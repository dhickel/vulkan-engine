use crate::data::gpu_data::AsByteSlice;
use crate::vulkan::vk_types::{VkBuffer, VkBufferAndDescriptorLimits, VkSubAlloc};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DeviceAddress;
use shaderc::ResourceKind::StorageBuffer;
use std::cmp::{Ordering, PartialEq};
use std::marker::PhantomData;
use std::ops::{Add, IndexMut};
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

#[derive(Clone, Copy, PartialEq)]
pub enum BufferPlacement {
    ContiguousPreferred,
    ContiguousOnly,
    EndOnly,
}

pub enum VkSingleAllocResult {
    Success(VkSubAlloc),
    OutOfSpace(Vec<u8>),
    Error(String),
}

pub enum VkMultiAllocResult {
    Success(Vec<VkSubAlloc>),
    Error(String),
    OutOfSpace {
        fulfilled: Vec<VkSubAlloc>,
        remaining: Vec<Vec<u8>>,
    },
}

pub struct VkStorageAllocator<T> {
    buffer: VkStorageBuffer,
    extra_buffers: Vec<VkStorageBuffer>,
    _type: std::marker::PhantomData<T>,
}

impl<T> VkStorageAllocator<T>
where
    T: AsByteSlice,
{
    pub fn new(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        limits: &VkBufferAndDescriptorLimits,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        max_allocations: u64,
    ) -> Result<Self, String> {
        let type_size = std::mem::size_of::<T>() as u64;
        let stride = type_size.next_multiple_of(limits.min_storage_buffer_offset_alignment);

        let max_allocations_by_range = (limits.max_storage_buffer_range / stride) as u64;
        let mut max_allocations = max_allocations.min(max_allocations_by_range);
        let mut buffer_size = max_allocations as usize * stride as usize;

        let allocator = allocator
            .lock()
            .map_err(|err| format!("Failed to acquire allocator lock: {:?}", err))?;

        let mut iter = 10;
        let buffer = loop {
            match vk_util::allocate_buffer(&allocator, buffer_size, usage_flags, memory_usage) {
                Ok(allocation) => break allocation,
                Err(err) => {
                    iter -= 1;
                    if iter < 0 || max_allocations == 0 {
                        return Err(format!("Failed to allocate, likely due to lack of memory | Last allocation Attempt: {} bytes", buffer_size));
                    }
                    max_allocations = max_allocations - (max_allocations / 4);
                    buffer_size = max_allocations as usize * stride as usize;
                }
            }
        };

        let buffer_addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        let buffer_address = unsafe { device.get_buffer_device_address(&buffer_addr_info) };
        let buffer = VkStorageBuffer::new(buffer_size as u64, stride, buffer_address, buffer);

        Ok(Self {
            buffer,
            extra_buffers: vec![],
            _type: PhantomData::default(),
        })
    }

    pub fn item_space_left(&self) -> u64 {
        self.buffer.max_size - self.buffer.curr_size
    }
}

pub struct VkRawStorageAllocator {
    free_sections: Vec<u32>,
    buffer: VkBuffer,
}

struct VkStorageBuffer {
    max_size: u64,
    curr_size: u64,
    stride: u64,
    buffer_tail: FreeChunk,
    buffer_start_addr: DeviceAddress,
    buffer_end_addr: DeviceAddress,
    buffer: VkBuffer,
    free_chunks: FreeChunkVec,
}

impl VkStorageBuffer {
    pub fn new(
        buffer_size: u64,
        stride: u64,
        buffer_address: DeviceAddress,
        buffer: VkBuffer,
    ) -> Self {
        Self {
            max_size: buffer_size,
            curr_size: 0,
            stride,
            buffer_tail: FreeChunk {
                start_addr: buffer_address,
                size: buffer_size,
            },
            buffer_start_addr: buffer_address,
            buffer_end_addr: buffer_address.add(buffer_size),
            buffer,
            free_chunks: FreeChunkVec::default(),
        }
    }

    pub fn add_item(
        &mut self,
        mut bytes: Vec<u8>,
        buffer_placement: BufferPlacement,
    ) -> VkSingleAllocResult {
        if !(bytes.len() & (self.stride as usize - 1) == 0) {
            let padding = bytes.len().next_multiple_of(self.stride as usize) - bytes.len();
            bytes.extend(std::iter::repeat(0).take(padding));
        }
        let byte_len = bytes.len() as u64;

        if matches!(
            buffer_placement,
            BufferPlacement::ContiguousPreferred | BufferPlacement::ContiguousOnly
        ) && self.free_chunks.max_free() >= byte_len
        {
            let (index, free_size) = unsafe { self.free_chunks.find_best_fit_unchecked(byte_len) };

            let chunk = unsafe { self.free_chunks.get_chunk_unchecked(index) };
            let addr = chunk.start_addr;

            //(record cmd buffer to pass item here)
            unsafe { self.free_chunks.update_chunk(index, byte_len) };
            // return alloc
            todo!()
        }

        // Couldn't allocate from free space, revert to tail allocation
        if byte_len > self.buffer_tail.size {
            return VkSingleAllocResult::OutOfSpace(bytes);
        }

        if self.buffer_tail.size < byte_len {
            return VkSingleAllocResult::OutOfSpace(bytes);
        }
        let addr = self.buffer_tail.start_addr;
        // record and pass here
        self.buffer_tail.remove_from_chunk(byte_len);
        //return alloc
        todo!()
    }

    pub fn add_items(
        &mut self,
        mut item_bytes: Vec<Vec<u8>>,
        buffer_placement: BufferPlacement,
    ) -> VkMultiAllocResult {
        let max_bytes = item_bytes
            .iter()
            .map(|b| b.len().next_multiple_of(self.stride as usize))
            .max()
            .unwrap_or(0) as u64;

        let mut max_bytes: u64 = 0;
        let mut sizes = Vec::<u64>::with_capacity(item_bytes.len());
        let mut flat_bytes = Vec::<u8>::with_capacity(max_bytes as usize);

        // Pad bytes to align if needed, keep track of these sizes for the return sub allocation vec
        // This size array is used to map 1-1 with bytes vecs in to allocations out to hand back to the caller
        for mut chunk in item_bytes.into_iter() {
            let original_len = chunk.len();
            let size = if original_len & (self.stride as usize - 1) == 0 {
                original_len
            } else {
                // pad to match stride
                let padding = chunk.len().next_multiple_of(self.stride as usize) - original_len;
                chunk.extend(std::iter::repeat(0).take(padding));
                chunk.len()
            };
            max_bytes += size as u64;
            sizes.push(size as u64);
            flat_bytes.append(&mut chunk);
        }

        // Handle end only
        if buffer_placement == BufferPlacement::EndOnly {
            if self.buffer_tail.size >= max_bytes {
                todo!("Contigious add end") // FIXME
            } else {
                return VkMultiAllocResult::OutOfSpace {
                    fulfilled: vec![],
                    remaining: vec![],
                };
            }
        }

        if matches!(
            buffer_placement,
            BufferPlacement::ContiguousOnly | BufferPlacement::ContiguousPreferred
        ) {
            if self.free_chunks.max_free() >= max_bytes {
                todo!("Contigious add") // FIXME
            } else if self.buffer_tail.size >= max_bytes {
                todo!("Contigious add") // FIXME
            } else if buffer_placement == BufferPlacement::ContiguousOnly {
                return None;
            }
        }

        // Fall through if contiguous preferred attempting to allocate most as contiguous, then best fits
        if buffer_placement == BufferPlacement::ContiguousPreferred {
            let mut total_bytes_left = max_bytes;
            let mut upload_slices = Vec::with_capacity(10);
            let mut sub_allocs = Vec::with_capacity(10);

            // A contiguous allocation was not found, select the largest chunk from either the
            // end of the buffer or the freed allocations
            let mut free_index: Option<usize> = None;
            let mut free_chunk = if self.buffer_tail.size > self.free_chunks.max_free() {
                &self.buffer_tail
            } else if let Some((index, size)) = self.free_chunks.get_largest_chunk() {
                free_index = Some(index);
                unsafe { self.free_chunks.get_chunk_unchecked(index) }
            } else {
                panic!("Fatal: No buffer space left, invalid state should not be reach");
            };

            // Keep track of the current index into the item sizes, and a pointer value into the
            // current head of flatten bytes for range selection
            let mut curr_idx = 0;
            let mut curr_bytes_head = 0;
            while total_bytes_left > 0 {
                // On a new outer iteration, a new chunk has been assigned, the tail starts at the head
                // and the amount of bytes left in a chunk is kept track of along with the current address
                // into the buffer which is assigned to each sub allocation returned
                let chunk_bytes_left = free_chunk.size;
                let mut address = free_chunk.start_addr;
                let mut curr_bytes_tail = curr_bytes_head;

                for i in curr_idx..sizes.len() {
                    let size = sizes[i];

                    if chunk_bytes_left
                        .checked_sub(size)
                        .is_some_and(|val| val > 0)
                    {
                        // set curr_idx to the current position in the items left to be allocated
                        curr_idx = i;

                        // push slice to be uploaded to gpu, reset current head to the tail for the
                        // next upload to start from
                        let upload_slice = &flat_bytes[curr_bytes_head..curr_bytes_tail];
                        upload_slices.push(upload_slice);
                        curr_bytes_head = curr_bytes_tail;

                        // Update chunk, updates size, removes chunk from free chunks if size == 0
                        let used_byte_amount = free_chunk.size - chunk_bytes_left;
                        if let Some(index) = free_index {
                            unsafe { self.free_chunks.update_chunk(index, used_byte_amount) };
                        } else {
                            self.buffer_tail.remove_from_chunk(used_byte_amount);
                        }

                        // Select the next chunk with the best fit
                        if let Some((index, size)) =
                            self.free_chunks.find_best_fit(total_bytes_left)
                        {
                            free_chunk = unsafe { self.free_chunks.get_chunk_unchecked(index) };
                            free_index = Some(index);
                        } else if self.buffer_tail.size < total_bytes_left {
                            panic!(
                                "Fatal: No buffer space left, invalid state should not be reach"
                            );
                        } else {
                            free_chunk = &self.buffer_tail;
                            free_index = None;
                        }
                        break;
                    }

                    // Create the sub allocation to be return, using the current address
                    // along with its size after alignment which is needed when freeing
                    let sub_alloc = VkSubAlloc { address, size };
                    // Update the address to the next location
                    address = address.add(size);
                    sub_allocs.push(sub_alloc);

                    total_bytes_left -= size;
                    curr_bytes_tail += size as usize;
                }
            }
            return None;
        }
        return None;
    }

    unsafe fn split_vec_by_sizes(
        mut data: Vec<u8>,
        start_index: usize,
        sizes: &[usize],
    ) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(sizes.len());
        let mut current_index = start_index;
        let data_ptr = data.as_mut_ptr().add(start_index);
        let data_len = data.len() - start_index;
        let data_cap = data.capacity() - start_index;

        // Forget to avoid freeing
        std::mem::forget(data);

        for &size in sizes {
            if current_index - start_index + size > data_len {
                panic!("Remaining sizes left to split, but no data left")
            }

            let vec = Vec::from_raw_parts(data_ptr.add(current_index), size, size);
            result.push(vec);
            current_index += size;
        }

        // Create a vec for the remaining data to ensure all memory is accounted for to avoid leaking
        let remaining_size = data_len - (current_index - start_index);
        let remaining = Vec::from_raw_parts(
            data_ptr.add(current_index - start_index),
            remaining_size,
            data_cap - (current_index - start_index),
        );
        result.push(remaining);

        result
    }

    fn record_upload_single() -> vk::CommandBuffer {}

    fn record_upload_multiple(
        bytes: &Vec<Vec<u8>>,
        addresses: &[DeviceAddress],
    ) -> vk::CommandBuffer {
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeChunk {
    pub start_addr: vk::DeviceAddress,
    pub size: u64,
}

impl PartialOrd for FreeChunk {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FreeChunk {
    fn cmp(&self, other: &Self) -> Ordering {
        self.size.cmp(&other.size)
    }
}

impl FreeChunk {
    pub fn remove_from_chunk(&mut self, amount: u64) {
        assert!(
            amount <= self.size,
            "Cannot remove more than the chunk's size"
        );
        self.size = self.size - amount;
        self.start_addr = self.start_addr.add(amount);
    }
}

struct FreeChunkVec {
    chunks: Vec<FreeChunk>,
    total_free: u64,
}

impl Default for FreeChunkVec {
    fn default() -> Self {
        Self {
            chunks: Vec::with_capacity(20),
            total_free: 0,
        }
    }
}

impl FreeChunkVec {
    pub fn insert(&mut self, new_chunk: FreeChunk) {
        self.total_free += new_chunk.size;
        let mut next_size_index = None;

        for (i, chunk) in self.chunks.iter_mut().enumerate() {
            if chunk.start_addr + chunk.size == new_chunk.start_addr {
                chunk.size += new_chunk.size;
                return;
            }
            if new_chunk.start_addr + new_chunk.size == chunk.start_addr {
                chunk.start_addr = new_chunk.start_addr;
                chunk.size += new_chunk.size;
                return;
            }
            if next_size_index.is_none() && chunk.size > new_chunk.size {
                next_size_index = Some(i);
            }
        }

        if let Some(index) = next_size_index {
            self.chunks.insert(index, new_chunk);
        } else {
            self.chunks.push(new_chunk);
        }
    }

    pub fn max_free(&self) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.size)
            .max()
            .unwrap_or(0)
    }

    pub fn de_frag(&mut self) {
        if self.chunks.is_empty() {
            return;
        }
        self.chunks.sort_by_key(|chunk| chunk.start_addr);

        let mut i = 0;
        while i < self.chunks.len() - 1 {
            let current_end = self.chunks[i].start_addr + self.chunks[i].size;
            let next_start = self.chunks[i + 1].start_addr;

            if current_end == next_start {
                let next_size = self.chunks[i + 1].size;
                self.chunks[i].size += next_size;
                self.chunks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub unsafe fn update_chunk(&mut self, index: usize, amount: u64) {
        let chunk = self.chunks.get_unchecked_mut(index);
        chunk.remove_from_chunk(amount);
        self.total_free - amount;

        if chunk.size == 0 {
            self.chunks.swap_remove(index);
        }
    }

    pub unsafe fn get_chunk_unchecked(&self, index: usize) -> &FreeChunk {
        self.chunks.get_unchecked(index)
    }

    pub fn get_chunk(&self, index: usize) -> Option<&FreeChunk> {
        self.chunks.get(index)
    }

    pub fn get_largest_chunk(&self) -> Option<(usize, u64)> {
        self.chunks
            .iter()
            .enumerate()
            .max_by_key(|(i, chunk)| chunk.size)
            .map(|(i, chunk)| (i, chunk.size))
    }

    pub unsafe fn get_largest_chunk_unchecked(&self) -> (usize, u64) {
        self.chunks
            .iter()
            .enumerate()
            .max_by_key(|(i, chunk)| chunk.size)
            .map(|(i, chunk)| (i, chunk.size))
            .unwrap()
    }

    pub unsafe fn find_best_fit_unchecked(&self, size: u64) -> (usize, u64) {
        self.chunks
            .iter()
            .enumerate()
            .find(|(_, chunk)| chunk.size >= size)
            .map(|(index, chunk)| (index, chunk.size))
            .unwrap()
    }

    pub fn find_best_fit(&self, size: u64) -> Option<(usize, u64)> {
        self.chunks
            .iter()
            .enumerate()
            .find(|(_, chunk)| chunk.size >= size)
            .map(|(index, chunk)| (index, chunk.size))
    }
}
