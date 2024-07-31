use crate::data::gpu_data::AsByteSlice;
use crate::vulkan::vk_types::{VkBuffer, VkBufferAndDescriptorLimits, VkSubAlloc};
use crate::vulkan::vk_util;
use ash::vk;
use ash::vk::DeviceAddress;
use std::cmp::{max_by, Ordering, PartialEq};
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

#[derive(Clone, PartialEq)]
pub enum VkAllocResult {
    Success(Vec<VkSubAlloc>),
    OutOfSpace(PartialAlloc),
    Error {
        error_msg: String,
        alloc_state: PartialAlloc,
    },
}

#[derive(Clone, PartialEq)]
pub struct PartialAlloc {
    pub fulfilled: Vec<VkSubAlloc>,
    pub remaining: Vec<Vec<u8>>,
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
        max_upload_size: u64,
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
        let buffer = VkStorageBuffer::new(
            buffer_size as u64,
            stride,
            max_upload_size,
            buffer_address,
            buffer,
        );

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
    alignment: u64,
    max_upload_bytes: u64,
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
        max_upload_bytes: u64,
        buffer_address: DeviceAddress,
        buffer: VkBuffer,
    ) -> Self {
        Self {
            max_size: buffer_size,
            curr_size: 0,
            alignment: stride,
            max_upload_bytes,
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

    fn bytes_exceeded_error(
        &self,
        index: usize,
        byte_len: usize,
        alloc: PartialAlloc,
    ) -> VkAllocResult {
        VkAllocResult::Error {
            error_msg: format!(
                "Bytes exceed max upload limit for item: {} | Upload Limit: {}, Bytes{} ",
                index, self.max_upload_bytes, byte_len
            )
            .to_string(),
            alloc_state: alloc,
        }
    }

    fn to_partial_alloc(successes: Vec<VkSubAlloc>, failures: &[Vec<u8>]) -> PartialAlloc {
        PartialAlloc {
            fulfilled: successes,
            remaining: failures.to_vec(),
        }
    }

    fn tail_as_mem_chunk(&self) -> MemChunk {
        MemChunk::Tail {
            size: self.buffer_tail.size,
            address: self.buffer_tail.start_addr,
        }
    }

    fn select_best_chunk(&self, total_bytes: u64) -> MemChunk {
        // Prefer to use contiguous space from a free chunk
        if let Some(free_chunk) = self.free_chunks.find_best_fit(self.max_size) {
            free_chunk
        }
        // Revert to  tail for contiguous space if possible
        else if self.buffer_tail.size >= total_bytes
            || self.buffer_tail.size > self.free_chunks.max_free()
        {
            self.tail_as_mem_chunk()
        }
        // At this point disjoint allocation is happening, prefer the largest contiguous space
        else if let Some(free_chunk) = self.free_chunks.get_largest_chunk() {
            if free_chunk.size() < self.buffer_tail.size {
                self.tail_as_mem_chunk()
            } else {
                free_chunk
            }
        }
        // No space left at all
        else {
            MemChunk::Null
        }
    }

    pub fn add_items(
        &mut self,
        mut item_bytes: Vec<Vec<u8>>,
        buffer_placement: BufferPlacement,
    ) -> VkAllocResult {
        let mut total_bytes: u64 = 0;
        let mut alloc_sizes = Vec::with_capacity(item_bytes.len());
        let mut sub_allocations = Vec::<VkSubAlloc>::with_capacity(item_bytes.len());

        // Pad if needed and calc each item's final size, make sure no obj exceeds max
        for (index, item) in item_bytes.iter_mut().enumerate() {
            if item.len() & (self.alignment as usize - 1) == 0 {
                let padding = item.len().next_multiple_of(self.alignment as usize) - item.len();
                item.extend(std::iter::repeat(0).take(padding));
            }

            if item.len() > self.max_upload_bytes as usize {
                return self.bytes_exceeded_error(
                    index,
                    item.len(),
                    Self::to_partial_alloc(sub_allocations, &item_bytes),
                );
            }

            total_bytes = item.len() as u64;
            alloc_sizes.push(item.len() as u64)
        }

        // check this here to early abort if no contiguous space or tail space (end-only) found
        if (buffer_placement == BufferPlacement::EndOnly && self.buffer_tail.size < total_bytes)
            || (buffer_placement == BufferPlacement::ContiguousOnly
                && self.free_chunks.max_free() < total_bytes
                && self.buffer_tail.size < total_bytes)
        {
            return VkAllocResult::OutOfSpace(Self::to_partial_alloc(sub_allocations, &item_bytes));
        }

        // Either return tail chunk, or select best chunk for ContiguousOnly/Preferred
        // Due to prior checks these can be combined and a proper chunk will be returned
        let mut curr_mem_chunk = match buffer_placement {
            BufferPlacement::EndOnly => self.tail_as_mem_chunk(),
            BufferPlacement::ContiguousOnly | BufferPlacement::ContiguousPreferred => {
                self.select_best_chunk(total_bytes)
            }
        };

        // Short circuit if no space
        if curr_mem_chunk == MemChunk::Null {
            return VkAllocResult::OutOfSpace(Self::to_partial_alloc(sub_allocations, &item_bytes));
        }

        let mut start_range = 0;
        let mut end_range = 0;

        let mut curr_allot = 0_u64;
        let mut total_chunk_allot = 0_u64;
        let mut bytes_left = total_bytes;

        let mut curr_address = curr_mem_chunk.address();

        for (i, bytes) in item_bytes.iter().enumerate() {
            let byte_size = bytes.len() as u64;

            // Test if at upload bytes limit, submitting if next item is over max
            if (curr_allot + byte_size) > self.max_upload_bytes {
                // upload this slice here, check success and extend sub allocations
                let upload_slice = &item_bytes[start_range..end_range];
                // update address to the start of next allocation group
                curr_address = curr_address.add(curr_allot);
                // on success add to returned sub allocations, and update the used chunk

                // set start range for next allocation to current yet to be allocated index,
                // set new end range to current, and reset curr allotment
                start_range = i;
                end_range = i;
                bytes_left -= curr_allot;
                curr_allot = 0;
            }

            // Test if current memory chunk being allocated to is out of free memory
            // for next allocation
            if (total_chunk_allot + byte_size) > curr_mem_chunk.size() {
                // upload this slice here, check success and extend sub allocations
                let upload_slice = &item_bytes[start_range..end_range];
                // update address to the start of next allocation group
                curr_address = curr_address.add(curr_allot);
                // on success add to returned sub allocations, and update the used chunk

                match curr_mem_chunk {
                    MemChunk::Tail { size, .. } => {
                        self.buffer_tail.remove_from_chunk(total_chunk_allot)
                    }
                    MemChunk::FreeChunk { index, .. } => {
                        self.free_chunks.update_chunk(index, total_chunk_allot)
                    }
                    MemChunk::Null => panic!("Fatal: Branch should not be reached"),
                }

                start_range = i;
                end_range = i;
                bytes_left -= curr_allot;
                curr_allot = 0;
                total_chunk_allot = 0;

                // Select next chunk and assign address, or return current allotment state if out of space
                curr_mem_chunk = self.select_best_chunk(bytes_left);
                match curr_mem_chunk {
                    MemChunk::Null => {
                        return VkAllocResult::OutOfSpace(Self::to_partial_alloc(
                            sub_allocations,
                            &item_bytes[start_range..],
                        ));
                    }
                    MemChunk::Tail { address, .. } | MemChunk::FreeChunk { address, .. } => {
                        curr_address = address
                    }
                }
            }

            // Continue assigning upload allotments
            curr_allot += byte_size;
            total_chunk_allot += byte_size;
            end_range = i + 1;
        }

        if curr_allot > 0 {
            // upload this slice here, check success and extend sub allocations
            let upload_slice = &item_bytes[start_range..end_range];
            bytes_left -= curr_allot;
        }

        if total_chunk_allot > 0 {
            match curr_mem_chunk {
                MemChunk::Tail { size, .. } => {
                    self.buffer_tail.remove_from_chunk(total_chunk_allot)
                }
                MemChunk::FreeChunk { index, .. } => {
                    self.free_chunks.update_chunk(index, total_chunk_allot)
                }
                MemChunk::Null => panic!("Fatal: Branch should not be reached"),
            }
        }

        assert_eq!(0, bytes_left, "Failed to upload all bytes");
        assert_eq!(
            sub_allocations.len(),
            item_bytes.len(),
            "Allocation amount did not match item amount"
        );

        VkAllocResult::Success(sub_allocations)
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
    pub fn insert(&mut self, mut new_chunk: FreeChunk) {
        self.total_free += new_chunk.size;
        let mut front_adj: Option<usize> = None;
        let mut back_adj: Option<usize> = None;

        for (i, iter_chunk) in self.chunks.iter_mut().enumerate() {
            if (iter_chunk.start_addr + iter_chunk.size) == new_chunk.start_addr {
                front_adj = Some(i)
            }
            if (new_chunk.start_addr + new_chunk.size) == iter_chunk.start_addr {
                back_adj = Some(i)
            }
        }

        // if front_adj, set new chunks start addr to it and increase size
        if let Some(index) = front_adj {
            let front_chunk = unsafe { self.chunks.get_unchecked(index) };
            new_chunk.start_addr = front_chunk.start_addr;
            new_chunk.size += front_chunk.size;
        }

        // if back_adj, increase new chunks size
        if let Some(index) = back_adj {
            let back_chunk_size = unsafe { self.chunks.get_unchecked(index).size };
            new_chunk.size += back_chunk_size;
        }

        // Now that new chunk has "consumed" any adjacent chunks
        // Swap remove is done on back_adj first to avoid invalidating indices
        if let Some(index) = back_adj {
            self.chunks.swap_remove(index);
        }
        
        if let Some(index) = front_adj {
            self.chunks.swap_remove(index);
        }
        
        self.chunks.push(new_chunk)
    }

    pub fn max_free(&self) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.size)
            .max()
            .unwrap_or(0)
    }
    
    pub fn update_chunk(&mut self, index: usize, amount: u64) {
        unsafe {
            let chunk = self.chunks.get_unchecked_mut(index);
            chunk.remove_from_chunk(amount);
            self.total_free - amount;

            if chunk.size == 0 {
                self.chunks.swap_remove(index);
            }
        }
    }
    
    pub fn get_largest_chunk(&self) -> Option<MemChunk> {
        let chunk = self
            .chunks
            .iter()
            .enumerate()
            .max_by_key(|(i, chunk)| chunk.size);

        if let Some((index, chunk)) = chunk {
            Some(MemChunk::FreeChunk {
                index,
                size: chunk.size,
                address: chunk.start_addr,
            })
        } else {
            None
        }
    }

    pub fn find_best_fit(&self, size: u64) -> Option<MemChunk> {
        let chunk = self
            .chunks
            .iter()
            .enumerate()
            .find(|(_, chunk)| chunk.size >= size);

        if let Some((index, chunk)) = chunk {
            Some(MemChunk::FreeChunk {
                index,
                size: chunk.size,
                address: chunk.start_addr,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum MemChunk {
    Null,
    Tail {
        size: u64,
        address: DeviceAddress,
    },
    FreeChunk {
        index: usize,
        size: u64,
        address: DeviceAddress,
    },
}

impl MemChunk {
    pub fn size(&self) -> u64 {
        match self {
            MemChunk::Null => 0,
            MemChunk::FreeChunk { size, .. } | MemChunk::Tail { size, .. } => *size,
        }
    }
    pub fn address(&self) -> DeviceAddress {
        match self {
            MemChunk::Null => panic!("Called address on Null chunk"),
            MemChunk::Tail { address, .. } | MemChunk::FreeChunk { address, .. } => *address,
        }
    }
}
