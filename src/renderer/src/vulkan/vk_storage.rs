use crate::data::gpu_data::AsByteSlice;
use crate::vulkan::vk_types::{
    VkBuffer, VkBufferAndDescriptorLimits, VkCommandPool, VkHostBuffer, VkHostBufferLease,
    VkSubAlloc,
};
use crate::vulkan::vk_util;
use crate::vulkan::vk_util::immediate_submit;
use ash::vk;
use ash::vk::DeviceAddress;
use std::cmp::{max_by, Ordering, PartialEq};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, IndexMut, Sub};
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

#[derive(Clone, Copy, PartialEq)]
pub enum BufferPlacement {
    ContiguousPreferred,
    ContiguousOnly,
    EndOnly,
}

#[derive(Clone, PartialEq)]
pub enum VkAllocResult<'a> {
    Success(Vec<VkSubAlloc>),
    OutOfSpace(PartialAlloc<'a>),
    Error {
        error_msg: String,
        successful_allocs: Vec<VkSubAlloc>,
    },
}

#[derive(Clone, PartialEq)]
pub struct PartialAlloc<'a> {
    pub fulfilled: Vec<VkSubAlloc>,
    pub remaining: Vec<&'a [u8]>,
}

impl<'a> PartialAlloc<'a> {
    pub fn new(successes: Vec<VkSubAlloc>, failures: Vec<&'a [u8]>) -> PartialAlloc {
        PartialAlloc {
            fulfilled: successes,
            remaining: failures,
        }
    }
}

pub struct VkStorageAllocator {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    buffer: VkStorageBuffer,
    transfer_buffer: Arc<VkHostBuffer>,
    extra_buffers: Vec<VkStorageBuffer>,
    usage_flags: vk::BufferUsageFlags,
    memory_usage: vk_mem::MemoryUsage,
}

impl VkStorageAllocator {
    pub fn new(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        transfer_buffer: Arc<VkHostBuffer>,
        limits: &VkBufferAndDescriptorLimits,
        buffer_size: usize,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        max_upload_size: u64,
    ) -> Result<Self, String> {
        let alignment = limits.min_storage_buffer_offset_alignment;

        let buffer = Self::allocate_buffer(
            device,
            allocator.clone(),
            alignment,
            usage_flags,
            memory_usage,
            buffer_size,
            max_upload_size,
        )?;

        Ok(Self {
            device: device.clone(),
            allocator,
            buffer,
            transfer_buffer,
            extra_buffers: vec![],
            usage_flags,
            memory_usage,
        })
    }

    fn allocate_buffer(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        alignment: u64,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        buffer_size: usize,
        max_upload_size: u64,
    ) -> Result<VkStorageBuffer, String> {
        let allocator = allocator
            .lock()
            .map_err(|err| format!("Failed to acquire allocator lock: {:?}", err))?;

        let mut buffer_size = buffer_size;
        let mut iter = 10;
        let buffer = loop {
            match vk_util::allocate_buffer(&allocator, buffer_size, usage_flags, memory_usage) {
                Ok(allocation) => break allocation,
                Err(err) => {
                    iter -= 1;
                    if iter < 0 || buffer_size == 0 {
                        return Err(format!("Failed to allocate, likely due to lack of memory | Last allocation Attempt: {} bytes", buffer_size));
                    }
                    buffer_size = buffer_size.checked_sub(buffer_size / 4)
                        .ok_or(format!("Failed to allocate, likely due to lack of memory | Last allocation Attempt: {} bytes", buffer_size))?;
                }
            }
        };

        let buffer_addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        let buffer_address = unsafe { device.get_buffer_device_address(&buffer_addr_info) };
        Ok(VkStorageBuffer::new(
            0,
            buffer_size as u64,
            alignment,
            max_upload_size,
            buffer_address,
            buffer,
        ))
    }

    pub fn allocate_bytes<'a>(
        &mut self,
        data: Vec<&'a [u8]>,
        buffer_placement: BufferPlacement,
    ) -> VkAllocResult<'a> {
        // Acquires a semaphore that will release when dropped
        let host_lease = self.transfer_buffer.acquire();

        let result = self
            .buffer
            .add_items(data, buffer_placement, &self.device, &host_lease);

        if let VkAllocResult::OutOfSpace(mut partial_alloc) = result {
            let mut new_buffer = match Self::allocate_buffer(
                &self.device,
                self.allocator.clone(),
                self.buffer.alignment,
                self.usage_flags,
                self.memory_usage,
                self.buffer.max_size as usize,
                self.buffer.max_upload_bytes,
            ) {
                Ok(buffer) => buffer,
                Err(err) => {
                    return VkAllocResult::Error {
                        error_msg: err,
                        successful_allocs: partial_alloc.fulfilled,
                    };
                }
            };

            match new_buffer.add_items(
                partial_alloc.remaining,
                buffer_placement,
                &self.device,
                &host_lease,
            ) {
                VkAllocResult::Success(mut items) => {
                    items.append(&mut partial_alloc.fulfilled);
                    VkAllocResult::Success(items)
                }
                VkAllocResult::OutOfSpace(mut other_partial) => {
                    other_partial.fulfilled.append(&mut partial_alloc.fulfilled);
                    return VkAllocResult::Error {
                        error_msg: "Out of space on fresh buffer".to_string(),
                        successful_allocs: other_partial.fulfilled,
                    };
                }
                VkAllocResult::Error {
                    error_msg,
                    mut successful_allocs,
                } => {
                    partial_alloc.fulfilled.append(&mut successful_allocs);
                    return VkAllocResult::Error {
                        error_msg: "Out of space on fresh buffer".to_string(),
                        successful_allocs: partial_alloc.fulfilled,
                    };
                }
            }
        } else {
            result
        }
    }
}

struct VkStorageBuffer {
    buffer_index: u32,
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
        buffer_index: u32,
        buffer_size: u64,
        stride: u64,
        max_upload_bytes: u64,
        buffer_address: DeviceAddress,
        buffer: VkBuffer,
    ) -> Self {
        Self {
            buffer_index,
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

    fn bytes_exceeded_error(&self, index: usize, byte_len: usize) -> VkAllocResult<'static> {
        VkAllocResult::Error {
            error_msg: format!(
                "Bytes exceed max upload limit for item: {} | Upload Limit: {}, Bytes{} ",
                index, self.max_upload_bytes, byte_len
            )
            .to_string(),
            successful_allocs: vec![],
        }
    }

    fn partial_error(
        &self,
        error_msg: String,
        successful_allocs: Vec<VkSubAlloc>,
    ) -> VkAllocResult<'static> {
        VkAllocResult::Error {
            error_msg,
            successful_allocs,
        }
    }

    fn tail_as_mem_chunk(&self) -> MemChunk {
        MemChunk::Tail {
            size: self.buffer_tail.size,
            address: self.buffer_tail.start_addr,
        }
    }

    fn add_sub_allocations(
        &self,
        alloc_start_addr: DeviceAddress,
        alloc_sizes: &[u64],
        sub_alloc_vec: &mut Vec<VkSubAlloc>,
    ) {
        let mut curr_address = alloc_start_addr;
        for size in alloc_sizes {
            let alloc = VkSubAlloc {
                address: curr_address,
                offset: curr_address.sub(self.buffer_start_addr),
                buffer: self.buffer.buffer,
                size: *size,
                sub_buffer_index: self.buffer_index,
            };

            curr_address.add_assign(size);
            sub_alloc_vec.push(alloc);
        }
    }

    fn await_allocation_fence(
        &self,
        device: &ash::Device,
        fence: [vk::Fence; 1],
    ) -> Result<(), String> {
        unsafe {
            device
                .wait_for_fences(&fence, true, (6e+10) as u64)
                .map_err(|err| format!("Error awaiting host transfer fence: {:?}", err))
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

    pub fn get_offset_from(&self, alloc_address: DeviceAddress) -> u64 {
        alloc_address.sub(self.buffer_start_addr)
    }

    pub fn allocate_data(
        &self,
        device: &ash::Device,
        host_lease: &VkHostBufferLease,
        curr_address: u64,
        upload_slice: &[&[u8]],
    ) -> Result<(), String> {
        vk_util::transfer_data_host_to_device(
            device,
            host_lease.cmd_pool.buffer,
            host_lease.buffer,
            &self.buffer,
            curr_address.sub(self.buffer_start_addr),
            upload_slice,
            self.alignment,
        )
    }

    pub fn add_items<'a>(
        &mut self,
        mut item_bytes: Vec<&'a [u8]>,
        buffer_placement: BufferPlacement,
        device: &ash::Device,
        host_lease: &VkHostBufferLease,
    ) -> VkAllocResult<'a> {
        let mut total_bytes: u64 = 0;
        let mut alloc_sizes = Vec::with_capacity(item_bytes.len());
        let mut sub_allocations = Vec::<VkSubAlloc>::with_capacity(item_bytes.len());

        // Pad if needed and calc each item's final size, make sure no obj exceeds max
        for (index, item) in item_bytes.iter_mut().enumerate() {
            let size = if item.len() & (self.alignment as usize - 1) == 0 {
                item.len().next_multiple_of(self.alignment as usize)
            } else {
                item.len()
            };

            if size > self.max_upload_bytes as usize {
                return self.bytes_exceeded_error(index, item.len());
            }

            total_bytes = item.len() as u64;
            alloc_sizes.push(item.len() as u64)
        }

        // check this here and early abort if no contiguous space or tail space (end-only) found
        if (buffer_placement == BufferPlacement::EndOnly && self.buffer_tail.size < total_bytes)
            || (buffer_placement == BufferPlacement::ContiguousOnly
                && self.free_chunks.max_free() < total_bytes
                && self.buffer_tail.size < total_bytes)
        {
            return VkAllocResult::OutOfSpace(PartialAlloc::new(sub_allocations, item_bytes));
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
            return VkAllocResult::OutOfSpace(PartialAlloc::new(sub_allocations, item_bytes));
        }

        let mut start_range = 0;
        let mut end_range = 0;

        let mut curr_allot = 0_u64;
        let mut total_chunk_allot = 0_u64;
        let mut bytes_left = total_bytes;

        let mut curr_address = curr_mem_chunk.address();

        for (i, bytes) in item_bytes.iter().enumerate() {
            let byte_size = bytes.len() as u64;

            // Check upload bytes and chunk byte limits
            let max_upload_reached = (curr_allot + byte_size) > self.max_upload_bytes;
            let end_buffer_reached = (total_chunk_allot + byte_size) > curr_mem_chunk.size();

            // handle upload if needed, and reset params
            if max_upload_reached || end_buffer_reached {
                let upload_slice = &item_bytes[start_range..end_range];
                let upload_offset = self.get_offset_from(curr_address);

                if let Err(error_msg) =
                    self.allocate_data(device, &host_lease, curr_address, upload_slice)
                {
                    return self.partial_error(error_msg, sub_allocations);
                }

                // Channel submit here

                if let Err(error_msg) = self.await_allocation_fence(device, host_lease.fence) {
                    return self.partial_error(error_msg, sub_allocations);
                }

                self.add_sub_allocations(
                    curr_address,
                    &alloc_sizes[start_range..end_range],
                    &mut sub_allocations,
                );

                start_range = i;
                end_range = i;
                bytes_left -= curr_allot;
                curr_allot = 0;

                curr_address = curr_address.add(curr_allot);
            }

            // get new buffer if needed
            if end_buffer_reached {
                match curr_mem_chunk {
                    MemChunk::Tail { size, .. } => {
                        self.buffer_tail.remove_from_chunk(total_chunk_allot)
                    }
                    MemChunk::FreeChunk { index, .. } => {
                        self.free_chunks.update_chunk(index, total_chunk_allot)
                    }
                    MemChunk::Null => panic!("Fatal: Branch should not be reached"),
                }

                total_chunk_allot = 0;

                // Select next chunk and assign address, or return current allotment state if out of space
                curr_mem_chunk = self.select_best_chunk(bytes_left);
                match curr_mem_chunk {
                    MemChunk::Null => {
                        return VkAllocResult::OutOfSpace(PartialAlloc::new(
                            sub_allocations,
                            item_bytes[start_range..].to_vec(),
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

        // check and upload any remaining data
        if curr_allot > 0 {
            let upload_slice = &item_bytes[start_range..end_range];
            let upload_offset = self.get_offset_from(curr_address);

            if let Err(error_msg) =
                self.allocate_data(device, &host_lease, curr_address, upload_slice)
            {
                return self.partial_error(error_msg, sub_allocations);
            }

            // Channel submit here

            if let Err(error_msg) = self.await_allocation_fence(device, host_lease.fence) {
                return self.partial_error(error_msg, sub_allocations);
            }

            self.add_sub_allocations(
                curr_address,
                &alloc_sizes[start_range..end_range],
                &mut sub_allocations,
            );

            bytes_left -= curr_allot;

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

        // Now new chunk has "consumed" any adjacent chunks
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
