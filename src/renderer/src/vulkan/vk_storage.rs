use crate::data::gpu_data::AsByteSlice;
use crate::vulkan::vk_types::{VkBuffer, VkBufferAndDescriptorLimits, VkCmdSubmitInfo, VkDestroyable, VkHostBuffer, VkQueueType, VkSubAlloc, VkSubmitParam};
use crate::vulkan::vk_util;
use ash::{Device, vk};
use ash::vk::DeviceAddress;
use std::cmp::{max_by, Ordering, PartialEq};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, IndexMut, Sub};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SendError};
use std::time::Duration;
use log::{debug, info};
use vk_mem::Allocator;


#[derive(Clone, Copy, PartialEq)]
pub enum BufferPlacement {
    ContiguousPreferred,
    ContiguousOnly,
    EndOnly,
}


#[derive(Clone, PartialEq)]
enum VkBufferResult<'a> {
    Success(Vec<VkSubAlloc>),
    OutOfSpace(PartialAlloc<'a>),
    Error {
        error_msg: String,
        successful_allocs: Vec<VkSubAlloc>,
    },
}


pub enum VkAllocResult {
    Success(Vec<VkSubAlloc>),
    Failure {
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
    pub fn new(successes: Vec<VkSubAlloc>, failures: Vec<&'a [u8]>) -> PartialAlloc<'a> {
        PartialAlloc {
            fulfilled: successes,
            remaining: failures,
        }
    }
}


pub struct VkSubAllocator {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    buffer: VkStorageBuffer,
    transfer_buffer: Arc<Mutex<VkHostBuffer>>,
    extra_buffers: Vec<VkStorageBuffer>,
    usage_flags: vk::BufferUsageFlags,
    memory_usage: vk_mem::MemoryUsage,

}


impl VkDestroyable for VkSubAllocator {
    fn destroy(&mut self, device: &Device, allocator: &Allocator) {
        todo!()
    }
}


impl VkSubAllocator {
    pub fn new(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        transfer_buffer: Arc<Mutex<VkHostBuffer>>,
        buffer_size: u64,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        dst_barrier: vk::BufferMemoryBarrier<'static>,
        alignment: u64,
    ) -> Result<Self, String> {
        let buffer = Self::allocate_buffer(
            device,
            allocator.clone(),
            alignment,
            usage_flags,
            memory_usage,
            dst_barrier,
            buffer_size,
            transfer_buffer.lock().unwrap().buffer.size as u64,
            0,
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

    pub fn new_storage_buffer(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        transfer_buffer: Arc<Mutex<VkHostBuffer>>,
        buffer_size: u64,
        alignment: u64,
        flags: vk::BufferUsageFlags 
    ) -> Result<Self, String> {
        let usage_flags = flags | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let memory_usage = vk_mem::MemoryUsage::AutoPreferDevice;

        let (transfer_queue_index, graphics_queue_index) = {
            let tb = transfer_buffer.lock().unwrap();
            (tb.transfer_queue_index, tb.graphics_queue_index)
        };

        let dst_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .src_queue_family_index(transfer_queue_index)
            .dst_queue_family_index(graphics_queue_index);
        

        Self::new(
            device,
            allocator,
            transfer_buffer,
            buffer_size,
            usage_flags,
            memory_usage,
            dst_barrier,
            alignment,
        )
    }


    pub fn new_uniform_buffer(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        transfer_buffer: Arc<Mutex<VkHostBuffer>>,
        buffer_size: u64,
        limits: &VkBufferAndDescriptorLimits,
    ) -> Result<Self, String> {
        let usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        let memory_usage = vk_mem::MemoryUsage::AutoPreferDevice;

        let (transfer_queue_index, graphics_queue_index) = {
            let tb = transfer_buffer.lock().unwrap();
            (tb.transfer_queue_index, tb.graphics_queue_index)
        };

        let dst_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::UNIFORM_READ)
            .src_queue_family_index(transfer_queue_index)
            .dst_queue_family_index(graphics_queue_index);

        let alignment = limits.min_storage_buffer_offset_alignment;

        Self::new(
            device,
            allocator,
            transfer_buffer,
            buffer_size,
            usage_flags,
            memory_usage,
            dst_barrier,
            alignment,
        )
    }

    fn allocate_buffer(
        device: &ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        alignment: u64,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        dst_barrier: vk::BufferMemoryBarrier<'static>,
        buffer_size: u64,
        max_upload_size: u64,
        buffer_index: u32,
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
            buffer_index,
            buffer_size as u64,
            alignment,
            max_upload_size,
            buffer_address,
            buffer,
            dst_barrier,
        ))
    }

    pub fn allocate_bytes(
        &mut self,
        data: &[&[u8]],
        buffer_placement: BufferPlacement,
    ) -> VkAllocResult {
        // Acquire the host buffer lock, as some host buffers may he shared
        let host_buffer = match self.transfer_buffer.lock() {
            Ok(buffer) => buffer,
            Err(error_msg) => {
                return VkAllocResult::Failure {
                    error_msg: format!("Error acquiring host buffer lock: {:?}", error_msg).to_string(),
                    successful_allocs: vec![],
                };
            }
        };

        match self
            .buffer
            .add_items(data, buffer_placement, &self.device, &host_buffer)
        {
            VkBufferResult::OutOfSpace(mut partial_alloc) => {
                let mut new_buffer = match Self::allocate_buffer(
                    &self.device,
                    self.allocator.clone(),
                    self.buffer.alignment,
                    self.usage_flags,
                    self.memory_usage,
                    self.buffer.dst_barrier,
                    self.buffer.max_size,
                    self.buffer.max_upload_bytes,
                    self.extra_buffers.len() as u32,
                ) {
                    Ok(buffer) => buffer,
                    Err(err) => return VkAllocResult::Failure {
                        error_msg: "Out if space, Can't alloc more".to_string(),
                        successful_allocs: partial_alloc.fulfilled,
                    }
                };

                match new_buffer.add_items(
                    &mut partial_alloc.remaining,
                    buffer_placement,
                    &self.device,
                    &host_buffer,
                ) {
                    VkBufferResult::Success(mut items) => {
                        items.append(&mut partial_alloc.fulfilled);
                        VkAllocResult::Success(items)
                    } // TODO, maybe loop and keep creating? This will only trigger on an alloc > new buffer size
                    VkBufferResult::OutOfSpace(mut other_partial) => {
                        partial_alloc.fulfilled.append(&mut other_partial.fulfilled);
                        VkAllocResult::Failure {
                            error_msg: "Out if space on new buffer".to_string(),
                            successful_allocs: partial_alloc.fulfilled,
                        }
                    }
                    VkBufferResult::Error { error_msg, mut successful_allocs } => {
                        partial_alloc.fulfilled.append(&mut successful_allocs);
                        VkAllocResult::Failure {
                            error_msg: format!("Error encountered during allocation: {:?}", error_msg).to_string(),
                            successful_allocs: partial_alloc.fulfilled,
                        }
                    }
                }
            }
            VkBufferResult::Error { error_msg, successful_allocs } => {
                VkAllocResult::Failure {
                    error_msg: format!("Error encountered during allocation: {:?}", error_msg).to_string(),
                    successful_allocs: vec![],
                }
            }
            VkBufferResult::Success(allocs) => VkAllocResult::Success(allocs)
        }
    }

    pub fn deallocate(&mut self, sub_alloc: VkSubAlloc) {
        if sub_alloc.sub_buffer_index == 0 {
            self.buffer.delete_item(sub_alloc);
        } else if let Some(buffer) = self.extra_buffers.get_mut(sub_alloc.sub_buffer_index as usize) {
            buffer.delete_item(sub_alloc)
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
    dst_barrier: vk::BufferMemoryBarrier<'static>,
}


impl VkStorageBuffer {
    pub fn new(
        buffer_index: u32,
        buffer_size: u64,
        stride: u64,
        max_upload_bytes: u64,
        buffer_address: DeviceAddress,
        buffer: VkBuffer,
        dst_barrier: vk::BufferMemoryBarrier<'static>,
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
            dst_barrier,
        }
    }

    fn bytes_exceeded_error(&self, index: usize, byte_len: usize) -> VkBufferResult<'static> {
        VkBufferResult::Error {
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
    ) -> VkBufferResult<'static> {
        VkBufferResult::Error {
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
                alloc_address: curr_address,
                offset: curr_address.sub(self.buffer_start_addr),
                buffer: self.buffer.buffer,
                size: *size,
                sub_buffer_index: self.buffer_index,
            };

            curr_address.add_assign(size);
            sub_alloc_vec.push(alloc);
        }
    }

    // fn await_allocation_fence(
    //     &self,
    //     device: &ash::Device,
    //     fence: [vk::Fence; 1],
    // ) -> Result<(), String> {
    //     unsafe {
    //         device
    //             .wait_for_fences(&fence, true, 1e+10 as u64)
    //             .map_err(|err| format!("Error awaiting host transfer fence: {:?}", err))?;
    // 
    //         device
    //             .reset_fences(&fence)
    //             .map_err(|err| format!("Error resetting host transfer fence: {:?}", err))
    //     }
    // }

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

    fn get_offset_from(&self, alloc_address: DeviceAddress) -> u64 {
        alloc_address.sub(self.buffer_start_addr)
    }

    fn allocate_data(
        &self,
        device: &ash::Device,
        host_buffer: &VkHostBuffer,
        curr_address: u64,
        upload_slice: &[&[u8]],
    ) -> Result<(), String> {
        vk_util::record_host_to_storage_buffer(
            device,
            host_buffer,
            &self.buffer,
            curr_address.sub(self.buffer_start_addr),
            upload_slice,
            self.alignment,
        )
    }

    pub fn delete_item(&mut self, sub_alloc: VkSubAlloc) {
        let free_chunk = FreeChunk {
            start_addr: sub_alloc.alloc_address,
            size: sub_alloc.size,
        };
        self.free_chunks.insert(free_chunk)
    }

    pub fn add_items<'a>(
        &mut self,
        item_bytes: &[&'a [u8]],
        buffer_placement: BufferPlacement,
        device: &ash::Device,
        host_buffer: &VkHostBuffer,
    ) -> VkBufferResult<'a> {
        let mut total_bytes: u64 = 0;
        let mut alloc_sizes = Vec::with_capacity(item_bytes.len());
        let mut sub_allocations = Vec::<VkSubAlloc>::with_capacity(item_bytes.len());

        // Pad if needed and calc each item's final size, make sure no obj exceeds max
        for (index, item) in item_bytes.iter().enumerate() {
            let size = if item.len() & (self.alignment as usize - 1) == 0 {
                item.len().next_multiple_of(self.alignment as usize)
            } else {
                item.len()
            };

            if size > self.max_upload_bytes as usize {
                return self.bytes_exceeded_error(index, item.len());
            }

            total_bytes += item.len() as u64;
            alloc_sizes.push(item.len() as u64)
        }

        // check this here and early abort if no contiguous space or tail space (end-only) found
        if (buffer_placement == BufferPlacement::EndOnly && self.buffer_tail.size < total_bytes)
            || (buffer_placement == BufferPlacement::ContiguousOnly
            && self.free_chunks.max_free() < total_bytes && self.buffer_tail.size < total_bytes)
        {
            return VkBufferResult::OutOfSpace(PartialAlloc::new(sub_allocations, item_bytes.to_vec()));
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
            return VkBufferResult::OutOfSpace(PartialAlloc::new(sub_allocations, item_bytes.to_vec()));
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

                debug!("Recording upload buffer");
                if let Err(error_msg) = self.allocate_data(device, &host_buffer, curr_address, upload_slice) {
                    return self.partial_error(error_msg, sub_allocations);
                }

                debug!("Submitting VkStorage Commands");
                host_buffer.submit_transfer_commands(VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER)).unwrap();
                host_buffer.submit_graphics_commands(VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER)).unwrap();

                if let Err(error) = host_buffer.await_done(30) {
                    return self.partial_error(format!("Error awaiting upload response: {:?}", error), sub_allocations);
                } else {
                    host_buffer.reset_buffers(device);
                    debug!("Storage upload latch passed")
                }


                self.add_sub_allocations(curr_address, &alloc_sizes[start_range..end_range], &mut sub_allocations);

                start_range = i;
                bytes_left -= curr_allot;
                curr_allot = 0;

                curr_address = curr_address.add(curr_allot);
            }

            // get new buffer if needed
            if end_buffer_reached {
                match curr_mem_chunk {
                    MemChunk::Tail { size, .. } => self.buffer_tail.remove_from_chunk(total_chunk_allot),
                    MemChunk::FreeChunk { index, .. } => self.free_chunks.update_chunk(index, total_chunk_allot),
                    MemChunk::Null => panic!("Fatal: Branch should not be reached"),
                }

                total_chunk_allot = 0;

                // Select next chunk and assign address, or return current allotment state if out of space
                curr_mem_chunk = self.select_best_chunk(bytes_left);
                match curr_mem_chunk {
                    MemChunk::Null => {
                        return VkBufferResult::OutOfSpace(PartialAlloc::new(
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

            if let Err(error_msg) = self.allocate_data(device, &host_buffer, curr_address, upload_slice) {
                return self.partial_error(error_msg, sub_allocations);
            }


            debug!("Submitting VkStorage Commands");
            host_buffer.submit_transfer_commands(VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER)).unwrap();
            host_buffer.submit_graphics_commands(VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER)).unwrap();

            if let Err(error) = host_buffer.await_done(30) {
                return self.partial_error(format!("Error awaiting upload response: {:?}", error), sub_allocations);
            } else {
                host_buffer.reset_buffers(device);
                debug!("Storage upload latch passed")
            }


            self.add_sub_allocations(curr_address, &alloc_sizes[start_range..end_range], &mut sub_allocations);
            bytes_left -= curr_allot;

            match curr_mem_chunk {
                MemChunk::Tail { size, .. } => self.buffer_tail.remove_from_chunk(total_chunk_allot),
                MemChunk::FreeChunk { index, .. } => self.free_chunks.update_chunk(index, total_chunk_allot),
                MemChunk::Null => panic!("Fatal: Branch should not be reached"),
            }
        }

        assert_eq!(0, bytes_left, "Failed to upload all bytes");
        assert_eq!(
            sub_allocations.len(),
            item_bytes.len(),
            "Allocation amount did not match item amount"
        );

        VkBufferResult::Success(sub_allocations)
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
            self.total_free -= amount;

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
