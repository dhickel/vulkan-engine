//! # Vulkan Debug Capture Helpers
//!
//! Utility routines for copying GPU images to CPU and writing debug snapshots to disk.

use crate::vulkan::vk_util;
use ash::vk;
use image::{ImageBuffer, Rgba};

pub fn capture_and_save_image_view(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    format: vk::Format,
    extent: vk::Extent3D,
    path: &str,
) {
    let buffer_size = (extent.width * extent.height * 4) as vk::DeviceSize;
    let buffer = vk_util::allocate_buffer(
        allocator,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk_mem::MemoryUsage::Auto,
    )
    .unwrap();

    let buffer_memory = buffer.alloc_info.mapped_data;

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()[0]
    };
    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .unwrap();
    }

    // Get the image from the image view

    // Transition image layout for transfer
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Copy image to buffer
    let copy_region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(extent);

    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.buffer,
            &[copy_region],
        );
    }

    // End and submit command buffer
    unsafe {
        device.end_command_buffer(command_buffer).unwrap();
        let command_buffer = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffer);
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
    }

    // Map memory and create image
    let memory_ptr = buffer_memory as *mut u8;
    let data_slice =
        unsafe { std::slice::from_raw_parts(memory_ptr as *const u8, buffer_size as usize) };

    let img =
        ImageBuffer::<Rgba<u8>, _>::from_raw(extent.width, extent.height, data_slice.to_vec())
            .ok_or("Failed to create image from raw data")
            .unwrap();

    img.save(path).unwrap();

    // Clean up
    unsafe {
        // device.unmap_memory(buffer.alloc_info.device_memory);
        // device.free_memory(buffer.alloc_info.device_memory, None);
        device.destroy_buffer(buffer.buffer, None);
    }
}
