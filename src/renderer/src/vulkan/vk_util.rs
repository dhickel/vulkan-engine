use std::cmp::max;
use std::ffi::CStr;

use crate::vulkan::vk_types::*;
use ash::vk;
use ash::vk::{
    AccessFlags2, ImageType, PipelineLayoutCreateInfo, PipelineStageFlags2, RenderingInfo,
};
use std::io::{Read, Seek, SeekFrom};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vk_mem::{Alloc, Allocator};

pub fn command_pool_create_info<'a>(
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
) -> vk::CommandPoolCreateInfo<'a> {
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(flags)
}

pub fn command_buffer_allocate_info<'a>(
    command_pool: vk::CommandPool,
    count: u32,
    level: vk::CommandBufferLevel,
) -> vk::CommandBufferAllocateInfo<'a> {
    vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .command_buffer_count(count)
        .level(level)
}

pub fn fence_create_info<'a>(flags: vk::FenceCreateFlags) -> vk::FenceCreateInfo<'a> {
    vk::FenceCreateInfo::default().flags(flags)
}

pub fn semaphore_create_info<'a>(flags: vk::SemaphoreCreateFlags) -> vk::SemaphoreCreateInfo<'a> {
    vk::SemaphoreCreateInfo::default().flags(flags)
}

pub fn command_buffer_begin_info<'a>(
    flags: vk::CommandBufferUsageFlags,
) -> vk::CommandBufferBeginInfo<'a> {
    vk::CommandBufferBeginInfo::default().flags(flags)
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn semaphore_submit_info<'a>(
    stage_flags: vk::PipelineStageFlags2,
    semaphore: vk::Semaphore,
) -> vk::SemaphoreSubmitInfo<'a> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(semaphore)
        .stage_mask(stage_flags)
        .device_index(0)
        .value(0)
}

pub fn command_buffer_submit_info<'a>(cmd: vk::CommandBuffer) -> vk::CommandBufferSubmitInfo<'a> {
    vk::CommandBufferSubmitInfo::default()
        .command_buffer(cmd)
        .device_mask(0)
}

pub fn submit_info_2<'a>(
    cmd_info: &'a [vk::CommandBufferSubmitInfo],
    signal_semaphore: &'a [vk::SemaphoreSubmitInfo],
    wait_semaphore: &'a [vk::SemaphoreSubmitInfo],
) -> vk::SubmitInfo2<'a> {
    vk::SubmitInfo2::default()
        .command_buffer_infos(cmd_info)
        .wait_semaphore_infos(wait_semaphore)
        .signal_semaphore_infos(signal_semaphore)
}

pub fn render_pass_begin_info<'a>(
    render_pass: vk::RenderPass,
    window_extent: vk::Extent2D,
    frame_buffer: vk::Framebuffer,
) -> vk::RenderPassBeginInfo<'a> {
    vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .render_area(
            vk::Rect2D::default()
                .offset(vk::Offset2D::default().x(0).y(0))
                .extent(window_extent),
        )
        .framebuffer(frame_buffer)
}

pub fn image_create_info<'a>(
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    extent: vk::Extent3D,
    image_type: vk::ImageType,
    sample_flags: vk::SampleCountFlags,
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(image_type)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(sample_flags)
        .usage(usage_flags)
}

pub fn create_image(
    device: &ash::Device,
    allocator: &Allocator,
    size: vk::Extent3D,
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    mip_mapped: bool,
) -> VkImageAlloc {
    let mut image_info = image_create_info(
        format,
        usage_flags,
        size,
        ImageType::TYPE_2D,
        vk::SampleCountFlags::TYPE_1,
    );

    let mips = if mip_mapped {
        let max_dimension = max(size.width, size.height) as f64;
        let mips = (max_dimension.log2().floor() as u32) + 1;
        image_info = image_info.mip_levels(mips);
        mips
    } else {
        1
    };

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    alloc_info.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    let (image, allocation) = unsafe { allocator.create_image(&image_info, &alloc_info).unwrap() };

    let aspect_flag = if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let mut view_info =
        image_view_create_info(format, vk::ImageViewType::TYPE_2D, image, aspect_flag);
    view_info.subresource_range.level_count = mips;

    let image_view = unsafe { device.create_image_view(&view_info, None).unwrap() };

    VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: size,
        image_format: format,
    }
}

pub fn image_view_create_info<'a>(
    format: vk::Format,
    view_type: vk::ImageViewType,
    image: vk::Image,
    aspect_flags: vk::ImageAspectFlags,
) -> vk::ImageViewCreateInfo<'a> {
    vk::ImageViewCreateInfo::default()
        .format(format)
        .image(image)
        .view_type(view_type)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .aspect_mask(aspect_flags),
        )
}

pub fn rendering_attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
    clear: Option<vk::ClearValue>,
) -> vk::RenderingAttachmentInfo<'a> {
    let info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        })
        .store_op(vk::AttachmentStoreOp::STORE);

    if let Some(clear) = clear {
        let info = info.clear_value(clear);
    };
    info
}

pub fn rendering_info<'a>(
    extent: vk::Extent2D,
    color_attachment: &'a [vk::RenderingAttachmentInfo],
    depth_attachment: &'a [vk::RenderingAttachmentInfo],
) -> RenderingInfo<'a> {
    let mut render_info = vk::RenderingInfo::default()
        .render_area(
            vk::Rect2D::default()
                .offset(vk::Offset2D::default().x(0).y(0))
                .extent(extent),
        )
        .layer_count(1);

    if !color_attachment.is_empty() {
        render_info = render_info.color_attachments(color_attachment);
    }
    // if !depth_attachment.is_empty() { // FIXME lifetime issues if optional but cant use a vec like above
    //     render_info = render_info.depth_attachment(depth_attachment)
    // }

    render_info
}

pub fn depth_attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'a> {
    let mut render_info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE);

    unsafe {
        render_info.clear_value.depth_stencil.depth = 0.0;
    }
    render_info
}

pub fn pipeline_shader_stage_create_info(
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    entry: &CStr,
) -> vk::PipelineShaderStageCreateInfo {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .name(entry)
        .module(module)
}

pub fn pipeline_layout_create_info<'a>() -> PipelineLayoutCreateInfo<'a> {
    vk::PipelineLayoutCreateInfo::default()
}

pub fn find_memory_type(
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    prop_flags: vk::MemoryPropertyFlags,
) -> u32 {
    todo!()
}

pub fn blit_copy_image_to_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    src_size: vk::Extent2D,
    dest: vk::Image,
    dest_size: vk::Extent2D,
) {
    let src_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D::default()
            .x(src_size.width as i32)
            .y(src_size.height as i32)
            .z(1),
    ];

    let dst_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D::default()
            .x(dest_size.width as i32)
            .y(dest_size.height as i32)
            .z(1),
    ];

    let src_sub_resource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let dst_sub_resource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let blit_region = [vk::ImageBlit2::default()
        .src_offsets(src_offsets)
        .dst_offsets(dst_offsets)
        .src_subresource(src_sub_resource)
        .dst_subresource(dst_sub_resource)];

    let blit_info = vk::BlitImageInfo2::default()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(dest)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&blit_region);

    unsafe { device.cmd_blit_image2(cmd, &blit_info) }
}

pub fn transition_image(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let image_barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(image_subresource_range(aspect_mask))
        .image(image)];

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barrier);

    unsafe { device.cmd_pipeline_barrier2(cmd_buffer, &dep_info) }
}

pub fn load_shader_module(
    device: &LogicalDevice,
    file_path: &str,
) -> Result<vk::ShaderModule, String> {
    // Open the file with the cursor at the end to determine the file size
    let mut file =
        std::fs::File::open(file_path).map_err(|e| format!("Failed to open file: {}", e))?;
    let file_size = file
        .seek(SeekFrom::End(0))
        .map_err(|e| format!("Failed to seek file: {}", e))?;

    // spirv expects the buffer to be on uint32, so make sure to reserve a Vec
    // big enough for the entire file
    let mut buffer = vec![0u32; (file_size / 4) as usize];

    // Put file cursor at the beginning
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("Failed to seek file: {}", e))?;

    // Load the entire file into the buffer
    file.read_exact(bytemuck::cast_slice_mut(&mut buffer))
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Create a new shader module, using the buffer we loaded
    let create_info = vk::ShaderModuleCreateInfo::default().code(&buffer);

    // Check that the creation goes well
    let shader_module = unsafe {
        device
            .device
            .create_shader_module(&create_info, None)
            .map_err(|e| format!("Failed to create shader module: {:?}", e))?
    };

    Ok(shader_module)
}

pub fn allocate_buffer(
    allocator: &Allocator,
    size: usize,
    usage_flags: vk::BufferUsageFlags,
    memory_usage: vk_mem::MemoryUsage,
) -> Result<VkBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size as vk::DeviceSize)
        .usage(usage_flags);

    let mut alloc_create_info = vk_mem::AllocationCreateInfo::default();
    alloc_create_info.usage = memory_usage;
    alloc_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED
        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;

    let (buffer, allocation) = unsafe {
        allocator
            .create_buffer(&buffer_info, &alloc_create_info)
            .map_err(|err| format!("Failed to allocate buffer, reason: {:?}", err))?
    };

    let alloc_info = allocator.get_allocation_info(&allocation);

    Ok(VkBuffer {
        buffer,
        allocation,
        alloc_info,
    })
}

pub fn destroy_buffer(allocator: &Allocator, mut buffer: VkBuffer) {
    unsafe {
        allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation)
    }
}
