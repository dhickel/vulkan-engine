use std::cmp::{max, PartialEq};
use std::ffi::CStr;

use crate::data::gpu_data::{TextureMeta, Vertex, VkCubeMap, VkMeshBuffers};
use crate::vulkan::vk_types::*;
use ash::vk;
use ash::vk::{AccessFlags2, ClearValue, DependencyFlags, DeviceAddress, DeviceSize, Extent2D, Extent3D, ImageLayout, ImageType, PipelineCache, PipelineLayoutCreateInfo, PipelineStageFlags2, Rect2D, RenderingInfo};
use log::{error, info};
use std::io::{Bytes, Read, Seek, SeekFrom};
use std::mem::align_of;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use vk_mem::{Alloc, Allocator};

use crate::data::data_cache::{
    LodBias, VkSamplerCache, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType,
    VkSamplerInfo,
};
use crate::data::data_util;
use crate::vulkan::vk_descriptor::{PoolSizeRatio, VkDescriptorAllocator};
use crate::vulkan::vk_render::VkSingleDescriptor;
use crate::vulkan::{vk_init, vk_util};
use ash::prelude::VkResult;
use shaderc::{CompileOptions, Compiler, ShaderKind};
use std::fs;
use std::fs::metadata;
use std::ops::{Add, Sub};
use std::path::{Path, PathBuf};
use crate::data::data_util::calc_mips_count;


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
    mips_levels: u32,
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(image_type)
        .format(format)
        .extent(extent)
        .mip_levels(mips_levels)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
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
    mips_levels: u32,
) -> VkImageAlloc {
    let image_info = image_create_info(
        format,
        usage_flags,
        size,
        ImageType::TYPE_2D,
        vk::SampleCountFlags::TYPE_1,
        mips_levels,
    );

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
    view_info.subresource_range.level_count = mips_levels;

    let image_view = unsafe { device.create_image_view(&view_info, None).unwrap() };

    VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: size,
        image_format: format,
        mip_levels: mips_levels,
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


pub fn attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
    clear: Option<vk::ClearValue>,
) -> vk::RenderingAttachmentInfo<'a> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        })
        .store_op(vk::AttachmentStoreOp::STORE);

    if let Some(clear) = clear {
        info = info.clear_value(clear);
    };
    info
}


pub fn rendering_info<'a>(
    extent: vk::Extent2D,
    color_attachment: &'a [vk::RenderingAttachmentInfo],
    depth_attachment: Option<&'a vk::RenderingAttachmentInfo>,
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

    if let Some(depth) = depth_attachment {
        render_info = render_info.depth_attachment(depth);
    }
    render_info
}


pub fn depth_attachment_info<'a>(
    view: vk::ImageView,
    layout: vk::ImageLayout,
) -> vk::RenderingAttachmentInfo<'a> {
    let clear_value = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    };

    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(clear_value)
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


pub fn transition_image_layered(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    layer_count: u32,
    mips_level: u32,
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
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mips_level,
            base_array_layer: 0,
            layer_count,
        })
        .image(image)];

    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barrier);

    unsafe { device.cmd_pipeline_barrier2(cmd_buffer, &dep_info) }
}


pub fn load_shader_module(
    device: &ash::Device,
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
            .create_shader_module(&create_info, None)
            .map_err(|e| format!("Failed to create shader module: {:?}", e))?
    };

    Ok(shader_module)
}


pub fn allocate_buffer(
    allocator: &Allocator,
    size: u64,
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
        size,
        allocation,
        alloc_info,
    })
}


pub fn allocate_host_buffer(
    allocator: &Allocator,
    size: u64,
) -> Result<VkBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let alloc_create_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferHost,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    let (buffer, allocation) = unsafe {
        allocator
            .create_buffer(&buffer_info, &alloc_create_info)
            .map_err(|err| format!("Failed to allocate buffer, reason: {:?}", err))?
    };

    let alloc_info = unsafe { allocator.get_allocation_info(&allocation) };

    Ok(VkBuffer {
        buffer,
        size,
        allocation,
        alloc_info,
    })
}


pub fn allocate_and_write_buffer(
    allocator: &Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<VkBuffer, String> {
    let buffer_size = data.len() as u64;
    let mut buffer = allocate_buffer(allocator, buffer_size, usage, vk_mem::MemoryUsage::Auto)?;

    unsafe {
        let data_ptr = allocator
            .map_memory(&mut buffer.allocation)
            .map_err(|err| format!("Failed to map memory: {:?}", err))?;

        std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        allocator.unmap_memory(&mut buffer.allocation);
    }
    Ok(buffer)
}


pub fn destroy_buffer(allocator: &Allocator, mut buffer: VkBuffer) {
    unsafe { allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation) }
}


pub fn destroy_image(allocator: &Allocator, mut image: VkImageAlloc) {
    unsafe { allocator.destroy_image(image.image, &mut image.allocation) }
}

//////////////////
// ENGINE UTIL ///
//////////////////

pub fn generate_brdf_lut(
    device: &ash::Device,
    allocator: &Allocator,
    pipeline: vk::Pipeline,
    graphics_cmd_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> VkBrdfLut {
    info!("Generating BRDF LUT");
    let start = SystemTime::now();


    let format = vk::Format::R16G16B16A16_SFLOAT;
    let size = Extent3D::default().width(512).height(512).depth(1);
    let dim_extent = Extent2D::default().width(512).height(512);
    let dim_rect = Rect2D::default().extent(dim_extent);

    let brd_img = create_image(
        device,
        allocator,
        size,
        format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        1,
    );

    let brd_sampler = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(1.0)
        .max_anisotropy(1.0)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);

    let brd_sampler = unsafe { device.create_sampler(&brd_sampler, None).unwrap() };

    let color_attachment_format = vk::Format::R16G16B16A16_SFLOAT;
    let pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&[color_attachment_format]);

    let clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let color_attachment = [attachment_info(
        brd_img.image_view,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        Some(clear_value),
    )];

    let rendering_info = vk::RenderingInfo::default()
        .render_area(dim_rect)
        .layer_count(1)
        .color_attachments(&color_attachment);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: dim_extent.height as f32,
        height: dim_extent.width as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: dim_extent,
    };

    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(graphics_cmd_buffer, &begin_info)
            .unwrap();

        transition_image(
            device,
            graphics_cmd_buffer,
            brd_img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        device.cmd_begin_rendering(graphics_cmd_buffer, &rendering_info);

        device.cmd_set_viewport(graphics_cmd_buffer, 0, &[viewport]);
        device.cmd_set_scissor(graphics_cmd_buffer, 0, &[scissor]);
        device.cmd_bind_pipeline(graphics_cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        device.cmd_draw(graphics_cmd_buffer, 3, 1, 0, 0);

        device.cmd_end_rendering(graphics_cmd_buffer);

        // Transition to final shader formatting
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);


        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(brd_img.image)
            .subresource_range(subresource_range)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);

        device.cmd_pipeline_barrier(
            graphics_cmd_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        device.end_command_buffer(graphics_cmd_buffer).unwrap();

        let cmd_info = [command_buffer_submit_info(graphics_cmd_buffer)];
        let submit_info = [submit_info_2(&cmd_info, &[], &[])];
        device
            .queue_submit2(graphics_queue, &submit_info, vk::Fence::null())
            .unwrap();

        device.device_wait_idle().unwrap();
        let end = SystemTime::now().duration_since(start).unwrap().as_millis();

        info!("BRDF LUT generation took: {} ms", end);
    }

    VkBrdfLut {
        sampler: brd_sampler,
        image_alloc: brd_img,
        extent: dim_extent,
    }
}


pub fn upload_skybox(
    device: &ash::Device,
    allocator: &Allocator,
    tex_meta: TextureMeta,
    transfer_pool: &VkCommandPool,
    transfer_queue: vk::Queue,
) -> VkCubeMap {
    let face_width = tex_meta.width / 6;

    let staging_buffer = allocate_and_write_buffer(
        allocator,
        &tex_meta.bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )
        .unwrap();

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(tex_meta.format)
        .mip_levels(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .extent(
            vk::Extent3D::default()
                .width(face_width)
                .height(tex_meta.height)
                .depth(1),
        )
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .array_layers(6)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe { device.create_image(&image_create_info, None).unwrap() };

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Unknown,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    let (allocation, device_memory, alloc_offset) = unsafe {
        let alloc = allocator
            .allocate_memory_for_image(image, &alloc_info)
            .unwrap();

        let alloc_info = allocator.get_allocation_info(&alloc);
        let device_memory = alloc_info.device_memory;
        let offset = alloc_info.offset;

        device
            .bind_image_memory(image, device_memory, offset)
            .unwrap();

        (alloc, device_memory, offset)
    };

    let cmd_buffer = transfer_pool.buffers[0];
    unsafe {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .unwrap();

        // Map regions for each face
        let regions: Vec<vk::BufferImageCopy> = (0..6)
            .map(|i| {
                let buffer_offset = i * (tex_meta.bytes.len() / 6) as u64;
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(face_width)
                    .buffer_image_height(tex_meta.height)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: face_width,
                        height: tex_meta.height,
                        depth: 1,
                    })
            })
            .collect();

        // Transition image for copy
        transition_image_layered(
            device,
            cmd_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            6,
            1,
        );

        // Copy buffer to image
        device.cmd_copy_buffer_to_image(
            cmd_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );

        // Transition image for shader read
        transition_image_layered(
            device,
            cmd_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            6,
            1,
        );

        device.end_command_buffer(cmd_buffer).unwrap();

        let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

        // Submit the command buffer and signal the fence correctly
        device
            .queue_submit2(transfer_queue, &submit_info, vk::Fence::null())
            .unwrap();

        device.device_wait_idle();

        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::NEVER)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = device.create_sampler(&sampler_create_info, None).unwrap();

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(tex_meta.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });

        let image_view = device.create_image_view(&view_create_info, None).unwrap();

        destroy_buffer(allocator, staging_buffer);

        let full_extent = Extent3D {
            width: tex_meta.width,
            height: tex_meta.height,
            depth: 1,
        };

        let face_extent = Extent3D {
            width: face_width,
            height: tex_meta.height,
            depth: 1,
        };

        VkCubeMap {
            texture_meta: Some(tex_meta),
            full_extent,
            face_extent,
            allocation,
            image,
            image_view,
            sampler,
        }
    }
}


fn create_buffer_image_copy(
    face_index: u32, // 0-based index of the face in the horizontal strip
    face_width: u32,
    face_height: u32,
    total_width: u32,
    layer: u32,
) -> vk::BufferImageCopy {
    let buffer_offset = (face_index * face_width * face_height * 4) as u64;
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(total_width)
        .buffer_image_height(face_height)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: layer,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: face_width,
            height: face_height,
            depth: 1,
        })
}


pub fn compile_shaders(shader_dir: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a shader compiler
    let compiler = Compiler::new().ok_or("Failed to create shader compiler")?;
    let mut options = CompileOptions::new().ok_or("Failed to create compile options")?;
    options.add_macro_definition("GL_EXT_buffer_reference", Some("1"));
    options.add_macro_definition("GL_GOOGLE_include_directive", Some("1"));

    let shader_dir_path = Path::new(shader_dir);
    options.set_include_callback(move |name, _include_type, _source_file, _depth| {
        let path = shader_dir_path.join(name);
        if path.exists() {
            Ok(shaderc::ResolvedInclude {
                resolved_name: path.to_str().unwrap().to_string(),
                content: fs::read_to_string(&path)
                    .map_err(|err| " Failed to read shader includes".to_string())?,
            })
        } else {
            Err(format!("Failed to find include file: {}", name))
        }
    });

    // Iterate over the shader files in the shader directory
    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            if ["vert", "frag", "comp"].contains(&extension.to_str().unwrap()) {
                // Read the shader source code
                let shader_source = fs::read_to_string(&path)?;

                // Compile the shader
                let shader_kind = match extension.to_str().unwrap() {
                    "vert" => ShaderKind::Vertex,
                    "frag" => ShaderKind::Fragment,
                    "comp" => ShaderKind::Compute,
                    _ => continue, // Skip unsupported shader types
                };

                let binary_result = compiler.compile_into_spirv(
                    &shader_source,
                    shader_kind,
                    path.file_name().unwrap().to_str().unwrap(),
                    "main",
                    Some(&options),
                )?;

                // Write the compiled SPIR-V to the output directory
                let output_path = PathBuf::from(out_dir).join(format!(
                    "{}.spv",
                    path.file_name().unwrap().to_str().unwrap()
                ));
                fs::write(&output_path, binary_result.as_binary_u8())?;
                println!("Compiled shader: {:?}", output_path);
            }
        }
    }

    Ok(())
}


pub(crate) fn create_cubemap(
    device: &ash::Device,
    allocator: &Allocator,
    format: vk::Format,
    dim: u32,
    num_mips: u32,
) -> Result<(VkImageAlloc, vk::Sampler), String> {
    let extent = vk::Extent3D {
        width: dim,
        height: dim,
        depth: 1,
    };

    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(num_mips)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::Unknown,
        flags: vk_mem::AllocationCreateFlags::MAPPED
            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        ..Default::default()
    };

    let (image, allocation) = unsafe {
        allocator
            .create_image(&image_create_info, &alloc_info)
            .map_err(|err| format!("Error allocating cubemap memory: {:?}", err).to_string())?
    };

    // Image View creation
    let view_create_info = vk::ImageViewCreateInfo::default()
        .view_type(vk::ImageViewType::CUBE)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: num_mips,
            base_array_layer: 0,
            layer_count: 6,
        })
        .image(image);

    let image_view = unsafe {
        device
            .create_image_view(&view_create_info, None)
            .map_err(|err| format!("Error creating cubemap view: {:?}", err).to_string())?
    };

    // Sampler creation
    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mip_lod_bias(0.0)
        .max_anisotropy(1.0)
        .min_lod(0.0)
        .max_lod(num_mips as f32)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);

    let sampler = unsafe {
        device
            .create_sampler(&sampler_create_info, None)
            .map_err(|err| format!("Error creating cubemap sampler: {:?}", err).to_string())?
    };

    let vk_image = VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: extent,
        image_format: format,
        mip_levels: num_mips,
    };

    Ok((vk_image, sampler))
}

/////////////////
// UPLOAD UTIL //
/////////////////

pub fn record_host_to_storage_buffer(
    device: &ash::Device,
    host_info: &VkHostBuffer,
    device_buffer: &VkBuffer,
    device_offset: u64,
    bytes: &[&[u8]],
    alignment: u64,
) -> Result<(), String> {
    let transfer_cmd_buffer = host_info.transfer_pool.buffers[0];
    let graphics_cmd_buffer = host_info.graphics_pool.buffers[0]; // Assuming you have a graphics command pool
    let host_buffer = &host_info.buffer;

    let alignment = if alignment < 4 { 4 } else { alignment };

    let mut total_size = 0;

    let mut host_ptr = host_buffer.alloc_info.mapped_data as *mut u8;
    for chunk in bytes {
        let size = chunk.len().next_multiple_of(alignment as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), host_ptr, size);
            host_ptr = host_ptr.add(size);
            total_size += size;
        }
    }

    let copy_info = [vk::BufferCopy::default()
        .dst_offset(device_offset)
        .src_offset(0)
        .size(total_size as vk::DeviceSize)];

    let begin_info = vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        // Record transfer command buffer
        device.begin_command_buffer(transfer_cmd_buffer, &begin_info)
            .map_err(|err| format!("Error beginning transfer buffer: {}", err))?;

        let src_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(host_buffer.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        device.cmd_pipeline_barrier(
            transfer_cmd_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[src_barrier],
            &[],
        );

        device.cmd_copy_buffer(
            transfer_cmd_buffer,
            host_buffer.buffer,
            device_buffer.buffer,
            &copy_info,
        );

        let release_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ | vk::AccessFlags::INDEX_READ)
            .src_queue_family_index(host_info.transfer_queue_index)
            .dst_queue_family_index(host_info.graphics_queue_index)
            .buffer(device_buffer.buffer)
            .offset(device_offset)
            .size(total_size as vk::DeviceSize);

        device.cmd_pipeline_barrier(
            transfer_cmd_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[release_barrier],
            &[],
        );

        device.end_command_buffer(transfer_cmd_buffer)
            .map_err(|err| format!("Error ending transfer buffer: {}", err))?;

        // Record graphics command buffer
        device.begin_command_buffer(graphics_cmd_buffer, &begin_info)
            .map_err(|err| format!("Error beginning graphics buffer: {}", err))?;

        let acquire_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ | vk::AccessFlags::INDEX_READ)
            .src_queue_family_index(host_info.transfer_queue_index)
            .dst_queue_family_index(host_info.graphics_queue_index)
            .buffer(device_buffer.buffer)
            .offset(device_offset)
            .size(total_size as vk::DeviceSize);

        device.cmd_pipeline_barrier(
            graphics_cmd_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[acquire_barrier],
            &[],
        );

        device.end_command_buffer(graphics_cmd_buffer)
            .map_err(|err| format!("Error ending graphics buffer: {}", err))?;
    }

    Ok(())
}


pub fn record_host_to_image_buffer(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    sampler_cache: &mut VkSamplerCache,
    host_info: &VkHostBuffer,
    image_meta: &[&TextureMeta],
    alignment: u64,
) -> Result<Vec<(VkImageAlloc, vk::Sampler)>, String> {
    let alignment = if alignment < 4 { 4 } else { alignment };

    let host_buffer = &host_info.buffer;
    let transfer_cmd_buffer = host_info.transfer_pool.buffers[0];
    let graphics_cmd_buffer = host_info.graphics_pool.buffers[0];

    let mut host_ptr = host_buffer.alloc_info.mapped_data as *mut u8;
    let mut offset: DeviceSize = 0;
    let image_offsets: Vec<DeviceSize> = image_meta.iter().map(|meta| {
        let size = meta.bytes.len().next_multiple_of(alignment as usize);

        unsafe {
            std::ptr::copy_nonoverlapping(meta.bytes.as_ptr(), host_ptr, size);
            let host_ptr = host_ptr.add(size);
            let curr_offset = offset;
            offset = offset.add(size as u64);
            curr_offset as DeviceSize
        }
    }).collect();


    let mut image_allocs: Vec<VkImageAlloc> = image_meta.iter().map(|meta| {
        create_image
            (device,
                allocator,
                Extent3D::default().height(meta.height).width(meta.width).depth(1),
                meta.format,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                calc_mips_count(meta.width, meta.height))
    }).collect();


    let begin_info = vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device.begin_command_buffer(transfer_cmd_buffer, &begin_info).unwrap();
        device.begin_command_buffer(graphics_cmd_buffer, &begin_info).unwrap();
    }

    // Perform buffer to image copies
    for (image_alloc, offset) in image_allocs.iter().zip(image_offsets.iter()) {
        vk_util::record_image_barrier(
            device,
            transfer_cmd_buffer,
            image_alloc.image,
            None,
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER),
            Some((vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)),
            None,
        );

        let copy_region = [vk::BufferImageCopy::default()
            .buffer_offset(*offset)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(image_alloc.image_extent)];

        unsafe {
            device.cmd_copy_buffer_to_image(
                transfer_cmd_buffer,
                host_buffer.buffer,
                image_alloc.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_region,
            );
        }

        // Record transfer->graphics barrier
        vk_util::record_image_barrier(
            device,
            transfer_cmd_buffer,
            image_alloc.image,
            None,
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER),
            Some((vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ)),
            Some((host_info.transfer_queue_index, host_info.graphics_queue_index)),
        );

        vk_util::record_image_barrier(
            device,
            graphics_cmd_buffer,
            image_alloc.image,
            None,
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER),
            Some((vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ)),
            Some((host_info.transfer_queue_index, host_info.graphics_queue_index)),
        );


        record_mip_maps_generation(
            device,
            graphics_cmd_buffer,
            image_alloc.image,
            image_alloc.image_extent.height,
            image_alloc.image_extent.width,
            image_alloc.mip_levels,
        );
    }
    
    let mut upload_images = Vec::<(VkImageAlloc, vk::Sampler)>::with_capacity(image_allocs.len());

    for image_alloc in image_allocs.into_iter() {
        let sampler_info = VkSamplerInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: LodBias::Sharp,
            anisotropy_enable: false, // FIXME, we need to implement this in engine
            max_anisotropy: 0,
            compare_enable: false,
            compare_op: Default::default(),
            min_lod: 0,
            max_lod: image_alloc.mip_levels,
            border_color: Default::default(),
            unnormalized_coordinates: false,
        };

        let sampler = sampler_cache.get_or_create_sampler(device, sampler_info);
        upload_images.push((image_alloc, sampler));
    };

    unsafe {
        device.end_command_buffer(transfer_cmd_buffer).unwrap();
        device.end_command_buffer(graphics_cmd_buffer).unwrap();
    }

    Ok(upload_images)
}


pub fn record_mip_maps_generation(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    width: u32,
    height: u32,
    mip_levels: u32,
) {
    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: i - 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: if mip_width > 1 { mip_width / 2 } else { 1 } as i32,
                    y: if mip_height > 1 { mip_height / 2 } else { 1 } as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: i,
                base_array_layer: 0,
                layer_count: 1,
            });

        let mips_subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(i)
            .level_count(1)
            .layer_count(1);

        vk_util::record_image_barrier(
            device,
            cmd_buffer,
            image,
            Some(mips_subresource),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER),
            Some((vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)),
            None,
        );


        unsafe {
            device.cmd_blit_image(
                cmd_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }

        vk_util::record_image_barrier(
            device,
            cmd_buffer,
            image,
            Some(mips_subresource),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER),
            Some((vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ)),
            None,
        );

        mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
        mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };
    }
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(mip_levels);

    vk_util::record_image_barrier(
        device,
        cmd_buffer,
        image,
        Some(subresource_range),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER),
        Some((vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ)),
        None,
    )
}


pub fn record_image_barrier(
    device: &ash::Device,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    subresource_range: Option<vk::ImageSubresourceRange>,
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
    (src_stage_mask, dst_stage_mask): (vk::PipelineStageFlags, vk::PipelineStageFlags),
    src_dst_access_mask: Option<(vk::AccessFlags, vk::AccessFlags)>,
    src_dst_queue_index: Option<((u32, u32))>,
) {
    let mut barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image);

    if let Some(range) = subresource_range {
        barrier.subresource_range = range;
    } else {
        barrier.subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);
    }


    if let Some((src_family, dst_family)) = src_dst_queue_index {
        barrier.src_queue_family_index = src_family;
        barrier.dst_queue_family_index = dst_family;
    }

    if let Some((src, dst)) = src_dst_access_mask {
        barrier.src_access_mask = src;
        barrier.dst_access_mask = dst
    } else {
        match old_layout {
            ImageLayout::UNDEFINED => barrier.src_access_mask = vk::AccessFlags::empty(),
            ImageLayout::PREINITIALIZED => barrier.src_access_mask = vk::AccessFlags::HOST_WRITE,
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => barrier.src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => barrier.src_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ImageLayout::TRANSFER_SRC_OPTIMAL => barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ,
            ImageLayout::TRANSFER_DST_OPTIMAL => barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => barrier.src_access_mask = vk::AccessFlags::SHADER_READ,
            _ => panic!("Not implemented")
        }

        match new_layout {
            ImageLayout::TRANSFER_DST_OPTIMAL => barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE,
            ImageLayout::TRANSFER_SRC_OPTIMAL => barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ,
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => barrier.dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => barrier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                if barrier.src_access_mask != vk::AccessFlags::empty() {
                    barrier.src_access_mask = vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE
                }
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ
            }
            _ => panic!("Not implemented")
        }
    }

    let barrier = [barrier];
    unsafe {
        device.cmd_pipeline_barrier(
            cmd_buffer,
            src_stage_mask,
            dst_stage_mask,
            DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }
}