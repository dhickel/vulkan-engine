use std::cmp::{max, PartialEq};
use std::ffi::CStr;

use crate::data::gpu_data::{TextureMeta, Vertex, VkCubeMap, VkGpuMeshBuffers};
use crate::vulkan::vk_types::*;
use ash::vk;
use ash::vk::{
    AccessFlags2, ClearValue, Extent2D, Extent3D, ImageType, PipelineCache,
    PipelineLayoutCreateInfo, PipelineStageFlags2, Rect2D, RenderingInfo,
};
use log::info;
use std::io::{Read, Seek, SeekFrom};
use std::mem::align_of;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use vk_mem::{Alloc, Allocator};

use crate::data::data_cache::{
    LodBias, SamplerCache, VkDescLayoutCache, VkDescType, VkPipelineCache, VkPipelineType,
    VkSamplerInfo,
};
use crate::data::data_util;
use crate::vulkan::vk_descriptor::{PoolSizeRatio, VkDescriptorAllocator};
use crate::vulkan::vk_render::{DataCache, VkSingleDescriptor};
use crate::vulkan::{vk_init, vk_util};
use shaderc::{CompileOptions, Compiler, ShaderKind};
use std::fs;
use std::path::{Path, PathBuf};

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
        data_util::calc_mips_count(size.width, size.height)
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
        mip_levels: mips,
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

pub fn allocate_and_write_buffer(
    allocator: &Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<VkBuffer, String> {
    let buffer_size = data.len();
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

pub fn destroy_mesh_buffers(allocator: &Allocator, mut buffer: VkGpuMeshBuffers) {
    unsafe {
        allocator.destroy_buffer(
            buffer.index_buffer.buffer,
            &mut buffer.index_buffer.allocation,
        );
        allocator.destroy_buffer(
            buffer.vertex_buffer.buffer,
            &mut buffer.vertex_buffer.allocation,
        )
    }
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
    cmd_pool: &VkCommandPool,
) -> VkBrdfLut {
    info!("Generating BRDF LUT");
    let start = SystemTime::now();

    let cmd_buffer = *cmd_pool.buffers.get(0).unwrap();
    let queue = cmd_pool.queue;

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
        false,
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
            .begin_command_buffer(cmd_buffer, &begin_info)
            .unwrap();

        transition_image(
            device,
            cmd_buffer,
            brd_img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        device.cmd_begin_rendering(cmd_buffer, &rendering_info);

        device.cmd_set_viewport(cmd_buffer, 0, &[viewport]);
        device.cmd_set_scissor(cmd_buffer, 0, &[scissor]);
        device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        device.cmd_draw(cmd_buffer, 3, 1, 0, 0);

        device.cmd_end_rendering(cmd_buffer);

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
            cmd_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        device.end_command_buffer(cmd_buffer).unwrap();

        let cmd_info = [command_buffer_submit_info(cmd_buffer)];
        let submit_info = [submit_info_2(&cmd_info, &[], &[])];
        device
            .queue_submit2(queue, &submit_info, vk::Fence::null())
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
    pipeline: vk::Pipeline,
    cmd_pool: &VkCommandPool,
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

    let cmd_buffer = *cmd_pool.buffers.get(0).unwrap();
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
            .queue_submit2(cmd_pool.queue, &submit_info, vk::Fence::null())
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

pub fn immediate_submit<F>(device: &ash::Device, immediate: &VkImmediate, function: F)
where
    F: FnOnce(&ash::Device, vk::CommandBuffer),
{
    unsafe {
        let cmd_buffer = immediate.command_pool.buffers[0];
        let queue = immediate.command_pool.queue;

        // Reset the fence correctly
        device.reset_fences(&immediate.fence).unwrap();

        device
            .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
            .unwrap();

        let begin_info =
            vk_util::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(cmd_buffer, &begin_info)
            .unwrap();

        function(device, cmd_buffer);

        device.end_command_buffer(cmd_buffer).unwrap();

        let cmd_info = [vk_util::command_buffer_submit_info(cmd_buffer)];
        let submit_info = [vk_util::submit_info_2(&cmd_info, &[], &[])];

        // Submit the command buffer and signal the fence correctly
        device
            .queue_submit2(queue, &submit_info, immediate.fence[0])
            .unwrap();

        // Wait for the fence to be signaled
        device
            .wait_for_fences(&immediate.fence, true, u64::MAX)
            .unwrap();
    }
}

pub fn upload_mesh(
    device: &ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    immediate: &VkImmediate,
    indices: &[u32],
    vertices: &[Vertex],
) -> VkGpuMeshBuffers {
    let i_buffer_size = indices.len() * std::mem::size_of::<u32>();
    let v_buffer_size = vertices.len() * std::mem::size_of::<Vertex>();

    let allocator = allocator.lock().unwrap();

    let index_buffer = vk_util::allocate_buffer(
        &allocator,
        i_buffer_size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk_mem::MemoryUsage::AutoPreferDevice,
    )
    .expect("Failed to allocate index buffer");

    let vertex_buffer = vk_util::allocate_buffer(
        &allocator,
        v_buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk_mem::MemoryUsage::AutoPreferDevice,
    )
    .expect("Failed to allocate vertex buffer");

    let v_buffer_addr_info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);

    let vertex_buffer_addr = unsafe { device.get_buffer_device_address(&v_buffer_addr_info) };

    let new_surface = VkGpuMeshBuffers {
        index_buffer,
        vertex_buffer,
        vertex_buffer_addr,
    };

    let staging_buffer = vk_util::allocate_buffer(
        &allocator,
        v_buffer_size + i_buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk_mem::MemoryUsage::AutoPreferHost,
    )
    .expect("Failed to allocate staging buffer");

    let staging_data = staging_buffer.alloc_info.mapped_data as *mut u8;

    unsafe {
        // Copy vertex data to the staging buffer
        std::ptr::copy_nonoverlapping(vertices.as_ptr() as *const u8, staging_data, v_buffer_size);

        // Copy index data to the staging buffer
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            staging_data.add(v_buffer_size),
            i_buffer_size,
        );
    }

    immediate_submit(device, immediate, |device, cmd| {
        let vertex_copy = [vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(v_buffer_size as vk::DeviceSize)];

        let index_copy = [vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(v_buffer_size as vk::DeviceSize)
            .size(i_buffer_size as vk::DeviceSize)];

        unsafe {
            device.cmd_copy_buffer(
                cmd,
                staging_buffer.buffer,
                new_surface.vertex_buffer.buffer,
                &vertex_copy,
            );
        }

        unsafe {
            device.cmd_copy_buffer(
                cmd,
                staging_buffer.buffer,
                new_surface.index_buffer.buffer,
                &index_copy,
            );
        }
    });

    vk_util::destroy_buffer(&allocator, staging_buffer);

    new_surface
}

pub fn upload_image(
    device: &ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    immediate: &VkImmediate,
    sampler_cache: &mut SamplerCache,
    data: &[u8],
    size: vk::Extent3D,
    format: vk::Format,
    usage_flags: vk::ImageUsageFlags,
    mip_mapped: bool,
) -> (VkImageAlloc, vk::Sampler) {
    let allocator = allocator.lock().unwrap();

    // Calculate mip levels
    let mip_levels = if mip_mapped {
        (size.width.max(size.height) as f32).log2().floor() as u32 + 1
    } else {
        1
    };

    let upload_buffer =
        vk_util::allocate_and_write_buffer(&allocator, data, vk::BufferUsageFlags::TRANSFER_SRC)
            .unwrap();

    let new_image = create_image(
        device,
        &allocator,
        size,
        format,
        usage_flags | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_mapped,
    );

    immediate_submit(device, immediate, |device, cmd| {
        vk_util::transition_image(
            device,
            cmd,
            new_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let copy_region = [vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(size)];

        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                upload_buffer.buffer,
                new_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_region,
            );
        }

        // Generate mipmaps
        if mip_levels > 1 {
            let mut mip_width = size.width;
            let mut mip_height = size.height;

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
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i - 1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: if mip_width > 1 { mip_width / 2 } else { 1 } as i32,
                            y: if mip_height > 1 { mip_height / 2 } else { 1 } as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe {
                    device.cmd_blit_image(
                        cmd,
                        new_image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit],
                        vk::Filter::LINEAR,
                    );
                }

                // Update dimensions for next iteration
                mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
                mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };
            }
        }

        vk_util::transition_image(
            device,
            cmd,
            new_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    });
    vk_util::destroy_buffer(&allocator, upload_buffer);

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
        max_lod: mip_levels,
        border_color: Default::default(),
        unnormalized_coordinates: false,
    };

    let sampler = sampler_cache.get_or_create_sampler(device, sampler_info);

    (new_image, sampler)
}
