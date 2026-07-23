//! # BSP Vulkan Integration
//!
//! Feature-gated (`renderer/bsp`) BSP pipeline layout construction, pipeline
//! creation, material descriptor allocation, and lightmap atlas upload.
//!
//! This module is compiled only from `vk_pipeline.rs` when the `bsp` feature is
//! active.

use crate::data::data_cache::{CoreShaderType, VkDescLayoutCache, VkDescType, VkPipelineType};
use crate::vulkan::vk_pipeline::{create_pipeline_from_spec, BlendingMode, PipelineSpec};
use crate::vulkan::vk_types::VkImageAlloc;
use crate::vulkan::vk_util;
use ash::vk;
use vk_mem::{Alloc, Allocator};

// ── BSP pipeline layout helper ─────────────────────────────────────────

/// Build a BSP pipeline layout with set 0 (scene), set 1 (material), and set 2 (frame values).
#[cfg(feature = "bsp")]
pub(crate) fn create_bsp_pipeline_layout(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
) -> Result<vk::PipelineLayout, String> {
    let set_layouts = [
        desc_layout_cache.get(VkDescType::BspScene),
        desc_layout_cache.get(VkDescType::BspMaterial),
        desc_layout_cache.get(VkDescType::BspFrameValues),
    ];

    // BSP push constants: mat4 model (64 bytes) + vertex_buffer_addr (8 bytes),
    // rounded to the renderer's 16-byte push-constant ABI boundary.
    let push_const_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<crate::data::gpu_data::BspModelPushConsts>() as u32);

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_const_range));

    unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|err| format!("failed to create BSP pipeline layout: {err:?}"))
    }
}

// ── BSP material descriptor helpers ────────────────────────────────────

/// Write a BSP material descriptor set (set 1) with the provided bindings.
///
/// Writes to an already-allocated `set`. Does NOT allocate the set.
///
/// Binds:
/// - b0: albedo texture (combined image sampler)
/// - b1: fullbright mask (combined image sampler)
/// - b2: lightmap atlas (combined image sampler, 2D array)
/// - b3: surface UBO (BspSurfaceUniform, 80 bytes)
#[cfg(feature = "bsp")]
pub(crate) fn write_bsp_material_descriptor(
    device: &ash::Device,
    set: vk::DescriptorSet,
    albedo_view: vk::ImageView,
    albedo_sampler: vk::Sampler,
    fullbright_view: vk::ImageView,
    fullbright_sampler: vk::Sampler,
    lightmap_view: vk::ImageView,
    lightmap_sampler: vk::Sampler,
    surf_ubo_buffer: vk::Buffer,
    surf_ubo_offset: u64,
    surf_ubo_range: u64,
) {
    let albedo_info = vk::DescriptorImageInfo::default()
        .image_view(albedo_view)
        .sampler(albedo_sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let fullbright_info = vk::DescriptorImageInfo::default()
        .image_view(fullbright_view)
        .sampler(fullbright_sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let lightmap_info = vk::DescriptorImageInfo::default()
        .image_view(lightmap_view)
        .sampler(lightmap_sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let ubo_info = vk::DescriptorBufferInfo::default()
        .buffer(surf_ubo_buffer)
        .offset(surf_ubo_offset)
        .range(surf_ubo_range);

    let write_descriptors = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&albedo_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&fullbright_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&lightmap_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&ubo_info)),
    ];

    unsafe {
        device.update_descriptor_sets(&write_descriptors, &[]);
    }
}

/// Allocate a single BSP material descriptor set from a pool.
#[cfg(feature = "bsp")]
pub(crate) fn allocate_bsp_material_set(
    device: &ash::Device,
    material_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
) -> Result<vk::DescriptorSet, String> {
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&material_set_layout));

    let sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|err| format!("failed to allocate BSP material set: {err:?}"))?
    };
    Ok(sets[0])
}

/// Create a descriptor pool suitable for BSP material descriptor sets.
///
/// Each BSP material set uses 3× COMBINED_IMAGE_SAMPLER + 1× UNIFORM_BUFFER.
/// The pool is sized for `max_sets` allocations.
#[cfg(feature = "bsp")]
pub(crate) fn create_bsp_material_descriptor_pool(
    device: &ash::Device,
    max_sets: u32,
) -> Result<vk::DescriptorPool, String> {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_sets * 3),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(max_sets),
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(max_sets)
        .pool_sizes(&pool_sizes);

    unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|err| format!("failed to create BSP material descriptor pool: {err:?}"))
    }
}

// ── Lightmap atlas image creation ──────────────────────────────────────

/// Create a 2D-array image for the lightmap atlas.
///
/// Returns the image allocation (with an array image view) and a sampler.
#[cfg(feature = "bsp")]
pub(crate) fn create_lightmap_atlas_image(
    device: &ash::Device,
    allocator: &Allocator,
    width: u32,
    height: u32,
    layer_count: u32,
) -> Result<VkImageAlloc, String> {
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(extent)
        .mip_levels(1)
        .array_layers(layer_count)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ..Default::default()
    };

    let (image, allocation) = unsafe {
        allocator
            .create_image(&image_info, &alloc_info)
            .map_err(|e| format!("failed to allocate lightmap atlas image: {e:?}"))?
    };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(layer_count),
        );

    let image_view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("failed to create lightmap atlas image view: {e:?}"))?
    };

    // NOTE: if image_view creation failed above, we leak the image allocation.
    // This is acceptable because the error will propagate and the caller will
    // drop the allocator (which owns the image via VMA).

    Ok(VkImageAlloc {
        image,
        image_view,
        allocation,
        image_extent: extent,
        image_format: vk::Format::R8G8B8A8_UNORM,
        mip_levels: 1,
    })
}

/// Create a default sampler suitable for the lightmap atlas.
#[cfg(feature = "bsp")]
pub(crate) fn create_lightmap_sampler(device: &ash::Device) -> Result<vk::Sampler, String> {
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .mip_lod_bias(0.0)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .compare_enable(false)
        .min_lod(0.0)
        .max_lod(0.0)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        .unnormalized_coordinates(false);

    unsafe {
        device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("failed to create lightmap sampler: {e:?}"))
    }
}

/// Upload RGBA8 pixel data to the lightmap atlas image through a staging buffer.
///
/// This records and submits a one-time transfer command buffer and waits for
/// completion. The image must have been created with UNDEFINED layout.
#[cfg(feature = "bsp")]
pub(crate) fn upload_lightmap_atlas_data(
    device: &ash::Device,
    allocator: &Allocator,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    image: vk::Image,
    width: u32,
    height: u32,
    layer_count: u32,
    rgba_data: &[u8],
) -> Result<(), String> {
    let expected_size = (width * height * layer_count * 4) as usize;
    if rgba_data.len() < expected_size {
        return Err(format!(
            "lightmap atlas data too small: have {} bytes, need {expected_size}",
            rgba_data.len()
        ));
    }

    let total_bytes = expected_size as u64;

    // Create staging buffer.
    let staging_info = vk::BufferCreateInfo::default()
        .size(total_bytes)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferHost,
        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        ..Default::default()
    };

    let (staging_buffer, mut staging_allocation) = unsafe {
        allocator
            .create_buffer(&staging_info, &staging_alloc_info)
            .map_err(|e| format!("failed to create lightmap staging buffer: {e:?}"))?
    };

    // Copy data to staging buffer.
    unsafe {
        let mapped = allocator
            .map_memory(&mut staging_allocation)
            .map_err(|e| format!("failed to map lightmap staging memory: {e:?}"))?;
        std::ptr::copy_nonoverlapping(rgba_data.as_ptr(), mapped as *mut u8, expected_size);
        allocator.unmap_memory(&mut staging_allocation);
    }

    // Allocate and record command buffer.
    let cmd_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd = unsafe {
        device
            .allocate_command_buffers(&cmd_alloc)
            .map_err(|e| format!("failed to allocate lightmap upload cmd: {e:?}"))?[0]
    };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(cmd, &begin_info)
            .map_err(|e| format!("failed to begin lightmap upload cmd: {e:?}"))?;
    }

    // Transition UNDEFINED → TRANSFER_DST_OPTIMAL inline.
    vk_util::transition_image_layered(
        device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        layer_count,
        1,
    );

    // Buffer image copy for each layer.
    let buffer_copy_regions: Vec<vk::BufferImageCopy> = (0..layer_count)
        .map(|layer| {
            vk::BufferImageCopy::default()
                .buffer_offset((layer * width * height * 4) as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
        })
        .collect();

    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &buffer_copy_regions,
        );
    }

    // Transition TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL inline.
    vk_util::transition_image_layered(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        layer_count,
        1,
    );

    unsafe {
        device
            .end_command_buffer(cmd)
            .map_err(|e| format!("failed to end lightmap upload cmd: {e:?}"))?;
    }

    // Submit.
    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe {
        device
            .create_fence(&fence_info, None)
            .map_err(|e| format!("failed to create lightmap upload fence: {e:?}"))?
    };

    unsafe {
        device
            .queue_submit(transfer_queue, &[submit_info], fence)
            .map_err(|e| format!("failed to submit lightmap upload: {e:?}"))?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("failed to wait for lightmap upload: {e:?}"))?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(command_pool, &[cmd]);
    }

    // Free staging buffer.
    unsafe {
        allocator.destroy_buffer(staging_buffer, &mut staging_allocation);
    }

    Ok(())
}

struct BspPipelineSpec {
    pipeline_type: VkPipelineType,
    frag_module: vk::ShaderModule,
    depth_test: (bool, vk::CompareOp),
    cull_mode: vk::CullModeFlags,
    blend: BlendingMode,
}

/// Create all five BSP pipeline variants.
///
/// Returns `(pipelines, shared_layout)`. On any failure, every pipeline created
/// in this function and the shared layout are destroyed before the error is
/// returned.
#[cfg(feature = "bsp")]
pub fn create_bsp_pipelines(
    device: &ash::Device,
    shader_modules: &[vk::ShaderModule; CoreShaderType::COUNT],
    desc_layout_cache: &VkDescLayoutCache,
    color_attachment_format: vk::Format,
    depth_attachment_format: vk::Format,
) -> Result<(Vec<(VkPipelineType, vk::Pipeline)>, vk::PipelineLayout), String> {
    let bsp_vs = shader_modules[CoreShaderType::BspLightmappedVert as usize];
    let bsp_lightmapped_fs = shader_modules[CoreShaderType::BspLightmappedFrag as usize];
    let bsp_sky_fs = shader_modules[CoreShaderType::BspSkyFrag as usize];
    let bsp_liquid_fs = shader_modules[CoreShaderType::BspLiquidFrag as usize];

    let layout = create_bsp_pipeline_layout(device, desc_layout_cache)?;

    let specs = [
        BspPipelineSpec {
            pipeline_type: VkPipelineType::BspOpaque,
            frag_module: bsp_lightmapped_fs,
            depth_test: (true, vk::CompareOp::LESS),
            cull_mode: vk::CullModeFlags::BACK,
            blend: BlendingMode::Disabled,
        },
        BspPipelineSpec {
            pipeline_type: VkPipelineType::BspFullbright,
            frag_module: bsp_lightmapped_fs,
            depth_test: (true, vk::CompareOp::LESS),
            cull_mode: vk::CullModeFlags::BACK,
            blend: BlendingMode::Disabled,
        },
        BspPipelineSpec {
            pipeline_type: VkPipelineType::BspAlphaMask,
            frag_module: bsp_lightmapped_fs,
            depth_test: (true, vk::CompareOp::LESS),
            cull_mode: vk::CullModeFlags::NONE,
            blend: BlendingMode::Disabled,
        },
        BspPipelineSpec {
            pipeline_type: VkPipelineType::BspSky,
            frag_module: bsp_sky_fs,
            depth_test: (false, vk::CompareOp::LESS),
            cull_mode: vk::CullModeFlags::BACK,
            blend: BlendingMode::Disabled,
        },
        BspPipelineSpec {
            pipeline_type: VkPipelineType::BspLiquid,
            frag_module: bsp_liquid_fs,
            depth_test: (false, vk::CompareOp::LESS),
            cull_mode: vk::CullModeFlags::NONE,
            blend: BlendingMode::AlphaBlend,
        },
    ];

    let mut pipelines = Vec::with_capacity(specs.len());
    for spec in specs {
        let pipeline_spec = PipelineSpec {
            vert_module: bsp_vs,
            frag_module: spec.frag_module,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: spec.cull_mode,
            front_face: vk::FrontFace::CLOCKWISE,
            color_attachment_format: Some(color_attachment_format),
            depth_format: Some(depth_attachment_format),
            depth_test: Some(spec.depth_test),
            blend: spec.blend,
            layout,
        };

        match create_pipeline_from_spec(device, &pipeline_spec) {
            Ok(pipeline) => pipelines.push((spec.pipeline_type, pipeline)),
            Err(err) => {
                unsafe {
                    for (_, pipeline) in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    device.destroy_pipeline_layout(layout, None);
                }
                return Err(err);
            }
        }
    }

    Ok((pipelines, layout))
}
