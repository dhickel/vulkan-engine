//! # BSP Vulkan Integration
//!
//! Feature-gated (`renderer/bsp`) BSP pipeline layout construction and pipeline
//! creation for the five BSP material variants.
//!
//! This module is compiled only from `vk_pipeline.rs` when the `bsp` feature is
//! active.

use super::{create_pipeline_from_spec, BlendingMode, PipelineSpec};
use crate::data::data_cache::{CoreShaderType, VkDescLayoutCache, VkDescType, VkPipelineType};
use ash::vk;

// ── BSP pipeline layout helper ─────────────────────────────────────────

/// Build a BSP pipeline layout with set 0 (scene) and set 1 (material).
#[cfg(feature = "bsp")]
fn create_bsp_pipeline_layout(
    device: &ash::Device,
    desc_layout_cache: &VkDescLayoutCache,
) -> Result<vk::PipelineLayout, String> {
    let set_layouts = [
        desc_layout_cache.get(VkDescType::BspScene),
        desc_layout_cache.get(VkDescType::BspMaterial),
    ];

    // BSP push constants: mat4 model (64 bytes) + vertex_buffer_addr (8 bytes),
    // rounded to the renderer's 16-byte push-constant ABI boundary.
    let push_const_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80);

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_const_range));

    unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|err| format!("failed to create BSP pipeline layout: {err:?}"))
    }
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
            depth_test: (true, vk::CompareOp::ALWAYS),
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
