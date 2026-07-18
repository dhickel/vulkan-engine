//! Shadow map render pass.
//!
//! Renders opaque geometry depth from the directional light's perspective into
//! a 2048² D32 shadow map. Uses a minimal depth-only pipeline.

use crate::data::data_cache::VkPipelineType;
use crate::rendergraph::{RenderGraphContext, RenderPassNode};
use crate::vulkan::vk_pipeline::PushConstShadowDepth;
use crate::vulkan::vk_shadow::compute_draw_light_view_projection;
use ash::vk;

pub struct ShadowPass;

impl RenderPassNode for ShadowPass {
    fn name(&self) -> &'static str {
        "ShadowPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        // The map must be cleared and transitioned before a PBR draw can bind it,
        // even when this frame has no active directional light.
        if !ctx.submission.flags.draw_geometry || ctx.submission.draw_items.is_empty() {
            return Ok(());
        }

        let mut recording = ctx.shadow_ctx();

        let frame_index = recording.frame_index();
        let shadow_extent = recording.shadow_resources().shadow_map_extent;
        let (shadow_map_image, shadow_map_view) = {
            let shadow_frame = recording.shadow_resources().get_frame(frame_index);
            (shadow_frame.shadow_map.image, shadow_frame.shadow_map_view)
        };

        let shadow_draws = recording.resolve_shadow_draw_objects();
        let light_view_proj = recording
            .submission()
            .directional_light
            .as_ref()
            .filter(|light| light.intensity > 0.0)
            .and_then(|light| {
                compute_draw_light_view_projection(light.direction, shadow_draws.iter())
            });

        let cmd_buffer = recording.cmd_buffer();

        // Transition shadow map to depth attachment optimal
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image(shadow_map_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let barriers = [barrier];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe {
            recording.device().cmd_pipeline_barrier2(cmd_buffer, &dep_info);
        }

        // Begin depth-only rendering
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(shadow_map_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: shadow_extent,
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment);

        unsafe {
            recording.device().cmd_begin_rendering(cmd_buffer, &rendering_info);
        }

        // Bind shadow pipeline
        let shadow_pipeline = recording
            .vulkan_cache()
            .pipelines
            .get_pipeline(VkPipelineType::ShadowDepth);
        unsafe {
            recording.device().cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pipeline.pipeline,
            );
        }

        // Set viewport and scissor
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: shadow_extent.width as f32,
            height: shadow_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: shadow_extent,
        };
        unsafe {
            recording.device().cmd_set_viewport(cmd_buffer, 0, &[viewport]);
            recording.device().cmd_set_scissor(cmd_buffer, 0, &[scissor]);
        }

        if let Some(light_view_proj) = light_view_proj {
            for draw in &shadow_draws {
                unsafe {
                    recording.device().cmd_bind_index_buffer(
                        cmd_buffer,
                        draw.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                let push_consts = PushConstShadowDepth {
                    light_model_view_projection: light_view_proj * draw.transform,
                    vertex_buffer_addr: draw.vertex_buffer_addr,
                    _pad: [0; 2],
                };

                unsafe {
                    recording.device().cmd_push_constants(
                        cmd_buffer,
                        shadow_pipeline.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&push_consts),
                    );

                    recording.device().cmd_draw_indexed(
                        cmd_buffer,
                        draw.index_count,
                        1,
                        draw.first_index,
                        0,
                        0,
                    );
                }
            }
        }

        unsafe {
            recording.device().cmd_end_rendering(cmd_buffer);
        }

        // Transition shadow map to SHADER_READ_ONLY_OPTIMAL for sampling
        let read_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(shadow_map_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let read_barriers = [read_barrier];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&read_barriers);
        unsafe {
            recording.device().cmd_pipeline_barrier2(cmd_buffer, &dep_info);
        }

        Ok(())
    }
}
