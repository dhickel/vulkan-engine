//! Shadow map render pass.
//!
//! Renders opaque geometry depth from the directional light's perspective into
//! a shadow map. Supports two paths:
//! - Legacy: single 2048² D32 shadow map.
//! - CSM (`csm` feature): three 1024² D32 layers with per-cascade culling.

use crate::data::data_cache::VkPipelineType;
use crate::data::gpu_data::RenderObject;
#[cfg(feature = "csm")]
use crate::data::gpu_data::CSM_CASCADE_COUNT;
use crate::rendergraph::{RenderGraphContext, RenderPassNode};
use crate::vulkan::vk_pipeline::PushConstShadowDepth;
use crate::vulkan::vk_shadow::compute_draw_light_view_projection;
#[cfg(feature = "csm")]
use crate::vulkan::vk_shadow::{
    compute_csm_cascades, cull_casters_for_cascade, derive_camera_near_far_from_corners,
    frustum_corners_from_vp,
};
use ash::vk;

pub struct ShadowPass;

impl RenderPassNode for ShadowPass {
    fn name(&self) -> &'static str {
        "ShadowPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if !ctx.submission.flags.draw_geometry || ctx.submission.draw_items.is_empty() {
            return Ok(());
        }

        let mut recording = ctx.shadow_ctx();

        #[cfg(feature = "csm")]
        let frame_index = recording.frame_index();
        let shadow_draws = recording.resolve_shadow_draw_objects();

        // Copy light data before borrowing to avoid the move conflict.
        let light_data = recording
            .submission()
            .directional_lights
            .iter()
            .copied()
            .find(|light| light.enable_shadows)
            .or(recording.submission().directional_light);
        let light_active = light_data.map_or(false, |l| l.intensity > 0.0);

        if !light_active || shadow_draws.is_empty() {
            return Ok(());
        }

        let light = light_data.unwrap();
        #[cfg(feature = "csm")]
        let light_dir = light.direction;

        #[cfg(feature = "csm")]
        let cmd_buffer = recording.cmd_buffer();
        #[cfg(feature = "csm")]
        let device = recording.device();

        // CSM feature gate: when compiled, only the CSM path renders shadows;
        // the legacy single-map path is inactive. This ensures the scene
        // descriptor (which binds the CSM array view) always samples the
        // same image the shadow pass renders to.
        #[cfg(feature = "csm")]
        {
            if !light.enable_shadows {
                return Ok(());
            }
            let csm_resources = match recording.csm_shadow_resources() {
                Some(r) => r,
                None => return Ok(()),
            };
            let csm_frame = csm_resources.get_frame(frame_index);
            let csm_extent = csm_resources.extent;
            let csm_image = csm_frame.csm_image.image;

            let camera = &recording.submission().camera;
            let view = camera.view;
            let projection = camera.projection;

            let vp = projection * view;
            let Some(corners) = frustum_corners_from_vp(&vp) else {
                return Ok(());
            };
            let (camera_near, camera_far) = derive_camera_near_far_from_corners(&view, &corners);
            let camera_far = camera_far.min(crate::vulkan::vk_shadow::CSM_MAX_DISTANCE);

            let cascades = match compute_csm_cascades(
                &view,
                &projection,
                light_dir,
                camera_near,
                camera_far,
                &shadow_draws,
            ) {
                Some(c) => c,
                None => return Ok(()),
            };

            log::debug!(
                "CSM frame {}: cascades={} candidates/emitted={:?}",
                frame_index,
                cascades.len(),
                cascades
                    .iter()
                    .map(|cascade| (cascade.candidate_casters, cascade.emitted_casters))
                    .collect::<Vec<_>>()
            );

            // Transition entire array to DEPTH_ATTACHMENT_OPTIMAL.
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
                .image(csm_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: CSM_CASCADE_COUNT,
                });

            let barriers = [barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe {
                device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
            }

            // Bind shadow pipeline once.
            let shadow_pipeline = recording
                .vulkan_cache()
                .pipelines
                .get_pipeline(VkPipelineType::ShadowDepth);
            unsafe {
                device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    shadow_pipeline.pipeline,
                );
            }

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: csm_extent.width as f32,
                height: csm_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: csm_extent,
            };
            unsafe {
                device.cmd_set_viewport(cmd_buffer, 0, &[viewport]);
                device.cmd_set_scissor(cmd_buffer, 0, &[scissor]);
            }

            // Render each cascade layer.
            for (i, cascade) in cascades.iter().enumerate() {
                let layer_idx = i as u32;

                let visible =
                    cull_casters_for_cascade(shadow_draws.iter(), &cascade.light_view_proj);

                let layer_view = csm_frame.csm_layer_views[layer_idx as usize];
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(layer_view)
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
                        extent: csm_extent,
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                unsafe {
                    device.cmd_begin_rendering(cmd_buffer, &rendering_info);
                }

                for draw in &visible {
                    unsafe {
                        device.cmd_bind_index_buffer(
                            cmd_buffer,
                            draw.index_buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                    }

                    let push_consts = PushConstShadowDepth {
                        light_model_view_projection: cascade.light_view_proj * draw.transform,
                        vertex_buffer_addr: draw.vertex_buffer_addr,
                        _pad: [0; 2],
                    };

                    unsafe {
                        device.cmd_push_constants(
                            cmd_buffer,
                            shadow_pipeline.layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytemuck::bytes_of(&push_consts),
                        );

                        device.cmd_draw_indexed(
                            cmd_buffer,
                            draw.index_count,
                            1,
                            draw.first_index,
                            0,
                            0,
                        );
                    }
                }

                unsafe {
                    device.cmd_end_rendering(cmd_buffer);
                }

                // Barrier: depth write → depth read for next layer.
                if i + 1 < CSM_CASCADE_COUNT as usize {
                    let layer_barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                        )
                        .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .image(csm_image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: layer_idx,
                            layer_count: 1,
                        });

                    let layer_barriers = [layer_barrier];
                    let dep_info =
                        vk::DependencyInfo::default().image_memory_barriers(&layer_barriers);
                    unsafe {
                        device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
                    }
                }
            }

            // Final transition: all layers to SHADER_READ_ONLY_OPTIMAL.
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
                .image(csm_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: CSM_CASCADE_COUNT,
                });

            let read_barriers = [read_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&read_barriers);
            unsafe {
                device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
            }

            return Ok(());
        }

        // Legacy single-map path (when CSM feature is not compiled).
        #[cfg(not(feature = "csm"))]
        self.execute_legacy(recording, &shadow_draws, &light)
    }
}

impl ShadowPass {
    fn execute_legacy(
        &self,
        recording: crate::vulkan::vk_commands::ShadowRecording<'_>,
        shadow_draws: &[RenderObject],
        light: &crate::scene::render_submission::FrameDirectionalLight,
    ) -> Result<(), String> {
        let frame_index = recording.frame_index();
        let shadow_extent = recording.shadow_resources().shadow_map_extent;
        let (shadow_map_image, shadow_map_view) = {
            let shadow_frame = recording.shadow_resources().get_frame(frame_index);
            (shadow_frame.shadow_map.image, shadow_frame.shadow_map_view)
        };

        let light_view_proj =
            compute_draw_light_view_projection(light.direction, shadow_draws.iter());
        log::debug!(
            "legacy shadow frame {}: draws={} fitted={}",
            frame_index,
            shadow_draws.len(),
            light_view_proj.is_some()
        );

        let cmd_buffer = recording.cmd_buffer();
        let device = recording.device();

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
            device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
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
            device.cmd_begin_rendering(cmd_buffer, &rendering_info);
        }

        // Bind shadow pipeline
        let shadow_pipeline = recording
            .vulkan_cache()
            .pipelines
            .get_pipeline(VkPipelineType::ShadowDepth);
        unsafe {
            device.cmd_bind_pipeline(
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
            device.cmd_set_viewport(cmd_buffer, 0, &[viewport]);
            device.cmd_set_scissor(cmd_buffer, 0, &[scissor]);
        }

        if let Some(light_view_proj) = light_view_proj {
            for draw in shadow_draws {
                unsafe {
                    device.cmd_bind_index_buffer(
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
                    device.cmd_push_constants(
                        cmd_buffer,
                        shadow_pipeline.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&push_consts),
                    );

                    device.cmd_draw_indexed(
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
            device.cmd_end_rendering(cmd_buffer);
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
            device.cmd_pipeline_barrier2(cmd_buffer, &dep_info);
        }

        Ok(())
    }
}
