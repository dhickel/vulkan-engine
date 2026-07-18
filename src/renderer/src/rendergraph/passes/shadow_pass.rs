//! Shadow map render pass.
//!
//! Renders opaque geometry depth from the directional light's perspective into
//! a 2048² D32 shadow map. Uses a minimal depth-only pipeline.

use crate::data::data_cache::VkPipelineType;
use crate::rendergraph::{RenderGraphContext, RenderPassNode};
use crate::vulkan::vk_pipeline::PushConstShadowDepth;
use crate::vulkan::vk_shadow::compute_light_view_projection;
use ash::vk;

pub struct ShadowPass;

impl RenderPassNode for ShadowPass {
    fn name(&self) -> &'static str {
        "ShadowPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        // Skip if no directional light or no geometry to render
        if ctx.submission.directional_light.is_none()
            || !ctx.submission.flags.draw_geometry
            || ctx.submission.draw_items.is_empty()
        {
            return Ok(());
        }

        let frame_index = ctx.frame.index;
        let shadow_frame = ctx.renderer.shadow_resources.get_frame(frame_index);
        let shadow_map = &shadow_frame.shadow_map;
        let shadow_map_view = shadow_frame.shadow_map_view;

        let dir_light = ctx.submission.directional_light.as_ref().unwrap();
        let light_dir = dir_light.direction.normalize();

        // Compute a conservative scene AABB from draw item transforms
        let mut aabb_min = glam::Vec3::splat(f32::INFINITY);
        let mut aabb_max = glam::Vec3::splat(f32::NEG_INFINITY);
        for item in ctx.submission.draw_items.iter() {
            let pos = item.transform.w_axis.truncate();
            aabb_min = aabb_min.min(pos - glam::Vec3::splat(2.0));
            aabb_max = aabb_max.max(pos + glam::Vec3::splat(2.0));
        }

        if aabb_min.x.is_infinite() {
            // No valid geometry - skip
            return Ok(());
        }

        let (_view, _proj, light_view_proj) =
            compute_light_view_projection(light_dir, aabb_min, aabb_max);

        let cmd_pool = ctx
            .frame
            .cmd_pools
            .get(crate::vulkan::vk_types::VkQueueType::Graphics);
        let cmd_buffer = cmd_pool.buffers[0];

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
            .image(shadow_map.image)
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
            ctx.renderer
                .device
                .cmd_pipeline_barrier2(cmd_buffer, &dep_info);
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
                extent: vk::Extent2D {
                    width: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM,
                    height: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM,
                },
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment);

        unsafe {
            ctx.renderer
                .device
                .cmd_begin_rendering(cmd_buffer, &rendering_info);
        }

        // Bind shadow pipeline
        let shadow_pipeline = ctx
            .renderer
            .vulkan_cache
            .pipelines
            .get_pipeline(VkPipelineType::ShadowDepth);
        unsafe {
            ctx.renderer.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pipeline.pipeline,
            );
        }

        // Set viewport and scissor
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM as f32,
            height: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM,
                height: crate::vulkan::vk_shadow::VkShadowResources::SHADOW_MAP_DIM,
            },
        };
        unsafe {
            ctx.renderer
                .device
                .cmd_set_viewport(cmd_buffer, 0, &[viewport]);
            ctx.renderer
                .device
                .cmd_set_scissor(cmd_buffer, 0, &[scissor]);
        }

        // Resolve geometry and draw each mesh
        let mesh_cache = ctx
            .renderer
            .data_cache
            .mesh_cache
            .lock()
            .expect("mesh_cache lock poisoned");

        for draw_item in ctx.submission.draw_items.iter() {
            let mesh = match mesh_cache.get_loaded_id(draw_item.mesh_id) {
                Ok(m) => m,
                Err(_) => continue,
            };

            unsafe {
                ctx.renderer.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    mesh.index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            let push_consts = PushConstShadowDepth {
                model_matrix: draw_item.transform,
                vertex_buffer_addr: mesh.vertex_buffer.alloc_address,
                joint_count: 0,
                pad: 0,
            };

            unsafe {
                ctx.renderer.device.cmd_push_constants(
                    cmd_buffer,
                    shadow_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push_consts),
                );

                ctx.renderer.device.cmd_draw_indexed(
                    cmd_buffer,
                    mesh.index_count,
                    1,
                    mesh.get_first_index(),
                    0,
                    0,
                );
            }
        }

        drop(mesh_cache);

        unsafe {
            ctx.renderer.device.cmd_end_rendering(cmd_buffer);
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
            .image(shadow_map.image)
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
            ctx.renderer
                .device
                .cmd_pipeline_barrier2(cmd_buffer, &dep_info);
        }

        Ok(())
    }
}
