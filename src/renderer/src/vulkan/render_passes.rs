use ash::vk;
use crate::vulkan::vk_types::*;
use crate::vulkan::render_graph::{RenderPass, RenderPassContext};
use crate::vulkan::vk_util;
use crate::data::data_cache::{MeshCache, VkPipelineType};
use crate::data::gpu_data::{VkModelPushConsts, AsByteSlice};

pub struct GeometryPass;

impl RenderPass for GeometryPass {
    fn name(&self) -> &str {
        "Geometry Pass"
    }

    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        let curr_frame = context.frame;
        let frame_index = curr_frame.index;

        // Transition images
        vk_util::transition_image(
            context.device,
            cmd,
            curr_frame.draw.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        vk_util::transition_image(
            context.device,
            cmd,
            curr_frame.depth.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        vk_util::transition_image(
            context.device,
            cmd,
            curr_frame.draw.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let color_attachment = [vk_util::attachment_info(
            curr_frame.draw.image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Some(clear_value),
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            curr_frame.depth.image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let extent = context.window_state.get_curr_extent();

        let rendering_info =
            vk_util::rendering_info(extent, &color_attachment, Some(&depth_attachment));

        unsafe {
            context.device.cmd_begin_rendering(cmd, &rendering_info);

            // Write the new scene view/pos
            let scene_desc = context.scene_descriptors.as_mut().unwrap()
                .update_scene_uniform(context.device, *context.scene_data, frame_index);

            // Initial setup - though loop handles bindings, we might need initial state if loop is empty?
            // Loop handles everything.

            let mut curr_joint_desc = context.data_cache.mesh_cache.lock().unwrap().get_default_joint_desc();

            context.device
                .cmd_set_viewport(cmd, 0, context.window_state.get_viewport());
            context.device
                .cmd_set_scissor(cmd, 0, context.window_state.get_scissor());


            let device = context.device;
            let pipelines = context.pipelines;

            let active_pipelines = &context.render_context.draw_context.active_pipelines;
            let render_objects = &context.render_context.draw_context.render_objects;

            for pipeline_type in active_pipelines {
                let mat_pipeline = pipelines.get_pipeline(*pipeline_type);

                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    mat_pipeline.pipeline,
                );

                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    mat_pipeline.layout,
                    0,
                    &[scene_desc],
                    &[],
                );

                let mut force_rebind_joints = true;

                for obj in render_objects.get_unchecked(*pipeline_type as usize) {
                    let material = &(*obj.material);

                    // Joint desc
                    if force_rebind_joints || obj.joint_desc != curr_joint_desc {
                        curr_joint_desc = obj.joint_desc;
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            mat_pipeline.layout,
                            1,
                            &[curr_joint_desc],
                            &[],
                        );
                        force_rebind_joints = false;
                    }

                    // Material desc (set 2)
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        mat_pipeline.layout,
                        2,
                        &[material.image_descriptor],
                        &[],
                    );

                    device.cmd_bind_index_buffer(
                        cmd,
                        obj.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    let push_consts = VkModelPushConsts::new(
                        obj.transform,
                        obj.vertex_buffer_addr,
                        material.meta_alloc.alloc_address,
                    );

                    device.cmd_push_constants(
                        cmd,
                        mat_pipeline.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        push_consts.as_byte_slice(),
                    );

                    device.cmd_draw_indexed(cmd, obj.index_count, 1, obj.first_index, 0, 0);
                }
            }

            context.device.cmd_end_rendering(cmd);
        }

        context.render_context.draw_context.clear();
    }
}

pub struct SkyboxPass;

impl RenderPass for SkyboxPass {
    fn name(&self) -> &str { "Skybox Pass" }

    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        let curr_frame = context.frame;

        let color_attachment = [vk_util::attachment_info(
            curr_frame.draw.image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(curr_frame.depth.image_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let extent = context.window_state.get_curr_extent();

        let rendering_info = vk_util::rendering_info(extent, &color_attachment, Some(&depth_attachment));

        let skybox_pipeline = context.pipelines.get_pipeline(VkPipelineType::Skybox);
        let skybox_desc = if let Some(desc) = &context.render_context.sky_box.descriptor {
            desc.descriptor
        } else {
            panic!("No skybox descriptor")
        };

        context.render_context.sky_box.skybox_consts.projection = context.scene_data.projection;
        context.render_context.sky_box.skybox_consts.model = context.scene_data.view;

        let mesh = context.data_cache.mesh_cache.lock().unwrap().get_loaded_id_unchecked(MeshCache::SKYBOX_MESH);

        unsafe {
            context.device.cmd_begin_rendering(cmd, &rendering_info);

            context.device
                .cmd_set_viewport(cmd, 0, context.window_state.get_viewport());
            context.device
                .cmd_set_scissor(cmd, 0, context.window_state.get_scissor());

            context.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                skybox_pipeline.pipeline,
            );

            context.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                skybox_pipeline.layout,
                0,
                &skybox_desc,
                &[],
            );

            context.device.cmd_bind_index_buffer(
                cmd,
                mesh.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            context.device.cmd_push_constants(
                cmd,
                skybox_pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                context.render_context.sky_box.skybox_consts.as_byte_slice(),
            );

            context.device
                .cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);

            context.device.cmd_end_rendering(cmd);
        }
    }
}

pub struct CopyPass;

impl RenderPass for CopyPass {
    fn name(&self) -> &str { "Copy Pass" }

    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        let extent = context.window_state.get_curr_extent();
        let draw_image = context.frame.draw.image;
        let present_image = context.frame.present_image;

        vk_util::transition_image(
            context.device,
            cmd,
            draw_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        vk_util::transition_image(
            context.device,
            cmd,
            present_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        vk_util::blit_copy_image_to_image(
            context.device,
            cmd,
            draw_image,
            extent,
            present_image,
            extent,
        );

        vk_util::transition_image(
            context.device,
            cmd,
            present_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }
}

pub struct UiPass;

impl RenderPass for UiPass {
    fn name(&self) -> &str { "UI Pass" }

    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        let image_view = context.frame.present_image_view;

         let attachment_info = [vk_util::attachment_info(
            image_view,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            None,
        )];

        let render_info =
            vk_util::rendering_info(context.window_state.get_curr_extent(), &attachment_info, None);

        unsafe {
            context.device.cmd_begin_rendering(cmd, &render_info);
        }

        context.imgui
            .context
            .new_frame()
            .show_demo_window(&mut context.imgui.opened);

        let draw_data = context.imgui.context.render();

        context.imgui.renderer.cmd_draw(cmd, draw_data).unwrap();

        unsafe {
            context.device.cmd_end_rendering(cmd);
        }

        // Transition for presentation
        vk_util::transition_image(
            context.device,
            cmd,
            context.frame.present_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
    }
}
