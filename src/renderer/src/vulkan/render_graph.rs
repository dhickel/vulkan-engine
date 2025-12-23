use ash::vk;
use crate::vulkan::vk_types::*;
use crate::vulkan::vk_util;
use crate::data::data_cache::{VkPipelineCache, VkDataCache};
use crate::data::gpu_data::SceneDataUBO;

pub struct RenderPassContext<'a> {
    pub device: &'a ash::Device,
    pub pipelines: &'a VkPipelineCache,
    pub frame: &'a VkFrame,
    pub window_state: &'a VkWindowState,
    pub scene_descriptors: Option<&'a mut VkSceneDescriptors>,
    pub scene_data: &'a SceneDataUBO,
    pub data_cache: &'a VkDataCache,
    pub render_context: &'a mut RenderContext,
    pub imgui: &'a mut VkImgui,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResource {
    DrawImage,
    DepthImage,
    PresentImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags2,
    pub stage: vk::PipelineStageFlags2,
}

impl ResourceState {
    pub fn new(layout: vk::ImageLayout, access: vk::AccessFlags2, stage: vk::PipelineStageFlags2) -> Self {
        Self { layout, access, stage }
    }
}

pub trait RenderPass {
    fn name(&self) -> &str;

    /// Executes the draw commands for this pass.
    ///
    /// # Arguments
    ///
    /// * `cmd` - The command buffer to record into.
    /// * `context` - The render context containing all global resources.
    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext);

    /// Returns the required resources and their state for this pass.
    fn required_resources(&self) -> Vec<(GraphResource, ResourceState)> {
        Vec::new()
    }
}

/// Manages a sequence of render passes.
pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
    resource_states: std::collections::HashMap<GraphResource, ResourceState>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resource_states: std::collections::HashMap::new(),
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.passes.push(pass);
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        // Reset resource states for the new frame
        self.resource_states.clear();
        self.resource_states.insert(GraphResource::DrawImage, ResourceState::new(
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TOP_OF_PIPE,
        ));
        self.resource_states.insert(GraphResource::DepthImage, ResourceState::new(
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TOP_OF_PIPE,
        ));
        self.resource_states.insert(GraphResource::PresentImage, ResourceState::new(
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TOP_OF_PIPE,
        ));

        for pass in &mut self.passes {
            let requirements = pass.required_resources();
            let mut barriers = Vec::new();

            for (resource, required_state) in requirements {
                let current_state = self.resource_states.entry(resource).or_insert(ResourceState::new(
                     vk::ImageLayout::UNDEFINED,
                     vk::AccessFlags2::empty(),
                     vk::PipelineStageFlags2::TOP_OF_PIPE,
                ));

                let layout_changed = current_state.layout != required_state.layout;

                // Check for hazards (RAW, WAR, WAW). RAR is safe.
                // If either previous or current access involves a write, we need a barrier.
                let write_mask = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::TRANSFER_WRITE
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::MEMORY_WRITE
                    | vk::AccessFlags2::HOST_WRITE;

                let is_hazard = (current_state.access.intersects(write_mask) || required_state.access.intersects(write_mask))
                    && current_state.layout != vk::ImageLayout::UNDEFINED;

                if layout_changed || is_hazard {
                    // Transition needed
                    let image = match resource {
                        GraphResource::DrawImage => context.frame.draw.image,
                        GraphResource::DepthImage => context.frame.depth.image,
                        GraphResource::PresentImage => context.frame.present_image,
                    };

                    let aspect_mask = if resource == GraphResource::DepthImage {
                        vk::ImageAspectFlags::DEPTH
                    } else {
                        vk::ImageAspectFlags::COLOR
                    };

                    let image_barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(current_state.stage)
                        .src_access_mask(current_state.access)
                        .dst_stage_mask(required_state.stage)
                        .dst_access_mask(required_state.access)
                        .old_layout(current_state.layout)
                        .new_layout(required_state.layout)
                        .subresource_range(vk_util::image_subresource_range(aspect_mask))
                        .image(image);

                    barriers.push(image_barrier);
                }

                // Update state
                *current_state = required_state;
            }

            if !barriers.is_empty() {
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe { context.device.cmd_pipeline_barrier2(cmd, &dep_info) };
            }

            pass.draw(cmd, context);
        }
    }
}
