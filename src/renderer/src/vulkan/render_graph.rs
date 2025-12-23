use ash::vk;
use crate::vulkan::vk_types::*;
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

pub trait RenderPass {
    fn name(&self) -> &str;

    /// Executes the draw commands for this pass.
    ///
    /// # Arguments
    ///
    /// * `cmd` - The command buffer to record into.
    /// * `context` - The render context containing all global resources.
    fn draw(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext);
}

/// Manages a sequence of render passes.
pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.passes.push(pass);
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer, context: &mut RenderPassContext) {
        for pass in &mut self.passes {
            pass.draw(cmd, context);
        }
    }
}
