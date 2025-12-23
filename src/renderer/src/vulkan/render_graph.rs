use ash::vk;
use crate::vulkan::vk_types::VkFrame;

/// Trait defining a single pass in the render graph.
pub trait RenderPass {
    fn name(&self) -> &str;

    /// Executes the draw commands for this pass.
    ///
    /// # Arguments
    ///
    /// * `cmd` - The command buffer to record into.
    /// * `frame` - The current frame data (sync primitives, descriptors, etc).
    fn draw(&mut self, cmd: vk::CommandBuffer, frame: &VkFrame);
}

/// Manages a sequence of render passes.
/// Future improvements: Handle resource dependencies and automatic barriers.
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

    pub fn execute(&mut self, cmd: vk::CommandBuffer, frame: &VkFrame) {
        for pass in &mut self.passes {
            pass.draw(cmd, frame);
        }
    }
}
