use crate::rendergraph::passes::{
    GeometryPass, ImguiPass, PrepareTargetsPass, PresentCopyPass, SkyboxPass,
};
use crate::scene::render_submission::RenderSubmission;
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_types::VkFrame;

pub mod passes;

pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPassNode>>,
}

pub struct RenderGraphContext<'a> {
    pub submission: &'a RenderSubmission,
    pub frame: &'a mut VkFrame,
    pub renderer: &'a mut VkRenderCore,
}

pub trait RenderPassNode {
    fn name(&self) -> &'static str;
    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String>;
}

impl RenderGraph {
    pub fn new(passes: Vec<Box<dyn RenderPassNode>>) -> Self {
        Self { passes }
    }

    pub fn default_graph() -> Self {
        Self::new(vec![
            Box::new(PrepareTargetsPass),
            Box::new(SkyboxPass),
            Box::new(GeometryPass),
            Box::new(PresentCopyPass),
            Box::new(ImguiPass),
        ])
    }

    pub fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        for pass in self.passes.iter() {
            pass.execute(ctx)
                .map_err(|err| format!("render pass '{}' failed: {err}", pass.name()))?;
        }
        Ok(())
    }
}
