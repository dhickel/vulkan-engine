//! # RenderGraph Architecture
//!
//! ## Purpose
//! The RenderGraph is a high-level abstraction that decouples the "what" to render from the
//! "how" it's recorded in Vulkan. It organizes the frame into a sequence of logical "passes"
//! (e.g., Geometry, Skybox, UI).
//!
//! ## Key Concepts
//! - **RenderPassNode**: A single stage in the frame (e.g., `GeometryPass`). It implements
//!   `execute` to record Vulkan commands for that stage.
//! - **RenderGraphContext**: Provides passes with everything they need: the frame's resources,
//!   the scene submission, and access to the core renderer.
//! - **Pass Ordering**: The order of passes in the graph is the order they are recorded into
//!   the command buffer.
//!
//! ## Why use a RenderGraph?
//! - **Modularity**: New rendering features (like shadows or post-processing) can be added as
//!   new nodes without touching the core frame loop.
//! - **Clarity**: It provides a clear high-level overview of the frame's structure.
//! - **Future-Proofing**: It allows for automatic optimization of resource transitions and
//!   memory aliases between passes.

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
