//! # RenderGraph Architecture
//!
//! ## Stability
//!
//! **Alpha unstable.** This module is only public when the `advanced-interop` Cargo
//! feature is enabled. When enabled, its types and traits may change or be removed
//! across alpha sprints without prior notice. There is no API compatibility guarantee.
//! Beginner applications should not depend on this module.
//!
//! ## Safety
//!
//! [`RenderGraphContext`] exposes [`RenderGraphContext::renderer`] as `&mut VkRenderCore`,
//! giving custom [`RenderPassNode`] implementations unrestricted mutable access to the
//! Vulkan backend. Custom pass registration has **no** resource declaration,
//! synchronization validation, or ordering constraint checks. Inserting a custom pass
//! without understanding the implicit state-carryover dependencies documented in
//! `docs/internal/07-rendergraph-dependencies-and-aliasing.md` can cause Vulkan validation
//! errors, GPU hangs, or undefined behavior.
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
    DebugCapturePass, GeometryPass, ImguiPass, PrepareTargetsPass, PresentCopyPass, SkyboxPass,
    TerminalPresentPass,
};
use crate::scene::render_submission::RenderSubmission;
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_types::{VkFrame, VkQueueType};
use std::time::Instant;

pub mod passes;

pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPassNode>>,
}

/// Context passed to each render pass during graph execution.
///
/// # Safety
///
/// The [`renderer`](Self::renderer) field provides `&mut VkRenderCore` — unrestricted mutable
/// access to the Vulkan backend. Pass implementations must respect all implicit state-carryover
/// dependencies documented in `docs/internal/07-rendergraph-dependencies-and-aliasing.md`.
/// Incorrect layout transitions, descriptor mutations, or resource ownership violations can
/// cause Vulkan validation errors or undefined behavior.
pub struct RenderGraphContext<'a> {
    pub submission: &'a RenderSubmission,
    pub frame: &'a mut VkFrame,
    pub renderer: &'a mut VkRenderCore,
}

/// A single stage in the render graph.
///
/// # Stability
///
/// **Alpha unstable.** This trait is only available when the `advanced-interop` feature is
/// enabled. Its signature may change across alpha sprints. Custom implementations receive
/// [`RenderGraphContext`] which gives unrestricted mutable access to the Vulkan backend.
/// There is no resource declaration, synchronization validation, or ordering constraint
/// checking for custom pass implementations.
pub trait RenderPassNode {
    fn name(&self) -> &'static str;
    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String>;
}

#[derive(Clone, Debug, Default)]
pub struct RenderGraphPassTiming {
    pub name: &'static str,
    pub cpu_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RenderGraphExecutionReport {
    pub total_cpu_ms: f32,
    pub pass_timings: Vec<RenderGraphPassTiming>,
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
            Box::new(DebugCapturePass),
            Box::new(TerminalPresentPass),
        ])
    }

    pub fn execute(
        &self,
        ctx: &mut RenderGraphContext,
    ) -> Result<RenderGraphExecutionReport, String> {
        let graph_start = Instant::now();
        let mut pass_timings = Vec::with_capacity(self.passes.len());

        for pass in self.passes.iter() {
            let cmd_pool = ctx.frame.cmd_pools.get(VkQueueType::Graphics);
            let cmd_buffer = cmd_pool.buffers[0];
            ctx.renderer.begin_gpu_pass_timing(cmd_buffer, pass.name());

            let pass_start = Instant::now();
            let result = pass.execute(ctx);
            let cpu_ms = elapsed_ms(pass_start);
            ctx.renderer.end_gpu_pass_timing(cmd_buffer);

            result.map_err(|err| format!("render pass '{}' failed: {err}", pass.name()))?;
            pass_timings.push(RenderGraphPassTiming {
                name: pass.name(),
                cpu_ms,
            });
        }
        Ok(RenderGraphExecutionReport {
            total_cpu_ms: elapsed_ms(graph_start),
            pass_timings,
        })
    }
}

fn elapsed_ms(start: Instant) -> f32 {
    start.elapsed().as_secs_f64() as f32 * 1000.0
}
