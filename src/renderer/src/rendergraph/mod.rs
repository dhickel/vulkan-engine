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
//! Default passes receive narrow pass-specific recording contexts; no pass receives
//! unrestricted mutable access to `VkRenderCore`. Custom pass registration still has no
//! resource declaration, synchronization validation, or ordering checks, and the unstable
//! context intentionally exposes no generalized Vulkan recording escape hatch.
//!
//! ## Purpose
//! The RenderGraph is a high-level abstraction that decouples the "what" to render from the
//! "how" it's recorded in Vulkan. It organizes the frame into a sequence of logical "passes"
//! (e.g., Geometry, Skybox, UI).
//!
//! ## Key Concepts
//! - **RenderPassNode**: A single stage in the frame (e.g., `GeometryPass`). It implements
//!   `execute` to record Vulkan commands for that stage.
//! - **RenderGraphContext**: Provides the frame submission plus internal narrow recording
//!   contexts for each default pass.
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
    DebugCapturePass, GeometryPass, ImguiPass, PrepareTargetsPass, PresentCopyPass, ShadowPass,
    SkyboxPass, TerminalPresentPass,
};
use crate::scene::render_submission::RenderSubmission;
use crate::vulkan::vk_commands::RecordingDispatcher;
use crate::vulkan::vk_types::{VkFrame, VkQueueType};
use std::time::Instant;

pub mod passes;

pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPassNode>>,
}

/// Context passed to each render pass during graph execution.
///
/// # Recording access
///
/// The private recording dispatcher yields lifetime-bound pass-specific contexts. It owns
/// no resources and cannot access frame lifecycle, presentation, queues, or swapchain state.
///
/// # Stability
///
/// Pass implementations must respect all implicit state-carryover dependencies documented
/// in `docs/internal/07-rendergraph-dependencies-and-aliasing.md`.
pub struct RenderGraphContext<'a> {
    pub submission: &'a RenderSubmission,
    pub frame: &'a mut VkFrame,
    pub(crate) recording: &'a mut RecordingDispatcher<'a>,
}

impl<'a> RenderGraphContext<'a> {
    pub(crate) fn new(
        submission: &'a RenderSubmission,
        frame: &'a mut VkFrame,
        recording: &'a mut RecordingDispatcher<'a>,
    ) -> Self {
        Self { submission, frame, recording }
    }
}

/// A single stage in the render graph.
///
/// # Stability
///
/// **Alpha unstable.** This trait is only available when the `advanced-interop` feature is
/// enabled. Its signature may change across alpha sprints. Custom implementations can inspect
/// the submission and frame but receive no generalized backend recording access. There is no
/// resource declaration, synchronization validation, or ordering constraint checking.
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
            Box::new(ShadowPass),
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
            ctx.begin_gpu_pass_timing(cmd_buffer, pass.name());

            let pass_start = Instant::now();
            let result = pass.execute(ctx);
            let cpu_ms = elapsed_ms(pass_start);
            ctx.end_gpu_pass_timing(cmd_buffer);

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
