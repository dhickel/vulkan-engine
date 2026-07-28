//! Debug-lines render pass. Gated behind the `debug-draw` Cargo feature.
//!
//! Renders world-space debug line segments as a depth-tested unlit overlay
//! after the main geometry pass and before the UI pass.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct DebugLinesPass;

impl RenderPassNode for DebugLinesPass {
    fn name(&self) -> &'static str {
        "DebugLinesPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if ctx.submission.debug_lines.is_empty() {
            return Ok(());
        }

        let mut recording = ctx.debug_lines_ctx();
        recording.draw_debug_lines()?;
        Ok(())
    }
}
