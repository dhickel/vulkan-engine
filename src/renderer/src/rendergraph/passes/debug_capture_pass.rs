//! Optional frame capture pass after final UI rendering and before terminal present transition.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct DebugCapturePass;

impl RenderPassNode for DebugCapturePass {
    fn name(&self) -> &'static str {
        "DebugCapturePass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        let mut recording = ctx.debug_capture_ctx();
        recording.record_due_frame_captures();
        Ok(())
    }
}
