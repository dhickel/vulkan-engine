//! Terminal swapchain presentation transition.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct TerminalPresentPass;

impl RenderPassNode for TerminalPresentPass {
    fn name(&self) -> &'static str {
        "TerminalPresentPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        let mut recording = ctx.terminal_present_ctx();
        if !recording.is_headless() {
            recording.transition_present_for_present()?;
        }
        Ok(())
    }
}
