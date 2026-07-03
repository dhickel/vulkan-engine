//! Terminal swapchain presentation transition.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct TerminalPresentPass;

impl RenderPassNode for TerminalPresentPass {
    fn name(&self) -> &'static str {
        "TerminalPresentPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        ctx.renderer.transition_present_for_present(ctx.frame);
        Ok(())
    }
}
