//! Prepare draw/depth targets for passes that will write color/depth this frame.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct PrepareTargetsPass;

impl RenderPassNode for PrepareTargetsPass {
    fn name(&self) -> &'static str {
        "PrepareTargetsPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if ctx.submission.has_draw_targets() {
            ctx.renderer.prepare_draw_targets(ctx.frame);
        }

        Ok(())
    }
}
