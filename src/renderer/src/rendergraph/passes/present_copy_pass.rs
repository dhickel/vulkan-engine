//! Blit/copy pass from offscreen draw target into present image.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct PresentCopyPass;

impl RenderPassNode for PresentCopyPass {
    fn name(&self) -> &'static str {
        "PresentCopyPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if ctx.submission.has_draw_targets() {
            ctx.renderer.copy_draw_to_present(ctx.frame);
        } else {
            ctx.renderer.prepare_present_color_attachment(ctx.frame);
        }
        Ok(())
    }
}
