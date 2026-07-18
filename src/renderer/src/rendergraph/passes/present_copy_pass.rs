//! Blit/copy pass from offscreen draw target into present image.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct PresentCopyPass;

impl RenderPassNode for PresentCopyPass {
    fn name(&self) -> &'static str {
        "PresentCopyPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        let has_draw_targets = ctx.submission.has_draw_targets();
        let mut recording = ctx.present_copy_ctx();
        if has_draw_targets {
            recording.copy_draw_to_present();
        } else {
            recording.prepare_present_color_attachment();
        }
        Ok(())
    }
}
