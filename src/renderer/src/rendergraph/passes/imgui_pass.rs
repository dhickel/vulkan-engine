//! UI overlay pass that draws ImGui into the present color attachment.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct ImguiPass;

impl RenderPassNode for ImguiPass {
    fn name(&self) -> &'static str {
        "ImguiPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if !ctx.submission.flags.draw_imgui {
            return Ok(());
        }

        ctx.renderer.draw_imgui_to_present(ctx.frame)
    }
}
