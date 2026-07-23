//! Environment background draw pass.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct SkyboxPass;

impl RenderPassNode for SkyboxPass {
    fn name(&self) -> &'static str {
        "SkyboxPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if !ctx.submission.flags.draw_skybox {
            return Ok(());
        }

        let mut recording = ctx.skybox_ctx();
        recording.draw_skybox_from_submission()?;
        Ok(())
    }
}
