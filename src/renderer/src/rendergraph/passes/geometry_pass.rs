//! Main scene geometry draw pass.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct GeometryPass;

impl RenderPassNode for GeometryPass {
    fn name(&self) -> &'static str {
        "GeometryPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if !ctx.submission.flags.draw_geometry {
            return Ok(());
        }

        let mut recording = ctx.geometry_ctx();
        recording.draw_geometry_from_submission()?;
        Ok(())
    }
}
