//! Sprite batch render pass. Always available.
//!
//! Renders world-space sprite quads with orthographic projection after
//! debug lines and before UI.

use crate::rendergraph::{RenderGraphContext, RenderPassNode};

pub struct SpritesPass;

impl RenderPassNode for SpritesPass {
    fn name(&self) -> &'static str {
        "SpritesPass"
    }

    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
        if ctx.submission.sprites.is_empty() {
            return Ok(());
        }

        let mut recording = ctx.sprites_ctx();
        recording.draw_sprites()?;
        Ok(())
    }
}
