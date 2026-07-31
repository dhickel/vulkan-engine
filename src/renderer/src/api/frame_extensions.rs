//! Immutable per-frame extensions for renderer behaviour overrides and
//! development-only debug visualization. All data is consumed by the renderer
//! during submission and cleared after each frame.
//!
//! # Design
//!
//! `FrameExtensions` is a plain DTO (Data Transfer Object) with no internal
//! mutation. It carries:
//!
//! * Optional transform overrides on a per-node basis (hierarchy-consistent,
//!   app-owned — no scene mutation).
//! * Debug-line segments for editor gizmos, physics wireframes, AI navmesh
//!   visualisation, and other development overlays.
//!
//! The renderer consumes this struct by value during submission build, so
//! extensions are implicitly cleared at frame boundaries.

use std::collections::HashMap;

use crate::scene::SceneNodeId;
use crate::vulkan::vk_sprites::SpriteInstance;
use glam::Mat4;
#[cfg(feature = "debug-draw")]
use glam::Vec3;

/// Immutable per-frame extension payload consumed by the renderer.
///
/// Set via [`Renderer::set_frame_extensions`] before a frame. The renderer
/// takes ownership of the value and replaces it with a default (empty) set
/// after submission.
#[derive(Clone, Debug, Default)]
pub struct FrameExtensions {
    /// Per-node world-transform overrides applied during submission traversal.
    /// An entry in this map replaces the node's computed world transform
    /// *and* propagates to its subtree. The scene-graph local transforms are
    /// never mutated.
    ///
    /// Nodes that are not in this map use their normal computed world transform.
    pub transform_overrides: HashMap<SceneNodeId, Mat4>,

    /// Debug line segments in world space.
    ///
    /// Each entry is `(from, to, color)`. Lines are depth-tested and rendered
    /// as an unlit overlay during the debug-line render pass (only when the
    /// `debug-draw` feature is enabled and lines are non-empty).
    #[cfg(feature = "debug-draw")]
    pub debug_lines: Vec<(Vec3, Vec3, Vec3)>,
    /// Sprite instances rendered by the sprite-batch pass.
    pub sprites: Vec<SpriteInstance>,
    /// Optional orthographic camera for the 2D layer.
    /// When `None`, an identity projection is used.
    pub sprite_camera: Option<super::sprite::Camera2D>,
}

impl FrameExtensions {
    /// Create an empty extension set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when no extensions are set.
    pub fn is_empty(&self) -> bool {
        let mut empty = self.transform_overrides.is_empty();
        #[cfg(feature = "debug-draw")]
        {
            empty = empty && self.debug_lines.is_empty();
        }
        empty = empty && self.sprites.is_empty() && self.sprite_camera.is_none();
        empty
    }
}
