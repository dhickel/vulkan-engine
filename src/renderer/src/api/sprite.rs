//! 2D sprite rendering API. Always available.
//!
//! Provides [`Camera2D`] for orthographic projection and [`SpriteRenderer`]
//! for batching colored quads in world space.
//!
//! # Usage
//! ```ignore
//! use renderer::api::sprite::{Camera2D, SpriteRenderer};
//!
//! let mut renderer = SpriteRenderer::new();
//! renderer.push_colored(Vec2::new(100.0, 200.0), Vec2::new(32.0, 32.0), [1.0,0.0,0.0,1.0]);
//!
//! let cam = Camera2D::from_extent(800.0, 600.0);
//! extensions.sprite_camera = Some(cam);
//! extensions.sprites = renderer.take_sprites();
//! ```

use crate::vulkan::vk_sprites::SpriteInstance;
use glam::{Mat4, Vec2};

/// Orthographic 2D camera.
///
/// By default the camera maps world-space coordinates directly to NDC
/// with (0,0) at the top-left of the viewport and +X right, +Y down.
#[derive(Debug, Clone)]
pub struct Camera2D {
    /// Viewport width in world units.
    pub width: f32,
    /// Viewport height in world units.
    pub height: f32,
    /// Camera center in world space (defaults to width/2, height/2).
    pub center: Vec2,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            center: Vec2::new(400.0, 300.0),
            near: -1.0,
            far: 1.0,
        }
    }
}

impl Camera2D {
    /// Create a camera that exactly fits the given viewport extent in world
    /// units, with (0,0) at top-left.
    pub fn from_extent(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            center: Vec2::new(width * 0.5, height * 0.5),
            near: -1.0,
            far: 1.0,
        }
    }

    /// Compute the view matrix (maps world coords centered at `center` to
    /// camera-local coords).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_translation(glam::Vec3::new(-self.center.x, -self.center.y, 0.0))
    }

    /// Compute the orthographic projection matrix.
    ///
    /// Maps right-handed world coords: +X right, +Y up, +Z into screen.
    /// The resulting NDC has Y flipped (Vulkan convention), which the
    /// viewport handles.
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::orthographic_rh(0.0, self.width, self.height, 0.0, self.near, self.far)
    }

    /// Set the camera center to a new world-space position.
    pub fn set_center(&mut self, center: Vec2) {
        self.center = center;
    }

    /// Pan the camera by a world-space delta.
    pub fn pan(&mut self, delta: Vec2) {
        self.center += delta;
    }

    /// Zoom by adjusting the viewport width/height around the center.
    pub fn zoom(&mut self, factor: f32) {
        let factor = factor.clamp(0.1, 10.0);
        self.width /= factor;
        self.height /= factor;
    }
}

/// Batched sprite renderer.
///
/// Collects sprite instances each frame. Call [`take_sprites`] before
/// submission to consume the batch.
#[derive(Debug, Clone, Default)]
pub struct SpriteRenderer {
    sprites: Vec<SpriteInstance>,
}

impl SpriteRenderer {
    /// Create an empty sprite renderer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a colored quad at `position` with the given `size`.
    ///
    /// Color is RGBA in linear space. The quad is axis-aligned unless
    /// [`push_rotated`] is used.
    pub fn push_colored(&mut self, position: Vec2, size: Vec2, color: [f32; 4]) {
        self.sprites
            .push(SpriteInstance::new(position, size).with_color(color));
    }

    /// Add a colored quad with rotation (radians) and z-order layer.
    pub fn push_sprite(
        &mut self,
        position: Vec2,
        size: Vec2,
        rotation: f32,
        color: [f32; 4],
        layer: i32,
    ) {
        self.sprites.push(
            SpriteInstance::new(position, size)
                .with_color(color)
                .with_rotation(rotation)
                .with_layer(layer),
        );
    }

    /// Returns the number of queued sprites.
    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// Returns `true` when no sprites are queued.
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    /// Clear all queued sprites.
    pub fn clear(&mut self) {
        self.sprites.clear();
    }

    /// Consume and return the queued sprites.
    ///
    /// Call this once per frame before passing to [`FrameExtensions`].
    pub fn take_sprites(&mut self) -> Vec<SpriteInstance> {
        std::mem::take(&mut self.sprites)
    }
}
