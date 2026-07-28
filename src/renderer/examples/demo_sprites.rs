//! Headless sprite-batch capture example.
//!
//! # Purpose
//! Demonstrates the `sprites-2d` feature: rendering colored quad sprites
//! with an orthographic Camera2D via `FrameExtensions`.
//!
//! # Usage
//! ```bash
//! cargo run -p renderer --example demo_sprites --features sprites-2d -- --headless
//! ```
//!
//! The captured frame is saved to the default capture directory.

use glam::Vec2;
use renderer::api::sprite::{Camera2D, SpriteRenderer};
use renderer::api::{CameraView, FrameExtensions, Renderer, RendererConfig, Scene};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let config = RendererConfig {
        app_name: "demo_sprites".to_string(),
        window_width: 800,
        window_height: 600,
        headless: true,
        ..Default::default()
    };

    let mut renderer = Renderer::new_headless(config)?;
    let mut scene = Scene::new();

    // Build sprite batch.
    let mut sprites = SpriteRenderer::new();

    // A red square at (50, 50), 64×64 pixels
    sprites.push_colored(Vec2::new(50.0, 50.0), Vec2::new(64.0, 64.0), [1.0, 0.0, 0.0, 1.0]);

    // A green square at (150, 50)
    sprites.push_colored(Vec2::new(150.0, 50.0), Vec2::new(64.0, 64.0), [0.0, 1.0, 0.0, 1.0]);

    // A blue square at (100, 150)
    sprites.push_colored(Vec2::new(100.0, 150.0), Vec2::new(64.0, 64.0), [0.0, 0.0, 1.0, 1.0]);

    // A rotated yellow square in the center
    sprites.push_sprite(
        Vec2::new(400.0, 300.0),
        Vec2::new(128.0, 128.0),
        45_f32.to_radians(),
        [1.0, 1.0, 0.0, 0.8],
        0,
    );

    // A semi-transparent white square overlapping the edge
    sprites.push_sprite(
        Vec2::new(700.0, 500.0),
        Vec2::new(96.0, 96.0),
        0.0,
        [1.0, 1.0, 1.0, 0.5],
        -1,
    );

    // Create a 2D camera that maps 800×600 world units to NDC.
    let camera_2d = Camera2D::from_extent(800.0, 600.0);

    // Transfer sprites to frame extensions.
    let mut extensions = FrameExtensions::new();
    extensions.sprite_camera = Some(camera_2d);
    extensions.sprites = sprites.take_sprites();

    let sprite_count = extensions.sprites.len();
    renderer.set_frame_extensions(extensions);

    // Use an identity 3D camera since we only care about the sprite layer.
    let camera_view = CameraView::from_matrices(
        glam::Mat4::IDENTITY,
        glam::Mat4::IDENTITY,
        glam::Vec3::ZERO,
    );

    let outcome = renderer.render_scene_headless_with_view(&mut scene, camera_view)?;

    println!(
        "Frame rendered: {:?} with {} sprites",
        outcome, sprite_count
    );

    Ok(())
}
