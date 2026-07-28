//! Headless debug-line capture example.
//!
//! # Purpose
//! Demonstrates the `debug-draw` feature: drawing world-space line primitives
//! (crosses, bounding boxes, spheres) via `FrameExtensions`.
//!
//! # Usage
//! ```bash
//! cargo run -p renderer --example demo_debug_lines --features debug-draw -- --headless
//! ```
//!
//! The captured frame is saved to the default capture directory.

use glam::{Mat4, Vec3};
use renderer::api::{CameraView, FrameExtensions, Renderer, RendererConfig, Scene};
use renderer::debug_draw::DebugDrawState;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let config = RendererConfig {
        app_name: "demo_debug_lines".to_string(),
        window_width: 800,
        window_height: 600,
        headless: true,
        ..Default::default()
    };

    let mut renderer = Renderer::new_headless(config)?;
    let mut scene = Scene::new();

    // Position the camera to look at the debug geometry.
    let eye = Vec3::new(0.0, 5.0, 10.0);
    let target = Vec3::ZERO;
    let up = Vec3::Y;
    let view = Mat4::look_at_rh(eye, target, up);
    let projection = Mat4::perspective_rh(70_f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);

    // Build debug geometry.
    let mut debug = DebugDrawState::new();

    // Red XYZ cross at origin
    debug.push_cross(
        Vec3::ZERO,
        4.0,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );

    // Cyan wireframe cube (AABB) around the origin
    debug.push_aabb(
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
        Vec3::new(0.0, 1.0, 1.0),
    );

    // Yellow sphere at (+3, 0, 0) with radius 1.5
    debug.push_sphere(Vec3::new(3.0, 0.0, 0.0), 1.5, Vec3::new(1.0, 1.0, 0.0));

    // Spiral of green lines
    let turns = 3;
    let points = 120;
    for i in 0..points {
        let t0 = i as f32 / points as f32 * turns as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / points as f32 * turns as f32 * std::f32::consts::TAU;
        let r = 3.0;
        let y0 = (i as f32 / points as f32 - 0.5) * 6.0;
        let y1 = ((i + 1) as f32 / points as f32 - 0.5) * 6.0;
        debug.push_line(
            Vec3::new(t0.cos() * r, y0, t0.sin() * r),
            Vec3::new(t1.cos() * r, y1, t1.sin() * r),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    // Transfer debug lines to frame extensions.
    let mut extensions = FrameExtensions::new();
    extensions.debug_lines = debug.take_lines();
    let line_count = extensions.debug_lines.len();
    renderer.set_frame_extensions(extensions);

    // Render one frame.
    let camera_view = CameraView::from_matrices(view, projection, eye);
    let outcome = renderer.render_scene_headless_with_view(&mut scene, camera_view)?;

    println!(
        "Frame rendered: {:?} with {} debug line segments",
        outcome, line_count
    );

    Ok(())
}
