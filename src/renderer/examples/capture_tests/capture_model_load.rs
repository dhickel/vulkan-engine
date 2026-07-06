//! Headless capture test: glTF Model Import Verification
//!
//! Loads DamagedHelmet.glb and renders it with default materials.
//! Tests the model import pipeline and material preservation.
//!
//! Run: cargo run -p renderer --example capture_model_load -- --headless

mod common;

use glam::{Mat4, Vec3};
use renderer::prelude::{PointLight, Scene};

use common::*;

const HELMET_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_model_load", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 1.2, 2.5);
    set_default_camera(
        renderer,
        &mut scene,
        eye,
        Vec3::new(0.0, 0.3, 0.0),
        60.0,
    );

    let root = scene.create_node_default(None).expect("capture root");
    scene.set_node_name(root, "Capture Root").expect("root name");

    // ── Load model ──────────────────────────────────────────────────────
    {
        let fragment = renderer
            .assets()
            .load_model(HELMET_PATH)
            .expect("load DamagedHelmet.glb");
        let mount = scene
            .merge_fragment(Some(root), fragment)
            .expect("merge helmet");
        scene
            .set_transform(
                mount.mounted_root,
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(1.0),
                    glam::Quat::from_rotation_y(std::f32::consts::PI * 0.25),
                    Vec3::new(0.0, 0.0, 0.0),
                ),
            )
            .expect("helmet transform");
    }

    // ── Lighting ────────────────────────────────────────────────────────
    scene
        .create_point_light(PointLight {
            position: Vec3::new(2.0, 3.0, 2.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 15.0,
            range: 10.0,
        })
        .expect("key light");

    scene
        .create_point_light(PointLight {
            position: Vec3::new(-2.0, 1.0, -1.0),
            color: Vec3::new(0.3, 0.4, 0.8),
            intensity: 8.0,
            range: 8.0,
        })
        .expect("fill light");

    scene
}
