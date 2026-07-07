//! Headless capture test: Environment / IBL Reflections
//!
//! Renders highly-metallic spheres at different roughness levels to
//! verify skybox rendering and IBL reflection quality.
//!
//! Run: cargo run -p renderer --example capture_environment -- --headless

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_environment", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 1.5, 8.0);
    set_default_camera(renderer, &mut scene, eye, Vec3::new(0.0, 1.0, 0.0), 55.0);

    let root = scene.create_node_default(None).expect("capture root");
    scene
        .set_node_name(root, "Capture Root")
        .expect("root name");

    let mut assets = renderer.assets();

    // ── Metallic spheres at different roughness ─────────────────────────
    let configs = [
        ("mirror", 1.0, 0.05, Vec3::new(-3.0, 1.0, 0.0)),
        ("blurry", 1.0, 0.3, Vec3::new(-1.0, 1.0, 0.0)),
        ("rough", 1.0, 0.7, Vec3::new(1.0, 1.0, 0.0)),
        ("diffuse_only", 0.0, 0.5, Vec3::new(3.0, 1.0, 0.0)),
    ];

    for (label, metallic, roughness, pos) in &configs {
        let mat = assets
            .create_material_pbr(PbrMaterialDesc {
                base_color: Vec4::new(0.9, 0.9, 0.9, 1.0),
                metallic: *metallic,
                roughness: *roughness,
                ..Default::default()
            })
            .expect("material");

        let mut mesh = build_sphere_mesh(&format!("sphere_{}", label), 48);
        mesh.material = Some(mat);
        let handle = assets.upload_procedural_mesh(mesh).expect("mesh");

        let node = scene
            .create_node(Some(root), Mat4::from_translation(*pos))
            .expect("node");
        scene.add_mesh(node, handle).expect("add mesh");
    }

    // ── Ground plane ────────────────────────────────────────────────────
    {
        let ground_mat = assets
            .create_material_pbr(PbrMaterialDesc {
                base_color: Vec4::new(0.3, 0.3, 0.3, 1.0),
                metallic: 0.0,
                roughness: 0.8,
                ..Default::default()
            })
            .expect("ground mat");
        let mut plane = build_plane_mesh("ground", 14.0);
        plane.material = Some(ground_mat);
        let handle = assets.upload_procedural_mesh(plane).expect("plane");
        let node = scene
            .create_node(Some(root), Mat4::IDENTITY)
            .expect("ground node");
        scene.add_mesh(node, handle).expect("add plane");
    }

    scene
}
