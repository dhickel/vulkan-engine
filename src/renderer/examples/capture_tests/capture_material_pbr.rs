//! Headless capture test: PBR Material Parameter Matrix
//!
//! Renders a 4×4 grid of spheres showing PBR material parameter sweeps:
//!   Row 1: metallic sweep (0.0 → 1.0), roughness=0.3
//!   Row 2: roughness sweep (0.05 → 0.9), metallic=1.0
//!   Row 3: base color sweep (red, green, blue, white), matte
//!   Row 4: emissive sweep (dim→bright red, dim→bright blue)
//!
//! Run: cargo run -p renderer --example capture_material_pbr -- --headless

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_material_pbr", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 3.0, 14.0);
    set_default_camera(renderer, &mut scene, eye, Vec3::ZERO, 50.0);

    let root = scene.create_node_default(None).expect("capture root");
    scene
        .set_node_name(root, "Capture Root")
        .expect("root name");

    // ── Lighting ────────────────────────────────────────────────────────
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 8.0, 4.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 80.0,
            range: 30.0,
        })
        .expect("key light");
    scene
        .create_point_light(PointLight {
            position: Vec3::new(-6.0, 3.0, -2.0),
            color: Vec3::new(0.3, 0.4, 0.8),
            intensity: 20.0,
            range: 15.0,
        })
        .expect("fill light");

    // ── Material grid ───────────────────────────────────────────────────
    let spacing = 2.5;
    let start_x = -spacing * 1.5;
    let start_y = spacing * 1.5;

    let rows: Vec<(&str, Vec<PbrMaterialDesc>)> = vec![
        // Row 1: metallic sweep (0.0, 0.33, 0.66, 1.0), roughness=0.3
        (
            "metallic",
            (0..4)
                .map(|i| PbrMaterialDesc {
                    metallic: i as f32 / 3.0,
                    roughness: 0.3,
                    base_color: Vec4::new(0.9, 0.7, 0.5, 1.0),
                    ..Default::default()
                })
                .collect(),
        ),
        // Row 2: roughness sweep (0.05, 0.3, 0.6, 0.9), metallic=1.0
        (
            "roughness",
            vec![0.05, 0.3, 0.6, 0.9]
                .into_iter()
                .map(|r| PbrMaterialDesc {
                    metallic: 1.0,
                    roughness: r,
                    base_color: Vec4::new(0.85, 0.85, 0.85, 1.0),
                    ..Default::default()
                })
                .collect(),
        ),
        // Row 3: base color sweep (red, green, blue, white), matte
        (
            "base_color",
            vec![
                Vec4::new(1.0, 0.15, 0.15, 1.0),
                Vec4::new(0.15, 1.0, 0.15, 1.0),
                Vec4::new(0.15, 0.3, 1.0, 1.0),
                Vec4::ONE,
            ]
            .into_iter()
            .map(|c| PbrMaterialDesc {
                metallic: 0.0,
                roughness: 0.5,
                base_color: c,
                ..Default::default()
            })
            .collect(),
        ),
        // Row 4: emissive sweep
        (
            "emissive",
            vec![
                (Vec3::new(2.0, 0.1, 0.1), 1.0),   // bright red
                (Vec3::new(0.3, 0.05, 0.05), 0.5), // dim red
                (Vec3::new(0.1, 0.1, 3.0), 1.0),   // bright blue
                (Vec3::new(0.05, 0.05, 0.4), 0.5), // dim blue
            ]
            .into_iter()
            .map(|(factor, strength)| PbrMaterialDesc {
                metallic: 0.0,
                roughness: 0.5,
                base_color: Vec4::new(0.1, 0.1, 0.1, 1.0),
                emissive_factor: factor,
                emissive_strength: strength,
                ..Default::default()
            })
            .collect(),
        ),
    ];

    let mut assets = renderer.assets();
    for (row_idx, (_label, materials)) in rows.iter().enumerate() {
        for (col_idx, desc) in materials.iter().enumerate() {
            let mat = assets
                .create_material_pbr(desc.clone())
                .expect("material creation");

            let x = start_x + col_idx as f32 * spacing;
            let y = start_y - row_idx as f32 * spacing;

            let node = scene
                .create_node(Some(root), Mat4::from_translation(Vec3::new(x, y, 0.0)))
                .expect("node");

            let mut mesh_data = build_sphere_mesh(&format!("sphere_r{}_c{}", row_idx, col_idx), 32);
            mesh_data.material = Some(mat);
            let mesh_handle = assets.upload_procedural_mesh(mesh_data).expect("mesh");
            scene.add_mesh(node, mesh_handle).expect("add mesh");
        }
    }

    scene
}
