//! Headless capture test: Point Light Properties
//!
//! Renders a ground plane + neutral spheres lit by 3 colored point lights
//! at different positions, testing intensity, color, range, and light accumulation.
//!
//! Run: cargo run -p renderer --example capture_lighting -- --headless

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_lighting", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 5.0, 10.0);
    set_default_camera(
        renderer,
        &mut scene,
        eye,
        Vec3::new(0.0, 0.0, -2.0),
        55.0,
    );

    let root = scene.create_node_default(None).expect("capture root");
    scene.set_node_name(root, "Capture Root").expect("root name");

    let mut assets = renderer.assets();

    // ── Materials ───────────────────────────────────────────────────────
    let gray_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.7, 0.7, 0.7, 1.0),
            metallic: 0.0,
            roughness: 0.6,
            ..Default::default()
        })
        .expect("gray material");

    let white_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("white material");

    let red_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(1.0, 0.2, 0.2, 1.0),
            metallic: 0.1,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("red material");

    let blue_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.2, 0.3, 1.0, 1.0),
            metallic: 0.1,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("blue material");

    // ── Ground plane ────────────────────────────────────────────────────
    {
        let mut plane = build_plane_mesh("ground", 20.0);
        plane.material = Some(gray_mat);
        let plane_handle = assets.upload_procedural_mesh(plane).expect("plane");
        let ground = scene
            .create_node(Some(root), Mat4::IDENTITY)
            .expect("ground node");
        scene.add_mesh(ground, plane_handle).expect("add plane");
    }

    // ── Test spheres ────────────────────────────────────────────────────
    let sphere_positions = [
        ("center", Vec3::new(0.0, 0.5, -1.0), white_mat),
        ("near_warm", Vec3::new(-1.5, 0.5, 0.0), white_mat),
        ("near_cool", Vec3::new(2.5, 0.5, 0.0), white_mat),
        ("far_left", Vec3::new(-4.0, 0.5, -2.0), red_mat),
        ("far_right", Vec3::new(4.0, 0.5, -2.0), blue_mat),
        ("behind", Vec3::new(0.0, 0.5, -4.0), white_mat),
    ];

    let sphere_mesh_data = build_sphere_mesh("light_sphere", 32);

    for (name, pos, material) in &sphere_positions {
        let mut mesh = sphere_mesh_data.clone();
        mesh.name = format!("sphere_{}", name);
        mesh.material = Some(*material);
        let mesh_handle = assets.upload_procedural_mesh(mesh).expect("sphere mesh");

        let node = scene
            .create_node(Some(root), Mat4::from_translation(*pos))
            .expect("node");
        scene.add_mesh(node, mesh_handle).expect("add mesh");
    }

    // ── Point lights ────────────────────────────────────────────────────
    // Light 1: warm white, above center
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 4.0, -1.0),
            color: Vec3::new(1.0, 0.9, 0.75),
            intensity: 40.0,
            range: 10.0,
        })
        .expect("warm light");

    // Light 2: cool blue, right side
    scene
        .create_point_light(PointLight {
            position: Vec3::new(4.0, 2.5, 0.0),
            color: Vec3::new(0.3, 0.5, 1.0),
            intensity: 50.0,
            range: 8.0,
        })
        .expect("cool light");

    // Light 3: dim red, back left
    scene
        .create_point_light(PointLight {
            position: Vec3::new(-4.0, 1.5, -2.0),
            color: Vec3::new(1.0, 0.15, 0.1),
            intensity: 25.0,
            range: 6.0,
        })
        .expect("red light");

    // Light 4: bright white, high above (ambient fill)
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 7.0, 0.0),
            color: Vec3::new(0.5, 0.5, 0.5),
            intensity: 15.0,
            range: 20.0,
        })
        .expect("ambient fill");

    scene
}
