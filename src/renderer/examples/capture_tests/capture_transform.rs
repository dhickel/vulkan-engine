//! Headless capture test: Transform Hierarchy Verification
//!
//! Renders a chain of nested cubes with incremental rotations and translations,
//! testing parent-child transform composition correctness.
//!
//! Run: cargo run -p renderer --example capture_transform -- --headless

mod common;

use glam::{Mat4, Quat, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_transform", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(3.0, 2.5, 7.0);
    set_default_camera(renderer, &mut scene, eye, Vec3::new(0.5, 0.5, 0.0), 55.0);

    let root = scene.create_node_default(None).expect("capture root");
    scene
        .set_node_name(root, "Capture Root")
        .expect("root name");

    let mut assets = renderer.assets();

    // ── Materials ───────────────────────────────────────────────────────
    let orange_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(1.0, 0.5, 0.1, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("orange");

    let teal_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.1, 0.7, 0.7, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("teal");

    let purple_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.6, 0.2, 0.8, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("purple");

    let pink_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(1.0, 0.3, 0.6, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("pink");

    let white_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("white");

    // ── Ground plane ────────────────────────────────────────────────────
    {
        let mut plane = build_plane_mesh("ground", 14.0);
        plane.material = Some(white_mat);
        let handle = assets.upload_procedural_mesh(plane).expect("plane");
        let node = scene
            .create_node(Some(root), Mat4::IDENTITY)
            .expect("ground");
        scene.add_mesh(node, handle).expect("add plane");
    }

    // ── Base platform (large cube at origin) ────────────────────────────
    let cube_mesh = build_cube_mesh("cube_base");

    let mut base_mesh = cube_mesh.clone();
    base_mesh.material = Some(orange_mat);
    base_mesh.name = "platform".to_string();
    let platform_handle = assets.upload_procedural_mesh(base_mesh).expect("platform");

    let platform = scene
        .create_node(
            Some(root),
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.5, 0.15, 1.5),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.075, 0.0),
            ),
        )
        .expect("platform");
    scene
        .add_mesh(platform, platform_handle)
        .expect("add platform");

    // ── Child 1: rotated cube offset from platform ──────────────────────
    let mut child1_mesh = cube_mesh.clone();
    child1_mesh.name = "arm_segment_1".to_string();
    child1_mesh.material = Some(teal_mat);
    let child1_handle = assets
        .upload_procedural_mesh(child1_mesh)
        .expect("child1 mesh");

    let child1 = scene
        .create_node(
            Some(platform),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.4, 0.4, 0.4),
                Quat::from_rotation_y(0.6),
                Vec3::new(0.0, 0.6, 0.0),
            ),
        )
        .expect("child1");
    scene.add_mesh(child1, child1_handle).expect("add child1");

    // ── Child 2: further offset, further rotation ───────────────────────
    let mut child2_mesh = cube_mesh.clone();
    child2_mesh.name = "arm_segment_2".to_string();
    child2_mesh.material = Some(purple_mat);
    let child2_handle = assets
        .upload_procedural_mesh(child2_mesh)
        .expect("child2 mesh");

    let child2 = scene
        .create_node(
            Some(child1),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.3, 0.3, 0.3),
                Quat::from_rotation_z(0.5),
                Vec3::new(0.0, 0.9, 0.0),
            ),
        )
        .expect("child2");
    scene.add_mesh(child2, child2_handle).expect("add child2");

    // ── Child 3: tip of the chain ───────────────────────────────────────
    let mut child3_mesh = cube_mesh.clone();
    child3_mesh.name = "arm_segment_3".to_string();
    child3_mesh.material = Some(pink_mat);
    let child3_handle = assets
        .upload_procedural_mesh(child3_mesh)
        .expect("child3 mesh");

    let _child3 = scene
        .create_node(
            Some(child2),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.2, 0.2, 0.2),
                Quat::from_rotation_x(0.7),
                Vec3::new(0.0, 0.8, 0.0),
            ),
        )
        .expect("child3");
    scene.add_mesh(_child3, child3_handle).expect("add child3");

    // ── Second independent chain (sibling) ──────────────────────────────
    let mut sibling_mesh = cube_mesh.clone();
    sibling_mesh.name = "sibling_cube".to_string();
    sibling_mesh.material = Some(white_mat);
    let sibling_handle = assets
        .upload_procedural_mesh(sibling_mesh)
        .expect("sibling mesh");

    let sibling = scene
        .create_node(
            Some(platform),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.3, 0.5, 0.3),
                Quat::from_rotation_y(-0.4),
                Vec3::new(0.0, 0.6, 0.0),
            ),
        )
        .expect("sibling");
    scene
        .add_mesh(sibling, sibling_handle)
        .expect("add sibling");

    // ── Lighting ────────────────────────────────────────────────────────
    scene
        .create_point_light(PointLight {
            position: Vec3::new(2.0, 5.0, 4.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 50.0,
            range: 18.0,
        })
        .expect("key light");

    scene
        .create_point_light(PointLight {
            position: Vec3::new(-3.0, 3.0, -2.0),
            color: Vec3::new(0.3, 0.4, 0.9),
            intensity: 20.0,
            range: 12.0,
        })
        .expect("fill light");

    scene
}
