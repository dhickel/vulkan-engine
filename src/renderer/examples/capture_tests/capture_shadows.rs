//! Headless capture test: directional light and shadow mapping.
//!
//! Renders two opaque cubes above a neutral receiver plane. The expected image
//! has hard-edged, softly filtered shadows extending away from the upper-left
//! directional light, with no shadow acne across the lit plane.
//!
//! Run: cargo run -p renderer --example capture_shadows -- --headless

mod common;

use common::*;
use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{DirectionalLight, PbrMaterialDesc, Scene};

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_shadows", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();
    set_default_camera(
        renderer,
        &mut scene,
        Vec3::new(6.0, 5.0, 9.0),
        Vec3::new(0.0, 0.7, 0.0),
        50.0,
    );

    let root = scene.create_node_default(None).expect("shadow root");
    scene.set_node_name(root, "Shadow Capture Root").unwrap();

    let mut assets = renderer.assets();
    let receiver_material = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.72, 0.72, 0.72, 1.0),
            metallic: 0.0,
            roughness: 0.8,
            ..Default::default()
        })
        .expect("receiver material");
    let caster_material = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.75, 0.18, 0.10, 1.0),
            metallic: 0.0,
            roughness: 0.55,
            ..Default::default()
        })
        .expect("caster material");

    let mut plane = build_plane_mesh("shadow_receiver", 14.0);
    plane.material = Some(receiver_material);
    let plane = assets.upload_procedural_mesh(plane).expect("receiver mesh");
    let receiver = scene
        .create_node(Some(root), Mat4::IDENTITY)
        .expect("receiver node");
    scene
        .add_mesh(receiver, plane)
        .expect("receiver attachment");

    let mut cube = build_cube_mesh("shadow_caster");
    cube.material = Some(caster_material);
    let cube = assets.upload_procedural_mesh(cube).expect("caster mesh");

    for (name, scale, position) in [
        (
            "Tall Caster",
            Vec3::new(1.5, 2.8, 1.5),
            Vec3::new(-1.4, 1.4, 0.2),
        ),
        (
            "Short Caster",
            Vec3::new(1.3, 1.3, 1.3),
            Vec3::new(1.7, 0.65, -1.0),
        ),
    ] {
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(scale, glam::Quat::IDENTITY, position),
            )
            .expect("caster node");
        scene.set_node_name(node, name).unwrap();
        scene.add_mesh(node, cube).expect("caster attachment");
    }

    drop(assets);
    scene
        .create_directional_light(DirectionalLight {
            direction: Vec3::new(-0.55, 1.0, 0.45),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 7.0,
        })
        .expect("directional light");

    scene
}
