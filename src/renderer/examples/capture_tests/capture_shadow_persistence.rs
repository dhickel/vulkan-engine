//! Headless capture test: shadow image layout persistence across early-return frames.
//!
//! Validates M-A5: when the shadow pass early-returns (no shadow-casting light,
//! empty draw list, frustum/cascade failure), the shadow images must transition
//! to SHADER_READ_ONLY_OPTIMAL before the geometry pass binds them as descriptors.
//!
//! The test scene is used by timeout-bound capture runs to verify that no-work
//! shadow paths leave descriptor-bound images initialized and shader-readable.
//!
//! Run: cargo run -p renderer --example capture_shadow_persistence -- --headless

mod common;

use common::*;
use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{DirectionalLight, PbrMaterialDesc, Scene};

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_shadow_persistence", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();
    set_default_camera(
        renderer,
        &mut scene,
        Vec3::new(4.0, 3.5, 6.0),
        Vec3::new(0.0, 0.6, 0.0),
        45.0,
    );

    let root = scene.create_node_default(None).expect("persist root");
    scene
        .set_node_name(root, "Shadow Persistence Root")
        .unwrap();

    let mut assets = renderer.assets();
    let floor_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.6, 0.6, 0.65, 1.0),
            metallic: 0.0,
            roughness: 0.9,
            ..Default::default()
        })
        .expect("floor material");
    let caster_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.8, 0.25, 0.15, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("caster material");

    let mut plane = build_plane_mesh("persist_receiver", 10.0);
    plane.material = Some(floor_mat);
    let plane_mesh = assets.upload_procedural_mesh(plane).expect("receiver mesh");
    let receiver = scene
        .create_node(Some(root), Mat4::IDENTITY)
        .expect("receiver node");
    scene
        .add_mesh(receiver, plane_mesh)
        .expect("receiver attach");

    // Single caster for the rendered baseline; validation runs pair this scene
    // with no-work shadow frames to catch stale or undefined shadow sampling.
    let mut cube = build_cube_mesh("persist_caster");
    cube.material = Some(caster_mat);
    let cube_mesh = assets.upload_procedural_mesh(cube).expect("caster mesh");
    let caster = scene
        .create_node(
            Some(root),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.8, 1.6, 0.8),
                glam::Quat::IDENTITY,
                Vec3::new(0.0, 0.8, 0.0),
            ),
        )
        .expect("caster node");
    scene.set_node_name(caster, "Persist Caster").unwrap();
    scene.add_mesh(caster, cube_mesh).expect("caster attach");

    drop(assets);

    let directional = scene
        .create_directional_light(DirectionalLight {
            direction: Vec3::new(-0.4, 1.2, 0.6),
            color: Vec3::new(1.0, 0.92, 0.78),
            intensity: 6.0,
        })
        .expect("directional light");

    #[cfg(not(feature = "csm"))]
    let _ = directional;

    #[cfg(feature = "csm")]
    scene
        .set_directional_shadow_config(
            directional,
            renderer::prelude::DirectionalShadowConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .expect("enable CSM shadows");

    scene
}
