//! Headless capture test: frustum culling on/off visual parity.
//!
//! The scene includes an off-screen mesh-bearing parent with an in-frustum
//! child. Culling must remove off-screen draws without hiding that child.
//!
//! Run with `--culling=on` (default) or `--culling=off` and compare captures.

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    let culling_enabled = std::env::args()
        .find_map(|arg| arg.strip_prefix("--culling=").map(str::to_owned))
        .map_or(true, |value| value != "off");
    let app_name = if culling_enabled {
        "capture_culling_on"
    } else {
        "capture_culling_off"
    };

    run_headless_capture_test(app_name, &args, move |renderer| {
        build_scene(renderer, culling_enabled)
    });
}

fn build_scene(renderer: &mut renderer::prelude::Renderer, culling_enabled: bool) -> Scene {
    let mut scene = Scene::new();
    scene.set_frustum_culling(culling_enabled);

    set_default_camera(
        renderer,
        &mut scene,
        Vec3::new(0.0, 1.0, 6.0),
        Vec3::new(0.0, 0.5, 0.0),
        55.0,
    );

    let root = scene.create_node_default(None).expect("capture root");
    let mut assets = renderer.assets();
    let material = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.65, 0.95, 1.0),
            metallic: 0.1,
            roughness: 0.35,
            ..Default::default()
        })
        .expect("culling material");
    let mut cube = build_cube_mesh("culling_cube");
    cube.material = Some(material);
    let cube = assets.upload_procedural_mesh(cube).expect("culling cube");

    let center = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(-0.75, 0.0, 0.0)),
        )
        .expect("center cube");
    scene.add_mesh(center, cube).expect("center cube mesh");

    // The parent's own mesh is outside the right plane. Its child cancels the
    // parent translation and must remain visible when culling is enabled.
    let offscreen_parent = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        )
        .expect("offscreen parent");
    scene
        .add_mesh(offscreen_parent, cube)
        .expect("offscreen parent mesh");
    let visible_child = scene
        .create_node(
            Some(offscreen_parent),
            Mat4::from_translation(Vec3::new(-49.25, 0.0, 0.0)),
        )
        .expect("visible child");
    scene
        .add_mesh(visible_child, cube)
        .expect("visible child mesh");

    let behind_camera = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 10.0)),
        )
        .expect("behind-camera cube");
    scene
        .add_mesh(behind_camera, cube)
        .expect("behind-camera cube mesh");

    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 4.0, 4.0),
            color: Vec3::ONE,
            intensity: 45.0,
            range: 15.0,
        })
        .expect("culling light");

    scene
}
