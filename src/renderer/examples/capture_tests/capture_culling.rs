//! Headless capture test: frustum culling on/off visual parity.
//!
//! The scene includes:
//! - A center cube for reference.
//! - An off-screen parent with an in-frustum child (descendant independence).
//! - A large mesh whose origin lies outside but geometry intersects the frustum.
//! - A rotated / non-uniformly-scaled mesh.
//! - A conservative-visible (unknown) mesh that must always render.
//! - A behind-camera mesh that must be culled.
//!
//! Run with `--culling=on` (default) or `--culling=off` and compare captures.

mod common;

use glam::{Mat4, Quat, Vec3, Vec4};
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
    let cube_mesh = assets.upload_procedural_mesh(cube).expect("culling cube");
    let cube_bounds = assets.mesh_scene_bounds(cube_mesh).expect("cube bounds");

    // ---- Center reference cube ----
    let center = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(-0.75, 0.0, 0.0)),
        )
        .expect("center cube");
    scene
        .add_mesh_with_bounds(center, cube_mesh, cube_bounds)
        .expect("center cube mesh");

    // ---- Off-screen parent with in-frustum child ----
    // The parent is far right. Its visible child cancels the translation.
    let offscreen_parent = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        )
        .expect("offscreen parent");
    scene
        .add_mesh_with_bounds(offscreen_parent, cube_mesh, cube_bounds)
        .expect("offscreen parent mesh");
    let visible_child = scene
        .create_node(
            Some(offscreen_parent),
            Mat4::from_translation(Vec3::new(-49.25, 0.0, 0.0)),
        )
        .expect("visible child");
    scene
        .add_mesh_with_bounds(visible_child, cube_mesh, cube_bounds)
        .expect("visible child mesh");

    // ---- Large mesh: origin outside frustum but geometry intersects ----
    // Place a 4x-scaled cube so its origin is far-left but its right half
    // extends into view. Culling must test the 8-corner bounds, not the origin.
    let large_node = scene
        .create_node(
            Some(root),
            Mat4::from_scale_rotation_translation(
                Vec3::new(4.0, 1.0, 2.0),
                Quat::IDENTITY,
                Vec3::new(-5.5, 0.0, 0.0),
            ),
        )
        .expect("large off-center mesh");
    scene
        .add_mesh_with_bounds(large_node, cube_mesh, cube_bounds)
        .expect("large mesh");

    // ---- Rotated + non-uniformly-scaled mesh ----
    // Non-uniform scale with rotation stresses the 8-corner transform.
    let rotated_node = scene
        .create_node(
            Some(root),
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.5, 0.5, 1.0),
                Quat::from_rotation_z(0.7),
                Vec3::new(1.2, 0.8, 0.0),
            ),
        )
        .expect("rotated mesh");
    scene
        .add_mesh_with_bounds(rotated_node, cube_mesh, cube_bounds)
        .expect("rotated mesh");

    // ---- Conservative-visible unknown mesh ----
    // Use `add_mesh` (without bounds) to force ConservativeVisible.
    // This mesh must always render regardless of frustum position.
    let unknown_node = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(1.2, -0.6, 0.0)),
        )
        .expect("unknown mesh node");
    scene
        .add_mesh(unknown_node, cube_mesh)
        .expect("unknown mesh");

    // ---- Behind-camera cube (should be culled) ----
    let behind_camera = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 10.0)),
        )
        .expect("behind-camera cube");
    scene
        .add_mesh_with_bounds(behind_camera, cube_mesh, cube_bounds)
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
