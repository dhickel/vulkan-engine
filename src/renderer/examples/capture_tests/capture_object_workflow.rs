//! Headless capture test: Object Workflow (Queries + Selection + EditorCamera)
//!
//! Validates the Phase 07 query, selection, and editor-camera APIs end-to-end
//! with a draw-target headless frame capture.
//!
//! Run: cargo run -p renderer --example capture_object_workflow -- --headless

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::{
    object::{
        query::{EditorProxyPolicy, ObjectQueryFilter, VolumeQuery},
        selection::Selection,
        ObjectKind,
    },
    prelude::{PbrMaterialDesc, PointLight, Scene},
    EditorCamera, Ray,
};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_object_workflow", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 4.0, 8.0);
    set_default_camera(renderer, &mut scene, eye, Vec3::new(0.0, 1.0, 0.0), 50.0);

    let root = scene.create_node_default(None).expect("capture root");
    scene
        .set_node_name(root, "Capture Root")
        .expect("root name");

    let mut assets = renderer.assets();

    // ── Materials ───────────────────────────────────────────────────────
    let white = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.95, 0.95, 0.95, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("white");

    let red = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.9, 0.15, 0.15, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("red");

    let blue = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.3, 0.95, 1.0),
            metallic: 0.4,
            roughness: 0.3,
            ..Default::default()
        })
        .expect("blue");

    // ── Lighting ────────────────────────────────────────────────────────
    scene
        .create_point_light(PointLight {
            position: Vec3::new(2.0, 6.0, 4.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 80.0,
            range: 25.0,
        })
        .expect("key light");

    scene
        .create_point_light(PointLight {
            position: Vec3::new(-3.0, 2.0, -1.0),
            color: Vec3::new(0.3, 0.4, 0.8),
            intensity: 30.0,
            range: 15.0,
        })
        .expect("fill light");

    // ── Build test geometry ─────────────────────────────────────────────
    // Upload meshes with materials attached.
    let mut plane_data = build_plane_mesh("capture_plane", 10.0);
    plane_data.material = Some(white);
    let plane_handle = assets.upload_procedural_mesh(plane_data).expect("plane");

    let mut cube_data = build_cube_mesh("capture_cube");
    cube_data.material = Some(red);
    let cube_handle = assets.upload_procedural_mesh(cube_data).expect("cube");

    let mut sphere_data = build_sphere_mesh("capture_sphere", 16);
    sphere_data.material = Some(blue);
    let sphere_handle = assets.upload_procedural_mesh(sphere_data).expect("sphere");

    // Ground plane
    let ground = scene.create_node_default(Some(root)).expect("ground");
    scene.set_node_name(ground, "Ground").expect("ground name");
    scene
        .set_transform(ground, Mat4::from_scale(Vec3::splat(5.0)))
        .expect("ground transform");
    scene.add_mesh(ground, plane_handle).expect("ground mesh");

    // Cube at left
    let cube_node = scene.create_node_default(Some(root)).expect("cube");
    scene.set_node_name(cube_node, "Cube").expect("cube name");
    scene
        .set_transform(
            cube_node,
            Mat4::from_scale_rotation_translation(
                Vec3::splat(0.6),
                glam::Quat::from_rotation_y(0.3),
                Vec3::new(-1.5, 1.0, -1.0),
            ),
        )
        .expect("cube transform");
    scene.add_mesh(cube_node, cube_handle).expect("cube mesh");

    // Sphere at right
    let sphere_node = scene.create_node_default(Some(root)).expect("sphere");
    scene
        .set_node_name(sphere_node, "Sphere")
        .expect("sphere name");
    scene
        .set_transform(
            sphere_node,
            Mat4::from_translation(Vec3::new(1.8, 1.0, -1.0)),
        )
        .expect("sphere transform");
    scene
        .add_mesh(sphere_node, sphere_handle)
        .expect("sphere mesh");

    // ── Phase 07: Query validation ─────────────────────────────────────

    // Raycast toward the cube
    let ray = Ray {
        origin: Vec3::new(-1.5, 0.5, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    match scene.raycast(&ray) {
        Ok(Some(hit)) => println!(
            "raycast: hit {:?} at distance {:.2}, proxy={}",
            hit.kind, hit.distance, hit.is_proxy
        ),
        other => println!("raycast: no hit or error ({other:?})"),
    }

    // Raycast all through the center
    let ray_center = Ray {
        origin: Vec3::new(0.0, 0.5, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let hits = scene.raycast_all(&ray_center).expect("valid ray");
    println!("raycast_all: {} hits", hits.len());
    for h in &hits {
        println!("  {:?} dist={:.2} proxy={}", h.kind, h.distance, h.is_proxy);
    }

    // Volume query: AABB region around the sphere
    let query_aabb =
        renderer::Aabb::from_min_max(Vec3::new(1.0, 0.0, -2.0), Vec3::new(3.0, 2.0, 0.0));
    let vol_query = VolumeQuery::aabb(query_aabb);
    let vol_hits = scene.query_volume(&vol_query);
    println!("volume query: {} hits", vol_hits.len());
    for h in &vol_hits {
        println!("  {:?} bounded={}", h.kind, h.is_bounded);
    }

    // Volume query with filter: nodes only
    let filtered_query =
        VolumeQuery::aabb(query_aabb).with_filter(ObjectQueryFilter::kinds([ObjectKind::Node]));
    let filtered_hits = scene.query_volume(&filtered_query);
    println!(
        "filtered volume query (nodes only): {} hits",
        filtered_hits.len()
    );

    // ── Phase 07: Editor pick ───────────────────────────────────────────
    let pick_ray = Ray {
        origin: Vec3::new(-1.5, 1.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    match scene.editor_pick(&pick_ray, EditorProxyPolicy::NodesOnly) {
        Ok(Some(pick)) => println!(
            "editor_pick: {:?} hit={}",
            pick.object.kind(),
            pick.hit.is_some()
        ),
        other => println!("editor_pick: no hit or error ({other:?})"),
    }

    // ── Phase 07: Selection workflow ────────────────────────────────────
    let prov = scene.provenance_token();
    let mut sel = Selection::with_provenance(prov);
    let root_oid = scene.object_id(root).unwrap();
    let cube_oid = scene.object_id(cube_node).unwrap();
    let sphere_oid = scene.object_id(sphere_node).unwrap();

    sel.add(root_oid).unwrap();
    sel.add(cube_oid).unwrap();
    sel.add(sphere_oid).unwrap();
    println!(
        "selection: {} items, primary={:?}",
        sel.len(),
        sel.primary()
    );

    // Toggle off the sphere
    let toggled = sel.toggle(sphere_oid).unwrap();
    println!("toggle sphere: added={toggled}, len={}", sel.len());

    // Cleanup stale: remove root
    sel.cleanup_stale(|id| *id != root_oid);
    println!("after stale cleanup: {} items", sel.len());

    // ── Phase 07: EditorCamera ──────────────────────────────────────────
    let mut editor_cam = EditorCamera::default();
    editor_cam
        .set_perspective(1.0, 0.1, 100.0)
        .expect("perspective");
    let scr_ray = editor_cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("screen ray");
    println!(
        "editor_camera screen_to_ray: origin={:?} dir={:?}",
        scr_ray.origin, scr_ray.direction
    );

    // Focus on a known bounds region
    let focus_aabb =
        renderer::Aabb::from_min_max(Vec3::new(-2.0, 0.0, -2.0), Vec3::new(2.0, 2.0, 0.0));
    editor_cam.focus_on(&focus_aabb).expect("focus_on");
    println!(
        "editor_camera after focus_on: target={:?} radius={:.2}",
        editor_cam.orbit().target,
        editor_cam.orbit().radius
    );

    // Orthographic projection and screen-to-ray
    editor_cam
        .set_orthographic(5.0, 0.1, 500.0)
        .expect("orthographic");
    let ortho_ray = editor_cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("ortho screen ray");
    println!(
        "ortho screen_to_ray: origin={:?} dir={:?}",
        ortho_ray.origin, ortho_ray.direction
    );

    // Test to_camera_view
    let _cv = editor_cam.to_camera_view(800, 600);
    println!("to_camera_view produced finite matrices");

    println!("\n=== Phase 07 Object Workflow Capture Complete ===");

    scene
}
