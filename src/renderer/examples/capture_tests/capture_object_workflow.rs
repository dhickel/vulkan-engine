//! Headless capture test: Object Workflow (Queries + Selection + EditorCamera)
//!
//! Validates the Phase 07 query, selection, and editor-camera APIs
//! in a headless scene setup.  No GPU rendering required — this is a
//! pure-CPU workflow test that exercises the new APIs under
//! deterministic conditions.
//!
//! Run: cargo run -p renderer --example capture_object_workflow

use glam::{Mat4, Vec3};
use renderer::{
    object::{
        ObjectKind,
        query::{EditorProxyPolicy, VolumeQuery},
        selection::Selection,
    },
    Aabb, EditorCamera, MeshBoundsEntry, PointLight, Ray, Scene, SceneBounds,
};

fn main() {
    println!("=== Phase 07 Object Workflow Capture ===");

    // --- Scene setup ---
    let mut scene = Scene::new();
    let cube = Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5));

    // Create hierarchy using public API
    let root = scene
        .create_node_default(None)
        .expect("root");
    scene
        .add_mesh_with_bounds(root, renderer::MeshHandle::new(1, 0), SceneBounds::Known(cube))
        .expect("add mesh to root");

    let child_a = scene
        .create_node_default(Some(root))
        .expect("child_a");
    scene.set_transform(child_a, Mat4::from_translation(Vec3::new(2.0, 0.0, -3.0)))
        .expect("set transform child_a");
    scene
        .add_mesh_with_bounds(
            child_a,
            renderer::MeshHandle::new(2, 0),
            SceneBounds::Known(cube),
        )
        .expect("add mesh to child_a");

    let child_b = scene
        .create_node_default(Some(root))
        .expect("child_b");
    scene.set_transform(child_b, Mat4::from_translation(Vec3::new(-2.0, 0.0, -6.0)))
        .expect("set transform child_b");
    scene
        .add_mesh_with_bounds(
            child_b,
            renderer::MeshHandle::new(3, 0),
            SceneBounds::Proxy(cube),
        )
        .expect("add mesh to child_b");

    // Add some lights
    let pl = scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 3.0, -4.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 20.0,
        })
        .expect("create point light");

    println!("Scene: root={root:?}, child_a={child_a:?}, child_b={child_b:?}, pl={pl:?}");

    // --- Raycast test ---
    let ray = Ray {
        origin: Vec3::new(2.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    match scene.raycast(&ray) {
        Ok(Some(hit)) => {
            println!(
                "raycast hit: object={:?} distance={:.3} kind={:?} proxy={}",
                hit.object, hit.distance, hit.kind, hit.is_proxy
            );
            assert!(!hit.is_proxy, "child_a should have known bounds");
        }
        other => panic!("expected raycast hit, got {other:?}"),
    }

    // --- Raycast all ---
    let ray_center = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let hits = scene.raycast_all(&ray_center).expect("valid ray");
    println!("raycast_all returned {} hits:", hits.len());
    for h in &hits {
        println!(
            "  object={:?} distance={:.3} kind={:?} proxy={}",
            h.object, h.distance, h.kind, h.is_proxy
        );
    }
    assert!(hits.len() >= 1, "should hit at least root");
    assert!(
        hits.windows(2).all(|w| w[0].distance <= w[1].distance),
        "hits must be sorted by distance"
    );

    // --- Volume query ---
    let query_aabb =
        Aabb::from_min_max(Vec3::new(1.0, -1.0, -4.0), Vec3::new(3.0, 1.0, -2.0));
    let vol_query = VolumeQuery::aabb(query_aabb);
    let vol_hits = scene.query_volume(&vol_query);
    println!("volume query returned {} hits", vol_hits.len());
    for h in &vol_hits {
        println!(
            "  object={:?} kind={:?} bounded={}",
            h.object, h.kind, h.is_bounded
        );
    }
    // child_a should be in this region.
    assert!(
        vol_hits.iter().any(|h| h.kind == ObjectKind::Node),
        "volume query should hit child_a"
    );

    // --- Editor pick ---
    let pick_result = scene
        .editor_pick(&ray_center, EditorProxyPolicy::NodesOnly)
        .expect("valid ray");
    println!("editor_pick (NodesOnly): {pick_result:?}");
    assert!(pick_result.is_some());

    let pick_all = scene
        .editor_pick(&ray_center, EditorProxyPolicy::All)
        .expect("valid ray");
    println!("editor_pick (All): {pick_all:?}");
    assert!(pick_all.is_some());

    // Light-only pick
    let ray_to_light = Ray {
        origin: Vec3::new(0.0, 3.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let pick_light = scene
        .editor_pick(&ray_to_light, EditorProxyPolicy::NodesAndBoundedLights)
        .expect("valid ray");
    println!("editor_pick (NodesAndBoundedLights): {pick_light:?}");

    // --- Selection workflow ---
    let root_oid = scene.object_id(root).unwrap();
    let child_a_oid = scene.object_id(child_a).unwrap();
    let child_b_oid = scene.object_id(child_b).unwrap();

    let mut sel = Selection::new();
    sel.add(root_oid);
    sel.add(child_a_oid);
    sel.add(child_b_oid);
    println!("selection ({} items):", sel.len());
    for id in sel.iter() {
        println!("  {id}");
    }
    assert_eq!(sel.len(), 3);
    assert!(sel.primary().is_some());

    // Cleanup stale: remove child_a from selection (simulate using
    // ObjectKind filtering)
    let child_a_kind = child_a_oid.kind();
    sel.cleanup_stale(|id| id.kind() != child_a_kind || id != &child_a_oid);
    println!("after stale cleanup: {} items", sel.len());
    assert_eq!(sel.len(), 2);

    // Remap: drop child_a, map everything else to root
    sel.remap(|id| {
        if *id == child_a_oid {
            None
        } else {
            Some(root_oid)
        }
    });
    println!("after remap: {} items", sel.len());
    assert_eq!(sel.len(), 1);

    // --- EditorCamera ---
    let mut editor_cam = EditorCamera::default();
    editor_cam.set_perspective(1.0, 0.1, 100.0).expect("perspective");
    let screen_ray = editor_cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("ray from screen");
    println!(
        "screen_to_ray origin={:?} dir={:?}",
        screen_ray.origin, screen_ray.direction
    );

    // Focus on the scene bounds
    let scene_aabb =
        Aabb::from_min_max(Vec3::new(-3.0, -1.0, -7.0), Vec3::new(3.0, 1.0, 1.0));
    editor_cam.focus_on(&scene_aabb).expect("focus");
    println!(
        "after focus_on: target={:?} radius={:.3}",
        editor_cam.orbit().target,
        editor_cam.orbit().radius
    );

    println!("\n=== All Phase 07 Object Workflow Tests Passed ===");
}
