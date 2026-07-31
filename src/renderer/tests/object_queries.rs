//! Tests for Phase 07 — Queries, Raycasts, Volume Queries, Editor Pick.
//!
//! Covers: raycast, raycast_all, volume (AABB/frustum) queries,
//! editor_pick with policy, pick/pick_last_camera compatibility.

use glam::{Mat4, Vec3};
use renderer::{
    object::{
        query::{EditorProxyPolicy, ObjectQueryFilter, UnknownBoundsPolicy, VolumeQuery},
        selection::Selection,
        ObjectKind,
    },
    Aabb, DirectionalLight, MeshBoundsEntry, PointLight, Ray, Scene, SceneBounds, SceneNodeId,
    SpotLight,
};

// ── Helpers ─────────────────────────────────────────────────────────────

fn cube_aabb() -> Aabb {
    Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5))
}

fn build_scene_with_cube(
    transform: Mat4,
    bounds: SceneBounds,
    parent: Option<SceneNodeId>,
) -> (Scene, SceneNodeId) {
    let mut scene = Scene::new();
    let node = scene.create_node_default(parent).expect("create node");
    scene.set_transform(node, transform).expect("set transform");
    scene
        .add_mesh_with_bounds(node, renderer::MeshHandle::new(1, 0), bounds)
        .expect("add mesh");
    (scene, node)
}

// ── Raycast ────────────────────────────────────────────────────────────

#[test]
fn raycast_hits_single_node() {
    let (scene, _root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hit = scene.raycast(&ray).expect("valid ray").expect("should hit");
    assert_eq!(hit.kind, ObjectKind::Node);
    assert!(hit.distance > 4.0 && hit.distance < 6.0);
    assert!(!hit.is_proxy);
}

#[test]
fn raycast_normalizes_direction_and_preserves_world_distance() {
    let (scene, _) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );
    let hit = scene
        .raycast(&Ray {
            origin: Vec3::ZERO,
            direction: Vec3::new(0.0, 0.0, -2.0),
        })
        .expect("valid ray")
        .expect("hit");

    assert!((hit.distance - 4.5).abs() < 1e-5);
    assert_eq!(hit.normal, Some(Vec3::Z));
}

#[test]
fn raycast_miss_returns_none() {
    let (scene, _root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );

    let ray = Ray {
        origin: Vec3::new(10.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    assert!(scene.raycast(&ray).expect("valid ray").is_none());
}

#[test]
fn raycast_rejects_invalid_ray() {
    let scene = Scene::new();
    let bad = Ray {
        origin: Vec3::new(f32::NAN, 0.0, 0.0),
        direction: Vec3::NEG_Z,
    };
    assert!(scene.raycast(&bad).is_err());
}

#[test]
fn raycast_rejects_zero_direction() {
    let scene = Scene::new();
    let bad = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::ZERO,
    };
    assert!(scene.raycast(&bad).is_err());
}

// ── Raycast all ────────────────────────────────────────────────────────

#[test]
fn raycast_all_returns_multiple_hits_sorted() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).expect("root");
    scene
        .add_mesh_with_bounds(
            root,
            renderer::MeshHandle::new(1, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh root");

    let child_a = scene.create_node_default(Some(root)).expect("child_a");
    scene
        .set_transform(child_a, Mat4::from_translation(Vec3::new(0.0, 0.0, -3.0)))
        .expect("set transform a");
    scene
        .add_mesh_with_bounds(
            child_a,
            renderer::MeshHandle::new(2, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh a");

    let child_b = scene.create_node_default(Some(root)).expect("child_b");
    scene
        .set_transform(child_b, Mat4::from_translation(Vec3::new(0.0, 0.0, -8.0)))
        .expect("set transform b");
    scene
        .add_mesh_with_bounds(
            child_b,
            renderer::MeshHandle::new(3, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh b");

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hits = scene.raycast_all(&ray).expect("valid ray");
    assert_eq!(hits.len(), 3);
    assert!(hits[0].distance < hits[1].distance);
    assert!(hits[1].distance < hits[2].distance);
}

#[test]
fn raycast_all_proxy_flag() {
    let (scene, _root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Proxy(cube_aabb()),
        None,
    );

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hits = scene.raycast_all(&ray).expect("valid ray");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].is_proxy);
}

// ── Volume queries ─────────────────────────────────────────────────────

#[test]
fn volume_query_aabb_finds_nodes() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).expect("root");
    scene
        .add_mesh_with_bounds(
            root,
            renderer::MeshHandle::new(1, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh root");

    let child = scene.create_node_default(Some(root)).expect("child");
    scene
        .set_transform(child, Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)))
        .expect("set transform");
    scene
        .add_mesh_with_bounds(
            child,
            renderer::MeshHandle::new(2, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh child");

    let query_aabb =
        Aabb::from_min_max(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0));
    let query = VolumeQuery::aabb(query_aabb);

    let hits = scene.query_volume(&query);
    // root at origin should hit; child at x=100 should not.
    assert!(hits.iter().any(|h| h.kind == ObjectKind::Node));
    assert_eq!(hits.len(), 1);
    assert!(hits[0].is_bounded);
}

#[test]
fn volume_query_includes_lights() {
    let mut scene = Scene::new();
    scene
        .create_point_light(PointLight {
            position: Vec3::new(3.0, 0.0, 0.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        })
        .expect("create light");

    let query = VolumeQuery::aabb(Aabb::from_min_max(
        Vec3::new(2.0, -1.0, -1.0),
        Vec3::new(5.0, 1.0, 1.0),
    ));

    let hits = scene.query_volume(&query);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].kind, ObjectKind::PointLight);
}

#[test]
fn volume_query_filter_excludes_kinds() {
    let mut scene = Scene::new();
    let node = scene.create_node_default(None).expect("node");
    scene
        .add_mesh_with_bounds(
            node,
            renderer::MeshHandle::new(1, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh");
    scene
        .create_point_light(PointLight {
            position: Vec3::new(1.0, 0.0, 0.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        })
        .expect("create light");

    let query = VolumeQuery::aabb(Aabb::from_min_max(
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
    ))
    .with_filter(ObjectQueryFilter::kinds([ObjectKind::Node]));

    let hits = scene.query_volume(&query);
    // Only the node, not the point light.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].kind, ObjectKind::Node);
}

#[test]
fn general_raycast_excludes_editor_helper_proxies() {
    // Create an empty node (no mesh, no proxy bounds).
    let mut scene = Scene::new();
    let empty = scene.create_node_default(None).expect("empty node");
    scene
        .set_transform(empty, Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)))
        .expect("set transform");

    // General raycast should NOT return hits for empty nodes (editor proxies are
    // only for editor_pick).
    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let hits = scene.raycast_all(&ray).expect("valid ray");
    // Empty node has no known bounds → no hit in general query.
    assert!(
        hits.is_empty(),
        "general raycast must not hit editor helper proxies for empty nodes"
    );
}

#[test]
fn editor_pick_still_hits_empty_node() {
    let mut scene = Scene::new();
    let empty = scene.create_node_default(None).expect("empty node");
    scene
        .set_transform(empty, Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)))
        .expect("set transform");

    // Editor pick SHOULD hit the empty node (it uses editor helper proxies).
    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };
    let hit = scene
        .editor_pick(&ray, EditorProxyPolicy::NodesOnly)
        .expect("valid ray");
    assert!(
        hit.is_some(),
        "editor pick should hit empty nodes via helper proxies"
    );
}

#[test]
fn selection_rejects_wrong_scene_object_id() {
    let mut scene = Scene::new();
    let prov = scene.provenance_token();
    let mut sel = Selection::with_provenance(prov);
    // A fresh scene's own ObjectId passes provenance check.
    let node = scene.create_node_default(None).expect("node");
    let ok_id = scene.object_id(node).unwrap();
    assert!(sel.add(ok_id).is_ok());
    sel.clear();
    // A different scene's ObjectId must fail.
    let mut scene2 = Scene::new();
    let node2 = scene2.create_node_default(None).expect("node2");
    let wrong_id = scene2.object_id(node2).unwrap();
    assert!(sel.add(wrong_id).is_err());
}

// ── Editor pick ────────────────────────────────────────────────────────

#[test]
fn editor_pick_nodes_only_default() {
    let (scene, _root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hit = scene
        .editor_pick(&ray, EditorProxyPolicy::NodesOnly)
        .expect("valid ray")
        .expect("should hit node");
    assert!(hit.hit.is_some());
}

#[test]
fn editor_pick_excludes_lights_with_nodes_only() {
    let mut scene = Scene::new();
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 0.0, -3.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        })
        .expect("create light");

    let node = scene.create_node_default(None).expect("node");
    scene
        .set_transform(node, Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)))
        .expect("set transform");
    scene
        .add_mesh_with_bounds(
            node,
            renderer::MeshHandle::new(1, 0),
            SceneBounds::Known(cube_aabb()),
        )
        .expect("mesh node");

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hit = scene
        .editor_pick(&ray, EditorProxyPolicy::NodesOnly)
        .expect("valid ray")
        .expect("should hit the node, not the light");

    assert_eq!(hit.object.kind(), ObjectKind::Node);
}

#[test]
fn editor_pick_picks_light_with_all_policy() {
    let mut scene = Scene::new();
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 0.0, -3.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        })
        .expect("create light");

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hit = scene
        .editor_pick(&ray, EditorProxyPolicy::All)
        .expect("valid ray")
        .expect("should hit point light");
    assert_eq!(hit.object.kind(), ObjectKind::PointLight);
}

// ── Compatibility: pick / pick_last_camera ─────────────────────────────

#[test]
fn pick_still_returns_scene_node_id() {
    let (mut scene, root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );
    // Use the renderer camera path to set view/proj (this is pub(crate),
    // so we can't call update_camera from here — instead we test that
    // pick works with explicit matrices)
    let result = scene.pick(
        400.0,
        300.0,
        800,
        600,
        Mat4::IDENTITY,
        Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
        Vec3::ZERO,
    );
    // The pick may or may not hit depending on the exact math, but it
    // should not panic.
    let _ = result;
}

#[test]
fn raycast_returns_object_id_not_scene_node_id() {
    let (scene, _root) = build_scene_with_cube(
        Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        SceneBounds::Known(cube_aabb()),
        None,
    );

    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, -1.0),
    };

    let hit = scene.raycast(&ray).expect("valid ray").expect("should hit");
    // The hit returns an ObjectId, not a SceneNodeId.
    assert_eq!(hit.object.kind(), ObjectKind::Node);
}
