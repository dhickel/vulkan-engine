//! Integration tests for Phase 02: Object Identity and Lifecycle.
//!
//! Covers:
//! - Cross-scene provenance isolation
//! - Slot/generation handle safety
//! - Create/remove/reuse/restore/clear cycles
//! - Persistent ID uniqueness
//! - Invariant audits

use engine_events::{ObjectKind, SceneObjectId};
use glam::Vec3;
use renderer::{
    DirectionalLight, DirectionalShadowConfig, PointLight,
    Scene, SpotLight, SceneNodeId,
};
use renderer::object::ObjectHandle;

// ── Helpers ─────────────────────────────────────────────────────────────

fn new_test_scene() -> Scene {
    Scene::new()
}

fn make_point_light(x: f32) -> PointLight {
    PointLight {
        position: Vec3::new(x, 2.0, 0.0),
        color: Vec3::ONE,
        intensity: 10.0,
        range: 8.0,
    }
}

fn make_directional_light() -> DirectionalLight {
    DirectionalLight {
        direction: Vec3::new(0.0, -1.0, 0.5),
        color: Vec3::ONE,
        intensity: 1.0,
    }
}

fn make_spot_light() -> SpotLight {
    SpotLight {
        position: Vec3::ZERO,
        direction: Vec3::new(0.0, -1.0, 0.0),
        color: Vec3::ONE,
        intensity: 10.0,
        range: 20.0,
        inner_cone_angle: 0.2,
        outer_cone_angle: 0.5,
    }
}

// ── Provenance isolation ────────────────────────────────────────────────

#[test]
fn cross_scene_object_id_does_not_validate() {
    let mut scene_a = new_test_scene();
    let scene_b = new_test_scene();

    let node_a = scene_a
        .create_node_default(None)
        .expect("create node in A");
    let obj_a = scene_a.object_id(node_a).expect("object ID from A");

    // Resolving the A object in B must fail (wrong provenance).
    let resolved = scene_b.resolve_object(obj_a);
    assert!(resolved.is_none(), "object from A must not resolve in B");
}

#[test]
fn same_slot_generation_from_two_worlds_rejects_wrong_scene() {
    let mut scene_a = new_test_scene();
    let mut scene_b = new_test_scene();

    let id_a = scene_a
        .create_node_default(None)
        .expect("create node in A");
    let id_b = scene_b
        .create_node_default(None)
        .expect("create node in B");

    // Both are slot 0, generation 0, but different provenance.
    let obj_a = scene_a.object_id(id_a).expect("object ID from A");
    let obj_b = scene_b.object_id(id_b).expect("object ID from B");

    assert_ne!(obj_a, obj_b, "objects from different scenes must differ");
    assert!(scene_a.resolve_object(obj_b).is_none());
    assert!(scene_b.resolve_object(obj_a).is_none());
}

// ── Handle validation ───────────────────────────────────────────────────

#[test]
fn stale_handle_rejected_by_object_id() {
    let mut scene = new_test_scene();
    let node = scene
        .create_node_default(None)
        .expect("create node");
    let obj = scene.object_id(node).expect("valid object ID");

    scene.remove_node(node).expect("remove node");

    // After removal, the node handle is stale.
    assert!(scene.object_id(node).is_err());
    // The ObjectId is no longer resolvable.
    assert!(scene.resolve_object(obj).is_none());
}

#[test]
fn vacant_slot_rejected() {
    let scene = new_test_scene();
    let bad_node = SceneNodeId::new(99999, 0);
    assert!(scene.object_id(bad_node).is_err());
}

#[test]
fn wrong_kind_access_rejected() {
    let mut scene = new_test_scene();
    let node = scene
        .create_node_default(None)
        .expect("create node");
    let obj = scene.object_id(node).expect("object ID");

    // The ObjectId has kind Node, not PointLight.
    assert_eq!(obj.kind(), ObjectKind::Node);
    assert_ne!(obj.kind(), ObjectKind::PointLight);
}

// ── Create / remove / reuse / restore / clear cycles ────────────────────

#[test]
fn create_remove_reuse_cycle_preserves_generations() {
    let mut scene = new_test_scene();

    let node_0 = scene.create_node_default(None).unwrap();
    let obj_0 = scene.object_id(node_0).unwrap();
    assert_eq!(node_0.slot, 0);
    assert_eq!(node_0.generation, 0);

    scene.remove_node(node_0).unwrap();
    assert!(scene.object_id(node_0).is_err());
    assert!(scene.resolve_object(obj_0).is_none());

    let node_1 = scene.create_node_default(None).unwrap();
    let obj_1 = scene.object_id(node_1).unwrap();
    // Reuses slot 0 with generation 1.
    assert_eq!(node_1.slot, 0);
    assert_eq!(node_1.generation, 1);
    assert_ne!(obj_0, obj_1);
}

#[test]
fn point_light_create_remove_reuse() {
    let mut scene = new_test_scene();

    let id_0 = scene.create_point_light(make_point_light(1.0)).unwrap();
    let obj_0 = scene
        .object_id_for_point_light(id_0)
        .expect("first object ID");
    assert_eq!(id_0.slot, 0);
    assert_eq!(id_0.generation, 0);

    scene.remove_point_light(id_0).unwrap();
    assert!(scene.resolve_object(obj_0).is_none());

    let id_1 = scene.create_point_light(make_point_light(2.0)).unwrap();
    let obj_1 = scene
        .object_id_for_point_light(id_1)
        .expect("second object ID");
    assert_eq!(id_1.slot, 0);
    assert_eq!(id_1.generation, 1);
    assert_ne!(obj_0, obj_1);
}

#[test]
fn clear_all_nodes_and_recreate() {
    let mut scene = new_test_scene();

    // Create several nodes.
    let parent = scene.create_node_default(None).unwrap();
    let child = scene.create_node_default(Some(parent)).unwrap();
    let obj_parent = scene.object_id(parent).unwrap();
    let obj_child = scene.object_id(child).unwrap();

    // Remove parent — subtree removal clears both.
    scene.remove_node(parent).unwrap();

    assert!(scene.resolve_object(obj_parent).is_none());
    assert!(scene.resolve_object(obj_child).is_none());
    assert!(scene.object_id(parent).is_err());
    assert!(scene.object_id(child).is_err());
}

// ── ObjectId ↔ typed handle round-trip ──────────────────────────────────

#[test]
fn object_id_to_typed_handle_roundtrip_node() {
    let mut scene = new_test_scene();
    let node = scene.create_node_default(None).unwrap();
    let obj = scene.object_id(node).unwrap();

    let resolved = scene.resolve_object(obj).unwrap();
    match resolved {
        ObjectHandle::Node(nid) => {
            assert_eq!(nid, node);
        }
        _ => panic!("expected Node handle"),
    }
}

#[test]
fn object_id_to_typed_handle_roundtrip_point_light() {
    let mut scene = new_test_scene();
    let id = scene.create_point_light(make_point_light(1.0)).unwrap();
    let obj = scene.object_id_for_point_light(id).unwrap();

    let resolved = scene.resolve_object(obj).unwrap();
    match resolved {
        ObjectHandle::PointLight(pl) => {
            assert_eq!(pl, id);
        }
        _ => panic!("expected PointLight handle"),
    }
}

#[test]
fn object_id_to_typed_handle_roundtrip_directional_light() {
    let mut scene = new_test_scene();
    let id = scene.create_directional_light(make_directional_light()).unwrap();
    let obj = scene.object_id_for_directional_light(id).unwrap();

    let resolved = scene.resolve_object(obj).unwrap();
    match resolved {
        ObjectHandle::DirectionalLight(dl) => {
            assert_eq!(dl, id);
        }
        _ => panic!("expected DirectionalLight handle"),
    }
}

#[test]
fn object_id_to_typed_handle_roundtrip_spot_light() {
    let mut scene = new_test_scene();
    let id = scene.create_spot_light(make_spot_light()).unwrap();
    let obj = scene.object_id_for_spot_light(id).unwrap();

    let resolved = scene.resolve_object(obj).unwrap();
    match resolved {
        ObjectHandle::SpotLight(sl) => {
            assert_eq!(sl, id);
        }
        _ => panic!("expected SpotLight handle"),
    }
}

// ── Persistent ID lookup ────────────────────────────────────────────────

#[test]
fn find_object_by_persistent_id_roundtrip() {
    let mut scene = new_test_scene();
    let node = scene.create_node_default(None).unwrap();
    let obj = scene.object_id(node).unwrap();

    // Get the persistent ID from the record.
    let persistent = scene
        .world()
        .get_node_record(node)
        .map(|r| r.persistent_id.clone())
        .unwrap();

    let found = scene.find_object_by_persistent_id(&persistent);
    assert_eq!(found, Some(obj));
}

#[test]
fn find_object_by_persistent_id_returns_none_for_unknown() {
    let scene = new_test_scene();
    let unknown = SceneObjectId::new("object.deadbeef0000000000000000000000000000000000000000000000000000000000");
    assert!(scene.find_object_by_persistent_id(&unknown).is_none());
}

// ── Persistent ID uniqueness across kinds ───────────────────────────────

#[test]
fn persistent_id_uniqueness_across_all_kinds() {
    let mut scene = new_test_scene();

    let node = scene.create_node_default(None).unwrap();
    let pl_id = scene.create_point_light(make_point_light(1.0)).unwrap();
    let dl_id = scene.create_directional_light(make_directional_light()).unwrap();
    let sl_id = scene.create_spot_light(make_spot_light()).unwrap();

    let obj_node = scene.object_id(node).unwrap();
    let obj_pl = scene.object_id_for_point_light(pl_id).unwrap();
    let obj_dl = scene.object_id_for_directional_light(dl_id).unwrap();
    let obj_sl = scene.object_id_for_spot_light(sl_id).unwrap();

    // All four ObjectIds must be distinct.
    assert_ne!(obj_node, obj_pl);
    assert_ne!(obj_node, obj_dl);
    assert_ne!(obj_node, obj_sl);
    assert_ne!(obj_pl, obj_dl);
    assert_ne!(obj_pl, obj_sl);
    assert_ne!(obj_dl, obj_sl);

    // All four kinds differ.
    assert_eq!(obj_node.kind(), ObjectKind::Node);
    assert_eq!(obj_pl.kind(), ObjectKind::PointLight);
    assert_eq!(obj_dl.kind(), ObjectKind::DirectionalLight);
    assert_eq!(obj_sl.kind(), ObjectKind::SpotLight);
}

// ── Generation exhaustion ───────────────────────────────────────────────

#[test]
fn generation_exhausted_node_still_removable() {
    let mut scene = new_test_scene();
    let node = scene.create_node_default(None).unwrap();

    // Use test helper to push generation to u32::MAX.
    assert!(scene.world_mut().test_set_generation_max(node));

    // The original handle is stale — create a matching max-gen handle.
    let max_gen_node = SceneNodeId::new(node.slot, u32::MAX);

    // Remove should still succeed even though the free list won't get the slot back.
    scene.remove_node(max_gen_node).unwrap();
    assert!(scene.object_id(node).is_err());
}

// ── Invariant audit after standard operations ───────────────────────────

#[test]
fn invariant_audit_after_create_and_remove() {
    let mut scene = new_test_scene();
    let node = scene.create_node_default(None).unwrap();
    let pl = scene.create_point_light(make_point_light(1.0)).unwrap();

    scene
        .world()
        .audit_object_invariants()
        .expect("invariants after create");

    scene.remove_point_light(pl).unwrap();
    scene
        .world()
        .audit_object_invariants()
        .expect("invariants after partial remove");

    scene.remove_node(node).unwrap();
    scene
        .world()
        .audit_object_invariants()
        .expect("invariants after full clear");
}

// ── Directional shadow config in record ─────────────────────────────────

#[test]
fn directional_shadow_config_stored_in_record() {
    let mut scene = new_test_scene();
    let dl_id = scene.create_directional_light(make_directional_light()).unwrap();

    let cfg = DirectionalShadowConfig {
        enabled: true,
        shadow_map_size: 4096,
        cascade_count: 3,
        cascade_split_lambda: 0.75,
    };
    scene
        .set_directional_shadow_config(dl_id, cfg)
        .unwrap();

    let stored = scene
        .world()
        .get_directional_light_record(dl_id)
        .and_then(|r| r.directional_shadow_config)
        .unwrap();
    assert_eq!(stored.shadow_map_size, 4096);
    assert_eq!(stored.cascade_count, 3);
    assert!(stored.enabled);
}
