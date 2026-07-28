//! Phase 05 — Deterministic Persistence
//!
//! Validates:
//! - v2 fixtures pass scene validation
//! - scene save produces deterministic JSON with object_id and components
//! - v1 migration adds default object_id and components fields
//! - attachment limits are enforced at parse time
//! - fragment components survive merge
//! - duplicate object_id rejected at validation
//! - component API: attach, enumerate, hydrate on nodes

use renderer::{
    MeshHandle, Scene, SceneError, SceneNodeId,
};
use renderer::api::SceneFragment;
use renderer::object::{
    ComponentEnvelope, ComponentInstanceId, ComponentKey,
};
use renderer::api::{
    validate_scene_file, validate_scene_str,
};
use serde_json::json;

// ── tests ───────────────────────────────────────────────────────────────

#[test]
fn v2_object_components_fixture_passes_validation() {
    let result = validate_scene_file(
        "tests/fixtures/scenes/v2-object-components.engine.scene.json",
    );
    assert!(result.is_ok(), "fixture should pass validation: {result:?}");
}

#[test]
fn scene_round_trip_preserves_component_data() {
    let mut scene = Scene::new();
    scene.set_scene_id("scene.roundtrip");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "Root").unwrap();

    // Attach a component.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000aa",
    )
    .unwrap();
    let key = ComponentKey::new("test.xp").unwrap();
    let envelope = ComponentEnvelope::new(iid.clone(), key.clone(), 1, json!({"level": 42}))
        .unwrap();
    scene.attach_component(root, envelope).unwrap();

    // Verify it's there.
    let envelopes = scene.component_envelopes(root).unwrap();
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0].key.as_str(), "test.xp");
    assert_eq!(envelopes[0].data["level"], json!(42));

    // Remove component.
    scene.remove_component(root, &key, &iid).unwrap();
    let envelopes = scene.component_envelopes(root).unwrap();
    assert_eq!(envelopes.len(), 0);
}

#[test]
fn save_to_string_includes_object_id_and_components_fields() {
    let mut scene = Scene::new();
    scene.set_scene_id("scene.json_test");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "Root").unwrap();

    // Attach a component to produce components in the output.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000bb",
    )
    .unwrap();
    let key = ComponentKey::new("test.tags").unwrap();
    let envelope = ComponentEnvelope::new(iid, key, 1, json!({"items": ["hero"]})).unwrap();
    scene.attach_component(root, envelope).unwrap();

    let json = scene.save_to_string().unwrap();
    assert!(json.contains("\"format_version\": 2"));
    assert!(json.contains("\"object_id\""));
    assert!(json.contains("\"components\""), "JSON should contain components array");
    assert!(json.contains("\"test.tags\""));
}

#[test]
fn component_free_objects_save_with_identity_and_without_empty_components() {
    let mut scene = Scene::new();
    scene.set_scene_id("scene.compact");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "Root").unwrap();

    let saved: serde_json::Value = serde_json::from_str(&scene.save_to_string().unwrap()).unwrap();
    let node = &saved["nodes"][0];
    assert!(node["object_id"].as_str().is_some_and(|id| id.starts_with("object.")));
    assert!(node.get("components").is_none());
}

#[test]
fn v1_scene_validates_through_migration() {
    let v1_json = json!({
        "format_version": 1,
        "scene_id": "scene.v1_test",
        "root_nodes": ["node.root"],
        "nodes": [
            {
                "id": "node.root",
                "parent": null,
                "name": "Root",
                "transform": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0]
                },
                "asset": null
            }
        ],
        "lights": [
            {
                "id": "light.one",
                "kind": "point",
                "position": [0.0, 1.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 1.0,
                "range": 5.0
            }
        ],
        "directional_lights": [],
        "spot_lights": []
    });

    let json_str = serde_json::to_string_pretty(&v1_json).unwrap();
    let result = validate_scene_str(&json_str);
    assert!(result.is_ok(), "v1 scene should validate: {result:?}");
}

#[test]
fn attachment_limit_enforced() {
    let mut scene = Scene::new();
    let root = scene.create_node_default(None).unwrap();
    // Attach 256 components (the max).
    for i in 0u64..256 {
        let iid = ComponentInstanceId::new(
            format!("component.{i:064x}"),
        )
        .unwrap();
        let key = ComponentKey::new("test.foo").unwrap();
        let env = ComponentEnvelope::new(iid, key, 1, json!({"i": i})).unwrap();
        scene.attach_component(root, env).unwrap();
    }

    // The 257th should fail.
    let overflow_iid = ComponentInstanceId::new(
        "component.ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    .unwrap();
    let overflow_key = ComponentKey::new("test.overflow").unwrap();
    let overflow_env = ComponentEnvelope::new(overflow_iid, overflow_key, 1, json!({"x": 1}))
        .unwrap();
    let result = scene.attach_component(root, overflow_env);
    assert!(result.is_err(), "should reject 257th component: {result:?}");
}

#[test]
fn fragment_components_survive_merge() {
    let mut fragment = SceneFragment::new();
    let root = fragment
        .add_node(None, glam::Mat4::IDENTITY, vec![MeshHandle::new(7, 0)])
        .expect("fragment root");
    fragment.set_root(root).unwrap();

    // Attach component to fragment node via the public components field.
    {
        let node = fragment.node_mut(root).expect("fragment node");
        let iid = ComponentInstanceId::new(
            "component.00000000000000000000000000000000000000000000000000000000000000cc",
        )
        .unwrap();
        let key = ComponentKey::new("test.frag").unwrap();
        let envelope = ComponentEnvelope::new(iid, key, 1, json!({"from": "fragment"})).unwrap();
        node.components.push(envelope);
    }

    let mut scene = Scene::new();
    let mount = scene.merge_fragment(None, fragment).expect("merge");
    let scene_root = mount.mounted_root;

    // Verify component survived merge.
    let envelopes = scene.component_envelopes(scene_root).unwrap();
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0].key.as_str(), "test.frag");
    assert_eq!(envelopes[0].data["from"], json!("fragment"));
}

#[test]
fn duplicate_object_id_rejected_at_validation() {
    let scene_json = json!({
        "format_version": 2,
        "scene_id": "scene.dup",
        "root_nodes": ["node.a"],
        "nodes": [
            {
                "id": "node.a",
                "object_id": "object.0000000000000000000000000000000000000000000000000000000000000001",
                "parent": null,
                "name": "A",
                "transform": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0]
                },
                "asset": null
            },
            {
                "id": "node.b",
                "object_id": "object.0000000000000000000000000000000000000000000000000000000000000001",
                "parent": null,
                "name": "B",
                "transform": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0]
                },
                "asset": null
            }
        ]
    });

    let json_str = serde_json::to_string_pretty(&scene_json).unwrap();
    let result = validate_scene_str(&json_str);
    assert!(result.is_err(), "should reject duplicate object_id: {result:?}");
}

#[test]
fn deterministic_save_has_stable_structure() {
    // Two saves of the same scene should produce identical JSON.
    let mut scene = Scene::new();
    scene.set_scene_id("scene.deterministic");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "Root").unwrap();
    let child = scene
        .create_node(
            Some(root),
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
        )
        .unwrap();
    scene.set_node_name(child, "Child").unwrap();

    // Attach a component to child.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000dd",
    )
    .unwrap();
    let key = ComponentKey::new("test.xp").unwrap();
    let envelope = ComponentEnvelope::new(iid, key, 1, json!({"level": 5})).unwrap();
    scene.attach_component(child, envelope).unwrap();

    // Same scene saved twice produces identical JSON (object_id is stable
    // after first save because persistent IDs are already in the records).
    let json1 = scene.save_to_string().unwrap();
    let json2 = scene.save_to_string().unwrap();
    assert_eq!(json1, json2, "same scene saved twice must produce identical JSON");

    // Verify structure contains expected fields.
    assert!(json1.contains("\"format_version\": 2"));
    assert!(json1.contains("\"object_id\""));
    assert!(json1.contains("\"components\""));
    assert!(json1.contains("\"test.xp\""));
    assert!(json1.contains("\"level\": 5"));
}

#[test]
fn node_object_record_is_accessible() {
    let mut scene = Scene::new();
    scene.set_scene_id("scene.record");
    let root = scene.create_node_default(None).unwrap();

    let record = scene.get_node_record(root).expect("record should exist");
    assert!(record.persistent_id.to_string().starts_with("object."));
    assert!(record.stable_id.as_deref().unwrap().starts_with("node."));

    // Attach a component and verify it shows up.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000ee",
    )
    .unwrap();
    let key = ComponentKey::new("test.health").unwrap();
    let envelope = ComponentEnvelope::new(iid, key, 1, json!({"value": 100})).unwrap();
    scene.attach_component(root, envelope).unwrap();

    let record = scene.get_node_record(root).unwrap();
    assert_eq!(record.component_store.len(), 1);
}
