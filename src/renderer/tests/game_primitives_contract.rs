//! Phase 01 — Game Primitives Contract
//!
//! Validates:
//! - `SCENE_FORMAT_VERSION == 2`
//! - Opaque future game component saves and reloads without alteration
//! - Serialized output contains no ObjectId, slot, generation, Rapier
//!   handle, or playback handle fields
//! - ObjectCapabilities and ObjectTransform remain exhaustive for four kinds
//!
//! Validation: `cargo test -p renderer game_primitives_contract`

use engine_events::ObjectKind;
use glam::Vec3;
use renderer::api::validate_scene_str;
use renderer::object::{
    ComponentEnvelope, ComponentInstanceId, ComponentKey, ObjectCapabilities, ObjectTransform,
    TransformCapabilities,
};
use renderer::{PointLight, Scene};
use serde_json::json;

// ── Format version ─────────────────────────────────────────────────────

#[test]
fn scene_format_version_constant_is_2() {
    // `api::scene` is crate-private, so this spec-only integration test asserts
    // the canonical declaration without widening the production API surface.
    let source = include_str!("../src/api/scene.rs");
    let declaration = source
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("pub const SCENE_FORMAT_VERSION:")
        })
        .expect("SCENE_FORMAT_VERSION declaration must exist");
    assert_eq!(
        declaration.trim(),
        "pub const SCENE_FORMAT_VERSION: u32 = 2;",
        "SCENE_FORMAT_VERSION must remain 2"
    );

    // Assert the value used by serialized scene output and explicit validation.
    let mut scene = Scene::new();
    scene.set_scene_id("scene.v2_check");
    scene.create_node_default(None).unwrap();

    let saved = scene.save_to_string().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(
        parsed["format_version"].as_u64(),
        Some(2),
        "serialized format_version must be 2"
    );

    // Also validate a v2 scene explicitly.
    let v2_json = json!({
        "format_version": 2,
        "scene_id": "scene.v2_explicit",
        "root_nodes": [],
        "nodes": []
    });
    let v2_str = serde_json::to_string_pretty(&v2_json).unwrap();
    assert!(
        validate_scene_str(&v2_str).is_ok(),
        "explicit v2 scene must validate"
    );

    // v3 must be rejected.
    let v3_json = json!({
        "format_version": 3,
        "scene_id": "scene.v3_reject",
        "root_nodes": [],
        "nodes": []
    });
    let v3_str = serde_json::to_string_pretty(&v3_json).unwrap();
    assert!(
        validate_scene_str(&v3_str).is_err(),
        "v3 scene must be rejected"
    );
}

// ── Opaque future game component round-trip ────────────────────────────

#[test]
fn opaque_future_game_component_survives_save_roundtrip() {
    // Create a scene with an opaque "future game" component that uses only
    // durable references (SceneObjectId, AssetId strings) — no ObjectId,
    // slot, generation, Rapier handle, or playback handle fields.
    let mut scene = Scene::new();
    scene.set_scene_id("scene.future_game");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "PlayerSpawn").unwrap();

    // Attach a component representing a hypothetical future game system
    // with durable-only references.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000ff",
    )
    .unwrap();
    let key = ComponentKey::new("game.character").unwrap();
    let data = json!({
        "character_class": "warrior",
        "level": 5,
        "inventory_refs": [
            "asset.sword_of_truth",
            "asset.healing_potion"
        ],
        "spawn_ref": "object.0000000000000000000000000000000000000000000000000000000000000001",
        "stats": {
            "health": 100,
            "mana": 50
        }
    });
    let envelope = ComponentEnvelope::new(iid.clone(), key.clone(), 1, data.clone()).unwrap();
    scene.attach_component(root, envelope).unwrap();

    // Save to string.
    let saved = scene.save_to_string().unwrap();

    // Parse to verify structural integrity.
    let parsed: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(
        parsed["format_version"].as_u64(),
        Some(2),
        "format_version must be 2"
    );

    // The node must carry an object_id and components array.
    let nodes = parsed["nodes"].as_array().expect("nodes must be an array");
    assert_eq!(nodes.len(), 1);
    let node = &nodes[0];
    assert!(
        node["object_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("object.")),
        "node must have a durable object_id"
    );

    let components = node["components"]
        .as_array()
        .expect("node must have components array");
    assert_eq!(components.len(), 1);
    assert_eq!(components[0]["type_key"], "game.character");
    assert_eq!(components[0]["schema_version"], 1);
    assert_eq!(components[0]["data"]["character_class"], "warrior");
    assert_eq!(components[0]["data"]["level"], 5);
    assert_eq!(
        components[0]["data"]["inventory_refs"][0],
        "asset.sword_of_truth"
    );
    assert_eq!(
        components[0]["data"]["spawn_ref"],
        "object.0000000000000000000000000000000000000000000000000000000000000001"
    );

    // Verify the component data survived intact.
    assert_eq!(
        components[0]["data"], data,
        "component data must round-trip unchanged"
    );

    // Validate the persisted document after its opaque component payload has
    // crossed the save boundary. Unregistered component keys remain canonical
    // JSON and need no runtime adapter or Vulkan-backed AssetManager to load.
    let validation = validate_scene_str(&saved);
    assert!(
        validation.is_ok(),
        "opaque component scene must validate: {validation:?}"
    );
}

// ── No transient handles in serialized output ──────────────────────────

#[test]
fn serialized_output_contains_no_transient_handle_fields() {
    let mut scene = Scene::new();
    scene.set_scene_id("scene.clean");
    let root = scene.create_node_default(None).unwrap();
    scene.set_node_name(root, "Root").unwrap();

    // Attach a plain component.
    let iid = ComponentInstanceId::new(
        "component.00000000000000000000000000000000000000000000000000000000000000ee",
    )
    .unwrap();
    let key = ComponentKey::new("test.clean").unwrap();
    let envelope = ComponentEnvelope::new(iid, key, 1, json!({"value": 1})).unwrap();
    scene.attach_component(root, envelope).unwrap();

    // Create a point light to exercise more serialization paths.
    scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 1.0, 0.0),
            color: Vec3::new(1.0, 0.0, 0.0),
            intensity: 1.0,
            range: 5.0,
        })
        .unwrap();

    let saved = scene.save_to_string().unwrap();

    // Durable "object_id" must appear (it's the persistent identity).
    assert!(
        saved.contains("\"object_id\""),
        "durable object_id must appear in serialized scene"
    );

    // These transient handle field names must never appear as JSON keys.
    let forbidden_transient_keys = [
        "\"ObjectId\"",
        "\"slot\"",
        "\"generation\"",
        "\"rapier_handle\"",
        "\"playback_handle\"",
    ];

    for key in &forbidden_transient_keys {
        assert!(
            !saved.contains(key),
            "serialized scene must not contain transient key: {key}"
        );
    }
}

// ── ObjectCapabilities exhaustive for four kinds ───────────────────────

#[test]
fn object_capabilities_exhaustive_for_four_kinds() {
    // Every ObjectKind variant must have valid ObjectCapabilities.
    for kind in &[
        ObjectKind::Node,
        ObjectKind::PointLight,
        ObjectKind::DirectionalLight,
        ObjectKind::SpotLight,
    ] {
        let caps = ObjectCapabilities::for_kind(*kind);
        assert!(
            caps.supports_persistent_id,
            "{kind:?} must support persistent_id"
        );
        assert!(caps.supports_transform, "{kind:?} must support transform");
        assert!(
            caps.transform_caps.is_some(),
            "{kind:?} must have transform_caps"
        );
    }

    // Node-specific capabilities.
    let node_caps = ObjectCapabilities::for_kind(ObjectKind::Node);
    assert_eq!(
        node_caps.transform_caps,
        Some(TransformCapabilities::FullAffine)
    );
    assert!(node_caps.supports_children, "Node must support children");
    assert!(
        !node_caps.supports_grouping,
        "Node must not support grouping"
    );
    assert!(
        node_caps.supports_duplication,
        "Node must support duplication"
    );
    assert!(
        node_caps.supports_subtree_removal,
        "Node must support subtree removal"
    );

    // PointLight-specific capabilities.
    let pl_caps = ObjectCapabilities::for_kind(ObjectKind::PointLight);
    assert_eq!(
        pl_caps.transform_caps,
        Some(TransformCapabilities::TranslationOnly)
    );
    assert!(
        !pl_caps.supports_children,
        "PointLight must not support children"
    );
    assert!(
        pl_caps.supports_grouping,
        "PointLight must support grouping"
    );

    // DirectionalLight-specific capabilities.
    let dl_caps = ObjectCapabilities::for_kind(ObjectKind::DirectionalLight);
    assert_eq!(
        dl_caps.transform_caps,
        Some(TransformCapabilities::RigidDirectionOnly)
    );
    assert!(
        !dl_caps.supports_children,
        "DirectionalLight must not support children"
    );
    assert!(
        dl_caps.supports_grouping,
        "DirectionalLight must support grouping"
    );

    // SpotLight-specific capabilities.
    let sl_caps = ObjectCapabilities::for_kind(ObjectKind::SpotLight);
    assert_eq!(
        sl_caps.transform_caps,
        Some(TransformCapabilities::RigidWithPosition)
    );
    assert!(
        !sl_caps.supports_children,
        "SpotLight must not support children"
    );
    assert!(sl_caps.supports_grouping, "SpotLight must support grouping");
}

// ── ObjectTransform exhaustive for four kinds ──────────────────────────

#[test]
fn object_transform_exhaustive_for_four_kinds() {
    let local = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    let position = glam::Vec3::new(4.0, 5.0, 6.0);
    let direction = glam::Vec3::new(0.0, -1.0, 0.0);
    let expected_dir = direction.normalize();

    // Node transform preserves the full local matrix.
    let node_t = ObjectTransform::canonical_for_kind(ObjectKind::Node, local, position, direction);
    assert!(
        matches!(node_t, ObjectTransform::Node(m) if m == local),
        "Node transform must be full local matrix"
    );

    // PointLight transform uses position only.
    let pl_t =
        ObjectTransform::canonical_for_kind(ObjectKind::PointLight, local, position, direction);
    assert!(
        matches!(pl_t, ObjectTransform::PointLight(p) if p == position),
        "PointLight transform must be position only"
    );

    // DirectionalLight transform uses normalized direction only.
    let dl_t = ObjectTransform::canonical_for_kind(
        ObjectKind::DirectionalLight,
        local,
        position,
        direction,
    );
    assert!(
        matches!(dl_t, ObjectTransform::DirectionalLight(d) if d == expected_dir),
        "DirectionalLight transform must be normalized direction only"
    );

    // SpotLight transform uses position + normalized direction.
    let sl_t =
        ObjectTransform::canonical_for_kind(ObjectKind::SpotLight, local, position, direction);
    assert!(
        matches!(sl_t, ObjectTransform::SpotLight { position: p, direction: d }
            if p == position && d == expected_dir),
        "SpotLight transform must be position + normalized direction"
    );
}
