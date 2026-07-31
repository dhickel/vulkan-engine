//! Persistence integration tests: save capture, restore from persistence,
//! schema stability, rejection cases, and save/reload integration with
//! app-owned bridges and mutable behavior state.

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_runtime::coordinator::BspCoordinator;
use bsp_runtime::{
    CanonicalFloat, MutableBehaviorState, SerializedDoorState, SerializedPlatformState,
    SerializedTriggerState,
};
use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;
use std::collections::BTreeMap;

fn minimal_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());

    let mut current_offset: u32 = 124;
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset = current_offset;
    let entity_size = entity_bytes.len() as u32;
    current_offset += entity_size;

    let plane_offset = current_offset;
    let plane_size = 20u32;
    current_offset += plane_size;

    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size),
        (plane_offset, plane_size),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ];

    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    data.extend_from_slice(entity_bytes);
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());

    data
}

fn empty_mount() -> PreparedBspMount {
    PreparedBspMount::new()
}

// ── Save/Restore Round-Trip with Bridges ─────────────────────────────

#[test]
fn save_and_restore_with_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    // Load
    let result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    // Save: capture source link
    let envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();
    assert_eq!(envelope.schema_version, bsp_runtime::SchemaVersion::V1);

    // Unload
    coordinator.unload(&mut scene).unwrap();
    assert!(!coordinator.is_active());

    // Restore from persistence
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount())
        .unwrap();

    assert!(coordinator.is_active());
    assert_eq!(restored.prepare.source_identity, "maps/test");
}

#[test]
fn save_restore_preserves_mutable_behavior_with_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture with full mutable behavior
    let mut behavior = MutableBehaviorState::default();
    behavior.doors.push(SerializedDoorState {
        entity_index: 10,
        phase: 2, // Open
        travel: CanonicalFloat(1.0),
        wait_timer: CanonicalFloat(3.0),
    });
    behavior.doors.push(SerializedDoorState {
        entity_index: 20,
        phase: 0, // Closed
        travel: CanonicalFloat(0.0),
        wait_timer: CanonicalFloat(0.0),
    });
    let mut styles = BTreeMap::new();
    styles.insert(5, CanonicalFloat(0.75));
    styles.insert(10, CanonicalFloat(0.0));
    behavior.light_styles = styles;
    behavior.triggers.push(SerializedTriggerState {
        entity_index: 30,
        fired: true,
    });

    let envelope = coordinator.capture_source_link(behavior).unwrap();

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Restore
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount())
        .unwrap();

    assert!(coordinator.is_active());

    // Verify mutable behavior was stored in scene source-link
    let link = scene.bsp_source_link().unwrap();
    let doors = &link["bsp_source"]["mutable_behavior"]["doors"];
    assert_eq!(doors.as_array().unwrap().len(), 2);
    let triggers = &link["bsp_source"]["mutable_behavior"]["triggers"];
    assert_eq!(triggers.as_array().unwrap().len(), 1);
}

#[test]
fn restore_fails_on_source_hash_mismatch_with_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture
    let mut envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Tamper with content hash
    envelope.bsp_source.content_hash = "sha256:badbadbad".into();

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Restore should fail
    let result =
        coordinator
            .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount());
    assert!(result.is_err());
    // Active state should be unchanged
    assert!(!coordinator.is_active());
}

#[test]
fn restore_cancelled_proves_active_unchanged() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    // Capture
    let mut envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Tamper
    envelope.bsp_source.content_hash = "sha256:badbadbad".into();

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Failed restore — active state should still be empty
    let result =
        coordinator
            .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount());
    assert!(result.is_err());
    assert!(!coordinator.is_active());

    // But coordinator should still be usable
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/recovery");
    assert!(prepare.is_ok());
}

#[test]
fn restore_validates_mutable_behavior_before_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture with invalid behavior (sentinel entity index)
    let mut behavior = MutableBehaviorState::default();
    behavior.doors.push(SerializedDoorState {
        entity_index: u32::MAX,
        phase: 0,
        travel: CanonicalFloat(0.0),
        wait_timer: CanonicalFloat(0.0),
    });
    let envelope = coordinator.capture_source_link(behavior).unwrap();

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Restore should fail (invalid mutable behavior)
    let result =
        coordinator
            .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount());
    assert!(result.is_err());
}

#[test]
fn schema_version_round_trip_is_stable() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    // Load and save three times to prove schema stability
    for i in 0..3 {
        if i == 0 {
            let _result = coordinator
                .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
                .unwrap();
        } else {
            let envelope = coordinator
                .capture_source_link(MutableBehaviorState::default())
                .unwrap();
            coordinator.unload(&mut scene).unwrap();
            let _restored = coordinator
                .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| {
                    empty_mount()
                })
                .unwrap();
        }

        let link = scene.bsp_source_link().unwrap();
        assert_eq!(
            link["schema_version"], 1,
            "schema version stable at cycle {i}"
        );
        assert!(!link["bsp_source"]["asset_id"].as_str().unwrap().is_empty());
    }
}

#[test]
fn save_capture_excludes_gpu_handles() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture
    let envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    let json = serde_json::to_string(&envelope).unwrap();

    // Verify no GPU handles or transient generation fields appear
    assert!(!json.contains("VkImage"));
    assert!(!json.contains("VkBuffer"));
    assert!(!json.contains("VkDescriptorSet"));
    assert!(!json.contains("scene_node_id"));
    assert!(!json.contains("material_handle"));
    assert!(!json.contains("mesh_handle"));
    assert!(!json.contains("light_id"));
    assert!(!json.contains("cache_slot"));
    assert!(!json.contains("generated_geometry"));

    // Verify source-linked reconstruction data IS present
    assert!(json.contains("asset_id"));
    assert!(json.contains("content_hash"));
    assert!(json.contains("schema_version"));
}

#[test]
fn restore_builds_hidden_candidate_before_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    let envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Don't unload — the active world should remain visible during restore
    // Restore builds a hidden candidate, validates, then commits
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount())
        .unwrap();

    // Active world should be the restored one
    assert!(coordinator.is_active());
    assert_eq!(restored.prepare.source_identity, "maps/test");
}

#[test]
fn mutable_behavior_platform_state_round_trips() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture with platform state
    let mut behavior = MutableBehaviorState::default();
    behavior.platforms.push(SerializedPlatformState {
        entity_index: 42,
        phase: 2, // High
        travel: CanonicalFloat(1.0),
        wait_timer: CanonicalFloat(1.5),
    });
    let envelope = coordinator.capture_source_link(behavior).unwrap();

    // Verify platform state is in the serialized payload
    let json = serde_json::to_string(&envelope).unwrap();
    assert!(json.contains("\"platforms\""));
    assert!(json.contains("42"));

    // Unload and restore
    coordinator.unload(&mut scene).unwrap();
    let _restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| empty_mount())
        .unwrap();

    let link = scene.bsp_source_link().unwrap();
    let platforms = &link["bsp_source"]["mutable_behavior"]["platforms"];
    assert_eq!(platforms.as_array().unwrap().len(), 1);
}
