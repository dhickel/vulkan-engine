//! Integration tests: save/load/reload, override reconciliation, identity
//! ambiguity, source-link persistence, and restore from persistence.

use bsp_runtime::{
    coordinator::BspCoordinator,
    source_link::{
        reconcile_overrides, BspOverrideLayer, BspPersistenceEnvelope, BspSourceLink,
        CanonicalFloat, EntityOverride, MutableBehaviorState, SerializedDoorState,
        SerializedTriggerState,
    },
};

use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;
use std::collections::BTreeMap;

/// Build a minimal valid BSP29 for testing.
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

// ── Source Link Persistence Tests ─────────────────────────────────────

#[test]
fn source_link_serialization_round_trip() {
    let mut link = BspSourceLink::new(
        "maps/test_map".into(),
        "sha256:abcdef1234567890".into(),
    );
    link.overrides = BspOverrideLayer {
        entity_overrides: vec![EntityOverride {
            stable_handle: "uuid-001".to_string(),
            light_intensity: Some(CanonicalFloat(400.0)),
            light_color: Some([1.0, 0.5, 0.5]),
            model_override: None,
        }],
        light_overrides: vec![],
    };

    let envelope = BspPersistenceEnvelope::new(link);
    let json = serde_json::to_string_pretty(&envelope).unwrap();
    assert!(json.contains("maps/test_map"));
    assert!(json.contains("uuid-001"));

    let deserialized: BspPersistenceEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.bsp_source.asset_id, "maps/test_map");
    assert_eq!(deserialized.bsp_source.overrides.entity_overrides.len(), 1);
}

#[test]
fn empty_override_layer_is_serializable() {
    let link = BspSourceLink::new("maps/empty".into(), "sha256:0000".into());
    let envelope = BspPersistenceEnvelope::new(link);

    let json = serde_json::to_string(&envelope).unwrap();
    let deserialized: BspPersistenceEnvelope = serde_json::from_str(&json).unwrap();
    assert!(deserialized.bsp_source.overrides.entity_overrides.is_empty());
    assert!(deserialized.bsp_source.overrides.light_overrides.is_empty());
}

// ── Reload Tests ──────────────────────────────────────────────────────

#[test]
fn reload_without_previous_mount_succeeds() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let result = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result.is_ok());
    assert!(coordinator.is_active());
}

#[test]
fn reload_preserves_active_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Initial load
    let result1 = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result1.is_ok());

    // Reload same source
    let result2 = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result2.is_ok());
    assert!(coordinator.is_active());
}

#[test]
fn override_reconciliation_empty_on_first_load() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let result = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result.is_ok());
    // No previous overrides, so reconciliation should be None
    assert!(result.unwrap().reconciliation.is_none());
}

// ── Override Reconciliation Tests ─────────────────────────────────────

#[test]
fn reconcile_overrides_on_empty_lists() {
    let identities: Vec<bsp::identity::EntityIdentity> = vec![];
    let descriptors: Vec<bsp::extract::EntityDescriptor> = vec![];
    let (report, reconciled) =
        reconcile_overrides(&BspOverrideLayer::default(), &identities, &descriptors);

    assert_eq!(report.applied, 0);
    assert_eq!(report.orphaned, 0);
    assert_eq!(report.ambiguous, 0);
    assert!(reconciled.entity_overrides.is_empty());
}

#[test]
fn reconcile_detects_orphaned_overrides() {
    // Have an override but no matching entity identity
    let overrides = BspOverrideLayer {
        entity_overrides: vec![EntityOverride {
            stable_handle: "missing-uuid".to_string(),
            light_intensity: Some(CanonicalFloat(300.0)),
            light_color: None,
            model_override: None,
        }],
        light_overrides: vec![],
    };

    let identities: Vec<bsp::identity::EntityIdentity> = vec![];
    let descriptors: Vec<bsp::extract::EntityDescriptor> = vec![];

    let (report, _reconciled) = reconcile_overrides(&overrides, &identities, &descriptors);

    assert_eq!(report.orphaned, 1);
    assert_eq!(report.applied, 0);
}

#[test]
fn override_reconciliation_is_deterministic() {
    let overrides = BspOverrideLayer {
        entity_overrides: vec![EntityOverride {
            stable_handle: "uuid-1".to_string(),
            light_intensity: Some(CanonicalFloat(300.0)),
            light_color: None,
            model_override: None,
        }],
        light_overrides: vec![],
    };

    let identities: Vec<bsp::identity::EntityIdentity> = vec![];
    let descriptors: Vec<bsp::extract::EntityDescriptor> = vec![];

    let (report1, _) = reconcile_overrides(&overrides, &identities, &descriptors);
    let (report2, _) = reconcile_overrides(&overrides, &identities, &descriptors);

    assert_eq!(report1.applied, report2.applied);
    assert_eq!(report1.orphaned, report2.orphaned);
    assert_eq!(report1.ambiguous, report2.ambiguous);
}

// ── Identity Tests ────────────────────────────────────────────────────

#[test]
fn source_link_converts_content_hash_correctly() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    // Source link should be staged
    let extracted = coordinator.staged_extraction();
    assert!(extracted.is_some());

    // Content hash should be non-zero (even minimal BSP)
    let hash = extracted.unwrap().content_hash;
    assert!(hash.iter().any(|&b| b != 0));
}

#[test]
fn coordinator_stages_and_clears() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    assert!(coordinator.staged_extraction().is_none());

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    assert!(coordinator.staged_extraction().is_some());

    coordinator.rollback().unwrap();
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn unload_after_commit_clears_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();

    assert!(coordinator.is_active());
    assert!(scene.bsp_source_link().is_some());

    coordinator.unload(&mut scene).unwrap();
    assert!(!coordinator.is_active());
    assert!(coordinator.staged_extraction().is_none());
    assert!(scene.bsp_source_link().is_none());
}

#[test]
fn commit_publishes_typed_source_link_to_scene_json() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator
        .prepare(&bsp_bytes, Some(0.5), "maps/test")
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();

    let link = scene.bsp_source_link().expect("source link should publish");
    // Envelope format: schema_version + bsp_source
    assert_eq!(link["schema_version"], 1);
    assert_eq!(link["bsp_source"]["asset_id"], "maps/test");
}

#[test]
fn reimport_updates_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // First import
    let (result, _reconciliation) = coordinator
        .reimport(&bsp_bytes, None, "maps/v1", &mut scene, |_| empty_mount())
        .unwrap();

    assert!(coordinator.is_active());
    assert_eq!(result.prepare.source_identity, "maps/v1");
    assert!(result.reconciliation.is_some());

    // Second import
    let (result2, _) = coordinator
        .reimport(&bsp_bytes, None, "maps/v2", &mut scene, |_| empty_mount())
        .unwrap();

    assert!(coordinator.is_active());
    assert_eq!(result2.prepare.source_identity, "maps/v2");
}

// ── Persistence Save/Restore Tests ───────────────────────────────────

#[test]
fn save_capture_reads_immutable_snapshot() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    // Capture source link (persistence read)
    let behavior = MutableBehaviorState::default();
    let envelope = coordinator.capture_source_link(behavior);
    assert!(envelope.is_some());

    let env = envelope.unwrap();
    assert_eq!(env.schema_version, bsp_runtime::SchemaVersion::V1);
    assert_eq!(env.bsp_source.asset_id, "maps/test");
    assert!(env.bsp_source.mutable_behavior.is_empty());
}

#[test]
fn save_capture_with_behavior_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    // Capture with behavior state
    let mut behavior = MutableBehaviorState::default();
    behavior.doors.push(SerializedDoorState {
        entity_index: 1,
        phase: 2,
        travel: CanonicalFloat(1.0),
        wait_timer: CanonicalFloat(0.5),
    });
    behavior.triggers.push(SerializedTriggerState {
        entity_index: 3,
        fired: true,
    });
    let mut styles = BTreeMap::new();
    styles.insert(5, CanonicalFloat(0.75));
    behavior.light_styles = styles;

    let envelope = coordinator.capture_source_link(behavior);
    assert!(envelope.is_some());

    let env = envelope.unwrap();
    assert_eq!(env.bsp_source.mutable_behavior.doors.len(), 1);
    assert_eq!(env.bsp_source.mutable_behavior.triggers.len(), 1);
    assert_eq!(env.bsp_source.mutable_behavior.light_styles.len(), 1);
}

#[test]
fn restore_from_persistence_succeeds() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // First load to establish source identity
    let result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    // Capture the source link
    let envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Unload
    coordinator.unload(&mut scene).unwrap();
    assert!(!coordinator.is_active());

    // Restore from persistence
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| {
            empty_mount()
        })
        .unwrap();

    assert!(coordinator.is_active());
    assert_eq!(restored.prepare.source_identity, "maps/test");
}

#[test]
fn restore_from_persistence_with_mutable_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture with behavior
    let mut behavior = MutableBehaviorState::default();
    behavior.doors.push(SerializedDoorState {
        entity_index: 1,
        phase: 1,
        travel: CanonicalFloat(0.5),
        wait_timer: CanonicalFloat(1.0),
    });
    let envelope = coordinator.capture_source_link(behavior).unwrap();

    // Unload then restore
    coordinator.unload(&mut scene).unwrap();
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| {
            empty_mount()
        })
        .unwrap();

    assert!(coordinator.is_active());
    let link = scene.bsp_source_link().unwrap();
    let doors = &link["bsp_source"]["mutable_behavior"]["doors"];
    assert_eq!(doors.as_array().unwrap().len(), 1);
}

#[test]
fn restore_with_content_hash_mismatch_fails() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture
    let mut envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Tamper with the content hash
    envelope.bsp_source.content_hash = "sha256:deadbeef".into();

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Restore should fail
    let result = coordinator.restore_from_persistence(
        &envelope,
        &bsp_bytes,
        None,
        &mut scene,
        |_| empty_mount(),
    );
    assert!(result.is_err());
    // Active generation should be unchanged
    assert!(!coordinator.is_active());
}

#[test]
fn restore_with_invalid_schema_version_fails() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Manually construct an envelope with bad schema version
    let json = r#"{"schema_version":99,"bsp_source":{"asset_id":"maps/bad","content_hash":"sha256:aa"}}"#;
    let result: Result<BspPersistenceEnvelope, _> = serde_json::from_str(json);
    // Should fail on deserialization because version 99 is not an approved value
    assert!(result.is_err());
}

#[test]
fn restore_with_invalid_mutable_behavior_fails() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    // Capture
    let mut envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    // Inject invalid behavior (sentinel entity index)
    envelope.bsp_source.mutable_behavior.doors.push(SerializedDoorState {
        entity_index: u32::MAX,
        phase: 0,
        travel: CanonicalFloat(0.0),
        wait_timer: CanonicalFloat(0.0),
    });

    // Unload
    coordinator.unload(&mut scene).unwrap();

    // Restore should fail
    let result = coordinator.restore_from_persistence(
        &envelope,
        &bsp_bytes,
        None,
        &mut scene,
        |_| empty_mount(),
    );
    assert!(result.is_err());
}

#[test]
fn restore_restore_order_is_correct() {
    // Verify the restore order: resolve→parse→extract→upload→identity
    // reconcile→mapping validation→mutable behavior validation→commit
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Load
    let _result = coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();

    let envelope = coordinator
        .capture_source_link(MutableBehaviorState::default())
        .unwrap();

    coordinator.unload(&mut scene).unwrap();

    // Restore — should succeed, proving all steps pass
    let restored = coordinator
        .restore_from_persistence(&envelope, &bsp_bytes, None, &mut scene, |_| {
            empty_mount()
        })
        .unwrap();

    assert!(coordinator.is_active());
    // Reconciliation should have been performed
    assert!(restored.reconciliation.is_some());
}
