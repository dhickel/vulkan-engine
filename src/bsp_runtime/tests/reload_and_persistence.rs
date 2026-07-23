//! Integration tests: save/load/reload, override reconciliation, identity
//! ambiguity, and source-link persistence.

use bsp_runtime::{
    coordinator::BspCoordinator,
    source_link::{
        reconcile_overrides, BspOverrideLayer, BspSourceLink, BspSourceReference, EntityOverride,
    },
};

use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;

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
    let source = BspSourceReference {
        asset_id: "maps/test_map".to_string(),
        content_hash: "sha256:abcdef1234567890".to_string(),
        compiler_provenance: None,
        import_settings: None,
        entity_identity_map: vec![],
    };

    let link = BspSourceLink {
        bsp_source: source,
        bsp_overrides: BspOverrideLayer {
            entity_overrides: vec![EntityOverride {
                stable_handle: "uuid-001".to_string(),
                light_intensity: Some(400.0),
                light_color: Some([1.0, 0.5, 0.5]),
                model_override: None,
            }],
            light_overrides: vec![],
        },
    };

    let json = serde_json::to_string_pretty(&link).unwrap();
    assert!(json.contains("maps/test_map"));
    assert!(json.contains("uuid-001"));
    assert!(json.contains("400.0"));

    let deserialized: BspSourceLink = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.bsp_source.asset_id, "maps/test_map");
    assert_eq!(deserialized.bsp_overrides.entity_overrides.len(), 1);
}

#[test]
fn empty_override_layer_is_serializable() {
    let source = BspSourceReference {
        asset_id: "maps/empty".to_string(),
        content_hash: "sha256:0000".to_string(),
        compiler_provenance: None,
        import_settings: None,
        entity_identity_map: vec![],
    };
    let link = BspSourceLink {
        bsp_source: source,
        bsp_overrides: BspOverrideLayer::default(),
    };

    let json = serde_json::to_string(&link).unwrap();
    let deserialized: BspSourceLink = serde_json::from_str(&json).unwrap();
    assert!(deserialized.bsp_overrides.entity_overrides.is_empty());
    assert!(deserialized.bsp_overrides.light_overrides.is_empty());
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
            light_intensity: Some(300.0),
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
            light_intensity: Some(300.0),
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
    assert_eq!(link["bsp_source"]["asset_id"], "maps/test");
    assert_eq!(link["bsp_source"]["import_settings"]["scale"], 0.5);
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
