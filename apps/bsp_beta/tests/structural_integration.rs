//! Structural integration tests for the BSP beta application.
//!
//! Verifies: coordinator-based preparation, bridge registration,
//! atomic commit, source-link lifecycle, and reload/reimport
//! through the coordinator with app-owned bridges.
//!
//! Phase 05: Bridge activation produces active receipts stored in the
//! active mount. Candidate-to-active transfer tested; coordinator
//! replacement/unload cleanup is Phase 06.

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_runtime::coordinator::BspCoordinator;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;

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
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
        (0, 0), (0, 0), (0, 0),
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

// ── Coordinator + App Bridge Integration ─────────────────────────────

#[test]
fn coordinator_with_physics_and_runtime_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let prepare = coordinator
        .prepare(&bsp_bytes, Some(0.0254), "maps/test")
        .unwrap();

    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();

    coordinator.validate(prepare.token).unwrap();

    let commit = coordinator.commit(prepare.token, &mut scene).unwrap();
    assert_eq!(commit.bridge_count, 2);
    assert!(coordinator.is_active());
    assert!(scene.bsp_source_link().is_some());

    let link = scene.bsp_source_link().unwrap();
    assert_eq!(link["schema_version"], 1);
    assert_eq!(link["bsp_source"]["asset_id"], "maps/test");
}

#[test]
fn reload_preserves_world_with_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let result = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result.is_ok());
    assert!(coordinator.is_active());
}

#[test]
fn reimport_switches_source_atomically_with_bridges() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let (result, _reconciliation) = coordinator
        .reimport(&bsp_bytes, None, "maps/v1", &mut scene, |_| empty_mount())
        .unwrap();
    assert_eq!(result.prepare.source_identity, "maps/v1");
    assert!(coordinator.is_active());

    let (result2, _) = coordinator
        .reimport(&bsp_bytes, None, "maps/v2", &mut scene, |_| empty_mount())
        .unwrap();
    assert_eq!(result2.prepare.source_identity, "maps/v2");
    assert!(coordinator.is_active());

    let link = scene.bsp_source_link().unwrap();
    assert_eq!(link["bsp_source"]["asset_id"], "maps/v2");
}

#[test]
fn unload_removes_bridge_resources() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let result = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result.is_ok());
    assert!(coordinator.is_active());

    coordinator.unload(&mut scene).unwrap();
    assert!(!coordinator.is_active());
    assert!(scene.bsp_source_link().is_none());
}

#[test]
fn coordinator_state_reset_between_cycles() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let result = coordinator.reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount());
    assert!(result.is_ok());

    coordinator.unload(&mut scene).unwrap();

    let result2 = coordinator.reload(&bsp_bytes, None, "maps/test2", &mut scene, |_| {
        empty_mount()
    });
    assert!(result2.is_ok());
    assert_eq!(result2.unwrap().prepare.source_identity, "maps/test2");
}

#[test]
fn prepare_from_world_entrypoint_works() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let load_options = bsp::LoadOptions {
        strict: false,
        source_identity: "maps/test_package".to_string(),
        ..bsp::LoadOptions::default()
    };
    let world = bsp::BspLoader::load(&bsp_bytes, &load_options).unwrap();

    let prepare = coordinator
        .prepare_from_world(world, Some(0.0254), "maps/test_package")
        .unwrap();

    assert_eq!(prepare.source_identity, "maps/test_package");
    assert!(coordinator.staged_extraction().is_some());
}

#[test]
fn candidate_becomes_none_after_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();

    assert!(coordinator.staged_extraction().is_some());

    coordinator.commit(prepare.token, &mut scene).unwrap();

    assert!(coordinator.staged_extraction().is_none());
    assert!(coordinator.is_active());
}

// ── Failure Injection Tests ──────────────────────────────────────────

#[test]
fn rollback_returns_coordinator_to_clean_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    assert!(coordinator.staged_extraction().is_some());

    coordinator.rollback().unwrap();
    assert!(coordinator.staged_extraction().is_none());

    let result = coordinator.prepare(&bsp_bytes, None, "maps/test2");
    assert!(result.is_ok());
}

#[test]
fn cancel_by_new_prepare_works() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    let _prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    assert!(coordinator.staged_extraction().is_some());
    let extracted = coordinator.staged_extraction().unwrap();
    assert_eq!(extracted.entity_descriptors.len(), 1);
}

#[test]
fn validate_after_new_prepare_uses_new_candidate() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));

    let _prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();
    assert!(coordinator.validate(prepare2.token).is_ok());

    let mut scene = Scene::new();
    let commit = coordinator.commit(prepare2.token, &mut scene).unwrap();
    assert_eq!(commit.bridge_count, 1);
}

// ── Phase 05: Receipt Creation and Candidate-to-Active Transfer ─────

#[test]
fn active_mount_contains_bridge_receipts_after_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("physics", Box::new(PhysicsBridge::new()));
    coordinator.register_bridge("runtime", Box::new(RuntimeBridge::new()));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();

    let commit = coordinator.commit(prepare.token, &mut scene).unwrap();
    assert_eq!(commit.bridge_count, 2);
    assert!(coordinator.is_active());

    // Active mount exists; Phase 06 wires coordinator replacement/unload
    // to consume these receipts.
}
