//! Integration tests: prepare failure, validation failure, cancellation,
//! stale generation, bridge failure, rollback, candidate lifecycle,
//! renderer lease, and failure injection (Phase 05).

use bsp_runtime::{
    bridge::{
        AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe, LightEntityRecipe,
        WorldCollisionRecipe,
    },
    coordinator::BspCoordinator,
    error::BspRuntimeError,
};

use renderer::api::bsp::PreparedBspMount;
use renderer::api::{PointLight, Scene};

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

fn light_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&29u32.to_le_bytes());

    let mut current_offset: u32 = 124;

    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"light\" \"origin\" \"0 0 64\" \"light\" \"200\"}\0";
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

fn test_point_light() -> PointLight {
    PointLight {
        position: glam::Vec3::ZERO,
        color: glam::Vec3::ONE,
        intensity: 1.0,
        range: 1.0,
    }
}

// ── Test Bridges ──────────────────────────────────────────────────────

struct NoopBridge {
    name: String,
    prepare_called: bool,
}

impl NoopBridge {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            prepare_called: false,
        }
    }
}

impl AppBridge for NoopBridge {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        self.prepare_called = true;
        Ok(BridgeToken::new(vec![1, 2, 3]))
    }

    fn validate(&self, _token: &BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn commit(&mut self, _token: BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {}
}

struct FailingPrepareBridge;

impl AppBridge for FailingPrepareBridge {
    fn name(&self) -> &str {
        "failing-prepare"
    }

    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        Err("intentional prepare failure".to_string())
    }

    fn validate(&self, _token: &BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn commit(&mut self, _token: BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {}
}

struct FailingValidateBridge;

impl AppBridge for FailingValidateBridge {
    fn name(&self) -> &str {
        "failing-validate"
    }

    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        Ok(BridgeToken::new(vec![]))
    }

    fn validate(&self, _token: &BridgeToken) -> Result<(), String> {
        Err("intentional validation failure".to_string())
    }

    fn commit(&mut self, _token: BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {}
}

struct PanicCommitBridge;

impl AppBridge for PanicCommitBridge {
    fn name(&self) -> &str {
        "panic-commit"
    }

    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        Ok(BridgeToken::new(vec![9]))
    }

    fn validate(&self, _token: &BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn commit(&mut self, _token: BridgeToken) -> Result<(), String> {
        panic!("intentional commit panic")
    }

    fn rollback(&mut self, _token: BridgeToken) {}
}

struct PanicRollbackBridge;

impl AppBridge for PanicRollbackBridge {
    fn name(&self) -> &str {
        "panic-rollback"
    }

    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        Ok(BridgeToken::new(vec![7]))
    }

    fn validate(&self, _token: &BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn commit(&mut self, _token: BridgeToken) -> Result<(), String> {
        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {
        panic!("intentional rollback panic")
    }
}

// ── Core Transaction Tests ────────────────────────────────────────────

#[test]
fn prepare_succeeds_on_valid_bsp() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let result = coordinator.prepare(&bsp_bytes, None, "maps/test");
    assert!(result.is_ok());
    let prepare = result.unwrap();
    assert_eq!(prepare.source_identity, "maps/test");
    assert!(!prepare.has_pvs);
}

#[test]
fn prepare_fails_on_invalid_bsp() {
    let bsp_bytes = b"not a bsp file";
    let mut coordinator = BspCoordinator::new();
    let result = coordinator.prepare(bsp_bytes, None, "maps/bad");
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::SourceUnavailable { .. } => {}
        e => panic!("expected SourceUnavailable, got {:?}", e),
    }
}

#[test]
fn validate_succeeds_after_prepare() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    assert!(coordinator.validate(prepare.token).is_ok());
}

#[test]
fn validate_fails_without_prepare() {
    let mut coordinator = BspCoordinator::new();
    let token = bsp_runtime::BspGenerationToken { generation: 0 };
    let result = coordinator.validate(token);
    assert!(result.is_err());
}

#[test]
fn stale_generation_rejected() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();

    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    let result = coordinator.validate(prepare1.token);
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::StaleGeneration { .. } => {}
        e => panic!("expected StaleGeneration, got {:?}", e),
    }
}

#[test]
fn commit_fails_with_stale_generation() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();

    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    coordinator.validate(prepare2.token).unwrap();

    let mount = empty_mount();
    let result = coordinator.commit_with_mount(prepare1.token, &mut scene, mount);
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::StaleGeneration { .. } => {}
        e => panic!("expected StaleGeneration, got {:?}", e),
    }
}

#[test]
fn bridge_prepare_failure_triggers_rollback() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("failing", Box::new(FailingPrepareBridge));

    let result = coordinator.prepare(&bsp_bytes, None, "maps/test");
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::BridgeFailure { phase, .. } => {
            assert!(matches!(phase, bsp_runtime::error::BridgePhase::Prepare));
        }
        e => panic!("expected BridgeFailure, got {:?}", e),
    }

    assert!(!coordinator.is_poisoned());
    assert!(coordinator.staged_extraction().is_none());

    let result2 = coordinator.prepare(&bsp_bytes, None, "maps/test");
    assert!(result2.is_err());
}

#[test]
fn bridge_validate_failure_triggers_rollback() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("failing", Box::new(FailingValidateBridge));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    let result = coordinator.validate(prepare.token);
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::BridgeFailure { phase, .. } => {
            assert!(matches!(phase, bsp_runtime::error::BridgePhase::Validate));
        }
        e => panic!("expected BridgeFailure, got {:?}", e),
    }
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn rollback_is_idempotent() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    assert!(coordinator.rollback().is_ok());
    assert!(coordinator.rollback().is_ok());
    assert!(coordinator.rollback().is_ok());

    let result = coordinator.prepare(&bsp_bytes, None, "maps/test2");
    assert!(result.is_ok());
}

#[test]
fn commit_and_unload_cycle() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();

    let mount = empty_mount();
    let commit = coordinator.commit_with_mount(prepare.token, &mut scene, mount);
    assert!(commit.is_ok());
    assert!(coordinator.is_active());

    let result = coordinator.unload(&mut scene);
    assert!(result.is_ok());
    assert!(!coordinator.is_active());
}

#[test]
fn prepare_with_bridge_and_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("noop", Box::new(NoopBridge::new("noop")));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();

    let mount = empty_mount();
    let commit = coordinator.commit_with_mount(prepare.token, &mut scene, mount);
    assert!(commit.is_ok());
    assert!(commit.unwrap().bridge_count == 1);
    assert!(coordinator.is_active());
}

#[test]
fn double_prepare_cancels_first() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let _prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    assert!(coordinator.validate(prepare2.token).is_ok());
}

#[test]
fn coordinator_poisoned_state_detected() {
    let coordinator = BspCoordinator::new();
    assert!(!coordinator.is_poisoned());
}

#[test]
fn bridge_commit_panic_poisons_coordinator_without_scene_publication() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.register_bridge("panic", Box::new(PanicCommitBridge));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();
    let result = coordinator.commit_with_mount(prepare.token, &mut scene, empty_mount());

    assert!(matches!(result, Err(BspRuntimeError::CoordinatorPoisoned)));
    assert!(coordinator.is_poisoned());
    assert!(scene.bsp_source_link().is_none());
}

#[test]
fn bridge_rollback_panic_poisons_coordinator() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("panic", Box::new(PanicRollbackBridge));

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    let result = coordinator.rollback();

    assert!(matches!(
        result,
        Err(BspRuntimeError::RollbackFailure { .. })
    ));
    assert!(coordinator.is_poisoned());
}

#[test]
fn commit_without_validate_fails() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    let result = coordinator.commit_with_mount(prepare.token, &mut scene, empty_mount());
    assert!(result.is_err());

    coordinator.validate(prepare.token).unwrap();
    assert!(coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .is_ok());
}

// ── Phase 05: Candidate Lifecycle Tests ─────────────────────────────

#[test]
fn candidate_is_staged_on_prepare() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    assert!(coordinator.staged_extraction().is_none());

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    assert!(coordinator.staged_extraction().is_some());
    assert!(coordinator.staged_entity_descriptors().is_some());
}

#[test]
fn renderer_lease_transitions_not_started_to_ready() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    // Set mount ready directly (synchronous path)
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();

    // Validate then commit
    coordinator.validate(prepare.token).unwrap();
    let mut scene = Scene::new();
    let result = coordinator.commit(prepare.token, &mut scene);
    assert!(result.is_ok());
}

#[test]
fn commit_requires_renderer_mount_ready() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();

    // Commit without setting mount ready should fail
    let result = coordinator.commit(prepare.token, &mut scene);
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::BridgeFailure { message, .. } => {
            assert!(message.contains("renderer mount not ready"));
        }
        e => panic!("expected BridgeFailure about mount, got {:?}", e),
    }
}

#[test]
fn renderer_lease_idempotent() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    // Setting mount ready multiple times should be idempotent
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();

    coordinator.validate(prepare.token).unwrap();
    let mut scene = Scene::new();
    assert!(coordinator.commit(prepare.token, &mut scene).is_ok());
}

#[test]
fn light_candidate_requires_scene_publication_validation() {
    let bsp_bytes = light_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/lit").unwrap();
    assert_eq!(prepare.light_count, 1);
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();

    let result = coordinator.validate(prepare.token);
    assert!(matches!(
        result,
        Err(BspRuntimeError::BridgeFailure {
            phase: bsp_runtime::error::BridgePhase::Validate,
            ..
        })
    ));
    assert!(coordinator.staged_extraction().is_none());

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/lit").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator
        .validate_for_scene(prepare.token, &mut scene)
        .unwrap();
    let commit = coordinator.commit(prepare.token, &mut scene).unwrap();
    assert_eq!(commit.light_count, 1);
}

#[test]
fn validate_for_scene_rejects_light_capacity_before_commit() {
    let bsp_bytes = light_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let scene_light = test_point_light();
    let mut scene = Scene::new();

    let mut filled = 0;
    while scene.create_point_light(scene_light).is_ok() {
        filled += 1;
        assert!(filled < 128, "point-light cap should be finite");
    }
    assert!(filled > 0);

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/lit").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();

    let result = coordinator.validate_for_scene(prepare.token, &mut scene);
    assert!(matches!(
        result,
        Err(BspRuntimeError::BridgeFailure {
            phase: bsp_runtime::error::BridgePhase::Validate,
            ..
        })
    ));
    assert!(coordinator.staged_extraction().is_none());
    assert!(!coordinator.is_poisoned());
}

#[test]
fn teardown_cleans_up_candidate_and_active_state() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Prepare and commit
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();

    assert!(coordinator.is_active());

    // Teardown
    coordinator.teardown(&mut scene);
    assert!(!coordinator.is_active());
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn new_prepare_cancels_previous_candidate() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();

    // Set mount ready on first candidate
    coordinator
        .set_renderer_mount_ready(prepare1.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare1.token).unwrap();

    // Second prepare cancels first
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    assert!(coordinator.staged_extraction().is_some());

    // First token is now stale
    let mut scene = Scene::new();
    let result = coordinator.commit(prepare1.token, &mut scene);
    assert!(result.is_err());
}

#[test]
fn poisioned_coordinator_rejects_all_operations() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("panic", Box::new(PanicCommitBridge));

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();

    let mut scene = Scene::new();
    let _ = coordinator.commit_with_mount(prepare.token, &mut scene, empty_mount());
    assert!(coordinator.is_poisoned());

    // All operations should be rejected
    assert!(matches!(
        coordinator.prepare(&bsp_bytes, None, "maps/test"),
        Err(BspRuntimeError::CoordinatorPoisoned)
    ));
    assert!(matches!(
        coordinator.unload(&mut scene),
        Err(BspRuntimeError::CoordinatorPoisoned)
    ));
    assert!(matches!(
        coordinator.rollback(),
        Err(BspRuntimeError::CoordinatorPoisoned)
    ));
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Lifecycle Fault Injection — Rapid Replacement
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn rapid_replacement_three_times_no_leak() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    for i in 0..3 {
        let result = coordinator.reload(
            &bsp_bytes,
            None,
            &format!("maps/test{i}"),
            &mut scene,
            |_| empty_mount(),
        );
        assert!(result.is_ok(), "reload {i} failed");
        assert!(coordinator.is_active());
    }
    // Should still be valid after three rapid replacements
    let link = scene.bsp_source_link().unwrap();
    assert_eq!(link["bsp_source"]["asset_id"], "maps/test2");
}

#[test]
fn unload_then_reload_works() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator
        .reload(&bsp_bytes, None, "maps/a", &mut scene, |_| empty_mount())
        .unwrap();
    coordinator.unload(&mut scene).unwrap();

    coordinator
        .reload(&bsp_bytes, None, "maps/b", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
}

#[test]
fn stale_generation_mid_pipeline_rejected() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    // prepare1 token is stale — validate should fail
    let result = coordinator.validate(prepare1.token);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        BspRuntimeError::StaleGeneration { .. }
    ));
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Lifecycle Fault Injection — Readiness Failure
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn commit_without_set_renderer_ready_fails() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();

    // Commit without setting renderer mount ready
    let result = coordinator.commit(prepare.token, &mut scene);
    assert!(result.is_err());
    assert!(!coordinator.is_poisoned());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Lifecycle Fault Injection — Hidden Install Interruption
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prepare_then_rollback_then_prepare_again() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.rollback().unwrap();

    // After rollback, prepare again
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    assert!(coordinator.staged_extraction().is_some());

    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare2.token).unwrap();
    let mut scene = Scene::new();
    assert!(coordinator.commit(prepare2.token, &mut scene).is_ok());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Lifecycle Fault Injection — Shared Resources
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn shared_bridge_across_multiple_prepares() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator.register_bridge("shared", Box::new(NoopBridge::new("shared")));

    // First prepare + commit
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare1.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare1.token).unwrap();
    coordinator.commit(prepare1.token, &mut scene).unwrap();

    // Reload with same bridge
    let result = coordinator.reload(&bsp_bytes, None, "maps/test2", &mut scene, |_| {
        empty_mount()
    });
    assert!(result.is_ok());
    assert!(coordinator.is_active());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Lifecycle Fault Injection — Shutdown Cleanup
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn teardown_on_clean_coordinator_is_idempotent() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.teardown(&mut scene);
    coordinator.teardown(&mut scene); // second call should not panic
    assert!(!coordinator.is_active());
}

#[test]
fn teardown_on_active_coordinator_cleans_up() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    coordinator
        .reload(&bsp_bytes, None, "maps/test", &mut scene, |_| empty_mount())
        .unwrap();
    assert!(coordinator.is_active());

    coordinator.teardown(&mut scene);
    assert!(!coordinator.is_active());
    assert!(scene.bsp_source_link().is_none());
}

#[test]
fn teardown_on_poisoned_coordinator_does_not_panic() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("panic", Box::new(PanicCommitBridge));
    let mut scene = Scene::new();

    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator.validate(prepare.token).unwrap();
    let _ = coordinator.commit_with_mount(prepare.token, &mut scene, empty_mount());
    assert!(coordinator.is_poisoned());

    // Teardown should NOT check poisoned flag
    coordinator.teardown(&mut scene);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Frames-in-Flight Retirement
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mount_commit_and_immediate_reload_retires_old_mount() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Commit a mount
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    coordinator.commit(prepare.token, &mut scene).unwrap();

    // Immediately reload — old mount should be retired
    let result = coordinator.reload(&bsp_bytes, None, "maps/v2", &mut scene, |_| empty_mount());
    assert!(result.is_ok());
    let link = scene.bsp_source_link().unwrap();
    assert_eq!(link["bsp_source"]["asset_id"], "maps/v2");
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Unsubmitted Cancellation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cancel_unsubmitted_candidate_via_new_prepare() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    // Prepare a candidate but never submit
    let _prepare1 = coordinator
        .prepare(&bsp_bytes, None, "maps/abandoned")
        .unwrap();

    // New prepare cancels the unsubmitted one
    let prepare2 = coordinator
        .prepare(&bsp_bytes, None, "maps/replacement")
        .unwrap();
    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare2.token).unwrap();

    let mut scene = Scene::new();
    assert!(coordinator.commit(prepare2.token, &mut scene).is_ok());
}
