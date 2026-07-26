//! Integration tests: prepare failure, validation failure, cancellation,
//! stale generation, bridge failure, rollback, candidate lifecycle,
//! renderer lease, failure injection, and Phase 05 bridge receipt ownership.
//!
//! Phase 05 adds: activation panic quarantine, rollback panic quarantine,
//! teardown error/panic quarantine, duplicate teardown rejection,
//! B failure preserving active A, and non-Clone compile verification.

use bsp_runtime::{
    bridge::{
        ActiveBridgeState, AppBridge, BehaviorEntityRecipe, EntityCollisionRecipe,
        LightEntityRecipe, PreparedBridgeState, WorldCollisionRecipe,
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

fn light_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let mut current_offset: u32 = 124;
    let entity_bytes =
        b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"light\" \"origin\" \"0 0 64\" \"light\" \"200\"}\0";
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

fn test_point_light() -> PointLight {
    PointLight {
        position: glam::Vec3::ZERO,
        color: glam::Vec3::ONE,
        intensity: 1.0,
        range: 1.0,
    }
}

// ── Phase 05 Recording Bridges ────────────────────────────────────────

/// A prepared state that records what happened (for test assertions).
#[derive(Debug)]
struct RecordingPrepared {
    name: String,
}
impl PreparedBridgeState for RecordingPrepared {
    fn registration_name(&self) -> &str {
        &self.name
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// An active state that records what happened.
#[derive(Debug)]
struct RecordingActive {
    name: String,
}
impl ActiveBridgeState for RecordingActive {
    fn registration_name(&self) -> &str {
        &self.name
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// A bridge that records calls for test assertions.
struct RecordingBridge {
    name: String,
    pub prepare_called: bool,
    pub validate_called: bool,
    pub activate_called: bool,
    pub teardown_called: bool,
    pub rollback_called: bool,
}

impl RecordingBridge {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            prepare_called: false,
            validate_called: false,
            activate_called: false,
            teardown_called: false,
            rollback_called: false,
        }
    }
}

impl AppBridge for RecordingBridge {
    fn name(&self) -> &str {
        &self.name
    }
    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        self.prepare_called = true;
        Ok(Box::new(RecordingPrepared {
            name: self.name.clone(),
        }))
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        self.activate_called = true;
        Box::new(RecordingActive {
            name: self.name.clone(),
        })
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        self.teardown_called = true;
        Ok(())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {
        self.rollback_called = true;
    }
}

/// A bridge that always fails prepare.
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
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        Err("intentional prepare failure".to_string())
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        Box::new(RecordingActive {
            name: "failing-prepare".into(),
        })
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {}
}

/// A bridge that fails validation.
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
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        Ok(Box::new(RecordingPrepared {
            name: "failing-validate".into(),
        }))
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Err("intentional validation failure".to_string())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        Box::new(RecordingActive {
            name: "failing-validate".into(),
        })
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {}
}

/// A bridge that panics during activation.
struct PanicActivateBridge;
impl AppBridge for PanicActivateBridge {
    fn name(&self) -> &str {
        "panic-activate"
    }
    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        Ok(Box::new(RecordingPrepared {
            name: "panic-activate".into(),
        }))
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        panic!("intentional activation panic")
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {}
}

/// A bridge that panics during rollback.
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
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        Ok(Box::new(RecordingPrepared {
            name: "panic-rollback".into(),
        }))
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        Box::new(RecordingActive {
            name: "panic-rollback".into(),
        })
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {
        panic!("intentional rollback panic")
    }
}

/// A bridge that fails teardown.
struct FailingTeardownBridge;
impl AppBridge for FailingTeardownBridge {
    fn name(&self) -> &str {
        "failing-teardown"
    }
    fn prepare(
        &mut self,
        _world: &WorldCollisionRecipe,
        _entities: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        Ok(Box::new(RecordingPrepared {
            name: "failing-teardown".into(),
        }))
    }
    fn validate(&self, _prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        Ok(())
    }
    fn activate(
        &mut self,
        _prepared: &mut dyn PreparedBridgeState,
    ) -> Box<dyn ActiveBridgeState> {
        Box::new(RecordingActive {
            name: "failing-teardown".into(),
        })
    }
    fn teardown(&mut self, _active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        Err("intentional teardown failure".to_string())
    }
    fn rollback(&mut self, _prepared: &mut dyn PreparedBridgeState) {}
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
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
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
    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();
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
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
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
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let commit = coordinator.commit(prepare.token, &mut scene);
    assert!(commit.is_ok());
    assert!(coordinator.is_active());
    assert!(scene.has_bsp_mount());
    let result = coordinator.unload(&mut scene);
    assert!(result.is_ok());
    assert!(!coordinator.is_active());
    assert!(!scene.has_bsp_mount());
}

#[test]
fn prepare_with_bridge_and_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.register_bridge("rec", Box::new(RecordingBridge::new("rec")));
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let commit = coordinator.commit(prepare.token, &mut scene);
    assert!(commit.is_ok());
    assert_eq!(commit.unwrap().bridge_count, 1);
    assert!(coordinator.is_active());
}

#[test]
fn double_prepare_cancels_first() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let _prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();
    assert!(coordinator.validate(prepare2.token).is_ok());
}

#[test]
fn coordinator_poisoned_state_detected() {
    let coordinator = BspCoordinator::new();
    assert!(!coordinator.is_poisoned());
}

#[test]
fn bridge_activate_panic_poisons_coordinator_without_scene_publication() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.register_bridge("panic", Box::new(PanicActivateBridge));
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let result = coordinator.commit(prepare.token, &mut scene);
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
    assert!(result.is_ok());
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    let result2 = coordinator.commit(prepare2.token, &mut scene);
    assert!(result2.is_err());
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
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
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
    let result = coordinator.commit(prepare.token, &mut scene);
    assert!(result.is_err());
    match result.unwrap_err() {
        BspRuntimeError::InvalidCandidateTransition { detail, .. } => {
            assert!(detail.contains("ValidatedForScene"));
        }
        e => panic!("expected InvalidCandidateTransition, got {:?}", e),
    }
}

#[test]
fn duplicate_renderer_ready_lease_is_detached() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    let retirements_before = coordinator.retirement_diagnostics();
    let duplicate = coordinator.set_renderer_mount_ready(prepare.token, empty_mount());
    assert!(matches!(
        duplicate,
        Err(BspRuntimeError::DuplicateReadyLease { .. })
    ));
    assert_eq!(coordinator.retirement_diagnostics(), retirements_before + 1);
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
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
    coordinator.teardown(&mut scene);
    assert!(!coordinator.is_active());
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn new_prepare_cancels_previous_candidate() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare1.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare1.token).unwrap();
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();
    assert!(coordinator.staged_extraction().is_some());
    let mut scene = Scene::new();
    let result = coordinator.commit(prepare1.token, &mut scene);
    assert!(result.is_err());
}

#[test]
fn poisioned_coordinator_rejects_all_operations() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    coordinator.register_bridge("panic", Box::new(PanicActivateBridge));
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let mut scene = Scene::new();
    let _ = coordinator.commit(prepare.token, &mut scene);
    assert!(coordinator.is_poisoned());
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

// ── Phase 05: Bridge Receipt Ownership Tests ─────────────────────────

#[test]
fn active_receipts_are_stored_in_mount_after_commit() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let mut rec = RecordingBridge::new("rec");
    coordinator.register_bridge("rec", Box::new(RecordingBridge::new("rec")));
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    assert!(coordinator.commit(prepare.token, &mut scene).is_ok());
}

#[test]
fn b_failure_preserves_active_a() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Commit active A
    let prepare_a = coordinator.prepare(&bsp_bytes, None, "maps/a").unwrap();
    coordinator
        .commit_with_mount(prepare_a.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
    let link_before = scene.bsp_source_link().unwrap().clone();

    // Register a failing bridge and try to prepare B
    coordinator.register_bridge("failing", Box::new(FailingValidateBridge));
    let prepare_b = coordinator.prepare(&bsp_bytes, None, "maps/b").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare_b.token, empty_mount())
        .unwrap();
    let result = coordinator.validate(prepare_b.token);
    assert!(result.is_err());

    // Active A should still be present
    assert!(coordinator.is_active());
    let link_after = scene.bsp_source_link().unwrap();
    assert_eq!(link_after, &link_before);
}

#[test]
fn bridge_validate_failure_rolls_back_candidate_only() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    let prepare = coordinator
        .prepare(&bsp_bytes, None, "maps/active")
        .unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();

    let link_before = scene.bsp_source_link().unwrap().clone();
    let ret_before = coordinator.retirement_diagnostics();

    coordinator.register_bridge("failing", Box::new(FailingValidateBridge));
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/new").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare2.token, empty_mount())
        .unwrap();

    let result = coordinator.validate(prepare2.token);
    assert!(result.is_err());

    assert!(coordinator.is_active());
    let link_after = scene.bsp_source_link().unwrap();
    assert_eq!(link_after, &link_before);
    assert_eq!(coordinator.retirement_diagnostics(), ret_before + 1);
    assert!(coordinator.staged_extraction().is_none());
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
// Phase 09: Shared Resources
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn shared_bridge_across_multiple_prepares() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.register_bridge("shared", Box::new(RecordingBridge::new("shared")));
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/test1").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare1.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare1.token).unwrap();
    coordinator.commit(prepare1.token, &mut scene).unwrap();
    let result = coordinator.reload(&bsp_bytes, None, "maps/test2", &mut scene, |_| {
        empty_mount()
    });
    assert!(result.is_ok());
    assert!(coordinator.is_active());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Shutdown Cleanup
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn teardown_on_clean_coordinator_is_idempotent() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    coordinator.teardown(&mut scene);
    coordinator.teardown(&mut scene);
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
    coordinator.register_bridge("panic", Box::new(PanicActivateBridge));
    let mut scene = Scene::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    let _ = coordinator.commit(prepare.token, &mut scene);
    assert!(coordinator.is_poisoned());
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
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    coordinator.validate(prepare.token).unwrap();
    coordinator.commit(prepare.token, &mut scene).unwrap();
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
    let _prepare1 = coordinator
        .prepare(&bsp_bytes, None, "maps/abandoned")
        .unwrap();
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

// ═══════════════════════════════════════════════════════════════════════
// Phase 05: Transactional Mount Ownership — Fault Injection
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn stale_renderer_completion_is_rejected_and_does_not_mutate_candidate() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/v2").unwrap();
    let retirements_before = coordinator.retirement_diagnostics();
    let result = coordinator.complete_renderer_upload(prepare1.token, empty_mount());
    assert!(matches!(
        result,
        Err(BspRuntimeError::StaleGeneration { .. })
    ));
    assert!(coordinator.staged_extraction().is_some());
    assert_eq!(coordinator.retirement_diagnostics(), retirements_before + 1);
}

#[test]
fn duplicate_renderer_ready_lease_is_rejected() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .unwrap();
    let retirements_before = coordinator.retirement_diagnostics();
    let result = coordinator.complete_renderer_upload(prepare.token, empty_mount());
    assert!(matches!(
        result,
        Err(BspRuntimeError::DuplicateReadyLease { .. })
    ));
    assert_eq!(coordinator.retirement_diagnostics(), retirements_before + 1);
    coordinator.validate(prepare.token).unwrap();
}

#[test]
fn stale_generation_preserves_active_mount_on_commit_rejection() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    coordinator
        .commit_with_mount(prepare1.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
    let link_before = scene.bsp_source_link().unwrap().clone();
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/v2").unwrap();
    let result = coordinator.commit_with_mount(prepare1.token, &mut scene, empty_mount());
    assert!(result.is_err());
    let link_after = scene.bsp_source_link().unwrap();
    assert_eq!(link_after, &link_before);
}

#[test]
fn unload_retires_active_mount_and_records_retirement() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();
    let retirements_before = coordinator.retirement_diagnostics();
    coordinator.unload(&mut scene).unwrap();
    assert!(!coordinator.is_active());
    assert!(coordinator.retirement_diagnostics() > retirements_before);
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn replacement_commits_retire_old_mount() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    coordinator
        .commit_with_mount(prepare1.token, &mut scene, empty_mount())
        .unwrap();
    let ret_v1 = coordinator.retirement_diagnostics();
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/v2").unwrap();
    coordinator
        .commit_with_mount(prepare2.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.retirement_diagnostics() > ret_v1);
    assert!(coordinator.is_active());
    let prepare3 = coordinator.prepare(&bsp_bytes, None, "maps/v3").unwrap();
    coordinator
        .commit_with_mount(prepare3.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.retirement_diagnostics() > ret_v1 + 1);
}

#[test]
fn rollback_does_not_affect_active_mount() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let prepare = coordinator
        .prepare(&bsp_bytes, None, "maps/active")
        .unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
    let link_before = scene.bsp_source_link().unwrap().clone();
    let _prepare2 = coordinator
        .prepare(&bsp_bytes, None, "maps/candidate")
        .unwrap();
    coordinator.rollback().unwrap();
    assert!(coordinator.is_active());
    let link_after = scene.bsp_source_link().unwrap();
    assert_eq!(link_after, &link_before);
}

#[test]
fn teardown_retires_active_and_counts() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    coordinator
        .commit_with_mount(prepare.token, &mut scene, empty_mount())
        .unwrap();
    let ret_before = coordinator.retirement_diagnostics();
    coordinator.teardown(&mut scene);
    assert!(!coordinator.is_active());
    assert!(coordinator.retirement_diagnostics() > ret_before);
}

#[test]
fn candidate_is_cleared_after_rollback() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();
    assert!(coordinator.staged_extraction().is_some());
    coordinator.rollback().unwrap();
    assert!(coordinator.staged_extraction().is_none());
    coordinator.rollback().unwrap();
    assert!(coordinator.staged_extraction().is_none());
}

#[test]
fn stale_token_validate_does_not_affect_newer_candidate() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let prepare1 = coordinator.prepare(&bsp_bytes, None, "maps/v1").unwrap();
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/v2").unwrap();
    coordinator
        .set_renderer_mount_ready(
            bsp_runtime::BspGenerationToken {
                generation: coordinator.current_generation(),
            },
            empty_mount(),
        )
        .unwrap();
    coordinator
        .validate(bsp_runtime::BspGenerationToken {
            generation: coordinator.current_generation(),
        })
        .unwrap();
    let result = coordinator.validate(prepare1.token);
    assert!(result.is_err());
    assert!(coordinator.staged_extraction().is_some());
}

#[test]
fn rapid_prepare_rollback_prepare_cycle_is_clean() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    let _p1 = coordinator.prepare(&bsp_bytes, None, "maps/c1").unwrap();
    coordinator.rollback().unwrap();
    let p2 = coordinator.prepare(&bsp_bytes, None, "maps/c2").unwrap();
    coordinator
        .commit_with_mount(p2.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
    let _p3 = coordinator.prepare(&bsp_bytes, None, "maps/c3").unwrap();
    coordinator.rollback().unwrap();
    assert!(coordinator.is_active());
    assert!(coordinator.staged_extraction().is_none());
    let p4 = coordinator.prepare(&bsp_bytes, None, "maps/c4").unwrap();
    coordinator
        .commit_with_mount(p4.token, &mut scene, empty_mount())
        .unwrap();
    assert!(coordinator.is_active());
}

#[test]
fn retirement_count_increases_with_each_replacement() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();
    for i in 0..5 {
        let prepare = coordinator
            .prepare(&bsp_bytes, None, format!("maps/v{i}"))
            .unwrap();
        coordinator
            .commit_with_mount(prepare.token, &mut scene, empty_mount())
            .unwrap();
    }
    assert_eq!(coordinator.retirement_diagnostics(), 4);
    assert!(coordinator.is_active());
}
