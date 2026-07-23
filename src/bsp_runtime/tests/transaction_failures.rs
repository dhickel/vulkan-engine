//! Integration tests: prepare failure, validation failure, cancellation,
//! stale generation, bridge failure, and rollback.

use bsp_runtime::{
    bridge::{
        AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe, LightEntityRecipe,
        WorldCollisionRecipe,
    },
    coordinator::BspCoordinator,
    error::BspRuntimeError,
};

use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;

/// Build a minimal valid BSP29 for testing.
fn minimal_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();

    // Header: version (4 bytes) + 15 lump descriptors (120 bytes) = 124 bytes
    data.extend_from_slice(&29u32.to_le_bytes());

    // Lump table: all lumps empty except entities and a plane
    let mut current_offset: u32 = 124;

    // Entities: minimal entity string (null-terminated)
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset = current_offset;
    let entity_size = entity_bytes.len() as u32;
    current_offset += entity_size;

    // Plane: one plane
    let plane_offset = current_offset;
    let plane_size = 20u32;
    current_offset += plane_size;

    // Build lump table
    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size), // entities
        (plane_offset, plane_size),   // planes
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0), // miptex, vertices, vis, nodes, texinfo
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0), // faces, lightmaps, clipnodes, leaves, markfaces
        (0, 0),
        (0, 0),
        (0, 0), // edges, surfedges, models
    ];

    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }

    // Write entity data
    data.extend_from_slice(entity_bytes);

    // Write plane data: (0,0,1), dist=0, type=0
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());

    data
}

/// Build an empty PreparedBspMount for testing.
fn empty_mount() -> PreparedBspMount {
    PreparedBspMount::new()
}

/// A bridge that always succeeds.
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

/// A bridge that fails validate.
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

// ── Tests ──────────────────────────────────────────────────────────────

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

    // Start second prepare, invalidating first
    let _prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    // First generation should now be stale
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

    // Start second prepare
    let prepare2 = coordinator.prepare(&bsp_bytes, None, "maps/test2").unwrap();

    // Validate second prepare
    coordinator.validate(prepare2.token).unwrap();

    // Try to commit with first generation token
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

    // Coordinator should not be poisoned by prepare failure
    assert!(!coordinator.is_poisoned());

    // The failing bridge is still registered; second prepare also fails
    let result2 = coordinator.prepare(&bsp_bytes, None, "maps/test");
    assert!(result2.is_err());
    match result2.unwrap_err() {
        BspRuntimeError::BridgeFailure { phase, .. } => {
            assert!(matches!(phase, bsp_runtime::error::BridgePhase::Prepare));
        }
        e => panic!("expected BridgeFailure, got {:?}", e),
    }
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
}

#[test]
fn rollback_is_idempotent() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();

    // Prepare but don't commit
    let _prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    // Rollback multiple times
    assert!(coordinator.rollback().is_ok());
    assert!(coordinator.rollback().is_ok());
    assert!(coordinator.rollback().is_ok());

    // Should be able to prepare again
    let result = coordinator.prepare(&bsp_bytes, None, "maps/test2");
    assert!(result.is_ok());
}

#[test]
fn commit_and_unload_cycle() {
    let bsp_bytes = minimal_bsp_bytes();
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Prepare
    let prepare = coordinator.prepare(&bsp_bytes, None, "maps/test").unwrap();

    // Validate
    coordinator.validate(prepare.token).unwrap();

    // Commit
    let mount = empty_mount();
    let commit = coordinator.commit_with_mount(prepare.token, &mut scene, mount);
    assert!(commit.is_ok());
    assert!(coordinator.is_active());

    // Unload
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

    // validate should work with prepare2's token
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
