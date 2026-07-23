//! App bridge trait and aggregator for BSP transaction participants.
//!
//! Bridges are narrow integration hooks that receive DTOs during prepare,
//! confirm readiness during validate, publish state during commit, and
//! clean up during rollback. Each bridge owns distinct resource classes
//! (e.g., physics, behavior state) and never reaches into renderer internals.
//!
//! # Phase 05: Hidden Prepared Batches
//!
//! Bridge prepare creates hidden resource batches. Bridge tokens are stored
//! in the [`BspCandidate`]. Validation reserves all app-world IDs/capacity
//! and rejects invalid values before commit. After validation passes,
//! activation during commit is non-fallible; a panic poisons the coordinator.

use crate::candidate::BspCandidate;
use crate::error::{BridgeFailure, BridgePhase, BspRuntimeError};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Lightweight DTO describing world collision geometry for the app bridge.
#[derive(Debug, Clone)]
pub struct WorldCollisionRecipe {
    /// World collision planes in engine space: (normal, distance).
    pub planes: Vec<(glam::Vec3, f32)>,
}

/// Lightweight DTO describing entity collision geometry for the app bridge.
#[derive(Debug, Clone)]
pub struct EntityCollisionRecipe {
    /// Source entity index in the BSP entity lump.
    pub entity_index: u32,
    /// Classname of the entity.
    pub classname: String,
    /// Origin in engine space.
    pub origin: glam::Vec3,
    /// Whether this is a trigger (sensor-only).
    pub is_trigger: bool,
    /// Collision recipes from convex reconstruction.
    pub recipes: Vec<bsp::collision::CollisionRecipe>,
}

/// DTO for light entities passed to the app bridge during prepare.
#[derive(Debug, Clone)]
pub struct LightEntityRecipe {
    pub entity_index: u32,
    pub origin: glam::Vec3,
    pub intensity: f32,
    pub color: [f32; 3],
    pub radius: f32,
    pub style: Option<String>,
}

/// DTO for behavior entities (doors, buttons, platforms) during prepare.
#[derive(Debug, Clone)]
pub struct BehaviorEntityRecipe {
    pub entity_index: u32,
    pub classname: String,
    pub origin: glam::Vec3,
    pub targetname: Option<String>,
    pub target: Option<String>,
}

/// Token returned by a bridge during prepare, presented at validate/commit/rollback.
#[derive(Debug, Clone)]
pub struct BridgeToken {
    /// Token payload: opaque to the coordinator.
    pub payload: Vec<u8>,
}

impl BridgeToken {
    pub fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }
}

/// A single named bridge instance stored in the coordinator.
pub(crate) struct BridgeEntry {
    pub name: String,
    pub bridge: Box<dyn AppBridge>,
}

/// Trait for app-owned integration hooks.
///
/// Each bridge receives narrow DTOs and returns a [`BridgeToken`] during
/// prepare. The token is presented at validate, commit, and rollback.
/// Bridges are called with no engine lock held.
///
/// # Phase 05 Contract
///
/// - **Prepare**: Creates hidden resources from DTOs. Returns a token.
///   Resources are NOT yet published to the active simulation/scene.
/// - **Validate**: App confirms all prepared resources are valid. Must
///   reserve all app-world IDs/capacity and reject invalid values.
///   This is the last chance to fail.
/// - **Commit**: App publishes prepared resources. **Must be non-fallible**
///   after validate passes. A panic poisons the coordinator.
/// - **Rollback**: App removes any resources created during prepare or
///   commit. Must be idempotent.
///
/// # Panic Safety
/// If any bridge hook panics during commit or rollback, the coordinator
/// enters a poisoned state. Panics during prepare or validate are caught
/// and returned as errors.
pub trait AppBridge: Send {
    /// Return the human-readable name of this bridge.
    fn name(&self) -> &str;

    /// Called during prepare. The bridge creates resources from the provided
    /// DTOs but does NOT publish them to the active simulation/scene.
    ///
    /// Returns a token that identifies the prepared state.
    fn prepare(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String>;

    /// Called during validate. The bridge confirms all prepared resources are
    /// valid and ready for publication. Must reserve all app-world IDs/capacity.
    fn validate(&self, token: &BridgeToken) -> Result<(), String>;

    /// Called during commit. The bridge publishes prepared resources to the
    /// active simulation/scene. This must be non-fallible after validate passes.
    fn commit(&mut self, token: BridgeToken) -> Result<(), String>;

    /// Called during rollback. The bridge removes any resources created during
    /// prepare or commit. Must be idempotent — subsequent calls are no-ops.
    fn rollback(&mut self, token: BridgeToken);
}

/// Aggregator that invokes hooks across all registered bridges and collects
/// failures. Used internally by the coordinator.
///
/// # Phase 05 Changes
///
/// - `prepare_with_tokens`: Returns tokens for storage in the candidate.
/// - `validate_candidate`: Validates bridges using candidate-stored tokens.
/// - `commit_candidate`: Commits bridges from candidate-stored tokens.
/// - `rollback_tokens`: Rolls back bridges from externally-owned tokens.
pub(crate) struct BridgeAggregator {
    bridges: Vec<BridgeEntry>,
    /// Tokens returned during the last prepare, indexed by bridge position.
    /// These are consumed by commit/rollback and replaced on new prepare.
    prepared_tokens: Vec<Option<BridgeToken>>,
}

impl BridgeAggregator {
    pub fn new() -> Self {
        Self {
            bridges: Vec::new(),
            prepared_tokens: Vec::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, bridge: Box<dyn AppBridge>) {
        self.bridges.push(BridgeEntry {
            name: name.into(),
            bridge,
        });
        self.prepared_tokens.push(None);
    }

    /// Phase 05: Prepare all bridges and return tokens for candidate storage.
    ///
    /// The tokens are returned to the caller for storage in the [`BspCandidate`].
    pub fn prepare_with_tokens(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Vec<Option<BridgeToken>>, BspRuntimeError> {
        let mut failures = Vec::new();
        let mut tokens: Vec<Option<BridgeToken>> = Vec::with_capacity(self.bridges.len());

        for entry in self.bridges.iter_mut() {
            let result = catch_unwind(AssertUnwindSafe(|| {
                entry
                    .bridge
                    .prepare(world_collider, entity_colliders, lights, behaviors)
            }));
            match result {
                Ok(Ok(token)) => {
                    tokens.push(Some(token));
                }
                Ok(Err(msg)) => {
                    failures.push(BridgeFailure::new(
                        entry.name.clone(),
                        BridgePhase::Prepare,
                        msg,
                    ));
                    tokens.push(None);
                }
                Err(_) => {
                    failures.push(BridgeFailure::new(
                        entry.name.clone(),
                        BridgePhase::Prepare,
                        "bridge panicked during prepare".to_string(),
                    ));
                    tokens.push(None);
                }
            }
        }

        if !failures.is_empty() {
            // Roll back any successfully prepared bridges
            let rollback_failures = self.rollback_tokens(std::mem::take(&mut tokens));
            if !rollback_failures.is_empty() {
                return Err(BspRuntimeError::CoordinatorPoisoned);
            }

            let first = failures.remove(0);
            for f in &failures {
                log::warn!(
                    "additional bridge prepare failure: [{}] {}",
                    f.bridge_name,
                    f.message
                );
            }
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: first.bridge_name,
                phase: first.phase,
                message: first.message,
            });
        }

        Ok(tokens)
    }

    /// Validate bridges using tokens stored in the candidate.
    pub fn validate_candidate(&self, candidate: &BspCandidate) -> Result<(), BspRuntimeError> {
        for (i, entry) in self.bridges.iter().enumerate() {
            let Some(ref token) = candidate.bridge_tokens.get(i).and_then(|t| t.as_ref()) else {
                return Err(BspRuntimeError::BridgeFailure {
                    bridge_name: entry.name.clone(),
                    phase: BridgePhase::Validate,
                    message: "no prepared token for bridge".to_string(),
                });
            };
            let result = catch_unwind(AssertUnwindSafe(|| entry.bridge.validate(token)));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Validate,
                        message: msg,
                    });
                }
                Err(_) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Validate,
                        message: "bridge panicked during validate".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Commit bridges using tokens stored in the candidate.
    ///
    /// After validation, commit must be non-fallible. A bridge panic or
    /// failure during commit poisons the coordinator.
    pub fn commit_candidate(
        &mut self,
        candidate: &mut BspCandidate,
    ) -> Result<(), BspRuntimeError> {
        for (idx, entry) in self.bridges.iter_mut().enumerate() {
            let token = candidate
                .bridge_tokens
                .get_mut(idx)
                .and_then(|t| t.take())
                .ok_or_else(|| BspRuntimeError::BridgeFailure {
                    bridge_name: entry.name.clone(),
                    phase: BridgePhase::Commit,
                    message: "no prepared token for bridge".to_string(),
                })?;

            let result = catch_unwind(AssertUnwindSafe(|| entry.bridge.commit(token)));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Commit,
                        message: msg,
                    });
                }
                Err(_) => return Err(BspRuntimeError::CoordinatorPoisoned),
            }
        }
        Ok(())
    }

    /// Legacy: Prepare bridges and store tokens internally.
    #[allow(dead_code)]
    pub fn prepare(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<(), BspRuntimeError> {
        let tokens =
            self.prepare_with_tokens(world_collider, entity_colliders, lights, behaviors)?;
        self.prepared_tokens = tokens;
        Ok(())
    }

    /// Legacy: Validate bridges using internally stored tokens.
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), BspRuntimeError> {
        for (i, entry) in self.bridges.iter().enumerate() {
            let Some(ref token) = self.prepared_tokens[i] else {
                return Err(BspRuntimeError::BridgeFailure {
                    bridge_name: entry.name.clone(),
                    phase: BridgePhase::Validate,
                    message: "no prepared token for bridge".to_string(),
                });
            };
            let result = catch_unwind(AssertUnwindSafe(|| entry.bridge.validate(token)));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Validate,
                        message: msg,
                    });
                }
                Err(_) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Validate,
                        message: "bridge panicked during validate".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Legacy: Commit bridges using internally stored tokens.
    #[allow(dead_code)]
    pub fn commit(&mut self) -> Result<(), BspRuntimeError> {
        for (i, entry) in self.bridges.iter_mut().enumerate() {
            let token =
                self.prepared_tokens[i]
                    .take()
                    .ok_or_else(|| BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Commit,
                        message: "no prepared token for bridge".to_string(),
                    })?;

            let result = catch_unwind(AssertUnwindSafe(|| entry.bridge.commit(token)));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    return Err(BspRuntimeError::BridgeFailure {
                        bridge_name: entry.name.clone(),
                        phase: BridgePhase::Commit,
                        message: msg,
                    });
                }
                Err(_) => return Err(BspRuntimeError::CoordinatorPoisoned),
            }
        }
        Ok(())
    }

    /// Roll back bridges from externally-owned tokens (Phase 05).
    pub fn rollback_tokens(&mut self, tokens: Vec<Option<BridgeToken>>) -> Vec<BridgeFailure> {
        let mut failures = Vec::new();
        for (i, entry) in self.bridges.iter_mut().enumerate() {
            if let Some(Some(token)) = tokens.get(i).cloned() {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    entry.bridge.rollback(token);
                }));
                if result.is_err() {
                    failures.push(BridgeFailure::new(
                        entry.name.clone(),
                        BridgePhase::Rollback,
                        "bridge panicked during rollback".to_string(),
                    ));
                }
            }
        }
        failures
    }

    /// Legacy: Roll back bridges from internally stored tokens.
    #[allow(dead_code)]
    pub fn rollback(&mut self) -> Vec<BridgeFailure> {
        let mut failures = Vec::new();
        for (i, entry) in self.bridges.iter_mut().enumerate() {
            if let Some(token) = self.prepared_tokens[i].take() {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    entry.bridge.rollback(token);
                }));
                if result.is_err() {
                    failures.push(BridgeFailure::new(
                        entry.name.clone(),
                        BridgePhase::Rollback,
                        "bridge panicked during rollback".to_string(),
                    ));
                }
            }
        }
        for token_opt in &mut self.prepared_tokens {
            *token_opt = None;
        }
        failures
    }

    // ── Query ───────────────────────────────────────────────────────

    /// Check if any bridges are registered.
    pub fn has_bridges(&self) -> bool {
        !self.bridges.is_empty()
    }

    /// Return the number of registered bridges.
    pub fn len(&self) -> usize {
        self.bridges.len()
    }
}

impl Default for BridgeAggregator {
    fn default() -> Self {
        Self::new()
    }
}
