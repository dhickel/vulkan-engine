//! App bridge trait and aggregator for BSP transaction participants.
//!
//! Bridges are narrow integration hooks that receive DTOs during prepare,
//! confirm readiness during validate, publish state during activation, and
//! clean up during rollback or teardown. Each bridge owns distinct resource
//! classes (e.g., physics, behavior state) and never reaches into renderer
//! internals.
//!
//! # Phase 05: Active Bridge Receipts
//!
//! Prepared tokens and active receipts are opaque, move-only, registration-bound
//! values that carry bridge-private payloads. The aggregator wraps and unwraps
//! them; neither callers nor the coordinator can fabricate a token or exchange
//! one bridge's value with another.
//!
//! Activation is publication-only and returns an active receipt without
//! `Result`. Panics are caught without losing the borrowed prepared value.
//! Rollback consumes only prepared state; teardown consumes only active state.
//! Both are exact-once at the aggregate boundary.

use crate::error::{
    BridgePhase, BspRuntimeError, INVARIANT_DOUBLE_ACTIVATION, INVARIANT_DUPLICATE_TEARDOWN,
    INVARIANT_REGISTRATION_MISMATCH, INVARIANT_TEARDOWN_OF_PREPARED,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ── DTO Types ──────────────────────────────────────────────────────────

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
    pub killtarget: Option<String>,
    /// Movement direction (unit vector), e.g. "0 0 1" for upward.
    pub movedir: Option<[f32; 3]>,
    /// Movement speed in Quake units per second.
    pub speed: Option<f32>,
    /// Seconds to wait before auto-closing/returning.
    pub wait: Option<f32>,
    /// Lip (how far from fully closed).
    pub lip: Option<f32>,
    /// Height (for platforms).
    pub height: Option<f32>,
    /// Light style string.
    pub light_style: Option<String>,
}

// ── Bridge State Traits ────────────────────────────────────────────────

/// Opaque prepared bridge state returned by [`AppBridge::prepare`].
///
/// Carries bridge-specific resources created during prepare. The aggregator
/// wraps this in a [`PreparedBridgeToken`]; no other code may construct a token.
pub trait PreparedBridgeState: Send + std::fmt::Debug {
    /// Human-readable name of the bridge that owns this state.
    fn registration_name(&self) -> &str;

    /// Downcast to a concrete type via `Any`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Downcast to a concrete type via `Any` (mutable).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Opaque active bridge state returned by [`AppBridge::activate`].
///
/// Carries the published bridge resources. The aggregator wraps this in an
/// [`ActiveBridgeReceipt`]; no other code may construct a receipt.
pub trait ActiveBridgeState: Send + std::fmt::Debug {
    /// Human-readable name of the bridge that owns this state.
    fn registration_name(&self) -> &str;

    /// Downcast to a concrete type via `Any`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Downcast to a concrete type via `Any` (mutable).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// ── Registration Key ───────────────────────────────────────────────────

/// Private registration key tying a token/receipt to its bridge and generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BridgeRegistrationKey {
    /// Human-readable bridge name.
    pub name: String,
    /// Index in the aggregator's registration order.
    pub index: usize,
    /// Generation at which the token/receipt was created.
    pub generation: u64,
}

// ── Prepared Token ─────────────────────────────────────────────────────

/// Opaque, move-only, registration-bound prepared bridge token.
///
/// Created by the [`BridgeAggregator`] during prepare. Neither callers nor the
/// coordinator can construct, clone, or exchange a token. A token is consumed
/// by activation or rollback and cannot be reused.
pub struct PreparedBridgeToken {
    pub(crate) key: BridgeRegistrationKey,
    pub(crate) payload: Box<dyn PreparedBridgeState>,
}

impl PreparedBridgeToken {
    /// The human-readable name of the bridge this token belongs to.
    pub fn bridge_name(&self) -> &str {
        &self.key.name
    }

    /// The registration index.
    pub fn index(&self) -> usize {
        self.key.index
    }

    /// The generation this token was created at.
    pub fn generation(&self) -> u64 {
        self.key.generation
    }

    /// Borrow the prepared payload (for diagnostics only — the aggregator
    /// owns mutation).
    pub fn payload(&self) -> &dyn PreparedBridgeState {
        &*self.payload
    }
}

impl std::fmt::Debug for PreparedBridgeToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedBridgeToken")
            .field("bridge", &self.key.name)
            .field("index", &self.key.index)
            .field("generation", &self.key.generation)
            .finish()
    }
}

// ── Active Receipt ─────────────────────────────────────────────────────

/// Opaque, move-only, registration-bound active bridge receipt.
///
/// Created by the [`BridgeAggregator`] during activation. Holds the published
/// bridge state. Consumed by teardown; cannot be reused.
pub struct ActiveBridgeReceipt {
    pub(crate) key: BridgeRegistrationKey,
    pub(crate) payload: Box<dyn ActiveBridgeState>,
}

impl ActiveBridgeReceipt {
    /// The human-readable name of the bridge this receipt belongs to.
    pub fn bridge_name(&self) -> &str {
        &self.key.name
    }

    /// The registration index.
    pub fn index(&self) -> usize {
        self.key.index
    }

    /// The generation this receipt was created at.
    pub fn generation(&self) -> u64 {
        self.key.generation
    }

    /// Borrow the active payload.
    pub fn payload(&self) -> &dyn ActiveBridgeState {
        &*self.payload
    }

    /// Mutably borrow the active payload.
    pub fn payload_mut(&mut self) -> &mut dyn ActiveBridgeState {
        &mut *self.payload
    }
}

impl std::fmt::Debug for ActiveBridgeReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBridgeReceipt")
            .field("bridge", &self.key.name)
            .field("index", &self.key.index)
            .field("generation", &self.key.generation)
            .finish()
    }
}

// ── Active Receipt Collection ──────────────────────────────────────────

/// Complete collection of active bridge receipts for one committed generation.
///
/// Stored in [`ActiveBspMount`](crate::candidate::ActiveBspMount). Owns every
/// active bridge receipt in registration order. Teardown consumes the entire
/// aggregate once.
pub struct ActiveBridgeReceipts {
    pub(crate) receipts: Vec<Option<ActiveBridgeReceipt>>,
    pub(crate) generation: u64,
}

impl ActiveBridgeReceipts {
    /// Create an empty receipt aggregate for the given generation.
    pub(crate) fn empty(generation: u64) -> Self {
        Self {
            receipts: Vec::new(),
            generation,
        }
    }

    /// Return the generation this aggregate was created at.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of receipt slots (including empty ones).
    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    /// Returns true if there are no receipt slots.
    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    /// Borrow a receipt by registration index.
    pub fn get(&self, index: usize) -> Option<&ActiveBridgeReceipt> {
        self.receipts.get(index).and_then(|r| r.as_ref())
    }

    /// Mutably borrow a receipt by registration index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ActiveBridgeReceipt> {
        self.receipts.get_mut(index).and_then(|r| r.as_mut())
    }

    /// Iterate over all present receipts in registration order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ActiveBridgeReceipt)> {
        self.receipts
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().map(|r| (i, r)))
    }
}

impl std::fmt::Debug for ActiveBridgeReceipts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBridgeReceipts")
            .field("generation", &self.generation)
            .field("count", &self.receipts.iter().flatten().count())
            .finish()
    }
}

// ── Quarantine Types ───────────────────────────────────────────────────

/// A prepared token that failed rollback, retained for diagnostics.
///
/// Ownership stays in the quarantine; neither caller nor coordinator can
/// silently drop or reuse the token.
pub struct PreparedBridgeQuarantine {
    /// Bridge name at the time of failure.
    pub bridge_name: String,
    /// Bridge registration index.
    pub bridge_index: usize,
    /// Generation at time of failure.
    pub generation: u64,
    /// Phase during which the failure occurred.
    pub phase: BridgePhase,
    /// Human-readable failure message.
    pub message: String,
    /// The trapped prepared token.
    pub(crate) token: PreparedBridgeToken,
}

impl std::fmt::Debug for PreparedBridgeQuarantine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedBridgeQuarantine")
            .field("bridge_name", &self.bridge_name)
            .field("bridge_index", &self.bridge_index)
            .field("generation", &self.generation)
            .field("phase", &self.phase)
            .field("message", &self.message)
            .finish()
    }
}

/// An active receipt that failed teardown, retained for diagnostics.
///
/// Ownership stays in the quarantine.
pub struct ActiveBridgeQuarantine {
    /// Bridge name at the time of failure.
    pub bridge_name: String,
    /// Bridge registration index.
    pub bridge_index: usize,
    /// Generation at time of failure.
    pub generation: u64,
    /// Phase during which the failure occurred.
    pub phase: BridgePhase,
    /// Human-readable failure message.
    pub message: String,
    /// The trapped active receipt.
    pub(crate) receipt: ActiveBridgeReceipt,
}

impl std::fmt::Debug for ActiveBridgeQuarantine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBridgeQuarantine")
            .field("bridge_name", &self.bridge_name)
            .field("bridge_index", &self.bridge_index)
            .field("generation", &self.generation)
            .field("phase", &self.phase)
            .field("message", &self.message)
            .finish()
    }
}

/// Quarantine produced when a bridge panics during sequential activation.
///
/// Retains: completed active receipts (prefix), the current prepared token,
/// all suffix prepared tokens, bridge identity, generation, and panic phase.
pub struct ActivationQuarantine {
    /// Successfully activated receipts before the panic.
    pub completed_receipts: Vec<Option<ActiveBridgeReceipt>>,
    /// The prepared token of the bridge that panicked.
    pub current_token: Option<PreparedBridgeToken>,
    /// Remaining prepared tokens after the panicking bridge.
    pub suffix_tokens: Vec<Option<PreparedBridgeToken>>,
    /// Index of the bridge that panicked.
    pub panic_index: usize,
    /// Name of the bridge that panicked.
    pub panic_bridge: String,
    /// Generation at time of panic.
    pub generation: u64,
}

impl std::fmt::Debug for ActivationQuarantine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationQuarantine")
            .field("panic_bridge", &self.panic_bridge)
            .field("panic_index", &self.panic_index)
            .field("generation", &self.generation)
            .field(
                "completed_count",
                &self.completed_receipts.iter().flatten().count(),
            )
            .field("suffix_count", &self.suffix_tokens.iter().flatten().count())
            .finish()
    }
}

/// Quarantine produced when teardown fails or panics for one or more bridges.
///
/// Retains: the failed receipt (if one errored), all unattempted receipts,
/// and generation. Deterministic registration order is preserved.
pub struct TeardownQuarantine {
    /// The first receipt that failed teardown, with its error message.
    pub failed: Option<(ActiveBridgeReceipt, String)>,
    /// Bridge name of the failure.
    pub failed_bridge: Option<String>,
    /// Bridge index of the failure.
    pub failed_index: Option<usize>,
    /// All unattempted receipts (after the failure or panic).
    pub unattempted: Vec<Option<ActiveBridgeReceipt>>,
    /// Generation at time of teardown.
    pub generation: u64,
    /// Whether a panic occurred (vs. an error return).
    pub was_panic: bool,
}

impl std::fmt::Debug for TeardownQuarantine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TeardownQuarantine")
            .field("failed_bridge", &self.failed_bridge)
            .field("failed_index", &self.failed_index)
            .field("generation", &self.generation)
            .field("was_panic", &self.was_panic)
            .field(
                "unattempted_count",
                &self.unattempted.iter().flatten().count(),
            )
            .finish()
    }
}

// ── App Bridge Trait ───────────────────────────────────────────────────

/// Trait for app-owned integration hooks.
///
/// Each bridge receives narrow DTOs and produces a typed prepared state during
/// prepare. The aggregator wraps it in a [`PreparedBridgeToken`]. Validation
/// borrows the prepared state. Activation consumes the prepared state and
/// produces an [`ActiveBridgeReceipt`]. Teardown consumes the active receipt.
///
/// # Phase 05 Contract
///
/// - **Prepare**: Creates hidden resources from DTOs. Returns a
///   `Box<dyn PreparedBridgeState>`. All fallible work (including physics body
///   and collider creation) happens here. Resources are NOT published to the
///   active simulation/scene.
/// - **Validate**: Borrows prepared state. Confirms all resources are valid.
///   This is the last chance to fail.
/// - **Activate**: Consumes prepared state and publishes it. **Must be
///   non-fallible** after validate passes. Returns `Box<dyn ActiveBridgeState>`.
///   A panic is caught without losing the prepared value.
/// - **Rollback**: Destroys prepared resources. Receives `&mut` to the prepared
///   payload; the aggregator retains ownership of the wrapper.
/// - **Teardown**: Removes active resources. Receives `&mut` to the active
///   payload. On error, the receipt is quarantined rather than dropped.
pub trait AppBridge: Send {
    /// Return the human-readable name of this bridge.
    fn name(&self) -> &str;

    /// Called during prepare. The bridge creates resources from the provided
    /// DTOs but does NOT publish them to the active simulation/scene.
    ///
    /// Returns a boxed prepared state that the aggregator wraps in a token.
    fn prepare(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Box<dyn PreparedBridgeState>, String>;

    /// Called during validate. The bridge confirms all prepared resources are
    /// valid and ready for publication. Borrows the prepared payload immutably.
    fn validate(&self, prepared: &dyn PreparedBridgeState) -> Result<(), String>;

    /// Called during activation. The bridge publishes prepared resources and
    /// returns an active state. This must be non-fallible after validate passes.
    ///
    /// The aggregator retains ownership of the wrapper; it passes a mutable
    /// borrow of the prepared payload and wraps the returned active state.
    fn activate(&mut self, prepared: &mut dyn PreparedBridgeState) -> Box<dyn ActiveBridgeState>;

    /// Called during teardown. The bridge removes active resources. Returns an
    /// error if cleanup fails; the receipt is quarantined on error.
    ///
    /// Receives `&mut` to the active payload while the aggregator retains the
    /// wrapper. On error or panic, the receipt is moved to [`TeardownQuarantine`].
    fn teardown(&mut self, active: &mut dyn ActiveBridgeState) -> Result<(), String>;

    /// Called during rollback. The bridge destroys prepared resources.
    /// Idempotent — subsequent calls should be no-ops.
    ///
    /// Receives `&mut` to the prepared payload while the aggregator retains
    /// the wrapper. On error or panic, the token is moved to
    /// [`PreparedBridgeQuarantine`].
    fn rollback(&mut self, prepared: &mut dyn PreparedBridgeState);
}

// ── Bridge Entry ───────────────────────────────────────────────────────

/// A single named bridge instance stored in the aggregator.
pub(crate) struct BridgeEntry {
    pub name: String,
    pub bridge: Box<dyn AppBridge>,
}

// ── Bridge Aggregator ──────────────────────────────────────────────────

/// Aggregator that invokes hooks across all registered bridges and collects
/// failures. Used internally by the coordinator.
///
/// # Phase 05 Ownership
///
/// - `prepare_all`: Returns `Vec<Option<PreparedBridgeToken>>` for candidate storage.
/// - `validate_all`: Borrows candidate-stored prepared tokens.
/// - `activate_all`: Consumes prepared tokens, returns `ActiveBridgeReceipts`
///   or an [`ActivationQuarantine`] on panic.
/// - `rollback_all`: Consumes prepared tokens; returns
///   [`PreparedBridgeQuarantine`] on failure.
/// - `teardown_all`: Consumes active receipts; returns
///   [`TeardownQuarantine`] on failure.
pub(crate) struct BridgeAggregator {
    bridges: Vec<BridgeEntry>,
}

impl BridgeAggregator {
    pub fn new() -> Self {
        Self {
            bridges: Vec::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, bridge: Box<dyn AppBridge>) {
        let name = name.into();
        let index = self.bridges.len();
        self.bridges.push(BridgeEntry { name, bridge });
        // Ensure consistent indexing — index is assigned at push time.
        let _ = index;
    }

    // ── Prepare ──────────────────────────────────────────────────────

    /// Prepare all bridges and return wrapped prepared tokens for candidate storage.
    ///
    /// On failure, successfully prepared tokens are rolled back before returning.
    pub fn prepare_all(
        &mut self,
        generation: u64,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        lights: &[LightEntityRecipe],
        behaviors: &[BehaviorEntityRecipe],
    ) -> Result<Vec<Option<PreparedBridgeToken>>, BspRuntimeError> {
        let mut failures = Vec::new();
        let mut tokens: Vec<Option<PreparedBridgeToken>> = Vec::with_capacity(self.bridges.len());

        for (idx, entry) in self.bridges.iter_mut().enumerate() {
            let key = BridgeRegistrationKey {
                name: entry.name.clone(),
                index: idx,
                generation,
            };

            let result = catch_unwind(AssertUnwindSafe(|| {
                entry
                    .bridge
                    .prepare(world_collider, entity_colliders, lights, behaviors)
            }));
            match result {
                Ok(Ok(payload)) => {
                    tokens.push(Some(PreparedBridgeToken { key, payload }));
                }
                Ok(Err(msg)) => {
                    failures.push((idx, entry.name.clone(), BridgePhase::Prepare, msg));
                    tokens.push(None);
                }
                Err(_) => {
                    failures.push((
                        idx,
                        entry.name.clone(),
                        BridgePhase::Prepare,
                        "bridge panicked during prepare".to_string(),
                    ));
                    tokens.push(None);
                }
            }
        }

        if !failures.is_empty() {
            // Roll back any successfully prepared tokens
            if let Err(q) = self.rollback_all(std::mem::take(&mut tokens)) {
                // Rollback itself panicked — coordinator is poisoned
                log::error!(
                    "BSP bridge aggregator: rollback panic during prepare failure recovery: {:?}",
                    q
                );
                return Err(BspRuntimeError::CoordinatorPoisoned);
            }

            let (_, name, phase, msg) = failures.remove(0);
            for (_, n, _, m) in &failures {
                log::warn!("additional bridge prepare failure: [{}] {}", n, m);
            }
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: name,
                phase,
                message: msg,
            });
        }

        Ok(tokens)
    }

    // ── Validate ─────────────────────────────────────────────────────

    /// Validate all bridges using prepared tokens stored in a candidate.
    ///
    /// Borrows the prepared payloads immutably. Tokens remain owned by the
    /// caller (candidate).
    pub fn validate_all(
        &self,
        tokens: &[Option<PreparedBridgeToken>],
    ) -> Result<(), BspRuntimeError> {
        for (idx, entry) in self.bridges.iter().enumerate() {
            let Some(ref token) = tokens.get(idx).and_then(|t| t.as_ref()) else {
                return Err(BspRuntimeError::BridgeFailure {
                    bridge_name: entry.name.clone(),
                    phase: BridgePhase::Validate,
                    message: "no prepared token for bridge".to_string(),
                });
            };

            let result = catch_unwind(AssertUnwindSafe(|| entry.bridge.validate(&*token.payload)));
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

    // ── Activate ─────────────────────────────────────────────────────

    /// Activate all bridges sequentially, consuming prepared tokens and
    /// returning active receipts.
    ///
    /// Activation is publication-only and non-fallible (by contract after
    /// validation). On panic, the aggregator retains completed receipts,
    /// the current token, and suffix tokens in an [`ActivationQuarantine`].
    pub fn activate_all(
        &mut self,
        mut tokens: Vec<Option<PreparedBridgeToken>>,
    ) -> Result<ActiveBridgeReceipts, ActivationQuarantine> {
        let generation = tokens
            .iter()
            .find_map(|t| t.as_ref().map(|t| t.key.generation))
            .unwrap_or(0);
        let total = tokens.len();
        let mut completed: Vec<Option<ActiveBridgeReceipt>> = Vec::with_capacity(total);

        for idx in 0..total {
            // Take the token at this index
            let mut token = match tokens.get_mut(idx).and_then(|t| t.take()) {
                Some(t) => t,
                None => {
                    // Absent token at this index — invariant violation
                    // Return what we have so far as quarantine
                    let suffix = tokens.split_off(idx + 1);
                    let current = tokens.into_iter().nth(idx).flatten();
                    return Err(ActivationQuarantine {
                        completed_receipts: completed,
                        current_token: current,
                        suffix_tokens: suffix,
                        panic_index: idx,
                        panic_bridge: self
                            .bridges
                            .get(idx)
                            .map(|e| e.name.clone())
                            .unwrap_or_else(|| "<unknown>".into()),
                        generation,
                    });
                }
            };

            let bridge_name = token.key.name.clone();
            let panic_idx = idx;

            // Activate the bridge — non-fallible by contract
            let result = catch_unwind(AssertUnwindSafe(|| {
                let entry = &mut self.bridges[idx];
                entry.bridge.activate(&mut *token.payload)
            }));

            match result {
                Ok(active_payload) => {
                    let receipt = ActiveBridgeReceipt {
                        key: token.key,
                        payload: active_payload,
                    };
                    completed.push(Some(receipt));
                }
                Err(_panic) => {
                    // Panic during activation — retain all state
                    let suffix = tokens.split_off(idx + 1);
                    return Err(ActivationQuarantine {
                        completed_receipts: completed,
                        current_token: Some(token),
                        suffix_tokens: suffix,
                        panic_index: panic_idx,
                        panic_bridge: bridge_name,
                        generation,
                    });
                }
            }
        }

        Ok(ActiveBridgeReceipts {
            receipts: completed,
            generation,
        })
    }

    // ── Rollback ─────────────────────────────────────────────────────

    /// Roll back all prepared tokens, consuming them.
    ///
    /// On failure or panic, the individual token is retained in a
    /// [`PreparedBridgeQuarantine`] and the remaining tokens are still
    /// attempted. The first failure is returned.
    pub fn rollback_all(
        &mut self,
        mut tokens: Vec<Option<PreparedBridgeToken>>,
    ) -> Result<(), PreparedBridgeQuarantine> {
        let generation = tokens
            .iter()
            .find_map(|t| t.as_ref().map(|t| t.key.generation))
            .unwrap_or(0);
        let mut first_quarantine: Option<PreparedBridgeQuarantine> = None;

        for idx in 0..tokens.len() {
            let Some(mut token) = tokens.get_mut(idx).and_then(|t| t.take()) else {
                continue;
            };

            let bridge_name = token.key.name.clone();
            let gen = token.key.generation;

            let result = catch_unwind(AssertUnwindSafe(|| {
                let entry = &mut self.bridges[idx];
                entry.bridge.rollback(&mut *token.payload);
            }));

            if let Err(_panic) = result {
                let quarantine = PreparedBridgeQuarantine {
                    bridge_name,
                    bridge_index: idx,
                    generation: gen,
                    phase: BridgePhase::Rollback,
                    message: "bridge panicked during rollback".to_string(),
                    token,
                };
                if first_quarantine.is_none() {
                    first_quarantine = Some(quarantine);
                } else {
                    log::error!(
                        "additional bridge rollback panic: [{}/{}]",
                        quarantine.bridge_name,
                        quarantine.message
                    );
                }
            }
        }

        if let Some(q) = first_quarantine {
            Err(q)
        } else {
            Ok(())
        }
    }

    // ── Teardown ─────────────────────────────────────────────────────

    /// Tear down all active receipts, consuming them.
    ///
    /// On error or panic, the failing receipt and all unattempted receipts
    /// are retained in a [`TeardownQuarantine`]. A successful aggregate
    /// teardown consumes every receipt once.
    pub fn teardown_all(
        &mut self,
        mut receipts: ActiveBridgeReceipts,
    ) -> Result<(), TeardownQuarantine> {
        let generation = receipts.generation;
        let total = receipts.receipts.len();
        let mut first_error: Option<(ActiveBridgeReceipt, String, usize, String, bool)> = None;

        for idx in 0..total {
            let Some(mut receipt) = receipts.receipts.get_mut(idx).and_then(|r| r.take()) else {
                continue;
            };

            let bridge_name = receipt.key.name.clone();

            let result = catch_unwind(AssertUnwindSafe(|| {
                let entry = &mut self.bridges[idx];
                entry.bridge.teardown(&mut *receipt.payload)
            }));

            match result {
                Ok(Ok(())) => {
                    // Successful teardown — receipt is consumed and dropped
                }
                Ok(Err(msg)) => {
                    first_error = Some((receipt, msg, idx, bridge_name, false));
                    break;
                }
                Err(_panic) => {
                    first_error = Some((
                        receipt,
                        "bridge panicked during teardown".to_string(),
                        idx,
                        bridge_name,
                        true,
                    ));
                    break;
                }
            }
        }

        if let Some((failed_receipt, msg, idx, name, was_panic)) = first_error {
            // Collect unattempted receipts
            let unattempted: Vec<Option<ActiveBridgeReceipt>> =
                receipts.receipts.into_iter().skip(idx + 1).collect();

            Err(TeardownQuarantine {
                failed: Some((failed_receipt, msg.clone())),
                failed_bridge: Some(name),
                failed_index: Some(idx),
                unattempted,
                generation,
                was_panic,
            })
        } else {
            Ok(())
        }
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

// ── Invariant Check Helpers ────────────────────────────────────────────

/// Verify that a token belongs to the expected bridge index.
pub(crate) fn check_token_index(
    token: &PreparedBridgeToken,
    expected_index: usize,
    expected_name: &str,
) -> Result<(), BspRuntimeError> {
    if token.key.index != expected_index {
        return Err(BspRuntimeError::BridgeFailure {
            bridge_name: expected_name.to_string(),
            phase: BridgePhase::Activate,
            message: format!(
                "{}: token index {} does not match registration index {}",
                INVARIANT_REGISTRATION_MISMATCH, token.key.index, expected_index
            ),
        });
    }
    Ok(())
}

/// Verify that a receipt belongs to the expected bridge index.
pub(crate) fn check_receipt_index(
    receipt: &ActiveBridgeReceipt,
    expected_index: usize,
    expected_name: &str,
) -> Result<(), BspRuntimeError> {
    if receipt.key.index != expected_index {
        return Err(BspRuntimeError::BridgeFailure {
            bridge_name: expected_name.to_string(),
            phase: BridgePhase::Teardown,
            message: format!(
                "{}: receipt index {} does not match registration index {}",
                INVARIANT_REGISTRATION_MISMATCH, receipt.key.index, expected_index
            ),
        });
    }
    Ok(())
}
