//! Transactional source-linked BSP runtime integration coordinator.
//!
//! The `bsp_runtime` crate is the dedicated integration coordinator that
//! atomically prepares, publishes, persists, unloads/reloads/reimports,
//! replaces, and rolls back complete source-linked BSP assets without
//! partial, stale, or conflicting publication.
//!
//! # Architecture
//!
//! - [`BspCoordinator`] owns the two-step prepare/validate/commit transaction,
//!   generation-token guards, idempotent rollback, and unload/reload/reimport
//!   semantics.
//! - [`AppBridge`] is the narrow integration trait for app-owned resources
//!   (physics, behavior state).
//! - [`CacheIdentity`] provides deterministic cache keys for every combination
//!   of inputs that changes extracted output.
//! - [`BspSourceLink`] and [`BspSourceReference`] provide durable source-linked
//!   persistence (store asset reference + settings, not expanded scene copies).
//!
//! # Crate Dependency Rules
//!
//! This crate depends on `bsp`, `renderer` (with the `bsp` feature), and
//! `engine_events`. It does NOT depend on `physics`, any app crate, or the
//! root `engine` crate.

pub mod behavior;
pub mod bridge;
pub mod cache;
pub(crate) mod candidate;
pub mod coordinator;
pub mod error;
pub mod generation;
pub mod package;
pub mod snapshot;
pub mod source_link;

// Re-export key types for convenience
pub use behavior::{
    Activation, BehaviorEntityInfo, ButtonPhase, ButtonState, DoorPhase, DoorState,
    LightStyleState, PlatformPhase, PlatformState, StructuralBehaviorAdapter, TriggerEvent,
    TriggerState,
};
pub use bridge::AppBridge;
pub use cache::CacheIdentity;
pub use candidate::BspCandidate;
pub use coordinator::BspCoordinator;
pub use error::{BridgePhase, BspRuntimeError};
pub use generation::BspGenerationToken;
pub use snapshot::{
    BspSimulationSnapshot, ExternalInstance, SnapshotActivation, SnapshotBuilder,
    SnapshotEntityPose, SnapshotEpoch, SnapshotGeneration, SnapshotLightStyles,
};
pub use source_link::{
    BspOverrideLayer, BspSourceLink, BspSourceReference, EntityOverride, LightOverride,
    OverrideReconciliation,
};
