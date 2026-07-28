//! Renderer runtime object identity and lifecycle vocabulary.
//!
//! Re-exports persistent vocabulary from [`engine_events`] and owns the
//! unforgeable runtime [`ObjectId`] type. No scene storage is changed in
//! this phase.

pub mod component;
pub mod identity;

// Persistent vocabulary re-exports (dependency-neutral, from engine_events)
pub use engine_events::{
    ObjectKind, SceneObjectId, SceneObjectLifecycleAction, SceneObjectLifecycleSnapshot,
    SceneObjectLifecycleEvent,
};

// Renderer runtime types
pub use identity::ObjectId;

// Re-export ObjectHandle from scene internals for test validation.
pub use crate::scene::object_store::ObjectHandle;

// Component system re-exports.
pub use component::{
    canonical_bytes, enforce_limits, hydrate_all, hydrate_all_by_key, hydrate_and_store,
    hydrate_envelope, ComponentAdapter, ComponentEnvelope, ComponentError, ComponentInstanceId,
    ComponentKey, ComponentPropertyDescriptor, ComponentPropertyType, ComponentPropertyValue,
    ComponentRegistry, ComponentStore,
};

/// Records a remapping from an old runtime [`ObjectId`] to a new one,
/// anchored to a persistent [`SceneObjectId`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectRemap {
    pub old: ObjectId,
    pub new: ObjectId,
    pub persistent: SceneObjectId,
}

/// Immutable outcome of an object lifecycle operation.
///
/// Carries runtime remaps and event-ready persistent snapshots so callers
/// can emit [`SceneObjectLifecycleEvent`]s after a mutation batch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObjectLifecycleOutcome {
    pub remaps: Vec<ObjectRemap>,
    pub snapshots: Vec<SceneObjectLifecycleSnapshot>,
}
