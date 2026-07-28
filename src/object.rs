//! Object identity, kind, and summary primitives re-exported from the
//! renderer and engine-events crates.
//!
//! This module is a thin re-export layer; implementation remains in the
//! `renderer` and `engine_events` crates.

pub use engine_events::{ObjectKind, SceneObjectId};
pub use renderer::object::query::{
    EditorPickResult, EditorProxyPolicy, ObjectQueryFilter, RayHit, UnknownBoundsPolicy, VolumeHit,
    VolumeQuery, VolumeShape,
};
pub use renderer::object::selection::{Selection, SelectionChange};
pub use renderer::object::{
    ComponentAdapter, ComponentEnvelope, ComponentError, ComponentInstanceId, ComponentKey,
    ComponentPropertyDescriptor, ComponentPropertyType, ComponentPropertyValue, ComponentRegistry,
    ObjectCapabilities, ObjectDuplicateRequest, ObjectId, ObjectMutationOutcome, ObjectParent,
    ObjectRemovalSnapshot, ObjectSummary, ObjectTransform, TransformCapabilities,
};

/// Convenience helper: get the [`ObjectKind`] from an [`ObjectId`].
///
/// This is a synonym for [`ObjectId::kind`] provided for discoverability
/// at the facade layer.
#[inline]
pub fn object_kind(id: &ObjectId) -> ObjectKind {
    id.kind()
}

/// Convenience helper: convert an [`ObjectKind`] to a display label.
#[inline]
pub fn object_kind_label(kind: ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Node => "Node",
        ObjectKind::PointLight => "PointLight",
        ObjectKind::DirectionalLight => "DirectionalLight",
        ObjectKind::SpotLight => "SpotLight",
    }
}
