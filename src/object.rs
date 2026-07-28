//! Object identity, kind, and summary primitives re-exported from the
//! renderer and engine-events crates.
//!
//! This module is a thin re-export layer. For advanced operations (queries,
//! selection, component adapters), use `renderer::object` directly.

pub use engine_events::{ObjectKind, SceneObjectId};
pub use renderer::object::identity::ObjectId;
pub use renderer::object::ObjectSummary;

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
