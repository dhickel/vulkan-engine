//! Scene query DTOs for raycasts, volume queries, and editor picking.
//!
//! These types are pure data — the Scene API consumes them and returns
//! matching results without mutating the scene.

use crate::data::camera::{Aabb, AabbRayHit, Frustum, Ray};
use crate::object::identity::ObjectId;
use engine_events::{ObjectKind, SceneObjectId};
use glam::Vec3;
use std::collections::BTreeSet;

// ── RayHit ────────────────────────────────────────────────────────────

/// Rich raycast result describing which object was hit, where, and how.
#[derive(Clone, Debug, PartialEq)]
pub struct RayHit {
    /// The hit object.
    pub object: ObjectId,
    /// Persistent scene identity.
    pub persistent_id: SceneObjectId,
    /// Object kind for sort-key stability.
    pub kind: ObjectKind,
    /// Distance from ray origin along the ray direction (t-value).
    pub distance: f32,
    /// World-space hit point.
    pub point: Vec3,
    /// World-space entry-face normal at the hit point. `None` when the ray
    /// began inside the bounds.
    pub normal: Option<Vec3>,
    /// Whether the hit came from a proxy AABB rather than known geometry.
    pub is_proxy: bool,
}

// ── Query options ─────────────────────────────────────────────────────

/// Policy for objects whose bounds are unknown or conservative-visible.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnknownBoundsPolicy {
    /// Include unknown-bounds objects as indeterminate hits with
    /// `is_bounded: false`. Conservative default: never produce false
    /// negatives.
    IncludeAsIndeterminate,
    /// Exclude unknown-bounds objects from results entirely.
    Exclude,
    /// Include unknown-bounds objects as "conservative hits" with
    /// distance = f32::INFINITY. These sort after all known hits but
    /// before nothing-found.
    IncludeAsInfinite,
}

impl Default for UnknownBoundsPolicy {
    fn default() -> Self {
        Self::IncludeAsIndeterminate
    }
}

/// Policy for which objects an editor pick should consider.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EditorProxyPolicy {
    /// Only nodes are eligible for editor pick (default).
    NodesOnly,
    /// Nodes and point/spot lights with known bounds.
    NodesAndBoundedLights,
    /// Nodes, point/spot lights, and directional lights (editor-visible).
    All,
}

impl Default for EditorProxyPolicy {
    fn default() -> Self {
        Self::NodesOnly
    }
}

/// Filter controls for scene queries: which object kinds, layers, and
/// visibility state to include.
#[derive(Clone, Debug)]
pub struct ObjectQueryFilter {
    /// Only include objects whose [`ObjectKind`] is in this set.
    /// An empty set means "all kinds".
    pub kind_set: BTreeSet<ObjectKind>,
    /// Layer mask: an object passes the filter when
    /// `(object_layer_mask & self.layer_mask) != 0`.
    /// The default `u64::MAX` matches every layer.
    pub layer_mask: u64,
    /// When `true`, invisible objects are excluded from results.
    pub require_visible: bool,
}

impl Default for ObjectQueryFilter {
    fn default() -> Self {
        Self {
            kind_set: BTreeSet::new(),
            layer_mask: u64::MAX,
            require_visible: false,
        }
    }
}

impl ObjectQueryFilter {
    /// Create a filter that only matches the given object kinds.
    pub fn kinds(kinds: impl IntoIterator<Item = ObjectKind>) -> Self {
        Self {
            kind_set: kinds.into_iter().collect(),
            ..Default::default()
        }
    }

    /// Restrict to a specific layer mask value.
    pub fn with_layer_mask(mut self, mask: u64) -> Self {
        self.layer_mask = mask;
        self
    }

    /// Require only visible objects.
    pub fn with_require_visible(mut self, require: bool) -> Self {
        self.require_visible = require;
        self
    }

    /// Returns `true` when `kind` passes this filter's kind set.
    pub fn allows_kind(&self, kind: ObjectKind) -> bool {
        self.kind_set.is_empty() || self.kind_set.contains(&kind)
    }
}

/// Configuration for a volume query (AABB or frustum).
#[derive(Clone, Debug)]
pub struct VolumeQuery {
    /// The query volume type.
    pub volume: VolumeShape,
    /// Policy for unknown-bounds objects.
    pub unknown_bounds: UnknownBoundsPolicy,
    /// Object-kind / layer / visibility filter.
    pub filter: ObjectQueryFilter,
}

impl VolumeQuery {
    /// Create a volume query from an AABB.
    pub fn aabb(aabb: Aabb) -> Self {
        Self {
            volume: VolumeShape::Aabb(aabb),
            unknown_bounds: UnknownBoundsPolicy::default(),
            filter: ObjectQueryFilter::default(),
        }
    }

    /// Create a volume query from a frustum.
    pub fn frustum(frustum: Frustum) -> Self {
        Self {
            volume: VolumeShape::Frustum(frustum),
            unknown_bounds: UnknownBoundsPolicy::default(),
            filter: ObjectQueryFilter::default(),
        }
    }

    /// Set the unknown-bounds policy.
    pub fn with_unknown_bounds(mut self, policy: UnknownBoundsPolicy) -> Self {
        self.unknown_bounds = policy;
        self
    }

    /// Set the object filter.
    pub fn with_filter(mut self, filter: ObjectQueryFilter) -> Self {
        self.filter = filter;
        self
    }
}

/// The shape of a volume query.
#[derive(Clone, Debug)]
pub enum VolumeShape {
    /// An axis-aligned bounding box.
    Aabb(Aabb),
    /// A view frustum.
    Frustum(Frustum),
}

// ── VolumeHit ─────────────────────────────────────────────────────────

/// Result from a volume query.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeHit {
    /// The intersecting object.
    pub object: ObjectId,
    /// Persistent scene identity.
    pub persistent_id: SceneObjectId,
    /// Object kind.
    pub kind: ObjectKind,
    /// Whether the hit came from a known/proxy bound (true) or is
    /// conservative (false).
    pub is_bounded: bool,
}

// ── EditorPickResult ──────────────────────────────────────────────────

/// Result of an editor pick operation.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorPickResult {
    /// The picked object.
    pub object: ObjectId,
    /// The hit metadata (None when the object was picked via
    /// conservative-visible fallback).
    pub hit: Option<AabbRayHit>,
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Validate that a ray is usable for intersection tests.
pub(crate) fn validate_ray(ray: &Ray) -> Result<(), &'static str> {
    if !ray.origin.is_finite() {
        return Err("ray origin is non-finite");
    }
    if !ray.direction.is_finite() {
        return Err("ray direction is non-finite");
    }
    if ray.direction.length_squared() == 0.0 {
        return Err("ray direction is zero");
    }
    Ok(())
}

/// Deterministic sort: IEEE total distance order, then persistent identity,
/// then object kind.
pub(crate) fn sort_ray_hits(hits: &mut [RayHit]) {
    hits.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.persistent_id.cmp(&b.persistent_id))
            .then_with(|| a.kind.cmp(&b.kind))
    });
}

/// Deterministic sort for VolumeHit: SceneObjectId, then ObjectKind.
pub(crate) fn sort_volume_hits(hits: &mut [VolumeHit]) {
    hits.sort_by(|a, b| {
        a.persistent_id
            .cmp(&b.persistent_id)
            .then_with(|| a.kind.cmp(&b.kind))
    });
}
