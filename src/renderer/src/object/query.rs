//! Scene query DTOs for raycasts, volume queries, and editor picking.
//!
//! These types are pure data — the Scene API consumes them and returns
//! matching results without mutating the scene.

use crate::data::camera::{Aabb, AabbRayHit, Frustum, Ray};
use crate::object::identity::ObjectId;
use engine_events::{ObjectKind, SceneObjectId};
use glam::Vec3;

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
    /// World-space surface normal at the hit point.
    pub normal: Vec3,
    /// Whether the hit came from a proxy AABB rather than known geometry.
    pub is_proxy: bool,
}

// ── Query options ─────────────────────────────────────────────────────

/// Policy for objects whose bounds are unknown or conservative-visible.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnknownBoundsPolicy {
    /// Exclude unknown-bounds objects from results (safe default).
    Exclude,
    /// Include unknown-bounds objects as "conservative hits" with
    /// distance = f32::INFINITY. These sort after all known hits but
    /// before nothing-found.
    IncludeAsInfinite,
}

impl Default for UnknownBoundsPolicy {
    fn default() -> Self {
        Self::Exclude
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

/// Configuration for a volume query (AABB or frustum).
#[derive(Clone, Debug)]
pub struct VolumeQuery {
    /// The query volume type.
    pub volume: VolumeShape,
    /// Policy for unknown-bounds objects.
    pub unknown_bounds: UnknownBoundsPolicy,
}

impl VolumeQuery {
    /// Create a volume query from an AABB.
    pub fn aabb(aabb: Aabb) -> Self {
        Self {
            volume: VolumeShape::Aabb(aabb),
            unknown_bounds: UnknownBoundsPolicy::default(),
        }
    }

    /// Create a volume query from a frustum.
    pub fn frustum(frustum: Frustum) -> Self {
        Self {
            volume: VolumeShape::Frustum(frustum),
            unknown_bounds: UnknownBoundsPolicy::default(),
        }
    }

    /// Set the unknown-bounds policy.
    pub fn with_unknown_bounds(mut self, policy: UnknownBoundsPolicy) -> Self {
        self.unknown_bounds = policy;
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
    let len_sq = ray.direction.length_squared();
    if len_sq < 1e-10 {
        return Err("ray direction is degenerate (near-zero length)");
    }
    Ok(())
}

/// Deterministic sort: distance (finite f32 via to_bits), then
/// SceneObjectId, then ObjectKind.
pub(crate) fn sort_ray_hits(hits: &mut [RayHit]) {
    hits.sort_by(|a, b| {
        a.distance
            .to_bits()
            .cmp(&b.distance.to_bits())
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
