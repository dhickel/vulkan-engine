//! 2D physics types behind the `rapier2d` feature gate.
//!
//! These mirror the 3D descriptor and shape types in the parent module
//! but use `[f32; 2]` for translations, `f32` for rotation angles, and
//! 2D-specific collider shapes.

use crate::{BodyKind, PhysicsBodyId, PhysicsColliderId};

// ── 2D body descriptor ──────────────────────────────────────────────

/// Descriptor for creating a 2D rigid body.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyDescriptor2D {
    pub id: PhysicsBodyId,
    pub kind: BodyKind,
    pub translation: [f32; 2],
}

impl BodyDescriptor2D {
    pub fn new(id: impl Into<PhysicsBodyId>, kind: BodyKind, translation: [f32; 2]) -> Self {
        Self {
            id: id.into(),
            kind,
            translation,
        }
    }
}

// ── 2D body pose ────────────────────────────────────────────────────

/// Position and rotation of a 2D body.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BodyPose2D {
    pub translation: [f32; 2],
    /// Rotation angle in radians.
    pub rotation: f32,
}

impl BodyPose2D {
    pub fn from_translation(translation: [f32; 2]) -> Self {
        Self {
            translation,
            rotation: 0.0,
        }
    }
}

// ── 2D collider shapes ──────────────────────────────────────────────

/// Collider shape variants for 2D physics.
#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShape2D {
    /// Axis-aligned rectangle.
    Cuboid { half_extents: [f32; 2] },
    /// Circle.
    Ball { radius: f32 },
    /// Capsule aligned with the Y axis.
    Capsule { half_height: f32, radius: f32 },
    /// Convex polygon from an ordered set of points.
    ConvexPolygon { points: Vec<[f32; 2]> },
    /// Static triangle mesh (triangulated 2D mesh, only on static bodies).
    TriMesh {
        vertices: Vec<[f32; 2]>,
        indices: Vec<[u32; 3]>,
    },
}

// ── 2D collider descriptor ──────────────────────────────────────────

/// Descriptor for creating a 2D collider.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderDescriptor2D {
    pub id: PhysicsColliderId,
    pub parent_body: PhysicsBodyId,
    pub shape: ColliderShape2D,
    pub is_trigger: bool,
    pub translation: [f32; 2],
    /// Rotation angle in radians.
    pub rotation: f32,
}

impl ColliderDescriptor2D {
    pub fn new(
        id: impl Into<PhysicsColliderId>,
        parent_body: impl Into<PhysicsBodyId>,
        shape: ColliderShape2D,
    ) -> Self {
        Self {
            id: id.into(),
            parent_body: parent_body.into(),
            shape,
            is_trigger: false,
            translation: [0.0; 2],
            rotation: 0.0,
        }
    }

    pub fn trigger(mut self, is_trigger: bool) -> Self {
        self.is_trigger = is_trigger;
        self
    }

    pub fn translation(mut self, translation: [f32; 2]) -> Self {
        self.translation = translation;
        self
    }

    pub fn rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }
}

// ── 2D ray query / hit ──────────────────────────────────────────────

/// Ray-cast query in 2D.
#[derive(Clone, Debug, PartialEq)]
pub struct RayQuery2D {
    pub origin: [f32; 2],
    pub direction: [f32; 2],
    pub max_time_of_impact: f32,
    pub solid: bool,
}

impl RayQuery2D {
    pub fn new(origin: [f32; 2], direction: [f32; 2], max_time_of_impact: f32) -> Self {
        Self {
            origin,
            direction,
            max_time_of_impact,
            solid: true,
        }
    }
}

/// Result of a 2D ray cast.
#[derive(Clone, Debug, PartialEq)]
pub struct RayHit2D {
    pub body: PhysicsBodyId,
    pub collider: PhysicsColliderId,
    pub time_of_impact: f32,
}

// ── Registration request / outcome ──────────────────────────────────

/// Atomic request: register one 2D body and zero or more colliders.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyRegistrationRequest2D {
    pub body: BodyDescriptor2D,
    pub colliders: Vec<ColliderDescriptor2D>,
}

/// Outcome of a successful [`PhysicsWorld2D::register_body`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistrationOutcome2D {
    pub body_id: PhysicsBodyId,
    pub collider_ids: Vec<PhysicsColliderId>,
}

// ── Collider replacement request ────────────────────────────────────

/// Request to replace an existing 2D collider's shape and properties.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderReplacementRequest2D {
    pub collider_id: PhysicsColliderId,
    pub shape: ColliderShape2D,
    pub is_trigger: bool,
    pub translation: [f32; 2],
    pub rotation: f32,
}

// ── Overlap result ──────────────────────────────────────────────────

/// Result of a 2D overlap query.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OverlapResult2D {
    pub body: PhysicsBodyId,
    pub collider: PhysicsColliderId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_descriptor_2d_roundtrip() {
        let d = BodyDescriptor2D::new("body.a", BodyKind::Dynamic, [1.0, 2.0]);
        assert_eq!(d.id, PhysicsBodyId::new("body.a"));
        assert_eq!(d.kind, BodyKind::Dynamic);
        assert_eq!(d.translation, [1.0, 2.0]);
    }

    #[test]
    fn collider_descriptor_2d_builder() {
        let d = ColliderDescriptor2D::new(
            "collider.a",
            "body.a",
            ColliderShape2D::Ball { radius: 0.5 },
        )
        .trigger(true)
        .translation([1.0, 0.0])
        .rotation(std::f32::consts::PI);
        assert!(d.is_trigger);
        assert_eq!(d.translation, [1.0, 0.0]);
        assert_eq!(d.rotation, std::f32::consts::PI);
    }

    #[test]
    fn body_pose_2d_default_rotation_is_zero() {
        let pose = BodyPose2D::from_translation([3.0, 4.0]);
        assert_eq!(pose.translation, [3.0, 4.0]);
        assert_eq!(pose.rotation, 0.0);
    }

    #[test]
    fn ray_query_2d_construction() {
        let q = RayQuery2D::new([0.0, 0.0], [1.0, 0.0], 10.0);
        assert_eq!(q.direction, [1.0, 0.0]);
        assert!(q.solid);
    }
}
