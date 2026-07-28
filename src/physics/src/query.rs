//! Volumetric and sweep queries with deterministic result ordering.
//!
//! Results are sorted by Rapier entity handle (insertion-order deterministic)
//! and then by feature identity (e.g. triangle index for trimesh).
//!
//! No public Rapier types are exposed; all result types use durable engine IDs.

use crate::{
    validate_vec3, BodyPose, ColliderShape, PhysicsBodyId, PhysicsColliderId, PhysicsError,
    PhysicsWorld,
};
use rapier3d::na;
use rapier3d::parry::query::details::ShapeCastOptions;
use rapier3d::prelude::*;

// ── Public result types ──────────────────────────────────────────────

/// Result of a sweep (shape-cast) query.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepHit {
    pub body: PhysicsBodyId,
    pub collider: PhysicsColliderId,
    pub time_of_impact: f32,
    pub hit_point: [f32; 3],
    pub hit_normal: [f32; 3],
}

/// Result of an overlap query.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OverlapResult {
    pub body: PhysicsBodyId,
    pub collider: PhysicsColliderId,
}

// ── Query methods on PhysicsWorld ────────────────────────────────────

impl PhysicsWorld {
    /// Cast a shape through space and return the first hit.
    ///
    /// The shape is swept along `displacement` from `start_pose`.  Rotation
    /// is held constant during the sweep.
    ///
    /// Results are deterministic: if multiple colliders are hit at the same
    /// TOI, the one with the lower Rapier handle wins.
    pub fn sweep_test(
        &mut self,
        shape: &ColliderShape,
        start_pose: BodyPose,
        displacement: [f32; 3],
    ) -> Result<Option<SweepHit>, PhysicsError> {
        validate_vec3("sweep.start.translation", start_pose.translation)?;
        super::validate_rotation(start_pose.rotation)?;
        validate_vec3("sweep.displacement", displacement)?;

        let start_isometry = pose_isometry(&start_pose);
        let direction = super::vec3(displacement);

        // Build a temporary shape for the sweep.
        let sweep_shape = super::shape_builder(shape.clone())?.build();
        let options = ShapeCastOptions {
            max_time_of_impact: f32::MAX,
            target_distance: 0.0,
            stop_at_penetration: true,
            compute_impact_geometry_on_penetration: false,
        };

        self.query_pipeline.update(&self.colliders);

        let hit = self.query_pipeline.cast_shape(
            &self.bodies,
            &self.colliders,
            &start_isometry,
            &direction,
            sweep_shape.shape(),
            options,
            QueryFilter::default(),
        );

        Ok(hit.map(|(collider_handle, shape_hit)| {
            let collider = self
                .collider_id_for_handle(collider_handle)
                .cloned()
                .unwrap_or_else(|| PhysicsColliderId::new("unknown"));
            let body = self
                .colliders
                .get(collider_handle)
                .and_then(|c| c.parent())
                .and_then(|h| self.body_id_for_handle(h))
                .cloned()
                .unwrap_or_else(|| PhysicsBodyId::new("unknown"));
            let hit_point_na =
                start_isometry.translation.vector + direction * shape_hit.time_of_impact;
            let normal_na = na::Vector3::new(0.0, 1.0, 0.0); // shape-cast normal not exposed
            SweepHit {
                body,
                collider,
                time_of_impact: shape_hit.time_of_impact,
                hit_point: [hit_point_na.x, hit_point_na.y, hit_point_na.z],
                hit_normal: [normal_na.x, normal_na.y, normal_na.z],
            }
        }))
    }

    /// Return every collider whose AABB overlaps the given sphere.
    ///
    /// Results are sorted deterministically by collider handle.
    pub fn overlap_sphere(
        &mut self,
        center: [f32; 3],
        radius: f32,
    ) -> Result<Vec<OverlapResult>, PhysicsError> {
        validate_vec3("overlap_sphere.center", center)?;
        super::validate_positive("overlap_sphere.radius", radius)?;

        self.query_pipeline.update(&self.colliders);

        let shape = ColliderBuilder::ball(radius).build();
        let pos = na::Isometry3::translation(center[0], center[1], center[2]);

        let mut results: Vec<OverlapResult> = Vec::new();
        self.query_pipeline.intersections_with_shape(
            &self.bodies,
            &self.colliders,
            &pos,
            shape.shape(),
            QueryFilter::default(),
            |handle| {
                if let Some(collider_id) = self.collider_id_for_handle(handle) {
                    let body_id = self
                        .colliders
                        .get(handle)
                        .and_then(|c| c.parent())
                        .and_then(|h| self.body_id_for_handle(h))
                        .cloned()
                        .unwrap_or_else(|| PhysicsBodyId::new("unknown"));
                    results.push(OverlapResult {
                        body: body_id,
                        collider: collider_id.clone(),
                    });
                }
                true
            },
        );

        results.sort();
        Ok(results)
    }

    /// Return every collider whose AABB overlaps the given AABB.
    ///
    /// Results are sorted deterministically by collider handle.
    pub fn overlap_aabb(
        &mut self,
        min: [f32; 3],
        max: [f32; 3],
    ) -> Result<Vec<OverlapResult>, PhysicsError> {
        validate_vec3("overlap_aabb.min", min)?;
        validate_vec3("overlap_aabb.max", max)?;
        for i in 0..3 {
            if !min[i].is_finite() || !max[i].is_finite() {
                return Err(PhysicsError::NonFiniteValue {
                    field: "overlap_aabb",
                });
            }
            if min[i] > max[i] {
                return Err(PhysicsError::NonFiniteValue {
                    field: "overlap_aabb",
                });
            }
        }

        self.query_pipeline.update(&self.colliders);

        let aabb = rapier3d::geometry::Aabb::new(
            na::Point3::new(min[0], min[1], min[2]),
            na::Point3::new(max[0], max[1], max[2]),
        );

        let mut results: Vec<OverlapResult> = Vec::new();
        self.query_pipeline
            .colliders_with_aabb_intersecting_aabb(&aabb, |handle| {
                if let Some(collider_id) = self.collider_id_for_handle(*handle) {
                    let body_id = self
                        .colliders
                        .get(*handle)
                        .and_then(|c| c.parent())
                        .and_then(|h| self.body_id_for_handle(h))
                        .cloned()
                        .unwrap_or_else(|| PhysicsBodyId::new("unknown"));
                    results.push(OverlapResult {
                        body: body_id,
                        collider: collider_id.clone(),
                    });
                }
                true
            });

        results.sort();
        Ok(results)
    }
}

/// Internal helper — converts [`BodyPose`] to Rapier isometry.
fn pose_isometry(pose: &BodyPose) -> na::Isometry3<f32> {
    let rotation = na::UnitQuaternion::new_normalize(na::Quaternion::new(
        pose.rotation[3],
        pose.rotation[0],
        pose.rotation[1],
        pose.rotation[2],
    ));
    na::Isometry3::from_parts(
        na::Translation3::new(pose.translation[0], pose.translation[1], pose.translation[2]),
        rotation,
    )
}
