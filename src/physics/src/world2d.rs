//! 2D physics world behind the `rapier2d` feature gate.
//!
//! Mirrors the 3D [`PhysicsWorld`](crate::PhysicsWorld) API shape:
//! durable engine IDs, transactional registration, contact records,
//! ray casts, and overlap queries — all using 2D Rapier types.

use std::collections::{BTreeMap, BTreeSet};

use rapier2d::na;
use rapier2d::prelude::*;

use crate::{
    validate_positive, BodyKind, BodyMode, PhysicsBodyId, PhysicsColliderId, PhysicsContactKind,
    PhysicsContactPhase, PhysicsContactRecord, PhysicsError,
};

use crate::components2d::{
    BodyDescriptor2D, BodyPose2D, BodyRegistrationRequest2D, ColliderDescriptor2D,
    ColliderReplacementRequest2D, ColliderShape2D, OverlapResult2D, RayHit2D, RayQuery2D,
    RegistrationOutcome2D,
};

use crate::RemovalOutcome;

type PairKey = (PhysicsColliderId, PhysicsColliderId);

/// Velocity/sleep state Rapier cannot retain for fixed or position-based
/// kinematic bodies. Restored if the body becomes dynamic again.
#[derive(Copy, Clone)]
struct ReconfiguredBodyState2D {
    linear_velocity: na::Vector2<f32>,
    angular_velocity: f32,
    sleeping: bool,
}

/// 2D physics world wrapping rapier2d while preserving durable engine-facing IDs.
pub struct PhysicsWorld2D {
    pub(crate) bodies: RigidBodySet,
    pub(crate) colliders: ColliderSet,
    gravity: na::Vector2<f32>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pub(crate) query_pipeline: QueryPipeline,
    body_handles: BTreeMap<PhysicsBodyId, RigidBodyHandle>,
    body_ids: Vec<(RigidBodyHandle, PhysicsBodyId)>,
    collider_handles: BTreeMap<PhysicsColliderId, ColliderHandle>,
    collider_ids: Vec<(ColliderHandle, PhysicsColliderId)>,
    reconfigured_body_states: BTreeMap<PhysicsBodyId, ReconfiguredBodyState2D>,
    active_pairs: BTreeMap<PairKey, PhysicsContactKind>,
    last_contacts: Vec<PhysicsContactRecord>,
}

impl PhysicsWorld2D {
    pub fn new() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            gravity: na::Vector2::new(0.0, -9.81),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            body_handles: BTreeMap::new(),
            body_ids: Vec::new(),
            collider_handles: BTreeMap::new(),
            collider_ids: Vec::new(),
            reconfigured_body_states: BTreeMap::new(),
            active_pairs: BTreeMap::new(),
            last_contacts: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, x: f32, y: f32) {
        self.gravity = na::Vector2::new(x, y);
    }

    // ── Creation ─────────────────────────────────────────────────

    pub fn create_body(
        &mut self,
        descriptor: BodyDescriptor2D,
    ) -> Result<PhysicsBodyId, PhysicsError> {
        validate_vec2("body.translation", descriptor.translation)?;
        if self.body_handles.contains_key(&descriptor.id) {
            return Err(PhysicsError::DuplicateBodyId(descriptor.id));
        }

        let translation = vec2(descriptor.translation);
        let body = match descriptor.kind {
            BodyKind::Static => RigidBodyBuilder::fixed(),
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        }
        .translation(translation)
        .build();

        let id = descriptor.id;
        let handle = self.bodies.insert(body);
        self.body_handles.insert(id.clone(), handle);
        self.body_ids.push((handle, id.clone()));
        Ok(id)
    }

    pub fn create_collider(
        &mut self,
        descriptor: ColliderDescriptor2D,
    ) -> Result<PhysicsColliderId, PhysicsError> {
        validate_vec2("collider.translation", descriptor.translation)?;
        if !descriptor.rotation.is_finite() {
            return Err(PhysicsError::InvalidRotation);
        }
        if self.collider_handles.contains_key(&descriptor.id) {
            return Err(PhysicsError::DuplicateColliderId(descriptor.id));
        }
        let body_handle = *self
            .body_handles
            .get(&descriptor.parent_body)
            .ok_or_else(|| PhysicsError::MissingBody(descriptor.parent_body.clone()))?;

        if matches!(descriptor.shape, ColliderShape2D::TriMesh { .. }) {
            if let Some(body) = self.bodies.get(body_handle) {
                if !body.is_fixed() {
                    return Err(PhysicsError::TrimeshOnDynamicBody);
                }
            }
        }

        let rotation = na::UnitComplex::new(descriptor.rotation);
        let mut builder = shape_builder_2d(descriptor.shape)?
            .sensor(descriptor.is_trigger)
            .position(na::Isometry2::from_parts(
                vec2(descriptor.translation).into(),
                rotation,
            ));
        builder = builder.active_events(ActiveEvents::COLLISION_EVENTS);

        let id = descriptor.id;
        let handle =
            self.colliders
                .insert_with_parent(builder.build(), body_handle, &mut self.bodies);
        self.collider_handles.insert(id.clone(), handle);
        self.collider_ids.push((handle, id.clone()));
        self.query_pipeline.update(&self.colliders);
        Ok(id)
    }

    // ── Pose queries ─────────────────────────────────────────────

    pub fn body_position_by_id(&self, id: &PhysicsBodyId) -> Option<[f32; 2]> {
        self.body_pose_by_id(id).map(|pose| pose.translation)
    }

    pub fn body_pose_by_id(&self, id: &PhysicsBodyId) -> Option<BodyPose2D> {
        let handle = *self.body_handles.get(id)?;
        let body = self.bodies.get(handle)?;
        let translation = body.translation();
        let rotation = body.rotation().angle();
        Some(BodyPose2D {
            translation: [translation.x, translation.y],
            rotation,
        })
    }

    pub fn set_body_position_by_id(
        &mut self,
        id: &PhysicsBodyId,
        translation: [f32; 2],
    ) -> Result<(), PhysicsError> {
        self.set_body_pose_by_id(
            id,
            BodyPose2D {
                translation,
                rotation: 0.0,
            },
        )
    }

    pub fn set_body_pose_by_id(
        &mut self,
        id: &PhysicsBodyId,
        pose: BodyPose2D,
    ) -> Result<(), PhysicsError> {
        validate_vec2("body.translation", pose.translation)?;
        if !pose.rotation.is_finite() {
            return Err(PhysicsError::InvalidRotation);
        }
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        let body = self
            .bodies
            .get_mut(handle)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        let isometry = pose_isometry_2d(&pose);
        if body.is_kinematic() {
            body.set_next_kinematic_position(isometry);
        }
        body.set_position(isometry, true);
        self.query_pipeline.update(&self.colliders);
        Ok(())
    }

    // ── Removal ─────────────────────────────────────────────────

    pub fn remove_collider(&mut self, id: &PhysicsColliderId) -> bool {
        self.remove_collider_with_outcome(id).is_some()
    }

    pub fn remove_collider_with_outcome(
        &mut self,
        id: &PhysicsColliderId,
    ) -> Option<RemovalOutcome> {
        let handle = self.collider_handles.remove(id)?;
        let exited = self.remove_pairs_involving_collider(id);
        self.last_contacts.extend(exited.clone());
        self.last_contacts.sort();
        self.colliders
            .remove(handle, &mut self.island_manager, &mut self.bodies, true);
        self.collider_ids
            .retain(|(candidate, _)| *candidate != handle);
        self.query_pipeline.update(&self.colliders);
        Some(RemovalOutcome {
            removed_body: None,
            removed_colliders: vec![id.clone()],
            exited_pairs: exited,
        })
    }

    pub fn remove_body(&mut self, id: &PhysicsBodyId) -> bool {
        self.remove_body_with_outcome(id).is_some()
    }

    pub fn remove_body_with_outcome(&mut self, id: &PhysicsBodyId) -> Option<RemovalOutcome> {
        let handle = self.body_handles.remove(id)?;
        let attached: Vec<(PhysicsColliderId, ColliderHandle)> = self
            .collider_ids
            .iter()
            .filter_map(|(collider_handle, collider_id)| {
                self.colliders.get(*collider_handle).and_then(|collider| {
                    (collider.parent() == Some(handle))
                        .then(|| (collider_id.clone(), *collider_handle))
                })
            })
            .collect();

        let mut exited_pairs: Vec<PhysicsContactRecord> = Vec::new();
        let mut removed_collider_ids: Vec<PhysicsColliderId> = Vec::new();

        for (collider_id, collider_handle) in &attached {
            let mut exited = self.remove_pairs_involving_collider(collider_id);
            exited_pairs.append(&mut exited);
            self.collider_handles.remove(collider_id);
            self.collider_ids.retain(|(c, _)| *c != *collider_handle);
            removed_collider_ids.push(collider_id.clone());
        }

        exited_pairs.sort();
        removed_collider_ids.sort();
        self.last_contacts.extend(exited_pairs.iter().cloned());
        self.last_contacts.sort();

        self.bodies.remove(
            handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
        self.body_ids.retain(|(candidate, _)| *candidate != handle);
        self.reconfigured_body_states.remove(id);
        self.query_pipeline.update(&self.colliders);

        Some(RemovalOutcome {
            removed_body: Some(id.clone()),
            removed_colliders: removed_collider_ids,
            exited_pairs,
        })
    }

    pub fn collider_exists(&self, id: &PhysicsColliderId) -> bool {
        self.collider_handles.contains_key(id)
    }

    // ── Ray cast ────────────────────────────────────────────────

    pub fn cast_ray(&mut self, query: RayQuery2D) -> Result<Option<RayHit2D>, PhysicsError> {
        validate_vec2("ray.origin", query.origin)?;
        validate_vec2("ray.direction", query.direction)?;
        validate_positive("ray.max_time_of_impact", query.max_time_of_impact)?;
        if vec2(query.direction).norm_squared() == 0.0 {
            return Err(PhysicsError::ZeroDirection);
        }

        self.query_pipeline.update(&self.colliders);
        let ray = Ray::new(point2(query.origin), vec2(query.direction));
        let hit = self.query_pipeline.cast_ray(
            &self.bodies,
            &self.colliders,
            &ray,
            query.max_time_of_impact,
            query.solid,
            QueryFilter::default(),
        );

        Ok(hit.and_then(|(collider_handle, toi)| {
            let collider = self.collider_id_for_handle(collider_handle)?.clone();
            let body_handle = self.colliders.get(collider_handle)?.parent()?;
            let body = self.body_id_for_handle(body_handle)?.clone();
            Some(RayHit2D {
                body,
                collider,
                time_of_impact: toi,
            })
        }))
    }

    // ── Overlap queries ─────────────────────────────────────────

    /// Return every collider whose AABB overlaps the given circle.
    ///
    /// Results are sorted deterministically by collider handle.
    pub fn overlap_circle(
        &mut self,
        center: [f32; 2],
        radius: f32,
    ) -> Result<Vec<OverlapResult2D>, PhysicsError> {
        validate_vec2("overlap_circle.center", center)?;
        validate_positive("overlap_circle.radius", radius)?;

        self.query_pipeline.update(&self.colliders);

        let shape = ColliderBuilder::ball(radius).build();
        let pos = na::Isometry2::translation(center[0], center[1]);

        let mut results: Vec<(ColliderHandle, OverlapResult2D)> = Vec::new();
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
                    results.push((
                        handle,
                        OverlapResult2D {
                            body: body_id,
                            collider: collider_id.clone(),
                        },
                    ));
                }
                true
            },
        );

        results.sort_by_key(|(handle, _)| handle.into_raw_parts());
        Ok(results.into_iter().map(|(_, result)| result).collect())
    }

    /// Return every collider whose AABB overlaps the given AABB.
    ///
    /// Results are sorted deterministically by collider handle.
    pub fn overlap_aabb(
        &mut self,
        min: [f32; 2],
        max: [f32; 2],
    ) -> Result<Vec<OverlapResult2D>, PhysicsError> {
        validate_vec2("overlap_aabb.min", min)?;
        validate_vec2("overlap_aabb.max", max)?;
        for i in 0..2 {
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

        let aabb = rapier2d::geometry::Aabb::new(
            na::Point2::new(min[0], min[1]),
            na::Point2::new(max[0], max[1]),
        );

        let mut results: Vec<(ColliderHandle, OverlapResult2D)> = Vec::new();
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
                    results.push((
                        *handle,
                        OverlapResult2D {
                            body: body_id,
                            collider: collider_id.clone(),
                        },
                    ));
                }
                true
            });

        results.sort_by_key(|(handle, _)| handle.into_raw_parts());
        Ok(results.into_iter().map(|(_, result)| result).collect())
    }

    // ── Simulation step ─────────────────────────────────────────

    pub fn step(&mut self, dt: f32) -> Result<(), PhysicsError> {
        validate_positive("dt", dt).map_err(|_| PhysicsError::NonPositiveDeltaTime)?;
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
        self.last_contacts = self.collect_contact_records();
        Ok(())
    }

    pub fn last_contact_records(&self) -> &[PhysicsContactRecord] {
        &self.last_contacts
    }

    // ── Atomic registration ─────────────────────────────────────

    /// Register one body and zero or more colliders atomically.
    ///
    /// All descriptors are validated and Rapier objects are built before
    /// any state is mutated.  On failure the world is unchanged.
    pub fn register_body(
        &mut self,
        request: BodyRegistrationRequest2D,
    ) -> Result<RegistrationOutcome2D, PhysicsError> {
        let body_id = request.body.id.clone();
        let body_kind = request.body.kind;
        validate_vec2("body.translation", request.body.translation)?;
        if self.body_handles.contains_key(&body_id) {
            return Err(PhysicsError::DuplicateBodyId(body_id));
        }

        let mut request_collider_ids = BTreeSet::new();
        for c in &request.colliders {
            validate_vec2("collider.translation", c.translation)?;
            if !c.rotation.is_finite() {
                return Err(PhysicsError::InvalidRotation);
            }
            if !request_collider_ids.insert(c.id.clone())
                || self.collider_handles.contains_key(&c.id)
            {
                return Err(PhysicsError::DuplicateColliderId(c.id.clone()));
            }
            if c.parent_body != body_id {
                return Err(PhysicsError::MissingBody(c.parent_body.clone()));
            }
            validate_collider_shape_2d(&c.shape, body_kind)?;
        }

        let translation = vec2(request.body.translation);
        let rapier_body = match body_kind {
            BodyKind::Static => RigidBodyBuilder::fixed(),
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        }
        .translation(translation)
        .build();

        let mut rapier_colliders: Vec<(PhysicsColliderId, ColliderBuilder)> = Vec::new();
        for c in &request.colliders {
            let rotation = na::UnitComplex::new(c.rotation);
            let builder = shape_builder_2d(c.shape.clone())?
                .sensor(c.is_trigger)
                .position(na::Isometry2::from_parts(
                    vec2(c.translation).into(),
                    rotation,
                ))
                .active_events(ActiveEvents::COLLISION_EVENTS);
            rapier_colliders.push((c.id.clone(), builder));
        }

        let body_handle = self.bodies.insert(rapier_body);
        self.body_handles.insert(body_id.clone(), body_handle);
        self.body_ids.push((body_handle, body_id.clone()));

        let mut collider_ids = Vec::new();
        for (collider_id, builder) in rapier_colliders {
            let chandle =
                self.colliders
                    .insert_with_parent(builder.build(), body_handle, &mut self.bodies);
            self.collider_handles.insert(collider_id.clone(), chandle);
            self.collider_ids.push((chandle, collider_id.clone()));
            collider_ids.push(collider_id);
        }

        self.query_pipeline.update(&self.colliders);

        Ok(RegistrationOutcome2D {
            body_id,
            collider_ids,
        })
    }

    // ── Body reconfiguration ────────────────────────────────────

    /// Change a body's kind (static/dynamic/kinematic) without recreating it.
    pub fn reconfigure_body_mode(
        &mut self,
        id: &PhysicsBodyId,
        mode: BodyMode,
    ) -> Result<(), PhysicsError> {
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;

        let body = self
            .bodies
            .get(handle)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;

        if body.is_fixed() && mode != BodyMode::Static {
            let has_trimesh = self.collider_ids.iter().any(|(ch, _)| {
                self.colliders
                    .get(*ch)
                    .map_or(false, |c| c.parent() == Some(handle))
                    && matches!(
                        self.colliders.get(*ch).map(|c| c.shape().as_typed_shape()),
                        Some(rapier2d::prelude::TypedShape::TriMesh(_))
                    )
            });
            if has_trimesh {
                return Err(PhysicsError::TrimeshOnDynamicBody);
            }
        }

        let previous =
            self.reconfigured_body_states
                .get(id)
                .copied()
                .unwrap_or(ReconfiguredBodyState2D {
                    linear_velocity: *body.linvel(),
                    angular_velocity: body.angvel(),
                    sleeping: body.is_sleeping(),
                });
        let pose = *body.position();

        let body = self
            .bodies
            .get_mut(handle)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        let target_type = match mode {
            BodyMode::Static => RigidBodyType::Fixed,
            BodyMode::Dynamic => RigidBodyType::Dynamic,
            BodyMode::Kinematic => RigidBodyType::KinematicPositionBased,
        };
        body.set_body_type(target_type, false);
        body.set_position(pose, true);

        if mode == BodyMode::Dynamic {
            body.set_linvel(previous.linear_velocity, false);
            body.set_angvel(previous.angular_velocity, false);
            if previous.sleeping {
                body.sleep();
            } else {
                body.wake_up(true);
            }
            self.reconfigured_body_states.remove(id);
        } else {
            self.reconfigured_body_states.insert(id.clone(), previous);
        }

        Ok(())
    }

    // ── Collider replacement ────────────────────────────────────

    /// Replace an existing collider's shape, sensor flag, and local pose.
    ///
    /// The parent body is unchanged.  Validation runs before the old collider
    /// is removed, so a failure leaves the world intact.
    pub fn replace_collider(
        &mut self,
        request: ColliderReplacementRequest2D,
    ) -> Result<(), PhysicsError> {
        validate_vec2("collider.translation", request.translation)?;
        if !request.rotation.is_finite() {
            return Err(PhysicsError::InvalidRotation);
        }

        let old_handle = *self
            .collider_handles
            .get(&request.collider_id)
            .ok_or_else(|| PhysicsError::MissingCollider(request.collider_id.clone()))?;

        let old_collider = self
            .colliders
            .get(old_handle)
            .ok_or_else(|| PhysicsError::MissingCollider(request.collider_id.clone()))?;

        let body_handle = old_collider
            .parent()
            .ok_or_else(|| PhysicsError::MissingCollider(request.collider_id.clone()))?;

        let body_kind = self
            .bodies
            .get(body_handle)
            .map(body_kind_from_rapier_2d)
            .unwrap_or(BodyKind::Static);
        validate_collider_shape_2d(&request.shape, body_kind)?;

        let rotation = na::UnitComplex::new(request.rotation);
        let builder = shape_builder_2d(request.shape)?
            .sensor(request.is_trigger)
            .position(na::Isometry2::from_parts(
                vec2(request.translation).into(),
                rotation,
            ))
            .active_events(ActiveEvents::COLLISION_EVENTS);

        let exited = self.remove_pairs_involving_collider(&request.collider_id);
        self.last_contacts.extend(exited);

        self.colliders
            .remove(old_handle, &mut self.island_manager, &mut self.bodies, true);

        let new_handle =
            self.colliders
                .insert_with_parent(builder.build(), body_handle, &mut self.bodies);

        self.collider_ids.retain(|(c, _)| *c != old_handle);
        self.collider_ids
            .push((new_handle, request.collider_id.clone()));
        self.collider_handles
            .insert(request.collider_id, new_handle);

        self.query_pipeline.update(&self.colliders);
        Ok(())
    }

    // ── Force / impulse / velocity / teleport ───────────────────

    /// Apply a force at the body's center of mass.
    ///
    /// Static and kinematic bodies are silently ignored.
    pub fn apply_force(&mut self, id: &PhysicsBodyId, force: [f32; 2]) -> Result<(), PhysicsError> {
        validate_vec2("force", force)?;
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.add_force(vec2(force), true);
            }
        }
        Ok(())
    }

    /// Apply an impulse at the body's center of mass.
    ///
    /// Static and kinematic bodies are silently ignored.
    pub fn apply_impulse(
        &mut self,
        id: &PhysicsBodyId,
        impulse: [f32; 2],
    ) -> Result<(), PhysicsError> {
        validate_vec2("impulse", impulse)?;
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.apply_impulse(vec2(impulse), true);
            }
        }
        Ok(())
    }

    /// Apply a torque impulse at the body's center of mass.
    ///
    /// In 2D, torque is a scalar. Static and kinematic bodies are silently ignored.
    pub fn apply_torque_impulse(
        &mut self,
        id: &PhysicsBodyId,
        torque: f32,
    ) -> Result<(), PhysicsError> {
        if !torque.is_finite() {
            return Err(PhysicsError::NonFiniteValue { field: "torque" });
        }
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.apply_torque_impulse(torque, true);
            }
        }
        Ok(())
    }

    /// Wake a sleeping body.
    ///
    /// Static and kinematic bodies are silently ignored.
    pub fn wake_body(&mut self, id: &PhysicsBodyId) -> Result<(), PhysicsError> {
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.wake_up(true);
            }
        }
        Ok(())
    }

    /// Put a body to sleep.
    ///
    /// Static and kinematic bodies are ignored.
    pub fn sleep_body(&mut self, id: &PhysicsBodyId) -> Result<(), PhysicsError> {
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.sleep();
            }
        }
        Ok(())
    }

    /// Set the linear velocity of a body.
    ///
    /// Static and kinematic bodies are silently ignored.
    pub fn set_linear_velocity(
        &mut self,
        id: &PhysicsBodyId,
        velocity: [f32; 2],
    ) -> Result<(), PhysicsError> {
        validate_vec2("velocity", velocity)?;
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.set_linvel(vec2(velocity), true);
            }
        }
        Ok(())
    }

    /// Set the angular velocity of a body.
    ///
    /// Static and kinematic bodies are silently ignored.
    pub fn set_angular_velocity(
        &mut self,
        id: &PhysicsBodyId,
        velocity: f32,
    ) -> Result<(), PhysicsError> {
        if !velocity.is_finite() {
            return Err(PhysicsError::NonFiniteValue { field: "velocity" });
        }
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.set_angvel(velocity, true);
            }
        }
        Ok(())
    }

    /// Teleport a dynamic body to a new pose, preserving velocities.
    ///
    /// Static and kinematic bodies are silently ignored. Invalid poses and
    /// missing IDs still return their normal validation errors.
    pub fn teleport_body(
        &mut self,
        id: &PhysicsBodyId,
        pose: BodyPose2D,
    ) -> Result<(), PhysicsError> {
        validate_vec2("body.translation", pose.translation)?;
        if !pose.rotation.is_finite() {
            return Err(PhysicsError::InvalidRotation);
        }
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        if let Some(body) = self.bodies.get_mut(handle) {
            if body.is_dynamic() {
                body.set_position(pose_isometry_2d(&pose), true);
                self.query_pipeline.update(&self.colliders);
            }
        }
        Ok(())
    }

    // ── Body introspection ──────────────────────────────────────

    pub fn body_is_static(&self, id: &PhysicsBodyId) -> bool {
        self.body_handles
            .get(id)
            .and_then(|h| self.bodies.get(*h))
            .map_or(false, |b| b.is_fixed())
    }

    pub fn body_is_dynamic(&self, id: &PhysicsBodyId) -> bool {
        self.body_handles
            .get(id)
            .and_then(|h| self.bodies.get(*h))
            .map_or(false, |b| b.is_dynamic())
    }

    pub fn body_is_kinematic(&self, id: &PhysicsBodyId) -> bool {
        self.body_handles
            .get(id)
            .and_then(|h| self.bodies.get(*h))
            .map_or(false, |b| b.is_kinematic())
    }

    pub fn body_linear_velocity(&self, id: &PhysicsBodyId) -> Option<[f32; 2]> {
        let handle = *self.body_handles.get(id)?;
        let body = self.bodies.get(handle)?;
        let v = self
            .reconfigured_body_states
            .get(id)
            .filter(|_| !body.is_dynamic())
            .map(|state| state.linear_velocity)
            .unwrap_or(*body.linvel());
        Some([v.x, v.y])
    }

    pub fn body_angular_velocity(&self, id: &PhysicsBodyId) -> Option<f32> {
        let handle = *self.body_handles.get(id)?;
        let body = self.bodies.get(handle)?;
        let v = self
            .reconfigured_body_states
            .get(id)
            .filter(|_| !body.is_dynamic())
            .map(|state| state.angular_velocity)
            .unwrap_or(body.angvel());
        Some(v)
    }

    pub fn body_exists(&self, id: &PhysicsBodyId) -> bool {
        self.body_handles.contains_key(id)
    }

    // ── Internal accessors ──────────────────────────────────────

    #[allow(dead_code)]
    pub(crate) fn body_handle_for(&self, id: &PhysicsBodyId) -> Option<RigidBodyHandle> {
        self.body_handles.get(id).copied()
    }

    #[allow(dead_code)]
    pub(crate) fn collider_handle_for(&self, id: &PhysicsColliderId) -> Option<ColliderHandle> {
        self.collider_handles.get(id).copied()
    }

    #[allow(dead_code)]
    pub(crate) fn collider_parent_body(&self, handle: ColliderHandle) -> Option<RigidBodyHandle> {
        self.colliders.get(handle)?.parent()
    }

    fn body_id_for_handle(&self, handle: RigidBodyHandle) -> Option<&PhysicsBodyId> {
        self.body_ids
            .iter()
            .find_map(|(candidate, id)| (*candidate == handle).then_some(id))
    }

    fn collider_id_for_handle(&self, handle: ColliderHandle) -> Option<&PhysicsColliderId> {
        self.collider_ids
            .iter()
            .find_map(|(candidate, id)| (*candidate == handle).then_some(id))
    }

    // ── Contact collection ──────────────────────────────────────

    fn collect_contact_records(&mut self) -> Vec<PhysicsContactRecord> {
        let mut current = BTreeMap::new();

        for pair in self.narrow_phase.contact_pairs() {
            if pair.has_any_active_contact {
                if let Some(key) = self.pair_key(pair.collider1, pair.collider2) {
                    current.insert(key, PhysicsContactKind::Collision);
                }
            }
        }

        for (a, b, intersecting) in self.narrow_phase.intersection_pairs() {
            if !intersecting {
                continue;
            }
            let Some(a_collider) = self.colliders.get(a) else {
                continue;
            };
            let Some(b_collider) = self.colliders.get(b) else {
                continue;
            };
            if !(a_collider.is_sensor() || b_collider.is_sensor()) {
                continue;
            }
            if let Some(key) = self.trigger_pair_key(a, b, a_collider.is_sensor()) {
                current.insert(key, PhysicsContactKind::Trigger);
            }
        }

        let previous_keys: BTreeSet<_> = self.active_pairs.keys().cloned().collect();
        let current_keys: BTreeSet<_> = current.keys().cloned().collect();
        let mut records = Vec::new();

        for key in current_keys.difference(&previous_keys) {
            records.push(record_2d(PhysicsContactPhase::Enter, current[key], key));
        }
        for key in current_keys.intersection(&previous_keys) {
            records.push(record_2d(PhysicsContactPhase::Stay, current[key], key));
        }
        for key in previous_keys.difference(&current_keys) {
            if let Some(kind) = self.active_pairs.get(key).copied() {
                records.push(record_2d(PhysicsContactPhase::Exit, kind, key));
            }
        }

        records.sort();
        self.active_pairs = current;
        records
    }

    fn pair_key(&self, a: ColliderHandle, b: ColliderHandle) -> Option<PairKey> {
        let a = self.collider_id_for_handle(a)?.clone();
        let b = self.collider_id_for_handle(b)?.clone();
        Some(if a <= b { (a, b) } else { (b, a) })
    }

    fn trigger_pair_key(
        &self,
        a: ColliderHandle,
        b: ColliderHandle,
        a_is_trigger: bool,
    ) -> Option<PairKey> {
        let a = self.collider_id_for_handle(a)?.clone();
        let b = self.collider_id_for_handle(b)?.clone();
        Some(if a_is_trigger { (a, b) } else { (b, a) })
    }

    /// Remove all active pairs that involve `collider_id` and return sorted exit records.
    fn remove_pairs_involving_collider(
        &mut self,
        collider_id: &PhysicsColliderId,
    ) -> Vec<PhysicsContactRecord> {
        let keys_to_remove: Vec<PairKey> = self
            .active_pairs
            .keys()
            .filter(|(a, b)| a == collider_id || b == collider_id)
            .cloned()
            .collect();
        let mut exited = Vec::new();
        for key in keys_to_remove {
            if let Some(kind) = self.active_pairs.remove(&key) {
                exited.push(PhysicsContactRecord {
                    phase: PhysicsContactPhase::Exit,
                    kind,
                    a: key.0.clone(),
                    b: key.1.clone(),
                });
            }
        }
        exited.sort();
        exited
    }
}

impl Default for PhysicsWorld2D {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

fn pose_isometry_2d(pose: &BodyPose2D) -> na::Isometry2<f32> {
    let rotation = na::UnitComplex::new(pose.rotation);
    na::Isometry2::from_parts(
        na::Translation2::new(pose.translation[0], pose.translation[1]),
        rotation,
    )
}

fn record_2d(
    phase: PhysicsContactPhase,
    kind: PhysicsContactKind,
    key: &PairKey,
) -> PhysicsContactRecord {
    PhysicsContactRecord {
        phase,
        kind,
        a: key.0.clone(),
        b: key.1.clone(),
    }
}

pub(crate) fn validate_vec2(field: &'static str, value: [f32; 2]) -> Result<(), PhysicsError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(PhysicsError::NonFiniteValue { field });
    }
    Ok(())
}

pub(crate) fn vec2(value: [f32; 2]) -> na::Vector2<f32> {
    na::Vector2::new(value[0], value[1])
}

fn point2(value: [f32; 2]) -> na::Point2<f32> {
    na::Point2::new(value[0], value[1])
}

fn body_kind_from_rapier_2d(body: &RigidBody) -> BodyKind {
    if body.is_fixed() {
        BodyKind::Static
    } else if body.is_kinematic() {
        BodyKind::Kinematic
    } else {
        BodyKind::Dynamic
    }
}

// ── Shape building and validation ────────────────────────────────────

/// Validates shape geometry and body-kind compatibility without mutating a world.
pub fn validate_collider_shape_2d(
    shape: &ColliderShape2D,
    body_kind: BodyKind,
) -> Result<(), PhysicsError> {
    if matches!(shape, ColliderShape2D::TriMesh { .. }) && body_kind != BodyKind::Static {
        return Err(PhysicsError::TrimeshOnDynamicBody);
    }
    shape_builder_2d(shape.clone()).map(|_| ())
}

pub(crate) fn shape_builder_2d(shape: ColliderShape2D) -> Result<ColliderBuilder, PhysicsError> {
    match shape {
        ColliderShape2D::Cuboid { half_extents } => {
            validate_vec2("cuboid.half_extents", half_extents)?;
            for (i, v) in half_extents.into_iter().enumerate() {
                validate_positive(
                    match i {
                        0 => "cuboid.half_extents.x",
                        _ => "cuboid.half_extents.y",
                    },
                    v,
                )?;
            }
            Ok(ColliderBuilder::cuboid(half_extents[0], half_extents[1]))
        }
        ColliderShape2D::Ball { radius } => {
            validate_positive("ball.radius", radius)?;
            Ok(ColliderBuilder::ball(radius))
        }
        ColliderShape2D::Capsule {
            half_height,
            radius,
        } => {
            validate_positive("capsule.half_height", half_height)?;
            validate_positive("capsule.radius", radius)?;
            Ok(ColliderBuilder::capsule_y(half_height, radius))
        }
        ColliderShape2D::ConvexPolygon { points } => {
            let validated = validate_convex_polygon(&points)?;
            let rapier_points: Vec<na::Point2<f32>> = validated
                .into_iter()
                .map(|v| na::Point2::new(v[0], v[1]))
                .collect();
            ColliderBuilder::convex_hull(&rapier_points).ok_or(PhysicsError::ConvexHullDegenerate)
        }
        ColliderShape2D::TriMesh { vertices, indices } => {
            validate_trimesh_2d(&vertices, &indices)?;
            let points: Vec<na::Point2<f32>> = vertices
                .into_iter()
                .map(|v| na::Point2::new(v[0], v[1]))
                .collect();
            Ok(ColliderBuilder::trimesh(points, indices))
        }
    }
}

fn validate_trimesh_2d(vertices: &[[f32; 2]], indices: &[[u32; 3]]) -> Result<(), PhysicsError> {
    if vertices.is_empty() || indices.is_empty() {
        return Err(PhysicsError::TrimeshEmpty);
    }
    let vertex_count = vertices.len();
    for (i, v) in vertices.iter().enumerate() {
        if v.iter().any(|c| !c.is_finite()) {
            return Err(PhysicsError::TrimeshNonFiniteVertex { index: i });
        }
    }
    for (i, tri) in indices.iter().enumerate() {
        let [a, b, c] = *tri;
        if a >= vertex_count as u32 || b >= vertex_count as u32 || c >= vertex_count as u32 {
            return Err(PhysicsError::TrimeshIndexOutOfBounds {
                index: i,
                vertex_count,
            });
        }
        if a == b || b == c || a == c {
            return Err(PhysicsError::TrimeshDegenerateTriangle { index: i });
        }
    }
    Ok(())
}

fn validate_convex_polygon(points: &[[f32; 2]]) -> Result<Vec<[f32; 2]>, PhysicsError> {
    if points.is_empty() {
        return Err(PhysicsError::ConvexHullEmpty);
    }
    for (i, p) in points.iter().enumerate() {
        if p.iter().any(|c| !c.is_finite()) {
            return Err(PhysicsError::ConvexHullNonFiniteVertex { index: i });
        }
    }
    let unique = dedup_points_2d(points);
    if unique.len() < 3 {
        return Err(PhysicsError::ConvexHullInsufficientPoints {
            unique_count: unique.len(),
        });
    }
    if convex_polygon_is_collinear(&unique) {
        return Err(PhysicsError::ConvexHullDegenerate);
    }
    Ok(unique)
}

fn dedup_points_2d(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    fn key(value: f32) -> u32 {
        if value == 0.0 {
            0
        } else {
            value.to_bits()
        }
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for &p in points {
        if seen.insert((key(p[0]), key(p[1]))) {
            result.push(p);
        }
    }
    result
}

fn convex_polygon_is_collinear(unique: &[[f32; 2]]) -> bool {
    if unique.len() < 3 {
        return true;
    }
    // Cross product of (p1-p0) and (p2-p0) must be non-zero for at least one triple.
    let p0 = unique[0];
    let p1 = unique[1];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    // Find any point that is not collinear with p0-p1.
    for p in &unique[2..] {
        let cross = dx * (p[1] - p0[1]) - dy * (p[0] - p0[0]);
        if cross.abs() > f32::EPSILON {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Creation and gravity ────────────────────────────────────

    #[test]
    fn world_creation() {
        let world = PhysicsWorld2D::new();
        assert!((world.gravity.y + 9.81).abs() < 0.01);
    }

    #[test]
    fn body_falls_under_gravity() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(0.0, -10.0);
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.player",
                BodyKind::Dynamic,
                [0.0, 10.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor2D::new(
                "collider.player",
                body.clone(),
                ColliderShape2D::Ball { radius: 0.5 },
            ))
            .unwrap();
        world.step(1.0).unwrap();
        let pos = world.body_position_by_id(&body).unwrap();
        assert!(pos[1] < 10.0, "body should fall under gravity");
    }

    #[test]
    fn descriptor_validation_rejects_invalid() {
        let mut world = PhysicsWorld2D::new();
        world
            .create_body(BodyDescriptor2D::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();

        let err = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.bad",
                "body.floor",
                ColliderShape2D::Ball { radius: 0.0 },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::NonPositiveDimension {
                field: "ball.radius"
            }
        );

        let err = world
            .create_body(BodyDescriptor2D::new(
                "body.bad",
                BodyKind::Dynamic,
                [0.0, f32::NAN],
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::NonFiniteValue {
                field: "body.translation"
            }
        );
    }

    #[test]
    fn durable_ids_map_to_runtime() {
        let mut world = PhysicsWorld2D::new();
        let body = PhysicsBodyId::new("body.ball");
        let collider = PhysicsColliderId::new("collider.ball");
        assert_eq!(
            world
                .create_body(BodyDescriptor2D::new(
                    body.clone(),
                    BodyKind::Dynamic,
                    [1.0, 2.0],
                ))
                .unwrap(),
            body
        );
        assert_eq!(
            world
                .create_collider(ColliderDescriptor2D::new(
                    collider.clone(),
                    body.clone(),
                    ColliderShape2D::Ball { radius: 0.25 },
                ))
                .unwrap(),
            collider
        );
        assert_eq!(world.body_position_by_id(&body), Some([1.0, 2.0]));
        assert!(world.collider_exists(&collider));
    }

    // ── Ray casts ───────────────────────────────────────────────

    #[test]
    fn ray_query_reports_hit_and_miss() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.target",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.target",
                body.clone(),
                ColliderShape2D::Cuboid {
                    half_extents: [0.5, 0.5],
                },
            ))
            .unwrap();

        let hit = world
            .cast_ray(RayQuery2D::new([0.0, -5.0], [0.0, 1.0], 10.0))
            .unwrap()
            .unwrap();
        assert_eq!(hit.body, body);
        assert_eq!(hit.collider, collider);
        assert!(hit.time_of_impact > 0.0);

        let miss = world
            .cast_ray(RayQuery2D::new([5.0, -5.0], [0.0, 1.0], 10.0))
            .unwrap();
        assert!(miss.is_none());
    }

    // ── Contact records ─────────────────────────────────────────

    #[test]
    fn contact_records_report_enter_and_stay() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(0.0, 0.0);
        world
            .create_body(BodyDescriptor2D::new(
                "body.a",
                BodyKind::Dynamic,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor2D::new(
                "collider.a",
                "body.a",
                ColliderShape2D::Cuboid {
                    half_extents: [0.5, 0.5],
                },
            ))
            .unwrap();
        world
            .create_body(BodyDescriptor2D::new(
                "body.b",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor2D::new(
                "collider.b",
                "body.b",
                ColliderShape2D::Cuboid {
                    half_extents: [0.5, 0.5],
                },
            ))
            .unwrap();

        world.step(1.0 / 60.0).unwrap();
        assert_eq!(
            world.last_contact_records(),
            &[PhysicsContactRecord {
                phase: PhysicsContactPhase::Enter,
                kind: PhysicsContactKind::Collision,
                a: PhysicsColliderId::new("collider.a"),
                b: PhysicsColliderId::new("collider.b"),
            }]
        );

        world.step(1.0 / 60.0).unwrap();
        assert_eq!(
            world.last_contact_records()[0].phase,
            PhysicsContactPhase::Stay
        );
    }

    #[test]
    fn trigger_records_are_separate() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(0.0, 0.0);
        world
            .create_body(BodyDescriptor2D::new(
                "body.trigger",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(
                ColliderDescriptor2D::new(
                    "collider.trigger",
                    "body.trigger",
                    ColliderShape2D::Ball { radius: 1.0 },
                )
                .trigger(true),
            )
            .unwrap();
        world
            .create_body(BodyDescriptor2D::new(
                "body.other",
                BodyKind::Dynamic,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor2D::new(
                "collider.other",
                "body.other",
                ColliderShape2D::Capsule {
                    half_height: 0.5,
                    radius: 0.25,
                },
            ))
            .unwrap();

        world.step(1.0 / 60.0).unwrap();
        assert_eq!(world.last_contact_records().len(), 1);
        assert_eq!(
            world.last_contact_records()[0].kind,
            PhysicsContactKind::Trigger
        );
    }

    // ── Removal ─────────────────────────────────────────────────

    #[test]
    fn remove_body_removes_colliders_and_ids() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.remove",
                BodyKind::Dynamic,
                [1.0, 2.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.remove",
                body.clone(),
                ColliderShape2D::Ball { radius: 1.0 },
            ))
            .unwrap();
        assert!(world.body_exists(&body));
        assert!(world.collider_exists(&collider));
        assert!(world.remove_body(&body));
        assert!(!world.body_exists(&body));
        assert!(!world.collider_exists(&collider));
        assert!(!world.remove_body(&body)); // idempotent
    }

    // ── Atomic registration ─────────────────────────────────────

    #[test]
    fn register_body_atomic_success() {
        let mut world = PhysicsWorld2D::new();
        let outcome = world
            .register_body(BodyRegistrationRequest2D {
                body: BodyDescriptor2D::new("body.reg", BodyKind::Dynamic, [1.0, 2.0]),
                colliders: vec![ColliderDescriptor2D::new(
                    "collider.reg",
                    "body.reg",
                    ColliderShape2D::Ball { radius: 0.5 },
                )],
            })
            .unwrap();
        assert_eq!(outcome.body_id, PhysicsBodyId::new("body.reg"));
        assert_eq!(outcome.collider_ids.len(), 1);
        assert!(world.body_exists(&outcome.body_id));
    }

    #[test]
    fn register_body_atomic_failure_rolls_back() {
        let mut world = PhysicsWorld2D::new();
        let result = world.register_body(BodyRegistrationRequest2D {
            body: BodyDescriptor2D::new("body.reg", BodyKind::Dynamic, [1.0, 2.0]),
            colliders: vec![ColliderDescriptor2D::new(
                "collider.reg",
                "body.wrong", // wrong parent
                ColliderShape2D::Ball { radius: 0.5 },
            )],
        });
        assert!(result.is_err());
        assert!(!world.body_exists(&PhysicsBodyId::new("body.reg")));
        assert!(!world.collider_exists(&PhysicsColliderId::new("collider.reg")));
    }

    // ── Body reconfiguration ────────────────────────────────────

    #[test]
    fn reconfigure_body_mode_preserves_pose() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.dyn",
                BodyKind::Dynamic,
                [3.0, 4.0],
            ))
            .unwrap();
        world
            .reconfigure_body_mode(&body, BodyMode::Kinematic)
            .unwrap();
        assert!(world.body_is_kinematic(&body));
        assert_eq!(world.body_position_by_id(&body), Some([3.0, 4.0]));
    }

    // ── Force / velocity / teleport ─────────────────────────────

    #[test]
    fn apply_force_and_read_velocity() {
        let mut world = PhysicsWorld2D::new();
        world.set_gravity(0.0, 0.0);
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.dyn",
                BodyKind::Dynamic,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor2D::new(
                "collider.dyn",
                body.clone(),
                ColliderShape2D::Ball { radius: 0.5 },
            ))
            .unwrap();

        world.apply_force(&body, [10.0, 0.0]).unwrap();
        world.step(1.0 / 60.0).unwrap();

        let vel = world.body_linear_velocity(&body).unwrap();
        assert!(vel[0] > 0.0, "force should accelerate body");
    }

    #[test]
    fn teleport_body_moves_dynamic() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.dyn",
                BodyKind::Dynamic,
                [0.0, 0.0],
            ))
            .unwrap();
        world
            .teleport_body(&body, BodyPose2D::from_translation([10.0, 20.0]))
            .unwrap();
        assert_eq!(world.body_position_by_id(&body), Some([10.0, 20.0]));
    }

    // ── Overlap queries ─────────────────────────────────────────

    #[test]
    fn overlap_circle_finds_collider() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.s",
                body.clone(),
                ColliderShape2D::Ball { radius: 0.5 },
            ))
            .unwrap();

        let results = world.overlap_circle([0.0, 0.0], 2.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].collider, collider);

        let far = world.overlap_circle([10.0, 10.0], 1.0).unwrap();
        assert!(far.is_empty());
    }

    #[test]
    fn overlap_aabb_finds_collider() {
        let mut world = PhysicsWorld2D::new();
        let body = world
            .create_body(BodyDescriptor2D::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.s",
                body.clone(),
                ColliderShape2D::Cuboid {
                    half_extents: [1.0, 1.0],
                },
            ))
            .unwrap();

        let results = world.overlap_aabb([-2.0, -2.0], [2.0, 2.0]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].collider, collider);

        let far = world.overlap_aabb([5.0, 5.0], [6.0, 6.0]).unwrap();
        assert!(far.is_empty());
    }

    // ── Convex polygon validation ───────────────────────────────

    #[test]
    fn convex_polygon_valid() {
        let mut world = PhysicsWorld2D::new();
        world
            .create_body(BodyDescriptor2D::new(
                "body.poly",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor2D::new(
            "collider.poly",
            "body.poly",
            ColliderShape2D::ConvexPolygon {
                points: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            },
        ));
        assert!(result.is_ok());
    }

    #[test]
    fn convex_polygon_collinear_rejected() {
        let mut world = PhysicsWorld2D::new();
        world
            .create_body(BodyDescriptor2D::new(
                "body.poly",
                BodyKind::Static,
                [0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor2D::new(
                "collider.poly",
                "body.poly",
                ColliderShape2D::ConvexPolygon {
                    points: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullDegenerate);
    }
}
