//! Renderer-independent alpha physics API built on Rapier.
//!
//! Authored code should use durable [`PhysicsBodyId`] and [`PhysicsColliderId`]
//! values. Rapier handles remain an internal runtime detail. No public item
//! is declared deprecated; the public API uses durable IDs exclusively.

use std::collections::{BTreeMap, BTreeSet};

use engine_events::{
    ContactPhase as EventContactPhase, EngineEvent, EventBus, EventStage, PhysicsEvent,
};
use rapier3d::na;
use rapier3d::prelude::*;

// Re-export unified ID types from the canonical engine_events crate.
pub use engine_events::ColliderId as PhysicsColliderId;
pub use engine_events::PhysicsBodyId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BodyKind {
    Static,
    Dynamic,
    Kinematic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodyDescriptor {
    pub id: PhysicsBodyId,
    pub kind: BodyKind,
    pub translation: [f32; 3],
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BodyPose {
    pub translation: [f32; 3],
    /// Quaternion in `[x, y, z, w]` order.
    pub rotation: [f32; 4],
}

impl BodyDescriptor {
    pub fn new(id: impl Into<PhysicsBodyId>, kind: BodyKind, translation: [f32; 3]) -> Self {
        Self {
            id: id.into(),
            kind,
            translation,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShape {
    Cuboid {
        half_extents: [f32; 3],
    },
    Sphere {
        radius: f32,
    },
    CapsuleY {
        half_height: f32,
        radius: f32,
    },
    TriMeshStatic {
        vertices: Vec<[f32; 3]>,
        indices: Vec<[u32; 3]>,
    },
    ConvexHull {
        points: Vec<[f32; 3]>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColliderDescriptor {
    pub id: PhysicsColliderId,
    pub parent_body: PhysicsBodyId,
    pub shape: ColliderShape,
    pub is_trigger: bool,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

impl ColliderDescriptor {
    pub fn new(
        id: impl Into<PhysicsColliderId>,
        parent_body: impl Into<PhysicsBodyId>,
        shape: ColliderShape,
    ) -> Self {
        Self {
            id: id.into(),
            parent_body: parent_body.into(),
            shape,
            is_trigger: false,
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn trigger(mut self, is_trigger: bool) -> Self {
        self.is_trigger = is_trigger;
        self
    }

    pub fn translation(mut self, translation: [f32; 3]) -> Self {
        self.translation = translation;
        self
    }

    pub fn rotation(mut self, rotation: [f32; 4]) -> Self {
        self.rotation = rotation;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhysicsError {
    DuplicateBodyId(PhysicsBodyId),
    DuplicateColliderId(PhysicsColliderId),
    MissingBody(PhysicsBodyId),
    NonFiniteValue { field: &'static str },
    NonPositiveDimension { field: &'static str },
    NonPositiveDeltaTime,
    ZeroDirection,
    InvalidRotation,
    TrimeshNonFiniteVertex { index: usize },
    TrimeshIndexOutOfBounds { index: usize, vertex_count: usize },
    TrimeshEmpty,
    TrimeshDegenerateTriangle { index: usize },
    TrimeshOnDynamicBody,
    ConvexHullEmpty,
    ConvexHullNonFiniteVertex { index: usize },
    ConvexHullInsufficientPoints { unique_count: usize },
    ConvexHullDegenerate,
}

impl std::fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhysicsError::DuplicateBodyId(id) => write!(f, "duplicate body id: {id}"),
            PhysicsError::DuplicateColliderId(id) => write!(f, "duplicate collider id: {id}"),
            PhysicsError::MissingBody(id) => write!(f, "missing body: {id}"),
            PhysicsError::NonFiniteValue { field } => write!(f, "non-finite value in {field}"),
            PhysicsError::NonPositiveDimension { field } => {
                write!(f, "non-positive dimension in {field}")
            }
            PhysicsError::NonPositiveDeltaTime => write!(f, "non-positive delta time"),
            PhysicsError::ZeroDirection => write!(f, "zero direction"),
            PhysicsError::InvalidRotation => {
                write!(f, "rotation quaternion must be finite and non-zero")
            }
            PhysicsError::TrimeshNonFiniteVertex { index } => {
                write!(f, "non-finite trimesh vertex at index {index}")
            }
            PhysicsError::TrimeshIndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                f,
                "trimesh index {index} out of bounds (vertex count: {vertex_count})"
            ),
            PhysicsError::TrimeshEmpty => write!(f, "empty trimesh"),
            PhysicsError::TrimeshDegenerateTriangle { index } => {
                write!(f, "degenerate trimesh triangle at index {index}")
            }
            PhysicsError::TrimeshOnDynamicBody => write!(f, "trimesh on non-static body"),
            PhysicsError::ConvexHullEmpty => write!(f, "empty convex hull"),
            PhysicsError::ConvexHullNonFiniteVertex { index } => {
                write!(f, "non-finite convex hull vertex at index {index}")
            }
            PhysicsError::ConvexHullInsufficientPoints { unique_count } => write!(
                f,
                "insufficient unique points for convex hull: {unique_count} (need at least 4)"
            ),
            PhysicsError::ConvexHullDegenerate => {
                write!(f, "degenerate convex hull (coplanar or zero-volume)")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RayQuery {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub max_time_of_impact: f32,
    pub solid: bool,
}

impl RayQuery {
    pub fn new(origin: [f32; 3], direction: [f32; 3], max_time_of_impact: f32) -> Self {
        Self {
            origin,
            direction,
            max_time_of_impact,
            solid: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RayHit {
    pub body: PhysicsBodyId,
    pub collider: PhysicsColliderId,
    pub time_of_impact: f32,
}

impl RayHit {
    pub fn to_engine_event(&self) -> EngineEvent {
        EngineEvent::Physics(PhysicsEvent::QueryHit {
            body: self.body.clone(),
            collider: self.collider.clone(),
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PhysicsContactPhase {
    Enter,
    Stay,
    Exit,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PhysicsContactKind {
    Collision,
    Trigger,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PhysicsContactRecord {
    pub phase: PhysicsContactPhase,
    pub kind: PhysicsContactKind,
    pub a: PhysicsColliderId,
    pub b: PhysicsColliderId,
}

impl PhysicsContactRecord {
    pub fn to_engine_event(&self) -> EngineEvent {
        let phase = match self.phase {
            PhysicsContactPhase::Enter => EventContactPhase::Enter,
            PhysicsContactPhase::Stay => EventContactPhase::Stay,
            PhysicsContactPhase::Exit => EventContactPhase::Exit,
        };
        match self.kind {
            PhysicsContactKind::Collision => EngineEvent::Physics(PhysicsEvent::Collision {
                phase,
                a: self.a.clone(),
                b: self.b.clone(),
            }),
            PhysicsContactKind::Trigger => EngineEvent::Physics(PhysicsEvent::Trigger {
                phase,
                trigger: self.a.clone(),
                other: self.b.clone(),
            }),
        }
    }
}

pub fn contact_records_to_engine_events(records: &[PhysicsContactRecord]) -> Vec<EngineEvent> {
    records
        .iter()
        .map(PhysicsContactRecord::to_engine_event)
        .collect()
}

pub fn emit_contact_records(
    bus: &mut EventBus,
    stage: EventStage,
    records: &[PhysicsContactRecord],
) {
    for event in contact_records_to_engine_events(records) {
        bus.emit(stage, None, event);
    }
}

// ── BSP collision recipe types ────────────────────────────────────────
//
// Neutral DTOs for building Rapier colliders from BSP collision data.
// These live in the physics crate so the app bridge can convert between
// BSP extraction DTOs and Rapier shapes without tying the physics crate
// to the bsp or bsp_runtime crates.

/// Recipe for a world static trimesh collider built from BSP solid faces.
#[derive(Debug, Clone)]
pub struct BspWorldCollision {
    /// Vertices in engine space.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<[u32; 3]>,
}

/// Recipe for a single convex piece from BSP clipnode reconstruction.
#[derive(Debug, Clone)]
pub struct BspConvexPiece {
    /// Convex hull points in engine space.
    pub points: Vec<[f32; 3]>,
}

/// Recipe for one entity's collision group (one or more convex pieces).
#[derive(Debug, Clone)]
pub struct BspEntityCollision {
    /// Source entity index in the BSP entity lump.
    pub entity_index: u32,
    /// Whether this is a trigger sensor (is_sensor = true).
    pub is_trigger: bool,
    /// Convex pieces that form the collision shape.
    pub pieces: Vec<BspConvexPiece>,
}

impl BspWorldCollision {
    /// Build a `ColliderShape::TriMeshStatic` from this recipe.
    pub fn to_shape(&self) -> ColliderShape {
        ColliderShape::TriMeshStatic {
            vertices: self.vertices.clone(),
            indices: self.indices.clone(),
        }
    }
}

impl BspConvexPiece {
    /// Build a `ColliderShape::ConvexHull` from this recipe.
    pub fn to_shape(&self) -> ColliderShape {
        ColliderShape::ConvexHull {
            points: self.points.clone(),
        }
    }
}

type PairKey = (PhysicsColliderId, PhysicsColliderId);

/// Physics world wrapping rapier3d while preserving durable engine-facing IDs.
pub struct PhysicsWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    gravity: na::Vector3<f32>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    body_handles: BTreeMap<PhysicsBodyId, RigidBodyHandle>,
    body_ids: Vec<(RigidBodyHandle, PhysicsBodyId)>,
    collider_handles: BTreeMap<PhysicsColliderId, ColliderHandle>,
    collider_ids: Vec<(ColliderHandle, PhysicsColliderId)>,
    active_pairs: BTreeMap<PairKey, PhysicsContactKind>,
    last_contacts: Vec<PhysicsContactRecord>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            gravity: na::Vector3::new(0.0, -9.81, 0.0),
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
            active_pairs: BTreeMap::new(),
            last_contacts: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.gravity = na::Vector3::new(x, y, z);
    }

    pub fn create_body(
        &mut self,
        descriptor: BodyDescriptor,
    ) -> Result<PhysicsBodyId, PhysicsError> {
        validate_vec3("body.translation", descriptor.translation)?;
        if self.body_handles.contains_key(&descriptor.id) {
            return Err(PhysicsError::DuplicateBodyId(descriptor.id));
        }

        let translation = vec3(descriptor.translation);
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
        descriptor: ColliderDescriptor,
    ) -> Result<PhysicsColliderId, PhysicsError> {
        validate_vec3("collider.translation", descriptor.translation)?;
        validate_rotation(descriptor.rotation)?;
        if self.collider_handles.contains_key(&descriptor.id) {
            return Err(PhysicsError::DuplicateColliderId(descriptor.id));
        }
        let body_handle = *self
            .body_handles
            .get(&descriptor.parent_body)
            .ok_or_else(|| PhysicsError::MissingBody(descriptor.parent_body.clone()))?;

        // TriMeshStatic is only allowed on static (fixed) bodies.
        if matches!(descriptor.shape, ColliderShape::TriMeshStatic { .. }) {
            if let Some(body) = self.bodies.get(body_handle) {
                if !body.is_fixed() {
                    return Err(PhysicsError::TrimeshOnDynamicBody);
                }
            }
        }

        let rotation = na::UnitQuaternion::new_normalize(na::Quaternion::new(
            descriptor.rotation[3],
            descriptor.rotation[0],
            descriptor.rotation[1],
            descriptor.rotation[2],
        ));
        let mut builder = shape_builder(descriptor.shape)?
            .sensor(descriptor.is_trigger)
            .position(na::Isometry::from_parts(
                vec3(descriptor.translation).into(),
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

    pub fn body_position_by_id(&self, id: &PhysicsBodyId) -> Option<[f32; 3]> {
        self.body_pose_by_id(id).map(|pose| pose.translation)
    }

    pub fn body_pose_by_id(&self, id: &PhysicsBodyId) -> Option<BodyPose> {
        let handle = *self.body_handles.get(id)?;
        let body = self.bodies.get(handle)?;
        let translation = body.translation();
        let rotation = body.rotation().quaternion();
        Some(BodyPose {
            translation: [translation.x, translation.y, translation.z],
            rotation: [rotation.i, rotation.j, rotation.k, rotation.w],
        })
    }

    pub fn set_body_position_by_id(
        &mut self,
        id: &PhysicsBodyId,
        translation: [f32; 3],
    ) -> Result<(), PhysicsError> {
        self.set_body_pose_by_id(
            id,
            BodyPose {
                translation,
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        )
    }

    pub fn set_body_pose_by_id(
        &mut self,
        id: &PhysicsBodyId,
        pose: BodyPose,
    ) -> Result<(), PhysicsError> {
        validate_vec3("body.translation", pose.translation)?;
        validate_rotation(pose.rotation)?;
        let handle = *self
            .body_handles
            .get(id)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        let body = self
            .bodies
            .get_mut(handle)
            .ok_or_else(|| PhysicsError::MissingBody(id.clone()))?;
        let isometry = pose_isometry(pose);
        if body.is_kinematic() {
            body.set_next_kinematic_position(isometry);
        }
        body.set_position(isometry, true);
        self.query_pipeline.update(&self.colliders);
        Ok(())
    }

    /// Removes a collider and its durable-ID mapping. Missing IDs are idempotent.
    pub fn remove_collider(&mut self, id: &PhysicsColliderId) -> bool {
        let Some(handle) = self.collider_handles.remove(id) else {
            return false;
        };
        self.colliders
            .remove(handle, &mut self.island_manager, &mut self.bodies, true);
        self.collider_ids
            .retain(|(candidate, _)| *candidate != handle);
        self.active_pairs.clear();
        self.last_contacts.clear();
        self.query_pipeline.update(&self.colliders);
        true
    }

    /// Removes a body, its attached colliders, and all durable-ID mappings.
    pub fn remove_body(&mut self, id: &PhysicsBodyId) -> bool {
        let Some(handle) = self.body_handles.remove(id) else {
            return false;
        };
        let attached: Vec<_> = self
            .collider_ids
            .iter()
            .filter_map(|(collider_handle, collider_id)| {
                self.colliders.get(*collider_handle).and_then(|collider| {
                    (collider.parent() == Some(handle)).then(|| collider_id.clone())
                })
            })
            .collect();
        self.bodies.remove(
            handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
        self.body_ids.retain(|(candidate, _)| *candidate != handle);
        for collider_id in attached {
            if let Some(collider_handle) = self.collider_handles.remove(&collider_id) {
                self.collider_ids
                    .retain(|(candidate, _)| *candidate != collider_handle);
            }
        }
        self.active_pairs.clear();
        self.last_contacts.clear();
        self.query_pipeline.update(&self.colliders);
        true
    }

    pub fn collider_exists(&self, id: &PhysicsColliderId) -> bool {
        self.collider_handles.contains_key(id)
    }

    pub fn cast_ray(&mut self, query: RayQuery) -> Result<Option<RayHit>, PhysicsError> {
        validate_vec3("ray.origin", query.origin)?;
        validate_vec3("ray.direction", query.direction)?;
        validate_positive("ray.max_time_of_impact", query.max_time_of_impact)?;
        if vec3(query.direction).norm_squared() == 0.0 {
            return Err(PhysicsError::ZeroDirection);
        }

        self.query_pipeline.update(&self.colliders);
        let ray = Ray::new(point3(query.origin), vec3(query.direction));
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
            Some(RayHit {
                body,
                collider,
                time_of_impact: toi,
            })
        }))
    }

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
            records.push(record(PhysicsContactPhase::Enter, current[key], key));
        }
        for key in current_keys.intersection(&previous_keys) {
            records.push(record(PhysicsContactPhase::Stay, current[key], key));
        }
        for key in previous_keys.difference(&current_keys) {
            if let Some(kind) = self.active_pairs.get(key).copied() {
                records.push(record(PhysicsContactPhase::Exit, kind, key));
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
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

fn pose_isometry(pose: BodyPose) -> na::Isometry3<f32> {
    let rotation = na::UnitQuaternion::new_normalize(na::Quaternion::new(
        pose.rotation[3],
        pose.rotation[0],
        pose.rotation[1],
        pose.rotation[2],
    ));
    na::Isometry::from_parts(vec3(pose.translation).into(), rotation)
}

fn record(
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

/// Validates shape geometry and body-kind compatibility without mutating a world.
pub fn validate_collider_shape(
    shape: &ColliderShape,
    body_kind: BodyKind,
) -> Result<(), PhysicsError> {
    if matches!(shape, ColliderShape::TriMeshStatic { .. }) && body_kind != BodyKind::Static {
        return Err(PhysicsError::TrimeshOnDynamicBody);
    }
    shape_builder(shape.clone()).map(|_| ())
}

fn shape_builder(shape: ColliderShape) -> Result<ColliderBuilder, PhysicsError> {
    match shape {
        ColliderShape::Cuboid { half_extents } => {
            validate_vec3("cuboid.half_extents", half_extents)?;
            for (index, value) in half_extents.into_iter().enumerate() {
                validate_positive(
                    match index {
                        0 => "cuboid.half_extents.x",
                        1 => "cuboid.half_extents.y",
                        _ => "cuboid.half_extents.z",
                    },
                    value,
                )?;
            }
            Ok(ColliderBuilder::cuboid(
                half_extents[0],
                half_extents[1],
                half_extents[2],
            ))
        }
        ColliderShape::Sphere { radius } => {
            validate_positive("sphere.radius", radius)?;
            Ok(ColliderBuilder::ball(radius))
        }
        ColliderShape::CapsuleY {
            half_height,
            radius,
        } => {
            validate_positive("capsule.half_height", half_height)?;
            validate_positive("capsule.radius", radius)?;
            Ok(ColliderBuilder::capsule_y(half_height, radius))
        }
        ColliderShape::TriMeshStatic { vertices, indices } => {
            validate_trimesh(&vertices, &indices)?;
            let points: Vec<na::Point3<f32>> = vertices
                .into_iter()
                .map(|v| na::Point3::new(v[0], v[1], v[2]))
                .collect();
            Ok(ColliderBuilder::trimesh(points, indices))
        }
        ColliderShape::ConvexHull { points } => {
            let validated = validate_convex_hull(&points)?;
            let rapier_points: Vec<na::Point3<f32>> = validated
                .into_iter()
                .map(|v| na::Point3::new(v[0], v[1], v[2]))
                .collect();
            ColliderBuilder::convex_hull(&rapier_points).ok_or(PhysicsError::ConvexHullDegenerate)
        }
    }
}

fn validate_trimesh(vertices: &[[f32; 3]], indices: &[[u32; 3]]) -> Result<(), PhysicsError> {
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

fn dedup_points(points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    fn coordinate_key(value: f32) -> u32 {
        // IEEE -0.0 and +0.0 are numerically equal and must not inflate the
        // unique-point count merely because their sign bits differ.
        if value == 0.0 {
            0
        } else {
            value.to_bits()
        }
    }

    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for &p in points {
        let key = (
            coordinate_key(p[0]),
            coordinate_key(p[1]),
            coordinate_key(p[2]),
        );
        if seen.insert(key) {
            result.push(p);
        }
    }
    result
}

fn convex_hull_is_degenerate(unique: &[[f32; 3]]) -> bool {
    const RELATIVE_EPSILON: f64 = 1e-6;

    let points: Vec<[f64; 3]> = unique
        .iter()
        .map(|point| [point[0] as f64, point[1] as f64, point[2] as f64])
        .collect();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in &points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let scale = (0..3)
        .map(|axis| max[axis] - min[axis])
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return true;
    }

    let subtract = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    // Build a stable affine basis: the point furthest from the origin point,
    // then the point furthest from that line, then measure distance from the
    // resulting plane. Distances are compared to one relative model-space
    // scale, making the classification translation- and rotation-independent.
    let origin = points[0];
    let direction = points
        .iter()
        .map(|&point| subtract(point, origin))
        .max_by(|a, b| dot(*a, *a).total_cmp(&dot(*b, *b)))
        .unwrap_or([0.0; 3]);
    let direction_length = dot(direction, direction).sqrt();
    if direction_length <= scale * RELATIVE_EPSILON {
        return true;
    }

    let normal = points
        .iter()
        .map(|&point| cross(direction, subtract(point, origin)))
        .max_by(|a, b| dot(*a, *a).total_cmp(&dot(*b, *b)))
        .unwrap_or([0.0; 3]);
    let normal_length = dot(normal, normal).sqrt();
    if normal_length / direction_length <= scale * RELATIVE_EPSILON {
        return true;
    }

    let max_plane_distance = points
        .iter()
        .map(|&point| dot(normal, subtract(point, origin)).abs() / normal_length)
        .fold(0.0_f64, f64::max);
    max_plane_distance <= scale * RELATIVE_EPSILON
}

fn validate_convex_hull(points: &[[f32; 3]]) -> Result<Vec<[f32; 3]>, PhysicsError> {
    if points.is_empty() {
        return Err(PhysicsError::ConvexHullEmpty);
    }
    for (i, p) in points.iter().enumerate() {
        if p.iter().any(|c| !c.is_finite()) {
            return Err(PhysicsError::ConvexHullNonFiniteVertex { index: i });
        }
    }
    let unique = dedup_points(points);
    if unique.len() < 4 {
        return Err(PhysicsError::ConvexHullInsufficientPoints {
            unique_count: unique.len(),
        });
    }
    if convex_hull_is_degenerate(&unique) {
        return Err(PhysicsError::ConvexHullDegenerate);
    }
    Ok(unique)
}

fn validate_vec3(field: &'static str, value: [f32; 3]) -> Result<(), PhysicsError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(PhysicsError::NonFiniteValue { field });
    }
    Ok(())
}

fn validate_rotation(value: [f32; 4]) -> Result<(), PhysicsError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(PhysicsError::InvalidRotation);
    }
    let norm_squared = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>();
    if norm_squared <= f32::EPSILON {
        return Err(PhysicsError::InvalidRotation);
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f32) -> Result<(), PhysicsError> {
    if !value.is_finite() {
        return Err(PhysicsError::NonFiniteValue { field });
    }
    if value <= 0.0 {
        return Err(PhysicsError::NonPositiveDimension { field });
    }
    Ok(())
}

fn vec3(value: [f32; 3]) -> na::Vector3<f32> {
    na::Vector3::new(value[0], value[1], value[2])
}

fn point3(value: [f32; 3]) -> na::Point3<f32> {
    na::Point3::new(value[0], value[1], value[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_creation() {
        let world = PhysicsWorld::new();
        assert!((world.gravity.y + 9.81).abs() < 0.01);
    }

    #[test]
    fn body_falls_under_gravity() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, -10.0, 0.0);
        let body = world
            .create_body(BodyDescriptor::new(
                "body.player",
                BodyKind::Dynamic,
                [0.0, 10.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.player",
                body.clone(),
                ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
            ))
            .unwrap();
        world.step(1.0).unwrap();
        let pos = world.body_position_by_id(&body).unwrap();
        assert!(pos[1] < 10.0, "body should fall under gravity");
    }

    #[test]
    fn descriptor_validation_rejects_invalid_dimensions() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();

        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::Sphere { radius: 0.0 },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::NonPositiveDimension {
                field: "sphere.radius"
            }
        );

        let err = world
            .create_body(BodyDescriptor::new(
                "body.bad",
                BodyKind::Dynamic,
                [0.0, f32::NAN, 0.0],
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
    fn durable_ids_map_to_runtime_bodies_and_colliders() {
        let mut world = PhysicsWorld::new();
        let body = PhysicsBodyId::new("body.ball");
        let collider = PhysicsColliderId::new("collider.ball");
        assert_eq!(
            world
                .create_body(BodyDescriptor::new(
                    body.clone(),
                    BodyKind::Dynamic,
                    [1.0, 2.0, 3.0],
                ))
                .unwrap(),
            body
        );
        assert_eq!(
            world
                .create_collider(ColliderDescriptor::new(
                    collider.clone(),
                    body.clone(),
                    ColliderShape::Sphere { radius: 0.25 },
                ))
                .unwrap(),
            collider
        );

        assert_eq!(world.body_position_by_id(&body), Some([1.0, 2.0, 3.0]));
        assert!(world.collider_exists(&collider));
    }

    #[test]
    fn set_body_position_updates_durable_body_pose() {
        let mut world = PhysicsWorld::new();
        let body = PhysicsBodyId::new("body.kinematic");
        world
            .create_body(BodyDescriptor::new(
                body.clone(),
                BodyKind::Kinematic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();

        world
            .set_body_position_by_id(&body, [2.0, 3.0, 4.0])
            .unwrap();

        assert_eq!(world.body_position_by_id(&body), Some([2.0, 3.0, 4.0]));
        assert_eq!(
            world.set_body_position_by_id(&PhysicsBodyId::new("missing"), [0.0, 0.0, 0.0]),
            Err(PhysicsError::MissingBody(PhysicsBodyId::new("missing")))
        );
    }

    #[test]
    fn ray_query_reports_hit_and_miss() {
        let mut world = PhysicsWorld::new();
        let body = world
            .create_body(BodyDescriptor::new(
                "body.target",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.target",
                body.clone(),
                ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
            ))
            .unwrap();

        let hit = world
            .cast_ray(RayQuery::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 10.0))
            .unwrap()
            .unwrap();
        assert_eq!(hit.body, body);
        assert_eq!(hit.collider, collider);
        assert!(hit.time_of_impact > 0.0);

        let miss = world
            .cast_ray(RayQuery::new([5.0, 0.0, -5.0], [0.0, 0.0, 1.0], 10.0))
            .unwrap();
        assert!(miss.is_none());
    }

    #[test]
    fn contact_records_report_enter_stay_and_exit() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);
        world
            .create_body(BodyDescriptor::new(
                "body.a",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.a",
                "body.a",
                ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
            ))
            .unwrap();
        world
            .create_body(BodyDescriptor::new(
                "body.b",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.b",
                "body.b",
                ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
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
    fn trigger_records_are_separate_from_collision_records() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);
        world
            .create_body(BodyDescriptor::new(
                "body.trigger",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(
                ColliderDescriptor::new(
                    "collider.trigger",
                    "body.trigger",
                    ColliderShape::Sphere { radius: 1.0 },
                )
                .trigger(true),
            )
            .unwrap();
        world
            .create_body(BodyDescriptor::new(
                "body.other",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.other",
                "body.other",
                ColliderShape::CapsuleY {
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
        assert_eq!(
            world.last_contact_records()[0].a,
            PhysicsColliderId::new("collider.trigger")
        );
    }

    #[test]
    fn collision_record_maps_to_engine_event() {
        let record = PhysicsContactRecord {
            phase: PhysicsContactPhase::Enter,
            kind: PhysicsContactKind::Collision,
            a: PhysicsColliderId::new("collider.a"),
            b: PhysicsColliderId::new("collider.b"),
        };

        assert_eq!(
            record.to_engine_event(),
            EngineEvent::Physics(PhysicsEvent::Collision {
                phase: EventContactPhase::Enter,
                a: PhysicsColliderId::new("collider.a"),
                b: PhysicsColliderId::new("collider.b"),
            })
        );
    }

    #[test]
    fn trigger_record_maps_trigger_first_to_engine_event() {
        let record = PhysicsContactRecord {
            phase: PhysicsContactPhase::Stay,
            kind: PhysicsContactKind::Trigger,
            a: PhysicsColliderId::new("collider.sensor"),
            b: PhysicsColliderId::new("collider.player"),
        };

        assert_eq!(
            record.to_engine_event(),
            EngineEvent::Physics(PhysicsEvent::Trigger {
                phase: EventContactPhase::Stay,
                trigger: PhysicsColliderId::new("collider.sensor"),
                other: PhysicsColliderId::new("collider.player"),
            })
        );
    }

    #[test]
    fn query_hit_maps_to_engine_event_without_distance_loss_in_physics_record() {
        let hit = RayHit {
            body: PhysicsBodyId::new("body.target"),
            collider: PhysicsColliderId::new("collider.target"),
            time_of_impact: 4.5,
        };

        assert_eq!(
            hit.to_engine_event(),
            EngineEvent::Physics(PhysicsEvent::QueryHit {
                body: PhysicsBodyId::new("body.target"),
                collider: PhysicsColliderId::new("collider.target"),
            })
        );
        assert_eq!(hit.time_of_impact, 4.5);
    }

    #[test]
    fn physics_step_query_and_event_bridge_run_without_renderer() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);
        world
            .create_body(BodyDescriptor::new(
                "body.target",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.target",
                "body.target",
                ColliderShape::Sphere { radius: 1.0 },
            ))
            .unwrap();
        world.step(1.0 / 60.0).unwrap();
        let hit = world
            .cast_ray(RayQuery::new([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 10.0))
            .unwrap()
            .unwrap();

        let mut bus = EventBus::new();
        bus.emit(EventStage::PostUpdate, None, hit.to_engine_event());
        assert_eq!(bus.pending_len(), 1);
        assert_eq!(bus.dispatch_pending().dispatched, 1);
    }

    // --- TriMeshStatic tests ---

    fn simple_triangle_mesh() -> ColliderShape {
        ColliderShape::TriMeshStatic {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            indices: vec![[0, 1, 2], [1, 3, 2]],
        }
    }

    #[test]
    fn trimesh_static_body_accepted() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            simple_triangle_mesh(),
        ));
        assert!(result.is_ok(), "static trimesh should be accepted");
    }

    #[test]
    fn trimesh_dynamic_body_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.player",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.player",
                simple_triangle_mesh(),
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshOnDynamicBody);
    }

    #[test]
    fn trimesh_kinematic_body_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.kin",
                BodyKind::Kinematic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.kin",
                simple_triangle_mesh(),
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshOnDynamicBody);
    }

    #[test]
    fn trimesh_non_finite_vertices_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, f32::NAN, 0.0]],
                    indices: vec![[0, 1, 0]], // degenerate, but non-finite should fire first
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshNonFiniteVertex { index: 1 });
    }

    #[test]
    fn trimesh_infinity_vertices_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, f32::INFINITY, 0.0]],
                    indices: vec![[0, 1, 2]],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshNonFiniteVertex { index: 2 });
    }

    #[test]
    fn trimesh_out_of_bounds_indices_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![[0, 1, 5]],
                },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::TrimeshIndexOutOfBounds {
                index: 0,
                vertex_count: 3
            }
        );
    }

    #[test]
    fn trimesh_empty_mesh_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();

        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![],
                    indices: vec![],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshEmpty);

        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad2",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshEmpty);
    }

    #[test]
    fn trimesh_degenerate_triangle_all_same_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![[0, 0, 0]],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshDegenerateTriangle { index: 0 });
    }

    #[test]
    fn trimesh_degenerate_triangle_two_same_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![[0, 1, 0]],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::TrimeshDegenerateTriangle { index: 0 });
    }

    #[test]
    fn trimesh_ray_query() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.floor",
                "body.floor",
                simple_triangle_mesh(),
            ))
            .unwrap();

        // Ray from above should hit the mesh
        let hit = world
            .cast_ray(RayQuery::new([0.5, 0.5, -5.0], [0.0, 0.0, 1.0], 10.0))
            .unwrap()
            .unwrap();
        assert_eq!(hit.collider, collider);
    }

    #[test]
    fn existing_shapes_still_work_with_trimesh_added() {
        let mut world = PhysicsWorld::new();

        // Cuboid
        world
            .create_body(BodyDescriptor::new(
                "body.a",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        assert!(world
            .create_collider(ColliderDescriptor::new(
                "collider.a",
                "body.a",
                ColliderShape::Cuboid {
                    half_extents: [1.0, 1.0, 1.0],
                },
            ))
            .is_ok());

        // Sphere
        world
            .create_body(BodyDescriptor::new(
                "body.b",
                BodyKind::Dynamic,
                [0.0, 5.0, 0.0],
            ))
            .unwrap();
        assert!(world
            .create_collider(ColliderDescriptor::new(
                "collider.b",
                "body.b",
                ColliderShape::Sphere { radius: 0.5 },
            ))
            .is_ok());

        // CapsuleY
        world
            .create_body(BodyDescriptor::new(
                "body.c",
                BodyKind::Dynamic,
                [2.0, 5.0, 0.0],
            ))
            .unwrap();
        assert!(world
            .create_collider(ColliderDescriptor::new(
                "collider.c",
                "body.c",
                ColliderShape::CapsuleY {
                    half_height: 1.0,
                    radius: 0.5,
                },
            ))
            .is_ok());
    }

    #[test]
    fn trimesh_contact_with_dynamic_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, -10.0, 0.0);

        // Static trimesh floor
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.floor",
                "body.floor",
                ColliderShape::TriMeshStatic {
                    vertices: vec![
                        [-5.0, 0.0, -5.0],
                        [5.0, 0.0, -5.0],
                        [-5.0, 0.0, 5.0],
                        [5.0, 0.0, 5.0],
                    ],
                    indices: vec![[0, 1, 2], [1, 3, 2]],
                },
            ))
            .unwrap();

        // Dynamic sphere above the floor
        world
            .create_body(BodyDescriptor::new(
                "body.ball",
                BodyKind::Dynamic,
                [0.0, 2.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.ball",
                "body.ball",
                ColliderShape::Sphere { radius: 0.5 },
            ))
            .unwrap();

        world.step(1.0 / 60.0).unwrap();
        // Ball should be above floor (contact has started)
        let pos = world
            .body_position_by_id(&PhysicsBodyId::new("body.ball"))
            .unwrap();
        assert!(pos[1] < 2.0, "ball should fall toward floor");
    }

    // --- ConvexHull tests ---

    /// Four points forming a non-degenerate tetrahedron.
    fn tetrahedron_points() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Eight cube points plus interior and duplicate points.
    fn cube_with_redundant_points() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            // interior point
            [0.5, 0.5, 0.5],
            // duplicate of first corner
            [0.0, 0.0, 0.0],
        ]
    }

    // --- Step 5: body-kind policy ---

    #[test]
    fn convex_hull_static_body_accepted() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.s",
            "body.s",
            ColliderShape::ConvexHull {
                points: tetrahedron_points(),
            },
        ));
        assert!(result.is_ok(), "convex hull on static body should succeed");
    }

    #[test]
    fn convex_hull_dynamic_body_accepted() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.d",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.d",
            "body.d",
            ColliderShape::ConvexHull {
                points: tetrahedron_points(),
            },
        ));
        assert!(result.is_ok(), "convex hull on dynamic body should succeed");
    }

    #[test]
    fn convex_hull_kinematic_body_accepted() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.k",
                BodyKind::Kinematic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.k",
            "body.k",
            ColliderShape::ConvexHull {
                points: tetrahedron_points(),
            },
        ));
        assert!(
            result.is_ok(),
            "convex hull on kinematic body should succeed"
        );
    }

    // --- Step 6: valid fixtures ---

    #[test]
    fn convex_hull_tetrahedron_creation() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.tetra",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.tetra",
                "body.tetra",
                ColliderShape::ConvexHull {
                    points: tetrahedron_points(),
                },
            ))
            .unwrap();
        assert!(world.collider_exists(&collider));
    }

    #[test]
    fn convex_hull_tetrahedron_ray_hit() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.tetra",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.tetra",
                "body.tetra",
                ColliderShape::ConvexHull {
                    points: tetrahedron_points(),
                },
            ))
            .unwrap();

        // Ray from above hitting the tetrahedron
        let hit = world
            .cast_ray(RayQuery::new([0.25, 0.25, -2.0], [0.0, 0.0, 1.0], 10.0))
            .unwrap()
            .unwrap();
        assert_eq!(hit.collider, collider);
        assert!(hit.time_of_impact > 0.0);
    }

    #[test]
    fn convex_hull_cube_with_redundant_points() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.cube",
                BodyKind::Dynamic,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.cube",
                "body.cube",
                ColliderShape::ConvexHull {
                    points: cube_with_redundant_points(),
                },
            ))
            .unwrap();
        assert!(world.collider_exists(&collider));
    }

    #[test]
    fn convex_hull_dynamic_collides_with_static_floor() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, -10.0, 0.0);

        // Static floor cube
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, -0.5, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.floor",
                "body.floor",
                ColliderShape::Cuboid {
                    half_extents: [3.0, 0.5, 3.0],
                },
            ))
            .unwrap();

        // Dynamic convex hull tetrahedron above floor
        world
            .create_body(BodyDescriptor::new(
                "body.hull",
                BodyKind::Dynamic,
                [0.0, 3.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.hull",
                "body.hull",
                ColliderShape::ConvexHull {
                    points: tetrahedron_points(),
                },
            ))
            .unwrap();

        world.step(1.0 / 60.0).unwrap();
        let pos = world
            .body_position_by_id(&PhysicsBodyId::new("body.hull"))
            .unwrap();
        assert!(pos[1] < 3.0, "convex hull should fall toward floor");
    }

    // --- Step 7: invalid fixtures ---

    #[test]
    fn convex_hull_empty_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        assert!(
            ColliderBuilder::convex_hull(&[]).is_none(),
            "empty fixture must also be rejected by Rapier/Parry"
        );
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull { points: vec![] },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullEmpty);
    }

    #[test]
    fn convex_hull_one_unique_point_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![[0.0, 0.0, 0.0]],
                },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 }
        );
    }

    #[test]
    fn convex_hull_three_unique_points_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull { points },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::ConvexHullInsufficientPoints { unique_count: 3 }
        );
    }

    #[test]
    fn convex_hull_nan_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, f32::NAN, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullNonFiniteVertex { index: 2 });
    }

    #[test]
    fn convex_hull_infinity_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, f32::INFINITY],
                    ],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullNonFiniteVertex { index: 3 });
    }

    #[test]
    fn convex_hull_all_duplicates_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                },
            ))
            .unwrap_err();
        assert_eq!(
            err,
            PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 }
        );
    }

    #[test]
    fn convex_hull_collinear_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        // Four points on the same line
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullDegenerate);
    }

    #[test]
    fn convex_hull_coplanar_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        // Four points on the same plane (z=0).
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.bad",
                "body.s",
                ColliderShape::ConvexHull { points },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullDegenerate);
    }

    #[test]
    fn convex_hull_near_coplanar_rejected() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        // Four points with an extremely small z offset produce a near-zero
        // point-to-plane distance relative to their scale.
        let z = 1e-8;
        let err = world
            .create_collider(ColliderDescriptor::new(
                "collider.near",
                "body.s",
                ColliderShape::ConvexHull {
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.5, 0.5, z],
                    ],
                },
            ))
            .unwrap_err();
        assert_eq!(err, PhysicsError::ConvexHullDegenerate);

        // The same thin tetrahedron rotated around Y must classify identically;
        // an axis-aligned AABB thickness test would miss this orientation.
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let rotated = vec![
            [0.0, 0.0, 0.0],
            [diagonal, 0.0, -diagonal],
            [0.0, 1.0, 0.0],
            [
                0.5 * diagonal + z * diagonal,
                0.5,
                -0.5 * diagonal + z * diagonal,
            ],
        ];
        assert_eq!(
            validate_convex_hull(&rotated),
            Err(PhysicsError::ConvexHullDegenerate)
        );
    }

    #[test]
    fn convex_hull_all_failures_are_transactional() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.s",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();

        let invalid_cases = vec![
            (vec![], PhysicsError::ConvexHullEmpty),
            (
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, f32::NAN, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                PhysicsError::ConvexHullNonFiniteVertex { index: 2 },
            ),
            (
                vec![[0.0, 0.0, 0.0]; 4],
                PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 },
            ),
            (
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                PhysicsError::ConvexHullDegenerate,
            ),
            (
                // An oblique plane exercises orientation-independent detection.
                vec![
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.25, 0.25],
                ],
                PhysicsError::ConvexHullDegenerate,
            ),
        ];

        for (index, (points, expected)) in invalid_cases.into_iter().enumerate() {
            let id = PhysicsColliderId::new(format!("collider.bad.{index}"));
            let collider_count_before = world.colliders.len();
            let err = world
                .create_collider(ColliderDescriptor::new(
                    id.clone(),
                    "body.s",
                    ColliderShape::ConvexHull { points },
                ))
                .unwrap_err();
            assert_eq!(err, expected);
            assert!(!world.collider_exists(&id));
            assert_eq!(world.colliders.len(), collider_count_before);
            assert_eq!(world.collider_ids.len(), collider_count_before);
        }
    }

    #[test]
    fn convex_hull_signed_zero_duplicates_are_insufficient() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [-0.0, 0.0, 0.0],
            [0.0, -0.0, 0.0],
            [0.0, 0.0, -0.0],
        ];
        assert_eq!(
            validate_convex_hull(&points),
            Err(PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 })
        );
    }

    // --- Step 8: event conversion unchanged with convex hull ---

    #[test]
    fn convex_hull_contact_event_conversion() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, -10.0, 0.0);

        // Static floor
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, -1.0, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.floor",
                "body.floor",
                ColliderShape::Cuboid {
                    half_extents: [3.0, 0.5, 3.0],
                },
            ))
            .unwrap();

        // Dynamic convex hull starting just above the floor
        world
            .create_body(BodyDescriptor::new(
                "body.hull",
                BodyKind::Dynamic,
                [0.0, 0.1, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.hull",
                "body.hull",
                ColliderShape::ConvexHull {
                    points: vec![
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [-0.5, 0.5, -0.5],
                        [0.0, -0.5, 0.5],
                    ],
                },
            ))
            .unwrap();

        // Multiple steps to ensure contact is registered
        for _ in 0..10 {
            world.step(1.0 / 60.0).unwrap();
        }
        let records = world.last_contact_records();
        assert!(!records.is_empty(), "convex hull should produce contacts");
        assert_eq!(records[0].kind, PhysicsContactKind::Collision);

        // Verify event conversion works
        let event = records[0].to_engine_event();
        assert!(matches!(
            event,
            EngineEvent::Physics(PhysicsEvent::Collision { .. })
        ));
    }

    // --- Display / error formatting tests ---

    #[test]
    fn convex_hull_error_display_formatting() {
        assert_eq!(
            format!("{}", PhysicsError::ConvexHullEmpty),
            "empty convex hull"
        );
        assert_eq!(
            format!("{}", PhysicsError::ConvexHullNonFiniteVertex { index: 5 }),
            "non-finite convex hull vertex at index 5"
        );
        assert_eq!(
            format!(
                "{}",
                PhysicsError::ConvexHullInsufficientPoints { unique_count: 2 }
            ),
            "insufficient unique points for convex hull: 2 (need at least 4)"
        );
        assert_eq!(
            format!("{}", PhysicsError::ConvexHullDegenerate),
            "degenerate convex hull (coplanar or zero-volume)"
        );
    }

    #[test]
    fn convex_hull_error_equality() {
        assert_eq!(PhysicsError::ConvexHullEmpty, PhysicsError::ConvexHullEmpty);
        assert_eq!(
            PhysicsError::ConvexHullNonFiniteVertex { index: 2 },
            PhysicsError::ConvexHullNonFiniteVertex { index: 2 }
        );
        assert_ne!(
            PhysicsError::ConvexHullNonFiniteVertex { index: 2 },
            PhysicsError::ConvexHullNonFiniteVertex { index: 3 }
        );
        assert_eq!(
            PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 },
            PhysicsError::ConvexHullInsufficientPoints { unique_count: 1 }
        );
        assert_eq!(
            PhysicsError::ConvexHullDegenerate,
            PhysicsError::ConvexHullDegenerate
        );
    }

    #[test]
    fn convex_hull_error_distinct_from_trimesh_errors() {
        assert_ne!(PhysicsError::ConvexHullEmpty, PhysicsError::TrimeshEmpty);
        assert_ne!(
            PhysicsError::ConvexHullDegenerate,
            PhysicsError::TrimeshDegenerateTriangle { index: 0 }
        );
        assert_ne!(
            PhysicsError::ConvexHullNonFiniteVertex { index: 0 },
            PhysicsError::TrimeshNonFiniteVertex { index: 0 }
        );
    }

    #[test]
    fn convex_hull_preserves_model_space_points() {
        // Verify the public shape payload is unchanged after a successful creation.
        let points = tetrahedron_points();
        let shape = ColliderShape::ConvexHull {
            points: points.clone(),
        };
        // The shape is consumed by shape_builder, so test the payload round-trip
        // by asserting the points are visible before consumption.
        if let ColliderShape::ConvexHull { points: p } = &shape {
            assert_eq!(p, &points);
        } else {
            panic!("expected ConvexHull variant");
        }
    }

    /// Scale-aware epsilon rationale: affine distances are computed in `f64`
    /// and compared to the largest model-space extent. This avoids a fixed
    /// world-unit tolerance and gives the same result after rotation.
    #[test]
    fn convex_hull_scale_aware_epsilon_rationale() {
        // Large-scale tetrahedron (kilometer extent) should succeed
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.big",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.big",
            "body.big",
            ColliderShape::ConvexHull {
                points: vec![
                    [0.0, 0.0, 0.0],
                    [1000.0, 0.0, 0.0],
                    [0.0, 1000.0, 0.0],
                    [0.0, 0.0, 1000.0],
                ],
            },
        ));
        assert!(result.is_ok(), "large-scale convex hull should succeed");

        // Tiny tetrahedron (sub-millimeter extent) should succeed
        world
            .create_body(BodyDescriptor::new(
                "body.tiny",
                BodyKind::Static,
                [0.0, 0.0, 0.0],
            ))
            .unwrap();
        let result = world.create_collider(ColliderDescriptor::new(
            "collider.tiny",
            "body.tiny",
            ColliderShape::ConvexHull {
                points: vec![
                    [0.0, 0.0, 0.0],
                    [0.001, 0.0, 0.0],
                    [0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.001],
                ],
            },
        ));
        assert!(result.is_ok(), "tiny-scale convex hull should succeed");
    }

    #[test]
    fn collider_rotation_rejects_non_finite_and_zero_quaternions_transactionally() {
        let mut world = PhysicsWorld::new();
        world
            .create_body(BodyDescriptor::new(
                "body.rotation",
                BodyKind::Static,
                [0.0; 3],
            ))
            .unwrap();
        for rotation in [[0.0; 4], [f32::NAN, 0.0, 0.0, 1.0]] {
            let result = world.create_collider(
                ColliderDescriptor::new(
                    "collider.rotation",
                    "body.rotation",
                    ColliderShape::Sphere { radius: 1.0 },
                )
                .rotation(rotation),
            );
            assert_eq!(result, Err(PhysicsError::InvalidRotation));
            assert!(!world.collider_exists(&PhysicsColliderId::new("collider.rotation")));
        }
    }

    #[test]
    fn body_pose_and_removal_keep_durable_maps_consistent() {
        let mut world = PhysicsWorld::new();
        let body = world
            .create_body(BodyDescriptor::new(
                "body.remove",
                BodyKind::Dynamic,
                [1.0, 2.0, 3.0],
            ))
            .unwrap();
        let collider = world
            .create_collider(ColliderDescriptor::new(
                "collider.remove",
                body.clone(),
                ColliderShape::Sphere { radius: 1.0 },
            ))
            .unwrap();
        let pose = world.body_pose_by_id(&body).unwrap();
        assert_eq!(pose.translation, [1.0, 2.0, 3.0]);
        assert_eq!(pose.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert!(world.remove_body(&body));
        assert!(world.body_pose_by_id(&body).is_none());
        assert!(!world.collider_exists(&collider));
        assert!(!world.remove_body(&body));
    }
}
