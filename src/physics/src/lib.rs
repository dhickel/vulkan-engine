//! Renderer-independent alpha physics API built on Rapier.
//!
//! Authored code should use durable [`PhysicsBodyId`] and [`PhysicsColliderId`]
//! values. Rapier handles remain an internal runtime detail except for the
//! deprecated compatibility helpers kept for the original smoke tests.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use engine_events::{
    ColliderId as EventColliderId, ContactPhase as EventContactPhase, EngineEvent, EventBus,
    EventStage, PhysicsBodyId as EventPhysicsBodyId, PhysicsEvent,
};
use rapier3d::na;
use rapier3d::prelude::*;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

string_id!(PhysicsBodyId);
string_id!(PhysicsColliderId);

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

impl BodyDescriptor {
    pub fn new(id: impl Into<PhysicsBodyId>, kind: BodyKind, translation: [f32; 3]) -> Self {
        Self {
            id: id.into(),
            kind,
            translation,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ColliderShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    CapsuleY { half_height: f32, radius: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColliderDescriptor {
    pub id: PhysicsColliderId,
    pub parent_body: PhysicsBodyId,
    pub shape: ColliderShape,
    pub is_trigger: bool,
    pub translation: [f32; 3],
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
            body: EventPhysicsBodyId::new(self.body.as_str()),
            collider: EventColliderId::new(self.collider.as_str()),
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
                a: EventColliderId::new(self.a.as_str()),
                b: EventColliderId::new(self.b.as_str()),
            }),
            PhysicsContactKind::Trigger => EngineEvent::Physics(PhysicsEvent::Trigger {
                phase,
                trigger: EventColliderId::new(self.a.as_str()),
                other: EventColliderId::new(self.b.as_str()),
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
        if self.collider_handles.contains_key(&descriptor.id) {
            return Err(PhysicsError::DuplicateColliderId(descriptor.id));
        }
        let body = *self
            .body_handles
            .get(&descriptor.parent_body)
            .ok_or_else(|| PhysicsError::MissingBody(descriptor.parent_body.clone()))?;

        let mut builder = shape_builder(descriptor.shape)?
            .sensor(descriptor.is_trigger)
            .translation(vec3(descriptor.translation));
        builder = builder.active_events(ActiveEvents::COLLISION_EVENTS);

        let id = descriptor.id;
        let handle = self
            .colliders
            .insert_with_parent(builder.build(), body, &mut self.bodies);
        self.collider_handles.insert(id.clone(), handle);
        self.collider_ids.push((handle, id.clone()));
        self.query_pipeline.update(&self.colliders);
        Ok(id)
    }

    pub fn body_position_by_id(&self, id: &PhysicsBodyId) -> Option<[f32; 3]> {
        self.body_handles
            .get(id)
            .and_then(|handle| self.body_position_for_handle(*handle))
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

    #[deprecated(note = "use create_body with BodyDescriptor instead")]
    pub fn create_dynamic_body(&mut self, x: f32, y: f32, z: f32) -> RigidBodyHandle {
        let body = RigidBodyBuilder::dynamic()
            .translation(na::Vector3::new(x, y, z))
            .build();
        self.bodies.insert(body)
    }

    #[deprecated(note = "use create_body with BodyDescriptor instead")]
    pub fn create_static_body(&mut self, x: f32, y: f32, z: f32) -> RigidBodyHandle {
        let body = RigidBodyBuilder::fixed()
            .translation(na::Vector3::new(x, y, z))
            .build();
        self.bodies.insert(body)
    }

    #[deprecated(note = "use create_collider with ColliderDescriptor instead")]
    pub fn attach_cuboid(
        &mut self,
        body: RigidBodyHandle,
        hx: f32,
        hy: f32,
        hz: f32,
    ) -> ColliderHandle {
        let collider = ColliderBuilder::cuboid(hx, hy, hz).build();
        let handle = self
            .colliders
            .insert_with_parent(collider, body, &mut self.bodies);
        self.query_pipeline.update(&self.colliders);
        handle
    }

    #[deprecated(note = "use body_position_by_id with PhysicsBodyId instead")]
    pub fn body_position(&self, handle: RigidBodyHandle) -> Option<[f32; 3]> {
        self.body_position_for_handle(handle)
    }

    fn body_position_for_handle(&self, handle: RigidBodyHandle) -> Option<[f32; 3]> {
        self.bodies.get(handle).map(|b| {
            let t = b.translation();
            [t.x, t.y, t.z]
        })
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
    }
}

fn validate_vec3(field: &'static str, value: [f32; 3]) -> Result<(), PhysicsError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(PhysicsError::NonFiniteValue { field });
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
                a: EventColliderId::new("collider.a"),
                b: EventColliderId::new("collider.b"),
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
                trigger: EventColliderId::new("collider.sensor"),
                other: EventColliderId::new("collider.player"),
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
                body: EventPhysicsBodyId::new("body.target"),
                collider: EventColliderId::new("collider.target"),
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
}
