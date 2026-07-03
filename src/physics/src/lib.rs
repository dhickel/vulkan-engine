//! Physics subsystem — rigid body simulation via rapier3d.
//!
//! Provides `PhysicsWorld` with gravity, `RigidBody` and `Collider` types,
//! and a step function for per-frame simulation.

use rapier3d::na;
use rapier3d::prelude::*;

/// Physics world wrapping rapier3d.
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
        }
    }

    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.gravity = na::Vector3::new(x, y, z);
    }

    pub fn create_dynamic_body(&mut self, x: f32, y: f32, z: f32) -> RigidBodyHandle {
        let body = RigidBodyBuilder::dynamic()
            .translation(na::Vector3::new(x, y, z))
            .build();
        self.bodies.insert(body)
    }

    pub fn create_static_body(&mut self, x: f32, y: f32, z: f32) -> RigidBodyHandle {
        let body = RigidBodyBuilder::fixed()
            .translation(na::Vector3::new(x, y, z))
            .build();
        self.bodies.insert(body)
    }

    pub fn attach_cuboid(
        &mut self,
        body: RigidBodyHandle,
        hx: f32,
        hy: f32,
        hz: f32,
    ) -> ColliderHandle {
        let collider = ColliderBuilder::cuboid(hx, hy, hz).build();
        self.colliders
            .insert_with_parent(collider, body, &mut self.bodies)
    }

    pub fn body_position(&self, handle: RigidBodyHandle) -> Option<[f32; 3]> {
        self.bodies.get(handle).map(|b| {
            let t = b.translation();
            [t.x, t.y, t.z]
        })
    }

    pub fn step(&mut self, dt: f32) {
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
            None,
            &(),
            &(),
        );
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
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
        let body = world.create_dynamic_body(0.0, 10.0, 0.0);
        world.attach_cuboid(body, 0.5, 0.5, 0.5);
        world.step(1.0);
        let pos = world.body_position(body).unwrap();
        assert!(pos[1] < 10.0, "body should fall under gravity");
    }
}
