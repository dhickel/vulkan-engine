//! App-owned Rapier physics bridge for BSP collision data.
//!
//! Implements the [`AppBridge`] trait to create and manage Rapier rigid bodies
//! and colliders from BSP collision recipes. This is the ONLY place where
//! physics dependencies appear for BSP data — `bsp` and `bsp_runtime` remain
//! physics-free.
//!
//! # Resource Lifecycle
//!
//! - **Prepare**: Creates Rapier bodies and colliders from collision recipes
//!   (not yet added to the simulation world). Stores them in the bridge struct.
//! - **Validate**: Confirms all bodies and colliders are valid and complete.
//! - **Commit**: Publishes staged bodies/colliders into the active
//!   `PhysicsWorld` simulation.
//! - **Rollback**: Clears staged state (idempotent).
//!
//! # Collision Mapping
//!
//! | BSP data | Rapier shape | Body kind |
//! |----------|-------------|-----------|
//! | World clipnodes | `TriMeshStatic` | Static |
//! | Brush entity clipnodes | `ConvexHull` | Kinematic |
//! | Trigger brush | `ConvexHull` (sensor) | Static |

use std::collections::HashMap;

use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};

/// Staged physics descriptors awaiting commit.
#[derive(Debug, Clone)]
pub struct StagedPhysics {
    /// Descriptors for bodies to create.
    pub bodies: Vec<physics::BodyDescriptor>,
    /// Descriptors for colliders to create.
    pub colliders: Vec<physics::ColliderDescriptor>,
    /// Entity index → body ID mapping for transform sync.
    pub entity_body_map: HashMap<u32, physics::PhysicsBodyId>,
    /// Entity index → collider IDs for removal.
    pub entity_collider_map: HashMap<u32, Vec<physics::PhysicsColliderId>>,
    /// World body ID for the static trimesh.
    pub world_body_id: Option<physics::PhysicsBodyId>,
    /// World collider ID when a concrete static collider is available.
    pub world_collider_id: Option<physics::PhysicsColliderId>,
}

/// A physics bridge that creates Rapier resources from BSP collision recipes.
///
/// The bridge stores staged physics state internally during prepare and
/// publishes it into the provided `PhysicsWorld` during commit. The bridge
/// token carries a generation marker; actual physics descriptors are stored
/// in the bridge struct to avoid serializing Rapier types.
pub struct PhysicsBridge {
    /// Staged physics from the last prepare. Cleared on commit or rollback.
    staged: Option<StagedPhysics>,
    /// Whether we've been committed (used for idempotent rollback).
    committed: bool,
    /// Currently published body IDs (for removal during unload).
    published_bodies: Vec<physics::PhysicsBodyId>,
    /// Currently published collider IDs (for removal during unload).
    published_colliders: Vec<physics::PhysicsColliderId>,
    /// Entity index → body ID for transform sync.
    pub entity_bodies: HashMap<u32, physics::PhysicsBodyId>,
}

impl PhysicsBridge {
    /// Create a new physics bridge.
    pub fn new() -> Self {
        Self {
            staged: None,
            committed: false,
            published_bodies: Vec::new(),
            published_colliders: Vec::new(),
            entity_bodies: HashMap::new(),
        }
    }

    /// Access the staged physics descriptors after prepare.
    pub fn staged(&self) -> Option<&StagedPhysics> {
        self.staged.as_ref()
    }

    /// Publish prepared physics into the simulation world.
    ///
    /// Call this after `commit` with the active `PhysicsWorld`.
    pub fn commit_to_world(
        &mut self,
        physics_world: &mut physics::PhysicsWorld,
    ) -> Result<(), String> {
        let staged = self
            .staged
            .take()
            .ok_or_else(|| "no staged physics to commit".to_string())?;

        for body_desc in &staged.bodies {
            physics_world
                .create_body(body_desc.clone())
                .map_err(|e| format!("failed to create body: {e}"))?;
            self.published_bodies.push(body_desc.id.clone());
        }
        for collider_desc in &staged.colliders {
            physics_world
                .create_collider(collider_desc.clone())
                .map_err(|e| format!("failed to create collider: {e}"))?;
            self.published_colliders.push(collider_desc.id.clone());
        }
        self.entity_bodies = staged.entity_body_map.clone();
        self.committed = true;
        Ok(())
    }

    /// Remove all published physics resources from the world.
    pub fn remove_from_world(&mut self, physics_world: &mut physics::PhysicsWorld) {
        for collider_id in self.published_colliders.drain(..) {
            physics_world.remove_collider(&collider_id);
        }
        for body_id in self.published_bodies.drain(..) {
            physics_world.remove_body(&body_id);
        }
        self.entity_bodies.clear();
        self.committed = false;
    }

    /// Sync a kinematic body's transform with a new world-space position.
    ///
    /// Returns `true` if the body was found.
    pub fn sync_body_transform(
        &self,
        entity_index: u32,
        new_position: [f32; 3],
        physics_world: &mut physics::PhysicsWorld,
    ) -> bool {
        let transform = glam::Mat4::from_translation(glam::Vec3::from_array(new_position));
        self.sync_body_pose(entity_index, transform, physics_world)
    }

    /// Sync a kinematic body's full pose with a new world-space transform.
    ///
    /// Returns `true` if the body was found and the pose was accepted.
    pub fn sync_body_pose(
        &self,
        entity_index: u32,
        transform: glam::Mat4,
        physics_world: &mut physics::PhysicsWorld,
    ) -> bool {
        let Some(body_id) = self.entity_bodies.get(&entity_index) else {
            return false;
        };

        let (_scale, rotation, translation) = transform.to_scale_rotation_translation();
        let pose = physics::BodyPose {
            translation: translation.to_array(),
            rotation: rotation.to_array(),
        };

        match physics_world.set_body_pose_by_id(body_id, pose) {
            Ok(()) => true,
            Err(error) => {
                log::warn!(
                    "failed to sync BSP entity {} body {} to transform {:?}: {}",
                    entity_index,
                    body_id,
                    transform,
                    error
                );
                false
            }
        }
    }

    /// Sync all moving entity poses from a snapshot into the physics world.
    ///
    /// Returns the count of successfully updated bodies.
    pub fn sync_from_snapshot(
        &self,
        snapshot: &bsp_runtime::BspSimulationSnapshot,
        physics_world: &mut physics::PhysicsWorld,
    ) -> usize {
        let mut updated = 0usize;
        for pose in &snapshot.entity_poses {
            if !pose.is_moving {
                continue;
            }
            if self.sync_body_pose(pose.entity_index, pose.transform, physics_world) {
                updated += 1;
            }
        }
        updated
    }
}

impl Default for PhysicsBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBridge for PhysicsBridge {
    fn name(&self) -> &str {
        "physics"
    }

    fn prepare(
        &mut self,
        world_collider: &WorldCollisionRecipe,
        entity_colliders: &[EntityCollisionRecipe],
        _lights: &[LightEntityRecipe],
        _behaviors: &[BehaviorEntityRecipe],
    ) -> Result<BridgeToken, String> {
        let mut bodies: Vec<physics::BodyDescriptor> = Vec::new();
        let mut colliders: Vec<physics::ColliderDescriptor> = Vec::new();
        let mut entity_body_map: HashMap<u32, physics::PhysicsBodyId> = HashMap::new();
        let mut entity_collider_map: HashMap<u32, Vec<physics::PhysicsColliderId>> = HashMap::new();

        // ── World static collision ────────────────────────────────────
        let world_body_id = if !world_collider.planes.is_empty() {
            let body_id = physics::PhysicsBodyId::new("bsp.world");
            let body_desc = physics::BodyDescriptor::new(
                body_id.clone(),
                physics::BodyKind::Static,
                [0.0, 0.0, 0.0],
            );
            bodies.push(body_desc);
            log::warn!(
                "World collision has {} planes but no concrete mesh recipe; world collider not created",
                world_collider.planes.len()
            );
            Some(body_id)
        } else {
            None
        };

        // ── Entity collision ───────────────────────────────────────────
        for entity_recipe in entity_colliders {
            let entity_index = entity_recipe.entity_index;

            // Skip entities with no collision recipes
            if entity_recipe.recipes.is_empty() {
                continue;
            }

            let body_id = physics::PhysicsBodyId::new(format!("bsp.entity.{}", entity_index));
            let body_kind = if entity_recipe.is_trigger {
                physics::BodyKind::Static
            } else {
                physics::BodyKind::Kinematic
            };

            let body_desc = physics::BodyDescriptor::new(
                body_id.clone(),
                body_kind,
                [
                    entity_recipe.origin.x,
                    entity_recipe.origin.y,
                    entity_recipe.origin.z,
                ],
            );
            bodies.push(body_desc);
            entity_body_map.insert(entity_index, body_id.clone());

            let mut piece_collider_ids: Vec<physics::PhysicsColliderId> = Vec::new();

            for recipe in &entity_recipe.recipes {
                for (piece_idx, piece) in recipe.pieces.iter().enumerate() {
                    if piece.vertices.is_empty() {
                        continue;
                    }

                    let collider_id = physics::PhysicsColliderId::new(format!(
                        "bsp.entity.{}.piece.{}",
                        entity_index, piece_idx
                    ));

                    // Convert glam::Vec3 to [f32; 3]
                    let points: Vec<[f32; 3]> =
                        piece.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();

                    let collider_desc = physics::ColliderDescriptor::new(
                        collider_id.clone(),
                        body_id.clone(),
                        physics::ColliderShape::ConvexHull { points },
                    )
                    .trigger(entity_recipe.is_trigger)
                    .translation([
                        entity_recipe.origin.x,
                        entity_recipe.origin.y,
                        entity_recipe.origin.z,
                    ]);

                    colliders.push(collider_desc);
                    piece_collider_ids.push(collider_id);
                }
            }

            if !piece_collider_ids.is_empty() {
                entity_collider_map.insert(entity_index, piece_collider_ids);
            }
        }

        let staged = StagedPhysics {
            bodies,
            colliders,
            entity_body_map,
            entity_collider_map,
            world_body_id,
            world_collider_id: None,
        };

        let token = BridgeToken::new(vec![1u8]); // generation marker
        self.staged = Some(staged);
        Ok(token)
    }

    fn validate(&self, token: &BridgeToken) -> Result<(), String> {
        if token.payload.is_empty() {
            return Err("empty bridge token".to_string());
        }

        let staged = self
            .staged
            .as_ref()
            .ok_or_else(|| "no staged physics".to_string())?;

        // Verify all body references in colliders have matching bodies
        let body_ids: std::collections::HashSet<_> = staged.bodies.iter().map(|b| &b.id).collect();
        for collider in &staged.colliders {
            if !body_ids.contains(&&collider.parent_body) {
                return Err(format!(
                    "collider {} references unknown body {}",
                    collider.id, collider.parent_body
                ));
            }
        }

        Ok(())
    }

    fn commit(&mut self, token: BridgeToken) -> Result<(), String> {
        if token.payload.is_empty() {
            return Err("empty bridge token".to_string());
        }

        // Allow re-commit: clear previous committed state before accepting new batch.
        // The coordinator guarantees that the previous active mount has been unloaded
        // before committing a new candidate.
        if self.committed {
            log::debug!("Physics bridge: resetting prior committed state for new mount");
            self.committed = false;
            self.staged = None;
            self.published_bodies.clear();
            self.published_colliders.clear();
            self.entity_bodies.clear();
        }

        log::debug!(
            "Physics bridge commit: {} bodies, {} colliders staged for publication",
            self.staged.as_ref().map_or(0, |s| s.bodies.len()),
            self.staged.as_ref().map_or(0, |s| s.colliders.len()),
        );

        self.committed = true;
        Ok(())
    }

    fn rollback(&mut self, _token: BridgeToken) {
        self.staged = None;
        self.committed = false;
    }
}
