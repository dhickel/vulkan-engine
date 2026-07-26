//! App-owned Rapier physics bridge for BSP collision data.
//!
//! Implements the [`AppBridge`] trait to create and manage Rapier rigid bodies
//! and colliders from BSP collision recipes. This is the ONLY place where
//! physics dependencies appear for BSP data — `bsp` and `bsp_runtime` remain
//! physics-free.
//!
//! # Phase 05: Active Bridge Receipts
//!
//! - **Prepare**: Creates a candidate-private `PhysicsWorld` with all bodies
//!   and colliders pre-created. All fallible creation work happens here.
//!   Undoes partial construction before returning an error.
//! - **Validate**: Confirms body/collider references are valid.
//! - **Activate**: Moves the prevalidated `PhysicsWorld` into an active receipt.
//!   No new `create_body`/`create_collider` calls.
//! - **Teardown**: Removes colliders before bodies from the receipt's world.
//!   Returns failure without relinquishing the receipt.
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
    ActiveBridgeState, AppBridge, BehaviorEntityRecipe, EntityCollisionRecipe, LightEntityRecipe,
    PreparedBridgeState, WorldCollisionRecipe,
};

// ── Prepared Physics State ─────────────────────────────────────────────

/// Candidate-private physics world built during prepare.
///
/// All bodies and colliders are created in this private world during prepare.
/// If construction fails, partial state is undone before returning an error.
pub struct PhysicsPreparedState {
    /// The candidate-private physics world with pre-created bodies/colliders.
    pub world: physics::PhysicsWorld,
    /// Entity index → body ID mapping for transform sync.
    pub entity_body_map: HashMap<u32, physics::PhysicsBodyId>,
    /// Entity index → collider IDs for teardown removal.
    pub entity_collider_map: HashMap<u32, Vec<physics::PhysicsColliderId>>,
    /// World body ID for the static trimesh.
    pub world_body_id: Option<physics::PhysicsBodyId>,
    /// World collider IDs.
    pub world_collider_ids: Vec<physics::PhysicsColliderId>,
    /// All body IDs in creation order (for teardown).
    pub all_body_ids: Vec<physics::PhysicsBodyId>,
    /// All collider IDs in creation order (for teardown).
    pub all_collider_ids: Vec<physics::PhysicsColliderId>,
}

impl std::fmt::Debug for PhysicsPreparedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsPreparedState")
            .field("entity_body_map", &self.entity_body_map)
            .field("all_body_count", &self.all_body_ids.len())
            .field("all_collider_count", &self.all_collider_ids.len())
            .finish()
    }
}

impl PreparedBridgeState for PhysicsPreparedState {
    fn registration_name(&self) -> &str {
        "physics"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Active Physics State ───────────────────────────────────────────────

/// Published physics world moved into the active receipt during activation.
///
/// Owns the complete `PhysicsWorld` with all BSP bodies and colliders.
/// Teardown removes resources in colliders-before-bodies order.
pub struct PhysicsActiveState {
    /// The active physics world.
    pub world: physics::PhysicsWorld,
    /// Entity index → body ID for transform sync.
    pub entity_body_map: HashMap<u32, physics::PhysicsBodyId>,
    /// Entity index → collider IDs for removal.
    pub entity_collider_map: HashMap<u32, Vec<physics::PhysicsColliderId>>,
    /// All body IDs in creation order.
    pub all_body_ids: Vec<physics::PhysicsBodyId>,
    /// All collider IDs in creation order.
    pub all_collider_ids: Vec<physics::PhysicsColliderId>,
}

impl std::fmt::Debug for PhysicsActiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsActiveState")
            .field("entity_body_map", &self.entity_body_map)
            .field("all_body_count", &self.all_body_ids.len())
            .field("all_collider_count", &self.all_collider_ids.len())
            .finish()
    }
}

impl ActiveBridgeState for PhysicsActiveState {
    fn registration_name(&self) -> &str {
        "physics"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Physics Bridge ─────────────────────────────────────────────────────

/// A physics bridge that creates Rapier resources from BSP collision recipes.
///
/// The bridge builds a candidate-private `PhysicsWorld` during prepare.
/// Activation moves this world into the active receipt. No post-activation
/// world insertion is performed.
pub struct PhysicsBridge {
    /// Entity index → body ID for external query (populated from active receipt).
    pub entity_bodies: HashMap<u32, physics::PhysicsBodyId>,
}

impl PhysicsBridge {
    /// Create a new physics bridge.
    pub fn new() -> Self {
        Self {
            entity_bodies: HashMap::new(),
        }
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
    ) -> Result<Box<dyn PreparedBridgeState>, String> {
        let mut world = physics::PhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);

        let mut entity_body_map: HashMap<u32, physics::PhysicsBodyId> = HashMap::new();
        let mut entity_collider_map: HashMap<u32, Vec<physics::PhysicsColliderId>> = HashMap::new();
        let mut all_body_ids: Vec<physics::PhysicsBodyId> = Vec::new();
        let mut all_collider_ids: Vec<physics::PhysicsColliderId> = Vec::new();
        let mut world_collider_ids: Vec<physics::PhysicsColliderId> = Vec::new();

        // Track what we've created so we can undo on failure.
        let clean_undo = |w: &mut physics::PhysicsWorld,
                              body_ids: &[physics::PhysicsBodyId],
                              collider_ids: &[physics::PhysicsColliderId]| {
            for cid in collider_ids.iter().rev() {
                w.remove_collider(cid);
            }
            for bid in body_ids.iter().rev() {
                w.remove_body(bid);
            }
        };

        // ── World static collision ────────────────────────────────────
        let world_body_id = if !world_collider.planes.is_empty() {
            let body_id = physics::PhysicsBodyId::new("bsp.world");
            let body_desc = physics::BodyDescriptor::new(
                body_id.clone(),
                physics::BodyKind::Static,
                [0.0, 0.0, 0.0],
            );
            world
                .create_body(body_desc)
                .map_err(|e| format!("failed to create world body: {e}"))?;
            all_body_ids.push(body_id.clone());

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

            // Create body in candidate-private world
            if let Err(e) = world.create_body(body_desc) {
                clean_undo(&mut world, &all_body_ids, &all_collider_ids);
                return Err(format!(
                    "failed to create body for entity {}: {}",
                    entity_index, e
                ));
            }
            all_body_ids.push(body_id.clone());
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

                    if let Err(e) = world.create_collider(collider_desc) {
                        clean_undo(&mut world, &all_body_ids, &all_collider_ids);
                        return Err(format!(
                            "failed to create collider for entity {} piece {}: {}",
                            entity_index, piece_idx, e
                        ));
                    }
                    all_collider_ids.push(collider_id.clone());
                    piece_collider_ids.push(collider_id);
                }
            }

            if !piece_collider_ids.is_empty() {
                entity_collider_map.insert(entity_index, piece_collider_ids);
            }
        }

        Ok(Box::new(PhysicsPreparedState {
            world,
            entity_body_map,
            entity_collider_map,
            world_body_id,
            world_collider_ids,
            all_body_ids,
            all_collider_ids,
        }))
    }

    fn validate(&self, prepared: &dyn PreparedBridgeState) -> Result<(), String> {
        let state: &PhysicsPreparedState = prepared
            .as_any()
            .downcast_ref::<PhysicsPreparedState>()
            .ok_or_else(|| "physics bridge received non-physics prepared state".to_string())?;

        // Verify all body references in colliders have matching bodies
        let body_ids: std::collections::HashSet<_> =
            state.all_body_ids.iter().collect();
        for cid in &state.all_collider_ids {
            // Check collider exists in the world
            if !state.world.collider_exists(cid) {
                return Err(format!("collider {} missing from prepared world", cid));
            }
        }

        // Verify entity body references
        for (entity_idx, body_id) in &state.entity_body_map {
            if !body_ids.contains(body_id) {
                return Err(format!(
                    "entity {} references unknown body {}",
                    entity_idx, body_id
                ));
            }
        }

        Ok(())
    }

    fn activate(&mut self, prepared: &mut dyn PreparedBridgeState) -> Box<dyn ActiveBridgeState> {
        let state: &mut PhysicsPreparedState = prepared
            .as_any_mut()
            .downcast_mut::<PhysicsPreparedState>()
            .expect("physics bridge received non-physics prepared state");

        // Copy entity body map for external query
        self.entity_bodies = state.entity_body_map.clone();

        // Move the world and metadata into the active state.
        // We use take + replace to move out of the prepared state.
        let mut world = physics::PhysicsWorld::new();
        std::mem::swap(&mut world, &mut state.world);

        Box::new(PhysicsActiveState {
            world,
            entity_body_map: std::mem::take(&mut state.entity_body_map),
            entity_collider_map: std::mem::take(&mut state.entity_collider_map),
            all_body_ids: std::mem::take(&mut state.all_body_ids),
            all_collider_ids: std::mem::take(&mut state.all_collider_ids),
        })
    }

    fn teardown(&mut self, active: &mut dyn ActiveBridgeState) -> Result<(), String> {
        let state: &mut PhysicsActiveState = active
            .as_any_mut()
            .downcast_mut::<PhysicsActiveState>()
            .ok_or_else(|| "physics bridge received non-physics active state".to_string())?;

        // Remove colliders before bodies (dependent order)
        for cid in state.all_collider_ids.iter() {
            state.world.remove_collider(cid);
        }
        for bid in state.all_body_ids.iter() {
            state.world.remove_body(bid);
        }

        state.all_collider_ids.clear();
        state.all_body_ids.clear();
        state.entity_body_map.clear();
        state.entity_collider_map.clear();
        self.entity_bodies.clear();

        Ok(())
    }

    fn rollback(&mut self, prepared: &mut dyn PreparedBridgeState) {
        let state: &mut PhysicsPreparedState = prepared
            .as_any_mut()
            .downcast_mut::<PhysicsPreparedState>()
            .expect("physics bridge received non-physics prepared state");

        // Remove colliders before bodies
        for cid in state.all_collider_ids.iter() {
            state.world.remove_collider(cid);
        }
        for bid in state.all_body_ids.iter() {
            state.world.remove_body(bid);
        }

        state.all_collider_ids.clear();
        state.all_body_ids.clear();
        state.entity_body_map.clear();
        state.entity_collider_map.clear();
    }
}
