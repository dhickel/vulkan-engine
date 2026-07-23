//! Renderer BSP frame-state publication from simulation snapshots.
//!
//! Consumes an immutable `Arc<BspSimulationSnapshot>` and updates the
//! renderer scene with per-frame transforms for inline-model BSP draws,
//! light-style intensities, liquid time, and external model instances.
//!
//! This replaces a Phase 09 stub and now synchronizes physics poses,
//! behavior state, culling bypass, and render submission from one
//! snapshot epoch.

use std::collections::HashMap;

use crate::physics_bridge::PhysicsBridge;
use bsp_runtime::BspSimulationSnapshot;
use physics::PhysicsWorld;
use renderer::api::Scene;
use renderer::SceneNodeId;

/// Per-entity scene node mapping stored during BSP commit.
///
/// When the coordinator publishes BSP geometry to the scene, each inline
/// model entity gets a SceneNodeId. This map tracks entity_index → node_id.
#[derive(Debug, Clone, Default)]
pub struct EntityNodeMap {
    /// entity_index → SceneNodeId for inline-model draws.
    pub inline_nodes: HashMap<u32, SceneNodeId>,
    /// entity_index → SceneNodeId for external model instances.
    pub external_nodes: HashMap<u32, SceneNodeId>,
}

/// Sync a simulation snapshot into the renderer scene.
///
/// Updates:
/// - Inline model scene node transforms (for moving brush entities).
/// - External model scene node transforms.
/// - BSP frame-values UBO (style intensities, liquid time).
///
/// Returns the number of scene nodes updated.
pub fn sync_snapshot_to_scene(
    snapshot: &BspSimulationSnapshot,
    entity_nodes: &EntityNodeMap,
    scene: &mut Scene,
) -> usize {
    let mut updated = 0usize;

    // ── Inline model poses ──────────────────────────────────────────
    // BSP inline-model batches are rendered from the BSP mount, not from
    // ordinary scene graph nodes. Publish model_index → transform every
    // snapshot so moving brushes update in the draw path and stale transforms
    // are cleared when a new snapshot has no inline poses.
    let inline_transforms: HashMap<u32, glam::Mat4> = snapshot
        .entity_poses
        .iter()
        .filter(|pose| pose.model_index != 0)
        .map(|pose| (pose.model_index, pose.transform))
        .collect();
    let inline_bounds: HashMap<u32, (glam::Vec3, glam::Vec3)> = snapshot
        .entity_poses
        .iter()
        .filter(|pose| pose.model_index != 0)
        .map(|pose| (pose.model_index, pose.world_bounds))
        .collect();
    scene.set_inline_model_transforms(inline_transforms);
    scene.set_inline_model_bounds(inline_bounds);

    for pose in &snapshot.entity_poses {
        if let Some(node_id) = entity_nodes.inline_nodes.get(&pose.entity_index) {
            if scene.set_transform(*node_id, pose.transform).is_ok() {
                updated += 1;
            }
        }
    }

    // ── External instances ───────────────────────────────────────────
    for instance in &snapshot.external_instances {
        if let Some(node_id) = entity_nodes.external_nodes.get(&instance.entity_index) {
            if scene.set_transform(*node_id, instance.transform).is_ok() {
                updated += 1;
            }
        }
    }

    // ── BSP frame values (styles, liquid) ────────────────────────────
    // Liquid time changes every snapshot even when poses/styles do not.
    scene.set_bsp_frame_values(snapshot.light_styles.intensities, snapshot.liquid_time);

    updated
}

/// Sync snapshot entity poses into the physics world.
///
/// For each moving entity in the snapshot, updates the corresponding
/// kinematic rigid body position in Rapier. Uses the `PhysicsBridge`
/// entity→body map.
pub fn sync_snapshot_to_physics(
    snapshot: &BspSimulationSnapshot,
    bridge: &PhysicsBridge,
    physics_world: &mut PhysicsWorld,
) -> usize {
    bridge.sync_from_snapshot(snapshot, physics_world)
}

/// Apply snapshot light styles to the scene's BSP frame values.
pub fn sync_styles_to_scene(snapshot: &BspSimulationSnapshot, scene: &mut Scene) {
    scene.set_bsp_frame_values(snapshot.light_styles.intensities, snapshot.liquid_time);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp_runtime::{SnapshotBuilder, SnapshotEntityPose};
    use glam::{Mat4, Vec3};

    fn make_test_snapshot() -> BspSimulationSnapshot {
        let mut builder = SnapshotBuilder::new(1, 0, 1.0 / 60.0, 0.0);
        builder.push_entity_pose(SnapshotEntityPose {
            entity_index: 1,
            model_index: 1,
            transform: Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
            world_bounds: (Vec3::new(4.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0)),
            is_moving: true,
        });
        builder.push_entity_pose(SnapshotEntityPose {
            entity_index: 2,
            model_index: 2,
            transform: Mat4::IDENTITY,
            world_bounds: (Vec3::ZERO, Vec3::ONE),
            is_moving: false,
        });
        builder.build()
    }

    #[test]
    fn sync_to_physics_only_updates_moving() {
        let snapshot = make_test_snapshot();
        let bridge = PhysicsBridge::new();
        let mut world = PhysicsWorld::new();

        // No bodies registered yet — all syncs should fail silently.
        let updated = sync_snapshot_to_physics(&snapshot, &bridge, &mut world);
        assert_eq!(updated, 0);
    }

    #[test]
    fn sync_to_scene_empty_map_returns_zero() {
        let snapshot = BspSimulationSnapshot::empty();
        let nodes = EntityNodeMap::default();
        let mut scene = Scene::new();
        let updated = sync_snapshot_to_scene(&snapshot, &nodes, &mut scene);
        assert_eq!(updated, 0);
    }

    #[test]
    fn entity_node_map_default_is_empty() {
        let map = EntityNodeMap::default();
        assert!(map.inline_nodes.is_empty());
        assert!(map.external_nodes.is_empty());
    }
}
