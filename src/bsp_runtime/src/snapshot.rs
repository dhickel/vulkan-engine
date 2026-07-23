//! Neutral simulation snapshot vocabulary.
//!
//! The snapshot is an immutable, deterministic, frame-epoch payload that
//! drives rendering, physics, culling, external props, light styles, liquid
//! time, behavior state, and save capture from one consistent epoch.
//!
//! Ownership:
//! - `bsp_runtime` owns the neutral DTOs (this file).
//! - `apps/bsp_beta` owns the builder, snapshot production, and consumption
//!   (see `apps/bsp_beta/src/snapshot.rs`).

use glam::{Mat4, Vec3};
use std::sync::Arc;

/// Monotonic generation counter embedded in every snapshot so consumers
/// can detect stale data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotGeneration(pub u64);

/// Deterministic simulation epoch: tick, delta, elapsed.
///
/// Every snapshot records the tick that produced it and the dt used for
/// that fixed step. Multiple snapshots may share the same tick (e.g.,
/// rendering interpolated frames) but never different ticks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotEpoch {
    /// Monotonic fixed-step tick.
    pub tick: u64,
    /// Fixed delta-time applied for this step (seconds).
    pub dt: f32,
    /// Accumulated simulation time (seconds). Wraps at 2^32 ticks for
    /// liquid/animation modulo, but the raw u64 tick is definitive.
    pub elapsed: f32,
}

/// A single entity pose in the snapshot — inline model or external instance.
///
/// Transforms are in engine space via the QuakeToEngine contract.
/// Pivot is the model origin in engine space; rotation is around pivot.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotEntityPose {
    /// Source entity index from the BSP entity lump.
    pub entity_index: u32,
    /// Model index (0 = worldspawn, >0 = inline model N).
    pub model_index: u32,
    /// World-space transform (translation + rotation + scale).
    pub transform: Mat4,
    /// Conservative AABB in world space (for culling). Always finite.
    pub world_bounds: (Vec3, Vec3),
    /// Whether this entity is currently in motion (kinematic).
    pub is_moving: bool,
}

/// External model instance identity and placement.
///
/// An external model is a renderer-loadable asset that replaces or
/// supplements a BSP entity representation. It maps through the
/// model-mappings table and uses `package_io` authorization.
#[derive(Debug, Clone)]
pub struct ExternalInstance {
    /// Source entity index.
    pub entity_index: u32,
    /// Resolved engine asset path (authorized by package_io).
    pub asset_path: String,
    /// World-space transform.
    pub transform: Mat4,
    /// Conservative AABB in world space.
    pub world_bounds: (Vec3, Vec3),
    /// Whether this external model has been loaded by the renderer.
    pub loaded: bool,
}

/// Snapshot of 64 light-style intensities.
///
/// Each style index (0..63) maps to a float multiplier (0.0..1.0).
/// Style 0 is always the static (default) lightmap and is pinned to 1.0.
/// The GPU consumes these as a 64-element float array (set 2 UBO).
#[derive(Debug, Clone)]
pub struct SnapshotLightStyles {
    /// Per-style intensity, indexed by style_id. 64 elements.
    pub intensities: [f32; 64],
}

impl Default for SnapshotLightStyles {
    fn default() -> Self {
        let mut intensities = [1.0_f32; 64];
        // Style 0 (static) is always 1.0.
        intensities[0] = 1.0;
        Self { intensities }
    }
}

/// Trigger/target activation state ref carried across snapshot epochs.
///
/// Does not contain full trigger occupant sets — those are app-private.
/// The snapshot carries only what downstream consumers need:
/// targets queued for activation and killtargets.
#[derive(Debug, Clone)]
pub struct SnapshotActivation {
    /// Entity indices that received activation this tick.
    pub activated_entities: Vec<u32>,
    /// Entity indices targeted for kill this tick.
    pub killed_entities: Vec<u32>,
}

/// Immutable simulation snapshot published at each fixed-step boundary.
///
/// All consumers (rendering, physics, culling, external props, styles,
/// liquid time, behavior, save capture) read from the same `Arc`.
#[derive(Debug, Clone)]
pub struct BspSimulationSnapshot {
    /// Monotonic generation:
    pub generation: SnapshotGeneration,
    /// Simulation epoch for this snapshot.
    pub epoch: SnapshotEpoch,
    /// Sorted entity poses (by entity_index).
    pub entity_poses: Vec<SnapshotEntityPose>,
    /// External model instances.
    pub external_instances: Vec<ExternalInstance>,
    /// 64 light-style intensities.
    pub light_styles: SnapshotLightStyles,
    /// Liquid animation time (seconds). Wraps at u16 max for GPU packing.
    pub liquid_time: f32,
    /// Activation events since the last snapshot.
    pub activations: SnapshotActivation,
    /// Whether any moving entity changed pose this tick.
    pub any_motion: bool,
    /// Whether any light style changed this tick.
    pub any_style_change: bool,
}

impl BspSimulationSnapshot {
    /// Create an empty snapshot at generation 0, tick 0.
    pub fn empty() -> Self {
        Self {
            generation: SnapshotGeneration(0),
            epoch: SnapshotEpoch {
                tick: 0,
                dt: 0.0,
                elapsed: 0.0,
            },
            entity_poses: Vec::new(),
            external_instances: Vec::new(),
            light_styles: SnapshotLightStyles::default(),
            liquid_time: 0.0,
            activations: SnapshotActivation {
                activated_entities: Vec::new(),
                killed_entities: Vec::new(),
            },
            any_motion: false,
            any_style_change: false,
        }
    }

    /// Wrap in an `Arc` for sharing among consumers.
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

// ── Snapshot Builder (neutral vocabulary) ──────────────────────────

/// Neutral builder API that `apps/bsp_beta` uses to construct snapshots.
///
/// The builder owns no physics or rendering state. It accepts entity pose
/// batches, external instances, style intensities, and activation events.
#[derive(Debug)]
pub struct SnapshotBuilder {
    generation: SnapshotGeneration,
    tick: u64,
    dt: f32,
    elapsed: f32,
    entity_poses: Vec<SnapshotEntityPose>,
    external_instances: Vec<ExternalInstance>,
    light_styles: SnapshotLightStyles,
    liquid_time: f32,
    activations: SnapshotActivation,
    any_motion: bool,
    any_style_change: bool,
}

impl SnapshotBuilder {
    /// Begin a new snapshot at the given generation, tick, and dt.
    pub fn new(generation: u64, tick: u64, dt: f32, elapsed: f32) -> Self {
        Self {
            generation: SnapshotGeneration(generation),
            tick,
            dt,
            elapsed,
            entity_poses: Vec::new(),
            external_instances: Vec::new(),
            light_styles: SnapshotLightStyles::default(),
            liquid_time: 0.0,
            activations: SnapshotActivation {
                activated_entities: Vec::new(),
                killed_entities: Vec::new(),
            },
            any_motion: false,
            any_style_change: false,
        }
    }

    /// Capacity hint for entity poses.
    pub fn with_entity_capacity(mut self, cap: usize) -> Self {
        self.entity_poses.reserve(cap);
        self
    }

    /// Add an entity pose.
    pub fn push_entity_pose(&mut self, pose: SnapshotEntityPose) {
        if pose.is_moving {
            self.any_motion = true;
        }
        self.entity_poses.push(pose);
    }

    /// Add an external instance.
    pub fn push_external_instance(&mut self, instance: ExternalInstance) {
        self.external_instances.push(instance);
    }

    /// Set a light style intensity.
    ///
    /// # Panics
    /// Panics if `style_id > 63`.
    pub fn set_light_style(&mut self, style_id: u8, intensity: f32) {
        assert!(style_id <= 63, "light style ID must be 0..63");
        let clamped = intensity.clamp(0.0, 1.0);
        if (self.light_styles.intensities[style_id as usize] - clamped).abs() > 1e-6 {
            self.any_style_change = true;
        }
        self.light_styles.intensities[style_id as usize] = clamped;
    }

    /// Copy all light styles from another set.
    pub fn set_light_styles(&mut self, styles: &SnapshotLightStyles) {
        self.light_styles = styles.clone();
        self.any_style_change = true;
    }

    /// Set liquid animation time.
    pub fn set_liquid_time(&mut self, time: f32) {
        self.liquid_time = time;
    }

    /// Record an activation event.
    pub fn push_activation(&mut self, entity_index: u32) {
        self.activations.activated_entities.push(entity_index);
    }

    /// Record a kill event.
    pub fn push_kill(&mut self, entity_index: u32) {
        self.activations.killed_entities.push(entity_index);
    }

    /// Sort entity poses by entity_index (required for deterministic output).
    pub fn sort_poses(&mut self) {
        self.entity_poses.sort_by_key(|p| p.entity_index);
    }

    /// Sort external instances by entity_index.
    pub fn sort_external_instances(&mut self) {
        self.external_instances.sort_by_key(|i| i.entity_index);
    }

    /// Finalize into a `BspSimulationSnapshot`.
    pub fn build(mut self) -> BspSimulationSnapshot {
        self.sort_poses();
        self.sort_external_instances();
        BspSimulationSnapshot {
            generation: self.generation,
            epoch: SnapshotEpoch {
                tick: self.tick,
                dt: self.dt,
                elapsed: self.elapsed,
            },
            entity_poses: self.entity_poses,
            external_instances: self.external_instances,
            light_styles: self.light_styles,
            liquid_time: self.liquid_time,
            activations: self.activations,
            any_motion: self.any_motion,
            any_style_change: self.any_style_change,
        }
    }

    /// Build wrapped in `Arc`.
    pub fn build_shared(self) -> Arc<BspSimulationSnapshot> {
        self.build().into_shared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_is_all_defaults() {
        let snap = BspSimulationSnapshot::empty();
        assert_eq!(snap.generation.0, 0);
        assert_eq!(snap.epoch.tick, 0);
        assert!(snap.entity_poses.is_empty());
        assert!(!snap.any_motion);
        assert!(!snap.any_style_change);
    }

    #[test]
    fn builder_sorts_poses() {
        let mut builder = SnapshotBuilder::new(1, 0, 1.0 / 60.0, 0.0);
        builder.push_entity_pose(SnapshotEntityPose {
            entity_index: 10,
            model_index: 1,
            transform: Mat4::IDENTITY,
            world_bounds: (Vec3::ZERO, Vec3::ONE),
            is_moving: false,
        });
        builder.push_entity_pose(SnapshotEntityPose {
            entity_index: 5,
            model_index: 2,
            transform: Mat4::IDENTITY,
            world_bounds: (Vec3::ZERO, Vec3::ONE),
            is_moving: false,
        });
        let snap = builder.build();
        assert_eq!(snap.entity_poses[0].entity_index, 5);
        assert_eq!(snap.entity_poses[1].entity_index, 10);
    }

    #[test]
    fn any_motion_flagged_when_entity_is_moving() {
        let mut builder = SnapshotBuilder::new(1, 0, 1.0 / 60.0, 0.0);
        builder.push_entity_pose(SnapshotEntityPose {
            entity_index: 1,
            model_index: 1,
            transform: Mat4::IDENTITY,
            world_bounds: (Vec3::ZERO, Vec3::ONE),
            is_moving: true,
        });
        let snap = builder.build();
        assert!(snap.any_motion);
    }

    #[test]
    fn light_style_bounds_checked() {
        let mut builder = SnapshotBuilder::new(1, 0, 0.016, 0.0);
        builder.set_light_style(5, 0.5);
        let snap = builder.build();
        assert!((snap.light_styles.intensities[5] - 0.5).abs() < 1e-6);
        // Default for unused styles is 1.0.
        assert!((snap.light_styles.intensities[0] - 1.0).abs() < 1e-6);
        assert!((snap.light_styles.intensities[10] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn style_change_detected() {
        let mut builder = SnapshotBuilder::new(1, 0, 0.016, 0.0);
        // Default 1.0 — setting to 0.5 is a change.
        builder.set_light_style(3, 0.5);
        assert!(builder.any_style_change);

        let mut builder2 = SnapshotBuilder::new(1, 0, 0.016, 0.0);
        // Default 1.0 — staying at 1.0 is not a change.
        builder2.set_light_style(3, 1.0);
        assert!(!builder2.any_style_change);
    }

    #[test]
    #[should_panic(expected = "light style ID must be 0..63")]
    fn light_style_panics_above_63() {
        let mut builder = SnapshotBuilder::new(0, 0, 0.0, 0.0);
        builder.set_light_style(64, 1.0);
    }

    #[test]
    fn activation_events_recorded() {
        let mut builder = SnapshotBuilder::new(1, 0, 0.016, 0.0);
        builder.push_activation(42);
        builder.push_kill(99);
        let snap = builder.build();
        assert_eq!(snap.activations.activated_entities, vec![42]);
        assert_eq!(snap.activations.killed_entities, vec![99]);
    }
}
