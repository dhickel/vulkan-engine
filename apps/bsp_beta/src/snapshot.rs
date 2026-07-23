//! App-owned simulation snapshot producer.
//!
//! At each fixed-step boundary, this module:
//! 1. Advances behavior state machines (doors, buttons, platforms, triggers).
//! 2. Computes per-entity world poses and conservative AABBs.
//! 3. Resolves trigger/target activation cascades.
//! 4. Computes light-style intensities.
//! 5. Builds external model instances from model-mappings.
//! 6. Publishes an immutable `Arc<BspSimulationSnapshot>`.
//!
//! All downstream consumers (renderer, physics, culling, save capture)
//! read the same snapshot epoch.

use std::collections::HashMap;
use std::sync::Arc;

use crate::runtime_bridge::RuntimeBridge;
use bsp_runtime::{
    BspSimulationSnapshot, ExternalInstance, SnapshotBuilder, SnapshotEntityPose,
    SnapshotLightStyles,
};
use glam::{Mat4, Vec3};

/// Model mapping entry: how to resolve a BSP entity to an engine asset.
#[derive(Debug, Clone)]
pub struct ModelMappingEntry {
    /// Source model string from the BSP entity (e.g., "models/player.mdl").
    pub source_model: String,
    /// Engine asset path authorized by package_io.
    pub asset_path: String,
}

/// Parsed model-mappings table.
#[derive(Debug, Clone, Default)]
pub struct ModelMappings {
    /// Entity override: entity_index → asset_path. Highest precedence.
    pub entity_overrides: HashMap<u32, String>,
    /// Exact source model match: source_model → asset_path.
    pub source_models: HashMap<String, String>,
    /// Exact classname match: classname → asset_path.
    pub classnames: HashMap<String, String>,
    /// Dev proxy model path (used when nothing else matches).
    pub dev_proxy: Option<String>,
}

impl ModelMappings {
    /// Resolve the asset path for an entity, following the precedence order:
    /// entity override > exact source-model > exact classname > dev proxy.
    ///
    /// Returns `None` if no mapping matches (entity is invisible + diagnostic).
    pub fn resolve(
        &self,
        entity_index: u32,
        source_model: Option<&str>,
        classname: &str,
    ) -> Option<&str> {
        // 1. Entity override (highest precedence)
        if let Some(path) = self.entity_overrides.get(&entity_index) {
            return Some(path.as_str());
        }

        // 2. Exact source model match
        if let Some(model) = source_model {
            if let Some(path) = self.source_models.get(model) {
                return Some(path.as_str());
            }
        }

        // 3. Exact classname match
        if let Some(path) = self.classnames.get(classname) {
            return Some(path.as_str());
        }

        // 4. Dev proxy (lowest precedence)
        self.dev_proxy.as_deref()
    }

    /// Parse model-mappings TOML into structured table.
    /// Rejects wildcards — all keys must be exact strings.
    pub fn parse(toml_str: &str) -> Result<Self, String> {
        let root: toml::Value =
            toml::from_str(toml_str).map_err(|e| format!("model-mappings TOML parse: {e}"))?;

        let mut mappings = Self::default();

        let models_table = root
            .get("models")
            .and_then(|v| v.as_table())
            .ok_or_else(|| "model-mappings: missing [models] table".to_string())?;

        for (key, value) in models_table {
            let path = value
                .as_str()
                .ok_or_else(|| format!("model-mappings: value for '{key}' is not a string"))?;

            match key.as_str() {
                k if k.starts_with("entity/") => {
                    // Entity override: "entity/<index>" → path
                    let index_str = k.strip_prefix("entity/").unwrap();
                    let index: u32 = index_str.parse().map_err(|_| {
                        format!("model-mappings: invalid entity index '{index_str}'")
                    })?;
                    if mappings
                        .entity_overrides
                        .insert(index, path.to_string())
                        .is_some()
                    {
                        return Err(format!("model-mappings: duplicate entity override for {k}"));
                    }
                }
                k if k.starts_with("model/") => {
                    // Exact source model: "model/<source>" → path
                    let model = k.strip_prefix("model/").unwrap();
                    if model.contains('*') || model.contains('?') {
                        return Err(format!(
                            "model-mappings: wildcard rejected in source model '{model}'"
                        ));
                    }
                    if mappings
                        .source_models
                        .insert(model.to_string(), path.to_string())
                        .is_some()
                    {
                        return Err(format!("model-mappings: duplicate source model '{model}'"));
                    }
                }
                k if k.starts_with("classname/") => {
                    // Exact classname: "classname/<name>" → path
                    let class = k.strip_prefix("classname/").unwrap();
                    if class.contains('*') || class.contains('?') {
                        return Err(format!(
                            "model-mappings: wildcard rejected in classname '{class}'"
                        ));
                    }
                    if mappings
                        .classnames
                        .insert(class.to_string(), path.to_string())
                        .is_some()
                    {
                        return Err(format!("model-mappings: duplicate classname '{class}'"));
                    }
                }
                "dev_proxy" => {
                    mappings.dev_proxy = Some(path.to_string());
                }
                other => {
                    return Err(format!(
                        "model-mappings: unrecognized key prefix '{other}'. Use entity/N, model/X, classname/X, or dev_proxy"
                    ));
                }
            }
        }

        Ok(mappings)
    }
}

/// App-owned simulation snapshot producer.
///
/// Owns the snapshot generation counter and produces a new
/// `Arc<BspSimulationSnapshot>` at each fixed-step tick.
pub struct SnapshotProducer {
    /// Monotonic snapshot generation.
    generation: u64,
    /// Accumulated simulation tick.
    tick: u64,
    /// Accumulated elapsed time.
    elapsed: f32,
    /// Model mappings for external instances.
    model_mappings: ModelMappings,
    /// Light-style state carried across ticks.
    light_styles: SnapshotLightStyles,
    /// Last known entity poses (for change detection).
    last_poses: HashMap<u32, SnapshotEntityPose>,
}

impl SnapshotProducer {
    /// Create a new snapshot producer with given model mappings.
    pub fn new(model_mappings: ModelMappings) -> Self {
        Self {
            generation: 0,
            tick: 0,
            elapsed: 0.0,
            model_mappings,
            light_styles: SnapshotLightStyles::default(),
            last_poses: HashMap::new(),
        }
    }

    /// Produce a new snapshot at the given fixed dt.
    ///
    /// Advances the runtime bridge, collects entity poses, resolves
    /// external instances, computes light styles, and builds the snapshot.
    pub fn produce(
        &mut self,
        dt: f32,
        runtime: &mut RuntimeBridge,
        inline_model_infos: &[InlineModelInfo],
        entity_classnames: &HashMap<u32, String>,
        entity_source_models: &HashMap<u32, String>,
    ) -> Arc<BspSimulationSnapshot> {
        self.generation += 1;
        self.tick += 1;
        self.elapsed += dt;

        let mut builder = SnapshotBuilder::new(self.generation, self.tick, dt, self.elapsed)
            .with_entity_capacity(inline_model_infos.len());

        // ── 1. Advance behavior ───────────────────────────────────────
        let pose_updates = runtime.update(dt);

        // ── 2. Collect entity poses from behavior output ──────────────
        let mut current_poses: HashMap<u32, [f32; 3]> = HashMap::new();
        for (ei, pos) in &pose_updates {
            current_poses.insert(*ei, *pos);
        }

        // Fill in static entities at their origin
        for info in inline_model_infos {
            let position = current_poses
                .get(&info.entity_index)
                .copied()
                .unwrap_or(info.origin);

            let is_moving = runtime.is_moving(info.entity_index);
            let transform = compute_entity_transform(
                position,
                info.angles,
                info.scale,
                info.origin, // pivot
            );

            // Compute conservative AABB from the model's local bounds
            // transformed by the world transform.
            let world_bounds = compute_world_aabb(&info.local_mins, &info.local_maxs, &transform);

            let pose = SnapshotEntityPose {
                entity_index: info.entity_index,
                model_index: info.model_index,
                transform,
                world_bounds,
                is_moving,
            };

            builder.push_entity_pose(pose);
        }

        // ── 3. Resolve external instances ────────────────────────────
        for info in inline_model_infos {
            let classname = entity_classnames
                .get(&info.entity_index)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let source_model = entity_source_models
                .get(&info.entity_index)
                .map(|s| s.as_str());

            let Some(asset_path) =
                self.model_mappings
                    .resolve(info.entity_index, source_model, classname)
            else {
                continue;
            };

            let position = current_poses
                .get(&info.entity_index)
                .copied()
                .unwrap_or(info.origin);

            let transform =
                compute_entity_transform(position, info.angles, info.scale, info.origin);
            let world_bounds = compute_world_aabb(&info.local_mins, &info.local_maxs, &transform);

            builder.push_external_instance(ExternalInstance {
                entity_index: info.entity_index,
                asset_path: asset_path.to_string(),
                transform,
                world_bounds,
                loaded: false, // renderer sets this after async load
            });
        }

        // ── 4. Compute light-style intensities ─────────────────────────
        // Start with the previous tick's light styles as base.
        builder.set_light_styles(&self.light_styles);
        // Override with runtime state (active/inactive styles).
        for (style_name, state) in &runtime.adapter.light_styles {
            if let Ok(idx) = style_name.parse::<u8>() {
                if idx <= 63 {
                    let intensity = if state.active { state.intensity } else { 0.0 };
                    builder.set_light_style(idx, intensity);
                }
            }
        }

        // ── 5. Liquid time ─────────────────────────────────────────────
        builder.set_liquid_time(self.elapsed);

        // ── 6. Detect pose changes ─────────────────────────────────────
        self.last_poses.clear();
        // We'll rebuild last_poses after build (next iteration)

        let snapshot = builder.build_shared();

        // Update light styles for next tick
        self.light_styles = snapshot.light_styles.clone();

        snapshot
    }

    /// Get the current generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Lightweight info about an inline model needed for pose computation.
#[derive(Debug, Clone)]
pub struct InlineModelInfo {
    pub entity_index: u32,
    pub model_index: u32,
    pub origin: [f32; 3],
    pub angles: Option<[f32; 3]>,
    pub scale: Option<f32>,
    pub local_mins: [f32; 3],
    pub local_maxs: [f32; 3],
}

/// Compute a world-space transform for an entity.
///
/// Transform = translate(origin) * rotate(angles) * scale * translate(-pivot),
/// where the pivot is the model origin (the point around which rotation occurs).
fn compute_entity_transform(
    position: [f32; 3],
    angles: Option<[f32; 3]>,
    scale: Option<f32>,
    pivot: [f32; 3],
) -> Mat4 {
    let scale_val = scale.unwrap_or(1.0);
    let translation = Mat4::from_translation(Vec3::from_array(position));
    let pivot_translate = Mat4::from_translation(Vec3::from_array(pivot));
    let pivot_neg = Mat4::from_translation(-Vec3::from_array(pivot));

    let rotation = if let Some(angles) = angles {
        let pitch = angles[0].to_radians();
        let yaw = angles[1].to_radians();
        let roll = angles[2].to_radians();
        Mat4::from_euler(glam::EulerRot::YXZ, yaw, pitch, roll)
    } else {
        Mat4::IDENTITY
    };

    let scale_mat = Mat4::from_scale(Vec3::splat(scale_val));

    // T * R * S around pivot: translate(pivot) * R * S * translate(-pivot)
    translation * pivot_translate * rotation * scale_mat * pivot_neg
}

/// Compute a conservative world-space AABB from local bounds and a world transform.
pub fn compute_world_aabb(
    local_mins: &[f32; 3],
    local_maxs: &[f32; 3],
    transform: &Mat4,
) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(local_mins[0], local_mins[1], local_mins[2]),
        Vec3::new(local_maxs[0], local_mins[1], local_mins[2]),
        Vec3::new(local_mins[0], local_maxs[1], local_mins[2]),
        Vec3::new(local_mins[0], local_mins[1], local_maxs[2]),
        Vec3::new(local_maxs[0], local_maxs[1], local_mins[2]),
        Vec3::new(local_maxs[0], local_mins[1], local_maxs[2]),
        Vec3::new(local_mins[0], local_maxs[1], local_maxs[2]),
        Vec3::new(local_maxs[0], local_maxs[1], local_maxs[2]),
    ];

    let mut world_min = Vec3::splat(f32::MAX);
    let mut world_max = Vec3::splat(f32::MIN);

    for corner in &corners {
        let wc = transform.transform_point3(*corner);
        world_min = world_min.min(wc);
        world_max = world_max.max(wc);
    }

    (world_min, world_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_mappings_parse_empty() {
        let toml = "[models]\n";
        let mappings = ModelMappings::parse(toml).unwrap();
        assert!(mappings.entity_overrides.is_empty());
    }

    #[test]
    fn model_mappings_parse_full() {
        let toml = r#"
[models]
"entity/5" = "assets/models/override.gltf"
"model/props/crate.mdl" = "assets/models/crate.gltf"
"classname/func_door" = "assets/models/door.gltf"
dev_proxy = "assets/models/proxy.gltf"
"#;
        let mappings = ModelMappings::parse(toml).unwrap();
        assert_eq!(
            mappings.entity_overrides.get(&5).map(|s| s.as_str()),
            Some("assets/models/override.gltf")
        );
        assert_eq!(
            mappings
                .source_models
                .get("props/crate.mdl")
                .map(|s| s.as_str()),
            Some("assets/models/crate.gltf")
        );
        assert_eq!(
            mappings.classnames.get("func_door").map(|s| s.as_str()),
            Some("assets/models/door.gltf")
        );
        assert_eq!(
            mappings.dev_proxy.as_deref(),
            Some("assets/models/proxy.gltf")
        );
    }

    #[test]
    fn model_mappings_precedence() {
        let toml = r#"
[models]
"entity/10" = "assets/a.gltf"
"model/props/barrel.mdl" = "assets/b.gltf"
"classname/func_door" = "assets/c.gltf"
"#;
        let mappings = ModelMappings::parse(toml).unwrap();

        // Entity override wins
        assert_eq!(
            mappings.resolve(10, Some("props/barrel.mdl"), "func_door"),
            Some("assets/a.gltf")
        );
        // Source model wins when no entity override
        assert_eq!(
            mappings.resolve(99, Some("props/barrel.mdl"), "func_door"),
            Some("assets/b.gltf")
        );
        // Classname wins when no source model
        assert_eq!(
            mappings.resolve(99, None, "func_door"),
            Some("assets/c.gltf")
        );
        // Nothing matches
        assert_eq!(mappings.resolve(99, None, "unknown"), None);
    }

    #[test]
    fn model_mappings_reject_wildcards() {
        let toml = r#"
[models]
"model/props/*" = "assets/a.gltf"
"#;
        assert!(ModelMappings::parse(toml).is_err());
    }

    #[test]
    fn compute_world_aabb_identity() {
        let transform = Mat4::IDENTITY;
        let (wmin, wmax) = compute_world_aabb(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0], &transform);
        assert!((wmin - Vec3::new(-1.0, -1.0, -1.0)).length() < 0.001);
        assert!((wmax - Vec3::new(1.0, 1.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn compute_world_aabb_translated() {
        let transform = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let (wmin, wmax) = compute_world_aabb(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0], &transform);
        assert!((wmin - Vec3::new(9.0, -1.0, -1.0)).length() < 0.001);
        assert!((wmax - Vec3::new(11.0, 1.0, 1.0)).length() < 0.001);
    }

    #[test]
    fn snapshot_producer_increments_generation() {
        let mappings = ModelMappings::default();
        let mut producer = SnapshotProducer::new(mappings);
        let mut runtime = RuntimeBridge::new();

        let snap = producer.produce(
            1.0 / 60.0,
            &mut runtime,
            &[],
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(snap.generation.0, 1);
        assert_eq!(snap.epoch.tick, 1);
    }
}
