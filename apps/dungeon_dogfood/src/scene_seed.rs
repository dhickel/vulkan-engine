use glam::Mat4;
use renderer::{
    AssetManager, EnvironmentSource, MeshHandle, PbrMaterialDesc, PointLight, PointLightId, Scene,
    SceneNodeId,
};
use thiserror::Error;

use crate::content::{resolve_content_path, select_light_preset, select_prop_index, ContentPack};
use crate::geometry::build_level_chunks;
use crate::layout::{tile_to_world, ParsedLevel};

#[derive(Debug, Error)]
pub enum SceneSeedError {
    #[error("asset error: {0}")]
    Asset(#[from] renderer::AssetError),
    #[error("scene error: {0}")]
    Scene(#[from] renderer::SceneError),
}

/// Scene resources created from level data
pub struct LevelScene {
    pub dungeon_material: renderer::MaterialHandle,
    pub chunk_meshes: Vec<MeshHandle>,
    pub chunk_nodes: Vec<SceneNodeId>,
    pub light_ids: Vec<PointLightId>,
    pub prop_roots: Vec<SceneNodeId>,
}

impl LevelScene {
    /// Seed scene from parsed level data.
    pub fn from_level(
        level: &ParsedLevel,
        content_pack: &ContentPack,
        scene: &mut Scene,
        assets: &mut AssetManager,
    ) -> Result<Self, SceneSeedError> {
        // Base dungeon material for procedural geometry.
        let dungeon_material = assets.create_material_pbr(PbrMaterialDesc {
            base_color: glam::Vec4::new(0.42, 0.44, 0.46, 1.0),
            metallic: 0.0,
            roughness: 0.86,
            ..Default::default()
        })?;

        // Upload chunked dungeon mesh geometry.
        let chunks = build_level_chunks(level, dungeon_material);
        let mut chunk_meshes = Vec::with_capacity(chunks.len());
        let mut chunk_nodes = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let mesh = assets.upload_procedural_mesh(chunk.mesh)?;
            let node = scene.create_node(None, Mat4::IDENTITY)?;
            scene.add_mesh(node, mesh)?;
            chunk_meshes.push(mesh);
            chunk_nodes.push(node);
        }

        // Spawn point lights from markers using deterministic 7/2/1 preset mapping.
        let mut light_ids = Vec::with_capacity(level.light_markers.len());
        for (marker_idx, &(x, y)) in level.light_markers.iter().enumerate() {
            let preset_id = select_light_preset(marker_idx);
            let Some(light_preset) = content_pack.light_preset(preset_id) else {
                log::warn!(
                    "Skipping light marker {} at ({}, {}): missing preset {:?}",
                    marker_idx,
                    x,
                    y,
                    preset_id
                );
                continue;
            };

            let world_pos = tile_to_world(x, y) + glam::Vec3::new(0.5, 1.7, -0.5);
            let light_id = scene.create_point_light(PointLight {
                position: world_pos,
                color: glam::Vec3::new(
                    light_preset.color[0],
                    light_preset.color[1],
                    light_preset.color[2],
                ),
                intensity: light_preset.intensity,
                range: light_preset.range,
            })?;
            light_ids.push(light_id);
        }

        // Spawn deterministic props from model markers.
        let enabled_props = content_pack.enabled_props();
        let mut prop_roots = Vec::new();
        let mut warned_torch_unlit_fallback = false;

        for (marker_idx, &(x, y)) in level.model_markers.iter().enumerate() {
            let prop_idx = select_prop_index(marker_idx, enabled_props.len());
            let prop = enabled_props[prop_idx];
            let placement = prop.placement_policy();

            if placement.prefer_unlit_fallback && !warned_torch_unlit_fallback {
                warned_torch_unlit_fallback = true;
                log::warn!(
                    "Content pack prop '{}' prefers unlit fallback; keeping this as temporary non-PBR behavior",
                    prop.id
                );
            }

            let prop_path = resolve_content_path(prop.path.as_path());
            let fragment = match assets.load_model(&prop_path) {
                Ok(fragment) => fragment,
                Err(err) => {
                    log::warn!(
                        "Skipping prop '{}' at marker {} ({}, {}): model load failed: {}",
                        prop.id,
                        marker_idx,
                        x,
                        y,
                        err
                    );
                    continue;
                }
            };

            let mount = match scene.merge_fragment(None, fragment) {
                Ok(mount) => mount,
                Err(err) => {
                    log::warn!(
                        "Skipping prop '{}' at marker {} ({}, {}): merge failed: {}",
                        prop.id,
                        marker_idx,
                        x,
                        y,
                        err
                    );
                    continue;
                }
            };

            let world_pos = tile_to_world(x, y) + glam::Vec3::new(0.5, placement.y_offset, -0.5);
            let transform = Mat4::from_scale_rotation_translation(
                placement.scale,
                glam::Quat::from_rotation_y(placement.yaw_radians),
                world_pos,
            );

            if let Err(err) = scene.set_transform(mount.mounted_root, transform) {
                log::warn!(
                    "Skipping prop '{}' at marker {} ({}, {}): transform failed: {}",
                    prop.id,
                    marker_idx,
                    x,
                    y,
                    err
                );
                continue;
            }

            prop_roots.push(mount.mounted_root);
        }

        // Load primary environment from content pack (warn-only on failure).
        let primary_env = content_pack.primary_environment();
        if let Err(err) = load_environment(scene, assets, primary_env.path.as_path()) {
            log::warn!(
                "Failed to load environment '{}': {}. Continuing with renderer default environment.",
                primary_env.id,
                err
            );
        }

        Ok(Self {
            dungeon_material,
            chunk_meshes,
            chunk_nodes,
            light_ids,
            prop_roots,
        })
    }
}

fn load_environment(
    scene: &mut Scene,
    assets: &mut AssetManager,
    path: &std::path::Path,
) -> Result<(), renderer::AssetError> {
    let resolved_path = resolve_content_path(path);
    let handle = assets.load_environment(EnvironmentSource::Auto(resolved_path))?;
    scene.set_skybox(handle);
    Ok(())
}
