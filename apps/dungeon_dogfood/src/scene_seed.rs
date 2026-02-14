use glam::Mat4;
use renderer::{AssetManager, MeshHandle, PbrMaterialDesc, PointLight, Scene, SceneNodeId};
use thiserror::Error;

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
}

impl LevelScene {
    /// Seed scene from parsed level data.
    pub fn from_level(
        level: &ParsedLevel,
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

        // Spawn point lights from markers.
        for &(x, y) in &level.light_markers {
            let world_pos = tile_to_world(x, y) + glam::Vec3::new(0.5, 1.7, -0.5);
            scene.create_point_light(PointLight {
                position: world_pos,
                color: glam::Vec3::new(1.0, 0.6, 0.3),
                intensity: 30.0,
                range: 6.0,
            })?;
        }

        Ok(Self {
            dungeon_material,
            chunk_meshes,
            chunk_nodes,
        })
    }
}
