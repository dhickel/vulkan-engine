use renderer::{AssetManager, PbrMaterialDesc, PointLight, Scene};
use thiserror::Error;

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
    // Cached material handles for dungeon geometry
    // (Geometry generation happens in Phase 04)
}

impl LevelScene {
    /// Seed scene from parsed level data
    ///
    /// Phase 03 responsibilities:
    /// - Create base materials (procedural, not loaded from files)
    /// - Spawn point lights from level markers
    ///
    /// Phase 04 will add:
    /// - Procedural geometry generation
    /// - Model prop instantiation
    /// - Collision baker
    pub fn from_level(
        level: &ParsedLevel,
        scene: &mut Scene,
        _assets: &mut AssetManager,
    ) -> Result<Self, SceneSeedError> {
        // Spawn point lights from markers
        for &(x, y) in &level.light_markers {
            // Convert tile coordinates to world space
            // Offset light position to center of tile and raise above floor
            let world_pos = tile_to_world(x, y) + glam::Vec3::new(0.5, 1.7, -0.5);

            scene.create_point_light(PointLight {
                position: world_pos,
                color: glam::Vec3::new(1.0, 0.6, 0.3), // Warm torch-like color
                intensity: 30.0,
                range: 6.0,
            })?;
        }

        // Phase 04 will add:
        // - Create floor/wall/ceiling materials
        // - Generate dungeon geometry meshes
        // - Instantiate model props from markers
        // - Build collision data

        Ok(Self {})
    }
}
