//! # Per-Frame Render Submission
//!
//! Flat draw payload emitted by `SceneWorld` and consumed by the rendergraph.
//! Keeps only stable handles and transforms so render code can resolve cache data safely.

use crate::data::data_cache::MeshCache;
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use glam::{Mat4, Vec3};

/// Maximum number of directional lights that can be uploaded to GPU per frame.
pub const MAX_DIRECTIONAL_LIGHTS_GPU: usize = 4;
/// Maximum number of point lights that can be uploaded to GPU per frame.
pub const MAX_POINT_LIGHTS_GPU: usize = 16;
/// Maximum number of spot lights that can be uploaded to GPU per frame.
pub const MAX_SPOT_LIGHTS_GPU: usize = 16;

#[derive(Debug, Copy, Clone)]
pub struct SubmissionFlags {
    pub draw_skybox: bool,
    pub draw_geometry: bool,
    pub draw_imgui: bool,
}

impl Default for SubmissionFlags {
    fn default() -> Self {
        Self {
            draw_skybox: true,
            draw_geometry: true,
            draw_imgui: true,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDrawItem {
    pub mesh_id: MeshHandle,
    pub transform: Mat4,
}

#[derive(Debug, Copy, Clone)]
pub struct FramePointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub enable_shadows: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameSpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cos: f32,
    pub outer_cos: f32,
}

pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
    pub point_lights: Vec<FramePointLight>,
    /// First directional light, retained for legacy consumers.
    pub directional_light: Option<FrameDirectionalLight>,
    /// Bounded collection used by multi-directional direct lighting.
    pub directional_lights: Vec<FrameDirectionalLight>,
    pub spot_lights: Vec<FrameSpotLight>,
    pub culling_stats: CullingStats,
    pub bounds_references: Vec<MeshHandle>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct CullingStats {
    pub known_nodes_tested: u32,
    pub proxy_nodes_tested: u32,
    pub conservative_nodes_by_reason: [u32; 5],
    pub subtree_bounds_tested: u32,
    pub subtrees_pruned: u32,
    pub submitted_draw_items: u32,
}

impl RenderSubmission {
    pub fn new(camera: SceneDataUBO, draw_capacity: usize) -> Self {
        Self {
            camera,
            draw_items: Vec::with_capacity(draw_capacity),
            flags: SubmissionFlags::default(),
            skybox_mesh_id: MeshCache::SKYBOX_MESH,
            skybox_env_id: EnvironmentHandle::new(0, 0),
            point_lights: Vec::with_capacity(MAX_POINT_LIGHTS_GPU),
            directional_light: None,
            directional_lights: Vec::with_capacity(MAX_DIRECTIONAL_LIGHTS_GPU),
            spot_lights: Vec::with_capacity(MAX_SPOT_LIGHTS_GPU),
            culling_stats: CullingStats::default(),
            bounds_references: Vec::with_capacity(draw_capacity),
        }
    }

    pub fn push_draw_item(&mut self, draw_item: FrameDrawItem) {
        self.draw_items.push(draw_item);
        self.culling_stats.submitted_draw_items += 1;
    }

    pub fn has_draw_targets(&self) -> bool {
        self.flags.draw_skybox || self.flags.draw_geometry
    }
}
