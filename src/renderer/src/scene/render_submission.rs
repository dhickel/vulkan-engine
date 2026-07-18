//! # Per-Frame Render Submission
//!
//! Flat draw payload emitted by `SceneWorld` and consumed by the rendergraph.
//! Keeps only stable handles and transforms so render code can resolve cache data safely.

use crate::data::data_cache::MeshCache;
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use glam::{Mat4, Vec3};

/// Maximum number of point lights that can be uploaded to GPU per frame.
pub const MAX_POINT_LIGHTS_GPU: usize = 16;

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
}

pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
    pub point_lights: Vec<FramePointLight>,
    pub directional_light: Option<FrameDirectionalLight>,
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
        }
    }

    pub fn push_draw_item(&mut self, draw_item: FrameDrawItem) {
        self.draw_items.push(draw_item);
    }

    pub fn has_draw_targets(&self) -> bool {
        self.flags.draw_skybox || self.flags.draw_geometry
    }
}
