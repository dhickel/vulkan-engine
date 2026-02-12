use crate::data::data_cache::MeshCache;
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use glam::Mat4;

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

pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
}

impl RenderSubmission {
    pub fn new(camera: SceneDataUBO, draw_capacity: usize) -> Self {
        Self {
            camera,
            draw_items: Vec::with_capacity(draw_capacity),
            flags: SubmissionFlags::default(),
            skybox_mesh_id: MeshCache::SKYBOX_MESH,
            skybox_env_id: EnvironmentHandle::new(0, 0),
        }
    }

    pub fn push_draw_item(&mut self, draw_item: FrameDrawItem) {
        self.draw_items.push(draw_item);
    }

    pub fn has_draw_targets(&self) -> bool {
        self.flags.draw_skybox || self.flags.draw_geometry
    }
}
