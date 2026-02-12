use crate::data::handles::EnvironmentHandle;
use crate::data::handles::MeshHandle;
use crate::data::gpu_data::SceneDataUBO;
use crate::scene::render_submission::{FrameDrawItem, RenderSubmission};
use glam::{Mat4, Vec3};

pub type SceneNodeId = u32;

#[derive(Clone, Debug)]
pub struct SceneNode {
    pub parent: Option<SceneNodeId>,
    pub children: Vec<SceneNodeId>,
    pub meshes: Vec<MeshHandle>,
    pub local_transform: Mat4,
    pub world_transform: Mat4,
    pub dirty: bool,
    pub layer_mask: u64,
    pub tags: Vec<String>,
}

impl Default for SceneNode {
    fn default() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            meshes: Vec::new(),
            local_transform: Mat4::IDENTITY,
            world_transform: Mat4::IDENTITY,
            dirty: true,
            layer_mask: u64::MAX,
            tags: Vec::new(),
        }
    }
}

pub struct SceneWorld {
    nodes: Vec<SceneNode>,
    root: Option<SceneNodeId>,
    camera: SceneDataUBO,
    skybox_env_id: EnvironmentHandle,
}

impl Default for SceneWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneWorld {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(256),
            root: None,
            camera: SceneDataUBO::default(),
            skybox_env_id: EnvironmentHandle::new(0, 0),
        }
    }

    pub fn root_id(&self) -> SceneNodeId {
        self.root.unwrap_or(0)
    }

    pub fn set_root(&mut self, id: SceneNodeId) {
        self.root = Some(id);
    }

    pub fn set_skybox_env_id(&mut self, env_id: EnvironmentHandle) {
        self.skybox_env_id = env_id;
    }

    pub fn add_node(&mut self, parent: Option<SceneNodeId>, mut node: SceneNode) -> SceneNodeId {
        let id = self.nodes.len() as SceneNodeId;
        node.parent = parent;
        self.nodes.push(node);

        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(parent_id as usize) {
                parent_node.children.push(id);
            }
        } else if self.root.is_none() {
            self.root = Some(id);
        }

        id
    }

    pub fn add_node_with_parts(
        &mut self,
        parent: Option<SceneNodeId>,
        local_transform: Mat4,
        meshes: Vec<MeshHandle>,
    ) -> SceneNodeId {
        let node = SceneNode {
            parent,
            children: Vec::new(),
            meshes,
            local_transform,
            world_transform: Mat4::IDENTITY,
            dirty: true,
            layer_mask: u64::MAX,
            tags: Vec::new(),
        };
        self.add_node(parent, node)
    }

    pub fn update_camera(&mut self, view: Mat4, projection: Mat4, cam_pos: Vec3) {
        self.camera.view = view;
        self.camera.projection = projection;
        self.camera.cam_pos = cam_pos;
    }

    pub fn build_submission(&mut self) -> RenderSubmission {
        let mut submission = RenderSubmission::new(self.camera, 400);
        submission.skybox_env_id = self.skybox_env_id;

        let Some(root_id) = self.root else {
            return submission;
        };

        self.refresh_world_recursive(root_id, Mat4::IDENTITY);

        self.collect_draw_items_recursive(root_id, &mut submission);

        submission
    }

    fn refresh_world_recursive(&mut self, node_id: SceneNodeId, parent_world: Mat4) {
        let (world, children) = {
            let node = &mut self.nodes[node_id as usize];
            if node.dirty {
                node.world_transform = parent_world.mul_mat4(&node.local_transform);
                node.dirty = false;
            }
            (node.world_transform, node.children.clone())
        };

        for child in children {
            self.refresh_world_recursive(child, world);
        }
    }

    fn collect_draw_items_recursive(
        &self,
        node_id: SceneNodeId,
        submission: &mut RenderSubmission,
    ) {
        let node = &self.nodes[node_id as usize];

        for mesh_id in node.meshes.iter().copied() {
            submission.push_draw_item(FrameDrawItem {
                mesh_id,
                transform: node.world_transform,
            });
        }

        for child in node.children.iter().copied() {
            self.collect_draw_items_recursive(child, submission);
        }
    }
}
