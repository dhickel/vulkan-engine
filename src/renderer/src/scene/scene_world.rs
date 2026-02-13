//! # Scene World Graph and Submission Bridge
//!
//! ## Purpose
//! Owns the runtime scene node hierarchy and converts it into a flat `RenderSubmission`
//! every frame. The scene stays renderer-agnostic and only emits stable mesh handles and
//! world transforms.
//!
//! ## Handle Safety Model
//! Scene nodes use a slot + generation handle (`SceneNodeId`) rather than raw indices.
//! This allows deletion without invalidating all outstanding references to the old slot.
//! Stale handles fail validation and are ignored during traversal.

use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::EnvironmentHandle;
use crate::data::handles::MeshHandle;
use crate::scene::render_submission::{FrameDrawItem, RenderSubmission};
use glam::{Mat4, Vec3};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SceneNodeId {
    pub slot: u32,
    pub generation: u32,
}

impl SceneNodeId {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

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

#[derive(Clone, Debug)]
struct SceneNodeEntry {
    generation: u32,
    node: Option<SceneNode>,
}

pub struct SceneWorld {
    nodes: Vec<SceneNodeEntry>,
    free_slots: Vec<u32>,
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
            free_slots: Vec::new(),
            root: None,
            camera: SceneDataUBO::default(),
            skybox_env_id: EnvironmentHandle::new(0, 0),
        }
    }

    pub fn root_id(&self) -> Option<SceneNodeId> {
        self.root.filter(|id| self.is_valid_node_id(*id))
    }

    pub fn set_root(&mut self, id: SceneNodeId) {
        if self.is_valid_node_id(id) {
            self.root = Some(id);
        }
    }

    pub fn set_skybox_env_id(&mut self, env_id: EnvironmentHandle) {
        self.skybox_env_id = env_id;
    }

    pub fn is_valid_node_id(&self, id: SceneNodeId) -> bool {
        let Some(entry) = self.nodes.get(id.slot as usize) else {
            return false;
        };
        entry.generation == id.generation && entry.node.is_some()
    }

    pub fn get_node(&self, id: SceneNodeId) -> Option<&SceneNode> {
        let entry = self.nodes.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_ref()
    }

    pub fn get_node_mut(&mut self, id: SceneNodeId) -> Option<&mut SceneNode> {
        let entry = self.nodes.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_mut()
    }

    pub fn add_node(&mut self, parent: Option<SceneNodeId>, mut node: SceneNode) -> SceneNodeId {
        let resolved_parent = parent.filter(|id| self.is_valid_node_id(*id));
        node.parent = resolved_parent;
        let id = self.allocate_node_slot(node);

        if let Some(parent_id) = resolved_parent {
            if let Some(parent_node) = self.get_node_mut(parent_id) {
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

    pub fn remove_node(&mut self, node_id: SceneNodeId) -> bool {
        self.remove_node_recursive(node_id)
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
        if !self.is_valid_node_id(root_id) {
            self.root = None;
            return submission;
        }

        // Parent world transform must be resolved before recursing into children.
        // Children multiply against this exact value, so order is critical.
        self.refresh_world_recursive(root_id, Mat4::IDENTITY);
        self.collect_draw_items_recursive(root_id, &mut submission);

        submission
    }

    fn allocate_node_slot(&mut self, node: SceneNode) -> SceneNodeId {
        if let Some(slot) = self.free_slots.pop() {
            let entry = &mut self.nodes[slot as usize];
            debug_assert!(entry.node.is_none(), "free slot list contained a live node");
            entry.node = Some(node);
            return SceneNodeId::new(slot, entry.generation);
        }

        let slot = self.nodes.len() as u32;
        self.nodes.push(SceneNodeEntry {
            generation: 0,
            node: Some(node),
        });
        SceneNodeId::new(slot, 0)
    }

    fn remove_node_recursive(&mut self, node_id: SceneNodeId) -> bool {
        let Some((parent, children)) = self
            .get_node(node_id)
            .map(|node| (node.parent, node.children.clone()))
        else {
            return false;
        };

        // Remove descendants first so parent links can be detached while this node is still valid.
        for child in children {
            let _ = self.remove_node_recursive(child);
        }

        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.get_node_mut(parent_id) {
                parent_node.children.retain(|child_id| *child_id != node_id);
            }
        }

        if self.root == Some(node_id) {
            self.root = None;
        }

        let Some(entry) = self.nodes.get_mut(node_id.slot as usize) else {
            return false;
        };
        if entry.generation != node_id.generation || entry.node.is_none() {
            return false;
        }

        entry.node = None;
        entry.generation = entry.generation.wrapping_add(1);
        self.free_slots.push(node_id.slot);
        true
    }

    fn refresh_world_recursive(&mut self, node_id: SceneNodeId, parent_world: Mat4) {
        let Some((world, children)) = ({
            let Some(node) = self.get_node_mut(node_id) else {
                return;
            };
            if node.dirty {
                node.world_transform = parent_world.mul_mat4(&node.local_transform);
                node.dirty = false;
            }
            Some((node.world_transform, node.children.clone()))
        }) else {
            return;
        };

        for child in children {
            if self.is_valid_node_id(child) {
                self.refresh_world_recursive(child, world);
            }
        }
    }

    fn collect_draw_items_recursive(
        &self,
        node_id: SceneNodeId,
        submission: &mut RenderSubmission,
    ) {
        let Some(node) = self.get_node(node_id) else {
            return;
        };

        for mesh_id in node.meshes.iter().copied() {
            submission.push_draw_item(FrameDrawItem {
                mesh_id,
                transform: node.world_transform,
            });
        }

        for child in node.children.iter().copied() {
            if self.is_valid_node_id(child) {
                self.collect_draw_items_recursive(child, submission);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SceneNode, SceneNodeId, SceneWorld};

    #[test]
    fn remove_marks_old_handle_stale_after_slot_reuse() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());

        assert!(scene.is_valid_node_id(root));
        assert!(scene.remove_node(root));
        assert!(!scene.is_valid_node_id(root));

        let replacement = scene.add_node(None, SceneNode::default());
        assert_eq!(replacement.slot, root.slot);
        assert_ne!(replacement.generation, root.generation);
        assert!(scene.is_valid_node_id(replacement));
    }

    #[test]
    fn remove_node_recursively_removes_children() {
        let mut scene = SceneWorld::new();
        let parent = scene.add_node(None, SceneNode::default());
        let child = scene.add_node(Some(parent), SceneNode::default());
        scene.set_root(parent);

        assert!(scene.remove_node(parent));
        assert!(!scene.is_valid_node_id(parent));
        assert!(!scene.is_valid_node_id(child));
        assert!(scene.root_id().is_none());
    }

    #[test]
    fn traversal_ignores_stale_child_references() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        scene.set_root(root);

        let stale_child = SceneNodeId::new(42, 7);
        scene.get_node_mut(root).unwrap().children.push(stale_child);

        // Should not panic even when a stale child handle is present.
        let submission = scene.build_submission();
        assert!(submission.draw_items.is_empty());
    }
}
