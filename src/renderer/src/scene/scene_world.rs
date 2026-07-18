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

use crate::api::scene::{DirectionalLight, DirectionalLightId, PointLight, PointLightId, SceneAssetReference};
use crate::data::camera::{Aabb, Frustum, Ray};
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::EnvironmentHandle;
use crate::data::handles::MeshHandle;
use crate::scene::render_submission::{
    FrameDirectionalLight, FrameDrawItem, FramePointLight, RenderSubmission,
    MAX_POINT_LIGHTS_GPU,
};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SceneNodeId {
    pub slot: u32,
    pub generation: u32,
}

impl SceneNodeId {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNode {
    #[serde(skip)]
    pub stable_id: Option<String>,
    #[serde(skip)]
    pub name: String,
    pub parent: Option<SceneNodeId>,
    pub children: Vec<SceneNodeId>,
    #[serde(skip)]
    pub meshes: Vec<MeshHandle>,
    #[serde(skip)]
    pub asset: Option<SceneAssetReference>,
    #[serde(skip)]
    pub material_overrides: BTreeMap<String, String>,
    pub local_transform: Mat4,
    #[serde(skip)]
    pub world_transform: Mat4,
    #[serde(skip)]
    pub dirty: bool,
    #[serde(skip)]
    pub layer_mask: u64,
    #[serde(skip)]
    pub tags: Vec<String>,
}

impl Default for SceneNode {
    fn default() -> Self {
        Self {
            stable_id: None,
            name: String::new(),
            parent: None,
            children: Vec::new(),
            meshes: Vec::new(),
            asset: None,
            material_overrides: BTreeMap::new(),
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

#[derive(Clone, Debug)]
pub(crate) struct RestorableSceneSubtree {
    node: SceneNode,
    children: Vec<RestorableSceneSubtree>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SceneNodeRefError {
    OutOfBounds,
    Vacant,
    GenerationMismatch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum DirectionalLightRefError {
    OutOfBounds,
    Vacant,
    GenerationMismatch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PointLightRefError {
    OutOfBounds,
    Vacant,
    GenerationMismatch,
}

#[derive(Clone, Debug)]
struct PointLightEntry {
    generation: u32,
    light: Option<PointLight>,
}

#[derive(Clone, Debug)]
struct DirectionalLightEntry {
    generation: u32,
    light: Option<DirectionalLight>,
}

pub struct SceneWorld {
    nodes: Vec<SceneNodeEntry>,
    free_slots: Vec<u32>,
    root: Option<SceneNodeId>,
    camera: SceneDataUBO,
    skybox_env_id: EnvironmentHandle,
    point_lights: Vec<PointLightEntry>,
    free_point_light_slots: Vec<u32>,
    directional_lights: Vec<DirectionalLightEntry>,
    free_directional_light_slots: Vec<u32>,
    /// When true, mesh-backed nodes outside the camera frustum are omitted
    /// from `build_submission`. Descendants are tested independently. Enabled
    /// by default.
    pub enable_frustum_culling: bool,
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
            point_lights: Vec::with_capacity(16),
            free_point_light_slots: Vec::new(),
            directional_lights: Vec::with_capacity(1),
            free_directional_light_slots: Vec::new(),
            enable_frustum_culling: true,
        }
    }

    pub(crate) fn root_id(&self) -> Option<SceneNodeId> {
        self.root.filter(|id| self.is_valid_node_id(*id))
    }

    pub(crate) fn set_root(&mut self, id: SceneNodeId) {
        if self.is_valid_node_id(id) {
            self.root = Some(id);
        }
    }

    pub(crate) fn skybox_env_id(&self) -> EnvironmentHandle {
        self.skybox_env_id
    }

    pub(crate) fn set_skybox_env_id(&mut self, env_id: EnvironmentHandle) {
        self.skybox_env_id = env_id;
    }

    /// Returns an iterator over active nodes suitable for serialization.
    pub(crate) fn serializable_nodes(&self) -> impl Iterator<Item = (SceneNodeId, &SceneNode)> {
        self.nodes.iter().enumerate().filter_map(|(slot, entry)| {
            entry
                .node
                .as_ref()
                .map(|node| (SceneNodeId::new(slot as u32, entry.generation), node))
        })
    }

    /// Returns all active point lights with their IDs.
    pub(crate) fn serializable_lights(&self) -> impl Iterator<Item = (PointLightId, &PointLight)> {
        self.point_lights
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry.light.as_ref().map(|light| {
                    (
                        PointLightId {
                            slot: slot as u32,
                            generation: entry.generation,
                        },
                        light,
                    )
                })
            })
    }

    pub(crate) fn validate_node_ref(&self, id: SceneNodeId) -> Result<(), SceneNodeRefError> {
        let Some(entry) = self.nodes.get(id.slot as usize) else {
            return Err(SceneNodeRefError::OutOfBounds);
        };
        if entry.generation != id.generation {
            return Err(SceneNodeRefError::GenerationMismatch);
        }
        if entry.node.is_none() {
            return Err(SceneNodeRefError::Vacant);
        }
        Ok(())
    }

    pub(crate) fn is_valid_node_id(&self, id: SceneNodeId) -> bool {
        self.validate_node_ref(id).is_ok()
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

    pub(crate) fn add_node(
        &mut self,
        parent: Option<SceneNodeId>,
        mut node: SceneNode,
    ) -> SceneNodeId {
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
            ..SceneNode::default()
        };
        self.add_node(parent, node)
    }

    pub fn remove_node(&mut self, node_id: SceneNodeId) -> bool {
        self.remove_node_recursive(node_id)
    }

    pub(crate) fn clone_subtree(&self, node_id: SceneNodeId) -> Option<RestorableSceneSubtree> {
        let node = self.get_node(node_id)?.clone();
        let children = node
            .children
            .iter()
            .copied()
            .filter_map(|child| self.clone_subtree(child))
            .collect();
        Some(RestorableSceneSubtree { node, children })
    }

    pub(crate) fn restore_subtree(&mut self, snapshot: RestorableSceneSubtree) -> SceneNodeId {
        let parent = snapshot
            .node
            .parent
            .filter(|parent_id| self.is_valid_node_id(*parent_id));
        self.restore_subtree_with_parent(snapshot, parent)
    }

    pub(crate) fn reparent_node(
        &mut self,
        node_id: SceneNodeId,
        new_parent: Option<SceneNodeId>,
    ) -> Result<(), ReparentError> {
        self.validate_node_ref(node_id)
            .map_err(ReparentError::InvalidNode)?;
        if let Some(parent_id) = new_parent {
            self.validate_node_ref(parent_id)
                .map_err(ReparentError::InvalidParent)?;
            if parent_id == node_id || self.is_descendant(parent_id, node_id) {
                return Err(ReparentError::Cycle);
            }
        }

        let old_parent = self
            .get_node(node_id)
            .ok_or(ReparentError::InvalidNode(SceneNodeRefError::Vacant))?
            .parent;
        if old_parent == new_parent {
            return Ok(());
        }

        if let Some(parent_id) = old_parent {
            if let Some(parent_node) = self.get_node_mut(parent_id) {
                parent_node.children.retain(|child| *child != node_id);
            }
        }

        if let Some(parent_id) = new_parent {
            if let Some(parent_node) = self.get_node_mut(parent_id) {
                if !parent_node.children.contains(&node_id) {
                    parent_node.children.push(node_id);
                }
            }
        }

        if let Some(node) = self.get_node_mut(node_id) {
            node.parent = new_parent;
            node.dirty = true;
        }

        if self.root == Some(node_id) && new_parent.is_some() {
            self.root = self.find_first_parentless_node();
        } else if self.root.is_none() && new_parent.is_none() {
            self.root = Some(node_id);
        }

        Ok(())
    }

    pub(crate) fn update_camera(&mut self, view: Mat4, projection: Mat4, cam_pos: Vec3) {
        self.camera.view = view;
        self.camera.projection = projection;
        self.camera.cam_pos = cam_pos;
    }

    pub(crate) fn camera_data(&self) -> SceneDataUBO {
        self.camera
    }

    /// Ray-pick the scene: iterate all active nodes, compute AABB from
    /// local transform bounds, and return the closest intersection.
    pub(crate) fn pick_ray(&self, ray: &Ray) -> Option<SceneNodeId> {
        let mut closest: Option<(f32, SceneNodeId)> = None;

        for (slot, entry) in self.nodes.iter().enumerate() {
            let Some(ref node) = entry.node else {
                continue;
            };

            let aabb = node_pick_bounds(node);

            if let Some(t) = aabb.intersect_ray(ray) {
                if closest.map_or(true, |(best_t, _)| t < best_t) {
                    closest = Some((t, SceneNodeId::new(slot as u32, entry.generation)));
                }
            }
        }

        closest.map(|(_, id)| id)
    }

    pub(crate) fn build_submission(&mut self) -> RenderSubmission {
        let mut submission = RenderSubmission::new(self.camera, 400);
        submission.skybox_env_id = self.skybox_env_id;

        // Collect the scene's single directional light.
        submission.directional_light =
            self.get_active_directional_light()
                .map(|light| FrameDirectionalLight {
                    direction: light.direction,
                    color: light.color,
                    intensity: light.intensity,
                });

        // Collect first N active lights (not first N slots) so sparse slot churn
        // does not accidentally submit zero lights.
        for entry in self.point_lights.iter() {
            if submission.point_lights.len() >= MAX_POINT_LIGHTS_GPU {
                break;
            }
            if let Some(light) = entry.light {
                submission.point_lights.push(FramePointLight {
                    position: light.position,
                    color: light.color,
                    intensity: light.intensity,
                    range: light.range,
                });
            }
        }

        let Some(root_id) = self.root else {
            return submission;
        };
        if !self.is_valid_node_id(root_id) {
            self.root = None;
            return submission;
        }

        // Parent world transform must be resolved before recursing into children.
        // Children multiply against this exact value, so order is critical.
        self.refresh_world_recursive(root_id, Mat4::IDENTITY, false);

        // Frustum culling is enabled by default and can be disabled through
        // the Scene facade for diagnostics or compatibility.
        let frustum = if self.enable_frustum_culling {
            Some(Frustum::from_view_projection(
                &(self.camera.projection * self.camera.view),
            ))
        } else {
            None
        };

        self.collect_draw_items_recursive_culled(root_id, &mut submission, frustum.as_ref());

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

    fn restore_subtree_with_parent(
        &mut self,
        snapshot: RestorableSceneSubtree,
        parent: Option<SceneNodeId>,
    ) -> SceneNodeId {
        let RestorableSceneSubtree { mut node, children } = snapshot;
        node.parent = parent;
        node.children.clear();
        node.dirty = true;
        let restored = self.add_node(parent, node);

        for child in children {
            self.restore_subtree_with_parent(child, Some(restored));
        }

        restored
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

    fn is_descendant(&self, possible_descendant: SceneNodeId, ancestor: SceneNodeId) -> bool {
        let Some(ancestor_node) = self.get_node(ancestor) else {
            return false;
        };

        for child in ancestor_node.children.iter().copied() {
            if child == possible_descendant || self.is_descendant(possible_descendant, child) {
                return true;
            }
        }

        false
    }

    fn find_first_parentless_node(&self) -> Option<SceneNodeId> {
        self.serializable_nodes()
            .find_map(|(id, node)| node.parent.is_none().then_some(id))
    }

    fn refresh_world_recursive(
        &mut self,
        node_id: SceneNodeId,
        parent_world: Mat4,
        parent_dirty: bool,
    ) {
        let Some((world, children, propagate_dirty)) = ({
            let Some(node) = self.get_node_mut(node_id) else {
                return;
            };
            let effective_dirty = node.dirty || parent_dirty;
            if effective_dirty {
                node.world_transform = parent_world.mul_mat4(&node.local_transform);
                node.dirty = false;
            }
            Some((node.world_transform, node.children.clone(), effective_dirty))
        }) else {
            return;
        };

        for child in children {
            if self.is_valid_node_id(child) {
                self.refresh_world_recursive(child, world, propagate_dirty);
            }
        }
    }

    fn collect_draw_items_recursive_culled(
        &self,
        node_id: SceneNodeId,
        submission: &mut RenderSubmission,
        frustum: Option<&Frustum>,
    ) {
        let Some(node) = self.get_node(node_id) else {
            return;
        };

        let meshes_visible = node.meshes.is_empty()
            || frustum.is_none_or(|frustum| frustum.intersects_aabb(&node_pick_bounds(node)));
        if meshes_visible {
            for mesh_id in node.meshes.iter().copied() {
                submission.push_draw_item(FrameDrawItem {
                    mesh_id,
                    transform: node.world_transform,
                });
            }
        }

        // Proxy bounds describe only this node, not its subtree. Always test
        // descendants independently so an off-screen grouping/parent node
        // cannot hide an in-frustum child.
        for child in node.children.iter().copied() {
            if self.is_valid_node_id(child) {
                self.collect_draw_items_recursive_culled(child, submission, frustum);
            }
        }
    }

    // Point light handle validation and lifecycle

    pub(crate) fn validate_point_light_ref(
        &self,
        id: PointLightId,
    ) -> Result<(), PointLightRefError> {
        let Some(entry) = self.point_lights.get(id.slot as usize) else {
            return Err(PointLightRefError::OutOfBounds);
        };
        if entry.generation != id.generation {
            return Err(PointLightRefError::GenerationMismatch);
        };
        if entry.light.is_none() {
            return Err(PointLightRefError::Vacant);
        }
        Ok(())
    }

    pub(crate) fn add_point_light(&mut self, light: PointLight) -> PointLightId {
        if let Some(slot) = self.free_point_light_slots.pop() {
            let entry = &mut self.point_lights[slot as usize];
            debug_assert!(
                entry.light.is_none(),
                "free slot list contained a live point light"
            );
            entry.light = Some(light);
            return PointLightId {
                slot,
                generation: entry.generation,
            };
        }

        let slot = self.point_lights.len() as u32;
        self.point_lights.push(PointLightEntry {
            generation: 0,
            light: Some(light),
        });
        PointLightId {
            slot,
            generation: 0,
        }
    }

    pub(crate) fn update_point_light(&mut self, id: PointLightId, light: PointLight) -> bool {
        let Some(entry) = self.point_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = Some(light);
        true
    }

    pub(crate) fn remove_point_light(&mut self, id: PointLightId) -> bool {
        let Some(entry) = self.point_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = None;
        entry.generation = entry.generation.wrapping_add(1);
        self.free_point_light_slots.push(id.slot);
        true
    }

    // future lighting query API
    #[allow(dead_code)]
    pub(crate) fn get_active_point_lights(&self) -> Vec<PointLight> {
        self.point_lights
            .iter()
            .filter_map(|entry| entry.light)
            .collect()
    }

    // Directional light handle validation and lifecycle

    pub(crate) fn validate_directional_light_ref(
        &self,
        id: DirectionalLightId,
    ) -> Result<(), DirectionalLightRefError> {
        let Some(entry) = self.directional_lights.get(id.slot as usize) else {
            return Err(DirectionalLightRefError::OutOfBounds);
        };
        if entry.generation != id.generation {
            return Err(DirectionalLightRefError::GenerationMismatch);
        };
        if entry.light.is_none() {
            return Err(DirectionalLightRefError::Vacant);
        }
        Ok(())
    }

    pub(crate) fn add_directional_light(
        &mut self,
        light: DirectionalLight,
    ) -> DirectionalLightId {
        if let Some(slot) = self.free_directional_light_slots.pop() {
            let entry = &mut self.directional_lights[slot as usize];
            debug_assert!(
                entry.light.is_none(),
                "free slot list contained a live directional light"
            );
            entry.light = Some(light);
            return DirectionalLightId {
                slot,
                generation: entry.generation,
            };
        }

        let slot = self.directional_lights.len() as u32;
        self.directional_lights.push(DirectionalLightEntry {
            generation: 0,
            light: Some(light),
        });
        DirectionalLightId {
            slot,
            generation: 0,
        }
    }

    pub(crate) fn update_directional_light(
        &mut self,
        id: DirectionalLightId,
        light: DirectionalLight,
    ) -> bool {
        let Some(entry) = self.directional_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = Some(light);
        true
    }

    pub(crate) fn remove_directional_light(&mut self, id: DirectionalLightId) -> bool {
        let Some(entry) = self.directional_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = None;
        entry.generation = entry.generation.wrapping_add(1);
        self.free_directional_light_slots.push(id.slot);
        true
    }

    /// Returns the active directional light (the public facade enforces one).
    pub(crate) fn get_active_directional_light(&self) -> Option<DirectionalLight> {
        self.directional_lights
            .iter()
            .find_map(|entry| entry.light)
    }
}

fn node_pick_bounds(node: &SceneNode) -> Aabb {
    // Current renderer draw submissions do not expose mesh CPU bounds here.
    // Use a transform-aware picking/culling proxy: mesh-backed nodes get one
    // unit of volume per axis, while empty grouping nodes receive a smaller
    // but still selectable proxy around their origin.
    let half_extent = if node.meshes.is_empty() { 0.25 } else { 0.5 };
    transformed_aabb(
        Mat4::from_scale(Vec3::splat(half_extent * 2.0)),
        node.world_transform,
    )
}

fn transformed_aabb(local_bounds: Mat4, world_transform: Mat4) -> Aabb {
    let local_min = local_bounds.transform_point3(Vec3::splat(-0.5));
    let local_max = local_bounds.transform_point3(Vec3::splat(0.5));
    let corners = [
        Vec3::new(local_min.x, local_min.y, local_min.z),
        Vec3::new(local_min.x, local_min.y, local_max.z),
        Vec3::new(local_min.x, local_max.y, local_min.z),
        Vec3::new(local_min.x, local_max.y, local_max.z),
        Vec3::new(local_max.x, local_min.y, local_min.z),
        Vec3::new(local_max.x, local_min.y, local_max.z),
        Vec3::new(local_max.x, local_max.y, local_min.z),
        Vec3::new(local_max.x, local_max.y, local_max.z),
    ];

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let world = world_transform.transform_point3(corner);
        min = min.min(world);
        max = max.max(world);
    }
    Aabb::from_min_max(min, max)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReparentError {
    InvalidNode(SceneNodeRefError),
    InvalidParent(SceneNodeRefError),
    Cycle,
}

#[cfg(test)]
mod tests {
    use super::{PointLightRefError, SceneNode, SceneNodeId, SceneWorld};
    use crate::data::camera::Ray;
    use crate::data::handles::MeshHandle;
    use glam::{Mat4, Vec3};

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

    #[test]
    fn submission_skips_top_level_orphans() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            SceneNode {
                meshes: vec![MeshHandle::new(1, 0)],
                ..SceneNode::default()
            },
        );
        scene.set_root(root);

        scene.add_node(
            None,
            SceneNode {
                meshes: vec![MeshHandle::new(2, 0)],
                ..SceneNode::default()
            },
        );

        let submission = scene.build_submission();
        assert_eq!(submission.draw_items.len(), 1);
        assert_eq!(submission.draw_items[0].mesh_id, MeshHandle::new(1, 0));
    }

    #[test]
    fn parent_transform_update_recomputes_child_world_transform() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            SceneNode {
                local_transform: Mat4::IDENTITY,
                ..SceneNode::default()
            },
        );
        let child = scene.add_node(
            Some(root),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                meshes: vec![MeshHandle::new(1, 0)],
                ..SceneNode::default()
            },
        );
        scene.set_root(root);
        scene.enable_frustum_culling = false;

        let initial_submission = scene.build_submission();
        assert_eq!(initial_submission.draw_items.len(), 1);
        assert_eq!(
            initial_submission.draw_items[0].transform,
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
        );

        {
            let root_node = scene.get_node_mut(root).expect("root should exist");
            root_node.local_transform = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
            root_node.dirty = true;
        }
        assert!(!scene.get_node(child).expect("child should exist").dirty);

        let moved_submission = scene.build_submission();
        assert_eq!(moved_submission.draw_items.len(), 1);
        assert_eq!(
            moved_submission.draw_items[0].transform,
            Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0))
        );
    }

    #[test]
    fn culling_tests_descendants_independently_of_parent_bounds() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        let offscreen_parent = scene.add_node(
            Some(root),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
                meshes: vec![MeshHandle::new(1, 0)],
                ..SceneNode::default()
            },
        );
        scene.add_node(
            Some(offscreen_parent),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(-100.0, 0.0, -5.0)),
                meshes: vec![MeshHandle::new(2, 0)],
                ..SceneNode::default()
            },
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );

        let submission = scene.build_submission();

        assert_eq!(submission.draw_items.len(), 1);
        assert_eq!(submission.draw_items[0].mesh_id, MeshHandle::new(2, 0));
    }

    #[test]
    fn parent_dirty_propagates_through_grandchild_chain() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            SceneNode {
                local_transform: Mat4::IDENTITY,
                ..SceneNode::default()
            },
        );
        let child = scene.add_node(
            Some(root),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                ..SceneNode::default()
            },
        );
        scene.add_node(
            Some(child),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
                meshes: vec![MeshHandle::new(2, 0)],
                ..SceneNode::default()
            },
        );
        scene.set_root(root);
        scene.enable_frustum_culling = false;

        let initial_submission = scene.build_submission();
        assert_eq!(initial_submission.draw_items.len(), 1);
        assert_eq!(
            initial_submission.draw_items[0].transform,
            Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0))
        );

        let root_node = scene.get_node_mut(root).expect("root should exist");
        root_node.local_transform = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        root_node.dirty = true;

        let moved_submission = scene.build_submission();
        assert_eq!(moved_submission.draw_items.len(), 1);
        assert_eq!(
            moved_submission.draw_items[0].transform,
            Mat4::from_translation(Vec3::new(8.0, 0.0, 0.0))
        );
    }

    #[test]
    fn pick_ray_uses_transformed_scaled_proxy_bounds() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            SceneNode {
                local_transform: Mat4::from_scale_rotation_translation(
                    Vec3::new(4.0, 1.0, 1.0),
                    glam::Quat::IDENTITY,
                    Vec3::new(3.0, 0.0, 0.0),
                ),
                meshes: vec![MeshHandle::new(1, 0)],
                ..SceneNode::default()
            },
        );
        scene.set_root(root);
        scene.enable_frustum_culling = false;
        scene.build_submission();

        let ray = Ray {
            origin: Vec3::new(3.0, 0.0, 5.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };

        assert_eq!(scene.pick_ray(&ray), Some(root));
    }

    #[test]
    fn point_light_handle_lifecycle() {
        use crate::api::scene::PointLight;

        let mut scene = SceneWorld::new();

        // Create a point light
        let light = PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::new(1.0, 0.8, 0.6),
            intensity: 50.0,
            range: 10.0,
        };
        let id = scene.add_point_light(light);
        assert_eq!(id.slot, 0);
        assert_eq!(id.generation, 0);

        // Validate it exists
        assert!(scene.validate_point_light_ref(id).is_ok());

        // Update the light
        let updated_light = PointLight {
            position: Vec3::new(5.0, 6.0, 7.0),
            color: Vec3::new(0.5, 0.5, 1.0),
            intensity: 100.0,
            range: 15.0,
        };
        assert!(scene.update_point_light(id, updated_light));

        // Remove the light
        assert!(scene.remove_point_light(id));

        // Validate it's now stale
        assert!(matches!(
            scene.validate_point_light_ref(id),
            Err(PointLightRefError::GenerationMismatch)
        ));

        // Create a new light (should reuse slot 0 but with generation 1)
        let new_id = scene.add_point_light(light);
        assert_eq!(new_id.slot, 0);
        assert_eq!(new_id.generation, 1);

        // Old handle should still be rejected
        assert!(matches!(
            scene.validate_point_light_ref(id),
            Err(PointLightRefError::GenerationMismatch)
        ));

        // New handle should be valid
        assert!(scene.validate_point_light_ref(new_id).is_ok());
    }

    #[test]
    fn submission_clamps_point_light_count() {
        use crate::api::scene::PointLight;
        use crate::scene::render_submission::MAX_POINT_LIGHTS_GPU;

        let mut scene = SceneWorld::new();

        // Create more than MAX_POINT_LIGHTS_GPU lights
        let light = PointLight {
            position: Vec3::new(0.0, 5.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 30.0,
            range: 8.0,
        };

        for i in 0..20 {
            let mut light_instance = light;
            light_instance.position.x = i as f32;
            scene.add_point_light(light_instance);
        }

        // Build submission and verify clamping
        let submission = scene.build_submission();
        assert_eq!(submission.point_lights.len(), MAX_POINT_LIGHTS_GPU);

        // Verify lights are correct (should be first MAX_POINT_LIGHTS_GPU)
        for (i, frame_light) in submission.point_lights.iter().enumerate() {
            assert_eq!(frame_light.position.x, i as f32);
        }
    }

    #[test]
    fn zero_light_submission_has_empty_list() {
        let mut scene = SceneWorld::new();

        // Build submission with no lights
        let submission = scene.build_submission();
        assert_eq!(submission.point_lights.len(), 0);
    }

    #[test]
    fn submission_collects_active_lights_from_sparse_slots() {
        use crate::api::scene::PointLight;

        let mut scene = SceneWorld::new();

        let base_light = PointLight {
            position: Vec3::new(0.0, 5.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 30.0,
            range: 8.0,
        };

        let mut ids = Vec::new();
        for i in 0..20 {
            let mut light_instance = base_light;
            light_instance.position.x = i as f32;
            ids.push(scene.add_point_light(light_instance));
        }

        for id in ids.iter().take(16) {
            assert!(scene.remove_point_light(*id));
        }

        let submission = scene.build_submission();
        assert_eq!(submission.point_lights.len(), 4);

        for (expected_x, frame_light) in (16..20).zip(submission.point_lights.iter()) {
            assert_eq!(frame_light.position.x, expected_x as f32);
        }
    }

    #[test]
    fn stale_point_light_update_rejected() {
        use crate::api::scene::PointLight;

        let mut scene = SceneWorld::new();

        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        };

        let id = scene.add_point_light(light);
        scene.remove_point_light(id);

        // Try to update after removal should fail
        assert!(!scene.update_point_light(id, light));
    }
}
