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

use crate::api::scene::{
    BoundsUnknownReason, DirectionalLight, DirectionalLightId, MeshBoundsEntry, PointLight,
    PointLightId, SceneAssetReference, SceneBounds, SpotLight, SpotLightId,
};
use crate::data::camera::{Aabb, Frustum, Ray};
use crate::data::gpu_data::SceneDataUBO;
use crate::data::handles::EnvironmentHandle;
use crate::data::handles::MeshHandle;
use crate::scene::render_submission::{
    FrameDirectionalLight, FrameDrawItem, FramePointLight, FrameSpotLight, RenderSubmission,
    MAX_DIRECTIONAL_LIGHTS_GPU, MAX_POINT_LIGHTS_GPU, MAX_SPOT_LIGHTS_GPU,
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
    /// Per-mesh bounds aligned with `meshes`; must never desynchronize.
    #[serde(skip)]
    pub mesh_bounds: Vec<MeshBoundsEntry>,
    /// Explicit local proxy bounds for geometry that is intentionally unavailable.
    #[serde(skip)]
    pub local_proxy_bounds: Option<Aabb>,
    /// Cached node-world bounds from local mesh aggregation.
    #[serde(skip)]
    pub node_world_bounds: Option<SceneBounds>,
    /// Cached subtree-world bounds from post-order aggregation.
    #[serde(skip)]
    pub subtree_world_bounds: Option<SceneBounds>,
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
            mesh_bounds: Vec::new(),
            local_proxy_bounds: None,
            node_world_bounds: None,
            subtree_world_bounds: None,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SpotLightRefError {
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

#[derive(Clone, Debug)]
struct SpotLightEntry {
    generation: u32,
    light: Option<SpotLight>,
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
    spot_lights: Vec<SpotLightEntry>,
    free_spot_light_slots: Vec<u32>,
    /// ID of the directional light that casts shadows (at most one).
    shadow_casting_directional: Option<DirectionalLightId>,
    /// When true, mesh-backed nodes outside the camera frustum are omitted
    /// from `build_submission`. Descendants are tested independently. Enabled
    /// by default.
    pub enable_frustum_culling: bool,
    /// BSP mount state for PVS-aware culling, light selection, and depth-sorting.
    #[cfg(feature = "bsp")]
    pub(crate) bsp_mount: crate::scene::bsp_visibility::BspMountState,
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
            directional_lights: Vec::with_capacity(4),
            free_directional_light_slots: Vec::new(),
            spot_lights: Vec::with_capacity(16),
            free_spot_light_slots: Vec::new(),
            shadow_casting_directional: None,
            enable_frustum_culling: true,
            #[cfg(feature = "bsp")]
            bsp_mount: crate::scene::bsp_visibility::BspMountState::new(),
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

    /// Returns all active directional lights with their IDs.
    pub(crate) fn serializable_directional_lights(
        &self,
    ) -> impl Iterator<Item = (DirectionalLightId, &DirectionalLight)> {
        self.directional_lights
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry.light.as_ref().map(|light| {
                    (
                        DirectionalLightId {
                            slot: slot as u32,
                            generation: entry.generation,
                        },
                        light,
                    )
                })
            })
    }

    /// Returns all active spot lights with their IDs.
    pub(crate) fn serializable_spot_lights(
        &self,
    ) -> impl Iterator<Item = (SpotLightId, &SpotLight)> {
        self.spot_lights
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry.light.as_ref().map(|light| {
                    (
                        SpotLightId {
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

    /// Invalidate all derived state (world bounds, subtree bounds) for a
    /// node and mark it dirty. Call this after any authoritative change
    /// (local transform, parent, meshes, mesh_bounds, proxy bounds).
    pub(crate) fn invalidate_derived_state(&mut self, id: SceneNodeId) {
        if let Some(node) = self.get_node_mut(id) {
            node.dirty = true;
            node.node_world_bounds = None;
            node.subtree_world_bounds = None;
        }
    }

    pub(crate) fn refresh_derived_state(&mut self) {
        let Some(root_id) = self.root else {
            return;
        };
        if !self.is_valid_node_id(root_id) {
            self.root = None;
            return;
        }
        self.refresh_world_recursive(root_id, Mat4::IDENTITY, false);
        self.compute_subtree_bounds_post_order(root_id);
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
            mesh_bounds: Vec::new(),
            local_transform,
            world_transform: Mat4::IDENTITY,
            dirty: true,
            layer_mask: u64::MAX,
            tags: Vec::new(),
            ..SceneNode::default()
        };
        self.add_node(parent, node)
    }

    pub fn add_node_with_parts_and_bounds(
        &mut self,
        parent: Option<SceneNodeId>,
        local_transform: Mat4,
        meshes: Vec<MeshHandle>,
        mesh_bounds: Vec<MeshBoundsEntry>,
    ) -> SceneNodeId {
        let node = SceneNode {
            parent,
            children: Vec::new(),
            meshes,
            mesh_bounds,
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

        self.invalidate_derived_state(node_id);
        if let Some(node) = self.get_node_mut(node_id) {
            node.parent = new_parent;
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
    /// world bounds when known, explicit proxy when set, or a small
    /// editor-origin helper for empty non-mesh group nodes.
    ///
    /// **Mutable path:** requires `refresh_derived_state` to have been
    /// called so that `node_world_bounds` is fresh.  The rendering path
    /// (`build_submission`) uses this.
    pub(crate) fn pick_ray(&self, ray: &Ray) -> Option<SceneNodeId> {
        let mut closest: Option<(f32, SceneNodeId)> = None;

        for (slot, entry) in self.nodes.iter().enumerate() {
            let Some(ref node) = entry.node else {
                continue;
            };

            let aabb = match node_pick_bounds_exact(node) {
                Some(aabb) => aabb,
                None => continue, // conservative-visible without explicit proxy — skip exact hit
            };

            if let Some(t) = aabb.intersect_ray(ray) {
                if closest.map_or(true, |(best_t, _)| t < best_t) {
                    closest = Some((t, SceneNodeId::new(slot as u32, entry.generation)));
                }
            }
        }

        closest.map(|(_, id)| id)
    }

    /// Pure read-only ray-pick that computes world transforms and bounds
    /// on the fly into scratch maps without mutating `dirty`, cached
    /// transforms, or cached bounds.
    ///
    /// Walks the tree from the root, honoring parent transforms,
    /// negative scale, known/proxy/conservative-visible rules, stale
    /// IDs, and empty-group editor proxies.
    pub(crate) fn pick_ray_readonly(&self, ray: &Ray) -> Option<SceneNodeId> {
        let root_id = match self.root {
            Some(id) if self.is_valid_node_id(id) => id,
            _ => return None,
        };

        // Scratch world transforms built during traversal.
        let mut world_transforms: std::collections::HashMap<SceneNodeId, Mat4> =
            std::collections::HashMap::with_capacity(self.nodes.len());

        let mut closest: Option<(f32, SceneNodeId)> = None;
        self.pick_readonly_walk(
            root_id,
            Mat4::IDENTITY,
            ray,
            &mut world_transforms,
            &mut closest,
        );

        closest.map(|(_, id)| id)
    }

    /// Recursive read-only walk for picking.
    fn pick_readonly_walk(
        &self,
        node_id: SceneNodeId,
        parent_world: Mat4,
        ray: &Ray,
        world_transforms: &mut std::collections::HashMap<SceneNodeId, Mat4>,
        closest: &mut Option<(f32, SceneNodeId)>,
    ) {
        let node = match self.get_node(node_id) {
            Some(n) => n,
            None => return,
        };

        let world = parent_world * node.local_transform;
        world_transforms.insert(node_id, world);

        // Compute world-space bounds from local mesh/proxy bounds using the
        // on-the-fly world transform (mirrors node_pick_bounds_exact +
        // compute_node_world_bounds but reads from scratch instead of
        // cached fields).
        if let Some(aabb) = pick_bounds_readonly(node, &world) {
            if let Some(t) = aabb.intersect_ray(ray) {
                if closest.map_or(true, |(best_t, _)| t < best_t) {
                    *closest = Some((t, node_id));
                }
            }
        }

        // Recurse children.
        for child_id in node.children.iter().copied() {
            if self.is_valid_node_id(child_id) {
                self.pick_readonly_walk(child_id, world, ray, world_transforms, closest);
            }
        }
    }

    pub(crate) fn build_submission(&mut self) -> RenderSubmission {
        let mut submission = RenderSubmission::new(self.camera, 400);
        submission.skybox_env_id = self.skybox_env_id;

        // Collect the bounded directional-light set while retaining the first
        // entry in the legacy compatibility field.
        let shadow_caster_id = self.shadow_casting_directional;
        for (slot, entry) in self.directional_lights.iter().enumerate() {
            if submission.directional_lights.len() >= MAX_DIRECTIONAL_LIGHTS_GPU {
                break;
            }
            let Some(light) = entry.light else { continue };
            let light_id = DirectionalLightId {
                slot: slot as u32,
                generation: entry.generation,
            };
            submission.directional_lights.push(FrameDirectionalLight {
                direction: light.direction,
                color: light.color,
                intensity: light.intensity,
                enable_shadows: Some(light_id) == shadow_caster_id,
            });
        }
        submission.directional_light = submission.directional_lights.first().copied();

        let frustum = if self.enable_frustum_culling {
            Some(Frustum::from_view_projection(
                &(self.camera.projection * self.camera.view),
            ))
        } else {
            None
        };

        #[cfg(feature = "bsp")]
        if self.bsp_mount.active {
            submission.bsp_frame_values = crate::scene::render_submission::BspFrameValuesState {
                style_intensities: self.bsp_mount.frame_style_intensities,
                liquid_time: self.bsp_mount.frame_liquid_time,
                arena_id: self.bsp_mount.arena_id,
            };
            self.update_bsp_pvs();
            let bsp_lights = self
                .bsp_mount
                .select_frame_lights_for_camera(self.camera.cam_pos, MAX_POINT_LIGHTS_GPU);
            submission.bsp_selected_lights = bsp_lights.clone();
            submission.point_lights.extend(bsp_lights);
        }

        // Collect first N active app lights (not first N slots) after BSP-imported
        // lights so the shared GPU cap is deterministic: PVS-selected BSP first,
        // then app-added dynamics in insertion/slot order.
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

        // Collect spot lights.
        for entry in self.spot_lights.iter() {
            if submission.spot_lights.len() >= MAX_SPOT_LIGHTS_GPU {
                break;
            }
            if let Some(light) = entry.light {
                let dir = light.direction.normalize();
                let inner_half = light.inner_cone_angle * 0.5;
                let outer_half = light.outer_cone_angle * 0.5;
                submission.spot_lights.push(FrameSpotLight {
                    position: light.position,
                    direction: dir,
                    color: light.color,
                    intensity: light.intensity,
                    range: light.range,
                    inner_cos: inner_half.cos(),
                    outer_cos: outer_half.cos(),
                });
            }
        }

        #[cfg(feature = "bsp")]
        if self.bsp_mount.active {
            self.collect_bsp_draw_items(&mut submission, frustum.as_ref());
        }

        let Some(root_id) = self.root else {
            return submission;
        };
        if !self.is_valid_node_id(root_id) {
            self.root = None;
            return submission;
        }

        // Parent world transform and bounds must be resolved before culling.
        self.refresh_derived_state();

        self.collect_draw_items_culled(root_id, &mut submission, frustum.as_ref());

        submission
    }

    #[cfg(feature = "bsp")]
    fn collect_bsp_draw_items(&self, submission: &mut RenderSubmission, frustum: Option<&Frustum>) {
        use crate::scene::bsp_visibility::{
            classify_bsp_visibility, mounted_visibility_decision, VisibilityDecision,
        };
        use crate::scene::render_submission::BspSubmissionFailure;

        let diagnostics = classify_bsp_visibility(
            &self.bsp_mount.mounted_batches,
            &self.bsp_mount,
        );
        submission.bsp_diagnostics = diagnostics.clone();
        log::debug!(
            "BSP submission: total={} eligible={} pvs_visible={} pvs_culled={} conservative={} invalid_member={}",
            diagnostics.total_mounted,
            diagnostics.pvs_eligible,
            diagnostics.pvs_visible,
            diagnostics.pvs_culled,
            diagnostics.conservative_visible,
            diagnostics.invalid_membership,
        );

        for (batch_index, mounted) in self.bsp_mount.mounted_batches.iter().enumerate() {
            // --- PVS Classification ---
            let pvs_decision = mounted_visibility_decision(mounted, &self.bsp_mount);
            match pvs_decision {
                VisibilityDecision::PvsCulled => {
                    continue;
                }
                VisibilityDecision::Visible => {}
            }

            // --- Inline model frustum cull ---
            let batch = &mounted.render;
            let batch_transform = if batch.is_inline_model {
                self.bsp_mount
                    .inline_model_transforms
                    .get(&batch.model_index)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY)
            } else {
                Mat4::IDENTITY
            };

            if batch.is_inline_model {
                if let Some((world_min, world_max)) = self
                    .bsp_mount
                    .inline_model_bounds
                    .get(&batch.model_index)
                    .copied()
                {
                    if !crate::scene::bsp_visibility::aabb_intersects_frustum(
                        world_min, world_max, frustum,
                    ) {
                        continue; // intentional frustum cull
                    }
                }
            }

            // --- Resource validation (fail-closed) ---
            let mesh_id = mounted.mesh;
            let bsp_material_id = mounted.material;
            let source_face_first = mounted
                .render
                .face_indices
                .first()
                .copied()
                .unwrap_or(0);
            let source_face_count = mounted.render.face_indices.len() as u32;
            let model_index = mounted.render.model_index;

            // Mesh slot zero is the explicit no-mesh sentinel. BSP material
            // slot zero is cache-issued for the first real material and must
            // reach cache generation validation during recording.
            if mesh_id == MeshHandle::new(0, 0) {
                let failure = BspSubmissionFailure {
                    batch_index,
                    source_face_first,
                    source_face_count,
                    pipeline_class: None,
                    model_index,
                    reason: format!(
                        "missing required BSP state: mesh={:?} material={:?}",
                        mesh_id, bsp_material_id
                    ),
                };
                if submission.bsp_failure.is_none() {
                    submission.bsp_failure = Some(failure);
                }
                continue;
            }

            submission
                .bsp_draw_items
                .push(crate::scene::render_submission::BspFrameDrawItem {
                    mesh_id,
                    bsp_material_id,
                    transform: batch_transform,
                    batch_index,
                    source_face_first,
                    source_face_count,
                    pipeline_class: None,
                    model_index,
                });
            submission.culling_stats.submitted_draw_items += 1;
        }
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
        if bump_generation(&mut entry.generation) {
            self.free_slots.push(node_id.slot);
        }
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

    // -----------------------------------------------------------------------
    // Bounds computation (Steps 6-8)
    // -----------------------------------------------------------------------

    /// Compute this node's world-space bounds by transforming every local
    /// mesh/proxy AABB through the node's world transform.
    fn compute_node_world_bounds(&self, node_id: SceneNodeId) -> Option<SceneBounds> {
        let node = self.get_node(node_id)?;

        // If an explicit proxy is set and no known meshes exist, use it.
        if let Some(proxy) = node.local_proxy_bounds {
            return proxy
                .transformed(&node.world_transform)
                .map(SceneBounds::Proxy);
        }

        // Aggregate all mesh local bounds into node-local union.
        let mut any_conservative = false;
        let mut local_union: Option<Aabb> = None;

        for entry in &node.mesh_bounds {
            match &entry.bounds {
                SceneBounds::Known(aabb) | SceneBounds::Proxy(aabb) => {
                    if aabb.is_finite() && aabb.is_ordered() {
                        match local_union {
                            Some(ref mut u) => {
                                u.extend_to_enclose(aabb);
                            }
                            None => local_union = Some(*aabb),
                        }
                    } else {
                        any_conservative = true;
                    }
                }
                SceneBounds::ConservativeVisible(_) => {
                    any_conservative = true;
                }
            }
        }

        if any_conservative {
            // Conservative-visible: node is always visible, prune disabled.
            return Some(SceneBounds::ConservativeVisible(
                BoundsUnknownReason::MissingGeometry,
            ));
        }

        // Transform the local union to world space.
        local_union
            .and_then(|local| local.transformed(&node.world_transform))
            .map(SceneBounds::Known)
    }

    /// Post-order: compute subtree bounds for this node and all descendants.
    /// Returns the computed subtree bounds.
    fn compute_subtree_bounds_post_order(&mut self, node_id: SceneNodeId) -> Option<SceneBounds> {
        // Ensure node exists.
        let children = {
            let Some(node) = self.get_node(node_id) else {
                return None;
            };
            node.children.clone()
        };

        // Compute children's subtree bounds first (post-order).
        let mut child_subtree_union: Option<Aabb> = None;
        let mut any_child_conservative = false;

        for child_id in children {
            if !self.is_valid_node_id(child_id) {
                continue;
            }
            if let Some(child_bounds) = self.compute_subtree_bounds_post_order(child_id) {
                match child_bounds {
                    SceneBounds::Known(aabb) | SceneBounds::Proxy(aabb) => {
                        match child_subtree_union {
                            Some(ref mut u) => {
                                u.extend_to_enclose(&aabb);
                            }
                            None => child_subtree_union = Some(aabb),
                        }
                    }
                    SceneBounds::ConservativeVisible(_) => {
                        any_child_conservative = true;
                    }
                }
            }
        }

        // Compute this node's own world bounds.
        // Cache it so culling can read it without recomputation.
        let node_world = self.compute_node_world_bounds(node_id);
        if let Some(ref mut node) = self.get_node_mut(node_id) {
            node.node_world_bounds = node_world;
        }

        // Build subtree: union of node_world + all child subtrees.
        let subtree = if any_child_conservative {
            Some(SceneBounds::ConservativeVisible(
                BoundsUnknownReason::MissingGeometry,
            ))
        } else {
            match (node_world, child_subtree_union) {
                (Some(SceneBounds::Known(nw)), Some(child_union)) => {
                    let mut u = nw;
                    u.extend_to_enclose(&child_union);
                    Some(SceneBounds::Known(u))
                }
                (Some(SceneBounds::Proxy(nw)), Some(child_union)) => {
                    let mut u = nw;
                    u.extend_to_enclose(&child_union);
                    Some(SceneBounds::Proxy(u))
                }
                (Some(SceneBounds::Known(aabb)), None) => Some(SceneBounds::Known(aabb)),
                (Some(SceneBounds::Proxy(aabb)), None) => Some(SceneBounds::Proxy(aabb)),
                (None, Some(child_union)) => Some(SceneBounds::Known(child_union)),
                (None, None) => None,
                (Some(SceneBounds::ConservativeVisible(reason)), _) => {
                    Some(SceneBounds::ConservativeVisible(reason))
                }
            }
        };

        // Cache subtree bounds.
        if let Some(ref mut node) = self.get_node_mut(node_id) {
            node.subtree_world_bounds = subtree;
        }

        subtree
    }

    /// Culling traversal using authoritative bounds.
    /// - Known/proxy node world bounds are tested against the frustum.
    /// - Conservative-visible nodes submit their meshes unconditionally.
    /// - Known subtree bounds may prune the entire branch.
    /// - Unknown subtree bounds always traverse children.
    fn collect_draw_items_culled(
        &self,
        node_id: SceneNodeId,
        submission: &mut RenderSubmission,
        frustum: Option<&Frustum>,
    ) {
        let Some(node) = self.get_node(node_id) else {
            return;
        };

        // Culling is disabled: submit everything.
        if frustum.is_none() {
            for mesh_id in node.meshes.iter().copied() {
                submission.push_draw_item(FrameDrawItem {
                    mesh_id,
                    transform: node.world_transform,
                });
            }
            for child in node.children.iter().copied() {
                if self.is_valid_node_id(child) {
                    self.collect_draw_items_culled(child, submission, frustum);
                }
            }
            return;
        }

        let frustum = frustum.unwrap();

        // If subtree is known and completely outside frustum, prune the whole branch.
        if let Some(ref subtree) = node.subtree_world_bounds {
            if subtree.is_trusted_for_pruning() {
                if let Some(aabb) = subtree.aabb() {
                    if !frustum.intersects_aabb(aabb) {
                        return; // entire subtree is off-screen
                    }
                }
            }
        }

        // Determine if this node's own meshes should be submitted.
        let node_visible = match &node.node_world_bounds {
            Some(SceneBounds::Known(aabb)) | Some(SceneBounds::Proxy(aabb)) => {
                frustum.intersects_aabb(aabb)
            }
            Some(SceneBounds::ConservativeVisible(_)) | None => {
                // Conservative-visible or empty: submit meshes unconditionally.
                true
            }
        };

        if node_visible && !node.meshes.is_empty() {
            for mesh_id in node.meshes.iter().copied() {
                submission.push_draw_item(FrameDrawItem {
                    mesh_id,
                    transform: node.world_transform,
                });
            }
        }

        // Traverse children. Conservative-visible subtrees always traverse.
        for child in node.children.iter().copied() {
            if self.is_valid_node_id(child) {
                self.collect_draw_items_culled(child, submission, Some(frustum));
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
        if bump_generation(&mut entry.generation) {
            self.free_point_light_slots.push(id.slot);
        }
        true
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

    pub(crate) fn add_directional_light(&mut self, light: DirectionalLight) -> DirectionalLightId {
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
        if bump_generation(&mut entry.generation) {
            self.free_directional_light_slots.push(id.slot);
        }
        if self.shadow_casting_directional == Some(id) {
            self.shadow_casting_directional = None;
        }
        true
    }

    /// Returns the active directional light (the public facade enforces one).
    pub(crate) fn get_active_directional_light(&self) -> Option<DirectionalLight> {
        self.directional_lights.iter().find_map(|entry| entry.light)
    }

    /// Returns all active directional lights.
    pub(crate) fn get_active_directional_lights(&self) -> Vec<DirectionalLight> {
        self.directional_lights
            .iter()
            .filter_map(|entry| entry.light)
            .collect()
    }

    pub(crate) fn active_directional_light_count(&self) -> usize {
        self.directional_lights
            .iter()
            .filter(|entry| entry.light.is_some())
            .count()
    }

    pub(crate) fn active_point_light_count(&self) -> usize {
        self.point_lights
            .iter()
            .filter(|entry| entry.light.is_some())
            .count()
    }

    pub(crate) fn reserve_point_light_slots(&mut self, total_slots: usize) {
        if total_slots > self.point_lights.capacity() {
            self.point_lights
                .reserve(total_slots - self.point_lights.len());
        }
    }

    pub(crate) fn active_spot_light_count(&self) -> usize {
        self.spot_lights
            .iter()
            .filter(|entry| entry.light.is_some())
            .count()
    }

    pub(crate) fn set_shadow_casting_directional(&mut self, id: Option<DirectionalLightId>) {
        self.shadow_casting_directional = id;
    }

    pub(crate) fn shadow_casting_directional(&self) -> Option<DirectionalLightId> {
        self.shadow_casting_directional
    }

    // ── Spot light lifecycle ────────────────────────────────────────────

    pub(crate) fn validate_spot_light_ref(&self, id: SpotLightId) -> Result<(), SpotLightRefError> {
        let Some(entry) = self.spot_lights.get(id.slot as usize) else {
            return Err(SpotLightRefError::OutOfBounds);
        };
        if entry.generation != id.generation {
            return Err(SpotLightRefError::GenerationMismatch);
        }
        if entry.light.is_none() {
            return Err(SpotLightRefError::Vacant);
        }
        Ok(())
    }

    pub(crate) fn add_spot_light(&mut self, light: SpotLight) -> SpotLightId {
        if let Some(slot) = self.free_spot_light_slots.pop() {
            let entry = &mut self.spot_lights[slot as usize];
            entry.light = Some(light);
            return SpotLightId {
                slot,
                generation: entry.generation,
            };
        }
        let slot = self.spot_lights.len() as u32;
        self.spot_lights.push(SpotLightEntry {
            generation: 0,
            light: Some(light),
        });
        SpotLightId {
            slot,
            generation: 0,
        }
    }

    pub(crate) fn update_spot_light(&mut self, id: SpotLightId, light: SpotLight) -> bool {
        let Some(entry) = self.spot_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = Some(light);
        true
    }

    pub(crate) fn remove_spot_light(&mut self, id: SpotLightId) -> bool {
        let Some(entry) = self.spot_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        entry.light = None;
        if bump_generation(&mut entry.generation) {
            self.free_spot_light_slots.push(id.slot);
        }
        true
    }

    pub(crate) fn get_active_spot_lights(&self) -> Vec<SpotLight> {
        self.spot_lights
            .iter()
            .filter_map(|entry| entry.light)
            .collect()
    }

    // ── BSP mount management ────────────────────────────────────────

    /// Set the BSP mount state for PVS-aware culling and light selection.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_mount(&mut self, mount: crate::scene::bsp_visibility::BspMountState) {
        self.bsp_mount = mount;
    }

    /// Clear the BSP mount, disabling PVS culling and BSP light selection.
    #[cfg(feature = "bsp")]
    pub(crate) fn clear_bsp_mount(&mut self) {
        self.bsp_mount.deactivate();
    }

    /// Detach the active BSP mount and return its previous scene state.
    ///
    /// The returned state is no longer referenced by scene submission. This
    /// method does not itself enqueue GPU resource retirement; a caller with
    /// renderer/core cache ownership must perform that work.
    #[cfg(feature = "bsp")]
    pub(crate) fn retire_bsp_mount(
        &mut self,
    ) -> Option<crate::scene::bsp_visibility::BspMountState> {
        if !self.bsp_mount.active {
            return None;
        }
        let mut retired = crate::scene::bsp_visibility::BspMountState::new();
        std::mem::swap(&mut self.bsp_mount, &mut retired);
        // The swapped-in mount is empty/inactive; the caller receives the
        // previous active mount state. GPU cache retirement requires a
        // separate renderer/core handoff.
        Some(retired)
    }

    /// Return whether a BSP mount is currently active.
    #[cfg(feature = "bsp")]
    pub(crate) fn has_bsp_mount(&self) -> bool {
        self.bsp_mount.active
    }

    /// Update BSP PVS for the current camera position.
    /// Called before `build_submission` when a BSP mount is active.
    #[cfg(feature = "bsp")]
    pub(crate) fn update_bsp_pvs(&mut self) {
        let cam_pos = self.camera.cam_pos;
        self.bsp_mount.update_pvs(cam_pos);
    }

    /// Set per-frame BSP frame values (style intensities, liquid time).
    ///
    /// These are uploaded to the BSP frame-values UBO each frame.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_frame_values(&mut self, style_intensities: [f32; 64], liquid_time: f32) {
        self.bsp_mount.frame_style_intensities = style_intensities;
        self.bsp_mount.frame_liquid_time = liquid_time;
    }

    /// Set per-model transforms for inline model draws.
    ///
    /// `transforms` is model_index → world-space Mat4.
    /// Model 0 (worldspawn) is always identity.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_inline_model_transforms(
        &mut self,
        transforms: std::collections::HashMap<u32, glam::Mat4>,
    ) {
        self.bsp_mount.inline_model_transforms = transforms;
    }

    /// Set per-model world-space bounds for inline model culling.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_inline_model_bounds(
        &mut self,
        bounds: std::collections::HashMap<u32, (glam::Vec3, glam::Vec3)>,
    ) {
        self.bsp_mount.inline_model_bounds = bounds;
    }
}

/// Compute a world-space pickable AABB for a node using an on-the-fly
/// world transform (`world`).  Mirrors the logic in
/// `node_pick_bounds_exact` + `compute_node_world_bounds` but operates
/// on scratch data without touching cached fields.
fn pick_bounds_readonly(node: &SceneNode, world: &Mat4) -> Option<Aabb> {
    // If an explicit local proxy is set and no known meshes exist, use it.
    if let Some(proxy) = node.local_proxy_bounds {
        return proxy.transformed(world);
    }

    // Aggregate local mesh bounds then transform.
    let mut any_conservative = false;
    let mut local_union: Option<Aabb> = None;

    for entry in &node.mesh_bounds {
        match &entry.bounds {
            SceneBounds::Known(aabb) | SceneBounds::Proxy(aabb) => {
                if aabb.is_finite() && aabb.is_ordered() {
                    match local_union {
                        Some(ref mut u) => {
                            u.extend_to_enclose(aabb);
                        }
                        None => local_union = Some(*aabb),
                    }
                } else {
                    any_conservative = true;
                }
            }
            SceneBounds::ConservativeVisible(_) => {
                any_conservative = true;
            }
        }
    }

    if any_conservative {
        // Conservative-visible mesh-bearing: skip exact hit.
        if !node.meshes.is_empty() {
            return None;
        }
        // Empty group node: fall through to editor proxy.
    }

    if let Some(local) = local_union {
        return local.transformed(world);
    }

    // No mesh bounds at all: editor-origin proxy for empty group nodes.
    if node.meshes.is_empty() {
        let half = 0.25;
        let local = Aabb::from_min_max(Vec3::splat(-half), Vec3::splat(half));
        return local.transformed(world);
    }

    // Mesh-bearing but no bounds: last-resort small transform-aware proxy.
    let half = 0.5;
    let local = Aabb::from_min_max(Vec3::splat(-half), Vec3::splat(half));
    local.transformed(world)
}

/// Return the pickable world AABB for a node.
/// - Known/proxy node_world_bounds are used directly.
/// - Empty non-mesh group nodes get a small editor-origin helper.
/// - Conservative-visible mesh-bearing nodes return `None` (skip exact hit).
fn node_pick_bounds_exact(node: &SceneNode) -> Option<Aabb> {
    if let Some(ref bounds) = node.node_world_bounds {
        match bounds {
            SceneBounds::Known(aabb) | SceneBounds::Proxy(aabb) => {
                if aabb.is_finite() {
                    return Some(*aabb);
                }
            }
            SceneBounds::ConservativeVisible(_) => {
                if !node.meshes.is_empty() {
                    return None; // mesh-bearing unknown: skip exact hit
                }
                // fall through to editor proxy for empty group nodes
            }
        }
    }

    // Editor-origin proxy for empty non-mesh group nodes.
    if node.meshes.is_empty() {
        let half = 0.25;
        let local = Aabb::from_min_max(Vec3::splat(-half), Vec3::splat(half));
        return Aabb::transformed(&local, &node.world_transform);
    }

    // Mesh-bearing but no bounds: use a small transform-aware proxy
    // as last-resort fallback.
    let half = 0.5;
    let local = Aabb::from_min_max(Vec3::splat(-half), Vec3::splat(half));
    Aabb::transformed(&local, &node.world_transform)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReparentError {
    InvalidNode(SceneNodeRefError),
    InvalidParent(SceneNodeRefError),
    Cycle,
}

/// Bump a slot generation by one. Returns `true` when the slot can safely
/// return to the free list. Returns `false` when the generation has reached
/// `u32::MAX` and the slot is terminally exhausted (do NOT return to free list).
fn bump_generation(generation: &mut u32) -> bool {
    match generation.checked_add(1) {
        Some(next) => {
            *generation = next;
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bump_generation, PointLightRefError, SceneNode, SceneNodeId, SceneNodeRefError, SceneWorld,
    };
    use crate::api::scene::{BoundsUnknownReason, MeshBoundsEntry, SceneBounds};
    use crate::data::camera::{Aabb, Ray};
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
        use crate::api::scene::{MeshBoundsEntry, SceneBounds};
        use crate::data::camera::Aabb;

        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        let unit_proxy =
            SceneBounds::Proxy(Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)));
        let offscreen_parent = scene.add_node(
            Some(root),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
                meshes: vec![MeshHandle::new(1, 0)],
                mesh_bounds: vec![MeshBoundsEntry {
                    mesh: MeshHandle::new(1, 0),
                    bounds: unit_proxy,
                }],
                ..SceneNode::default()
            },
        );
        scene.add_node(
            Some(offscreen_parent),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(-100.0, 0.0, -5.0)),
                meshes: vec![MeshHandle::new(2, 0)],
                mesh_bounds: vec![MeshBoundsEntry {
                    mesh: MeshHandle::new(2, 0),
                    bounds: unit_proxy,
                }],
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
    fn pick_ray_readonly_does_not_mutate_dirty_or_cached_state() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);

        let before = scene.get_node(root).expect("root should exist").clone();
        assert!(before.dirty);
        assert_eq!(before.world_transform, Mat4::IDENTITY);
        assert!(before.node_world_bounds.is_none());
        assert!(before.subtree_world_bounds.is_none());

        let ray = Ray {
            origin: Vec3::new(3.0, 0.0, 5.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
        };
        assert_eq!(scene.pick_ray_readonly(&ray), Some(root));

        let after = scene.get_node(root).expect("root should exist");
        assert_eq!(after.dirty, before.dirty);
        assert_eq!(after.world_transform, before.world_transform);
        assert_eq!(after.node_world_bounds, before.node_world_bounds);
        assert_eq!(after.subtree_world_bounds, before.subtree_world_bounds);
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

    // -----------------------------------------------------------------------
    // Conservative bounds tests (Phase 06)
    // -----------------------------------------------------------------------

    fn make_unit_known() -> SceneBounds {
        SceneBounds::Known(Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)))
    }

    fn make_unit_proxy() -> SceneBounds {
        SceneBounds::Proxy(Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)))
    }

    fn node_with_mesh_and_bounds(mesh_slot: u32, bounds: SceneBounds) -> SceneNode {
        SceneNode {
            meshes: vec![MeshHandle::new(mesh_slot, 0)],
            mesh_bounds: vec![MeshBoundsEntry {
                mesh: MeshHandle::new(mesh_slot, 0),
                bounds,
            }],
            ..SceneNode::default()
        }
    }

    fn node_with_mesh_and_transform(
        mesh_slot: u32,
        bounds: SceneBounds,
        local_transform: Mat4,
    ) -> SceneNode {
        SceneNode {
            local_transform,
            meshes: vec![MeshHandle::new(mesh_slot, 0)],
            mesh_bounds: vec![MeshBoundsEntry {
                mesh: MeshHandle::new(mesh_slot, 0),
                bounds,
            }],
            ..SceneNode::default()
        }
    }

    // -- Exact local AABB under translation --

    #[test]
    fn known_bounds_transform_to_world_correctly() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        scene.build_submission();

        let node = scene.get_node(root).unwrap();
        let world = node.node_world_bounds.unwrap();
        if let SceneBounds::Known(aabb) = world {
            assert_eq!(aabb.min, Vec3::new(9.5, -0.5, -0.5));
            assert_eq!(aabb.max, Vec3::new(10.5, 0.5, 0.5));
        } else {
            panic!("expected Known world bounds");
        }
    }

    // -- Negative scale --

    #[test]
    fn negative_scale_produces_correct_world_bounds() {
        let mut scene = SceneWorld::new();
        // Scale -1 on X flips the AABB across the YZ plane at origin.
        let root = scene.add_node(
            None,
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_scale(Vec3::new(-2.0, 1.0, 1.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        scene.build_submission();

        let node = scene.get_node(root).unwrap();
        let world = node.node_world_bounds.unwrap();
        if let SceneBounds::Known(aabb) = world {
            // Unit [-0.5,0.5] scaled by -2 on X becomes [-1, 1] but min/max swap.
            // The 8-corner transform handles this correctly.
            assert!((aabb.max.x - aabb.min.x - 2.0).abs() < 0.001);
            assert!(aabb.min.x < aabb.max.x);
        } else {
            panic!("expected Known world bounds under negative scale");
        }
    }

    // -- Rotation (45° around Z) --

    #[test]
    fn rotation_45z_produces_correct_world_bounds() {
        let mut scene = SceneWorld::new();
        let rot = glam::Quat::from_rotation_z(45.0_f32.to_radians());
        let root = scene.add_node(
            None,
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_rotation_translation(rot, Vec3::ZERO),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        scene.build_submission();

        let node = scene.get_node(root).unwrap();
        let world = node.node_world_bounds.unwrap();
        if let SceneBounds::Known(aabb) = world {
            // The diagonal of the unit cube is sqrt(2) ≈ 1.414. After 45° rotation,
            // the AABB should enclose the rotated cube.
            let extent = aabb.max - aabb.min;
            assert!(
                (extent.x - 2.0_f32.sqrt()).abs() < 0.01,
                "rotated extent {extent:?}"
            );
        } else {
            panic!("expected Known world bounds under rotation");
        }
    }

    // -- Subtree union --

    #[test]
    fn subtree_bounds_union_children() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        let child_a = scene.add_node(
            Some(root),
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
            ),
        );
        let child_b = scene.add_node(
            Some(root),
            node_with_mesh_and_transform(
                2,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.enable_frustum_culling = false;
        scene.build_submission();

        let root_node = scene.get_node(root).unwrap();
        let subtree = root_node.subtree_world_bounds.unwrap();
        if let SceneBounds::Known(aabb) = subtree {
            // Should enclose both children.
            assert!(aabb.min.x <= -2.5);
            assert!(aabb.max.x >= 2.5);
        } else {
            panic!("expected Known subtree bounds");
        }
    }

    // -- Dirty invalidation --

    #[test]
    fn dirty_flag_invalidates_bounds() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        scene.build_submission();

        let node = scene.get_node(root).unwrap();
        assert!(node.node_world_bounds.is_some());

        // Move the node far right and mark dirty.
        if let Some(n) = scene.get_node_mut(root) {
            n.local_transform = Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0));
            n.dirty = true;
            n.node_world_bounds = None;
            n.subtree_world_bounds = None;
        }

        // Bounds should be recomputed on next build.
        scene.build_submission();
        let node = scene.get_node(root).unwrap();
        if let Some(SceneBounds::Known(aabb)) = node.node_world_bounds {
            assert!(aabb.min.x > 99.0);
        } else {
            panic!("expected Known bounds after dirty recompute");
        }
    }

    // -- Missing/stale/skinned/deformed visibility --

    #[test]
    fn conservative_visible_meshes_always_submit() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            node_with_mesh_and_bounds(
                1,
                SceneBounds::ConservativeVisible(BoundsUnknownReason::Skinned),
            ),
        );
        scene.set_root(root);
        // Camera looks down -Z, node at origin. Conservative-visible must submit.
        scene.update_camera(
            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        let submission = scene.build_submission();
        assert_eq!(submission.draw_items.len(), 1);
        assert_eq!(submission.draw_items[0].mesh_id, MeshHandle::new(1, 0));
    }

    #[test]
    fn conservative_visible_dominates_known_in_same_node() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(
            None,
            SceneNode {
                meshes: vec![MeshHandle::new(1, 0), MeshHandle::new(2, 0)],
                mesh_bounds: vec![
                    MeshBoundsEntry {
                        mesh: MeshHandle::new(1, 0),
                        bounds: make_unit_known(),
                    },
                    MeshBoundsEntry {
                        mesh: MeshHandle::new(2, 0),
                        bounds: SceneBounds::ConservativeVisible(BoundsUnknownReason::Deformed),
                    },
                ],
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
        // Both meshes submit because one is conservative-visible.
        assert_eq!(submission.draw_items.len(), 2);
    }

    // -- Explicit proxy tagging --

    #[test]
    fn proxy_bounds_are_used_when_no_known_mesh() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        // Set local proxy bounds (simulating set_node_proxy_bounds).
        if let Some(n) = scene.get_node_mut(root) {
            n.local_proxy_bounds = Some(Aabb::from_min_max(
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(4.0, 5.0, 6.0),
            ));
        }
        scene.set_root(root);
        scene.enable_frustum_culling = false;
        scene.build_submission();

        let node = scene.get_node(root).unwrap();
        // Proxy should persist when no known meshes are present.
        assert!(matches!(
            node.node_world_bounds,
            Some(SceneBounds::Proxy(_))
        ));
        assert!(node.subtree_world_bounds.is_some());
    }

    // -- Descendant independence --

    #[test]
    fn off_screen_known_parent_does_not_hide_in_frustum_known_child() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        // Parent is far right (outside frustum).
        let parent = scene.add_node(
            Some(root),
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
            ),
        );
        // Child cancels the translation and sits at origin (in frustum).
        scene.add_node(
            Some(parent),
            node_with_mesh_and_transform(
                2,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        let submission = scene.build_submission();
        // Only the child should be submitted; parent is off-screen.
        let ids: Vec<_> = submission.draw_items.iter().map(|d| d.mesh_id).collect();
        assert_eq!(ids, vec![MeshHandle::new(2, 0)]);
    }

    // -- Safe subtree pruning --

    #[test]
    fn known_subtree_entirely_outside_frustum_is_pruned() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        // A parent node whose entire subtree is far left.
        let off_parent = scene.add_node(
            Some(root),
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(-50.0, 0.0, 0.0)),
            ),
        );
        // Child also far left.
        scene.add_node(
            Some(off_parent),
            node_with_mesh_and_transform(
                2,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        let submission = scene.build_submission();
        // No draws: the entire subtree is outside.
        assert!(submission.draw_items.is_empty());
    }

    #[test]
    fn conservative_visible_parent_does_not_prune_visible_child() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        // Parent is at origin, conservative-visible (always draws).
        let parent = scene.add_node(
            Some(root),
            node_with_mesh_and_bounds(
                1,
                SceneBounds::ConservativeVisible(BoundsUnknownReason::Skinned),
            ),
        );
        // Child is also at origin, known bounds, in frustum.
        // A known in-frustum child under a skip-pruning parent should still draw.
        scene.add_node(
            Some(parent),
            node_with_mesh_and_transform(2, make_unit_known(), Mat4::IDENTITY),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        let submission = scene.build_submission();
        // Both submit: parent always draws, child is in frustum.
        assert_eq!(submission.draw_items.len(), 2);
    }

    // -- Handle retirement (BoundsEntry) --

    #[test]
    fn bounds_entry_retirement_class_exists() {
        use crate::data::retirement::{FrameSerial, GpuRetirementQueue, RetirementClass};

        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial::new(5), 42);
        assert_eq!(q.pending_by_class(RetirementClass::BoundsEntry), 1);
        let reaped = q.reap_through(FrameSerial::new(5)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 42);
    }

    // -- Merge/undo metadata alignment --

    #[test]
    fn add_node_with_parts_and_bounds_preserves_mesh_bounds() {
        let mut scene = SceneWorld::new();
        let bounds_entry = MeshBoundsEntry {
            mesh: MeshHandle::new(1, 0),
            bounds: SceneBounds::Known(Aabb::from_min_max(
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            )),
        };
        let mounted = scene.add_node_with_parts_and_bounds(
            None,
            Mat4::IDENTITY,
            vec![MeshHandle::new(1, 0)],
            vec![bounds_entry],
        );
        let mounted_node = scene.get_node(mounted).unwrap();
        assert_eq!(mounted_node.mesh_bounds.len(), 1);
        assert_eq!(mounted_node.mesh_bounds[0].mesh, MeshHandle::new(1, 0));
        assert!(matches!(
            mounted_node.mesh_bounds[0].bounds,
            SceneBounds::Known(_)
        ));
    }

    // -- Empty group nodes remain traversable --

    #[test]
    fn empty_group_node_traverses_children() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        let group = scene.add_node(
            Some(root),
            SceneNode {
                local_transform: Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
                ..SceneNode::default()
            },
        );
        let visible = scene.add_node(
            Some(group),
            node_with_mesh_and_transform(
                1,
                make_unit_known(),
                Mat4::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
            ),
        );
        scene.set_root(root);
        scene.update_camera(
            Mat4::IDENTITY,
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            Vec3::ZERO,
        );
        let submission = scene.build_submission();
        // The group node has no meshes but its child is at origin.
        assert_eq!(submission.draw_items.len(), 1);
        assert_eq!(submission.draw_items[0].mesh_id, MeshHandle::new(1, 0));
    }

    // ── Generation exhaustion & handle safety ────────────────────────────

    #[test]
    fn bump_generation_returns_false_at_u32_max() {
        let mut gen = u32::MAX;
        assert!(!bump_generation(&mut gen));
        assert_eq!(gen, u32::MAX);
    }

    #[test]
    fn bump_generation_preserves_identity_through_normal_range() {
        let mut gen = 7;
        assert!(bump_generation(&mut gen));
        assert_eq!(gen, 8);
    }

    #[test]
    fn node_removal_at_max_generation_terminally_retires_slot() {
        let mut scene = SceneWorld::new();
        let node = scene.add_node(None, SceneNode::default());

        // Artificially push generation to u32::MAX and create a matching handle.
        scene.nodes[node.slot as usize].generation = u32::MAX;
        let max_gen_id = SceneNodeId::new(node.slot, u32::MAX);

        assert!(scene.remove_node(max_gen_id));
        assert!(!scene.is_valid_node_id(max_gen_id));

        // Slot must NOT be in free list (terminal exhaustion)
        assert!(!scene.free_slots.contains(&node.slot));

        // Any lookup for the slot still rejects
        assert!(!scene.is_valid_node_id(max_gen_id));
    }

    #[test]
    fn point_light_at_max_generation_terminally_retires() {
        use crate::api::scene::PointLight;

        let mut scene = SceneWorld::new();
        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 1.0,
        };
        let id = scene.add_point_light(light);

        scene.point_lights[id.slot as usize].generation = u32::MAX;
        let max_gen_id = crate::api::scene::PointLightId {
            slot: id.slot,
            generation: u32::MAX,
        };
        assert!(scene.remove_point_light(max_gen_id));
        assert!(!scene.free_point_light_slots.contains(&id.slot));
        // After removal, the slot is vacant (generation stays at u32::MAX, terminal).
        assert!(matches!(
            scene.validate_point_light_ref(max_gen_id),
            Err(PointLightRefError::Vacant)
        ));
    }

    #[test]
    fn out_of_range_slot_rejected() {
        let scene = SceneWorld::new();
        let bad_id = SceneNodeId::new(99999, 0);
        assert!(matches!(
            scene.validate_node_ref(bad_id),
            Err(SceneNodeRefError::OutOfBounds)
        ));
        assert!(scene.get_node(bad_id).is_none());
    }

    #[test]
    fn generation_mismatch_rejected() {
        let mut scene = SceneWorld::new();
        let id = scene.add_node(None, SceneNode::default());
        assert!(scene.is_valid_node_id(id));

        let wrong_gen = SceneNodeId::new(id.slot, id.generation + 1);
        assert!(!scene.is_valid_node_id(wrong_gen));
    }

    #[test]
    fn repeated_allocate_remove_cycle_preserves_handle_safety() {
        let mut scene = SceneWorld::new();

        for cycle in 0..10u32 {
            let id = scene.add_node(None, SceneNode::default());
            assert_eq!(id.slot, 0);
            assert_eq!(id.generation, cycle);
            assert!(scene.is_valid_node_id(id));
            assert!(scene.remove_node(id));
            assert!(!scene.is_valid_node_id(id));
        }

        // After 10 cycles, slot 0 generation should be 10
        let id = SceneNodeId::new(0, 9);
        assert!(!scene.is_valid_node_id(id));

        let fresh = scene.add_node(None, SceneNode::default());
        assert_eq!(fresh.slot, 0);
        assert_eq!(fresh.generation, 10);
        assert!(scene.is_valid_node_id(fresh));
    }

    #[test]
    fn repeated_light_remove_reallocate_rejects_stale() {
        use crate::api::scene::PointLight;

        let mut scene = SceneWorld::new();
        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 1.0,
        };

        let id0 = scene.add_point_light(light);
        assert_eq!(id0.generation, 0);
        assert!(scene.remove_point_light(id0));

        // Old handle must be rejected
        assert!(matches!(
            scene.validate_point_light_ref(id0),
            Err(PointLightRefError::GenerationMismatch)
        ));

        let id1 = scene.add_point_light(light);
        assert_eq!(id1.slot, 0);
        assert_eq!(id1.generation, 1);
        assert!(scene.validate_point_light_ref(id1).is_ok());
        // Old handle still rejected
        assert!(matches!(
            scene.validate_point_light_ref(id0),
            Err(PointLightRefError::GenerationMismatch)
        ));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_mount_submits_draws_and_lights_without_scene_root() {
        use crate::data::bsp_import::MountedBspBatch;

        let mut scene = SceneWorld::new();
        let mut mount = crate::scene::bsp_visibility::BspMountState::new();
        mount.activate();
        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                style_ids: [0, 255, 255, 255],
                model_index: 0,
            },
            leaf_signature: Vec::new(),
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let mesh = MeshHandle::new(7, 0);
        let material = crate::data::handles::BspMaterialHandle::new(0, 0);
        let mounted = MountedBspBatch::try_new(
            &batch,
            mesh,
            material,
            (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        )
        .expect("valid mounted batch");
        mount
            .set_render_assets_from_canonical(
                &[mounted],
                vec![mesh],
                vec![Some(material)],
                vec![bsp::extract::LightDescriptor {
                    entity_index: 0,
                    origin: Vec3::new(1.0, 0.0, 0.0),
                    intensity: 64.0,
                    color: [1.0, 0.5, 0.25],
                    radius: 128.0,
                    style: None,
                }],
            )
            .expect("canonical batch publish");
        scene.set_bsp_mount(mount);

        let submission = scene.build_submission();

        assert_eq!(submission.bsp_draw_items.len(), 1);
        assert_eq!(submission.bsp_draw_items[0].mesh_id, MeshHandle::new(7, 0));
        assert_eq!(
            submission.bsp_draw_items[0].bsp_material_id,
            crate::data::handles::BspMaterialHandle::new(0, 0)
        );
        assert_eq!(submission.point_lights.len(), 1);
        assert_eq!(submission.point_lights[0].color, Vec3::new(1.0, 0.5, 0.25));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_mount_retirement_returns_active_state_and_leaves_scene_empty() {
        use crate::data::bsp_import::MountedBspBatch;

        let mut scene = SceneWorld::new();
        let mut mount = crate::scene::bsp_visibility::BspMountState::new();
        mount.activate();
        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
                style_ids: [0, 255, 255, 255],
                model_index: 0,
            },
            leaf_signature: Vec::new(),
            face_indices: vec![0],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let mesh = MeshHandle::new(7, 0);
        let material = crate::data::handles::BspMaterialHandle::new(3, 0);
        let mounted = MountedBspBatch::try_new(
            &batch,
            mesh,
            material,
            (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        )
        .expect("valid mounted batch");
        mount
            .set_render_assets_from_canonical(&[mounted], vec![mesh], vec![Some(material)], vec![])
            .expect("canonical batch publish");
        scene.set_bsp_mount(mount);
        assert!(scene.has_bsp_mount());

        let retired = scene.retire_bsp_mount();
        assert!(retired.is_some());
        assert!(!scene.has_bsp_mount());

        // Scene submission after retirement should have no BSP draws.
        let submission = scene.build_submission();
        assert!(submission.bsp_draw_items.is_empty());
        assert!(submission.bsp_selected_lights.is_empty());
        assert!(submission.bsp_failure.is_none());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_mount_retire_on_inactive_returns_none() {
        let mut scene = SceneWorld::new();
        assert!(!scene.has_bsp_mount());
        let retired = scene.retire_bsp_mount();
        assert!(retired.is_none());
    }
}
