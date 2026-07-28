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
use crate::object::identity::{ObjectId, SceneRuntimeId};
use crate::scene::object_store::{
    mint_provenance, CreateDirectionalLightPlan, CreateNodePlan, CreatePointLightPlan,
    CreateSpotLightPlan, ObjectHandle, ObjectRecord, RemoveDirectionalLightPlan, RemoveNodePlan,
    RemovePointLightPlan, RemoveSpotLightPlan, SceneNodeRemovalSnapshot,
};
use crate::scene::render_submission::{
    FrameDirectionalLight, FrameDrawItem, FramePointLight, FrameSpotLight, RenderSubmission,
    MAX_DIRECTIONAL_LIGHTS_GPU, MAX_POINT_LIGHTS_GPU, MAX_SPOT_LIGHTS_GPU,
};
use engine_events::{ObjectKind, SceneObjectId};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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
    node: Option<(SceneNode, ObjectRecord)>,
}

#[derive(Clone, Debug)]
pub(crate) struct RestorableSceneSubtree {
    node: SceneNode,
    record: ObjectRecord,
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
    light: Option<(PointLight, ObjectRecord)>,
}

#[derive(Clone, Debug)]
struct DirectionalLightEntry {
    generation: u32,
    light: Option<(DirectionalLight, ObjectRecord)>,
}

#[derive(Clone, Debug)]
struct SpotLightEntry {
    generation: u32,
    light: Option<(SpotLight, ObjectRecord)>,
}

pub struct SceneWorld {
    /// Opaque runtime provenance, minted once at construction.
    provenance: SceneRuntimeId,
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
    /// Reverse index: persistent SceneObjectId → typed runtime handle.
    reverse_index: HashMap<SceneObjectId, ObjectHandle>,
    /// When true, mesh-backed nodes outside the camera frustum are omitted
    /// from `build_submission`. Descendants are tested independently. Enabled
    /// by default.
    pub enable_frustum_culling: bool,
    /// Published BSP mount for PVS-aware culling, light selection, and
    /// depth-sorting. `None` when no mount is active. The lease moves with
    /// the mount through every ownership transition.
    #[cfg(feature = "bsp")]
    pub(crate) bsp_mount: Option<crate::api::bsp::PublishedBspMount>,
    /// Phase 07: Pending evidence request data for the next submission build.
    /// Stored as (corpus_identity, request_identity, visibility).
    #[cfg(feature = "bsp")]
    pub(crate) pending_evidence_data: Option<(String, String, crate::api::bsp::BspEvidenceVisibility)>,
    /// Phase 07: Frame number for the pending evidence request.
    #[cfg(feature = "bsp")]
    pub(crate) pending_evidence_frame: u32,
}

impl Default for SceneWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneWorld {
    pub fn new() -> Self {
        Self {
            provenance: mint_provenance(),
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
            reverse_index: HashMap::new(),
            enable_frustum_culling: true,
            #[cfg(feature = "bsp")]
            bsp_mount: None,
            #[cfg(feature = "bsp")]
            pending_evidence_data: None,
            #[cfg(feature = "bsp")]
            pending_evidence_frame: 0,
        }
    }

    /// Access the world's provenance token.
    pub(crate) fn provenance(&self) -> SceneRuntimeId {
        self.provenance
    }

    /// Test-only: artificially push a node slot's generation to u32::MAX.
    #[doc(hidden)]
    pub fn test_set_generation_max(&mut self, id: SceneNodeId) -> bool {
        let Some(entry) = self.nodes.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.node.is_none() {
            return false;
        }
        entry.generation = u32::MAX;
        true
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
                .map(|(node, _record)| (SceneNodeId::new(slot as u32, entry.generation), node))
        })
    }

    /// Returns all active point lights with their IDs.
    pub(crate) fn serializable_lights(&self) -> impl Iterator<Item = (PointLightId, &PointLight)> {
        self.point_lights
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry.light.as_ref().map(|(light, _record)| {
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
                entry.light.as_ref().map(|(light, _record)| {
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
                entry.light.as_ref().map(|(light, _record)| {
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
        entry.node.as_ref().map(|(node, _)| node)
    }

    pub(crate) fn get_node_mut(&mut self, id: SceneNodeId) -> Option<&mut SceneNode> {
        let entry = self.nodes.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_mut().map(|(node, _)| node)
    }

    /// Get a shared reference to a node's record.
    pub fn get_node_record(&self, id: SceneNodeId) -> Option<&ObjectRecord> {
        let entry = self.nodes.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_ref().map(|(_, record)| record)
    }

    /// Get a mutable reference to a node's record.
    pub(crate) fn get_node_record_mut(&mut self, id: SceneNodeId) -> Option<&mut ObjectRecord> {
        let entry = self.nodes.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_mut().map(|(_, record)| record)
    }

    /// Get mutable references to both the node and its record.
    pub(crate) fn get_node_with_record_mut(
        &mut self,
        id: SceneNodeId,
    ) -> Option<(&mut SceneNode, &mut ObjectRecord)> {
        let entry = self.nodes.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_mut().map(|(node, record)| (node, record))
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
        node: SceneNode,
    ) -> SceneNodeId {
        let record = ObjectRecord::for_new_node(None);
        let plan = self.prepare_create_node(parent, node, record);
        self.commit_create_node(plan)
    }

    pub(crate) fn add_node_with_record(
        &mut self,
        parent: Option<SceneNodeId>,
        node: SceneNode,
        record: ObjectRecord,
    ) -> SceneNodeId {
        let plan = self.prepare_create_node(parent, node, record);
        self.commit_create_node(plan)
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

    // ── Prepare / commit lifecycle ───────────────────────────────────

    /// Validate and stage a node creation. Resolves parent, allocates a
    /// slot and generation, and produces an infallible commit plan.
    pub(crate) fn prepare_create_node(
        &mut self,
        parent: Option<SceneNodeId>,
        mut node: SceneNode,
        record: ObjectRecord,
    ) -> CreateNodePlan {
        let resolved_parent = parent.filter(|id| self.is_valid_node_id(*id));
        node.parent = resolved_parent;

        let (slot, generation, is_new_slot) = if let Some(free_slot) = self.free_slots.pop() {
            let entry = &mut self.nodes[free_slot as usize];
            debug_assert!(entry.node.is_none(), "free slot list contained a live node");
            (free_slot, entry.generation, false)
        } else {
            let slot = self.nodes.len() as u32;
            (slot, 0, true)
        };

        CreateNodePlan {
            slot,
            generation,
            node,
            record,
            parent: resolved_parent,
            is_new_slot,
        }
    }

    /// Commit a node creation plan. Infallible after successful prepare.
    pub(crate) fn commit_create_node(&mut self, plan: CreateNodePlan) -> SceneNodeId {
        let CreateNodePlan {
            slot,
            generation,
            node,
            record,
            parent,
            is_new_slot,
        } = plan;

        let persistent_id = record.persistent_id.clone();
        let id = SceneNodeId::new(slot, generation);

        // Update reverse index
        self.reverse_index
            .insert(persistent_id, ObjectHandle::Node(id));

        if is_new_slot {
            self.nodes.push(SceneNodeEntry {
                generation,
                node: Some((node, record)),
            });
        } else {
            let entry = &mut self.nodes[slot as usize];
            entry.node = Some((node, record));
        }

        // Attach to parent or set as root.
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.get_node_mut(parent_id) {
                parent_node.children.push(id);
            }
        } else if self.root.is_none() {
            self.root = Some(id);
        }

        id
    }

    pub(crate) fn remove_node(&mut self, node_id: SceneNodeId) -> bool {
        let plan = match self.prepare_remove_node(node_id) {
            Some(p) => p,
            None => return false,
        };
        self.commit_remove_node(plan);
        true
    }

    /// Prepare removal: validate and snapshot the entire subtree in
    /// post-order (children before parent). Returns the plan or `None`
    /// when the node is not valid.
    pub(crate) fn prepare_remove_node(&self, node_id: SceneNodeId) -> Option<RemoveNodePlan> {
        if self.validate_node_ref(node_id).is_err() {
            return None;
        }
        let mut snapshots = Vec::new();
        self.collect_removal_snapshots(node_id, None, &mut snapshots);

        let root_replaced =
            self.root == Some(node_id) && snapshots.iter().any(|s| s.id == node_id);

        Some(RemoveNodePlan {
            snapshots,
            root_replaced,
        })
    }

    /// Commit a node removal plan. Infallible after successful prepare.
    pub(crate) fn commit_remove_node(&mut self, plan: RemoveNodePlan) {
        for snap in plan.snapshots.iter().rev() {
            // Remove from parent's children list.
            if let Some(parent_id) = snap.parent {
                if let Some(parent_entry) = self.nodes.get_mut(parent_id.slot as usize) {
                    if let Some((ref mut parent_node, _)) = parent_entry.node {
                        parent_node.children.retain(|c| *c != snap.id);
                    }
                }
            }

            // Clear the slot.
            if let Some(entry) = self.nodes.get_mut(snap.id.slot as usize) {
                if entry.generation == snap.id.generation {
                    // Remove from reverse index.
                    self.reverse_index.remove(&snap.record.persistent_id);
                    entry.node = None;
                    if bump_generation(&mut entry.generation) {
                        self.free_slots.push(snap.id.slot);
                    }
                }
            }
        }

        if plan.root_replaced {
            self.root = None;
        }
    }

    /// Collect subtree snapshots in post-order (children before parent).
    fn collect_removal_snapshots(
        &self,
        node_id: SceneNodeId,
        parent_id: Option<SceneNodeId>,
        snapshots: &mut Vec<SceneNodeRemovalSnapshot>,
    ) {
        let Some(entry) = self.nodes.get(node_id.slot as usize) else {
            return;
        };
        if entry.generation != node_id.generation {
            return;
        }
        let Some((ref node, ref record)) = entry.node else {
            return;
        };

        let children: Vec<SceneNodeId> = node.children.clone();
        for child_id in children {
            if self.is_valid_node_id(child_id) {
                self.collect_removal_snapshots(child_id, Some(node_id), snapshots);
            }
        }

        snapshots.push(SceneNodeRemovalSnapshot {
            id: node_id,
            node: node.clone(),
            record: record.clone(),
            parent: parent_id,
            parent_index: snapshots
                .iter()
                .position(|s| s.id == parent_id.unwrap_or(SceneNodeId::new(u32::MAX, 0)))
                .unwrap_or(usize::MAX),
        });
    }

    pub(crate) fn clone_subtree(&self, node_id: SceneNodeId) -> Option<RestorableSceneSubtree> {
        let entry = self.nodes.get(node_id.slot as usize)?;
        if entry.generation != node_id.generation {
            return None;
        }
        let (node, record) = entry.node.as_ref()?;
        let node = node.clone();
        let record = record.clone();
        let children = node
            .children
            .iter()
            .copied()
            .filter_map(|child| self.clone_subtree(child))
            .collect();
        Some(RestorableSceneSubtree {
            node,
            record,
            children,
        })
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
            let Some((ref node, _)) = entry.node else {
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
            let Some((light, _)) = entry.light else { continue };
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
        if let Some(ref mut mount) = self.bsp_mount {
            // Update PVS first (requires &mut mount.state).
            let cam_pos = self.camera.cam_pos;
            mount.state.update_pvs(cam_pos);

            submission.bsp_frame_values = crate::scene::render_submission::BspFrameValuesState {
                style_intensities: mount.state.frame_style_intensities,
                liquid_time: mount.state.frame_liquid_time,
                arena_id: mount.state.arena_id,
            };
            let bsp_lights = mount
                .state
                .select_frame_lights_for_camera(cam_pos, MAX_POINT_LIGHTS_GPU);
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
            if let Some((light, _)) = entry.light {
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
            if let Some((light, _)) = entry.light {
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
        if let Some(ref mount) = self.bsp_mount {
            let evidence_active = self.evidence_active();
            let all_visible = evidence_active && self.pending_evidence_data.as_ref()
                .map(|(_, _, vis)| matches!(vis, crate::api::bsp::BspEvidenceVisibility::AllVisible))
                .unwrap_or(false);
            self.collect_bsp_draw_items_from_mount(mount, &mut submission, frustum.as_ref(), evidence_active, all_visible);
        }
        // Phase 07: Clear the pending evidence request after submission build.
        #[cfg(feature = "bsp")]
        {
            self.pending_evidence_data = None;
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
    fn collect_bsp_draw_items_from_mount(
        &self,
        mount: &crate::api::bsp::PublishedBspMount,
        submission: &mut RenderSubmission,
        frustum: Option<&Frustum>,
        evidence_active: bool,
        all_visible: bool,
    ) {
        use crate::scene::bsp_visibility::{
            classify_bsp_visibility, mounted_visibility_decision, VisibilityDecision,
        };
        use crate::scene::render_submission::{
            BspBatchSemanticIdentity, BspEvidenceCollector, BspSubmissionFailure,
        };
        use crate::api::bsp::BspCanonicalDigest;

        let state = &mount.state;
        let diagnostics = classify_bsp_visibility(&state.mounted_batches, state);
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

        // Phase 07: Populate evidence collector if a request is pending.
        if evidence_active {
            let (corpus, req_id, visibility) = self.pending_evidence_data.as_ref().unwrap();
            let arena_id = mount.state.arena_id;
            let request = crate::api::bsp::BspEvidenceRequest {
                corpus_identity: corpus.clone(),
                request_identity: req_id.clone(),
                visibility: *visibility,
                key: crate::api::bsp::BspEvidenceRequestKey(0),
            };
            let mut collector = BspEvidenceCollector::new(&request, arena_id, self.pending_evidence_frame);

            // Derive atlas bytes from upload demand.
            collector.atlas_bytes = mount.state.mounted_batches.first()
                .map(|_| 0u64)
                .unwrap_or(0);

            // Populate neutral identities from canonical mounted batches.
            for (batch_index, mounted) in state.mounted_batches.iter().enumerate() {
                let batch = &mounted.render;
                let key = &batch.key;
                let digest = BspCanonicalDigest::compute(
                    key.render_class,
                    key.material_identity,
                    key.lightmap_page,
                    &key.style_ids,
                    key.model_index,
                    &batch.face_indices,
                );
                let identity = BspBatchSemanticIdentity {
                    batch_index,
                    canonical_digest: digest.0,
                    face_count: batch.face_indices.len() as u32,
                    source_faces: batch.face_indices.clone(),
                    model_index: key.model_index,
                    is_static: key.model_index == 0 && !batch.is_inline_model,
                };
                if identity.is_static {
                    collector.neutral_identities.push(identity.clone());
                    collector.mounted_identities.push(identity);
                } else {
                    collector.inline_batch_count += 1;
                    collector.inline_face_count += batch.face_indices.len() as u32;
                }
            }

            collector.pvs_eligible = diagnostics.pvs_eligible;
            collector.pvs_culled = diagnostics.pvs_culled;

            *submission.bsp_evidence_collector.borrow_mut() = Some(collector);
        }

        for (batch_index, mounted) in state.mounted_batches.iter().enumerate() {
            let batch = &mounted.render;
            let model_index = mounted.render.model_index;
            let is_static = model_index == 0 && !batch.is_inline_model;

            // Compute canonical digest for this batch (needed for evidence and identity).
            let canonical_digest = BspCanonicalDigest::compute(
                batch.key.render_class,
                batch.key.material_identity,
                batch.key.lightmap_page,
                &batch.key.style_ids,
                batch.key.model_index,
                &batch.face_indices,
            );

            // --- PVS Classification ---
            let pvs_decision = if all_visible && is_static {
                // All-visible override: treat static PVS-eligible batches as visible.
                VisibilityDecision::Visible
            } else {
                mounted_visibility_decision(mounted, state)
            };

            match pvs_decision {
                VisibilityDecision::PvsCulled => {
                    if evidence_active && is_static {
                        if let Some(ref mut collector) = *submission.bsp_evidence_collector.borrow_mut() {
                            collector.cull_decisions.push((batch_index, crate::scene::render_submission::BspCullReason::PvsCulled));
                        }
                    }
                    continue;
                }
                VisibilityDecision::Visible => {}
            }

            // --- Inline model frustum cull ---
            let batch_transform = if batch.is_inline_model {
                state
                    .inline_model_transforms
                    .get(&batch.model_index)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY)
            } else {
                Mat4::IDENTITY
            };

            if batch.is_inline_model {
                if let Some((world_min, world_max)) = state
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
                // Evidence: record failure for static batches.
                if evidence_active && is_static {
                    if let Some(ref mut collector) = *submission.bsp_evidence_collector.borrow_mut() {
                        collector.failures.push(crate::api::bsp::BspEvidenceFailure::MissingDescriptor { batch_index });
                    }
                }
                continue;
            }

            let item = crate::scene::render_submission::BspFrameDrawItem {
                mesh_id,
                bsp_material_id,
                transform: batch_transform,
                batch_index,
                source_face_first,
                source_face_count,
                pipeline_class: None,
                model_index,
                canonical_digest: canonical_digest.0,
            };

            // Evidence: record submitted identity.
            if evidence_active && is_static {
                if let Some(ref mut collector) = *submission.bsp_evidence_collector.borrow_mut() {
                    collector.submitted_identities.push(BspBatchSemanticIdentity {
                        batch_index,
                        canonical_digest: canonical_digest.0,
                        face_count: source_face_count,
                        source_faces: mounted.render.face_indices.clone(),
                        model_index,
                        is_static: true,
                    });
                }
            }

            submission.bsp_draw_items.push(item);
            submission.culling_stats.submitted_draw_items += 1;
        }
    }

    fn restore_subtree_with_parent(
        &mut self,
        snapshot: RestorableSceneSubtree,
        parent: Option<SceneNodeId>,
    ) -> SceneNodeId {
        let RestorableSceneSubtree {
            mut node,
            record,
            children,
        } = snapshot;
        node.parent = parent;
        node.children.clear();
        node.dirty = true;
        let restored = self.add_node_with_record(parent, node, record);

        for child in children {
            self.restore_subtree_with_parent(child, Some(restored));
        }

        restored
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
        let record = ObjectRecord::for_new_point_light(None, None);
        let plan = self.prepare_create_point_light(light, record);
        self.commit_create_point_light(plan)
    }

    pub(crate) fn add_point_light_with_record(
        &mut self,
        light: PointLight,
        record: ObjectRecord,
    ) -> PointLightId {
        let plan = self.prepare_create_point_light(light, record);
        self.commit_create_point_light(plan)
    }

    pub(crate) fn prepare_create_point_light(
        &mut self,
        light: PointLight,
        record: ObjectRecord,
    ) -> CreatePointLightPlan {
        let (slot, generation, is_new_slot) =
            if let Some(free_slot) = self.free_point_light_slots.pop() {
                let entry = &mut self.point_lights[free_slot as usize];
                debug_assert!(
                    entry.light.is_none(),
                    "free slot list contained a live point light"
                );
                (free_slot, entry.generation, false)
            } else {
                let slot = self.point_lights.len() as u32;
                (slot, 0, true)
            };

        CreatePointLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        }
    }

    pub(crate) fn commit_create_point_light(
        &mut self,
        plan: CreatePointLightPlan,
    ) -> PointLightId {
        let CreatePointLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        } = plan;

        let persistent_id = record.persistent_id.clone();
        let id = PointLightId { slot, generation };
        self.reverse_index
            .insert(persistent_id, ObjectHandle::PointLight(id));

        if is_new_slot {
            self.point_lights.push(PointLightEntry {
                generation,
                light: Some((light, record)),
            });
        } else {
            let entry = &mut self.point_lights[slot as usize];
            entry.light = Some((light, record));
        }

        id
    }

    pub(crate) fn update_point_light(&mut self, id: PointLightId, light: PointLight) -> bool {
        let Some(entry) = self.point_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        let record = entry.light.as_ref().map(|(_, r)| r.clone());
        if let Some(record) = record {
            entry.light = Some((light, record));
            return true;
        }
        false
    }

    /// Get a point light's record.
    pub(crate) fn get_point_light_record(&self, id: PointLightId) -> Option<&ObjectRecord> {
        let entry = self.point_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(_, r)| r)
    }

    /// Get a mutable point light record.
    pub(crate) fn get_point_light_record_mut(
        &mut self,
        id: PointLightId,
    ) -> Option<&mut ObjectRecord> {
        let entry = self.point_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut().map(|(_, r)| r)
    }

    pub(crate) fn remove_point_light(&mut self, id: PointLightId) -> bool {
        let plan = match self.prepare_remove_point_light(id) {
            Some(p) => p,
            None => return false,
        };
        self.commit_remove_point_light(plan);
        true
    }

    pub(crate) fn prepare_remove_point_light(
        &self,
        id: PointLightId,
    ) -> Option<RemovePointLightPlan> {
        let entry = self.point_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (light, record) = entry.light.as_ref()?;
        Some(RemovePointLightPlan {
            id,
            light: *light,
            record: record.clone(),
        })
    }

    pub(crate) fn commit_remove_point_light(&mut self, plan: RemovePointLightPlan) {
        let Some(entry) = self.point_lights.get_mut(plan.id.slot as usize) else {
            return;
        };
        if entry.generation != plan.id.generation || entry.light.is_none() {
            return;
        }
        self.reverse_index.remove(&plan.record.persistent_id);
        entry.light = None;
        if bump_generation(&mut entry.generation) {
            self.free_point_light_slots.push(plan.id.slot);
        }
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
        let record = ObjectRecord::for_new_directional_light(None, None);
        let plan = self.prepare_create_directional_light(light, record);
        self.commit_create_directional_light(plan)
    }

    pub(crate) fn add_directional_light_with_record(
        &mut self,
        light: DirectionalLight,
        record: ObjectRecord,
    ) -> DirectionalLightId {
        let plan = self.prepare_create_directional_light(light, record);
        self.commit_create_directional_light(plan)
    }

    pub(crate) fn prepare_create_directional_light(
        &mut self,
        light: DirectionalLight,
        record: ObjectRecord,
    ) -> CreateDirectionalLightPlan {
        let (slot, generation, is_new_slot) =
            if let Some(free_slot) = self.free_directional_light_slots.pop() {
                let entry = &mut self.directional_lights[free_slot as usize];
                debug_assert!(
                    entry.light.is_none(),
                    "free slot list contained a live directional light"
                );
                (free_slot, entry.generation, false)
            } else {
                let slot = self.directional_lights.len() as u32;
                (slot, 0, true)
            };

        CreateDirectionalLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        }
    }

    pub(crate) fn commit_create_directional_light(
        &mut self,
        plan: CreateDirectionalLightPlan,
    ) -> DirectionalLightId {
        let CreateDirectionalLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        } = plan;

        let persistent_id = record.persistent_id.clone();
        let id = DirectionalLightId { slot, generation };
        self.reverse_index
            .insert(persistent_id, ObjectHandle::DirectionalLight(id));

        if is_new_slot {
            self.directional_lights.push(DirectionalLightEntry {
                generation,
                light: Some((light, record)),
            });
        } else {
            let entry = &mut self.directional_lights[slot as usize];
            entry.light = Some((light, record));
        }

        id
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
        let record = entry.light.as_ref().map(|(_, r)| r.clone());
        if let Some(record) = record {
            entry.light = Some((light, record));
            return true;
        }
        false
    }

    /// Get a directional light's record.
    pub fn get_directional_light_record(
        &self,
        id: DirectionalLightId,
    ) -> Option<&ObjectRecord> {
        let entry = self.directional_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(_, r)| r)
    }

    /// Get a mutable directional light record.
    pub(crate) fn get_directional_light_record_mut(
        &mut self,
        id: DirectionalLightId,
    ) -> Option<&mut ObjectRecord> {
        let entry = self.directional_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut().map(|(_, r)| r)
    }

    pub(crate) fn remove_directional_light(&mut self, id: DirectionalLightId) -> bool {
        let plan = match self.prepare_remove_directional_light(id) {
            Some(p) => p,
            None => return false,
        };
        self.commit_remove_directional_light(plan);
        true
    }

    pub(crate) fn prepare_remove_directional_light(
        &self,
        id: DirectionalLightId,
    ) -> Option<RemoveDirectionalLightPlan> {
        let entry = self.directional_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (light, record) = entry.light.as_ref()?;
        Some(RemoveDirectionalLightPlan {
            id,
            light: *light,
            record: record.clone(),
        })
    }

    pub(crate) fn commit_remove_directional_light(&mut self, plan: RemoveDirectionalLightPlan) {
        let Some(entry) = self.directional_lights.get_mut(plan.id.slot as usize) else {
            return;
        };
        if entry.generation != plan.id.generation || entry.light.is_none() {
            return;
        }
        self.reverse_index.remove(&plan.record.persistent_id);
        entry.light = None;
        if bump_generation(&mut entry.generation) {
            self.free_directional_light_slots.push(plan.id.slot);
        }
        if self.shadow_casting_directional == Some(plan.id) {
            self.shadow_casting_directional = None;
        }
    }

    /// Returns the active directional light (the public facade enforces one).
    pub(crate) fn get_active_directional_light(&self) -> Option<DirectionalLight> {
        self.directional_lights
            .iter()
            .find_map(|entry| entry.light.as_ref().map(|(l, _)| *l))
    }

    /// Returns all active directional lights.
    pub(crate) fn get_active_directional_lights(&self) -> Vec<DirectionalLight> {
        self.directional_lights
            .iter()
            .filter_map(|entry| entry.light.as_ref().map(|(l, _)| *l))
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

    // ── Reverse-index and ObjectId conversion ─────────────────────────

    /// Convert a node handle to an ObjectId.
    pub(crate) fn object_id_for_node(&self, id: SceneNodeId) -> Option<ObjectId> {
        let entry = self.nodes.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (_node, _record) = entry.node.as_ref()?;
        Some(ObjectId::from_parts(
            self.provenance,
            ObjectKind::Node,
            id.slot,
            id.generation,
        ))
    }

    /// Convert a point light handle to an ObjectId.
    pub(crate) fn object_id_for_point_light(&self, id: PointLightId) -> Option<ObjectId> {
        let entry = self.point_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (_light, _record) = entry.light.as_ref()?;
        Some(ObjectId::from_parts(
            self.provenance,
            ObjectKind::PointLight,
            id.slot,
            id.generation,
        ))
    }

    /// Convert a directional light handle to an ObjectId.
    pub(crate) fn object_id_for_directional_light(
        &self,
        id: DirectionalLightId,
    ) -> Option<ObjectId> {
        let entry = self.directional_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (_light, _record) = entry.light.as_ref()?;
        Some(ObjectId::from_parts(
            self.provenance,
            ObjectKind::DirectionalLight,
            id.slot,
            id.generation,
        ))
    }

    /// Convert a spot light handle to an ObjectId.
    pub(crate) fn object_id_for_spot_light(&self, id: SpotLightId) -> Option<ObjectId> {
        let entry = self.spot_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (_light, _record) = entry.light.as_ref()?;
        Some(ObjectId::from_parts(
            self.provenance,
            ObjectKind::SpotLight,
            id.slot,
            id.generation,
        ))
    }

    /// Resolve an ObjectId back to a typed handle.
    /// Validation order: provenance → kind → slot bounds → generation → occupancy.
    pub(crate) fn resolve_object(&self, id: ObjectId) -> Option<ObjectHandle> {
        if id.provenance() != self.provenance {
            return None;
        }
        match id.kind() {
            ObjectKind::Node => {
                let node_id = SceneNodeId::new(id.slot(), id.generation());
                if self.validate_node_ref(node_id).is_ok() {
                    Some(ObjectHandle::Node(node_id))
                } else {
                    None
                }
            }
            ObjectKind::PointLight => {
                let pl_id = PointLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                };
                if self.validate_point_light_ref(pl_id).is_ok() {
                    Some(ObjectHandle::PointLight(pl_id))
                } else {
                    None
                }
            }
            ObjectKind::DirectionalLight => {
                let dl_id = DirectionalLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                };
                if self.validate_directional_light_ref(dl_id).is_ok() {
                    Some(ObjectHandle::DirectionalLight(dl_id))
                } else {
                    None
                }
            }
            ObjectKind::SpotLight => {
                let sl_id = SpotLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                };
                if self.validate_spot_light_ref(sl_id).is_ok() {
                    Some(ObjectHandle::SpotLight(sl_id))
                } else {
                    None
                }
            }
        }
    }

    /// Like [`resolve_object`] but distinguishes WrongScene from other failures.
    pub(crate) fn resolve_object_with_error(
        &self,
        id: ObjectId,
    ) -> Result<ObjectHandle, crate::object::identity::ObjectError> {
        use crate::object::identity::ObjectError;
        if id.provenance() != self.provenance {
            return Err(ObjectError::WrongScene {
                object: id,
                expected_scene: format!("{:?}", self.provenance),
            });
        }
        self.resolve_object(id)
            .ok_or(ObjectError::InvalidObject(id))
    }

    /// Look up an ObjectId by persistent SceneObjectId.
    pub(crate) fn find_object_by_persistent_id(
        &self,
        persistent_id: &SceneObjectId,
    ) -> Option<ObjectId> {
        let handle = self.reverse_index.get(persistent_id)?;
        match *handle {
            ObjectHandle::Node(nid) => self.object_id_for_node(nid),
            ObjectHandle::PointLight(pl) => self.object_id_for_point_light(pl),
            ObjectHandle::DirectionalLight(dl) => self.object_id_for_directional_light(dl),
            ObjectHandle::SpotLight(sl) => self.object_id_for_spot_light(sl),
        }
    }

    /// Check all object invariants.
    pub(crate) fn audit_object_invariants_impl(&self) -> Result<(), String> {
        use std::collections::HashSet;

        let mut seen_persistent = HashSet::new();

        // Nodes: every occupied slot has record, every record has valid persistent ID.
        for (slot, entry) in self.nodes.iter().enumerate() {
            if let Some((ref node, ref record)) = entry.node {
                if !seen_persistent.insert(record.persistent_id.clone()) {
                    return Err(format!(
                        "duplicate persistent ID {} in node slot {}",
                        record.persistent_id, slot
                    ));
                }
                // Check reverse index contains this persistent ID.
                match self.reverse_index.get(&record.persistent_id) {
                    Some(ObjectHandle::Node(nid)) => {
                        if nid.slot != slot as u32 || nid.generation != entry.generation {
                            return Err(format!(
                                "reverse index mismatch for node slot {}: expected ({}, {}), got ({}, {})",
                                slot, slot, entry.generation, nid.slot, nid.generation
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "reverse index missing or wrong kind for node slot {}: {:?}",
                            slot, other
                        ));
                    }
                }
                // Hierarchy: parent must exist and reference this node.
                if let Some(parent_id) = node.parent {
                    if self.validate_node_ref(parent_id).is_err() {
                        return Err(format!(
                            "node slot {} has invalid parent {:?}",
                            slot, parent_id
                        ));
                    }
                }
            } else {
                // Vacant slot: must not be in reverse index (by slot).
                // (We can skip because reverse index is by persistent ID, not slot.)
            }
        }

        // Point lights.
        for (slot, entry) in self.point_lights.iter().enumerate() {
            if let Some((ref _light, ref record)) = entry.light {
                if !seen_persistent.insert(record.persistent_id.clone()) {
                    return Err(format!(
                        "duplicate persistent ID {} in point light slot {}",
                        record.persistent_id, slot
                    ));
                }
                match self.reverse_index.get(&record.persistent_id) {
                    Some(ObjectHandle::PointLight(pl)) => {
                        if pl.slot != slot as u32 || pl.generation != entry.generation {
                            return Err(format!(
                                "reverse index mismatch for point light slot {}",
                                slot
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "reverse index missing or wrong kind for point light slot {}: {:?}",
                            slot, other
                        ));
                    }
                }
            }
        }

        // Directional lights.
        for (slot, entry) in self.directional_lights.iter().enumerate() {
            if let Some((ref _light, ref record)) = entry.light {
                if !seen_persistent.insert(record.persistent_id.clone()) {
                    return Err(format!(
                        "duplicate persistent ID {} in directional light slot {}",
                        record.persistent_id, slot
                    ));
                }
                match self.reverse_index.get(&record.persistent_id) {
                    Some(ObjectHandle::DirectionalLight(dl)) => {
                        if dl.slot != slot as u32 || dl.generation != entry.generation {
                            return Err(format!(
                                "reverse index mismatch for directional light slot {}",
                                slot
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "reverse index missing or wrong kind for directional light slot {}: {:?}",
                            slot, other
                        ));
                    }
                }
                // Shadow-owner validity.
                if let Some(shadow_id) = self.shadow_casting_directional {
                    if shadow_id.slot == slot as u32
                        && shadow_id.generation != entry.generation
                    {
                        return Err(format!(
                            "shadow-casting directional light slot {} has stale generation",
                            slot
                        ));
                    }
                }
            }
        }

        // Spot lights.
        for (slot, entry) in self.spot_lights.iter().enumerate() {
            if let Some((ref _light, ref record)) = entry.light {
                if !seen_persistent.insert(record.persistent_id.clone()) {
                    return Err(format!(
                        "duplicate persistent ID {} in spot light slot {}",
                        record.persistent_id, slot
                    ));
                }
                match self.reverse_index.get(&record.persistent_id) {
                    Some(ObjectHandle::SpotLight(sl)) => {
                        if sl.slot != slot as u32 || sl.generation != entry.generation {
                            return Err(format!(
                                "reverse index mismatch for spot light slot {}",
                                slot
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "reverse index missing or wrong kind for spot light slot {}: {:?}",
                            slot, other
                        ));
                    }
                }
            }
        }

        Ok(())
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
        let record = ObjectRecord::for_new_spot_light(None, None);
        let plan = self.prepare_create_spot_light(light, record);
        self.commit_create_spot_light(plan)
    }

    pub(crate) fn add_spot_light_with_record(
        &mut self,
        light: SpotLight,
        record: ObjectRecord,
    ) -> SpotLightId {
        let plan = self.prepare_create_spot_light(light, record);
        self.commit_create_spot_light(plan)
    }

    pub(crate) fn prepare_create_spot_light(
        &mut self,
        light: SpotLight,
        record: ObjectRecord,
    ) -> CreateSpotLightPlan {
        let (slot, generation, is_new_slot) =
            if let Some(free_slot) = self.free_spot_light_slots.pop() {
                let entry = &mut self.spot_lights[free_slot as usize];
                (free_slot, entry.generation, false)
            } else {
                let slot = self.spot_lights.len() as u32;
                (slot, 0, true)
            };

        CreateSpotLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        }
    }

    pub(crate) fn commit_create_spot_light(
        &mut self,
        plan: CreateSpotLightPlan,
    ) -> SpotLightId {
        let CreateSpotLightPlan {
            slot,
            generation,
            light,
            record,
            is_new_slot,
        } = plan;

        let persistent_id = record.persistent_id.clone();
        let id = SpotLightId { slot, generation };
        self.reverse_index
            .insert(persistent_id, ObjectHandle::SpotLight(id));

        if is_new_slot {
            self.spot_lights.push(SpotLightEntry {
                generation,
                light: Some((light, record)),
            });
        } else {
            let entry = &mut self.spot_lights[slot as usize];
            entry.light = Some((light, record));
        }

        id
    }

    pub(crate) fn update_spot_light(&mut self, id: SpotLightId, light: SpotLight) -> bool {
        let Some(entry) = self.spot_lights.get_mut(id.slot as usize) else {
            return false;
        };
        if entry.generation != id.generation || entry.light.is_none() {
            return false;
        }
        let record = entry.light.as_ref().map(|(_, r)| r.clone());
        if let Some(record) = record {
            entry.light = Some((light, record));
            return true;
        }
        false
    }

    /// Get a spot light's record.
    pub(crate) fn get_spot_light_record(&self, id: SpotLightId) -> Option<&ObjectRecord> {
        let entry = self.spot_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(_, r)| r)
    }

    /// Get a mutable spot light record.
    pub(crate) fn get_spot_light_record_mut(
        &mut self,
        id: SpotLightId,
    ) -> Option<&mut ObjectRecord> {
        let entry = self.spot_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut().map(|(_, r)| r)
    }

    /// Direct access to point light slot for transform reads.
    pub fn point_light_entry(
        &self,
        id: PointLightId,
    ) -> Option<&PointLight> {
        let entry = self.point_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(l, _)| l)
    }

    /// Direct mutable access to point light slot for transform writes.
    pub(crate) fn point_light_entry_mut(
        &mut self,
        id: PointLightId,
    ) -> Option<&mut (PointLight, ObjectRecord)> {
        let entry = self.point_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut()
    }

    /// Direct access to directional light slot.
    pub fn directional_light_entry(
        &self,
        id: DirectionalLightId,
    ) -> Option<&DirectionalLight> {
        let entry = self.directional_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(l, _)| l)
    }

    /// Direct mutable access to directional light slot.
    pub(crate) fn directional_light_entry_mut(
        &mut self,
        id: DirectionalLightId,
    ) -> Option<&mut (DirectionalLight, ObjectRecord)> {
        let entry = self.directional_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut()
    }

    /// Direct access to spot light slot.
    pub fn spot_light_entry(
        &self,
        id: SpotLightId,
    ) -> Option<&SpotLight> {
        let entry = self.spot_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_ref().map(|(l, _)| l)
    }

    /// Direct mutable access to spot light slot.
    pub(crate) fn spot_light_entry_mut(
        &mut self,
        id: SpotLightId,
    ) -> Option<&mut (SpotLight, ObjectRecord)> {
        let entry = self.spot_lights.get_mut(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.light.as_mut()
    }



    pub(crate) fn remove_spot_light(&mut self, id: SpotLightId) -> bool {
        let plan = match self.prepare_remove_spot_light(id) {
            Some(p) => p,
            None => return false,
        };
        self.commit_remove_spot_light(plan);
        true
    }

    pub(crate) fn prepare_remove_spot_light(
        &self,
        id: SpotLightId,
    ) -> Option<RemoveSpotLightPlan> {
        let entry = self.spot_lights.get(id.slot as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let (light, record) = entry.light.as_ref()?;
        Some(RemoveSpotLightPlan {
            id,
            light: *light,
            record: record.clone(),
        })
    }

    pub(crate) fn commit_remove_spot_light(&mut self, plan: RemoveSpotLightPlan) {
        let Some(entry) = self.spot_lights.get_mut(plan.id.slot as usize) else {
            return;
        };
        if entry.generation != plan.id.generation || entry.light.is_none() {
            return;
        }
        self.reverse_index.remove(&plan.record.persistent_id);
        entry.light = None;
        if bump_generation(&mut entry.generation) {
            self.free_spot_light_slots.push(plan.id.slot);
        }
    }

    pub(crate) fn get_active_spot_lights(&self) -> Vec<SpotLight> {
        self.spot_lights
            .iter()
            .filter_map(|entry| entry.light.as_ref().map(|(l, _)| *l))
            .collect()
    }

    // ── BSP mount management ────────────────────────────────────────

    /// Set the published BSP mount for PVS-aware culling and light selection.
    ///
    /// The lease-bearing [`PublishedBspMount`] owns every GPU resource.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_mount(
        &mut self,
        mount: crate::api::bsp::PublishedBspMount,
    ) {
        self.bsp_mount = Some(mount);
    }

    /// Clear the BSP mount, disabling PVS culling and BSP light selection.
    ///
    /// Deprecated: this drops the resource lease. Use [`Self::retire_bsp_mount`]
    /// to obtain a [`DetachedBspMount`] for renderer retirement.
    #[cfg(feature = "bsp")]
    #[deprecated(since = "0.14.0", note = "use retire_bsp_mount() to preserve the lease")]
    pub(crate) fn clear_bsp_mount(&mut self) {
        let _ = self.retire_bsp_mount();
    }

    /// Detach the active BSP mount and return the lease-bearing
    /// [`DetachedBspMount`] for renderer retirement.
    ///
    /// The returned mount retains the full resource lease. The caller must
    /// pass it to the renderer's retirement path for fence-aware GPU teardown.
    #[cfg(feature = "bsp")]
    pub(crate) fn retire_bsp_mount(
        &mut self,
    ) -> Option<crate::api::bsp::DetachedBspMount> {
        let published = self.bsp_mount.take()?;
        Some(crate::api::bsp::DetachedBspMount::from_published(published))
    }

    /// Return whether a BSP mount is currently published.
    #[cfg(feature = "bsp")]
    pub(crate) fn has_bsp_mount(&self) -> bool {
        self.bsp_mount.is_some()
    }

    /// Phase 07: Store a pending evidence request for population during the next
    /// [`build_submission`] call.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_evidence_request(
        &mut self,
        corpus_identity: String,
        request_identity: String,
        visibility: crate::api::bsp::BspEvidenceVisibility,
        frame_number: u32,
    ) {
        self.pending_evidence_data = Some((corpus_identity, request_identity, visibility));
        self.pending_evidence_frame = frame_number;
    }

    /// Returns true if there's active evidence data (non-empty corpus).
    #[cfg(feature = "bsp")]
    fn evidence_active(&self) -> bool {
        self.pending_evidence_data.as_ref()
            .map(|(corpus, _, _)| !corpus.is_empty())
            .unwrap_or(false)
    }

    /// Update BSP PVS for the current camera position.
    /// Called before `build_submission` when a BSP mount is active.
    #[cfg(feature = "bsp")]
    pub(crate) fn update_bsp_pvs(&mut self) {
        let cam_pos = self.camera.cam_pos;
        if let Some(ref mut mount) = self.bsp_mount {
            mount.state.update_pvs(cam_pos);
        }
    }

    /// Set per-frame BSP frame values (style intensities, liquid time).
    ///
    /// These are uploaded to the BSP frame-values UBO each frame.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_frame_values(&mut self, style_intensities: [f32; 64], liquid_time: f32) {
        if let Some(ref mut mount) = self.bsp_mount {
            mount.state.frame_style_intensities = style_intensities;
            mount.state.frame_liquid_time = liquid_time;
        }
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
        if let Some(ref mut mount) = self.bsp_mount {
            mount.state.inline_model_transforms = transforms;
        }
    }

    /// Set per-model world-space bounds for inline model culling.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_inline_model_bounds(
        &mut self,
        bounds: std::collections::HashMap<u32, (glam::Vec3, glam::Vec3)>,
    ) {
        if let Some(ref mut mount) = self.bsp_mount {
            mount.state.inline_model_bounds = bounds;
        }
    }

    // ── Component accessors ──────────────────────────────────────────

    /// Attach a prevalidated component envelope to the node's record.
    pub(crate) fn attach_component(
        &mut self,
        node_id: SceneNodeId,
        envelope: crate::object::component::ComponentEnvelope,
    ) -> Result<(), crate::object::component::ComponentError> {
        let record = self
            .get_node_record_mut(node_id)
            .ok_or_else(|| {
                use crate::object::component::ComponentError;
                ComponentError::InvalidEnvelope("node not found".into())
            })?;
        record.component_store.attach(envelope)
    }

    /// Remove a component instance from the node's record by key and instance ID.
    pub(crate) fn remove_component(
        &mut self,
        node_id: SceneNodeId,
        key: &crate::object::component::ComponentKey,
        instance_id: &crate::object::component::ComponentInstanceId,
    ) -> Option<crate::object::component::ComponentEnvelope> {
        self.get_node_record_mut(node_id)
            .and_then(|record| record.component_store.remove(key, instance_id))
    }

    /// Enumerate all component envelopes for a node.
    pub(crate) fn component_envelopes(
        &self,
        node_id: SceneNodeId,
    ) -> Option<impl Iterator<Item = &crate::object::component::ComponentEnvelope>> {
        self.get_node_record(node_id)
            .map(|record| record.component_store.envelopes())
    }

    /// Enumerate component envelopes of a given type for a node.
    pub(crate) fn component_envelopes_by_key(
        &self,
        node_id: SceneNodeId,
        key: &crate::object::component::ComponentKey,
    ) -> Option<impl Iterator<Item = &crate::object::component::ComponentEnvelope>> {
        self.get_node_record(node_id)
            .map(|record| record.component_store.envelopes_by_key(key))
    }

    /// Get a typed component instance by key and instance ID.
    pub(crate) fn component_downcast<T: 'static>(
        &self,
        node_id: SceneNodeId,
        key: &crate::object::component::ComponentKey,
        instance_id: &crate::object::component::ComponentInstanceId,
    ) -> Result<&T, crate::object::component::ComponentError> {
        let record = self
            .get_node_record(node_id)
            .ok_or_else(|| {
                crate::object::component::ComponentError::InvalidEnvelope(
                    "node not found".into(),
                )
            })?;
        record.component_store.downcast::<T>(key, instance_id)
    }

    /// Iterate typed hydrated instances of a given component type on a node.
    pub(crate) fn component_typed_instances<T: Any + Send + Sync>(
        &self,
        node_id: SceneNodeId,
        key: &crate::object::component::ComponentKey,
    ) -> Option<Vec<(crate::object::component::ComponentEnvelope, Arc<T>)>> {
        self.get_node_record(node_id)
            .map(|record| record.component_store.typed_instances_owned::<T>(key))
    }

    /// Get a mutable reference to the component store for a node.
    pub(crate) fn component_store_mut(
        &mut self,
        node_id: SceneNodeId,
    ) -> Option<&mut crate::object::component::ComponentStore> {
        self.get_node_record_mut(node_id)
            .map(|record| &mut record.component_store)
    }

    // ── Object enumeration ───────────────────────────────────────────

    /// Return all occupied objects as (ObjectId, ObjectKind, persistent_id, stable_id, name, tags).
    pub(crate) fn all_objects(&self) -> Vec<ObjectId> {
        let mut ids: Vec<ObjectId> = Vec::new();

        for (slot, entry) in self.nodes.iter().enumerate() {
            if entry.node.is_some() {
                ids.push(ObjectId::from_parts(
                    self.provenance,
                    ObjectKind::Node,
                    slot as u32,
                    entry.generation,
                ));
            }
        }
        for (slot, entry) in self.point_lights.iter().enumerate() {
            if entry.light.is_some() {
                ids.push(ObjectId::from_parts(
                    self.provenance,
                    ObjectKind::PointLight,
                    slot as u32,
                    entry.generation,
                ));
            }
        }
        for (slot, entry) in self.directional_lights.iter().enumerate() {
            if entry.light.is_some() {
                ids.push(ObjectId::from_parts(
                    self.provenance,
                    ObjectKind::DirectionalLight,
                    slot as u32,
                    entry.generation,
                ));
            }
        }
        for (slot, entry) in self.spot_lights.iter().enumerate() {
            if entry.light.is_some() {
                ids.push(ObjectId::from_parts(
                    self.provenance,
                    ObjectKind::SpotLight,
                    slot as u32,
                    entry.generation,
                ));
            }
        }
        // Stable sort by kind, then persistent ID
        ids.sort_by(|a, b| {
            a.kind()
                .cmp(&b.kind())
                .then_with(|| a.slot().cmp(&b.slot()))
                .then_with(|| a.generation().cmp(&b.generation()))
        });
        ids
    }

    /// Return node name (or stable ID fallback).
    pub(crate) fn object_name(&self, id: ObjectId) -> Option<String> {
        match id.kind() {
            ObjectKind::Node => {
                let node = self.get_node(SceneNodeId::new(id.slot(), id.generation()))?;
                Some(if node.name.is_empty() {
                    node.stable_id.clone().unwrap_or_else(|| format!("Node {}", id.slot()))
                } else {
                    node.name.clone()
                })
            }
            ObjectKind::PointLight => {
                self.get_point_light_record(PointLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| {
                    r.stable_id
                        .clone()
                        .unwrap_or_else(|| format!("PointLight {}", id.slot()))
                })
            }
            ObjectKind::DirectionalLight => {
                self.get_directional_light_record(DirectionalLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| {
                    r.stable_id
                        .clone()
                        .unwrap_or_else(|| format!("DirectionalLight {}", id.slot()))
                })
            }
            ObjectKind::SpotLight => {
                self.get_spot_light_record(SpotLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| {
                    r.stable_id
                        .clone()
                        .unwrap_or_else(|| format!("SpotLight {}", id.slot()))
                })
            }
        }
    }

    /// Return the persistent ID for an object.
    pub(crate) fn object_persistent_id(&self, id: ObjectId) -> Option<SceneObjectId> {
        match id.kind() {
            ObjectKind::Node => {
                self.get_node_record(SceneNodeId::new(id.slot(), id.generation()))
                    .map(|r| r.persistent_id.clone())
            }
            ObjectKind::PointLight => {
                self.get_point_light_record(PointLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| r.persistent_id.clone())
            }
            ObjectKind::DirectionalLight => {
                self.get_directional_light_record(DirectionalLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| r.persistent_id.clone())
            }
            ObjectKind::SpotLight => {
                self.get_spot_light_record(SpotLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| r.persistent_id.clone())
            }
        }
    }

    /// Return the component store for any object (by immutable reference).
    pub(crate) fn object_component_store(
        &self,
        id: ObjectId,
    ) -> Option<&crate::object::component::ComponentStore> {
        match id.kind() {
            ObjectKind::Node => self
                .get_node_record(SceneNodeId::new(id.slot(), id.generation()))
                .map(|r| &r.component_store),
            ObjectKind::PointLight => {
                self.get_point_light_record(PointLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &r.component_store)
            }
            ObjectKind::DirectionalLight => {
                self.get_directional_light_record(DirectionalLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &r.component_store)
            }
            ObjectKind::SpotLight => {
                self.get_spot_light_record(SpotLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &r.component_store)
            }
        }
    }

    /// Return the component store for any object (by mutable reference).
    pub(crate) fn object_component_store_mut(
        &mut self,
        id: ObjectId,
    ) -> Option<&mut crate::object::component::ComponentStore> {
        match id.kind() {
            ObjectKind::Node => self
                .get_node_record_mut(SceneNodeId::new(id.slot(), id.generation()))
                .map(|r| &mut r.component_store),
            ObjectKind::PointLight => {
                self.get_point_light_record_mut(PointLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &mut r.component_store)
            }
            ObjectKind::DirectionalLight => {
                self.get_directional_light_record_mut(DirectionalLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &mut r.component_store)
            }
            ObjectKind::SpotLight => {
                self.get_spot_light_record_mut(SpotLightId {
                    slot: id.slot(),
                    generation: id.generation(),
                })
                .map(|r| &mut r.component_store)
            }
        }
    }

    /// Return object visibility state.
    pub(crate) fn object_visible(&self, id: ObjectId) -> Option<bool> {
        match id.kind() {
            ObjectKind::Node => self
                .get_node_record(SceneNodeId::new(id.slot(), id.generation()))
                .and_then(|r| r.visibility.as_ref())
                .map(|v| v.visible),
            _ => Some(true), // Lights are always "visible"
        }
    }

    /// Return the layer name for an object (nodes only).
    pub(crate) fn object_layer(&self, id: ObjectId) -> Option<String> {
        match id.kind() {
            ObjectKind::Node => self
                .get_node_record(SceneNodeId::new(id.slot(), id.generation()))
                .and_then(|r| r.visibility.as_ref())
                .map(|v| v.layer.clone()),
            _ => None,
        }
    }

    // ── Subtree removal with light detachment ───────────────────────

    /// Prepare removal of a node subtree, collecting surviving grouped lights
    /// whose group parent lies in the removed set.
    pub(crate) fn prepare_remove_node_subtree(
        &self,
        node_id: SceneNodeId,
    ) -> Result<
        (
            RestorableSceneSubtree,
            Vec<crate::scene::object_store::DetachedLightSnapshot>,
        ),
        String,
    > {
        // Validate
        self.validate_node_ref(node_id).map_err(|e| format!("{e:?}"))?;

        // Clone subtree
        let subtree = self
            .clone_subtree(node_id)
            .ok_or_else(|| "failed to clone subtree".to_string())?;

        // Collect persistent IDs of all nodes in the subtree
        let removed_persistent_ids = self.collect_subtree_persistent_ids(&subtree);

        // Find grouped lights whose group parent is in the removed set
        let detached_lights = self.collect_grouped_lights_for_removal(&removed_persistent_ids);

        Ok((subtree, detached_lights))
    }

    /// Commit subtree removal, detaching grouped lights before node removal.
    pub(crate) fn commit_remove_node_subtree(
        &mut self,
        node_id: SceneNodeId,
        detached_lights: &[crate::scene::object_store::DetachedLightSnapshot],
    ) {
        // Detach lights from their group parents
        for dl in detached_lights {
            match dl.kind {
                ObjectKind::PointLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::PointLight(pl_id) = *handle {
                            if let Some(record) = self.get_point_light_record_mut(pl_id) {
                                record.light_group_parent = None;
                            }
                        }
                    }
                }
                ObjectKind::DirectionalLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::DirectionalLight(dl_id) = *handle {
                            if let Some(record) = self.get_directional_light_record_mut(dl_id) {
                                record.light_group_parent = None;
                            }
                        }
                    }
                }
                ObjectKind::SpotLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::SpotLight(sl_id) = *handle {
                            if let Some(record) = self.get_spot_light_record_mut(sl_id) {
                                record.light_group_parent = None;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Remove the node subtree
        self.remove_node(node_id);
    }

    /// Collect persistent IDs of all nodes in a subtree.
    fn collect_subtree_persistent_ids(
        &self,
        subtree: &RestorableSceneSubtree,
    ) -> Vec<SceneObjectId> {
        let mut ids = vec![subtree.record.persistent_id.clone()];
        for child in &subtree.children {
            ids.extend(self.collect_subtree_persistent_ids(child));
        }
        ids
    }

    /// Collect lights whose group parent persistent ID is in `removed_ids`.
    fn collect_grouped_lights_for_removal(
        &self,
        removed_ids: &[SceneObjectId],
    ) -> Vec<crate::scene::object_store::DetachedLightSnapshot> {
        use crate::scene::object_store::DetachedLightSnapshot;
        let removed_set: std::collections::HashSet<&SceneObjectId> =
            removed_ids.iter().collect();
        let mut detached = Vec::new();

        for entry in &self.point_lights {
            if let Some((light, record)) = &entry.light {
                if let Some(ref parent) = record.light_group_parent {
                    if removed_set.contains(parent) {
                        detached.push(DetachedLightSnapshot {
                            kind: ObjectKind::PointLight,
                            persistent_id: record.persistent_id.clone(),
                            old_group_parent: parent.clone(),
                            point_light: Some(*light),
                            directional_light: None,
                            spot_light: None,
                        });
                    }
                }
            }
        }
        for entry in &self.directional_lights {
            if let Some((light, record)) = &entry.light {
                if let Some(ref parent) = record.light_group_parent {
                    if removed_set.contains(parent) {
                        detached.push(DetachedLightSnapshot {
                            kind: ObjectKind::DirectionalLight,
                            persistent_id: record.persistent_id.clone(),
                            old_group_parent: parent.clone(),
                            point_light: None,
                            directional_light: Some(*light),
                            spot_light: None,
                        });
                    }
                }
            }
        }
        for entry in &self.spot_lights {
            if let Some((light, record)) = &entry.light {
                if let Some(ref parent) = record.light_group_parent {
                    if removed_set.contains(parent) {
                        detached.push(DetachedLightSnapshot {
                            kind: ObjectKind::SpotLight,
                            persistent_id: record.persistent_id.clone(),
                            old_group_parent: parent.clone(),
                            point_light: None,
                            directional_light: None,
                            spot_light: Some(*light),
                        });
                    }
                }
            }
        }

        detached
    }

    // ── Subtree restoration ─────────────────────────────────────────

    /// Restore a subtree from a snapshot and reattach previously detached
    /// grouped lights.
    pub(crate) fn restore_subtree_with_lights(
        &mut self,
        subtree: RestorableSceneSubtree,
        detached_lights: &[crate::scene::object_store::DetachedLightSnapshot],
    ) -> (SceneNodeId, Vec<ObjectId>) {
        let new_root = self.restore_subtree(subtree);
        let new_ids = vec![self
            .object_id_for_node(new_root)
            .expect("just-created node must have object ID")];

        // Reattach grouped lights to the restored nodes.
        // Map old persistent IDs to new ones.
        let mut persistent_remap: HashMap<SceneObjectId, SceneObjectId> = HashMap::new();
        self.collect_restored_persistent_map(new_root, &mut persistent_remap);

        for dl in detached_lights {
            let new_parent = persistent_remap.get(&dl.old_group_parent).cloned();
            match dl.kind {
                ObjectKind::PointLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::PointLight(pl_id) = *handle {
                            if let Some(record) = self.get_point_light_record_mut(pl_id) {
                                record.light_group_parent = new_parent.clone();
                            }
                        }
                    }
                }
                ObjectKind::DirectionalLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::DirectionalLight(dl_id) = *handle {
                            if let Some(record) =
                                self.get_directional_light_record_mut(dl_id)
                            {
                                record.light_group_parent = new_parent.clone();
                            }
                        }
                    }
                }
                ObjectKind::SpotLight => {
                    if let Some(handle) = self.reverse_index.get(&dl.persistent_id) {
                        if let ObjectHandle::SpotLight(sl_id) = *handle {
                            if let Some(record) = self.get_spot_light_record_mut(sl_id) {
                                record.light_group_parent = new_parent.clone();
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        (new_root, new_ids)
    }

    /// Collect a map from old persistent IDs to new ones after restoration.
    fn collect_restored_persistent_map(
        &self,
        node_id: SceneNodeId,
        map: &mut HashMap<SceneObjectId, SceneObjectId>,
    ) {
        if let Some(record) = self.get_node_record(node_id) {
            // Restore creates new persistent IDs, so this records the new one.
            map.insert(record.persistent_id.clone(), record.persistent_id.clone());
        }
        if let Some(node) = self.get_node(node_id) {
            for child_id in node.children.clone() {
                if self.is_valid_node_id(child_id) {
                    self.collect_restored_persistent_map(child_id, map);
                }
            }
        }
    }

    // ── Duplication ──────────────────────────────────────────────────

    /// Duplicate a node subtree, minting new persistent IDs and runtime
    /// handles.  Grouped lights are never implicitly duplicated.
    pub(crate) fn duplicate_node(
        &mut self,
        node_id: SceneNodeId,
        parent: Option<SceneNodeId>,
    ) -> Result<SceneNodeId, String> {
        self.validate_node_ref(node_id)
            .map_err(|e| format!("{e:?}"))?;

        let subtree = self
            .clone_subtree(node_id)
            .ok_or_else(|| "failed to clone subtree for duplication".to_string())?;

        Ok(self.duplicate_subtree(subtree, parent))
    }

    /// Recursively duplicate a `RestorableSceneSubtree`, minting new
    /// persistent IDs for every node.
    fn duplicate_subtree(
        &mut self,
        subtree: RestorableSceneSubtree,
        parent: Option<SceneNodeId>,
    ) -> SceneNodeId {
        let mut new_record = subtree.record.clone();
        // Mint new persistent ID
        new_record.persistent_id = crate::scene::object_store::mint_persistent_id();
        // New stable ID
        new_record.stable_id = None;

        let mut new_node = subtree.node.clone();
        new_node.parent = parent;
        new_node.children.clear();
        new_node.dirty = true;

        let new_id = self.add_node_with_record(parent, new_node, new_record);

        for child in subtree.children {
            self.duplicate_subtree(child, Some(new_id));
        }

        new_id
    }

    /// Duplicate a point light, minting a new persistent ID. Preserves
    /// world-space state and grouping.
    pub(crate) fn duplicate_point_light(
        &mut self,
        id: PointLightId,
    ) -> Result<PointLightId, String> {
        let entry = self
            .point_lights
            .get(id.slot as usize)
            .ok_or("out of bounds".to_string())?;
        if entry.generation != id.generation {
            return Err("generation mismatch".to_string());
        }
        let (light, record) = entry
            .light
            .clone()
            .ok_or("vacant".to_string())?;

        let mut new_record = record.clone();
        new_record.persistent_id = crate::scene::object_store::mint_persistent_id();
        new_record.stable_id = None;

        Ok(self.add_point_light_with_record(light, new_record))
    }

    /// Duplicate a directional light.
    pub(crate) fn duplicate_directional_light(
        &mut self,
        id: DirectionalLightId,
    ) -> Result<DirectionalLightId, String> {
        let entry = self
            .directional_lights
            .get(id.slot as usize)
            .ok_or("out of bounds".to_string())?;
        if entry.generation != id.generation {
            return Err("generation mismatch".to_string());
        }
        let (light, record) = entry
            .light
            .clone()
            .ok_or("vacant".to_string())?;

        let mut new_record = record.clone();
        new_record.persistent_id = crate::scene::object_store::mint_persistent_id();
        new_record.stable_id = None;
        // Duplicated directional shadow config starts non-owning
        new_record.directional_shadow_config = None;

        Ok(self.add_directional_light_with_record(light, new_record))
    }

    /// Duplicate a spot light.
    pub(crate) fn duplicate_spot_light(
        &mut self,
        id: SpotLightId,
    ) -> Result<SpotLightId, String> {
        let entry = self
            .spot_lights
            .get(id.slot as usize)
            .ok_or("out of bounds".to_string())?;
        if entry.generation != id.generation {
            return Err("generation mismatch".to_string());
        }
        let (light, record) = entry
            .light
            .clone()
            .ok_or("vacant".to_string())?;

        let mut new_record = record.clone();
        new_record.persistent_id = crate::scene::object_store::mint_persistent_id();
        new_record.stable_id = None;

        Ok(self.add_spot_light_with_record(light, new_record))
    }

    /// Clone the component store contents for duplication, copying JSON
    /// unchanged unless a remap adapter is registered.
    pub(crate) fn clone_component_store(
        source: &crate::object::component::ComponentStore,
    ) -> crate::object::component::ComponentStore {
        // Just clone the store — envelopes are canonical JSON and copy as-is.
        // Callers should use `prepare_reference_remap` with a registry if
        // remapping is needed.
        source.clone()
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
        use crate::api::bsp::{BspResourceLease, PublishedBspMount};
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
        let lease = BspResourceLease {
            arena_id: 1,
            mesh_handles: vec![mesh],
            texture_handles: vec![],
            material_handles: vec![material],
        };
        mount.arena_id = Some(1);
        scene.set_bsp_mount(PublishedBspMount::new(mount, lease));

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
    fn bsp_mount_retirement_returns_detached_mount_and_leaves_scene_empty() {
        use crate::api::bsp::{BspResourceLease, PublishedBspMount};
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
        let lease = BspResourceLease {
            arena_id: 3,
            mesh_handles: vec![mesh],
            texture_handles: vec![],
            material_handles: vec![material],
        };
        mount.arena_id = Some(3);
        scene.set_bsp_mount(PublishedBspMount::new(mount, lease));
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

    #[test]
    fn invariant_audit_detects_duplicate_persistent_id() {
        let mut scene = SceneWorld::new();
        let root = scene.add_node(None, SceneNode::default());
        let child = scene.add_node(Some(root), SceneNode::default());

        // Both have unique persistent IDs, so audit should pass.
        scene
            .audit_object_invariants()
            .expect("invariants before corruption");

        // Artificially set child's persistent ID to match root's.
        let dup_persistent = scene
            .get_node_record(root)
            .map(|r| r.persistent_id.clone())
            .unwrap();
        if let Some(record) = scene.get_node_record_mut(child) {
            record.persistent_id = dup_persistent;
        }

        let result = scene.audit_object_invariants();
        assert!(result.is_err(), "audit must detect duplicate persistent ID");
        assert!(result.unwrap_err().contains("duplicate persistent"));
    }
}
