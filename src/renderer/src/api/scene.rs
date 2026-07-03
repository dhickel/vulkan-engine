use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::api::assets::{AssetManager, EnvironmentSource};
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use crate::scene::command::{Command, CommandHistory, CommandResult};
use crate::scene::render_submission::RenderSubmission;
use crate::scene::scene_world::{PointLightRefError, ReparentError, SceneNodeRefError, SceneWorld};
use crate::scene::SceneNodeId;

use super::errors::SceneError;

pub const SCENE_FORMAT_VERSION: u32 = 1;

/// Durable asset identity stored in editor scene files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneAssetReference {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_hint: Option<PathBuf>,
}

impl SceneAssetReference {
    pub fn new(id: impl Into<String>, path_hint: Option<PathBuf>) -> Self {
        Self {
            id: id.into(),
            path_hint,
        }
    }
}

/// Point light ID with slot+generation semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointLightId {
    pub slot: u32,
    pub generation: u32,
}

/// Point light definition.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

impl PointLight {
    /// Validate point light parameters.
    fn validate(&self) -> Result<(), SceneError> {
        if !self.range.is_finite() || self.range <= 0.0 {
            return Err(SceneError::InvalidPointLight(
                "range must be finite and > 0.0".to_string(),
            ));
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(SceneError::InvalidPointLight(
                "intensity must be finite and >= 0.0".to_string(),
            ));
        }
        if !self.color.is_finite() {
            return Err(SceneError::InvalidPointLight(
                "color must be finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Clamp color to valid range.
    fn sanitize_color(&self) -> Vec3 {
        self.color.max(Vec3::ZERO)
    }
}

/// Fragment-local node identifier used during scene merge.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SceneFragmentNodeId {
    pub index: u32,
}

impl SceneFragmentNodeId {
    pub const fn new(index: u32) -> Self {
        Self { index }
    }
}

/// One node in a detached scene fragment.
#[derive(Clone, Debug)]
pub struct SceneFragmentNode {
    pub parent: Option<SceneFragmentNodeId>,
    pub children: Vec<SceneFragmentNodeId>,
    pub local_transform: Mat4,
    pub meshes: Vec<MeshHandle>,
}

impl Default for SceneFragmentNode {
    fn default() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            local_transform: Mat4::IDENTITY,
            meshes: Vec::new(),
        }
    }
}

/// Detached scene hierarchy that can be mounted into a [`Scene`].
#[derive(Clone, Debug, Default)]
pub struct SceneFragment {
    nodes: Vec<SceneFragmentNode>,
    root: Option<SceneFragmentNodeId>,
    skybox: Option<EnvironmentHandle>,
}

impl SceneFragment {
    /// Thread: Any
    /// May Stall: No
    pub fn new() -> Self {
        Self::default()
    }

    /// Thread: Any
    /// May Stall: No
    pub fn root(&self) -> Option<SceneFragmentNodeId> {
        self.root
    }

    /// Thread: Any
    /// May Stall: No
    pub fn skybox(&self) -> Option<EnvironmentHandle> {
        self.skybox
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_skybox(&mut self, env: EnvironmentHandle) {
        self.skybox = Some(env);
    }

    /// Thread: Any
    /// May Stall: No
    pub fn clear_skybox(&mut self) {
        self.skybox = None;
    }

    /// Thread: Any
    /// May Stall: No
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Thread: Any
    /// May Stall: No
    pub fn node(&self, node: SceneFragmentNodeId) -> Option<&SceneFragmentNode> {
        self.nodes.get(node.index as usize)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn add_node(
        &mut self,
        parent: Option<SceneFragmentNodeId>,
        transform: Mat4,
        meshes: Vec<MeshHandle>,
    ) -> Result<SceneFragmentNodeId, SceneError> {
        if let Some(parent_id) = parent {
            if self.nodes.get(parent_id.index as usize).is_none() {
                return Err(SceneError::MergeFailed(format!(
                    "fragment parent node {} is out of bounds",
                    parent_id.index
                )));
            }
        }

        let id = SceneFragmentNodeId::new(self.nodes.len() as u32);
        self.nodes.push(SceneFragmentNode {
            parent,
            children: Vec::new(),
            local_transform: transform,
            meshes,
        });

        if let Some(parent_id) = parent {
            let parent_node = &mut self.nodes[parent_id.index as usize];
            parent_node.children.push(id);
        } else if self.root.is_none() {
            self.root = Some(id);
        }

        Ok(id)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn add_node_default(
        &mut self,
        parent: Option<SceneFragmentNodeId>,
    ) -> Result<SceneFragmentNodeId, SceneError> {
        self.add_node(parent, Mat4::IDENTITY, Vec::new())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_root(&mut self, node: SceneFragmentNodeId) -> Result<(), SceneError> {
        if self.nodes.get(node.index as usize).is_none() {
            return Err(SceneError::MergeFailed(format!(
                "fragment root node {} is out of bounds",
                node.index
            )));
        }
        self.root = Some(node);
        Ok(())
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<SceneFragmentNode>,
        Option<SceneFragmentNodeId>,
        Option<EnvironmentHandle>,
    ) {
        (self.nodes, self.root, self.skybox)
    }
}

/// Result of mounting a [`SceneFragment`] into a [`Scene`].
#[derive(Clone, Debug)]
pub struct SceneFragmentMount {
    pub mounted_root: SceneNodeId,
    pub node_mapping: HashMap<SceneFragmentNodeId, SceneNodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneNodeSummary {
    pub id: SceneNodeId,
    pub parent: Option<SceneNodeId>,
    pub stable_id: Option<String>,
    pub name: String,
    pub local_transform: Mat4,
    pub asset: Option<SceneAssetReference>,
    pub material_overrides: BTreeMap<String, String>,
    pub tags: Vec<String>,
    pub child_count: usize,
    pub mesh_count: usize,
}

/// Public scene facade.
pub struct Scene {
    world: SceneWorld,
    scene_id: String,
    display_name: Option<String>,
    next_stable_node_id: u64,
    skybox_asset: Option<SceneAssetReference>,
    materials: BTreeMap<String, SerializedMaterialOverride>,
    editor: serde_json::Value,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    /// Thread: Any
    /// May Stall: No
    pub fn new() -> Self {
        Self {
            world: SceneWorld::new(),
            scene_id: "scene.untitled".to_string(),
            display_name: None,
            next_stable_node_id: 1,
            skybox_asset: None,
            materials: BTreeMap::new(),
            editor: serde_json::json!({}),
        }
    }

    /// Thread: Any
    /// May Stall: No
    pub fn root(&self) -> Option<SceneNodeId> {
        self.world.root_id()
    }

    /// Thread: Any
    /// May Stall: No
    pub fn create_node(
        &mut self,
        parent: Option<SceneNodeId>,
        transform: Mat4,
    ) -> Result<SceneNodeId, SceneError> {
        if let Some(parent_id) = parent {
            self.validate_parent(parent_id)?;
        }

        let id = self
            .world
            .add_node_with_parts(parent, transform, Vec::new());
        self.ensure_node_persistence_metadata(id);
        Ok(id)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn create_node_default(
        &mut self,
        parent: Option<SceneNodeId>,
    ) -> Result<SceneNodeId, SceneError> {
        self.create_node(parent, Mat4::IDENTITY)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn remove_node(&mut self, node: SceneNodeId) -> Result<(), SceneError> {
        self.validate_node(node)?;
        if self.world.remove_node(node) {
            return Ok(());
        }

        Err(SceneError::InvalidNode(node))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_transform(&mut self, node: SceneNodeId, transform: Mat4) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.local_transform = transform;
        node_ref.dirty = true;

        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn reparent_node(
        &mut self,
        node: SceneNodeId,
        parent: Option<SceneNodeId>,
    ) -> Result<(), SceneError> {
        self.world
            .reparent_node(node, parent)
            .map_err(|err| map_reparent_error(node, parent, err))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_node_name(
        &mut self,
        node: SceneNodeId,
        name: impl Into<String>,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.name = name.into();
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_node_asset_reference(
        &mut self,
        node: SceneNodeId,
        asset: SceneAssetReference,
    ) -> Result<(), SceneError> {
        validate_asset_reference(&asset, "node asset")?;
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.asset = Some(asset);
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn clear_node_asset_reference(&mut self, node: SceneNodeId) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.asset = None;
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_node_material_override(
        &mut self,
        node: SceneNodeId,
        slot: impl Into<String>,
        material_override_id: impl Into<String>,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;
        let slot = slot.into();
        let material_override_id = material_override_id.into();
        validate_material_override(&slot, &material_override_id)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref
            .material_overrides
            .insert(slot, material_override_id);
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn clear_node_material_override(
        &mut self,
        node: SceneNodeId,
        slot: impl AsRef<str>,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;
        let slot = slot.as_ref().trim();
        if slot.is_empty() {
            return Err(SceneError::MergeFailed(
                "material override slot cannot be empty".to_string(),
            ));
        }

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.material_overrides.remove(slot);
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_node_tags(
        &mut self,
        node: SceneNodeId,
        tags: Vec<String>,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let mut normalized = Vec::new();
        for tag in tags {
            let tag = tag.trim();
            if tag.is_empty() {
                continue;
            }
            if !normalized.iter().any(|existing| existing == tag) {
                normalized.push(tag.to_string());
            }
        }

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.tags = normalized;
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn transform(&self, node: SceneNodeId) -> Result<Mat4, SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        Ok(node_ref.local_transform)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn node_stable_id(&self, node: SceneNodeId) -> Result<Option<String>, SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        Ok(node_ref.stable_id.clone())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn find_node_by_stable_id(&self, stable_id: &str) -> Option<SceneNodeId> {
        self.world
            .serializable_nodes()
            .find_map(|(id, node)| (node.stable_id.as_deref() == Some(stable_id)).then_some(id))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn node_summaries(&self) -> Vec<SceneNodeSummary> {
        let mut nodes: Vec<_> = self
            .world
            .serializable_nodes()
            .map(|(id, node)| SceneNodeSummary {
                id,
                parent: node.parent,
                stable_id: node.stable_id.clone(),
                name: if node.name.is_empty() {
                    node.stable_id
                        .clone()
                        .unwrap_or_else(|| format!("Node {}", id.slot))
                } else {
                    node.name.clone()
                },
                local_transform: node.local_transform,
                asset: node.asset.clone(),
                material_overrides: node.material_overrides.clone(),
                tags: node.tags.clone(),
                child_count: node.children.len(),
                mesh_count: node.meshes.len(),
            })
            .collect();
        nodes.sort_by_key(|node| (node.parent.is_some(), node.name.clone(), node.id.slot));
        nodes
    }

    /// Thread: Any
    /// May Stall: No
    pub fn is_valid_node(&self, node: SceneNodeId) -> bool {
        self.world.is_valid_node_id(node)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn add_mesh(&mut self, node: SceneNodeId, mesh: MeshHandle) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.meshes.push(mesh);

        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn clear_meshes(&mut self, node: SceneNodeId) -> Result<(), SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.meshes.clear();

        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_camera(&mut self, view: Mat4, projection: Mat4, position: Vec3) {
        self.world.update_camera(view, projection, position);
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_skybox(&mut self, env: EnvironmentHandle) {
        self.world.set_skybox_env_id(env);
        self.skybox_asset = None;
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_skybox_asset_reference(
        &mut self,
        env: EnvironmentHandle,
        asset: SceneAssetReference,
    ) -> Result<(), SceneError> {
        validate_asset_reference(&asset, "environment")?;
        self.world.set_skybox_env_id(env);
        self.skybox_asset = Some(asset);
        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn create_point_light(&mut self, light: PointLight) -> Result<PointLightId, SceneError> {
        light.validate()?;
        let sanitized = PointLight {
            color: light.sanitize_color(),
            ..light
        };
        Ok(self.world.add_point_light(sanitized))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn update_point_light(
        &mut self,
        id: PointLightId,
        light: PointLight,
    ) -> Result<(), SceneError> {
        light.validate()?;
        self.validate_point_light(id)?;

        let sanitized = PointLight {
            color: light.sanitize_color(),
            ..light
        };

        if self.world.update_point_light(id, sanitized) {
            return Ok(());
        }

        Err(SceneError::InvalidPointLight(format!(
            "failed to update point light (slot={}, generation={})",
            id.slot, id.generation
        )))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn remove_point_light(&mut self, id: PointLightId) -> Result<(), SceneError> {
        self.validate_point_light(id)?;

        if self.world.remove_point_light(id) {
            return Ok(());
        }

        Err(SceneError::InvalidPointLight(format!(
            "failed to remove point light (slot={}, generation={})",
            id.slot, id.generation
        )))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn merge_fragment(
        &mut self,
        parent: Option<SceneNodeId>,
        fragment: SceneFragment,
    ) -> Result<SceneFragmentMount, SceneError> {
        if let Some(parent_id) = parent {
            self.validate_parent(parent_id)?;
        }

        let (nodes, root, skybox) = fragment.into_parts();
        if nodes.is_empty() {
            return Err(SceneError::MergeFailed(
                "cannot merge an empty scene fragment".to_string(),
            ));
        }

        let fragment_root = resolve_fragment_root(&nodes, root)?;
        let merge_plan = build_fragment_merge_plan(&nodes, fragment_root)?;

        let mut mapping = HashMap::with_capacity(nodes.len());
        for (fragment_node, fragment_parent) in merge_plan {
            let fragment_idx = fragment_node.index as usize;
            let source = &nodes[fragment_idx];

            let scene_parent = if let Some(fragment_parent_id) = fragment_parent {
                let Some(mapped_parent) = mapping.get(&fragment_parent_id) else {
                    return Err(SceneError::MergeFailed(format!(
                        "fragment merge invariant violated: parent {} was not cloned",
                        fragment_parent_id.index
                    )));
                };
                Some(*mapped_parent)
            } else {
                parent
            };

            let new_node = self.world.add_node_with_parts(
                scene_parent,
                source.local_transform,
                source.meshes.clone(),
            );
            self.ensure_node_persistence_metadata(new_node);
            mapping.insert(fragment_node, new_node);
        }

        if let Some(env) = skybox {
            self.set_skybox(env);
        }

        let Some(mounted_root) = mapping.get(&fragment_root).copied() else {
            return Err(SceneError::MergeFailed(
                "fragment merge failed to map root node".to_string(),
            ));
        };

        Ok(SceneFragmentMount {
            mounted_root,
            node_mapping: mapping,
        })
    }

    /// Serialize the scene to a versioned JSON file. Runtime handles are not
    /// written; asset-backed content uses durable asset IDs plus optional path hints.
    ///
    /// Thread: Any
    /// May Stall: Yes (file I/O)
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let serialized = SerializedScene::from_scene(self);
        let json = serde_json::to_string_pretty(&serialized)
            .map_err(|e| SceneError::MergeFailed(format!("scene serialization failed: {e}")))?;
        std::fs::write(path.as_ref(), json)
            .map_err(|e| SceneError::MergeFailed(format!("failed to write scene file: {e}")))?;
        Ok(())
    }

    /// Load a scene from a versioned JSON file. Durable asset IDs are resolved
    /// through the provided [`AssetManager`], with path hints as fallback.
    ///
    /// Thread: Any
    /// May Stall: Yes (file I/O + asset loading)
    pub fn load(
        path: impl AsRef<std::path::Path>,
        assets: &mut crate::api::assets::AssetManager,
    ) -> Result<Self, crate::api::errors::RendererError> {
        let json = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            crate::api::errors::RendererError::Init(
                crate::api::errors::RendererInitError::StartupScene(format!(
                    "failed to read scene file: {e}"
                )),
            )
        })?;
        let serialized: SerializedScene = serde_json::from_str(&json).map_err(|e| {
            crate::api::errors::RendererError::Init(
                crate::api::errors::RendererInitError::StartupScene(format!(
                    "scene deserialization failed: {e}"
                )),
            )
        })?;
        serialized.into_scene(assets)
    }

    pub(crate) fn from_world(world: SceneWorld) -> Self {
        Self {
            world,
            scene_id: "scene.imported".to_string(),
            display_name: None,
            next_stable_node_id: 1,
            skybox_asset: None,
            materials: BTreeMap::new(),
            editor: serde_json::json!({}),
        }
    }

    pub(crate) fn update_camera(&mut self, view: Mat4, projection: Mat4, position: Vec3) {
        self.world.update_camera(view, projection, position);
    }

    /// Cast a ray into the scene and return the closest intersected node.
    /// Uses the camera's view/projection and screen coordinates.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn pick(
        &self,
        screen_x: f32,
        screen_y: f32,
        viewport_width: u32,
        viewport_height: u32,
        view: Mat4,
        projection: Mat4,
        camera_position: Vec3,
    ) -> Option<SceneNodeId> {
        let inv_vp = (projection * view).inverse();
        let ray = crate::data::camera::Ray::from_screen(
            (screen_x, screen_y),
            (viewport_width, viewport_height),
            inv_vp,
            camera_position,
        );
        self.world.pick_ray(&ray)
    }

    /// Cast a ray using the scene's last renderer-supplied camera matrices.
    ///
    /// Picking currently uses transform-aware editor proxy AABBs because the
    /// scene graph does not own CPU mesh bounds. Mesh-backed nodes use a
    /// one-unit local proxy and empty group nodes use a smaller origin proxy.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn pick_last_camera(
        &self,
        screen_x: f32,
        screen_y: f32,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Option<SceneNodeId> {
        let camera = self.world.camera_data();
        self.pick(
            screen_x,
            screen_y,
            viewport_width,
            viewport_height,
            camera.view,
            camera.projection,
            camera.cam_pos,
        )
    }

    /// Thread: Any
    /// May Stall: No
    pub fn execute_command(
        &mut self,
        history: &mut CommandHistory,
        command: Box<dyn Command>,
    ) -> Result<CommandResult, SceneError> {
        history.execute(command, &mut self.world)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn undo_command(
        &mut self,
        history: &mut CommandHistory,
    ) -> Result<CommandResult, SceneError> {
        history.undo(&mut self.world)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn redo_command(
        &mut self,
        history: &mut CommandHistory,
    ) -> Result<CommandResult, SceneError> {
        history.redo(&mut self.world)
    }

    pub(crate) fn build_submission(&mut self) -> RenderSubmission {
        self.world.build_submission()
    }

    fn validate_node(&self, node: SceneNodeId) -> Result<(), SceneError> {
        self.world
            .validate_node_ref(node)
            .map_err(|err| map_node_ref_error(node, err))
    }

    fn validate_parent(&self, parent: SceneNodeId) -> Result<(), SceneError> {
        self.world
            .validate_node_ref(parent)
            .map_err(|err| map_parent_ref_error(parent, err))
    }

    fn validate_point_light(&self, id: PointLightId) -> Result<(), SceneError> {
        self.world
            .validate_point_light_ref(id)
            .map_err(|err| map_point_light_ref_error(id, err))
    }

    fn ensure_node_persistence_metadata(&mut self, node: SceneNodeId) {
        let stable_id = format!("node.{:06}", self.next_stable_node_id);
        self.next_stable_node_id += 1;

        if let Some(node_ref) = self.world.get_node_mut(node) {
            if node_ref.stable_id.is_none() {
                node_ref.stable_id = Some(stable_id.clone());
            }
            if node_ref.name.is_empty() {
                node_ref.name = stable_id;
            }
        }
    }
}

fn map_node_ref_error(node: SceneNodeId, err: SceneNodeRefError) -> SceneError {
    match err {
        SceneNodeRefError::GenerationMismatch => SceneError::StaleNode(node),
        SceneNodeRefError::OutOfBounds | SceneNodeRefError::Vacant => SceneError::InvalidNode(node),
    }
}

fn map_parent_ref_error(parent: SceneNodeId, err: SceneNodeRefError) -> SceneError {
    match err {
        SceneNodeRefError::GenerationMismatch => SceneError::StaleNode(parent),
        SceneNodeRefError::OutOfBounds | SceneNodeRefError::Vacant => {
            SceneError::InvalidParent(parent)
        }
    }
}

fn map_point_light_ref_error(id: PointLightId, err: PointLightRefError) -> SceneError {
    match err {
        PointLightRefError::GenerationMismatch => SceneError::StalePointLight(id),
        PointLightRefError::OutOfBounds | PointLightRefError::Vacant => {
            SceneError::InvalidPointLight(format!(
                "point light out of bounds or vacant (slot={}, generation={})",
                id.slot, id.generation
            ))
        }
    }
}

fn map_reparent_error(
    node: SceneNodeId,
    parent: Option<SceneNodeId>,
    err: ReparentError,
) -> SceneError {
    match err {
        ReparentError::InvalidNode(ref_err) => map_node_ref_error(node, ref_err),
        ReparentError::InvalidParent(ref_err) => map_parent_ref_error(
            parent.expect("invalid parent error requires parent"),
            ref_err,
        ),
        ReparentError::Cycle => SceneError::CycleDetected,
    }
}

fn resolve_fragment_root(
    nodes: &[SceneFragmentNode],
    root: Option<SceneFragmentNodeId>,
) -> Result<SceneFragmentNodeId, SceneError> {
    if let Some(root_id) = root {
        let root_index = root_id.index as usize;
        if root_index >= nodes.len() {
            return Err(SceneError::MergeFailed(format!(
                "fragment root {} is out of bounds",
                root_id.index
            )));
        }
        return Ok(root_id);
    }

    let inferred_roots: Vec<SceneFragmentNodeId> = nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| {
            if node.parent.is_none() {
                Some(SceneFragmentNodeId::new(idx as u32))
            } else {
                None
            }
        })
        .collect();

    if inferred_roots.len() == 1 {
        return Ok(inferred_roots[0]);
    }

    Err(SceneError::MergeFailed(
        "fragment root is missing or ambiguous".to_string(),
    ))
}

fn build_fragment_merge_plan(
    nodes: &[SceneFragmentNode],
    root: SceneFragmentNodeId,
) -> Result<Vec<(SceneFragmentNodeId, Option<SceneFragmentNodeId>)>, SceneError> {
    let mut visit_state = vec![0_u8; nodes.len()];
    let mut resolved_parent = vec![None::<SceneFragmentNodeId>; nodes.len()];
    let mut discovered = vec![false; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());

    for (idx, node) in nodes.iter().enumerate() {
        if let Some(parent) = node.parent {
            if parent.index as usize >= nodes.len() {
                return Err(SceneError::MergeFailed(format!(
                    "fragment node {} has out-of-bounds parent {}",
                    idx, parent.index
                )));
            }
        }
    }

    let root_idx = root.index as usize;
    build_fragment_merge_plan_recursive(
        nodes,
        root,
        None,
        &mut visit_state,
        &mut discovered,
        &mut resolved_parent,
        &mut order,
    )?;

    if visit_state.iter().any(|state| *state != 2) {
        return Err(SceneError::MergeFailed(
            "fragment contains disconnected nodes".to_string(),
        ));
    }

    if discovered[root_idx] && resolved_parent[root_idx].is_some() {
        return Err(SceneError::MergeFailed(
            "fragment root cannot be visited as a child".to_string(),
        ));
    }

    Ok(order)
}

fn build_fragment_merge_plan_recursive(
    nodes: &[SceneFragmentNode],
    node_id: SceneFragmentNodeId,
    parent_id: Option<SceneFragmentNodeId>,
    visit_state: &mut [u8],
    discovered: &mut [bool],
    resolved_parent: &mut [Option<SceneFragmentNodeId>],
    order: &mut Vec<(SceneFragmentNodeId, Option<SceneFragmentNodeId>)>,
) -> Result<(), SceneError> {
    let idx = node_id.index as usize;
    if idx >= nodes.len() {
        return Err(SceneError::MergeFailed(format!(
            "fragment references unknown node {}",
            node_id.index
        )));
    }

    if visit_state[idx] == 1 {
        return Err(SceneError::CycleDetected);
    }

    if visit_state[idx] == 2 {
        if resolved_parent[idx] != parent_id {
            return Err(SceneError::MergeFailed(format!(
                "fragment node {} is referenced by multiple parents",
                node_id.index
            )));
        }
        return Ok(());
    }

    visit_state[idx] = 1;
    discovered[idx] = true;
    resolved_parent[idx] = parent_id;
    order.push((node_id, parent_id));

    for child in nodes[idx].children.iter().copied() {
        build_fragment_merge_plan_recursive(
            nodes,
            child,
            Some(node_id),
            visit_state,
            discovered,
            resolved_parent,
            order,
        )?;
    }

    visit_state[idx] = 2;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene serialization
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedTransform {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedAssetReference {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path_hint: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedVisibility {
    visible: bool,
    locked: bool,
    layer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedNode {
    id: String,
    parent: Option<String>,
    name: String,
    transform: SerializedTransform,
    asset: Option<SerializedAssetReference>,
    #[serde(default)]
    material_overrides: BTreeMap<String, String>,
    #[serde(default = "default_visibility")]
    visibility: SerializedVisibility,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefab: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedPointLight {
    id: String,
    kind: String,
    #[serde(default)]
    parent: Option<String>,
    position: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    range: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedEnvironment {
    asset: SerializedAssetReference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedMaterialOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base: Option<String>,
    #[serde(default)]
    parameters: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedScene {
    #[serde(default)]
    format_version: u32,
    scene_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    root_nodes: Vec<String>,
    nodes: Vec<SerializedNode>,
    lights: Vec<SerializedPointLight>,
    environment: Option<SerializedEnvironment>,
    #[serde(default)]
    materials: BTreeMap<String, SerializedMaterialOverride>,
    #[serde(default = "default_editor_metadata")]
    editor: serde_json::Value,
}

impl SerializedScene {
    fn from_scene(scene: &Scene) -> Self {
        let mut fallback_index = 1_u64;
        let nodes: Vec<SerializedNode> = scene
            .world
            .serializable_nodes()
            .map(|(_, node)| {
                let id = node.stable_id.clone().unwrap_or_else(|| {
                    let generated = format!("node.autogenerated.{fallback_index:06}");
                    fallback_index += 1;
                    generated
                });
                SerializedNode {
                    id,
                    parent: node.parent.and_then(|parent| {
                        scene
                            .world
                            .get_node(parent)
                            .and_then(|parent_node| parent_node.stable_id.clone())
                    }),
                    name: if node.name.is_empty() {
                        node.stable_id
                            .clone()
                            .unwrap_or_else(|| "Unnamed Node".to_string())
                    } else {
                        node.name.clone()
                    },
                    transform: SerializedTransform::from_mat4(node.local_transform),
                    asset: node
                        .asset
                        .as_ref()
                        .map(SerializedAssetReference::from_scene_ref),
                    material_overrides: node.material_overrides.clone(),
                    visibility: SerializedVisibility {
                        visible: true,
                        locked: false,
                        layer: "world".to_string(),
                    },
                    tags: node.tags.clone(),
                    prefab: None,
                }
            })
            .collect();

        let root_nodes = nodes
            .iter()
            .filter_map(|node| node.parent.is_none().then_some(node.id.clone()))
            .collect();

        let lights: Vec<SerializedPointLight> = scene
            .world
            .serializable_lights()
            .enumerate()
            .map(|(idx, (_, light))| SerializedPointLight {
                id: format!("light.{:06}", idx + 1),
                kind: "point".to_string(),
                parent: None,
                position: light.position.to_array(),
                color: light.color.to_array(),
                intensity: light.intensity,
                range: light.range,
            })
            .collect();

        SerializedScene {
            format_version: SCENE_FORMAT_VERSION,
            scene_id: scene.scene_id.clone(),
            display_name: scene.display_name.clone(),
            root_nodes,
            nodes,
            lights,
            environment: scene
                .skybox_asset
                .as_ref()
                .map(|asset| SerializedEnvironment {
                    asset: SerializedAssetReference::from_scene_ref(asset),
                }),
            materials: scene.materials.clone(),
            editor: scene.editor.clone(),
        }
    }

    fn into_scene(
        self,
        assets: &mut AssetManager,
    ) -> Result<Scene, crate::api::errors::RendererError> {
        self.into_scene_with_loader(assets)
    }

    fn into_scene_with_loader(
        self,
        loader: &mut impl SceneAssetLoader,
    ) -> Result<Scene, crate::api::errors::RendererError> {
        let load_plan = self.validate()?;
        let mut scene = Scene::new();
        scene.scene_id = self.scene_id.clone();
        scene.display_name = self.display_name.clone();
        scene.materials = self.materials.clone();
        scene.editor = self.editor.clone();

        let mut id_map: HashMap<String, SceneNodeId> = HashMap::new();

        for node_index in load_plan.creation_order {
            let serialized_node = &self.nodes[node_index];
            let parent = serialized_node
                .parent
                .as_ref()
                .and_then(|parent_id| id_map.get(parent_id).copied());
            let transform = serialized_node.transform.to_mat4();

            let node_id = if let Some(asset) = serialized_node.asset.as_ref() {
                let scene_asset = asset.to_scene_ref(&format!("node '{}'", serialized_node.id))?;
                let fragment = loader
                    .load_model_ref(&scene_asset)
                    .map_err(crate::api::errors::RendererError::from)?;
                let mount = scene
                    .merge_fragment(parent, fragment)
                    .map_err(crate::api::errors::RendererError::from)?;
                let mounted = mount.mounted_root;
                scene
                    .set_transform(mounted, transform)
                    .map_err(crate::api::errors::RendererError::from)?;
                scene
                    .set_node_asset_reference(mounted, scene_asset)
                    .map_err(crate::api::errors::RendererError::from)?;
                mounted
            } else {
                scene
                    .create_node(parent, transform)
                    .map_err(crate::api::errors::RendererError::from)?
            };

            if let Some(node_ref) = scene.world.get_node_mut(node_id) {
                node_ref.stable_id = Some(serialized_node.id.clone());
                node_ref.name = serialized_node.name.clone();
                node_ref.material_overrides = serialized_node.material_overrides.clone();
                node_ref.tags = serialized_node.tags.clone();
            }
            id_map.insert(serialized_node.id.clone(), node_id);
        }

        for light in self.lights {
            if light.kind != "point" {
                return Err(SceneError::InvalidPointLight(format!(
                    "unsupported light kind '{}'",
                    light.kind
                ))
                .into());
            }
            scene
                .create_point_light(PointLight {
                    position: Vec3::from_array(light.position),
                    color: Vec3::from_array(light.color),
                    intensity: light.intensity,
                    range: light.range,
                })
                .map_err(crate::api::errors::RendererError::from)?;
        }

        if let Some(environment) = self.environment {
            let asset = environment.asset.to_scene_ref("environment")?;
            let env = loader
                .load_environment_ref(&asset)
                .map_err(crate::api::errors::RendererError::from)?;
            scene
                .set_skybox_asset_reference(env, asset)
                .map_err(crate::api::errors::RendererError::from)?;
        }

        Ok(scene)
    }

    fn validate(&self) -> Result<SceneLoadPlan, SceneError> {
        if self.format_version != SCENE_FORMAT_VERSION {
            return Err(SceneError::UnsupportedSceneVersion {
                found: self.format_version,
                expected: SCENE_FORMAT_VERSION,
            });
        }

        let mut seen = HashSet::new();
        let mut index_by_id = HashMap::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            if !seen.insert(node.id.clone()) {
                return Err(SceneError::DuplicateSerializedNodeId(node.id.clone()));
            }
            index_by_id.insert(node.id.clone(), idx);
            if let Some(asset) = &node.asset {
                asset.to_scene_ref(&format!("node '{}'", node.id))?;
            }
        }

        for node in &self.nodes {
            if let Some(parent_id) = &node.parent {
                if !index_by_id.contains_key(parent_id) {
                    return Err(SceneError::BadSerializedParent {
                        node_id: node.id.clone(),
                        parent_id: parent_id.clone(),
                    });
                }
            }
        }

        if let Some(environment) = &self.environment {
            environment.asset.to_scene_ref("environment")?;
        }

        if self.root_nodes.is_empty() && !self.nodes.is_empty() {
            return Err(SceneError::DisconnectedGraph(
                "scene has nodes but no root_nodes".to_string(),
            ));
        }

        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();
        let mut order = Vec::with_capacity(self.nodes.len());
        for root_id in &self.root_nodes {
            let Some(root_index) = index_by_id.get(root_id).copied() else {
                return Err(SceneError::DisconnectedGraph(format!(
                    "root node '{root_id}' is not present in nodes"
                )));
            };
            self.visit_node(
                root_index,
                &index_by_id,
                &mut visiting,
                &mut visited,
                &mut order,
            )?;
        }

        if visited.len() != self.nodes.len() {
            return Err(SceneError::DisconnectedGraph(format!(
                "{} node(s) are unreachable from root_nodes",
                self.nodes.len() - visited.len()
            )));
        }

        let mut roots: Vec<String> = self
            .nodes
            .iter()
            .filter_map(|node| node.parent.is_none().then_some(node.id.clone()))
            .collect();
        roots.sort();
        let mut declared_roots = self.root_nodes.clone();
        declared_roots.sort();
        if roots != declared_roots {
            return Err(SceneError::DisconnectedGraph(format!(
                "root_nodes {:?} do not match parentless nodes {:?}",
                self.root_nodes, roots
            )));
        }
        if roots.len() > 1 {
            return Err(SceneError::DisconnectedGraph(
                "runtime scene currently supports one root node".to_string(),
            ));
        }

        Ok(SceneLoadPlan {
            creation_order: order,
        })
    }

    fn visit_node(
        &self,
        idx: usize,
        index_by_id: &HashMap<String, usize>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        order: &mut Vec<usize>,
    ) -> Result<(), SceneError> {
        let node = &self.nodes[idx];
        if visited.contains(&node.id) {
            return Ok(());
        }
        if !visiting.insert(node.id.clone()) {
            return Err(SceneError::CycleDetected);
        }

        order.push(idx);
        for (child_idx, child) in self.nodes.iter().enumerate() {
            if child.parent.as_deref() == Some(node.id.as_str()) {
                if !index_by_id.contains_key(&child.id) {
                    return Err(SceneError::DisconnectedGraph(format!(
                        "child '{}' is not indexed",
                        child.id
                    )));
                }
                self.visit_node(child_idx, index_by_id, visiting, visited, order)?;
            }
        }

        visiting.remove(&node.id);
        visited.insert(node.id.clone());
        Ok(())
    }
}

struct SceneLoadPlan {
    creation_order: Vec<usize>,
}

trait SceneAssetLoader {
    fn load_model_ref(
        &mut self,
        asset: &SceneAssetReference,
    ) -> Result<SceneFragment, super::errors::AssetError>;
    fn load_environment_ref(
        &mut self,
        asset: &SceneAssetReference,
    ) -> Result<EnvironmentHandle, super::errors::AssetError>;
}

impl SceneAssetLoader for AssetManager<'_> {
    fn load_model_ref(
        &mut self,
        asset: &SceneAssetReference,
    ) -> Result<SceneFragment, super::errors::AssetError> {
        self.load_model_asset(&asset.id).or_else(|err| {
            if let Some(path) = &asset.path_hint {
                self.load_model(path)
            } else {
                Err(err)
            }
        })
    }

    fn load_environment_ref(
        &mut self,
        asset: &SceneAssetReference,
    ) -> Result<EnvironmentHandle, super::errors::AssetError> {
        self.load_environment_asset(&asset.id).or_else(|err| {
            if let Some(path) = &asset.path_hint {
                self.load_environment(EnvironmentSource::Auto(path.clone()))
            } else {
                Err(err)
            }
        })
    }
}

impl SerializedTransform {
    fn from_mat4(transform: Mat4) -> Self {
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        Self {
            translation: translation.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
        }
    }

    fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from_array(self.scale),
            Quat::from_array(self.rotation).normalize(),
            Vec3::from_array(self.translation),
        )
    }
}

impl SerializedAssetReference {
    fn from_scene_ref(asset: &SceneAssetReference) -> Self {
        Self {
            id: Some(asset.id.clone()),
            path_hint: asset.path_hint.clone(),
        }
    }

    fn to_scene_ref(&self, context: &str) -> Result<SceneAssetReference, SceneError> {
        let Some(id) = self
            .id
            .as_ref()
            .map(|id| id.trim())
            .filter(|id| !id.is_empty())
        else {
            return Err(SceneError::MissingAssetId(context.to_string()));
        };
        Ok(SceneAssetReference {
            id: id.to_string(),
            path_hint: self.path_hint.clone(),
        })
    }
}

fn validate_asset_reference(asset: &SceneAssetReference, context: &str) -> Result<(), SceneError> {
    if asset.id.trim().is_empty() {
        return Err(SceneError::MissingAssetId(context.to_string()));
    }
    Ok(())
}

fn validate_material_override(slot: &str, material_override_id: &str) -> Result<(), SceneError> {
    if slot.trim().is_empty() {
        return Err(SceneError::MergeFailed(
            "material override slot cannot be empty".to_string(),
        ));
    }
    if material_override_id.trim().is_empty() {
        return Err(SceneError::MergeFailed(
            "material override id cannot be empty".to_string(),
        ));
    }
    Ok(())
}

fn default_visibility() -> SerializedVisibility {
    SerializedVisibility {
        visible: true,
        locked: false,
        layer: "world".to_string(),
    }
}

fn default_editor_metadata() -> serde_json::Value {
    serde_json::json!({})
}

#[cfg(test)]
mod tests {
    use super::{
        PointLight, Scene, SceneAssetLoader, SceneAssetReference, SceneFragment, SceneFragmentNode,
        SceneFragmentNodeId, SerializedScene,
    };
    use crate::api::errors::{AssetError, RendererError, SceneError};
    use crate::data::handles::{EnvironmentHandle, MeshHandle};
    use crate::scene::command::{CommandHistory, PlaceAssetCommand};
    use glam::{Mat4, Vec3};
    use std::path::PathBuf;

    #[test]
    fn stale_handle_rejected() {
        let mut scene = Scene::new();
        let node = scene.create_node_default(None).unwrap();
        scene.remove_node(node).unwrap();

        let result = scene.set_transform(node, Mat4::IDENTITY);
        assert!(matches!(result, Err(SceneError::StaleNode(id)) if id == node));
    }

    #[test]
    fn merge_fragment_rejects_cycles() {
        let mut scene = Scene::new();

        let mut fragment = SceneFragment::new();
        fragment
            .add_node_default(None)
            .expect("fragment root should be added");
        fragment
            .add_node_default(Some(SceneFragmentNodeId::new(0)))
            .expect("fragment child should be added");

        fragment.nodes[1].children.push(SceneFragmentNodeId::new(1));

        let result = scene.merge_fragment(None, fragment);
        assert!(matches!(result, Err(SceneError::CycleDetected)));
    }

    #[test]
    fn merge_fragment_returns_node_mapping() {
        let mut scene = Scene::new();
        let parent = scene.create_node_default(None).unwrap();

        let mut fragment = SceneFragment::new();
        let fragment_root = fragment
            .add_node(
                None,
                Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
                vec![],
            )
            .unwrap();
        let fragment_child = fragment
            .add_node(
                Some(fragment_root),
                Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)),
                vec![MeshHandle::new(99, 0)],
            )
            .unwrap();

        let mount = scene.merge_fragment(Some(parent), fragment).unwrap();
        assert_eq!(mount.node_mapping.len(), 2);

        let mounted_root = mount.node_mapping[&fragment_root];
        let mounted_child = mount.node_mapping[&fragment_child];

        assert_eq!(mount.mounted_root, mounted_root);
        assert_eq!(
            scene.transform(mounted_root).unwrap(),
            Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
        );
        assert_eq!(
            scene.transform(mounted_child).unwrap(),
            Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0))
        );
    }

    #[test]
    fn merge_fragment_requires_all_nodes_reachable_from_root() {
        let mut scene = Scene::new();

        let mut fragment = SceneFragment::new();
        fragment
            .add_node_default(None)
            .expect("fragment root should be added");
        fragment.nodes.push(SceneFragmentNode::default());

        let result = scene.merge_fragment(None, fragment);
        assert!(matches!(result, Err(SceneError::MergeFailed(_))));
    }

    #[test]
    fn point_light_create_update_remove() {
        let mut scene = Scene::new();

        let light = PointLight {
            position: Vec3::new(0.0, 10.0, 0.0),
            color: Vec3::new(1.0, 0.8, 0.6),
            intensity: 50.0,
            range: 10.0,
        };

        // Create light
        let id = scene
            .create_point_light(light)
            .expect("create should succeed");

        // Update light
        let updated = PointLight {
            position: Vec3::new(5.0, 10.0, 5.0),
            color: Vec3::new(0.5, 0.5, 1.0),
            intensity: 100.0,
            range: 15.0,
        };
        scene
            .update_point_light(id, updated)
            .expect("update should succeed");

        // Remove light
        scene.remove_point_light(id).expect("remove should succeed");

        // Stale handle should be rejected
        let result = scene.update_point_light(id, light);
        assert!(matches!(result, Err(SceneError::StalePointLight(_))));
    }

    #[test]
    fn point_light_validation_rejects_invalid_range() {
        let mut scene = Scene::new();

        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 50.0,
            range: 0.0, // Invalid: must be > 0
        };

        let result = scene.create_point_light(light);
        assert!(matches!(result, Err(SceneError::InvalidPointLight(_))));
    }

    #[test]
    fn point_light_validation_rejects_negative_intensity() {
        let mut scene = Scene::new();

        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: -10.0, // Invalid: must be >= 0
            range: 5.0,
        };

        let result = scene.create_point_light(light);
        assert!(matches!(result, Err(SceneError::InvalidPointLight(_))));
    }

    #[test]
    fn point_light_validation_clamps_negative_color() {
        let mut scene = Scene::new();

        let light = PointLight {
            position: Vec3::ZERO,
            color: Vec3::new(-1.0, 0.5, 1.0), // Negative component should be clamped
            intensity: 50.0,
            range: 5.0,
        };

        // Should succeed - negative colors are clamped to zero
        let id = scene
            .create_point_light(light)
            .expect("should clamp and succeed");
        scene.remove_point_light(id).unwrap();
    }

    #[test]
    fn scene_persistence_round_trips_nested_assets_environment_lights_and_overrides() {
        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();
        scene.set_node_name(root, "Room Root").unwrap();
        let child = scene
            .create_node(Some(root), Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)))
            .unwrap();
        scene.set_node_name(child, "North Wall").unwrap();
        scene
            .set_node_asset_reference(
                child,
                SceneAssetReference::new(
                    "core.wall.stone_2m",
                    Some(PathBuf::from("prefabs/wall_stone_2m.glb")),
                ),
            )
            .unwrap();
        scene
            .set_node_material_override(child, "0", "mat_override.damp_stone")
            .unwrap();
        let grandchild = scene
            .create_node(
                Some(child),
                Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            )
            .unwrap();
        scene.set_node_name(grandchild, "Torch Mount").unwrap();
        scene
            .create_point_light(PointLight {
                position: Vec3::new(0.0, 2.0, 0.0),
                color: Vec3::new(1.0, 0.8, 0.5),
                intensity: 8.0,
                range: 12.0,
            })
            .unwrap();
        scene
            .set_skybox_asset_reference(
                EnvironmentHandle::new(9, 0),
                SceneAssetReference::new(
                    "core.env.indoor_4k",
                    Some(PathBuf::from("sky_maps/indoor_4k.exr")),
                ),
            )
            .unwrap();

        let serialized = SerializedScene::from_scene(&scene);
        let json = serde_json::to_string_pretty(&serialized).unwrap();
        assert!(json.contains("\"format_version\": 1"));
        assert!(json.contains("\"id\": \"core.wall.stone_2m\""));
        assert!(json.contains("\"id\": \"core.env.indoor_4k\""));
        assert!(json.contains("\"mat_override.damp_stone\""));
        assert!(!json.contains("model_path"));
        assert!(!json.contains("skybox_path"));
        assert!(!json.contains("mesh_handle"));
        assert!(!json.contains("generation"));

        let parsed: SerializedScene = serde_json::from_str(&json).unwrap();
        let mut loader = FakeSceneAssetLoader::default();
        let loaded = parsed.into_scene_with_loader(&mut loader).unwrap();
        assert_eq!(
            loader.loaded_models,
            vec![SceneAssetReference::new(
                "core.wall.stone_2m",
                Some(PathBuf::from("prefabs/wall_stone_2m.glb"))
            )]
        );
        assert_eq!(
            loader.loaded_environments,
            vec![SceneAssetReference::new(
                "core.env.indoor_4k",
                Some(PathBuf::from("sky_maps/indoor_4k.exr"))
            )]
        );

        let round_tripped = SerializedScene::from_scene(&loaded);
        let root_node = round_tripped
            .nodes
            .iter()
            .find(|node| node.name == "Room Root")
            .expect("root node");
        let wall_node = round_tripped
            .nodes
            .iter()
            .find(|node| node.name == "North Wall")
            .expect("wall node");
        let torch_node = round_tripped
            .nodes
            .iter()
            .find(|node| node.name == "Torch Mount")
            .expect("torch node");

        assert_eq!(round_tripped.root_nodes, vec![root_node.id.clone()]);
        assert_eq!(wall_node.parent.as_deref(), Some(root_node.id.as_str()));
        assert_eq!(torch_node.parent.as_deref(), Some(wall_node.id.as_str()));
        assert_eq!(
            wall_node
                .asset
                .as_ref()
                .and_then(|asset| asset.id.as_deref()),
            Some("core.wall.stone_2m")
        );
        assert_eq!(
            wall_node.material_overrides.get("0").map(String::as_str),
            Some("mat_override.damp_stone")
        );
        assert_eq!(round_tripped.lights.len(), 1);
        assert_eq!(
            round_tripped
                .environment
                .as_ref()
                .and_then(|env| env.asset.id.as_deref()),
            Some("core.env.indoor_4k")
        );
    }

    #[test]
    fn scene_persistence_round_trips_placed_wall_chunk_asset_reference() {
        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();
        scene.set_node_name(root, "Room Root").unwrap();
        let mut fragment = SceneFragment::new();
        fragment
            .add_node(None, Mat4::IDENTITY, vec![MeshHandle::new(11, 0)])
            .unwrap();
        let mut history = CommandHistory::new(8);

        let result = scene
            .execute_command(
                &mut history,
                Box::new(PlaceAssetCommand::new(
                    Some(root),
                    Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0)),
                    fragment,
                    SceneAssetReference::new(
                        "editor_sample.wall.stone_2m",
                        Some(PathBuf::from("prefabs/wall_straight_2m.obj")),
                    ),
                    "Stone Wall 2m",
                    vec!["wall".to_string(), "chunk".to_string()],
                    "node.placed.wall.000001",
                )),
            )
            .unwrap();
        let placed = result.created_node.expect("placement node");
        assert_eq!(
            scene.node_stable_id(placed).unwrap().as_deref(),
            Some("node.placed.wall.000001")
        );

        let serialized = SerializedScene::from_scene(&scene);
        let json = serde_json::to_string_pretty(&serialized).unwrap();
        assert!(json.contains("\"id\": \"editor_sample.wall.stone_2m\""));
        assert!(json.contains("\"wall\""));
        assert!(!json.contains("mesh_handle"));

        let parsed: SerializedScene = serde_json::from_str(&json).unwrap();
        let mut loader = FakeSceneAssetLoader::default();
        let loaded = parsed.into_scene_with_loader(&mut loader).unwrap();
        assert_eq!(
            loader.loaded_models,
            vec![SceneAssetReference::new(
                "editor_sample.wall.stone_2m",
                Some(PathBuf::from("prefabs/wall_straight_2m.obj"))
            )]
        );

        let round_tripped = SerializedScene::from_scene(&loaded);
        let wall_node = round_tripped
            .nodes
            .iter()
            .find(|node| node.name == "Stone Wall 2m")
            .expect("wall node");
        assert_eq!(
            wall_node
                .asset
                .as_ref()
                .and_then(|asset| asset.id.as_deref()),
            Some("editor_sample.wall.stone_2m")
        );
        assert!(wall_node.tags.iter().any(|tag| tag == "wall"));
    }

    #[test]
    fn inspector_metadata_edits_round_trip_name_tags_and_material_override() {
        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();
        scene.set_node_name(root, "Editable Root").unwrap();
        scene
            .set_node_tags(root, vec![" gameplay ".to_string(), "wall".to_string()])
            .unwrap();
        scene
            .set_node_material_override(root, "0", "mat_override.editor_slot_0")
            .unwrap();

        let serialized = SerializedScene::from_scene(&scene);
        let json = serde_json::to_string_pretty(&serialized).unwrap();
        assert!(json.contains("\"Editable Root\""));
        assert!(json.contains("\"gameplay\""));
        assert!(json.contains("\"mat_override.editor_slot_0\""));

        let parsed: SerializedScene = serde_json::from_str(&json).unwrap();
        let mut loader = FakeSceneAssetLoader::default();
        let loaded = parsed.into_scene_with_loader(&mut loader).unwrap();
        let round_tripped = SerializedScene::from_scene(&loaded);
        let node = round_tripped
            .nodes
            .iter()
            .find(|node| node.name == "Editable Root")
            .expect("editable node");

        assert!(node.tags.iter().any(|tag| tag == "gameplay"));
        assert_eq!(
            node.material_overrides.get("0").map(String::as_str),
            Some("mat_override.editor_slot_0")
        );
    }

    #[test]
    fn inspector_material_override_rejects_empty_slot_or_override_id() {
        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();

        assert!(scene
            .set_node_material_override(root, "", "mat_override.editor")
            .is_err());
        assert!(scene.set_node_material_override(root, "0", "   ").is_err());
        scene
            .set_node_material_override(root, "0", "mat_override.editor")
            .unwrap();
        scene.clear_node_material_override(root, "0").unwrap();
        let serialized = SerializedScene::from_scene(&scene);
        assert!(serialized.nodes[0].material_overrides.is_empty());
    }

    #[test]
    fn scene_persistence_rejects_malformed_scene_documents() {
        assert_scene_load_error(
            r#"{
                "format_version": 2,
                "scene_id": "scene.bad",
                "root_nodes": [],
                "nodes": [],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| {
                matches!(
                    err,
                    RendererError::Scene(SceneError::UnsupportedSceneVersion { .. })
                )
            },
        );

        assert_scene_load_error(
            r#"{
                "format_version": 1,
                "scene_id": "scene.duplicate",
                "root_nodes": ["node.a"],
                "nodes": [
                    {"id":"node.a","parent":null,"name":"A","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null},
                    {"id":"node.a","parent":null,"name":"A2","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| matches!(err, RendererError::Scene(SceneError::DuplicateSerializedNodeId(id)) if id == "node.a"),
        );

        assert_scene_load_error(
            r#"{
                "format_version": 1,
                "scene_id": "scene.bad_parent",
                "root_nodes": ["node.child"],
                "nodes": [
                    {"id":"node.child","parent":"node.missing","name":"Child","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| matches!(err, RendererError::Scene(SceneError::BadSerializedParent { node_id, parent_id }) if node_id == "node.child" && parent_id == "node.missing"),
        );

        assert_scene_load_error(
            r#"{
                "format_version": 1,
                "scene_id": "scene.cycle",
                "root_nodes": ["node.a"],
                "nodes": [
                    {"id":"node.a","parent":"node.b","name":"A","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null},
                    {"id":"node.b","parent":"node.a","name":"B","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| matches!(err, RendererError::Scene(SceneError::CycleDetected)),
        );

        assert_scene_load_error(
            r#"{
                "format_version": 1,
                "scene_id": "scene.disconnected",
                "root_nodes": ["node.root"],
                "nodes": [
                    {"id":"node.root","parent":null,"name":"Root","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null},
                    {"id":"node.orphan","parent":null,"name":"Orphan","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| matches!(err, RendererError::Scene(SceneError::DisconnectedGraph(_))),
        );

        assert_scene_load_error(
            r#"{
                "format_version": 1,
                "scene_id": "scene.missing_asset",
                "root_nodes": ["node.asset"],
                "nodes": [
                    {"id":"node.asset","parent":null,"name":"Asset","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":{"path_hint":"models/crate.glb"}}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            |err| matches!(err, RendererError::Scene(SceneError::MissingAssetId(context)) if context.contains("node.asset")),
        );
    }

    fn assert_scene_load_error(json: &str, predicate: impl FnOnce(RendererError) -> bool) {
        let parsed: SerializedScene = serde_json::from_str(json).unwrap();
        let mut loader = FakeSceneAssetLoader::default();
        let err = match parsed.into_scene_with_loader(&mut loader) {
            Ok(_) => panic!("scene should fail validation"),
            Err(err) => err,
        };
        assert!(predicate(err));
    }

    #[derive(Default)]
    struct FakeSceneAssetLoader {
        loaded_models: Vec<SceneAssetReference>,
        loaded_environments: Vec<SceneAssetReference>,
    }

    impl SceneAssetLoader for FakeSceneAssetLoader {
        fn load_model_ref(
            &mut self,
            asset: &SceneAssetReference,
        ) -> Result<SceneFragment, AssetError> {
            self.loaded_models.push(asset.clone());
            let mut fragment = SceneFragment::new();
            fragment
                .add_node(None, Mat4::IDENTITY, vec![MeshHandle::new(7, 0)])
                .expect("fake fragment root");
            Ok(fragment)
        }

        fn load_environment_ref(
            &mut self,
            asset: &SceneAssetReference,
        ) -> Result<EnvironmentHandle, AssetError> {
            self.loaded_environments.push(asset.clone());
            Ok(EnvironmentHandle::new(9, 0))
        }
    }
}
