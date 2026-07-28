use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::api::assets::{AssetManager, EnvironmentSource};
use crate::data::handles::{EnvironmentHandle, MeshHandle};
use crate::data::validation::{ValidationArea, ValidationDiagnostic, ValidationError};
use crate::object::identity::ObjectId;
use crate::scene::command::{Command, CommandHistory, CommandResult};
use crate::scene::object_store::{
    mint_persistent_id, ObjectHandle, ObjectRecord,
};
use crate::scene::render_submission::{
    RenderSubmission, MAX_DIRECTIONAL_LIGHTS_GPU, MAX_POINT_LIGHTS_GPU, MAX_SPOT_LIGHTS_GPU,
};
use crate::scene::scene_world::{
    DirectionalLightRefError, PointLightRefError, ReparentError, SceneNodeRefError, SceneWorld,
    SpotLightRefError,
};
use crate::scene::SceneNodeId;
use engine_events::SceneObjectId;

use crate::data::camera::Aabb;
use crate::data::mesh_geometry::{MeshDeformation, MeshGeometryDto};

use super::errors::SceneError;

pub const SCENE_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Default)]
pub struct SceneValidationOptions {
    pub known_asset_ids: Option<HashSet<String>>,
}

impl SceneValidationOptions {
    pub fn with_known_asset_ids<I, S>(mut self, known_asset_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.known_asset_ids = Some(known_asset_ids.into_iter().map(Into::into).collect());
        self
    }
}

// ---------------------------------------------------------------------------
// Scene bounds types
// ---------------------------------------------------------------------------

/// Reason a node or mesh has no trusted conservative bound for culling/pruning.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundsUnknownReason {
    /// No geometry DTO was registered (orphan handle or upload failure).
    MissingGeometry,
    /// The handle generation no longer matches the registered geometry.
    StaleHandle,
    /// Geometry is skinned; bind-pose AABB is not a valid bound.
    Skinned,
    /// Geometry is procedurally or morph-target deformed.
    Deformed,
    /// The stored AABB is non-finite or invalid.
    InvalidGeometry,
}

/// Bounding volume state for one mesh or one scene node.
///
/// - `Known(Aabb)`: authoritative bound computed from trusted geometry.
/// - `Proxy(Aabb)`: explicit stand-in bound for a node whose geometry is
///   intentionally unavailable.
/// - `ConservativeVisible(reason)`: bound is unreliable; culling must treat
///   the node as always visible.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SceneBounds {
    Known(Aabb),
    Proxy(Aabb),
    ConservativeVisible(BoundsUnknownReason),
}

impl SceneBounds {
    /// Return the inner `Aabb` when known or proxy, otherwise `None`.
    pub fn aabb(&self) -> Option<&Aabb> {
        match self {
            SceneBounds::Known(a) | SceneBounds::Proxy(a) => Some(a),
            SceneBounds::ConservativeVisible(_) => None,
        }
    }

    /// True when this bound may be used for safe subtree pruning.
    pub fn is_trusted_for_pruning(&self) -> bool {
        matches!(self, SceneBounds::Known(_) | SceneBounds::Proxy(_))
    }

    /// True when the bound is conservative-visible (always visible, never pruned).
    pub fn is_conservative_visible(&self) -> bool {
        matches!(self, SceneBounds::ConservativeVisible(_))
    }
}

/// Convert a [`MeshGeometryDto`] to a [`SceneBounds`].
/// Rigid meshes with valid AABBs become `Known`; skinned/deformed/unknown
/// classification becomes `ConservativeVisible` with the exact reason.
pub(crate) fn scene_bounds_from_dto(dto: &MeshGeometryDto) -> SceneBounds {
    match dto.deformation {
        MeshDeformation::Rigid => {
            if let Some(ref local_aabb) = dto.local_aabb {
                if local_aabb.is_valid() {
                    let min = glam::Vec3::from_array(local_aabb.min);
                    let max = glam::Vec3::from_array(local_aabb.max);
                    return SceneBounds::Known(Aabb::from_min_max(min, max));
                }
            }
            SceneBounds::ConservativeVisible(BoundsUnknownReason::InvalidGeometry)
        }
        MeshDeformation::Skinned => SceneBounds::ConservativeVisible(BoundsUnknownReason::Skinned),
        MeshDeformation::Deformed => {
            SceneBounds::ConservativeVisible(BoundsUnknownReason::Deformed)
        }
        MeshDeformation::Unknown => {
            SceneBounds::ConservativeVisible(BoundsUnknownReason::MissingGeometry)
        }
    }
}

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

/// Directional light ID with slot+generation semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DirectionalLightId {
    pub slot: u32,
    pub generation: u32,
}

/// Directional light definition.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectionalLight {
    /// World-space direction from a shaded surface toward the light source.
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl DirectionalLight {
    /// Validate directional light parameters.
    fn validate(&self) -> Result<(), SceneError> {
        if !self.direction.is_finite() || self.direction.length_squared() < 1e-6 {
            return Err(SceneError::InvalidDirectionalLight(
                "direction must be finite and non-zero".to_string(),
            ));
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(SceneError::InvalidDirectionalLight(
                "intensity must be finite and >= 0.0".to_string(),
            ));
        }
        if !self.color.is_finite() {
            return Err(SceneError::InvalidDirectionalLight(
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

/// Point light ID with slot+generation semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PointLightId {
    pub slot: u32,
    pub generation: u32,
}

/// Spot light ID with slot+generation semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SpotLightId {
    pub slot: u32,
    pub generation: u32,
}

/// Shadow configuration for a directional light.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DirectionalShadowConfig {
    pub enabled: bool,
    pub shadow_map_size: u32,
    pub cascade_count: u32,
    pub cascade_split_lambda: f32,
}

impl Default for DirectionalShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            shadow_map_size: 2048,
            cascade_count: 1,
            cascade_split_lambda: 0.5,
        }
    }
}

/// Serialized form of [`DirectionalShadowConfig`] for scene persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedDirectionalShadowConfig {
    enabled: bool,
    #[serde(default = "default_shadow_map_size")]
    shadow_map_size: u32,
    #[serde(default = "default_cascade_count")]
    cascade_count: u32,
    #[serde(default = "default_cascade_split_lambda")]
    cascade_split_lambda: f32,
}

fn default_shadow_map_size() -> u32 {
    2048
}
fn default_cascade_count() -> u32 {
    1
}
fn default_cascade_split_lambda() -> f32 {
    0.5
}

impl From<DirectionalShadowConfig> for SerializedDirectionalShadowConfig {
    fn from(cfg: DirectionalShadowConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            shadow_map_size: cfg.shadow_map_size,
            cascade_count: cfg.cascade_count,
            cascade_split_lambda: cfg.cascade_split_lambda,
        }
    }
}

impl From<SerializedDirectionalShadowConfig> for DirectionalShadowConfig {
    fn from(s: SerializedDirectionalShadowConfig) -> Self {
        Self {
            enabled: s.enabled,
            shadow_map_size: s.shadow_map_size,
            cascade_count: s.cascade_count,
            cascade_split_lambda: s.cascade_split_lambda,
        }
    }
}

/// Spot light definition.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpotLight {
    pub position: Vec3,
    /// World-space direction from light origin toward the lit scene.
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
}

impl SpotLight {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        inner: f32,
        outer: f32,
    ) -> Self {
        Self {
            position,
            direction,
            color,
            intensity,
            range,
            inner_cone_angle: inner,
            outer_cone_angle: outer,
        }
    }

    fn validate(&self) -> Result<(), SceneError> {
        if !self.position.is_finite() {
            return Err(SceneError::InvalidSpotLight(
                "position must be finite".into(),
            ));
        }
        if !self.direction.is_finite() || self.direction.length_squared() < 1e-6 {
            return Err(SceneError::InvalidSpotLight(
                "direction must be finite and non-zero".into(),
            ));
        }
        if !self.range.is_finite() || self.range <= 0.0 {
            return Err(SceneError::InvalidSpotLight(
                "range must be finite and > 0.0".into(),
            ));
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(SceneError::InvalidSpotLight(
                "intensity must be finite and >= 0.0".into(),
            ));
        }
        if !self.color.is_finite() {
            return Err(SceneError::InvalidSpotLight("color must be finite".into()));
        }
        if !self.inner_cone_angle.is_finite()
            || self.inner_cone_angle < 0.0
            || self.inner_cone_angle > std::f32::consts::PI
        {
            return Err(SceneError::InvalidSpotLight(
                "inner_cone_angle must be in [0, PI]".into(),
            ));
        }
        if !self.outer_cone_angle.is_finite()
            || self.outer_cone_angle < 0.0
            || self.outer_cone_angle > std::f32::consts::PI
        {
            return Err(SceneError::InvalidSpotLight(
                "outer_cone_angle must be in [0, PI]".into(),
            ));
        }
        if self.inner_cone_angle > self.outer_cone_angle {
            return Err(SceneError::InvalidSpotLight(
                "inner_cone_angle must be <= outer_cone_angle".into(),
            ));
        }
        Ok(())
    }

    fn sanitize_color(&self) -> Vec3 {
        self.color.max(Vec3::ZERO)
    }
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
        if !self.position.is_finite() {
            return Err(SceneError::InvalidPointLight(
                "position must be finite".to_string(),
            ));
        }
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

/// Per-mesh bounds record within a scene fragment or scene node.
#[derive(Clone, Debug)]
pub struct MeshBoundsEntry {
    pub mesh: MeshHandle,
    pub bounds: SceneBounds,
}

/// One node in a detached scene fragment.
#[derive(Clone, Debug)]
pub struct SceneFragmentNode {
    pub parent: Option<SceneFragmentNodeId>,
    pub children: Vec<SceneFragmentNodeId>,
    pub local_transform: Mat4,
    pub meshes: Vec<MeshHandle>,
    pub mesh_bounds: Vec<MeshBoundsEntry>,
}

impl Default for SceneFragmentNode {
    fn default() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            local_transform: Mat4::IDENTITY,
            meshes: Vec::new(),
            mesh_bounds: Vec::new(),
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
        self.add_node_with_bounds(parent, transform, meshes, Vec::new())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn add_node_with_bounds(
        &mut self,
        parent: Option<SceneFragmentNodeId>,
        transform: Mat4,
        meshes: Vec<MeshHandle>,
        mesh_bounds: Vec<MeshBoundsEntry>,
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
            mesh_bounds,
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

pub fn validate_scene_str(content: &str) -> Result<(), ValidationError> {
    validate_scene_str_with_options(content, &SceneValidationOptions::default())
}

pub fn validate_scene_str_with_options(
    content: &str,
    options: &SceneValidationOptions,
) -> Result<(), ValidationError> {
    validate_scene_content(content, None, options)
}

pub fn validate_scene_file(path: impl AsRef<Path>) -> Result<(), ValidationError> {
    validate_scene_file_with_options(path, &SceneValidationOptions::default())
}

pub fn validate_scene_file_with_options(
    path: impl AsRef<Path>,
    options: &SceneValidationOptions,
) -> Result<(), ValidationError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "scene.io",
                ValidationArea::Scene,
                format!("failed to read scene file: {err}"),
            )
            .with_path(path),
        )
    })?;
    validate_scene_content(&content, Some(path), options)
}

/// Public scene facade.
pub struct Scene {
    world: SceneWorld,
    scene_id: String,
    display_name: Option<String>,
    next_stable_node_id: u64,
    skybox_asset: Option<SceneAssetReference>,
    materials: BTreeMap<String, SerializedMaterialOverride>,
    audio: Vec<SerializedAudioReference>,
    /// BSP source-link metadata (if a BSP mount provides it).
    ///
    /// Stored as a raw JSON value so the renderer does not depend on
    /// the `bsp_runtime` crate. The coordinator owns the typed form.
    bsp_source_link: Option<serde_json::Value>,
    /// Editor-specific metadata blob.
    ///
    /// This field is preserved for serialization compatibility but is not
    /// part of the stable public API. Use [`Scene::editor_metadata`] and
    /// [`Scene::set_editor_metadata`] to access this data.
    #[deprecated(
        since = "0.13.0",
        note = "use editor_metadata() / set_editor_metadata() accessors"
    )]
    #[doc(hidden)]
    pub editor: serde_json::Value,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    /// Thread: Any
    /// May Stall: No
    #[allow(deprecated)]
    pub fn new() -> Self {
        Self {
            world: SceneWorld::new(),
            scene_id: "scene.untitled".to_string(),
            display_name: None,
            next_stable_node_id: 1,
            skybox_asset: None,
            materials: BTreeMap::new(),
            audio: Vec::new(),
            bsp_source_link: None,
            editor: serde_json::json!({}),
        }
    }

    /// Returns the BSP source-link metadata, if set by the coordinator.
    pub fn bsp_source_link(&self) -> Option<&serde_json::Value> {
        self.bsp_source_link.as_ref()
    }

    /// Set the BSP source-link metadata from the coordinator.
    pub fn set_bsp_source_link(&mut self, link: serde_json::Value) {
        self.bsp_source_link = Some(link);
    }

    /// Clear the BSP source-link metadata.
    pub fn clear_bsp_source_link(&mut self) {
        self.bsp_source_link = None;
    }

    /// Return the number of point lights that can still be created before the
    /// renderer's fixed GPU point-light cap is reached.
    pub fn available_point_light_slots(&self) -> usize {
        MAX_POINT_LIGHTS_GPU.saturating_sub(self.world.active_point_light_count())
    }

    /// Reserve storage so a later prevalidated point-light publication can add
    /// `total_slots` point lights without growing the scene light slot table.
    ///
    /// Thread: Any
    /// May Stall: Yes
    pub fn reserve_point_light_storage(&mut self, total_slots: usize) {
        self.world.reserve_point_light_slots(total_slots);
    }

    /// Returns a reference to the editor metadata blob.
    pub fn editor_metadata(&self) -> &serde_json::Value {
        #[allow(deprecated)]
        &self.editor
    }

    /// Sets the editor metadata blob.
    pub fn set_editor_metadata(&mut self, metadata: serde_json::Value) {
        #[allow(deprecated)]
        {
            self.editor = metadata;
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
        if !transform.is_finite() {
            return Err(SceneError::InvalidMutation(
                "transform must contain only finite values".to_string(),
            ));
        }
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
        if !transform.is_finite() {
            return Err(SceneError::InvalidMutation(
                "transform must contain only finite values".to_string(),
            ));
        }
        node_ref.local_transform = transform;
        self.world.invalidate_derived_state(node);

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

    /// Returns the material parameters stored in scene-level materials for the given override ID.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn material_parameters(
        &self,
        override_id: &str,
    ) -> Option<&BTreeMap<String, serde_json::Value>> {
        self.materials
            .get(override_id)
            .map(|entry| &entry.parameters)
    }

    /// Sets material parameters for a given override ID in the scene-level materials map.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn set_material_parameters(
        &mut self,
        override_id: String,
        parameters: BTreeMap<String, serde_json::Value>,
    ) {
        self.materials
            .entry(override_id)
            .or_insert_with(|| SerializedMaterialOverride {
                base: None,
                parameters: BTreeMap::new(),
            })
            .parameters = parameters;
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
        // Bounds unknown until explicitly provided.
        node_ref.mesh_bounds.push(MeshBoundsEntry {
            mesh,
            bounds: SceneBounds::ConservativeVisible(BoundsUnknownReason::MissingGeometry),
        });
        self.world.invalidate_derived_state(node);

        Ok(())
    }

    /// Attach a mesh to a node with an explicit bound computed from trusted
    /// geometry. The bound must be model-space, consistent with the DTO.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn add_mesh_with_bounds(
        &mut self,
        node: SceneNodeId,
        mesh: MeshHandle,
        bounds: SceneBounds,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;
        if let SceneBounds::Known(aabb) = &bounds {
            if !aabb.is_finite() || !aabb.is_ordered() {
                return Err(SceneError::MergeFailed(
                    "Known scene bounds must be finite and ordered".to_string(),
                ));
            }
        }
        if let SceneBounds::Proxy(aabb) = &bounds {
            if !aabb.is_finite() || !aabb.is_ordered() {
                return Err(SceneError::MergeFailed(
                    "Proxy scene bounds must be finite and ordered".to_string(),
                ));
            }
        }

        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        node_ref.meshes.push(mesh);
        node_ref.mesh_bounds.push(MeshBoundsEntry { mesh, bounds });
        self.world.invalidate_derived_state(node);

        Ok(())
    }

    /// Set an explicit proxy bound for a node whose geometry is intentionally
    /// unavailable. A proxy does not override an existing known mesh bound.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn set_node_proxy_bounds(
        &mut self,
        node: SceneNodeId,
        local_aabb: Aabb,
    ) -> Result<(), SceneError> {
        self.validate_node(node)?;
        if !local_aabb.is_finite() || !local_aabb.is_ordered() {
            return Err(SceneError::MergeFailed(
                "proxy bounds must be finite and ordered".to_string(),
            ));
        }
        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        // Only set proxy when no geometry exists or all mesh bounds are known
        // missing. Skinned, deformed, stale, invalid, known, and existing proxy
        // mesh bounds remain conservatively governed by their own state.
        if node_ref.mesh_bounds.iter().any(|entry| {
            !matches!(
                entry.bounds,
                SceneBounds::ConservativeVisible(BoundsUnknownReason::MissingGeometry)
            )
        }) {
            return Err(SceneError::MergeFailed(
                "cannot set proxy bounds when mesh bounds are known, stale, skinned, deformed, invalid, or already proxied".to_string(),
            ));
        }
        node_ref.local_proxy_bounds = Some(local_aabb);
        self.world.invalidate_derived_state(node);
        Ok(())
    }

    /// Clear any explicit proxy bounds on a node.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn clear_node_proxy_bounds(&mut self, node: SceneNodeId) -> Result<(), SceneError> {
        self.validate_node(node)?;
        let Some(node_ref) = self.world.get_node_mut(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        if node_ref.local_proxy_bounds.take().is_some() {
            self.world.invalidate_derived_state(node);
        }
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
        node_ref.mesh_bounds.clear();
        node_ref.local_proxy_bounds = None;
        self.world.invalidate_derived_state(node);

        Ok(())
    }

    /// Thread: Any
    /// May Stall: No
    pub fn set_camera(&mut self, view: Mat4, projection: Mat4, position: Vec3) {
        self.world.update_camera(view, projection, position);
    }

    /// Enable or disable frustum culling. It is enabled by default. When
    /// enabled, mesh-backed nodes whose transform-aware proxy AABB is outside
    /// the camera frustum are skipped during submission, reducing GPU draws.
    /// Descendants are tested independently.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn set_frustum_culling(&mut self, enabled: bool) {
        self.world.enable_frustum_culling = enabled;
    }

    /// Returns whether frustum culling is currently enabled.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn frustum_culling_enabled(&self) -> bool {
        self.world.enable_frustum_culling
    }

    /// Thread: Any
    /// May Stall: No
    pub fn has_skybox(&self) -> bool {
        self.world.skybox_env_id() != EnvironmentHandle::new(0, 0)
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

    // ── BSP mount ────────────────────────────────────────────────────

    /// Mount a prepared BSP into the scene for PVS-aware rendering.
    ///
    /// While a BSP mount is active, imported BSP lights are selected via
    /// PVS-filtered deterministic scoring, and BSP static geometry is
    /// PVS-culled before frustum culling. Replacing an existing mount detaches
    /// it from scene publication before publishing the new lease.
    ///
    /// The caller must retire the returned [`DetachedBspMount`] through the
    /// renderer to complete fence-aware GPU resource teardown.
    ///
    /// Thread: Any
    /// May Stall: No
    #[cfg(feature = "bsp")]
    pub fn set_bsp_mount(
        &mut self,
        mount: crate::api::bsp::PreparedBspMount,
    ) -> Option<crate::api::bsp::DetachedBspMount> {
        let detached = self.retire_bsp_mount();
        self.world.set_bsp_mount(mount.into_published());
        detached
    }

    /// Retire the currently published BSP mount.
    ///
    /// The returned [`DetachedBspMount`] retains the full resource lease.
    /// The caller must pass it to the renderer's retirement path for
    /// fence-aware GPU teardown. Dropping the detached mount leaks GPU
    /// resources.
    #[cfg(feature = "bsp")]
    #[must_use = "a detached BSP mount must be retired through the renderer"]
    pub fn retire_bsp_mount(&mut self) -> Option<crate::api::bsp::DetachedBspMount> {
        self.world.retire_bsp_mount()
    }

    /// Return whether this scene currently publishes a BSP mount.
    #[cfg(feature = "bsp")]
    pub fn has_bsp_mount(&self) -> bool {
        self.world.has_bsp_mount()
    }

    /// Phase 07: Set a pending BSP evidence request for the next submission build.
    ///
    /// The request is consumed during [`Scene::build_submission`]. Only call this
    /// immediately before rendering when an evidence request is active.
    #[cfg(feature = "bsp")]
    pub(crate) fn set_bsp_evidence_request(
        &mut self,
        corpus_identity: String,
        request_identity: String,
        visibility: crate::api::bsp::BspEvidenceVisibility,
        frame_number: u32,
    ) {
        self.world.set_bsp_evidence_request(corpus_identity, request_identity, visibility, frame_number);
    }

    /// Clear the BSP mount, returning to non-BSP rendering.
    ///
    /// This compatibility helper uses the same scene-detachment boundary as
    /// an explicit unload. [`Scene::retire_bsp_mount`] acknowledges removal
    /// from submission, not queueing of GPU teardown.
    ///
    /// Thread: Any
    /// May Stall: No
    #[cfg(feature = "bsp")]
    pub fn clear_bsp_mount(&mut self) {
        let _retired = self.retire_bsp_mount();
    }

    /// Replace BSP-owned point lights after transaction preflight.
    ///
    /// The coordinator validates all input before commit. This method repeats
    /// the checks before mutating anything, then performs an infallible slot
    /// swap so an error leaves the old light set untouched.
    #[cfg(feature = "bsp")]
    #[doc(hidden)]
    pub fn replace_prevalidated_bsp_point_lights(
        &mut self,
        old_ids: &[PointLightId],
        new_lights: &[PointLight],
    ) -> Result<Vec<PointLightId>, SceneError> {
        for (index, id) in old_ids.iter().enumerate() {
            if old_ids[..index].contains(id) {
                return Err(SceneError::InvalidPointLight(format!(
                    "duplicate BSP-owned point-light id (slot={}, generation={})",
                    id.slot, id.generation
                )));
            }
            self.validate_point_light(*id)?;
        }
        for light in new_lights {
            light.validate()?;
        }

        let remaining = self
            .world
            .active_point_light_count()
            .checked_sub(old_ids.len())
            .ok_or_else(|| {
                SceneError::InvalidPointLight(
                    "BSP-owned point lights exceed the active scene count".to_string(),
                )
            })?;
        if remaining.saturating_add(new_lights.len()) > MAX_POINT_LIGHTS_GPU {
            return Err(SceneError::InvalidPointLight(format!(
                "BSP point-light replacement would exceed cap ({MAX_POINT_LIGHTS_GPU})"
            )));
        }

        for id in old_ids {
            let removed = self.world.remove_point_light(*id);
            debug_assert!(removed, "validated BSP point light must remove");
        }

        Ok(new_lights
            .iter()
            .map(|light| {
                self.world.add_point_light(PointLight {
                    color: light.sanitize_color(),
                    ..*light
                })
            })
            .collect())
    }

    /// Set per-frame BSP frame values (light-style intensities, liquid time).
    ///
    /// These are consumed by the BSP frame-values UBO each rendered frame.
    /// `style_intensities` is a 64-element array mapping style_id → intensity (0.0..1.0).
    /// `liquid_time` is the accumulated liquid animation time in seconds.
    ///
    /// Thread: Any
    /// May Stall: No
    #[cfg(feature = "bsp")]
    pub fn set_bsp_frame_values(&mut self, style_intensities: [f32; 64], liquid_time: f32) {
        self.world
            .set_bsp_frame_values(style_intensities, liquid_time);
    }

    /// Set per-model transforms for inline model BSP draws.
    ///
    /// `transforms` maps model_index (1..n) to world-space transform.
    /// Model 0 (worldspawn) is always identity and need not be included.
    /// These are applied as per-draw transforms for inline model batches.
    ///
    /// Thread: Any
    /// May Stall: No
    #[cfg(feature = "bsp")]
    pub fn set_inline_model_transforms(
        &mut self,
        transforms: std::collections::HashMap<u32, glam::Mat4>,
    ) {
        self.world.set_inline_model_transforms(transforms);
    }

    /// Set per-model world-space bounds for inline model BSP culling.
    ///
    /// Bounds are keyed by model_index (1..n) and come from the same simulation
    /// snapshot as the inline transforms.
    ///
    /// Thread: Any
    /// May Stall: No
    #[cfg(feature = "bsp")]
    pub fn set_inline_model_bounds(
        &mut self,
        bounds: std::collections::HashMap<u32, (glam::Vec3, glam::Vec3)>,
    ) {
        self.world.set_inline_model_bounds(bounds);
    }

    /// Thread: Any
    /// May Stall: No
    pub fn create_point_light(&mut self, light: PointLight) -> Result<PointLightId, SceneError> {
        light.validate()?;
        if self.world.active_point_light_count() >= MAX_POINT_LIGHTS_GPU {
            return Err(SceneError::InvalidPointLight(format!(
                "point-light cap ({MAX_POINT_LIGHTS_GPU}) reached"
            )));
        }
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
    pub fn create_directional_light(
        &mut self,
        light: DirectionalLight,
    ) -> Result<DirectionalLightId, SceneError> {
        light.validate()?;
        if self.world.get_active_directional_light().is_some() {
            return Err(SceneError::InvalidDirectionalLight(
                "the scene already has its single directional light".to_string(),
            ));
        }
        let sanitized = DirectionalLight {
            color: light.sanitize_color(),
            ..light
        };
        let id = self.world.add_directional_light(sanitized);
        // Preserve the pre-CSM zero/one route: in default builds its single
        // directional light continues to cast the legacy shadow automatically.
        #[cfg(not(feature = "csm"))]
        self.world.set_shadow_casting_directional(Some(id));
        Ok(id)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn update_directional_light(
        &mut self,
        id: DirectionalLightId,
        light: DirectionalLight,
    ) -> Result<(), SceneError> {
        light.validate()?;
        self.validate_directional_light(id)?;

        let sanitized = DirectionalLight {
            color: light.sanitize_color(),
            ..light
        };

        if self.world.update_directional_light(id, sanitized) {
            return Ok(());
        }

        Err(SceneError::InvalidDirectionalLight(format!(
            "failed to update directional light (slot={}, generation={})",
            id.slot, id.generation
        )))
    }

    /// Thread: Any
    /// May Stall: No
    pub fn remove_directional_light(&mut self, id: DirectionalLightId) -> Result<(), SceneError> {
        self.validate_directional_light(id)?;

        if self.world.remove_directional_light(id) {
            return Ok(());
        }

        Err(SceneError::InvalidDirectionalLight(format!(
            "failed to remove directional light (slot={}, generation={})",
            id.slot, id.generation
        )))
    }

    /// Returns the scene's directional light, if any.
    pub fn directional_light(&self) -> Option<DirectionalLight> {
        self.world.get_active_directional_light()
    }

    /// Set which directional light (if any) casts shadows.
    /// Only one directional light may be the shadow caster at a time.
    pub fn set_shadow_casting_directional(
        &mut self,
        id: Option<DirectionalLightId>,
    ) -> Result<(), SceneError> {
        if let Some(id) = id {
            self.validate_directional_light(id)?;
        }
        self.world.set_shadow_casting_directional(id);
        Ok(())
    }

    /// Returns the ID of the shadow-casting directional light, if any.
    pub fn shadow_casting_directional_light_id(&self) -> Option<DirectionalLightId> {
        self.world.shadow_casting_directional()
    }

    /// Add a directional light without enforcing the legacy single-light cap.
    /// Multiple directional lights are supported for direct illumination;
    /// shadow casting remains limited to at most one.
    pub fn add_directional_light(
        &mut self,
        light: DirectionalLight,
    ) -> Result<DirectionalLightId, SceneError> {
        light.validate()?;
        if self.world.active_directional_light_count() >= MAX_DIRECTIONAL_LIGHTS_GPU {
            return Err(SceneError::InvalidDirectionalLight(format!(
                "directional-light cap ({MAX_DIRECTIONAL_LIGHTS_GPU}) reached"
            )));
        }
        let sanitized = DirectionalLight {
            color: light.sanitize_color(),
            ..light
        };
        Ok(self.world.add_directional_light(sanitized))
    }

    /// Returns all active directional lights.
    pub fn directional_lights(&self) -> Vec<DirectionalLight> {
        self.world.get_active_directional_lights()
    }

    /// Enable or disable shadows for a directional light.
    /// At most one directional light may cast shadows. Returns
    /// `UnsupportedLightFeature` if a second directional is enabled.
    pub fn set_directional_shadow_config(
        &mut self,
        id: DirectionalLightId,
        config: DirectionalShadowConfig,
    ) -> Result<(), SceneError> {
        self.validate_directional_light(id)?;
        if config.enabled {
            // If a different directional is already the shadow caster, reject.
            if let Some(existing) = self.world.shadow_casting_directional() {
                if existing != id {
                    return Err(SceneError::UnsupportedLightFeature(
                        "at most one directional light may cast shadows at a time".into(),
                    ));
                }
            }
            self.world.set_shadow_casting_directional(Some(id));
        } else if self.world.shadow_casting_directional() == Some(id) {
            self.world.set_shadow_casting_directional(None);
        }
        // Store shadow config in the record.
        if let Some(record) = self.world.get_directional_light_record_mut(id) {
            record.directional_shadow_config = Some(config);
        }
        Ok(())
    }

    /// Configure spot-light shadow intent. Spot shadows are not yet supported;
    /// enabling them returns `UnsupportedLightFeature`.
    pub fn set_spot_light_shadow_config(
        &mut self,
        id: SpotLightId,
        enabled: bool,
    ) -> Result<(), SceneError> {
        self.validate_spot_light(id)?;
        if enabled {
            return Err(SceneError::UnsupportedLightFeature(
                "spot-light shadow rendering is not yet supported".into(),
            ));
        }
        Ok(())
    }

    /// Configure point-light shadow intent. Point shadows are not yet supported;
    /// enabling them returns `UnsupportedLightFeature`.
    pub fn set_point_light_shadow_config(
        &mut self,
        id: PointLightId,
        enabled: bool,
    ) -> Result<(), SceneError> {
        self.validate_point_light(id)?;
        if enabled {
            return Err(SceneError::UnsupportedLightFeature(
                "point-light shadow rendering is not yet supported".into(),
            ));
        }
        Ok(())
    }

    /// Create a spot light.
    pub fn create_spot_light(&mut self, light: SpotLight) -> Result<SpotLightId, SceneError> {
        light.validate()?;
        if self.world.active_spot_light_count() >= MAX_SPOT_LIGHTS_GPU {
            return Err(SceneError::InvalidSpotLight(format!(
                "spot-light cap ({MAX_SPOT_LIGHTS_GPU}) reached"
            )));
        }
        let sanitized = SpotLight {
            color: light.sanitize_color(),
            ..light
        };
        Ok(self.world.add_spot_light(sanitized))
    }

    /// Update a spot light.
    pub fn update_spot_light(
        &mut self,
        id: SpotLightId,
        light: SpotLight,
    ) -> Result<(), SceneError> {
        light.validate()?;
        self.validate_spot_light(id)?;
        let sanitized = SpotLight {
            color: light.sanitize_color(),
            ..light
        };
        if self.world.update_spot_light(id, sanitized) {
            return Ok(());
        }
        Err(SceneError::InvalidSpotLight(format!(
            "failed to update spot light"
        )))
    }

    /// Remove a spot light.
    pub fn remove_spot_light(&mut self, id: SpotLightId) -> Result<(), SceneError> {
        self.validate_spot_light(id)?;
        if self.world.remove_spot_light(id) {
            return Ok(());
        }
        Err(SceneError::InvalidSpotLight(format!(
            "failed to remove spot light"
        )))
    }

    /// Returns all active spot lights.
    pub fn spot_lights(&self) -> Vec<SpotLight> {
        self.world.get_active_spot_lights()
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

            let new_node = self.world.add_node_with_parts_and_bounds(
                scene_parent,
                source.local_transform,
                source.meshes.clone(),
                source.mesh_bounds.clone(),
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
    /// The save is failure-atomic: serialization and validation happen first,
    /// then a staged file is written, flushed, and renamed over the target.
    /// Partial writes or I/O errors are not published.
    ///
    /// Thread: Any
    /// May Stall: Yes (file I/O)
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        let serialized = SerializedScene::from_scene(self);
        let json = serde_json::to_string_pretty(&serialized).map_err(|e| {
            SceneError::SerializationError(format!("scene serialization failed: {e}"))
        })?;

        // Write through a same-directory staged-file owner that keeps the
        // reserved handle until publication and removes only its own stage.
        crate::api::scene_file_tx::save_scene_file(path, json.as_bytes()).map_err(|e| {
            SceneError::SerializationError(format!("failed to publish scene file: {e}"))
        })?;

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
        let serialized = Self::read_serialized_scene(path.as_ref())?;
        serialized.into_scene(assets)
    }

    #[cfg(test)]
    fn load_with_loader(
        path: impl AsRef<std::path::Path>,
        loader: &mut impl SceneAssetLoader,
    ) -> Result<Self, crate::api::errors::RendererError> {
        let serialized = Self::read_serialized_scene(path.as_ref())?;
        serialized.into_scene_with_loader(loader)
    }

    fn read_serialized_scene(
        path: &std::path::Path,
    ) -> Result<SerializedScene, crate::api::errors::RendererError> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            crate::api::errors::RendererError::Init(
                crate::api::errors::RendererInitError::StartupScene(format!(
                    "failed to read scene file: {e}"
                )),
            )
        })?;

        // Detect format version and migrate v1 → v2 before strict deserialization.
        let raw: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
            crate::api::errors::RendererError::Scene(SceneError::SerializationError(format!(
                "scene deserialization failed: {e}"
            )))
        })?;

        let version = raw
            .get("format_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let migrated = if version == 1 {
            migrate_v1_to_v2(raw)
        } else if version == SCENE_FORMAT_VERSION as u64 {
            raw
        } else {
            return Err(crate::api::errors::RendererError::Scene(
                SceneError::UnsupportedSceneVersion {
                    found: version as u32,
                    expected: SCENE_FORMAT_VERSION,
                },
            ));
        };

        let serialized: SerializedScene = serde_json::from_value(migrated).map_err(|e| {
            crate::api::errors::RendererError::Scene(SceneError::SerializationError(format!(
                "scene deserialization failed: {e}"
            )))
        })?;
        Ok(serialized)
    }

    #[allow(deprecated)]
    pub(crate) fn from_world(world: SceneWorld) -> Self {
        Self {
            world,
            scene_id: "scene.imported".to_string(),
            display_name: None,
            next_stable_node_id: 1,
            skybox_asset: None,
            materials: BTreeMap::new(),
            audio: Vec::new(),
            bsp_source_link: None,
            editor: serde_json::json!({}),
        }
    }

    // ── Object identity facade methods ──────────────────────────────────

    /// Returns the [`ObjectId`] for a node, or an error if the node is invalid.
    pub fn object_id(&self, node: SceneNodeId) -> Result<ObjectId, SceneError> {
        self.validate_node(node)?;
        self.world
            .object_id_for_node(node)
            .ok_or(SceneError::InvalidNode(node))
    }

    /// Returns the [`ObjectId`] for a point light.
    pub fn object_id_for_point_light(
        &self,
        id: PointLightId,
    ) -> Result<ObjectId, SceneError> {
        self.validate_point_light(id)?;
        self.world
            .object_id_for_point_light(id)
            .ok_or(SceneError::InvalidPointLight(format!("invalid point light")))
    }

    /// Returns the [`ObjectId`] for a directional light.
    pub fn object_id_for_directional_light(
        &self,
        id: DirectionalLightId,
    ) -> Result<ObjectId, SceneError> {
        self.validate_directional_light(id)?;
        self.world
            .object_id_for_directional_light(id)
            .ok_or(SceneError::InvalidDirectionalLight(format!(
                "invalid directional light"
            )))
    }

    /// Returns the [`ObjectId`] for a spot light.
    pub fn object_id_for_spot_light(
        &self,
        id: SpotLightId,
    ) -> Result<ObjectId, SceneError> {
        self.validate_spot_light(id)?;
        self.world
            .object_id_for_spot_light(id)
            .ok_or(SceneError::InvalidSpotLight(format!(
                "invalid spot light"
            )))
    }

    /// Resolve an [`ObjectId`] back to a typed [`ObjectHandle`].
    pub fn resolve_object(&self, id: ObjectId) -> Option<ObjectHandle> {
        self.world.resolve_object(id)
    }

    /// Access the underlying [`SceneWorld`] for audit operations.
    pub fn world(&self) -> &SceneWorld {
        &self.world
    }

    /// Access the underlying [`SceneWorld`] mutably for test operations.
    #[doc(hidden)]
    pub fn world_mut(&mut self) -> &mut SceneWorld {
        &mut self.world
    }

    /// Look up an [`ObjectId`] by persistent [`SceneObjectId`].
    pub fn find_object_by_persistent_id(
        &self,
        persistent_id: &SceneObjectId,
    ) -> Option<ObjectId> {
        self.world.find_object_by_persistent_id(persistent_id)
    }

    pub(crate) fn update_camera(&mut self, view: Mat4, projection: Mat4, position: Vec3) {
        self.world.update_camera(view, projection, position);
    }

    /// Cast a ray into the scene and return the closest intersected node.
    /// Uses the camera's view/projection and screen coordinates.
    ///
    /// This is a pure read-only query: it computes world transforms and
    /// bounds on the fly without mutating cached scene state. The
    /// rendering path (`build_submission`) continues to use the mutable
    /// `refresh_derived_state` path for performance.
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
        self.world.pick_ray_readonly(&ray)
    }

    /// Cast a ray using the scene's last renderer-supplied camera matrices.
    ///
    /// Picking currently uses transform-aware editor proxy AABBs because the
    /// scene graph does not own CPU mesh bounds. Mesh-backed nodes use a
    /// one-unit local proxy and empty group nodes use a smaller origin proxy.
    ///
    /// Returns the last camera view and projection matrices set by the renderer.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn camera_view_projection(&self) -> (glam::Mat4, glam::Mat4) {
        let camera = self.world.camera_data();
        (camera.view, camera.projection)
    }

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

    fn validate_directional_light(&self, id: DirectionalLightId) -> Result<(), SceneError> {
        self.world
            .validate_directional_light_ref(id)
            .map_err(|err| map_directional_light_ref_error(id, err))
    }

    fn validate_spot_light(&self, id: SpotLightId) -> Result<(), SceneError> {
        self.world
            .validate_spot_light_ref(id)
            .map_err(|err| map_spot_light_ref_error(id, err))
    }

    fn ensure_node_persistence_metadata(&mut self, node: SceneNodeId) {
        let stable_id = format!("node.{:06}", self.next_stable_node_id);
        self.next_stable_node_id += 1;

        if let Some(node_ref) = self.world.get_node_mut(node) {
            if node_ref.stable_id.is_none() {
                node_ref.stable_id = Some(stable_id.clone());
            }
            if node_ref.name.is_empty() {
                node_ref.name = stable_id.clone();
            }
        }
        // Also store the stable ID in the record.
        if let Some(record) = self.world.get_node_record_mut(node) {
            if record.stable_id.is_none() {
                record.stable_id = Some(stable_id);
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

fn map_directional_light_ref_error(
    id: DirectionalLightId,
    err: DirectionalLightRefError,
) -> SceneError {
    match err {
        DirectionalLightRefError::GenerationMismatch => SceneError::StaleDirectionalLight(id),
        DirectionalLightRefError::OutOfBounds | DirectionalLightRefError::Vacant => {
            SceneError::InvalidDirectionalLight(format!(
                "directional light out of bounds or vacant (slot={}, generation={})",
                id.slot, id.generation
            ))
        }
    }
}

fn map_spot_light_ref_error(id: SpotLightId, err: SpotLightRefError) -> SceneError {
    match err {
        SpotLightRefError::GenerationMismatch => SceneError::StaleSpotLight(id),
        SpotLightRefError::OutOfBounds | SpotLightRefError::Vacant => {
            SceneError::InvalidSpotLight(format!("spot light out of bounds or vacant"))
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
#[serde(deny_unknown_fields)]
struct SerializedTransform {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedAssetReference {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path_hint: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedVisibility {
    pub visible: bool,
    pub locked: bool,
    pub layer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedNode {
    id: String,
    parent: Option<String>,
    name: String,
    transform: SerializedTransform,
    asset: Option<SerializedAssetReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    collision: Option<SerializedCollisionComponent>,
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct SerializedDirectionalLight {
    id: String,
    kind: String,
    #[serde(default)]
    parent: Option<String>,
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    #[serde(default)]
    shadow_config: Option<SerializedDirectionalShadowConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedSpotLight {
    id: String,
    kind: String,
    #[serde(default)]
    parent: Option<String>,
    position: [f32; 3],
    direction: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    range: f32,
    #[serde(default)]
    inner_cone_angle: f32,
    #[serde(default)]
    outer_cone_angle: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedEnvironment {
    asset: SerializedAssetReference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedMaterialOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base: Option<String>,
    #[serde(default)]
    parameters: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedCollisionComponent {
    pub body: SerializedCollisionBody,
    #[serde(default)]
    pub colliders: Vec<SerializedCollisionCollider>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedCollisionBody {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedCollisionCollider {
    pub id: String,
    pub shape: SerializedCollisionShape,
    #[serde(default)]
    pub trigger: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<String>,
    #[serde(default)]
    pub offset: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedCollisionShape {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub half_extents: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub half_height: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedAudioClipReference {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path_hint: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedAudioReference {
    id: String,
    clip: SerializedAudioClipReference,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trigger: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    usage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    volume: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_gain: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedScene {
    #[serde(default)]
    format_version: u32,
    scene_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    root_nodes: Vec<String>,
    nodes: Vec<SerializedNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    lights: Vec<SerializedPointLight>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    directional_lights: Vec<SerializedDirectionalLight>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    spot_lights: Vec<SerializedSpotLight>,
    environment: Option<SerializedEnvironment>,
    #[serde(default)]
    materials: BTreeMap<String, SerializedMaterialOverride>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    audio: Vec<SerializedAudioReference>,
    #[serde(default = "default_editor_metadata")]
    editor: serde_json::Value,
    /// BSP source-link metadata (persisted for coordinator reload).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bsp_source: Option<serde_json::Value>,
}

impl SerializedScene {
    fn from_scene(scene: &Scene) -> Self {
        let mut fallback_index = 1_u64;
        let nodes: Vec<SerializedNode> = scene
            .world
            .serializable_nodes()
            .map(|(node_id, node)| {
                let record = scene.world.get_node_record(node_id);
                let persistence_visibility = record.and_then(|r| r.visibility.clone());
                let persistence_collision = record.and_then(|r| r.collision.clone());
                let persistence_prefab = record.and_then(|r| r.prefab.clone());
                let stable_id = record
                    .and_then(|r| r.stable_id.clone())
                    .or_else(|| node.stable_id.clone());
                let id = stable_id.unwrap_or_else(|| {
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
                    collision: persistence_collision,
                    material_overrides: node.material_overrides.clone(),
                    visibility: persistence_visibility.unwrap_or_else(default_visibility),
                    tags: node.tags.clone(),
                    prefab: persistence_prefab,
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
            .map(|(idx, (id, light))| {
                let record = scene.world.get_point_light_record(id);
                let stable_id = record.and_then(|r| r.stable_id.clone());
                let parent = record.and_then(|r| r.light_group_parent.clone());
                SerializedPointLight {
                    id: stable_id
                        .unwrap_or_else(|| format!("light.{:06}", idx + 1)),
                    kind: "point".to_string(),
                    parent: parent.map(|pid| pid.to_string()),
                    position: light.position.to_array(),
                    color: light.color.to_array(),
                    intensity: light.intensity,
                    range: light.range,
                }
            })
            .collect();

        let directional_lights: Vec<SerializedDirectionalLight> = scene
            .world
            .serializable_directional_lights()
            .enumerate()
            .map(|(idx, (id, light))| {
                let record = scene.world.get_directional_light_record(id);
                let stable_id = record.and_then(|r| r.stable_id.clone());
                let parent = record.and_then(|r| r.light_group_parent.clone());
                let shadow_config = record.and_then(|r| r.directional_shadow_config);
                SerializedDirectionalLight {
                    id: stable_id
                        .unwrap_or_else(|| format!("directional.{:06}", idx + 1)),
                    kind: "directional".to_string(),
                    parent: parent.map(|pid| pid.to_string()),
                    direction: light.direction.to_array(),
                    color: light.color.to_array(),
                    intensity: light.intensity,
                    shadow_config: shadow_config.map(SerializedDirectionalShadowConfig::from),
                }
            })
            .collect();

        let spot_lights: Vec<SerializedSpotLight> = scene
            .world
            .serializable_spot_lights()
            .enumerate()
            .map(|(idx, (id, light))| {
                let record = scene.world.get_spot_light_record(id);
                let stable_id = record.and_then(|r| r.stable_id.clone());
                let parent = record.and_then(|r| r.light_group_parent.clone());
                SerializedSpotLight {
                    id: stable_id
                        .unwrap_or_else(|| format!("spot.{:06}", idx + 1)),
                    kind: "spot".to_string(),
                    parent: parent.map(|pid| pid.to_string()),
                    position: light.position.to_array(),
                    direction: light.direction.to_array(),
                    color: light.color.to_array(),
                    intensity: light.intensity,
                    range: light.range,
                    inner_cone_angle: light.inner_cone_angle,
                    outer_cone_angle: light.outer_cone_angle,
                }
            })
            .collect();

        SerializedScene {
            format_version: SCENE_FORMAT_VERSION,
            scene_id: scene.scene_id.clone(),
            display_name: scene.display_name.clone(),
            root_nodes,
            nodes,
            lights,
            directional_lights,
            spot_lights,
            environment: scene
                .skybox_asset
                .as_ref()
                .map(|asset| SerializedEnvironment {
                    asset: SerializedAssetReference::from_scene_ref(asset),
                }),
            materials: scene.materials.clone(),
            audio: scene.audio.clone(),
            editor: scene.editor_metadata().clone(),
            bsp_source: scene.bsp_source_link.clone(),
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
        scene.audio = self.audio.clone();
        scene.set_editor_metadata(self.editor.clone());
        scene.bsp_source_link = self.bsp_source.clone();

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
            // Store persistence in the record.
            if let Some(record) = scene.world.get_node_record_mut(node_id) {
                record.stable_id = Some(serialized_node.id.clone());
                record.visibility = Some(serialized_node.visibility.clone());
                record.collision = serialized_node.collision.clone();
                record.prefab = serialized_node.prefab.clone();
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
            let pl = PointLight {
                position: Vec3::from_array(light.position),
                color: Vec3::from_array(light.color),
                intensity: light.intensity,
                range: light.range,
            };
            pl.validate().map_err(crate::api::errors::RendererError::from)?;
            if scene.world.active_point_light_count() >= MAX_POINT_LIGHTS_GPU {
                return Err(SceneError::InvalidPointLight(format!(
                    "point-light cap ({MAX_POINT_LIGHTS_GPU}) reached"
                ))
                .into());
            }
            let light_group_parent = light.parent.as_ref().map(|p| SceneObjectId::new(p.clone()));
            let record = ObjectRecord::for_new_point_light(
                Some(light.id.clone()),
                light_group_parent,
            );
            let id = scene.world.add_point_light_with_record(PointLight {
                color: pl.sanitize_color(),
                ..pl
            }, record);
        }

        for light in &self.directional_lights {
            if light.kind != "directional" {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "unsupported directional light kind '{}'",
                    light.kind
                ))
                .into());
            }
            let dl = DirectionalLight {
                direction: Vec3::from_array(light.direction),
                color: Vec3::from_array(light.color),
                intensity: light.intensity,
            };
            dl.validate().map_err(crate::api::errors::RendererError::from)?;
            if scene.world.active_directional_light_count() >= MAX_DIRECTIONAL_LIGHTS_GPU {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "directional-light cap ({MAX_DIRECTIONAL_LIGHTS_GPU}) reached"
                ))
                .into());
            }
            let light_group_parent = light.parent.as_ref().map(|p| SceneObjectId::new(p.clone()));
            let mut record = ObjectRecord::for_new_directional_light(
                Some(light.id.clone()),
                light_group_parent,
            );
            // Restore shadow configuration if present.
            if let Some(ref shadow_cfg) = light.shadow_config {
                let cfg: DirectionalShadowConfig = shadow_cfg.clone().into();
                record.directional_shadow_config = Some(cfg);
                if cfg.enabled {
                    // Shadow will be set below when ID is known
                }
            }
            let sanitized = DirectionalLight {
                color: dl.sanitize_color(),
                ..dl
            };
            let id = scene.world.add_directional_light_with_record(sanitized, record);
            // If shadow was enabled, set it now that we have the ID.
            if let Some(ref shadow_cfg) = light.shadow_config {
                let cfg: DirectionalShadowConfig = shadow_cfg.clone().into();
                if cfg.enabled {
                    scene.world.set_shadow_casting_directional(Some(id));
                }
            }
        }

        for light in &self.spot_lights {
            if light.kind != "spot" {
                return Err(SceneError::InvalidSpotLight(format!(
                    "unsupported spot light kind '{}'",
                    light.kind
                ))
                .into());
            }
            let sl = SpotLight {
                position: Vec3::from_array(light.position),
                direction: Vec3::from_array(light.direction),
                color: Vec3::from_array(light.color),
                intensity: light.intensity,
                range: light.range,
                inner_cone_angle: light.inner_cone_angle,
                outer_cone_angle: light.outer_cone_angle,
            };
            sl.validate().map_err(crate::api::errors::RendererError::from)?;
            if scene.world.active_spot_light_count() >= MAX_SPOT_LIGHTS_GPU {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot-light cap ({MAX_SPOT_LIGHTS_GPU}) reached"
                ))
                .into());
            }
            let light_group_parent = light.parent.as_ref().map(|p| SceneObjectId::new(p.clone()));
            let record = ObjectRecord::for_new_spot_light(
                Some(light.id.clone()),
                light_group_parent,
            );
            scene.world.add_spot_light_with_record(SpotLight {
                color: sl.sanitize_color(),
                ..sl
            }, record);
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
        if self.scene_id.trim().is_empty() {
            return Err(SceneError::DisconnectedGraph(
                "scene_id must not be empty".to_string(),
            ));
        }

        let mut seen = HashSet::new();
        let mut index_by_id = HashMap::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(SceneError::DisconnectedGraph(
                    "node id must not be empty".to_string(),
                ));
            }
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
            // Validate finite transforms.
            let t = &node.transform;
            if !t.translation.iter().all(|v| v.is_finite())
                || !t.rotation.iter().all(|v| v.is_finite())
                || !t.scale.iter().all(|v| v.is_finite())
            {
                return Err(SceneError::DisconnectedGraph(format!(
                    "node '{}' has non-finite transform components",
                    node.id
                )));
            }
        }

        // Validate point lights.
        for light in &self.lights {
            if light.kind != "point" {
                return Err(SceneError::InvalidPointLight(format!(
                    "unsupported light kind '{}'",
                    light.kind
                )));
            }
            if let Some(parent) = &light.parent {
                if !index_by_id.contains_key(parent) {
                    return Err(SceneError::BadSerializedParent {
                        node_id: light.id.clone(),
                        parent_id: parent.clone(),
                    });
                }
            }
            if !light.position.iter().all(|v| v.is_finite()) {
                return Err(SceneError::InvalidPointLight(format!(
                    "point light '{}' has non-finite position",
                    light.id
                )));
            }
            if !light.range.is_finite() || light.range <= 0.0 {
                return Err(SceneError::InvalidPointLight(format!(
                    "point light '{}' has invalid range",
                    light.id
                )));
            }
            if !light.intensity.is_finite() || light.intensity < 0.0 {
                return Err(SceneError::InvalidPointLight(format!(
                    "point light '{}' has invalid intensity",
                    light.id
                )));
            }
            if !light.color.iter().all(|v| v.is_finite()) {
                return Err(SceneError::InvalidPointLight(format!(
                    "point light '{}' has non-finite color",
                    light.id
                )));
            }
        }

        // Validate directional lights.
        for light in &self.directional_lights {
            if light.kind != "directional" {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "unsupported directional light kind '{}'",
                    light.kind
                )));
            }
            if let Some(parent) = &light.parent {
                if !index_by_id.contains_key(parent) {
                    return Err(SceneError::BadSerializedParent {
                        node_id: light.id.clone(),
                        parent_id: parent.clone(),
                    });
                }
            }
            if !light.direction.iter().all(|v| v.is_finite())
                || light.direction.iter().all(|v| v.abs() < 1e-6)
            {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "directional light '{}' has non-finite or zero direction",
                    light.id
                )));
            }
            if !light.intensity.is_finite() || light.intensity < 0.0 {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "directional light '{}' has invalid intensity",
                    light.id
                )));
            }
            if !light.color.iter().all(|v| v.is_finite()) {
                return Err(SceneError::InvalidDirectionalLight(format!(
                    "directional light '{}' has non-finite color",
                    light.id
                )));
            }
        }

        // Validate spot lights.
        for light in &self.spot_lights {
            if light.kind != "spot" {
                return Err(SceneError::InvalidSpotLight(format!(
                    "unsupported spot light kind '{}'",
                    light.kind
                )));
            }
            if let Some(parent) = &light.parent {
                if !index_by_id.contains_key(parent) {
                    return Err(SceneError::BadSerializedParent {
                        node_id: light.id.clone(),
                        parent_id: parent.clone(),
                    });
                }
            }
            if !light.position.iter().all(|v| v.is_finite()) {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has non-finite position",
                    light.id
                )));
            }
            if !light.direction.iter().all(|v| v.is_finite())
                || light.direction.iter().all(|v| v.abs() < 1e-6)
            {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has non-finite or zero direction",
                    light.id
                )));
            }
            if !light.range.is_finite() || light.range <= 0.0 {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has invalid range",
                    light.id
                )));
            }
            if !light.intensity.is_finite() || light.intensity < 0.0 {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has invalid intensity",
                    light.id
                )));
            }
            if !light.color.iter().all(|v| v.is_finite()) {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has non-finite color",
                    light.id
                )));
            }
            if !light.inner_cone_angle.is_finite()
                || light.inner_cone_angle < 0.0
                || light.inner_cone_angle > std::f32::consts::PI
            {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has invalid inner_cone_angle",
                    light.id
                )));
            }
            if !light.outer_cone_angle.is_finite()
                || light.outer_cone_angle < 0.0
                || light.outer_cone_angle > std::f32::consts::PI
            {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' has invalid outer_cone_angle",
                    light.id
                )));
            }
            if light.inner_cone_angle > light.outer_cone_angle {
                return Err(SceneError::InvalidSpotLight(format!(
                    "spot light '{}' inner_cone_angle must be <= outer_cone_angle",
                    light.id
                )));
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

fn validate_scene_content(
    content: &str,
    path: Option<&Path>,
    options: &SceneValidationOptions,
) -> Result<(), ValidationError> {
    let raw: serde_json::Value = serde_json::from_str(content).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "scene.parse",
                ValidationArea::Scene,
                format!("invalid scene JSON: {err}"),
            )
            .with_optional_path(path),
        )
    })?;

    let mut diagnostics = Vec::new();
    if raw.get("format_version").is_none() {
        diagnostics.push(
            ValidationDiagnostic::new(
                "scene.missing_format_version",
                ValidationArea::Scene,
                "missing required format_version",
            )
            .with_optional_path(path),
        );
    }
    collect_scene_runtime_handle_diagnostics(&raw, path, &mut diagnostics);
    if !diagnostics.is_empty() {
        return Err(ValidationError::new(diagnostics));
    }

    // Migrate v1 → v2 before strict deserialization so validation can
    // proceed on forward-compatible documents.
    let version = raw
        .get("format_version")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let migrated = if version == 1 {
        migrate_v1_to_v2(raw)
    } else {
        raw
    };

    let serialized: SerializedScene = serde_json::from_value(migrated).map_err(|err| {
        ValidationError::single(
            ValidationDiagnostic::new(
                "scene.parse",
                ValidationArea::Scene,
                format!("invalid scene schema: {err}"),
            )
            .with_optional_path(path),
        )
    })?;
    collect_scene_collision_diagnostics(&serialized, path, options, &mut diagnostics);
    collect_scene_audio_diagnostics(&serialized, path, options, &mut diagnostics);
    if !diagnostics.is_empty() {
        return Err(ValidationError::new(diagnostics));
    }
    serialized
        .validate()
        .map_err(|err| ValidationError::single(scene_error_to_diagnostic(err, path)))?;

    if let Some(known_asset_ids) = &options.known_asset_ids {
        let unknown = collect_unknown_scene_assets(&serialized, known_asset_ids, path);
        if !unknown.is_empty() {
            return Err(ValidationError::new(unknown));
        }
    }

    Ok(())
}

fn collect_scene_runtime_handle_diagnostics(
    raw: &serde_json::Value,
    path: Option<&Path>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let Some(nodes) = raw.get("nodes").and_then(serde_json::Value::as_array) else {
        return;
    };

    for node in nodes {
        let Some(node_obj) = node.as_object() else {
            continue;
        };
        let node_id = node_obj
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        if node_obj.get("id").is_some_and(is_json_runtime_handle_shape) {
            diagnostics.push(
                ValidationDiagnostic::new(
                    "scene.runtime_handle_identity",
                    ValidationArea::Node,
                    "node id must be a stable string, not a runtime handle",
                )
                .with_optional_path(path),
            );
        }
        for field in ["asset", "prefab", "collision"] {
            if let Some(value) = node_obj.get(field) {
                collect_json_handle_shapes(value, path, node_id.as_deref(), diagnostics);
            }
        }
    }

    if let Some(lights) = raw.get("lights").and_then(serde_json::Value::as_array) {
        for light in lights {
            let Some(light_obj) = light.as_object() else {
                continue;
            };
            if light_obj
                .get("id")
                .is_some_and(is_json_runtime_handle_shape)
            {
                diagnostics.push(
                    ValidationDiagnostic::new(
                        "scene.runtime_handle_identity",
                        ValidationArea::Scene,
                        "light id must be a stable string, not a runtime handle",
                    )
                    .with_optional_path(path),
                );
            }
        }
    }

    if let Some(environment) = raw.get("environment") {
        collect_json_handle_shapes(environment, path, Some("environment"), diagnostics);
    }

    if let Some(audio) = raw.get("audio") {
        collect_json_handle_shapes(audio, path, Some("audio"), diagnostics);
    }
}

fn collect_unknown_scene_assets(
    serialized: &SerializedScene,
    known_asset_ids: &HashSet<String>,
    path: Option<&Path>,
) -> Vec<ValidationDiagnostic> {
    let mut diagnostics = Vec::new();
    for node in &serialized.nodes {
        let Some(asset_id) = node.asset.as_ref().and_then(|asset| asset.id.as_deref()) else {
            continue;
        };
        if !known_asset_ids.contains(asset_id) {
            diagnostics.push(
                ValidationDiagnostic::new(
                    "scene.unknown_asset_id",
                    ValidationArea::Asset,
                    format!("unknown durable asset id '{asset_id}'"),
                )
                .with_optional_path(path)
                .with_durable_id(asset_id.to_string()),
            );
        }
    }
    if let Some(asset_id) = serialized
        .environment
        .as_ref()
        .and_then(|environment| environment.asset.id.as_deref())
    {
        if !known_asset_ids.contains(asset_id) {
            diagnostics.push(
                ValidationDiagnostic::new(
                    "scene.unknown_asset_id",
                    ValidationArea::Environment,
                    format!("unknown durable asset id '{asset_id}'"),
                )
                .with_optional_path(path)
                .with_durable_id(asset_id.to_string()),
            );
        }
    }
    diagnostics
}

fn collect_scene_collision_diagnostics(
    serialized: &SerializedScene,
    path: Option<&Path>,
    options: &SceneValidationOptions,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let mut collision_ids = HashSet::new();
    for node in &serialized.nodes {
        let Some(collision) = &node.collision else {
            continue;
        };
        validate_scene_collision_id(
            &collision.body.id,
            "body.id",
            &node.id,
            path,
            &mut collision_ids,
            diagnostics,
        );
        match collision.body.kind.as_str() {
            "static" | "dynamic" | "kinematic" => {}
            _ => diagnostics.push(scene_collision_diagnostic(
                "scene.collision_invalid_body_kind",
                "collision body kind must be static, dynamic, or kinematic",
                path,
                &node.id,
            )),
        }
        if collision.colliders.is_empty() {
            diagnostics.push(scene_collision_diagnostic(
                "scene.collision_missing_collider",
                "collision component requires at least one collider",
                path,
                &node.id,
            ));
        }
        for collider in &collision.colliders {
            validate_scene_collision_id(
                &collider.id,
                "collider.id",
                &node.id,
                path,
                &mut collision_ids,
                diagnostics,
            );
            validate_scene_collision_shape(&collider.shape, path, &node.id, diagnostics);
            if !collider.offset.iter().all(|value| value.is_finite()) {
                diagnostics.push(scene_collision_diagnostic(
                    "scene.collision_invalid_offset",
                    "collision collider offset must contain finite numbers",
                    path,
                    &node.id,
                ));
            }
            if let Some(asset_id) = &collider.asset {
                if is_invalid_durable_collision_id(asset_id) {
                    diagnostics.push(scene_collision_diagnostic(
                        "scene.collision_invalid_id",
                        format!("collision asset '{asset_id}' is not a durable id"),
                        path,
                        &node.id,
                    ));
                } else if options
                    .known_asset_ids
                    .as_ref()
                    .is_some_and(|known| !known.contains(asset_id))
                {
                    diagnostics.push(
                        scene_collision_diagnostic(
                            "scene.unknown_collision_asset_id",
                            format!("unknown collision asset id '{asset_id}'"),
                            path,
                            &node.id,
                        )
                        .with_durable_id(asset_id.clone()),
                    );
                }
            }
        }
    }
}

fn collect_scene_audio_diagnostics(
    serialized: &SerializedScene,
    path: Option<&Path>,
    options: &SceneValidationOptions,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    let mut audio_ids = HashSet::new();
    for audio in &serialized.audio {
        validate_scene_audio_id(
            &audio.id,
            "id",
            &audio.id,
            path,
            &mut audio_ids,
            diagnostics,
        );
        let Some(clip_id) = audio
            .clip
            .id
            .as_ref()
            .map(|id| id.trim())
            .filter(|id| !id.is_empty())
        else {
            diagnostics.push(scene_audio_diagnostic(
                "scene.audio_missing_clip_id",
                "audio clip reference requires a durable clip id",
                path,
                &audio.id,
            ));
            continue;
        };
        if is_invalid_durable_audio_id(clip_id) {
            diagnostics.push(scene_audio_diagnostic(
                "scene.audio_invalid_id",
                format!("audio clip id '{clip_id}' is not a durable id"),
                path,
                &audio.id,
            ));
        } else if options
            .known_asset_ids
            .as_ref()
            .is_some_and(|known| !known.contains(clip_id))
        {
            diagnostics.push(
                scene_audio_diagnostic(
                    "scene.unknown_audio_clip_id",
                    format!("unknown audio clip id '{clip_id}'"),
                    path,
                    &audio.id,
                )
                .with_durable_id(clip_id.to_string()),
            );
        }

        if let Some(usage) = &audio.usage {
            match usage.as_str() {
                "effect" | "music" | "ambient" | "voice" | "ui" => {}
                _ => diagnostics.push(scene_audio_diagnostic(
                    "scene.audio_invalid_usage",
                    "audio usage must be one of effect, music, ambient, voice, or ui",
                    path,
                    &audio.id,
                )),
            }
        }

        for (field, value) in [
            ("volume", audio.volume),
            ("default_gain", audio.default_gain),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                diagnostics.push(scene_audio_diagnostic(
                    "scene.audio_invalid_gain",
                    format!("audio {field} must be a positive finite number"),
                    path,
                    &audio.id,
                ));
            }
        }
    }
}

fn validate_scene_audio_id(
    id: &str,
    field: &str,
    durable_id: &str,
    path: Option<&Path>,
    audio_ids: &mut HashSet<String>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    if is_invalid_durable_audio_id(id) {
        diagnostics.push(scene_audio_diagnostic(
            "scene.audio_invalid_id",
            format!("audio {field} '{id}' is not a durable id"),
            path,
            durable_id,
        ));
        return;
    }
    if !audio_ids.insert(id.to_string()) {
        diagnostics.push(scene_audio_diagnostic(
            "scene.duplicate_audio_id",
            format!("duplicate audio id '{id}'"),
            path,
            durable_id,
        ));
    }
}

fn scene_audio_diagnostic(
    code: impl Into<String>,
    message: impl Into<String>,
    path: Option<&Path>,
    audio_id: &str,
) -> ValidationDiagnostic {
    ValidationDiagnostic::new(code, ValidationArea::Scene, message)
        .with_optional_path(path)
        .with_durable_id(audio_id.to_string())
}

fn validate_scene_collision_id(
    id: &str,
    field: &str,
    node_id: &str,
    path: Option<&Path>,
    collision_ids: &mut HashSet<String>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    if is_invalid_durable_collision_id(id) {
        diagnostics.push(scene_collision_diagnostic(
            "scene.collision_invalid_id",
            format!("collision {field} '{id}' is not a durable id"),
            path,
            node_id,
        ));
        return;
    }
    if !collision_ids.insert(id.to_string()) {
        diagnostics.push(scene_collision_diagnostic(
            "scene.duplicate_collision_id",
            format!("duplicate collision id '{id}'"),
            path,
            node_id,
        ));
    }
}

fn validate_scene_collision_shape(
    shape: &SerializedCollisionShape,
    path: Option<&Path>,
    node_id: &str,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    match shape.kind.as_str() {
        "box" | "cuboid" => {
            if !shape
                .half_extents
                .is_some_and(|values| values.iter().all(|value| value.is_finite() && *value > 0.0))
            {
                diagnostics.push(scene_collision_diagnostic(
                    "scene.collision_invalid_dimension",
                    "box collision shape requires positive finite half_extents",
                    path,
                    node_id,
                ));
            }
        }
        "sphere" => {
            validate_scene_positive_scalar(shape.radius, "radius", path, node_id, diagnostics)
        }
        "capsule" | "capsule_y" => {
            validate_scene_positive_scalar(shape.radius, "radius", path, node_id, diagnostics);
            validate_scene_positive_scalar(
                shape.half_height,
                "half_height",
                path,
                node_id,
                diagnostics,
            );
        }
        _ => diagnostics.push(scene_collision_diagnostic(
            "scene.collision_invalid_shape",
            "collision shape kind must be box, cuboid, sphere, capsule, or capsule_y",
            path,
            node_id,
        )),
    }
}

fn validate_scene_positive_scalar(
    value: Option<f32>,
    field: &str,
    path: Option<&Path>,
    node_id: &str,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    if !value.is_some_and(|value| value.is_finite() && value > 0.0) {
        diagnostics.push(scene_collision_diagnostic(
            "scene.collision_invalid_dimension",
            format!("collision {field} must be a positive finite number"),
            path,
            node_id,
        ));
    }
}

fn scene_collision_diagnostic(
    code: impl Into<String>,
    message: impl Into<String>,
    path: Option<&Path>,
    node_id: &str,
) -> ValidationDiagnostic {
    ValidationDiagnostic::new(code, ValidationArea::Scene, message)
        .with_optional_path(path)
        .with_durable_id(node_id.to_string())
}

fn collect_json_handle_shapes(
    value: &serde_json::Value,
    path: Option<&Path>,
    durable_id: Option<&str>,
    diagnostics: &mut Vec<ValidationDiagnostic>,
) {
    match value {
        serde_json::Value::Object(map) => {
            if is_json_runtime_handle_shape(value) || map.keys().any(|key| key.ends_with("_handle"))
            {
                let mut diagnostic = ValidationDiagnostic::new(
                    "scene.runtime_handle_identity",
                    ValidationArea::Scene,
                    "runtime handles are not durable scene identity",
                )
                .with_optional_path(path);
                if let Some(durable_id) = durable_id {
                    diagnostic = diagnostic.with_durable_id(durable_id.to_string());
                }
                diagnostics.push(diagnostic);
            }
            for child in map.values() {
                collect_json_handle_shapes(child, path, durable_id, diagnostics);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                collect_json_handle_shapes(child, path, durable_id, diagnostics);
            }
        }
        _ => {}
    }
}

fn is_json_runtime_handle_shape(value: &serde_json::Value) -> bool {
    value
        .as_object()
        .is_some_and(|map| map.contains_key("slot") && map.contains_key("generation"))
}

fn is_invalid_durable_collision_id(id: &str) -> bool {
    is_invalid_durable_audio_id(id)
}

fn is_invalid_durable_audio_id(id: &str) -> bool {
    let trimmed = id.trim();
    trimmed.is_empty()
        || trimmed != id
        || !trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || (trimmed.contains("slot") && trimmed.contains("generation"))
}

fn scene_error_to_diagnostic(err: SceneError, path: Option<&Path>) -> ValidationDiagnostic {
    match err {
        SceneError::UnsupportedSceneVersion { found, expected } => ValidationDiagnostic::new(
            "scene.unsupported_version",
            ValidationArea::Scene,
            format!("unsupported scene format version {found}; expected {expected}"),
        )
        .with_optional_path(path),
        SceneError::MissingAssetId(context) => ValidationDiagnostic::new(
            "scene.missing_asset_id",
            ValidationArea::Asset,
            format!("missing durable asset id for {context}"),
        )
        .with_optional_path(path),
        SceneError::BadSerializedParent { node_id, parent_id } => ValidationDiagnostic::new(
            "scene.missing_parent",
            ValidationArea::Node,
            format!("scene node '{node_id}' references missing parent '{parent_id}'"),
        )
        .with_optional_path(path)
        .with_durable_id(node_id),
        SceneError::DuplicateSerializedNodeId(id) => ValidationDiagnostic::new(
            "scene.duplicate_node_id",
            ValidationArea::Node,
            format!("duplicate serialized scene node id '{id}'"),
        )
        .with_optional_path(path)
        .with_durable_id(id),
        SceneError::DisconnectedGraph(message) => ValidationDiagnostic::new(
            "scene.invalid_graph",
            ValidationArea::Scene,
            format!("invalid scene graph: {message}"),
        )
        .with_optional_path(path),
        SceneError::CycleDetected => ValidationDiagnostic::new(
            "scene.cycle",
            ValidationArea::Scene,
            "cycle detected in scene hierarchy",
        )
        .with_optional_path(path),
        other => {
            ValidationDiagnostic::new("scene.invalid", ValidationArea::Scene, other.to_string())
                .with_optional_path(path)
        }
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

/// Migrate a v1 scene JSON Value to v2 format so it can be deserialized
/// by the strict (deny_unknown_fields) `SerializedScene` parser.
///
/// - Sets `format_version` to 2.
/// - Strips unrecognized top-level keys so v1 extras don't fail strict parsing.
/// - Adds `shadow_config: null` default to directional lights that lack it.
fn migrate_v1_to_v2(mut raw: serde_json::Value) -> serde_json::Value {
    const V2_TOP_LEVEL_KEYS: &[&str] = &[
        "format_version",
        "scene_id",
        "display_name",
        "root_nodes",
        "nodes",
        "lights",
        "directional_lights",
        "spot_lights",
        "environment",
        "materials",
        "audio",
        "editor",
    ];

    // Bump version.
    if let Some(obj) = raw.as_object_mut() {
        obj.insert(
            "format_version".to_string(),
            serde_json::Value::Number(serde_json::Number::from(2)),
        );

        // Strip unknown top-level keys.
        let known: std::collections::HashSet<&str> = V2_TOP_LEVEL_KEYS.iter().copied().collect();
        obj.retain(|key, _| known.contains(key.as_str()));

        // Add shadow_config default to each directional light.
        if let Some(directional_lights) = obj
            .get_mut("directional_lights")
            .and_then(|v| v.as_array_mut())
        {
            for light in directional_lights.iter_mut() {
                if let Some(light_obj) = light.as_object_mut() {
                    if !light_obj.contains_key("shadow_config") {
                        light_obj.insert("shadow_config".to_string(), serde_json::Value::Null);
                    }
                }
            }
        }
    }

    raw
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
        validate_scene_str, validate_scene_str_with_options, DirectionalLight,
        DirectionalShadowConfig, PointLight, Scene, SceneAssetLoader, SceneAssetReference,
        SceneFragment, SceneFragmentNode, SceneFragmentNodeId, SceneValidationOptions,
        SerializedScene,
    };
    use crate::api::errors::{AssetError, RendererError, SceneError};
    use crate::data::handles::{EnvironmentHandle, MeshHandle};
    use crate::scene::command::{CommandHistory, PlaceAssetCommand};
    use crate::scene::render_submission::MAX_DIRECTIONAL_LIGHTS_GPU;
    use glam::{Mat4, Quat, Vec3};
    use std::fs;
    use std::path::{Path, PathBuf};

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_mount_retirement_detaches_scene_state() {
        let mut scene = Scene::new();
        scene.set_bsp_mount(crate::api::bsp::PreparedBspMount::new());
        assert!(scene.has_bsp_mount());

        let detached = scene
            .retire_bsp_mount()
            .expect("published BSP mount must return a detached mount");
        assert!(!scene.has_bsp_mount());
        assert!(scene.retire_bsp_mount().is_none());
        drop(detached);
    }

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
    fn directional_light_create_update_remove_and_single_light_limit() {
        let mut scene = Scene::new();
        let light = DirectionalLight {
            direction: Vec3::new(0.25, 1.0, 0.5),
            color: Vec3::new(1.0, 0.9, 0.8),
            intensity: 3.0,
        };
        let id = scene.create_directional_light(light).unwrap();

        assert!(matches!(
            scene.create_directional_light(light),
            Err(SceneError::InvalidDirectionalLight(_))
        ));

        let updated = DirectionalLight {
            intensity: 4.0,
            ..light
        };
        scene.update_directional_light(id, updated).unwrap();
        assert_eq!(scene.directional_light(), Some(updated));
        let submitted = scene.build_submission().directional_light.unwrap();
        assert_eq!(submitted.direction, updated.direction);
        assert_eq!(submitted.color, updated.color);
        assert_eq!(submitted.intensity, updated.intensity);

        scene.remove_directional_light(id).unwrap();
        assert!(matches!(
            scene.update_directional_light(id, light),
            Err(SceneError::StaleDirectionalLight(_))
        ));
        assert!(scene.create_directional_light(light).is_ok());
    }

    #[test]
    fn additive_directionals_are_bounded_and_only_one_owns_shadows() {
        let mut scene = Scene::new();
        let light = DirectionalLight {
            direction: Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
        };
        let first = scene.create_directional_light(light).unwrap();
        let second = scene.add_directional_light(light).unwrap();
        scene.add_directional_light(light).unwrap();
        scene.add_directional_light(light).unwrap();
        assert!(matches!(
            scene.add_directional_light(light),
            Err(SceneError::InvalidDirectionalLight(_))
        ));

        scene
            .set_directional_shadow_config(
                first,
                DirectionalShadowConfig {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(matches!(
            scene.set_directional_shadow_config(
                second,
                DirectionalShadowConfig {
                    enabled: true,
                    ..Default::default()
                }
            ),
            Err(SceneError::UnsupportedLightFeature(_))
        ));
        let submission = scene.build_submission();
        assert_eq!(
            submission.directional_lights.len(),
            MAX_DIRECTIONAL_LIGHTS_GPU
        );
        assert_eq!(
            submission
                .directional_lights
                .iter()
                .filter(|light| light.enable_shadows)
                .count(),
            1
        );
    }

    #[test]
    fn directional_light_rejects_zero_or_non_finite_direction() {
        let mut scene = Scene::new();
        for direction in [Vec3::ZERO, Vec3::new(f32::NAN, 1.0, 0.0)] {
            let result = scene.create_directional_light(DirectionalLight {
                direction,
                color: Vec3::ONE,
                intensity: 1.0,
            });
            assert!(matches!(
                result,
                Err(SceneError::InvalidDirectionalLight(_))
            ));
        }
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
        assert!(json.contains("\"format_version\": 2"));
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
    fn editor_packaged_scene_save_copy_round_trips_model_and_wall_chunk() {
        let saved_scene = phase_02_saved_scene_copy_path();
        fs::create_dir_all(saved_scene.parent().expect("artifact parent")).unwrap();

        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();
        scene
            .set_node_name(root, "Phase 02 Round Trip Root")
            .unwrap();
        scene
            .set_node_tags(root, vec!["phase-02".to_string(), "root".to_string()])
            .unwrap();
        let mut history = CommandHistory::new(8);

        let model = place_test_asset(
            &mut scene,
            &mut history,
            root,
            "editor_sample.model.block",
            "models/block_prop.obj",
            "Block Prop",
            &["model", "prop", "sample", "phase-02"],
            "node.placed.editor_sample_model_block.000001",
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.25, 1.0, 1.25),
                Quat::IDENTITY,
                Vec3::new(-1.5, 0.0, -2.0),
            ),
            MeshHandle::new(21, 0),
        );
        scene
            .set_node_material_override(model, "0", "mat_override.phase02_block")
            .unwrap();

        let wall = place_test_asset(
            &mut scene,
            &mut history,
            root,
            "editor_sample.wall.stone_2m",
            "prefabs/wall_straight_2m.obj",
            "Stone Wall 2m",
            &["wall", "chunk", "prefab", "sample", "phase-02"],
            "node.placed.editor_sample_wall_stone_2m.000002",
            Mat4::from_translation(Vec3::new(1.5, 0.0, -2.0)),
            MeshHandle::new(22, 0),
        );

        scene.save(&saved_scene).unwrap();
        let json = fs::read_to_string(&saved_scene).unwrap();
        assert!(json.contains("\"id\": \"editor_sample.model.block\""));
        assert!(json.contains("\"id\": \"editor_sample.wall.stone_2m\""));
        assert!(json.contains("\"node.placed.editor_sample_model_block.000001\""));
        assert!(json.contains("\"node.placed.editor_sample_wall_stone_2m.000002\""));
        assert!(json.contains("\"mat_override.phase02_block\""));
        assert!(!json.contains("\"slot\""));
        assert!(!json.contains("\"generation\""));
        assert!(!json.contains("mesh_handle"));
        validate_scene_str_with_options(
            &json,
            &SceneValidationOptions::default()
                .with_known_asset_ids(["editor_sample.model.block", "editor_sample.wall.stone_2m"]),
        )
        .unwrap();

        let mut loader = FakeSceneAssetLoader::default();
        let loaded = Scene::load_with_loader(&saved_scene, &mut loader).unwrap();
        assert_eq!(
            loader.loaded_models,
            vec![
                SceneAssetReference::new(
                    "editor_sample.model.block",
                    Some(PathBuf::from("models/block_prop.obj")),
                ),
                SceneAssetReference::new(
                    "editor_sample.wall.stone_2m",
                    Some(PathBuf::from("prefabs/wall_straight_2m.obj")),
                ),
            ]
        );

        let model_stable_id = scene.node_stable_id(model).unwrap().unwrap();
        let wall_stable_id = scene.node_stable_id(wall).unwrap().unwrap();
        let summaries = loaded.node_summaries();
        let model_summary = summaries
            .iter()
            .find(|node| node.stable_id.as_deref() == Some(model_stable_id.as_str()))
            .expect("loaded model node");
        let wall_summary = summaries
            .iter()
            .find(|node| node.stable_id.as_deref() == Some(wall_stable_id.as_str()))
            .expect("loaded wall node");

        assert_eq!(
            model_summary
                .asset
                .as_ref()
                .map(|asset| (asset.id.as_str(), asset.path_hint.as_deref())),
            Some((
                "editor_sample.model.block",
                Some(Path::new("models/block_prop.obj"))
            ))
        );
        assert_eq!(
            wall_summary
                .asset
                .as_ref()
                .map(|asset| (asset.id.as_str(), asset.path_hint.as_deref())),
            Some((
                "editor_sample.wall.stone_2m",
                Some(Path::new("prefabs/wall_straight_2m.obj"))
            ))
        );
        assert_eq!(
            model_summary
                .material_overrides
                .get("0")
                .map(String::as_str),
            Some("mat_override.phase02_block")
        );
        assert!(model_summary.tags.iter().any(|tag| tag == "phase-02"));
        assert!(wall_summary.tags.iter().any(|tag| tag == "wall"));
        assert_transform_translation(model_summary.local_transform, Vec3::new(-1.5, 0.0, -2.0));
        assert_transform_translation(wall_summary.local_transform, Vec3::new(1.5, 0.0, -2.0));
        fs::remove_file(saved_scene).unwrap();
    }

    #[test]
    fn concurrent_scene_saves_publish_only_complete_documents() {
        let dir = unique_scene_temp_dir("concurrent-save");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("scene.engine.scene.json");

        let mut initial = Scene::new();
        initial.scene_id = "scene.initial".to_string();
        initial.save(&target).unwrap();

        let writer_count = 8usize;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(writer_count));
        let mut handles = Vec::new();
        for index in 0..writer_count {
            let target = target.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                let mut scene = Scene::new();
                scene.scene_id = format!("scene.concurrent.{index}");
                barrier.wait();
                scene.save(target).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let raw = fs::read_to_string(&target).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["format_version"].as_u64(), Some(2));
        let scene_id = parsed["scene_id"].as_str().unwrap();
        assert!(scene_id == "scene.initial" || scene_id.starts_with("scene.concurrent."));
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().contains("scene-save"))
            .collect();
        assert!(leftovers.is_empty(), "leftover staged files: {leftovers:?}");

        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn scene_save_rejects_symlink_target_without_touching_referent() {
        let dir = unique_scene_temp_dir("symlink-save");
        fs::create_dir_all(&dir).unwrap();
        let referent = dir.join("referent.json");
        fs::write(&referent, b"referent").unwrap();
        let link = dir.join("scene.engine.scene.json");
        std::os::unix::fs::symlink(&referent, &link).unwrap();

        let err = Scene::new().save(&link).unwrap_err();
        assert!(matches!(err, SceneError::SerializationError(_)));
        assert_eq!(fs::read(&referent).unwrap(), b"referent");
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());

        fs::remove_dir_all(dir).unwrap();
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
                "format_version": 99,
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
                "format_version": 2,
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
                "format_version": 2,
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
                "format_version": 2,
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
                "format_version": 2,
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
                "format_version": 2,
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

    #[test]
    fn scene_validation_accepts_valid_schema_without_loading_assets() {
        validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.valid",
                "root_nodes": ["node.root"],
                "nodes": [
                    {"id":"node.root","parent":null,"name":"Root","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":{"id":"core.model.crate","path_hint":"models/crate.glb"}}
                ],
                "lights": [],
                "environment": {"asset":{"id":"core.env.indoor","path_hint":"sky_maps/indoor.exr"}},
                "editor": {}
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn scene_validation_reports_missing_version_duplicate_nodes_and_missing_parent() {
        let missing_version = validate_scene_str(
            r#"{
                "scene_id": "scene.no_version",
                "root_nodes": [],
                "nodes": [],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(missing_version
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.missing_format_version"));

        let duplicate = validate_scene_str(
            r#"{
                "format_version": 2,
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
        )
        .unwrap_err();
        assert!(duplicate
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.duplicate_node_id"));

        let missing_parent = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.bad_parent",
                "root_nodes": ["node.child"],
                "nodes": [
                    {"id":"node.child","parent":"node.missing","name":"Child","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(missing_parent
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.missing_parent"));
    }

    #[test]
    fn scene_validation_rejects_runtime_handle_identity_and_missing_asset_id() {
        let runtime_handle = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.bad_handle",
                "root_nodes": ["node.a"],
                "nodes": [
                    {"id":{"slot":4,"generation":2},"parent":null,"name":"A","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":{"mesh_handle":{"slot":7,"generation":1}}}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(runtime_handle
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.runtime_handle_identity"));

        let missing_asset_id = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.path_only",
                "root_nodes": ["node.asset"],
                "nodes": [
                    {"id":"node.asset","parent":null,"name":"Asset","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":{"path_hint":"models/crate.glb"}}
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(missing_asset_id
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.missing_asset_id"));
    }

    #[test]
    fn scene_validation_can_report_unknown_asset_ids_from_known_package_records() {
        let err = validate_scene_str_with_options(
            r#"{
                "format_version": 2,
                "scene_id": "scene.unknown_asset",
                "root_nodes": ["node.asset"],
                "nodes": [
                    {"id":"node.asset","parent":null,"name":"Asset","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":{"id":"core.model.missing","path_hint":"models/missing.glb"}}
                ],
                "lights": [],
                "environment": {"asset":{"id":"core.env.known"}},
                "editor": {}
            }"#,
            &SceneValidationOptions::default().with_known_asset_ids(["core.env.known"]),
        )
        .unwrap_err();

        assert!(err
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.unknown_asset_id"
                && diagnostic.durable_id.as_deref() == Some("core.model.missing")));
    }

    #[test]
    fn scene_validation_accepts_collision_metadata_round_trip_schema() {
        let json = r#"{
            "format_version": 2,
            "scene_id": "scene.collision",
            "root_nodes": ["node.wall"],
            "nodes": [
                {
                    "id": "node.wall",
                    "parent": null,
                    "name": "Wall",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null,
                    "collision": {
                        "body": {"id": "body.wall", "kind": "static"},
                        "colliders": [
                            {
                                "id": "collider.wall",
                                "shape": {"kind": "box", "half_extents": [0.5, 1.25, 0.5]},
                                "trigger": false,
                                "asset": "core.collision.wall",
                                "offset": [0.0, 0.0, 0.0]
                            }
                        ]
                    }
                }
            ],
            "lights": [],
            "environment": null,
            "editor": {}
        }"#;

        validate_scene_str_with_options(
            json,
            &SceneValidationOptions::default().with_known_asset_ids(["core.collision.wall"]),
        )
        .unwrap();

        let parsed: SerializedScene = serde_json::from_str(json).unwrap();
        let pretty = serde_json::to_string_pretty(&parsed).unwrap();
        let round_tripped: SerializedScene = serde_json::from_str(&pretty).unwrap();
        assert_eq!(
            round_tripped.nodes[0].collision.as_ref().unwrap().colliders[0].id,
            "collider.wall"
        );
    }

    #[test]
    fn scene_validation_rejects_invalid_collision_metadata() {
        let invalid_dimensions = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.bad_collision",
                "root_nodes": ["node.bad"],
                "nodes": [
                    {
                        "id": "node.bad",
                        "parent": null,
                        "name": "Bad",
                        "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                        "asset": null,
                        "collision": {
                            "body": {"id": "body.bad", "kind": "static"},
                            "colliders": [
                                {"id": "collider.bad", "shape": {"kind": "sphere", "radius": 0.0}}
                            ]
                        }
                    }
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(invalid_dimensions
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.collision_invalid_dimension"));

        let duplicate_ids = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.duplicate_collision",
                "root_nodes": ["node.a"],
                "nodes": [
                    {
                        "id": "node.a",
                        "parent": null,
                        "name": "A",
                        "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                        "asset": null,
                        "collision": {
                            "body": {"id": "body.a", "kind": "static"},
                            "colliders": [
                                {"id": "collider.same", "shape": {"kind": "sphere", "radius": 0.5}}
                            ]
                        }
                    },
                    {
                        "id": "node.b",
                        "parent": "node.a",
                        "name": "B",
                        "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                        "asset": null,
                        "collision": {
                            "body": {"id": "body.b", "kind": "static"},
                            "colliders": [
                                {"id": "collider.same", "shape": {"kind": "sphere", "radius": 0.5}}
                            ]
                        }
                    }
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(duplicate_ids
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.duplicate_collision_id"));

        let unknown_collision_asset = validate_scene_str_with_options(
            r#"{
                "format_version": 2,
                "scene_id": "scene.unknown_collision_asset",
                "root_nodes": ["node.asset"],
                "nodes": [
                    {
                        "id": "node.asset",
                        "parent": null,
                        "name": "Asset",
                        "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                        "asset": null,
                        "collision": {
                            "body": {"id": "body.asset", "kind": "static"},
                            "colliders": [
                                {"id": "collider.asset", "shape": {"kind": "box", "half_extents": [0.5, 0.5, 0.5]}, "asset": "core.collision.missing"}
                            ]
                        }
                    }
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
            &SceneValidationOptions::default().with_known_asset_ids(["core.model.known"]),
        )
        .unwrap_err();
        assert!(unknown_collision_asset
            .diagnostics()
            .iter()
            .any(
                |diagnostic| diagnostic.code == "scene.unknown_collision_asset_id"
                    && diagnostic.durable_id.as_deref() == Some("core.collision.missing")
            ));

        let runtime_collision_handle = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.runtime_collision_handle",
                "root_nodes": ["node.handle"],
                "nodes": [
                    {
                        "id": "node.handle",
                        "parent": null,
                        "name": "Handle",
                        "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                        "asset": null,
                        "collision": {
                            "body": {"id": {"slot": 1, "generation": 0}, "kind": "static"},
                            "colliders": [
                                {"id": "collider.handle", "shape": {"kind": "sphere", "radius": 0.5}}
                            ]
                        }
                    }
                ],
                "lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(runtime_collision_handle
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.runtime_handle_identity"));
    }

    #[test]
    fn scene_validation_accepts_audio_references() {
        let json = r#"{
            "format_version": 2,
            "scene_id": "scene.audio",
            "root_nodes": ["node.root"],
            "nodes": [
                {
                    "id": "node.root",
                    "parent": null,
                    "name": "Root",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null
                }
            ],
            "lights": [],
            "environment": null,
            "materials": {},
            "audio": [
                {
                    "id": "scene.audio.pickup",
                    "clip": {"id": "core.audio.pickup", "path_hint": "audio/pickup.ogg"},
                    "trigger": "startup",
                    "usage": "effect",
                    "volume": 0.5,
                    "default_gain": 1.0
                }
            ],
            "editor": {}
        }"#;

        validate_scene_str_with_options(
            json,
            &SceneValidationOptions::default().with_known_asset_ids(["core.audio.pickup"]),
        )
        .unwrap();

        let parsed: SerializedScene = serde_json::from_str(json).unwrap();
        let pretty = serde_json::to_string_pretty(&parsed).unwrap();
        let round_tripped: SerializedScene = serde_json::from_str(&pretty).unwrap();
        assert_eq!(
            round_tripped.audio[0].clip.id.as_deref(),
            Some("core.audio.pickup")
        );
    }

    #[test]
    fn scene_validation_rejects_invalid_audio_references() {
        let unknown_clip = validate_scene_str_with_options(
            r#"{
                "format_version": 2,
                "scene_id": "scene.unknown_audio",
                "root_nodes": ["node.root"],
                "nodes": [
                    {"id":"node.root","parent":null,"name":"Root","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "audio": [
                    {"id": "scene.audio.pickup", "clip": {"id": "core.audio.missing"}, "volume": 0.5}
                ],
                "editor": {}
            }"#,
            &SceneValidationOptions::default().with_known_asset_ids(["core.audio.known"]),
        )
        .unwrap_err();
        assert!(unknown_clip
            .diagnostics()
            .iter()
            .any(
                |diagnostic| diagnostic.code == "scene.unknown_audio_clip_id"
                    && diagnostic.durable_id.as_deref() == Some("core.audio.missing")
            ));

        let invalid_gain = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.bad_audio_gain",
                "root_nodes": ["node.root"],
                "nodes": [
                    {"id":"node.root","parent":null,"name":"Root","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "audio": [
                    {"id": "scene.audio.pickup", "clip": {"id": "core.audio.pickup"}, "default_gain": 0.0}
                ],
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(invalid_gain
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.audio_invalid_gain"));

        let runtime_audio_handle = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.runtime_audio_handle",
                "root_nodes": ["node.root"],
                "nodes": [
                    {"id":"node.root","parent":null,"name":"Root","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"asset":null}
                ],
                "lights": [],
                "environment": null,
                "audio": [
                    {"id": "scene.audio.handle", "clip": {"id": {"slot": 4, "generation": 2}}}
                ],
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(runtime_audio_handle
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == "scene.runtime_handle_identity"));
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

    fn phase_02_saved_scene_copy_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "renderer-phase-02-saved-scene-copy-{}.engine.scene.json",
            std::process::id()
        ))
    }

    fn unique_scene_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "renderer-scene-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn place_test_asset(
        scene: &mut Scene,
        history: &mut CommandHistory,
        parent: crate::scene::SceneNodeId,
        asset_id: &str,
        path_hint: &str,
        display_name: &str,
        tags: &[&str],
        stable_id: &str,
        transform: Mat4,
        mesh: MeshHandle,
    ) -> crate::scene::SceneNodeId {
        let mut fragment = SceneFragment::new();
        fragment
            .add_node(None, Mat4::IDENTITY, vec![mesh])
            .expect("fragment root");

        scene
            .execute_command(
                history,
                Box::new(PlaceAssetCommand::new(
                    Some(parent),
                    transform,
                    fragment,
                    SceneAssetReference::new(asset_id, Some(PathBuf::from(path_hint))),
                    display_name,
                    tags.iter().map(|tag| tag.to_string()).collect(),
                    stable_id,
                )),
            )
            .unwrap()
            .created_node
            .expect("placed node")
    }

    fn assert_transform_translation(transform: Mat4, expected: Vec3) {
        let (_, _, translation) = transform.to_scale_rotation_translation();
        assert!(
            translation.abs_diff_eq(expected, 0.0001),
            "translation {translation:?} did not match {expected:?}"
        );
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

    #[test]
    fn frustum_culling_reduces_draw_count() {
        use super::{BoundsUnknownReason, MeshBoundsEntry, SceneBounds};
        use crate::data::camera::Aabb;

        let mut scene = Scene::new();
        assert!(scene.frustum_culling_enabled());

        // Camera at origin looking down -Z with a moderate FOV.
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let projection = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        scene.set_camera(view, projection, Vec3::new(0.0, 0.0, 0.0));

        let root = scene.create_node_default(None).unwrap();

        // Use an explicit unit proxy so culling can operate on known bounds.
        let unit_proxy =
            SceneBounds::Proxy(Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)));

        // Node in front of camera — should stay visible.
        let in_front = scene
            .create_node(
                Some(root),
                Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
            )
            .unwrap();
        scene
            .add_mesh_with_bounds(in_front, MeshHandle::new(1, 0), unit_proxy)
            .unwrap();

        // Node behind camera — should be culled.
        let behind = scene
            .create_node(Some(root), Mat4::from_translation(Vec3::new(0.0, 0.0, 5.0)))
            .unwrap();
        scene
            .add_mesh_with_bounds(behind, MeshHandle::new(2, 0), unit_proxy)
            .unwrap();

        // Count with culling off.
        scene.set_frustum_culling(false);
        assert!(!scene.frustum_culling_enabled());
        let submission_off = scene.build_submission();
        let count_off = submission_off.draw_items.len();
        assert_eq!(count_off, 2);

        // Count with culling on.
        scene.set_frustum_culling(true);
        assert!(scene.frustum_culling_enabled());
        let submission_on = scene.build_submission();
        let count_on = submission_on.draw_items.len();

        assert!(
            count_on < count_off,
            "Expected culling to reduce draw count (off={count_off}, on={count_on})"
        );
        assert_eq!(
            count_on, 1,
            "Expected exactly 1 visible mesh with culling on, got {count_on}"
        );
    }

    // ── H-A1: v1 forward-compatible migration tests ────────────────────

    #[test]
    fn v1_scene_migrates_to_v2_on_load() {
        // A minimal v1 scene with a directional light and one node.
        let v1_json = r#"{
            "format_version": 1,
            "scene_id": "scene.v1_forward",
            "root_nodes": ["node.root"],
            "nodes": [
                {
                    "id": "node.root",
                    "parent": null,
                    "name": "Root",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null
                }
            ],
            "lights": [],
            "directional_lights": [
                {
                    "id": "directional.000001",
                    "kind": "directional",
                    "direction": [0.5, 1.0, 0.25],
                    "color": [1.0, 0.95, 0.8],
                    "intensity": 2.5
                }
            ],
            "spot_lights": [],
            "environment": null,
            "editor": {}
        }"#;

        // Load through the migration path (simulates what `read_serialized_scene` does).
        let parsed: SerializedScene = {
            let raw: serde_json::Value = serde_json::from_str(v1_json).unwrap();
            let migrated = super::migrate_v1_to_v2(raw);
            serde_json::from_value(migrated).unwrap()
        };

        assert_eq!(parsed.format_version, 2);
        assert_eq!(parsed.scene_id, "scene.v1_forward");
        assert_eq!(parsed.root_nodes, vec!["node.root"]);
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.directional_lights.len(), 1);
        assert!(parsed.directional_lights[0].shadow_config.is_none());

        // Load through FakeSceneAssetLoader (full load path).
        let mut loader = FakeSceneAssetLoader::default();
        let scene = parsed.into_scene_with_loader(&mut loader).unwrap();
        assert_eq!(scene.directional_lights().len(), 1);
        assert!(scene.shadow_casting_directional_light_id().is_none());
    }

    #[test]
    fn v1_scene_with_extra_unknown_keys_still_loads() {
        // A v1 document produced by a tool that added editor-specific extras.
        let v1_json = r#"{
            "format_version": 1,
            "scene_id": "scene.v1_extras",
            "root_nodes": ["node.a"],
            "nodes": [
                {
                    "id": "node.a",
                    "parent": null,
                    "name": "A",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null
                }
            ],
            "lights": [],
            "directional_lights": [],
            "spot_lights": [],
            "environment": null,
            "editor": {},
            "custom_tool_metadata": {"version": "3.1.0"},
            "extra_field": true
        }"#;

        let parsed: SerializedScene = {
            let raw: serde_json::Value = serde_json::from_str(v1_json).unwrap();
            let migrated = super::migrate_v1_to_v2(raw);
            serde_json::from_value(migrated).unwrap()
        };

        assert_eq!(parsed.format_version, 2);
        assert_eq!(parsed.scene_id, "scene.v1_extras");

        let mut loader = FakeSceneAssetLoader::default();
        let scene = parsed.into_scene_with_loader(&mut loader).unwrap();
        assert!(scene.root().is_some());
    }

    #[test]
    fn validation_accepts_v1_document_through_migration() {
        let v1_json = r#"{
            "format_version": 1,
            "scene_id": "scene.v1_validation",
            "root_nodes": ["node.x"],
            "nodes": [
                {
                    "id": "node.x",
                    "parent": null,
                    "name": "X",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null
                }
            ],
            "lights": [],
            "directional_lights": [],
            "spot_lights": [],
            "environment": null,
            "editor": {}
        }"#;

        // This should pass because the migration bumps format_version to 2.
        validate_scene_str(v1_json).unwrap();
    }

    #[test]
    fn strict_v2_rejects_unknown_fields() {
        let top_level_extra = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.v2_strict_top",
                "root_nodes": [],
                "nodes": [],
                "lights": [],
                "directional_lights": [],
                "spot_lights": [],
                "environment": null,
                "editor": {},
                "custom_tool_metadata": {"version": "3.1.0"}
            }"#,
        )
        .unwrap_err();
        assert!(top_level_extra.diagnostics().iter().any(|diagnostic| {
            diagnostic.code == "scene.parse"
                && diagnostic
                    .message
                    .contains("unknown field `custom_tool_metadata`")
        }));

        let shadow_config_extra = validate_scene_str(
            r#"{
                "format_version": 2,
                "scene_id": "scene.v2_strict_shadow_config",
                "root_nodes": [],
                "nodes": [],
                "lights": [],
                "directional_lights": [
                    {
                        "id": "directional.strict",
                        "kind": "directional",
                        "direction": [0.0, 1.0, 0.0],
                        "color": [1.0, 1.0, 1.0],
                        "intensity": 1.0,
                        "shadow_config": {
                            "enabled": true,
                            "shadow_map_size": 2048,
                            "cascade_count": 1,
                            "cascade_split_lambda": 0.5,
                            "extra": true
                        }
                    }
                ],
                "spot_lights": [],
                "environment": null,
                "editor": {}
            }"#,
        )
        .unwrap_err();
        assert!(shadow_config_extra.diagnostics().iter().any(|diagnostic| {
            diagnostic.code == "scene.parse" && diagnostic.message.contains("unknown field `extra`")
        }));
    }

    // ── H-A4: shadow config persistence tests ──────────────────────────

    #[test]
    fn directional_shadow_config_round_trips_through_save_load() {
        let mut scene = Scene::new();
        let root = scene.create_node_default(None).unwrap();
        scene.set_node_name(root, "Shadow Test Root").unwrap();

        let light = DirectionalLight {
            direction: Vec3::new(0.25, 1.0, 0.5),
            color: Vec3::ONE,
            intensity: 1.5,
        };
        let id = scene.add_directional_light(light).unwrap();

        let cfg = DirectionalShadowConfig {
            enabled: true,
            shadow_map_size: 4096,
            cascade_count: 3,
            cascade_split_lambda: 0.75,
        };
        scene.set_directional_shadow_config(id, cfg).unwrap();

        assert_eq!(scene.shadow_casting_directional_light_id(), Some(id));
        let cfg_from_record = scene
            .world
            .get_directional_light_record(id)
            .and_then(|r| r.directional_shadow_config);
        assert!(cfg_from_record.is_some_and(|c| c.shadow_map_size == 4096));

        let serialized = SerializedScene::from_scene(&scene);
        let json = serde_json::to_string_pretty(&serialized).unwrap();
        assert!(json.contains("\"format_version\": 2"));
        assert!(json.contains("\"shadow_config\""));
        assert!(json.contains("\"shadow_map_size\": 4096"));
        assert!(json.contains("\"cascade_count\": 3"));
        assert!(json.contains("\"cascade_split_lambda\": 0.75"));
        assert!(json.contains("\"enabled\": true"));

        let parsed: SerializedScene = serde_json::from_str(&json).unwrap();
        let mut loader = FakeSceneAssetLoader::default();
        let loaded = parsed.into_scene_with_loader(&mut loader).unwrap();

        let loaded_id = loaded
            .directional_lights()
            .first()
            .map(|_| loaded.shadow_casting_directional_light_id())
            .flatten();
        assert!(loaded_id.is_some());
        let loaded_cfg = loaded
            .world
            .get_directional_light_record(loaded_id.unwrap())
            .and_then(|r| r.directional_shadow_config);
        assert!(loaded_cfg.is_some_and(|c| {
            c.enabled
                && c.shadow_map_size == 4096
                && c.cascade_count == 3
                && (c.cascade_split_lambda - 0.75).abs() < 0.001
        }));
    }

    #[test]
    fn shadow_config_defaults_are_applied_when_absent_from_json() {
        // A v2 scene with a directional light but no shadow_config.
        let json = r#"{
            "format_version": 2,
            "scene_id": "scene.no_shadow_cfg",
            "root_nodes": ["node.root"],
            "nodes": [
                {
                    "id": "node.root",
                    "parent": null,
                    "name": "Root",
                    "transform": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    "asset": null
                }
            ],
            "lights": [],
            "directional_lights": [
                {
                    "id": "directional.000001",
                    "kind": "directional",
                    "direction": [0.0, 1.0, 0.0],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 1.0
                }
            ],
            "spot_lights": [],
            "environment": null,
            "editor": {}
        }"#;

        let parsed: SerializedScene = serde_json::from_str(json).unwrap();
        assert!(parsed.directional_lights[0].shadow_config.is_none());

        let mut loader = FakeSceneAssetLoader::default();
        let scene = parsed.into_scene_with_loader(&mut loader).unwrap();
        // No shadow config was provided, so none should be active.
        assert!(scene.shadow_casting_directional_light_id().is_none());
        // All directional light records should have no shadow config.
        let has_shadow_config = scene
            .world
            .serializable_directional_lights()
            .any(|(id, _)| {
                scene
                    .world
                    .get_directional_light_record(id)
                    .and_then(|r| r.directional_shadow_config)
                    .is_some()
            });
        assert!(!has_shadow_config);
    }

    // ── Fixture-based integration tests ────────────────────────────────

    #[test]
    fn load_v1_fixture_file_through_migration() {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scenes/v1-forward-compatible.engine.scene.json");

        let mut loader = FakeSceneAssetLoader::default();
        let scene = Scene::load_with_loader(&fixture_path, &mut loader).unwrap();

        assert_eq!(scene.directional_lights().len(), 1);
        // v1 fixture has no shadow_config, so nothing should be enabled.
        assert!(scene.shadow_casting_directional_light_id().is_none());
        let has_shadow = scene
            .world
            .serializable_directional_lights()
            .any(|(id, _)| {
                scene
                    .world
                    .get_directional_light_record(id)
                    .and_then(|r| r.directional_shadow_config)
                    .is_some()
            });
        assert!(!has_shadow);
    }

    #[test]
    fn load_v2_shadow_owner_fixture_preserves_shadow_config() {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scenes/v2-shadow-owner.engine.scene.json");

        let mut loader = FakeSceneAssetLoader::default();
        let scene = Scene::load_with_loader(&fixture_path, &mut loader).unwrap();

        let directional = scene.directional_lights();
        assert_eq!(directional.len(), 2);

        // The shadow caster should be the first light (enabled in the fixture).
        let caster_id = scene.shadow_casting_directional_light_id();
        assert!(caster_id.is_some());
        let cfg = scene
            .world
            .get_directional_light_record(caster_id.unwrap())
            .and_then(|r| r.directional_shadow_config);
        assert!(cfg.is_some_and(|c| {
            c.enabled
                && c.shadow_map_size == 4096
                && c.cascade_count == 4
                && (c.cascade_split_lambda - 0.75).abs() < 0.001
        }));
    }

    #[test]
    fn v1_migration_adds_null_shadow_config_to_directional_lights() {
        // v1 directional lights lacked shadow_config; migration should add null.
        let v1_json = r#"{
            "format_version": 1,
            "scene_id": "scene.v1_directional",
            "root_nodes": [],
            "nodes": [],
            "lights": [],
            "directional_lights": [
                {
                    "id": "directional.000001",
                    "kind": "directional",
                    "direction": [0.0, -1.0, 0.0],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 1.0
                }
            ],
            "spot_lights": [],
            "environment": null,
            "editor": {}
        }"#;

        let raw: serde_json::Value = serde_json::from_str(v1_json).unwrap();
        let migrated = super::migrate_v1_to_v2(raw);
        let parsed: SerializedScene = serde_json::from_value(migrated).unwrap();

        assert_eq!(parsed.format_version, 2);
        assert!(parsed.directional_lights[0].shadow_config.is_none());
    }
}
