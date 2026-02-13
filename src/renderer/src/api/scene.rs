use std::collections::HashMap;

use glam::{Mat4, Vec3};

use crate::data::handles::{EnvironmentHandle, MeshHandle};
use crate::scene::render_submission::RenderSubmission;
use crate::scene::scene_world::{PointLightRefError, SceneNodeRefError, SceneWorld};
use crate::scene::SceneNodeId;

use super::errors::SceneError;

/// Point light ID with slot+generation semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PointLightId {
    pub slot: u32,
    pub generation: u32,
}

/// Point light definition.
#[derive(Copy, Clone, Debug, PartialEq)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Public scene facade.
#[derive(Default)]
pub struct Scene {
    world: SceneWorld,
}

impl Scene {
    /// Thread: Any
    /// May Stall: No
    pub fn new() -> Self {
        Self {
            world: SceneWorld::new(),
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

        Ok(self
            .world
            .add_node_with_parts(parent, transform, Vec::new()))
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
    pub fn transform(&self, node: SceneNodeId) -> Result<Mat4, SceneError> {
        self.validate_node(node)?;

        let Some(node_ref) = self.world.get_node(node) else {
            return Err(SceneError::InvalidNode(node));
        };
        Ok(node_ref.local_transform)
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

    pub(crate) fn from_world(world: SceneWorld) -> Self {
        Self { world }
    }

    pub(crate) fn world_mut(&mut self) -> &mut SceneWorld {
        &mut self.world
    }

    pub(crate) fn update_camera(&mut self, view: Mat4, projection: Mat4, position: Vec3) {
        self.world.update_camera(view, projection, position);
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

#[cfg(test)]
mod tests {
    use super::{PointLight, Scene, SceneFragment, SceneFragmentNode, SceneFragmentNodeId};
    use crate::api::errors::SceneError;
    use crate::data::handles::MeshHandle;
    use glam::{Mat4, Vec3};

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
        let id = scene.create_point_light(light).expect("create should succeed");

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
        let id = scene.create_point_light(light).expect("should clamp and succeed");
        scene.remove_point_light(id).unwrap();
    }
}
