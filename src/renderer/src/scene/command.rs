//! Command pattern for undo/redo support on scene mutations.
//!
//! Each command stores enough state to reverse its effect. Commands are pushed
//! onto a `CommandHistory` stack of bounded depth.

use crate::api::{SceneAssetReference, SceneError, SceneFragment, SceneFragmentNodeId};
use crate::data::handles::MeshHandle;
use crate::scene::scene_world::{
    RestorableSceneSubtree, SceneNode, SceneNodeId, SceneNodeRefError, SceneWorld,
};
use glam::Mat4;

/// A reversible scene mutation.
pub trait Command: Send {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError>;
    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError>;
    fn description(&self) -> &str;
    fn node_remap(&self) -> Option<SceneNodeRemap> {
        None
    }
    fn created_node(&self) -> Option<SceneNodeId> {
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SceneNodeRemap {
    pub old: SceneNodeId,
    pub new: SceneNodeId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandResult {
    pub description: String,
    pub node_remap: Option<SceneNodeRemap>,
    pub created_node: Option<SceneNodeId>,
}

/// Bounded undo/redo history stack.
pub struct CommandHistory {
    undo_stack: Vec<Box<dyn Command>>,
    redo_stack: Vec<Box<dyn Command>>,
    max_depth: usize,
}

impl CommandHistory {
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::with_capacity(max_depth),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    /// Execute a command and push it onto the undo stack.
    /// Clears the redo stack (new action invalidates redo history).
    pub fn execute(
        &mut self,
        mut cmd: Box<dyn Command>,
        world: &mut SceneWorld,
    ) -> Result<CommandResult, SceneError> {
        cmd.execute(world)?;
        let result = CommandResult {
            description: cmd.description().to_string(),
            node_remap: cmd.node_remap(),
            created_node: cmd.created_node(),
        };
        self.redo_stack.clear();
        if self.undo_stack.len() >= self.max_depth {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(cmd);
        Ok(result)
    }

    /// Undo the most recent command.
    pub fn undo(&mut self, world: &mut SceneWorld) -> Result<CommandResult, SceneError> {
        let mut cmd = self
            .undo_stack
            .pop()
            .ok_or_else(|| SceneError::MergeFailed("nothing to undo".into()))?;
        cmd.undo(world)?;
        let result = CommandResult {
            description: cmd.description().to_string(),
            node_remap: cmd.node_remap(),
            created_node: cmd.created_node(),
        };
        self.redo_stack.push(cmd);
        Ok(result)
    }

    /// Redo the most recently undone command.
    pub fn redo(&mut self, world: &mut SceneWorld) -> Result<CommandResult, SceneError> {
        let mut cmd = self
            .redo_stack
            .pop()
            .ok_or_else(|| SceneError::MergeFailed("nothing to redo".into()))?;
        cmd.execute(world)?;
        let result = CommandResult {
            description: cmd.description().to_string(),
            node_remap: cmd.node_remap(),
            created_node: cmd.created_node(),
        };
        if self.undo_stack.len() >= self.max_depth {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(cmd);
        Ok(result)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Concrete commands
// ---------------------------------------------------------------------------

/// Set a node's local transform, remembering the old one.
pub struct SetTransformCommand {
    node: SceneNodeId,
    new_transform: Mat4,
    old_transform: Mat4,
}

impl SetTransformCommand {
    pub fn new(node: SceneNodeId, transform: Mat4) -> Self {
        Self {
            node,
            new_transform: transform,
            old_transform: Mat4::IDENTITY,
        }
    }
}

impl Command for SetTransformCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        let node_ref = world
            .get_node_mut(self.node)
            .ok_or(SceneError::InvalidNode(self.node))?;
        self.old_transform = node_ref.local_transform;
        node_ref.local_transform = self.new_transform;
        node_ref.dirty = true;
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        let node_ref = world
            .get_node_mut(self.node)
            .ok_or(SceneError::InvalidNode(self.node))?;
        node_ref.local_transform = self.old_transform;
        node_ref.dirty = true;
        Ok(())
    }

    fn description(&self) -> &str {
        "set_transform"
    }
}

/// Add a node under a parent, remembering the node ID for removal.
pub struct AddNodeCommand {
    parent: Option<SceneNodeId>,
    transform: Mat4,
    meshes: Vec<MeshHandle>,
    created_node: Option<SceneNodeId>,
}

impl AddNodeCommand {
    pub fn new(parent: Option<SceneNodeId>, transform: Mat4, meshes: Vec<MeshHandle>) -> Self {
        Self {
            parent,
            transform,
            meshes,
            created_node: None,
        }
    }
}

impl Command for AddNodeCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        let id = world.add_node_with_parts(self.parent, self.transform, self.meshes.clone());
        self.created_node = Some(id);
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if let Some(node) = self.created_node {
            world.remove_node(node);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "add_node"
    }
}

/// Place a model or prefab fragment into a scene with a durable asset reference.
pub struct PlaceAssetCommand {
    parent: Option<SceneNodeId>,
    transform: Mat4,
    fragment: SceneFragment,
    asset: SceneAssetReference,
    display_name: String,
    tags: Vec<String>,
    stable_id_base: String,
    created_root: Option<SceneNodeId>,
}

impl PlaceAssetCommand {
    pub fn new(
        parent: Option<SceneNodeId>,
        transform: Mat4,
        fragment: SceneFragment,
        asset: SceneAssetReference,
        display_name: impl Into<String>,
        tags: Vec<String>,
        stable_id_base: impl Into<String>,
    ) -> Self {
        Self {
            parent,
            transform,
            fragment,
            asset,
            display_name: display_name.into(),
            tags,
            stable_id_base: stable_id_base.into(),
            created_root: None,
        }
    }
}

impl Command for PlaceAssetCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if let Some(parent) = self.parent {
            world
                .validate_node_ref(parent)
                .map_err(|err| map_parent_ref_error(parent, err))?;
        }

        let mount = mount_fragment_for_asset(
            world,
            self.parent,
            self.transform,
            self.fragment.clone(),
            self.asset.clone(),
            &self.display_name,
            &self.tags,
            &self.stable_id_base,
        )?;
        self.created_root = Some(mount);
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if let Some(root) = self.created_root.take() {
            world.remove_node(root);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "place_asset"
    }

    fn created_node(&self) -> Option<SceneNodeId> {
        self.created_root
    }
}

/// Remove a node, remembering its state for restoration.
pub struct RemoveNodeCommand {
    active_node: SceneNodeId,
    snapshot: Option<RestorableSceneSubtree>,
    last_remap: Option<SceneNodeRemap>,
    executed: bool,
}

fn mount_fragment_for_asset(
    world: &mut SceneWorld,
    parent: Option<SceneNodeId>,
    placement_transform: Mat4,
    fragment: SceneFragment,
    asset: SceneAssetReference,
    display_name: &str,
    tags: &[String],
    stable_id_base: &str,
) -> Result<SceneNodeId, SceneError> {
    let (nodes, root, skybox) = fragment.into_parts();
    if skybox.is_some() {
        return Err(SceneError::MergeFailed(
            "asset placement fragments cannot set scene skybox".to_string(),
        ));
    }
    if nodes.is_empty() {
        return Err(SceneError::MergeFailed(
            "cannot place an empty scene fragment".to_string(),
        ));
    }

    let fragment_root = resolve_fragment_root(&nodes, root)?;
    let merge_plan = build_fragment_merge_plan(&nodes, fragment_root)?;
    let mut mapping = std::collections::HashMap::with_capacity(nodes.len());

    for (sequence, (fragment_node, fragment_parent)) in merge_plan.into_iter().enumerate() {
        let source = &nodes[fragment_node.index as usize];
        let scene_parent = if let Some(fragment_parent_id) = fragment_parent {
            Some(*mapping.get(&fragment_parent_id).ok_or_else(|| {
                SceneError::MergeFailed(
                    "fragment merge invariant violated: parent was not cloned".to_string(),
                )
            })?)
        } else {
            parent
        };
        let is_root = fragment_node == fragment_root;
        let transform = if is_root {
            placement_transform * source.local_transform
        } else {
            source.local_transform
        };
        let mut node = SceneNode {
            parent: scene_parent,
            local_transform: transform,
            meshes: source.meshes.clone(),
            ..SceneNode::default()
        };
        node.stable_id = Some(if is_root {
            stable_id_base.to_string()
        } else {
            format!("{stable_id_base}.part.{sequence:03}")
        });
        node.name = if is_root {
            display_name.to_string()
        } else {
            format!("{display_name} Part {sequence}")
        };
        node.tags = tags.to_vec();
        if is_root {
            node.asset = Some(asset.clone());
        }

        let created = world.add_node(scene_parent, node);
        mapping.insert(fragment_node, created);
    }

    mapping.get(&fragment_root).copied().ok_or_else(|| {
        SceneError::MergeFailed("fragment merge failed to map root node".to_string())
    })
}

fn resolve_fragment_root(
    nodes: &[crate::api::SceneFragmentNode],
    root: Option<SceneFragmentNodeId>,
) -> Result<SceneFragmentNodeId, SceneError> {
    if let Some(root_id) = root {
        if root_id.index as usize >= nodes.len() {
            return Err(SceneError::MergeFailed(format!(
                "fragment root {} is out of bounds",
                root_id.index
            )));
        }
        return Ok(root_id);
    }

    let roots: Vec<_> = nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| {
            node.parent
                .is_none()
                .then_some(SceneFragmentNodeId::new(idx as u32))
        })
        .collect();
    if roots.len() == 1 {
        return Ok(roots[0]);
    }

    Err(SceneError::MergeFailed(
        "fragment root is missing or ambiguous".to_string(),
    ))
}

fn build_fragment_merge_plan(
    nodes: &[crate::api::SceneFragmentNode],
    root: SceneFragmentNodeId,
) -> Result<Vec<(SceneFragmentNodeId, Option<SceneFragmentNodeId>)>, SceneError> {
    let mut state = vec![0_u8; nodes.len()];
    let mut resolved_parent = vec![None; nodes.len()];
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

    visit_fragment_node(
        nodes,
        root,
        None,
        &mut state,
        &mut resolved_parent,
        &mut order,
    )?;

    if state.iter().any(|state| *state != 2) {
        return Err(SceneError::MergeFailed(
            "fragment contains disconnected nodes".to_string(),
        ));
    }

    Ok(order)
}

fn visit_fragment_node(
    nodes: &[crate::api::SceneFragmentNode],
    node_id: SceneFragmentNodeId,
    parent_id: Option<SceneFragmentNodeId>,
    state: &mut [u8],
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
    if state[idx] == 1 {
        return Err(SceneError::CycleDetected);
    }
    if state[idx] == 2 {
        if resolved_parent[idx] != parent_id {
            return Err(SceneError::MergeFailed(format!(
                "fragment node {} is referenced by multiple parents",
                node_id.index
            )));
        }
        return Ok(());
    }

    state[idx] = 1;
    resolved_parent[idx] = parent_id;
    order.push((node_id, parent_id));
    for child in nodes[idx].children.iter().copied() {
        visit_fragment_node(nodes, child, Some(node_id), state, resolved_parent, order)?;
    }
    state[idx] = 2;
    Ok(())
}

fn map_parent_ref_error(parent: SceneNodeId, err: SceneNodeRefError) -> SceneError {
    match err {
        SceneNodeRefError::GenerationMismatch => SceneError::StaleNode(parent),
        SceneNodeRefError::OutOfBounds | SceneNodeRefError::Vacant => {
            SceneError::InvalidParent(parent)
        }
    }
}

impl RemoveNodeCommand {
    pub fn new(node: SceneNodeId) -> Self {
        Self {
            active_node: node,
            snapshot: None,
            last_remap: None,
            executed: false,
        }
    }

    pub fn active_node(&self) -> SceneNodeId {
        self.active_node
    }
}

impl Command for RemoveNodeCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        let target = self.active_node;
        self.snapshot = Some(
            world
                .clone_subtree(target)
                .ok_or(SceneError::InvalidNode(target))?,
        );
        world.remove_node(target);
        self.last_remap = None;
        self.executed = true;
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        let Some(snapshot) = self.snapshot.clone().filter(|_| self.executed) else {
            return Ok(());
        };
        let restored = world.restore_subtree(snapshot);
        self.last_remap = Some(SceneNodeRemap {
            old: self.active_node,
            new: restored,
        });
        self.active_node = restored;
        Ok(())
    }

    fn description(&self) -> &str {
        "remove_node"
    }

    fn node_remap(&self) -> Option<SceneNodeRemap> {
        self.last_remap
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandHistory, PlaceAssetCommand, RemoveNodeCommand, SetTransformCommand};
    use crate::api::scene::{SceneAssetReference, SceneFragment};
    use crate::data::handles::MeshHandle;
    use crate::scene::scene_world::{SceneNode, SceneWorld};
    use glam::{Mat4, Vec3};

    #[test]
    fn transform_command_undo_redo_is_transactional() {
        let mut world = SceneWorld::new();
        let node = world.add_node(None, SceneNode::default());
        let mut history = CommandHistory::new(8);

        history
            .execute(
                Box::new(SetTransformCommand::new(
                    node,
                    Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
                )),
                &mut world,
            )
            .unwrap();
        assert_eq!(
            world.get_node(node).unwrap().local_transform,
            Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))
        );

        history.undo(&mut world).unwrap();
        assert_eq!(
            world.get_node(node).unwrap().local_transform,
            Mat4::IDENTITY
        );

        history.redo(&mut world).unwrap();
        assert_eq!(
            world.get_node(node).unwrap().local_transform,
            Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))
        );
    }

    #[test]
    fn remove_undo_restores_subtree_with_remap_and_stale_old_handle() {
        let mut world = SceneWorld::new();
        let root = world.add_node(
            None,
            SceneNode {
                name: "Root".to_string(),
                stable_id: Some("node.root".to_string()),
                ..SceneNode::default()
            },
        );
        let child = world.add_node(
            Some(root),
            SceneNode {
                name: "Child".to_string(),
                stable_id: Some("node.child".to_string()),
                local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                ..SceneNode::default()
            },
        );
        let mut history = CommandHistory::new(8);

        history
            .execute(Box::new(RemoveNodeCommand::new(root)), &mut world)
            .unwrap();
        assert!(!world.is_valid_node_id(root));
        assert!(!world.is_valid_node_id(child));

        let result = history.undo(&mut world).unwrap();
        let restored = result.node_remap.expect("remove undo should remap").new;
        assert_ne!(restored, root);
        assert!(!world.is_valid_node_id(root));
        assert_eq!(
            world.get_node(restored).unwrap().stable_id.as_deref(),
            Some("node.root")
        );
        let restored_child = world.get_node(restored).unwrap().children[0];
        assert_eq!(
            world.get_node(restored_child).unwrap().stable_id.as_deref(),
            Some("node.child")
        );

        history.redo(&mut world).unwrap();
        assert!(!world.is_valid_node_id(restored));
        assert!(!world.is_valid_node_id(restored_child));
    }

    #[test]
    fn place_asset_command_is_undoable_and_recreates_asset_reference() {
        let mut world = SceneWorld::new();
        let root = world.add_node(
            None,
            SceneNode {
                name: "Root".to_string(),
                stable_id: Some("node.root".to_string()),
                ..SceneNode::default()
            },
        );
        let mut fragment = SceneFragment::new();
        fragment
            .add_node(None, Mat4::IDENTITY, vec![MeshHandle::new(3, 0)])
            .expect("fragment node");
        let mut history = CommandHistory::new(8);

        let result = history
            .execute(
                Box::new(PlaceAssetCommand::new(
                    Some(root),
                    Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
                    fragment,
                    SceneAssetReference::new(
                        "editor_sample.wall.stone_2m",
                        Some("prefabs/wall_straight_2m.obj".into()),
                    ),
                    "Stone Wall 2m",
                    vec!["wall".to_string(), "chunk".to_string()],
                    "node.placed.wall.000001",
                )),
                &mut world,
            )
            .expect("place asset");

        let placed = result.created_node.expect("placement returns node");
        let placed_node = world.get_node(placed).expect("placed node exists");
        assert_eq!(placed_node.parent, Some(root));
        assert_eq!(placed_node.name, "Stone Wall 2m");
        assert_eq!(
            placed_node.asset.as_ref().map(|asset| asset.id.as_str()),
            Some("editor_sample.wall.stone_2m")
        );

        history.undo(&mut world).expect("undo placement");
        assert!(!world.is_valid_node_id(placed));

        let redo = history.redo(&mut world).expect("redo placement");
        let redone = redo.created_node.expect("redo returns new node");
        assert_ne!(redone, placed);
        assert_eq!(
            world
                .get_node(redone)
                .and_then(|node| node.asset.as_ref())
                .map(|asset| asset.id.as_str()),
            Some("editor_sample.wall.stone_2m")
        );
    }
}
