//! Built-in undoable object commands for the unified object API.
//!
//! Every command uses a state machine (`Prepared` / `Executed` / `Undone`)
//! to prevent accidental reuse and to anchor redo on persistent identity.
//!
//! ## Failure contract
//! - **Failed execute:** world unchanged, redo stack NOT cleared.
//! - **Failed undo:** command stays at undo top.
//! - **Failed redo:** command stays at redo top.
//!
//! All commands use the prepare/commit lifecycle for failure atomicity.

use crate::api::scene::{DirectionalLightId, PointLightId, SpotLightId};
use crate::api::{CommandError, SceneError};
use crate::object::component::{
    commit_full_state_replacement, ComponentEnvelope, ComponentInstanceId,
    ComponentKey,
};
use crate::object::identity::ObjectId;
use crate::object::{ObjectParent, ObjectRemap};
use crate::scene::command::Command;
use crate::scene::object_store::{DetachedLightSnapshot, ObjectHandle, ObjectRecord};
use crate::scene::scene_world::{RestorableSceneSubtree, SceneNodeId, SceneWorld};
use engine_events::{ObjectKind, SceneObjectId};
use glam::Mat4;
use std::collections::HashMap;

// ── Command state machine ───────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CommandState {
    /// Freshly created, never executed.
    Prepared,
    /// Currently live (execute succeeded, not yet undone).
    Executed,
    /// Undo succeeded, can be redone.
    Undone,
}

// ── SetObjectTransformCommand ───────────────────────────────────────────

/// Set the transform of any object kind using a canonical [`Mat4`].
///
/// Anchored by persistent [`SceneObjectId`]; redo resolves the current
/// runtime handle via the reverse index.
pub struct SetObjectTransformCommand {
    persistent_id: SceneObjectId,
    kind: ObjectKind,
    new_transform: Mat4,
    old_transform: Mat4,
    state: CommandState,
}

impl SetObjectTransformCommand {
    pub fn new(persistent_id: SceneObjectId, kind: ObjectKind, transform: Mat4) -> Self {
        Self {
            persistent_id,
            kind,
            new_transform: transform,
            old_transform: Mat4::IDENTITY,
            state: CommandState::Prepared,
        }
    }
}

impl Command for SetObjectTransformCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "SetObjectTransformCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let current_id = resolve_current_id(world, &self.persistent_id, self.kind)?;
        if !self.new_transform.is_finite() {
            return Err(SceneError::InvalidMutation(
                "transform must contain only finite values".into(),
            ));
        }

        use crate::object::{is_identity_basis_matrix, is_rigid_direction_only, is_rigid_matrix};

        match self.kind {
            ObjectKind::Node => {
                let nid = SceneNodeId::new(current_id.slot(), current_id.generation());
                let node = world
                    .get_node_mut(nid)
                    .ok_or(SceneError::InvalidNode(nid))?;
                self.old_transform = node.local_transform;
                node.local_transform = self.new_transform;
                world.invalidate_derived_state(nid);
            }
            ObjectKind::PointLight => {
                if !is_identity_basis_matrix(&self.new_transform) {
                    return Err(SceneError::InvalidMutation(
                        "point light transform must be translation-only".into(),
                    ));
                }
                let pl_id = PointLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .point_light_entry_mut(pl_id)
                    .ok_or(SceneError::StalePointLight(pl_id))?;
                self.old_transform = Mat4::from_translation(entry.0.position);
                entry.0.position = self.new_transform.w_axis.truncate();
            }
            ObjectKind::SpotLight => {
                if !is_rigid_matrix(&self.new_transform) {
                    return Err(SceneError::InvalidMutation(
                        "spot light transform must be rigid".into(),
                    ));
                }
                let sl_id = SpotLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .spot_light_entry_mut(sl_id)
                    .ok_or(SceneError::StaleSpotLight(sl_id))?;
                let old_pos = entry.0.position;
                let old_dir = entry.0.direction;
                self.old_transform =
                    crate::object::ObjectTransform::rigid_from_position_direction(old_pos, old_dir);
                entry.0.position = self.new_transform.w_axis.truncate();
                entry.0.direction =
                    crate::object::ObjectTransform::direction_from_rigid(&self.new_transform);
            }
            ObjectKind::DirectionalLight => {
                if !is_rigid_direction_only(&self.new_transform) {
                    return Err(SceneError::InvalidMutation(
                        "directional light transform must be rigid direction-only".into(),
                    ));
                }
                let dl_id = DirectionalLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .directional_light_entry_mut(dl_id)
                    .ok_or(SceneError::StaleDirectionalLight(dl_id))?;
                self.old_transform =
                    crate::object::ObjectTransform::rigid_from_direction(entry.0.direction);
                entry.0.direction =
                    crate::object::ObjectTransform::direction_from_rigid(&self.new_transform);
            }
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "SetObjectTransformCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let current_id = resolve_current_id(world, &self.persistent_id, self.kind)?;

        match self.kind {
            ObjectKind::Node => {
                let nid = SceneNodeId::new(current_id.slot(), current_id.generation());
                let node = world
                    .get_node_mut(nid)
                    .ok_or(SceneError::InvalidNode(nid))?;
                node.local_transform = self.old_transform;
                world.invalidate_derived_state(nid);
            }
            ObjectKind::PointLight => {
                let pl_id = PointLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .point_light_entry_mut(pl_id)
                    .ok_or(SceneError::StalePointLight(pl_id))?;
                entry.0.position = self.old_transform.w_axis.truncate();
            }
            ObjectKind::SpotLight => {
                let sl_id = SpotLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .spot_light_entry_mut(sl_id)
                    .ok_or(SceneError::StaleSpotLight(sl_id))?;
                entry.0.position = self.old_transform.w_axis.truncate();
                entry.0.direction =
                    crate::object::ObjectTransform::direction_from_rigid(&self.old_transform);
            }
            ObjectKind::DirectionalLight => {
                let dl_id = DirectionalLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let entry = world
                    .directional_light_entry_mut(dl_id)
                    .ok_or(SceneError::StaleDirectionalLight(dl_id))?;
                entry.0.direction =
                    crate::object::ObjectTransform::direction_from_rigid(&self.old_transform);
            }
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "set_object_transform"
    }
}

// ── SetObjectParentCommand ──────────────────────────────────────────────

/// Change the parent of an object (node reparent or light regroup).
///
/// Anchored by persistent [`SceneObjectId`]; redo resolves the current
/// runtime handle.
pub struct SetObjectParentCommand {
    persistent_id: SceneObjectId,
    kind: ObjectKind,
    new_parent: ObjectParent,
    new_parent_persistent: Option<SceneObjectId>,
    old_parent_persistent: Option<SceneObjectId>,
    old_node_parent_id: Option<SceneNodeId>,
    state: CommandState,
}

impl SetObjectParentCommand {
    pub fn new(
        persistent_id: SceneObjectId,
        kind: ObjectKind,
        parent: ObjectParent,
        parent_persistent: Option<SceneObjectId>,
    ) -> Self {
        Self {
            persistent_id,
            kind,
            new_parent: parent,
            new_parent_persistent: parent_persistent,
            old_parent_persistent: None,
            old_node_parent_id: None,
            state: CommandState::Prepared,
        }
    }
}

impl Command for SetObjectParentCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "SetObjectParentCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let current_id = resolve_current_id(world, &self.persistent_id, self.kind)?;

        match self.kind {
            ObjectKind::Node => {
                let node_id = SceneNodeId::new(current_id.slot(), current_id.generation());
                // Capture old parent before mutation.
                let node = world
                    .get_node(node_id)
                    .ok_or(SceneError::InvalidNode(node_id))?;
                self.old_node_parent_id = node.parent;
                self.old_parent_persistent = node.parent.and_then(|pid| {
                    world
                        .object_id_for_node(pid)
                        .and_then(|oid| world.object_persistent_id(oid))
                });

                // Resolve new parent.
                let new_parent_node_id = match &self.new_parent {
                    ObjectParent::None => None,
                    ObjectParent::Node(_oid) => {
                        let resolved = if let Some(ref pp) = self.new_parent_persistent {
                            world.find_object_by_persistent_id(pp)
                        } else {
                            None
                        };
                        let resolved = resolved.ok_or_else(|| {
                            SceneError::InvalidMutation("new parent object not found".into())
                        })?;
                        if resolved.kind() != ObjectKind::Node {
                            return Err(SceneError::InvalidMutation(
                                "parent must be a node".into(),
                            ));
                        }
                        Some(SceneNodeId::new(resolved.slot(), resolved.generation()))
                    }
                };

                world
                    .reparent_node(node_id, new_parent_node_id)
                    .map_err(|err| {
                        SceneError::InvalidMutation(format!("reparent failed: {err:?}"))
                    })?;
            }
            ObjectKind::PointLight => {
                let pl_id = PointLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_point_light_record_mut(pl_id)
                    .ok_or(SceneError::StalePointLight(pl_id))?;
                self.old_parent_persistent = record.light_group_parent.clone();
                record.light_group_parent = self.new_parent_persistent.clone();
            }
            ObjectKind::DirectionalLight => {
                let dl_id = DirectionalLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_directional_light_record_mut(dl_id)
                    .ok_or(SceneError::StaleDirectionalLight(dl_id))?;
                self.old_parent_persistent = record.light_group_parent.clone();
                record.light_group_parent = self.new_parent_persistent.clone();
            }
            ObjectKind::SpotLight => {
                let sl_id = SpotLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_spot_light_record_mut(sl_id)
                    .ok_or(SceneError::StaleSpotLight(sl_id))?;
                self.old_parent_persistent = record.light_group_parent.clone();
                record.light_group_parent = self.new_parent_persistent.clone();
            }
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "SetObjectParentCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let current_id = resolve_current_id(world, &self.persistent_id, self.kind)?;

        match self.kind {
            ObjectKind::Node => {
                let node_id = SceneNodeId::new(current_id.slot(), current_id.generation());
                world.reparent_node(node_id, self.old_node_parent_id).map_err(|err| {
                    SceneError::InvalidMutation(format!("undo reparent failed: {err:?}"))
                })?;
            }
            ObjectKind::PointLight => {
                let pl_id = PointLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_point_light_record_mut(pl_id)
                    .ok_or(SceneError::StalePointLight(pl_id))?;
                record.light_group_parent = self.old_parent_persistent.clone();
            }
            ObjectKind::DirectionalLight => {
                let dl_id = DirectionalLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_directional_light_record_mut(dl_id)
                    .ok_or(SceneError::StaleDirectionalLight(dl_id))?;
                record.light_group_parent = self.old_parent_persistent.clone();
            }
            ObjectKind::SpotLight => {
                let sl_id = SpotLightId {
                    slot: current_id.slot(),
                    generation: current_id.generation(),
                };
                let record = world
                    .get_spot_light_record_mut(sl_id)
                    .ok_or(SceneError::StaleSpotLight(sl_id))?;
                record.light_group_parent = self.old_parent_persistent.clone();
            }
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "set_object_parent"
    }
}

// ── RemoveObjectsCommand ─────────────────────────────────────────────────

/// Remove one or more objects, capturing full state for restoration.
///
/// Nodes are snapshotted as subtrees with detached grouped lights.
/// Lights are snapshotted individually.
///
/// On redo, objects are re-resolved by persistent identity. If an object
/// was restored via undo and its runtime handle changed, the redo follows
/// the new runtime handle.
pub struct RemoveObjectsCommand {
    /// Persistent IDs of the objects to remove, in deterministic order.
    source_persistent_ids: Vec<SceneObjectId>,
    /// Kinds, parallel to source_persistent_ids.
    source_kinds: Vec<ObjectKind>,
    /// Node subtree snapshots (keyed by persistent ID).
    node_snapshots: HashMap<SceneObjectId, RestorableSceneSubtree>,
    /// Detached lights that were grouped under removed nodes.
    detached_lights: Vec<DetachedLightSnapshot>,
    /// Point light snapshots (keyed by persistent ID).
    point_light_snapshots: HashMap<SceneObjectId, crate::api::scene::PointLight>,
    /// Directional light snapshots.
    directional_light_snapshots: HashMap<SceneObjectId, crate::api::scene::DirectionalLight>,
    /// Spot light snapshots.
    spot_light_snapshots: HashMap<SceneObjectId, crate::api::scene::SpotLight>,
    /// Light records for restoration.
    light_records: HashMap<SceneObjectId, ObjectRecord>,
    /// Set of all persistent IDs in removed node subtrees (for remaps).
    removed_subtree_ids: Vec<ObjectId>,
    /// Remaps populated after restore.
    remaps: Vec<ObjectRemap>,
    state: CommandState,
}

impl RemoveObjectsCommand {
    pub fn new(persistent_ids: Vec<SceneObjectId>, kinds: Vec<ObjectKind>) -> Self {
        Self {
            source_persistent_ids: persistent_ids,
            source_kinds: kinds,
            node_snapshots: HashMap::new(),
            detached_lights: Vec::new(),
            point_light_snapshots: HashMap::new(),
            directional_light_snapshots: HashMap::new(),
            spot_light_snapshots: HashMap::new(),
            light_records: HashMap::new(),
            removed_subtree_ids: Vec::new(),
            remaps: Vec::new(),
            state: CommandState::Prepared,
        }
    }
}

impl Command for RemoveObjectsCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "RemoveObjectsCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        // Phase 1: Snapshot everything.
        let mut node_snapshots_temp: Vec<(
            SceneObjectId,
            RestorableSceneSubtree,
            Vec<DetachedLightSnapshot>,
        )> = Vec::new();

        for (persistent_id, kind) in self
            .source_persistent_ids
            .iter()
            .zip(self.source_kinds.iter())
        {
            match kind {
                ObjectKind::Node => {
                    let handle = world
                        .find_object_by_persistent_id(persistent_id)
                        .ok_or_else(|| {
                            SceneError::InvalidMutation(format!(
                                "object with persistent id {persistent_id} not found"
                            ))
                        })?;
                    let node_id = SceneNodeId::new(handle.slot(), handle.generation());
                    world
                        .validate_node_ref(node_id)
                        .map_err(|_| SceneError::InvalidNode(node_id))?;

                    // Collect subtree IDs for remap before removal.
                    let subtree_ids = world.subtree_node_ids_preorder(node_id);
                    self.removed_subtree_ids.extend(subtree_ids);

                    let (subtree, detached) = world
                        .prepare_remove_node_subtree(node_id)
                        .map_err(|e| {
                            SceneError::InvalidMutation(format!("snapshot failed: {e}"))
                        })?;
                    node_snapshots_temp.push((persistent_id.clone(), subtree, detached));
                }
                ObjectKind::PointLight => {
                    let handle = world
                        .find_object_by_persistent_id(persistent_id)
                        .ok_or_else(|| {
                            SceneError::InvalidMutation(format!(
                                "object {persistent_id} not found"
                            ))
                        })?;
                    let pl_id = PointLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let plan = world.prepare_remove_point_light(pl_id).ok_or_else(|| {
                        SceneError::InvalidMutation("point light not found".into())
                    })?;
                    self.point_light_snapshots
                        .insert(persistent_id.clone(), plan.light);
                    self.light_records
                        .insert(persistent_id.clone(), plan.record.clone());
                }
                ObjectKind::DirectionalLight => {
                    let handle = world
                        .find_object_by_persistent_id(persistent_id)
                        .ok_or_else(|| {
                            SceneError::InvalidMutation(format!(
                                "object {persistent_id} not found"
                            ))
                        })?;
                    let dl_id = DirectionalLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let plan = world
                        .prepare_remove_directional_light(dl_id)
                        .ok_or_else(|| {
                            SceneError::InvalidMutation(
                                "directional light not found".into(),
                            )
                        })?;
                    self.directional_light_snapshots
                        .insert(persistent_id.clone(), plan.light);
                    self.light_records
                        .insert(persistent_id.clone(), plan.record.clone());
                }
                ObjectKind::SpotLight => {
                    let handle = world
                        .find_object_by_persistent_id(persistent_id)
                        .ok_or_else(|| {
                            SceneError::InvalidMutation(format!(
                                "object {persistent_id} not found"
                            ))
                        })?;
                    let sl_id = SpotLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let plan = world.prepare_remove_spot_light(sl_id).ok_or_else(|| {
                        SceneError::InvalidMutation("spot light not found".into())
                    })?;
                    self.spot_light_snapshots
                        .insert(persistent_id.clone(), plan.light);
                    self.light_records
                        .insert(persistent_id.clone(), plan.record.clone());
                }
            }
        }

        // Phase 2: Commit all removals (infallible after snapshot).
        for (persistent_id, subtree, detached) in &node_snapshots_temp {
            let handle = world
                .find_object_by_persistent_id(persistent_id)
                .ok_or_else(|| {
                    SceneError::InvalidMutation(format!(
                        "object {persistent_id} vanished during removal"
                    ))
                })?;
            let node_id = SceneNodeId::new(handle.slot(), handle.generation());
            world.commit_remove_node_subtree(node_id, detached);
            self.node_snapshots
                .insert(persistent_id.clone(), subtree.clone());
            self.detached_lights.extend(detached.clone());
        }

        // Commit light removals via the existing remove_* methods.
        for persistent_id in &self.source_persistent_ids {
            if self.point_light_snapshots.contains_key(persistent_id) {
                if let Some(handle) = world.find_object_by_persistent_id(persistent_id) {
                    let pl_id = PointLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    world.remove_point_light(pl_id);
                }
            }
            if self.directional_light_snapshots.contains_key(persistent_id) {
                if let Some(handle) = world.find_object_by_persistent_id(persistent_id) {
                    let dl_id = DirectionalLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    world.remove_directional_light(dl_id);
                }
            }
            if self.spot_light_snapshots.contains_key(persistent_id) {
                if let Some(handle) = world.find_object_by_persistent_id(persistent_id) {
                    let sl_id = SpotLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    world.remove_spot_light(sl_id);
                }
            }
        }

        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "RemoveObjectsCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let mut new_remaps: Vec<ObjectRemap> = Vec::new();

        // Restore node subtrees.
        for persistent_id in &self.source_persistent_ids {
            if let Some(subtree) = self.node_snapshots.get(persistent_id) {
                let subtree = subtree.clone();
                let old_ids: Vec<ObjectId> =
                    self.removed_subtree_ids.clone(); // pre-removal IDs
                let restored_root = world.restore_subtree(subtree);
                let new_ids: Vec<ObjectId> = world.subtree_node_ids_preorder(restored_root);

                // Match old→new by position since they're in preorder.
                for (old, new) in old_ids.into_iter().zip(new_ids.into_iter()) {
                    let p = world
                        .get_node_record(SceneNodeId::new(new.slot(), new.generation()))
                        .map(|r| r.persistent_id.clone())
                        .unwrap_or_else(|| {
                            world.object_persistent_id(new).unwrap_or_else(|| {
                                SceneObjectId::new(format!(
                                    "object.{:016x}{:016x}",
                                    new.slot(),
                                    new.generation()
                                ))
                            })
                        });
                    new_remaps.push(ObjectRemap {
                        old,
                        new,
                        persistent: p,
                    });
                }
            }
        }

        // Re-attach detached lights.
        for dl in &self.detached_lights {
            let resolved = world
                .find_object_by_persistent_id(&dl.persistent_id)
                .and_then(|oid| world.resolve_object(oid));
            match dl.kind {
                ObjectKind::PointLight => {
                    if let Some(ObjectHandle::PointLight(pl_id)) = resolved {
                        if let Some(record) = world.get_point_light_record_mut(pl_id) {
                            record.light_group_parent =
                                Some(dl.old_group_parent.clone());
                        }
                    }
                }
                ObjectKind::DirectionalLight => {
                    if let Some(ObjectHandle::DirectionalLight(dl_id)) = resolved {
                        if let Some(record) = world.get_directional_light_record_mut(dl_id) {
                            record.light_group_parent =
                                Some(dl.old_group_parent.clone());
                        }
                    }
                }
                ObjectKind::SpotLight => {
                    if let Some(ObjectHandle::SpotLight(sl_id)) = resolved {
                        if let Some(record) = world.get_spot_light_record_mut(sl_id) {
                            record.light_group_parent =
                                Some(dl.old_group_parent.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        // Re-create lights.
        for persistent_id in &self.source_persistent_ids {
            if let Some(light) = self.point_light_snapshots.get(persistent_id) {
                let record = self.light_records.get(persistent_id).cloned();
                if let Some(record) = record {
                    let new_id = world.add_point_light_with_record(*light, record);
                    let new_oid = world
                        .object_id_for_point_light(new_id)
                        .unwrap_or_else(|| {
                            ObjectId::from_parts(
                                world.provenance(),
                                ObjectKind::PointLight,
                                new_id.slot,
                                new_id.generation,
                            )
                        });
                    let old = ObjectId::from_parts(
                        world.provenance(),
                        ObjectKind::PointLight,
                        0,
                        0,
                    );
                    new_remaps.push(ObjectRemap {
                        old,
                        new: new_oid,
                        persistent: persistent_id.clone(),
                    });
                }
            }
            if let Some(light) = self.directional_light_snapshots.get(persistent_id) {
                let record = self.light_records.get(persistent_id).cloned();
                if let Some(record) = record {
                    let new_id =
                        world.add_directional_light_with_record(*light, record);
                    let new_oid = world
                        .object_id_for_directional_light(new_id)
                        .unwrap_or_else(|| {
                            ObjectId::from_parts(
                                world.provenance(),
                                ObjectKind::DirectionalLight,
                                new_id.slot,
                                new_id.generation,
                            )
                        });
                    let old = ObjectId::from_parts(
                        world.provenance(),
                        ObjectKind::DirectionalLight,
                        0,
                        0,
                    );
                    new_remaps.push(ObjectRemap {
                        old,
                        new: new_oid,
                        persistent: persistent_id.clone(),
                    });
                }
            }
            if let Some(light) = self.spot_light_snapshots.get(persistent_id) {
                let record = self.light_records.get(persistent_id).cloned();
                if let Some(record) = record {
                    let new_id = world.add_spot_light_with_record(*light, record);
                    let new_oid = world
                        .object_id_for_spot_light(new_id)
                        .unwrap_or_else(|| {
                            ObjectId::from_parts(
                                world.provenance(),
                                ObjectKind::SpotLight,
                                new_id.slot,
                                new_id.generation,
                            )
                        });
                    let old =
                        ObjectId::from_parts(world.provenance(), ObjectKind::SpotLight, 0, 0);
                    new_remaps.push(ObjectRemap {
                        old,
                        new: new_oid,
                        persistent: persistent_id.clone(),
                    });
                }
            }
        }

        self.remaps = new_remaps;
        Ok(())
    }

    fn description(&self) -> &str {
        "remove_objects"
    }

    fn object_remaps(&self) -> Vec<ObjectRemap> {
        self.remaps.clone()
    }
}

// ── DuplicateObjectsCommand ──────────────────────────────────────────────

/// Duplicate one or more objects. Anchors on persistent source identity
/// for redo; duplicates are reminted with new persistent IDs.
pub struct DuplicateObjectsCommand {
    /// Persistent IDs of source objects.
    source_persistent_ids: Vec<SceneObjectId>,
    /// Kinds of source objects.
    source_kinds: Vec<ObjectKind>,
    /// Optional parent node for duplicated node roots.
    parent_persistent_id: Option<SceneObjectId>,
    /// Persistent IDs of the created duplicates (for undo removal).
    created_persistent_ids: Vec<SceneObjectId>,
    /// Created root object IDs (for undo + remap reporting).
    created_roots: Vec<ObjectId>,
    /// All remaps from old to new.
    remaps: Vec<ObjectRemap>,
    state: CommandState,
}

impl DuplicateObjectsCommand {
    pub fn new(
        source_persistent_ids: Vec<SceneObjectId>,
        source_kinds: Vec<ObjectKind>,
        parent_persistent_id: Option<SceneObjectId>,
    ) -> Self {
        Self {
            source_persistent_ids,
            source_kinds,
            parent_persistent_id,
            created_persistent_ids: Vec::new(),
            created_roots: Vec::new(),
            remaps: Vec::new(),
            state: CommandState::Prepared,
        }
    }
}

impl Command for DuplicateObjectsCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "DuplicateObjectsCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        // Resolve parent node.
        let parent_node = self
            .parent_persistent_id
            .as_ref()
            .and_then(|pp| world.find_object_by_persistent_id(pp))
            .map(|h| SceneNodeId::new(h.slot(), h.generation()));

        self.created_persistent_ids.clear();
        self.created_roots.clear();
        self.remaps.clear();

        for (persistent_id, kind) in self
            .source_persistent_ids
            .iter()
            .zip(self.source_kinds.iter())
        {
            let handle = world
                .find_object_by_persistent_id(persistent_id)
                .ok_or_else(|| {
                    SceneError::InvalidMutation(format!(
                        "source object {persistent_id} not found"
                    ))
                })?;

            match kind {
                ObjectKind::Node => {
                    let node_id = SceneNodeId::new(handle.slot(), handle.generation());
                    let old_subtree_ids = world.subtree_node_ids_preorder(node_id);
                    let duplicated_root = world
                        .duplicate_node(node_id, parent_node)
                        .map_err(|e| {
                            SceneError::InvalidMutation(format!("duplicate failed: {e}"))
                        })?;
                    let new_subtree_ids =
                        world.subtree_node_ids_preorder(duplicated_root);

                    let new_root_oid = world
                        .object_id_for_node(duplicated_root)
                        .ok_or(SceneError::InvalidNode(duplicated_root))?;
                    self.created_roots.push(new_root_oid);

                    for (old, new) in
                        old_subtree_ids.into_iter().zip(new_subtree_ids.into_iter())
                    {
                        let p = world
                            .get_node_record(SceneNodeId::new(
                                new.slot(),
                                new.generation(),
                            ))
                            .map(|r| r.persistent_id.clone())
                            .unwrap_or_else(|| {
                                SceneObjectId::new(format!(
                                    "object.{:016x}{:016x}",
                                    new.slot(),
                                    new.generation()
                                ))
                            });
                        self.created_persistent_ids.push(p.clone());
                        self.remaps.push(ObjectRemap {
                            old,
                            new,
                            persistent: p,
                        });
                    }
                }
                ObjectKind::PointLight => {
                    let pl_id = PointLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let duplicated = world.duplicate_point_light(pl_id).map_err(|e| {
                        SceneError::InvalidMutation(format!(
                            "duplicate point light failed: {e}"
                        ))
                    })?;
                    let new_oid = ObjectId::from_parts(
                        world.provenance(),
                        ObjectKind::PointLight,
                        duplicated.slot,
                        duplicated.generation,
                    );
                    let p = world
                        .get_point_light_record(duplicated)
                        .map(|r| r.persistent_id.clone())
                        .unwrap_or_else(|| {
                            SceneObjectId::new(format!(
                                "object.{:016x}{:016x}",
                                duplicated.slot, duplicated.generation
                            ))
                        });
                    self.created_persistent_ids.push(p.clone());
                    self.created_roots.push(new_oid);
                    self.remaps.push(ObjectRemap {
                        old: handle,
                        new: new_oid,
                        persistent: p,
                    });
                }
                ObjectKind::DirectionalLight => {
                    let dl_id = DirectionalLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let duplicated = world
                        .duplicate_directional_light(dl_id)
                        .map_err(|e| {
                            SceneError::InvalidMutation(format!(
                                "duplicate directional light failed: {e}"
                            ))
                        })?;
                    let new_oid = ObjectId::from_parts(
                        world.provenance(),
                        ObjectKind::DirectionalLight,
                        duplicated.slot,
                        duplicated.generation,
                    );
                    let p = world
                        .get_directional_light_record(duplicated)
                        .map(|r| r.persistent_id.clone())
                        .unwrap_or_else(|| {
                            SceneObjectId::new(format!(
                                "object.{:016x}{:016x}",
                                duplicated.slot, duplicated.generation
                            ))
                        });
                    self.created_persistent_ids.push(p.clone());
                    self.created_roots.push(new_oid);
                    self.remaps.push(ObjectRemap {
                        old: handle,
                        new: new_oid,
                        persistent: p,
                    });
                }
                ObjectKind::SpotLight => {
                    let sl_id = SpotLightId {
                        slot: handle.slot(),
                        generation: handle.generation(),
                    };
                    let duplicated = world.duplicate_spot_light(sl_id).map_err(|e| {
                        SceneError::InvalidMutation(format!(
                            "duplicate spot light failed: {e}"
                        ))
                    })?;
                    let new_oid = ObjectId::from_parts(
                        world.provenance(),
                        ObjectKind::SpotLight,
                        duplicated.slot,
                        duplicated.generation,
                    );
                    let p = world
                        .get_spot_light_record(duplicated)
                        .map(|r| r.persistent_id.clone())
                        .unwrap_or_else(|| {
                            SceneObjectId::new(format!(
                                "object.{:016x}{:016x}",
                                duplicated.slot, duplicated.generation
                            ))
                        });
                    self.created_persistent_ids.push(p.clone());
                    self.created_roots.push(new_oid);
                    self.remaps.push(ObjectRemap {
                        old: handle,
                        new: new_oid,
                        persistent: p,
                    });
                }
            }
        }

        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "DuplicateObjectsCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        // Remove everything we created.
        for persistent_id in &self.created_persistent_ids {
            if let Some(handle) = world.find_object_by_persistent_id(persistent_id) {
                match handle.kind() {
                    ObjectKind::Node => {
                        world.remove_node(SceneNodeId::new(
                            handle.slot(),
                            handle.generation(),
                        ));
                    }
                    ObjectKind::PointLight => {
                        world.remove_point_light(PointLightId {
                            slot: handle.slot(),
                            generation: handle.generation(),
                        });
                    }
                    ObjectKind::DirectionalLight => {
                        world.remove_directional_light(DirectionalLightId {
                            slot: handle.slot(),
                            generation: handle.generation(),
                        });
                    }
                    ObjectKind::SpotLight => {
                        world.remove_spot_light(SpotLightId {
                            slot: handle.slot(),
                            generation: handle.generation(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "duplicate_objects"
    }

    fn object_remaps(&self) -> Vec<ObjectRemap> {
        self.remaps.clone()
    }
}

// ── AttachComponentCommand ───────────────────────────────────────────────

/// Attach a component envelope to a scene node.
pub struct AttachComponentCommand {
    node_persistent_id: SceneObjectId,
    envelope: ComponentEnvelope,
    state: CommandState,
}

impl AttachComponentCommand {
    pub fn new(node_persistent_id: SceneObjectId, envelope: ComponentEnvelope) -> Self {
        Self {
            node_persistent_id,
            envelope,
            state: CommandState::Prepared,
        }
    }
}

impl Command for AttachComponentCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "AttachComponentCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        world
            .attach_component(node_id, self.envelope.clone())
            .map_err(SceneError::from)?;
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "AttachComponentCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let key = self.envelope.key.clone();
        let instance_id = self.envelope.instance_id.clone();
        world
            .remove_component(node_id, &key, &instance_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(
                    "component instance not found for undo".into(),
                )
            })?;
        Ok(())
    }

    fn description(&self) -> &str {
        "attach_component"
    }
}

// ── RemoveComponentCommand ───────────────────────────────────────────────

/// Remove a component instance from a scene node, remembering it for undo.
pub struct RemoveComponentCommand {
    node_persistent_id: SceneObjectId,
    key: ComponentKey,
    instance_id: ComponentInstanceId,
    removed_envelope: Option<ComponentEnvelope>,
    state: CommandState,
}

impl RemoveComponentCommand {
    pub fn new(
        node_persistent_id: SceneObjectId,
        key: ComponentKey,
        instance_id: ComponentInstanceId,
    ) -> Self {
        Self {
            node_persistent_id,
            key,
            instance_id,
            removed_envelope: None,
            state: CommandState::Prepared,
        }
    }
}

impl Command for RemoveComponentCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "RemoveComponentCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let removed = world
            .remove_component(node_id, &self.key, &self.instance_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "component instance '{0}' not found",
                    self.instance_id
                ))
            })?;
        self.removed_envelope = Some(removed);
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "RemoveComponentCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        if let Some(ref envelope) = self.removed_envelope {
            world
                .attach_component(node_id, envelope.clone())
                .map_err(SceneError::from)?;
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "remove_component"
    }
}

// ── ReplaceComponentStateCommand ─────────────────────────────────────────

/// Replace the canonical state of a component instance.
///
/// Uses the adapter's serialise path to produce the canonical JSON. The
/// old state is captured for undo.
pub struct ReplaceComponentStateCommand {
    node_persistent_id: SceneObjectId,
    key: ComponentKey,
    instance_id: ComponentInstanceId,
    /// Pre-serialized new canonical data and hydrated value.
    new_envelope: ComponentEnvelope,
    new_hydrated: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    /// Captured old state for undo.
    old_envelope: Option<ComponentEnvelope>,
    state: CommandState,
}

impl ReplaceComponentStateCommand {
    /// Construct from a pre-serialized replacement produced by
    /// [`prepare_full_state_replacement`].
    pub fn new(
        node_persistent_id: SceneObjectId,
        key: ComponentKey,
        instance_id: ComponentInstanceId,
        new_envelope: ComponentEnvelope,
        new_hydrated: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            node_persistent_id,
            key,
            instance_id,
            new_envelope,
            new_hydrated,
            old_envelope: None,
            state: CommandState::Prepared,
        }
    }
}

impl Command for ReplaceComponentStateCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "ReplaceComponentStateCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let store = world
            .component_store_mut(node_id)
            .ok_or(SceneError::InvalidNode(node_id))?;

        // Capture old envelope.
        let old = store.envelope(&self.key, &self.instance_id).cloned();
        self.old_envelope = old;

        // Apply new state.
        commit_full_state_replacement(
            store,
            self.new_envelope.clone(),
            self.new_hydrated.clone(),
        )
        .map_err(SceneError::from)?;
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "ReplaceComponentStateCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let store = world
            .component_store_mut(node_id)
            .ok_or(SceneError::InvalidNode(node_id))?;

        if let Some(ref old_env) = self.old_envelope {
            // Use replace() which clears hydrated — canonical JSON is
            // restored and the hydrated view will be reconstructed on
            // next access.
            store.replace(old_env.clone()).map_err(SceneError::from)?;
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "replace_component_state"
    }
}

// ── SetComponentPropertyCommand ──────────────────────────────────────────

/// Set a single property on a component instance via the adapter.
///
/// The caller pre-serializes the edit using [`prepare_property_edit`]
/// and passes the resulting envelope + hydrated value at construction.
/// This command implements [`Command`] by carrying pre-serialized state.
pub struct SetComponentPropertyCommand {
    node_persistent_id: SceneObjectId,
    key: ComponentKey,
    instance_id: ComponentInstanceId,
    /// Pre-serialized new canonical data and hydrated value.
    new_envelope: ComponentEnvelope,
    new_hydrated: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    /// Captured old state for undo.
    old_envelope: Option<ComponentEnvelope>,
    state: CommandState,
}

impl SetComponentPropertyCommand {
    /// Construct from a pre-serialized property edit produced by
    /// [`prepare_property_edit`] (called by the scene API layer which
    /// owns the registry).
    pub fn new(
        node_persistent_id: SceneObjectId,
        key: ComponentKey,
        instance_id: ComponentInstanceId,
        new_envelope: ComponentEnvelope,
        new_hydrated: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            node_persistent_id,
            key,
            instance_id,
            new_envelope,
            new_hydrated,
            old_envelope: None,
            state: CommandState::Prepared,
        }
    }
}

impl Command for SetComponentPropertyCommand {
    fn execute(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state == CommandState::Executed {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed(
                    "SetComponentPropertyCommand already executed".into(),
                ),
            ));
        }
        self.state = CommandState::Executed;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let store = world
            .component_store_mut(node_id)
            .ok_or(SceneError::InvalidNode(node_id))?;

        // Capture old envelope.
        let old = store.envelope(&self.key, &self.instance_id).cloned();
        self.old_envelope = old;

        // Apply new state.
        commit_full_state_replacement(
            store,
            self.new_envelope.clone(),
            self.new_hydrated.clone(),
        )
        .map_err(SceneError::from)?;
        Ok(())
    }

    fn undo(&mut self, world: &mut SceneWorld) -> Result<(), SceneError> {
        if self.state != CommandState::Executed {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "SetComponentPropertyCommand not in executed state".into(),
            )));
        }
        self.state = CommandState::Undone;

        let handle = world
            .find_object_by_persistent_id(&self.node_persistent_id)
            .ok_or_else(|| {
                SceneError::InvalidMutation(format!(
                    "node {0} not found",
                    self.node_persistent_id
                ))
            })?;
        let node_id = SceneNodeId::new(handle.slot(), handle.generation());
        let store = world
            .component_store_mut(node_id)
            .ok_or(SceneError::InvalidNode(node_id))?;

        if let Some(ref old_env) = self.old_envelope {
            store.replace(old_env.clone()).map_err(SceneError::from)?;
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "set_component_property"
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Resolve the current runtime [`ObjectId`] from a persistent
/// [`SceneObjectId`] and expected kind.
fn resolve_current_id(
    world: &SceneWorld,
    persistent_id: &SceneObjectId,
    expected_kind: ObjectKind,
) -> Result<ObjectId, SceneError> {
    let id = world
        .find_object_by_persistent_id(persistent_id)
        .ok_or_else(|| {
            SceneError::InvalidMutation(format!(
                "object with persistent id {persistent_id} not found in scene"
            ))
        })?;
    if id.kind() != expected_kind {
        use crate::object::identity::ObjectError;
        return Err(SceneError::Object(ObjectError::WrongKind {
            object: id,
            expected: expected_kind,
            actual: id.kind(),
        }));
    }
    Ok(id)
}
