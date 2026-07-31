//! Tests for Phase 06 — Atomic Object Commands (integration tests).
//!
//! These tests use the public [`Scene`] API and exercise all 7 built-in
//! object commands through [`CommandHistory`].

use engine_events::SceneObjectId;
use glam::{Mat4, Vec3};
use renderer::object::{
    component::{ComponentEnvelope, ComponentInstanceId, ComponentKey},
    ObjectKind, ObjectParent,
};
use renderer::{
    AttachComponentCommand, Command, CommandError, CommandHistory, DuplicateObjectsCommand,
    PointLight, RemoveComponentCommand, RemoveObjectsCommand, ReplaceComponentStateCommand, Scene,
    SceneError, SetObjectParentCommand, SetObjectTransformCommand,
};

// ── Helpers ─────────────────────────────────────────────────────────────

fn make_scene() -> Scene {
    Scene::new()
}

fn make_history(depth: usize) -> CommandHistory {
    CommandHistory::new(depth)
}

fn add_root_node(scene: &mut Scene) -> (renderer::SceneNodeId, SceneObjectId) {
    let id = scene.create_node_default(None).expect("create root node");
    scene.set_node_name(id, "Root").expect("set name");
    let persistent = scene.get_node_record(id).unwrap().persistent_id.clone();
    (id, persistent)
}

fn add_point_light(scene: &mut Scene) -> (renderer::PointLightId, SceneObjectId) {
    let id = scene
        .create_point_light(PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
        })
        .expect("create point light");
    let persistent = scene
        .get_point_light_record(id)
        .unwrap()
        .persistent_id
        .clone();
    (id, persistent)
}

struct ExternalFailureCommand {
    fail_undo: bool,
    fail_redo: bool,
    executions: u8,
}

impl ExternalFailureCommand {
    fn failing_undo() -> Self {
        Self {
            fail_undo: true,
            fail_redo: false,
            executions: 0,
        }
    }

    fn failing_redo() -> Self {
        Self {
            fail_undo: false,
            fail_redo: true,
            executions: 0,
        }
    }
}

impl Command for ExternalFailureCommand {
    fn execute(&mut self, _world: &mut renderer::SceneWorld) -> Result<(), SceneError> {
        if self.fail_redo && self.executions > 0 {
            return Err(SceneError::CommandError(
                CommandError::CommandExecutionFailed("injected redo failure".into()),
            ));
        }
        self.executions += 1;
        Ok(())
    }

    fn undo(&mut self, _world: &mut renderer::SceneWorld) -> Result<(), SceneError> {
        if self.fail_undo {
            return Err(SceneError::CommandError(CommandError::UndoFailed(
                "injected undo failure".into(),
            )));
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "external_failure"
    }
}

fn make_test_envelope() -> ComponentEnvelope {
    use serde_json::json;
    let key = ComponentKey::new("test.component").unwrap();
    let instance_id = ComponentInstanceId::mint();
    ComponentEnvelope::new(instance_id, key, 1, json!({"value": 42})).unwrap()
}

// ── SetObjectTransformCommand ───────────────────────────────────────────

#[test]
fn set_transform_node_execute_undo_redo() {
    let mut scene = make_scene();
    let mut history = make_history(8);
    let (_nid, persistent) = add_root_node(&mut scene);

    let new_t = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
    let mut cmd = SetObjectTransformCommand::new(persistent.clone(), ObjectKind::Node, new_t);

    cmd.execute(scene.world_mut()).unwrap();
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_eq!(scene.transform(nid).unwrap(), new_t);

    cmd.undo(scene.world_mut()).unwrap();
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_eq!(scene.transform(nid).unwrap(), Mat4::IDENTITY);

    cmd.execute(scene.world_mut()).unwrap(); // redo
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_eq!(scene.transform(nid).unwrap(), new_t);
}

#[test]
fn set_transform_reuse_rejected() {
    let mut scene = make_scene();
    let (_nid, persistent) = add_root_node(&mut scene);
    let mut cmd = SetObjectTransformCommand::new(persistent, ObjectKind::Node, Mat4::IDENTITY);

    cmd.execute(scene.world_mut()).unwrap();
    let err = cmd.execute(scene.world_mut()).unwrap_err();
    assert!(matches!(
        err,
        SceneError::CommandError(CommandError::CommandExecutionFailed(_))
    ));
}

#[test]
fn set_transform_point_light() {
    let mut scene = make_scene();
    let (_pl_id, persistent) = add_point_light(&mut scene);

    let new_t = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let mut cmd = SetObjectTransformCommand::new(persistent.clone(), ObjectKind::PointLight, new_t);

    cmd.execute(scene.world_mut()).unwrap();

    cmd.undo(scene.world_mut()).unwrap();
    let pl_id = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_point_light_id(oid).ok())
        .unwrap();
    let transform = scene.point_light_transform(pl_id).unwrap();
    assert_eq!(transform.w_axis.truncate(), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn set_transform_rejects_non_finite() {
    let mut scene = make_scene();
    let (_nid, persistent) = add_root_node(&mut scene);

    let bad = Mat4::from_cols(glam::Vec4::NAN, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W);
    let mut cmd = SetObjectTransformCommand::new(persistent, ObjectKind::Node, bad);
    let err = cmd.execute(scene.world_mut()).unwrap_err();
    assert!(matches!(err, SceneError::InvalidMutation(_)));
}

#[test]
fn set_transform_point_light_rejects_non_translation() {
    let mut scene = make_scene();
    let (_pl_id, persistent) = add_point_light(&mut scene);

    let rot = Mat4::from_rotation_x(0.5) * Mat4::from_translation(Vec3::X);
    let mut cmd = SetObjectTransformCommand::new(persistent, ObjectKind::PointLight, rot);
    let err = cmd.execute(scene.world_mut()).unwrap_err();
    assert!(matches!(err, SceneError::InvalidMutation(_)));
}

// ── SetObjectParentCommand ──────────────────────────────────────────────

#[test]
fn set_parent_node_reparent_undo_redo() {
    let mut scene = make_scene();
    let (root_id, _root_persistent) = add_root_node(&mut scene);
    let child_nid = scene
        .create_node_default(Some(root_id))
        .expect("create child");
    let child_persistent = scene
        .get_node_record(child_nid)
        .unwrap()
        .persistent_id
        .clone();

    let (new_parent_id, new_parent_persistent) = add_root_node(&mut scene);

    // Reparent child to new_parent using persistent ID resolution.
    let new_parent_oid = scene
        .find_object_by_persistent_id(&new_parent_persistent)
        .unwrap();
    let mut cmd = SetObjectParentCommand::new(
        child_persistent.clone(),
        ObjectKind::Node,
        ObjectParent::Node(new_parent_oid),
        Some(new_parent_persistent.clone()),
    );

    cmd.execute(scene.world_mut()).unwrap();
    let child_nid = scene
        .find_object_by_persistent_id(&child_persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert!(
        scene.children(new_parent_id).contains(&child_nid),
        "child should be under new parent after reparent"
    );

    cmd.undo(scene.world_mut()).unwrap();
    let child_nid = scene
        .find_object_by_persistent_id(&child_persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert!(scene.children(root_id).contains(&child_nid));

    cmd.execute(scene.world_mut()).unwrap(); // redo
    let child_nid = scene
        .find_object_by_persistent_id(&child_persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert!(scene.children(new_parent_id).contains(&child_nid));
}

#[test]
fn set_parent_light_regroup_undo_redo() {
    let mut scene = make_scene();
    let (_node_id, node_persistent) = add_root_node(&mut scene);
    let (_pl_id, light_persistent) = add_point_light(&mut scene);

    let mut cmd = SetObjectParentCommand::new(
        light_persistent.clone(),
        ObjectKind::PointLight,
        ObjectParent::None,
        Some(node_persistent.clone()),
    );

    cmd.execute(scene.world_mut()).unwrap();
    let pl_id = scene
        .find_object_by_persistent_id(&light_persistent)
        .and_then(|oid| scene.try_get_point_light_id(oid).ok())
        .unwrap();
    assert_eq!(
        scene
            .get_point_light_record(pl_id)
            .unwrap()
            .light_group_parent,
        Some(node_persistent.clone())
    );

    cmd.undo(scene.world_mut()).unwrap();
    let pl_id = scene
        .find_object_by_persistent_id(&light_persistent)
        .and_then(|oid| scene.try_get_point_light_id(oid).ok())
        .unwrap();
    assert_eq!(
        scene
            .get_point_light_record(pl_id)
            .unwrap()
            .light_group_parent,
        None
    );
}

// ── RemoveObjectsCommand ─────────────────────────────────────────────────

#[test]
fn remove_node_undo_restores_with_remap() {
    let mut scene = make_scene();
    let (node_id, persistent) = add_root_node(&mut scene);
    scene
        .create_node_default(Some(node_id))
        .expect("create child");

    let mut cmd = RemoveObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::Node]);

    cmd.execute(scene.world_mut()).unwrap();
    assert!(!scene.is_valid_node(node_id));

    cmd.undo(scene.world_mut()).unwrap();
    let remaps = cmd.object_remaps();
    assert!(!remaps.is_empty(), "undo should produce remaps");

    let new_nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_ne!(new_nid, node_id);
    assert!(!scene.children(new_nid).is_empty());
}

#[test]
fn remove_node_then_redo_removes_again() {
    let mut scene = make_scene();
    let (_node_id, persistent) = add_root_node(&mut scene);

    let mut cmd = RemoveObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::Node]);

    cmd.execute(scene.world_mut()).unwrap();
    cmd.undo(scene.world_mut()).unwrap();
    let restored = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert!(scene.is_valid_node(restored));

    cmd.execute(scene.world_mut()).unwrap(); // redo
    assert!(!scene.is_valid_node(restored));
}

#[test]
fn remove_point_light_undo_restores() {
    let mut scene = make_scene();
    let (_pl_id, persistent) = add_point_light(&mut scene);

    let mut cmd = RemoveObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::PointLight]);

    cmd.execute(scene.world_mut()).unwrap();
    assert!(scene.find_object_by_persistent_id(&persistent).is_none());

    cmd.undo(scene.world_mut()).unwrap();
    assert!(scene.find_object_by_persistent_id(&persistent).is_some());
}

// ── DuplicateObjectsCommand ──────────────────────────────────────────────

#[test]
fn duplicate_node_produces_new_persistent_ids() {
    let mut scene = make_scene();
    let (_nid, persistent) = add_root_node(&mut scene);

    let mut cmd =
        DuplicateObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::Node], None);

    cmd.execute(scene.world_mut()).unwrap();
    let remaps = cmd.object_remaps();
    assert!(!remaps.is_empty());
    assert!(scene.find_object_by_persistent_id(&persistent).is_some());
    let dup_persistent = &remaps[0].persistent;
    assert_ne!(dup_persistent, &persistent);
    assert!(scene.find_object_by_persistent_id(dup_persistent).is_some());
}

#[test]
fn duplicate_then_undo_removes_duplicates() {
    let mut scene = make_scene();
    let (_nid, persistent) = add_root_node(&mut scene);

    let mut cmd =
        DuplicateObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::Node], None);

    cmd.execute(scene.world_mut()).unwrap();
    let remaps = cmd.object_remaps();
    let dup_persistent = remaps[0].persistent.clone();

    cmd.undo(scene.world_mut()).unwrap();
    assert!(scene.find_object_by_persistent_id(&persistent).is_some());
    assert!(scene
        .find_object_by_persistent_id(&dup_persistent)
        .is_none());
}

#[test]
fn duplicate_light() {
    let mut scene = make_scene();
    let (_pl_id, persistent) = add_point_light(&mut scene);

    let mut cmd =
        DuplicateObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::PointLight], None);

    cmd.execute(scene.world_mut()).unwrap();
    let remaps = cmd.object_remaps();
    assert_eq!(remaps.len(), 1);
    assert_ne!(remaps[0].persistent, persistent);
}

// ── AttachComponentCommand ───────────────────────────────────────────────

#[test]
fn attach_component_undo_removes() {
    let mut scene = make_scene();
    let (nid, persistent) = add_root_node(&mut scene);

    let envelope = make_test_envelope();
    let key = envelope.key.clone();
    let instance_id = envelope.instance_id.clone();

    let mut cmd = AttachComponentCommand::new(persistent.clone(), envelope.clone());

    cmd.execute(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    assert!(envs
        .iter()
        .any(|e| e.key == key && e.instance_id == instance_id));

    cmd.undo(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    assert!(!envs
        .iter()
        .any(|e| e.key == key && e.instance_id == instance_id));
}

// ── RemoveComponentCommand ───────────────────────────────────────────────

#[test]
fn remove_component_undo_reattaches() {
    let mut scene = make_scene();
    let (nid, persistent) = add_root_node(&mut scene);

    let envelope = make_test_envelope();
    let key = envelope.key.clone();
    let instance_id = envelope.instance_id.clone();

    scene.attach_component(nid, envelope.clone()).unwrap();

    let mut cmd = RemoveComponentCommand::new(persistent.clone(), key.clone(), instance_id.clone());

    cmd.execute(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    assert!(!envs
        .iter()
        .any(|e| e.key == key && e.instance_id == instance_id));

    cmd.undo(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    assert!(envs
        .iter()
        .any(|e| e.key == key && e.instance_id == instance_id));
}

// ── ReplaceComponentStateCommand ─────────────────────────────────────────

#[test]
fn replace_component_state_undo_restores_old() {
    let mut scene = make_scene();
    let (nid, persistent) = add_root_node(&mut scene);

    let envelope = make_test_envelope();
    let key = envelope.key.clone();
    let instance_id = envelope.instance_id.clone();

    scene.attach_component(nid, envelope.clone()).unwrap();

    use serde_json::json;
    let new_envelope =
        ComponentEnvelope::new(instance_id.clone(), key.clone(), 1, json!({"value": 99})).unwrap();
    let new_hydrated: std::sync::Arc<dyn std::any::Any + Send + Sync> =
        std::sync::Arc::new(json!({"value": 99}));

    let mut cmd = ReplaceComponentStateCommand::new(
        persistent.clone(),
        key.clone(),
        instance_id.clone(),
        new_envelope,
        new_hydrated,
    );

    cmd.execute(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    let env = envs
        .iter()
        .find(|e| e.key == key && e.instance_id == instance_id)
        .unwrap();
    assert_eq!(env.data, json!({"value": 99}));

    cmd.undo(scene.world_mut()).unwrap();
    let envs = scene.component_envelopes(nid).unwrap();
    let env = envs
        .iter()
        .find(|e| e.key == key && e.instance_id == instance_id)
        .unwrap();
    assert_eq!(env.data, json!({"value": 42}));
}

// ── CommandHistory failure preservation ──────────────────────────────────

#[test]
fn execute_failure_preserves_redo_stack() {
    let mut scene = make_scene();
    let mut history = make_history(8);
    let (_nid, persistent) = add_root_node(&mut scene);

    let cmd1 = SetObjectTransformCommand::new(
        persistent.clone(),
        ObjectKind::Node,
        Mat4::from_translation(Vec3::X),
    );
    history.execute(Box::new(cmd1), scene.world_mut()).unwrap();
    history.undo(scene.world_mut()).unwrap();
    assert!(history.can_redo());

    let bad_t = Mat4::from_cols(glam::Vec4::NAN, glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W);
    let cmd2 = SetObjectTransformCommand::new(persistent, ObjectKind::Node, bad_t);
    let err = history.execute(Box::new(cmd2), scene.world_mut());
    assert!(err.is_err());
    assert!(history.can_redo()); // redo stack preserved
}

#[test]
fn undo_failure_keeps_command_on_undo_stack() {
    let mut scene = make_scene();
    let mut history = make_history(8);
    let (_nid, persistent) = add_root_node(&mut scene);

    let cmd = SetObjectTransformCommand::new(
        persistent.clone(),
        ObjectKind::Node,
        Mat4::from_translation(Vec3::X),
    );
    history.execute(Box::new(cmd), scene.world_mut()).unwrap();

    // Remove the node to make undo fail.
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    scene.remove_node(nid).unwrap();

    let result = history.undo(scene.world_mut());
    assert!(result.is_err());
    assert!(history.can_undo());
}

#[test]
fn redo_failure_keeps_command_on_redo_stack() {
    let mut scene = make_scene();
    let mut history = make_history(8);
    let (_nid, persistent) = add_root_node(&mut scene);

    let cmd = SetObjectTransformCommand::new(
        persistent.clone(),
        ObjectKind::Node,
        Mat4::from_translation(Vec3::X),
    );
    history.execute(Box::new(cmd), scene.world_mut()).unwrap();
    history.undo(scene.world_mut()).unwrap();
    assert!(history.can_redo());

    // Remove the node to make redo fail.
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    scene.remove_node(nid).unwrap();

    let result = history.redo(scene.world_mut());
    assert!(result.is_err());
    assert!(history.can_redo());
}

#[test]
fn external_command_failures_preserve_history_stack_positions() {
    let mut scene = make_scene();
    let mut history = make_history(8);

    history
        .execute(
            Box::new(ExternalFailureCommand::failing_undo()),
            scene.world_mut(),
        )
        .unwrap();
    assert!(history.can_undo());
    assert!(!history.can_redo());
    assert!(history.undo(scene.world_mut()).is_err());
    assert!(history.can_undo(), "failed undo must remain at undo top");
    assert!(!history.can_redo());

    let mut redo_history = make_history(8);
    redo_history
        .execute(
            Box::new(ExternalFailureCommand::failing_redo()),
            scene.world_mut(),
        )
        .unwrap();
    redo_history.undo(scene.world_mut()).unwrap();
    assert!(!redo_history.can_undo());
    assert!(redo_history.can_redo());
    assert!(redo_history.redo(scene.world_mut()).is_err());
    assert!(!redo_history.can_undo());
    assert!(
        redo_history.can_redo(),
        "failed redo must remain at redo top"
    );
}

// ── Persistent identity resolution across handle changes ────────────────

#[test]
fn redo_resolves_by_persistent_id_after_restore_changes_handle() {
    let mut scene = make_scene();
    let mut history = make_history(8);
    let (orig_nid, persistent) = add_root_node(&mut scene);

    // Execute a transform command.
    let cmd = SetObjectTransformCommand::new(
        persistent.clone(),
        ObjectKind::Node,
        Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)),
    );
    history.execute(Box::new(cmd), scene.world_mut()).unwrap();

    // Remove the node and restore it (changing runtime handle).
    let remove_cmd = RemoveObjectsCommand::new(vec![persistent.clone()], vec![ObjectKind::Node]);
    history
        .execute(Box::new(remove_cmd), scene.world_mut())
        .unwrap();

    // Undo the remove — node gets restored with a new runtime handle.
    history.undo(scene.world_mut()).unwrap();

    // The new handle is different.
    let new_nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_ne!(new_nid, orig_nid);

    // Now undo the transform command — it must find the node by persistent ID.
    history.undo(scene.world_mut()).unwrap();
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_eq!(scene.transform(nid).unwrap(), Mat4::IDENTITY);

    // Redo the transform — still finds by persistent ID.
    history.redo(scene.world_mut()).unwrap();
    let nid = scene
        .find_object_by_persistent_id(&persistent)
        .and_then(|oid| scene.try_get_node_id(oid).ok())
        .unwrap();
    assert_eq!(
        scene.transform(nid).unwrap(),
        Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0))
    );
}

// ── Edge: descriptions ──────────────────────────────────────────────────

#[test]
fn commands_report_correct_descriptions() {
    let mut scene = make_scene();
    let (_nid, persistent) = add_root_node(&mut scene);

    let mut cmd = SetObjectTransformCommand::new(persistent, ObjectKind::Node, Mat4::IDENTITY);
    assert_eq!(cmd.description(), "set_object_transform");

    cmd.execute(scene.world_mut()).unwrap();
    assert_eq!(cmd.description(), "set_object_transform");
}
