//! Tests for the physics bridge: app-owned bodies/colliders, trigger sensor
//! contact phases, kinematic sync, and prepare/validate/activate/teardown.
//!
//! Phase 05: All body/collider creation happens during prepare in a
//! candidate-private world. Activation moves the prevalidated world. No
//! post-activation commit_to_world. Teardown removes colliders before bodies.

use bsp_beta::physics_bridge::{PhysicsActiveState, PhysicsBridge, PhysicsPreparedState};
use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};
use physics::{BodyKind, PhysicsBodyId, PhysicsColliderId, PhysicsContactKind, PhysicsContactPhase};

fn world_with_plane() -> WorldCollisionRecipe {
    WorldCollisionRecipe {
        planes: vec![([0.0, 1.0, 0.0].into(), 0.0)],
    }
}

fn empty_world() -> WorldCollisionRecipe {
    WorldCollisionRecipe { planes: vec![] }
}

fn make_entity_collision(entity_index: u32, is_trigger: bool) -> EntityCollisionRecipe {
    use bsp::collision::{CollisionRecipe, ConvexPiece};
    use glam::Vec3;

    EntityCollisionRecipe {
        entity_index,
        classname: if is_trigger {
            "trigger_multiple".into()
        } else {
            "func_door".into()
        },
        origin: Vec3::ZERO,
        is_trigger,
        recipes: vec![CollisionRecipe {
            entity_index,
            hull_index: 1,
            pieces: vec![ConvexPiece {
                plane_normals: vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z],
                plane_dists: vec![1.0; 6],
                vertices: vec![
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(-1.0, 1.0, 1.0),
                    Vec3::new(1.0, -1.0, 1.0),
                    Vec3::new(-1.0, -1.0, 1.0),
                    Vec3::new(1.0, 1.0, -1.0),
                    Vec3::new(-1.0, 1.0, -1.0),
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(-1.0, -1.0, -1.0),
                ],
            }],
            is_trigger,
            diagnostics: vec![],
        }],
    }
}

fn empty_lights() -> Vec<LightEntityRecipe> {
    vec![]
}

fn empty_behaviors() -> Vec<BehaviorEntityRecipe> {
    vec![]
}

fn prepare_and_activate(
    bridge: &mut PhysicsBridge,
    entities: &[EntityCollisionRecipe],
) -> (Box<dyn bsp_runtime::ActiveBridgeState>, PhysicsBodyId) {
    let mut prepared = bridge
        .prepare(&empty_world(), entities, &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();

    // Verify the prepared state has what we expect
    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    let body_count = ps.all_body_ids.len();
    let collider_count = ps.all_collider_ids.len();

    let active = bridge.activate(&mut *prepared);

    // Verify the active state
    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();
    assert_eq!(ast.all_body_ids.len(), body_count);
    assert_eq!(ast.all_collider_ids.len(), collider_count);

    (active, PhysicsBodyId::new("bsp.entity.42")) // dummy for tests that don't need it
}

#[test]
fn prepare_creates_world_static_body_from_collision_planes() {
    let mut bridge = PhysicsBridge::new();
    let prepared = bridge
        .prepare(&world_with_plane(), &[], &empty_lights(), &empty_behaviors())
        .unwrap();

    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    assert_eq!(ps.all_body_ids.len(), 1);
    // Body kind is not exposed on the prepared state directly, but we can
    // check the world has the body
    assert!(ps
        .world
        .body_position_by_id(&PhysicsBodyId::new("bsp.world"))
        .is_some());
    assert!(ps.all_collider_ids.is_empty());
}

#[test]
fn prepare_creates_kinematic_body_for_brush_entity() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(5, false);

    let prepared = bridge
        .prepare(&empty_world(), &[entity], &empty_lights(), &empty_behaviors())
        .unwrap();

    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    assert_eq!(ps.all_body_ids.len(), 1);
    assert!(ps
        .world
        .body_position_by_id(&PhysicsBodyId::new("bsp.entity.5"))
        .is_some());
    assert!(!ps.all_collider_ids.is_empty());
    assert!(ps
        .world
        .collider_exists(&PhysicsColliderId::new("bsp.entity.5.piece.0")));
}

#[test]
fn prepare_creates_trigger_sensor_for_trigger_entity() {
    let mut bridge = PhysicsBridge::new();
    let trigger_entity = make_entity_collision(10, true);

    let prepared = bridge
        .prepare(
            &empty_world(),
            &[trigger_entity],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    assert!(ps
        .world
        .body_position_by_id(&PhysicsBodyId::new("bsp.entity.10"))
        .is_some());
}

#[test]
fn validate_accepts_prepared_state() {
    let mut bridge = PhysicsBridge::new();
    let prepared = bridge
        .prepare(
            &empty_world(),
            &[make_entity_collision(1, false)],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    assert!(bridge.validate(&*prepared).is_ok());
}

#[test]
fn activate_and_teardown_lifecycle() {
    let mut bridge = PhysicsBridge::new();
    let mut prepared = bridge
        .prepare(&world_with_plane(), &[], &empty_lights(), &empty_behaviors())
        .unwrap();

    bridge.validate(&*prepared).unwrap();

    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    assert!(ps.all_body_ids.len() > 0);

    let mut active = bridge.activate(&mut *prepared);
    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();
    assert!(ast.all_body_ids.len() > 0);

    // Teardown
    assert!(bridge.teardown(&mut *active).is_ok());
    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();
    assert!(ast.all_body_ids.is_empty());
    assert!(ast.all_collider_ids.is_empty());
}

#[test]
fn double_activate_after_teardown_is_allowed() {
    let mut bridge = PhysicsBridge::new();
    let mut prepared = bridge
        .prepare(&world_with_plane(), &[], &empty_lights(), &empty_behaviors())
        .unwrap();

    bridge.validate(&*prepared).unwrap();
    let mut active = bridge.activate(&mut *prepared);
    let _ = bridge.teardown(&mut *active);

    // Second prepare/activate cycle
    let mut prepared2 = bridge
        .prepare(&world_with_plane(), &[], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared2).unwrap();
    let active2 = bridge.activate(&mut *prepared2);
    assert!(!active2
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap()
        .all_body_ids
        .is_empty());
}

#[test]
fn activate_publishes_entity_body_and_collider_in_private_world() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(42, false);
    let mut prepared = bridge
        .prepare(&empty_world(), &[entity], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();

    let active = bridge.activate(&mut *prepared);
    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();

    // The active state's world has the pre-created bodies/colliders
    assert!(ast
        .world
        .body_position_by_id(&PhysicsBodyId::new("bsp.entity.42"))
        .is_some());
    assert!(ast
        .world
        .collider_exists(&PhysicsColliderId::new("bsp.entity.42.piece.0")));
    assert!(bridge.entity_bodies.contains_key(&42));
}

#[test]
fn teardown_removes_colliders_before_bodies() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(12, false);
    let mut prepared = bridge
        .prepare(&empty_world(), &[entity], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();

    let mut active = bridge.activate(&mut *prepared);
    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();
    assert!(ast
        .world
        .collider_exists(&PhysicsColliderId::new("bsp.entity.12.piece.0")));

    bridge.teardown(&mut *active).unwrap();

    let ast: &PhysicsActiveState = active
        .as_any()
        .downcast_ref::<PhysicsActiveState>()
        .unwrap();
    assert!(!ast
        .world
        .collider_exists(&PhysicsColliderId::new("bsp.entity.12.piece.0")));
    assert!(ast.all_body_ids.is_empty());
}

#[test]
fn sync_body_transform_updates_registered_kinematic_body() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(7, false);
    let mut prepared = bridge
        .prepare(&empty_world(), &[entity], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();
    let mut active = bridge.activate(&mut *prepared);
    let ast: &mut PhysicsActiveState = active
        .as_any_mut()
        .downcast_mut::<PhysicsActiveState>()
        .unwrap();

    assert!(bridge.sync_body_transform(7, [1.0, 2.0, 3.0], &mut ast.world));
    assert_eq!(
        ast.world
            .body_position_by_id(&PhysicsBodyId::new("bsp.entity.7")),
        Some([1.0, 2.0, 3.0])
    );
    assert!(!bridge.sync_body_transform(999, [0.0, 0.0, 0.0], &mut ast.world));
}

#[test]
fn sync_from_snapshot_updates_full_kinematic_pose() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(8, false);
    let mut prepared = bridge
        .prepare(&empty_world(), &[entity], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();
    let mut active = bridge.activate(&mut *prepared);
    let ast: &mut PhysicsActiveState = active
        .as_any_mut()
        .downcast_mut::<PhysicsActiveState>()
        .unwrap();

    let rotation = glam::Quat::from_rotation_y(90.0_f32.to_radians());
    let transform =
        glam::Mat4::from_rotation_translation(rotation, glam::Vec3::new(4.0, 5.0, 6.0));
    let mut builder = bsp_runtime::SnapshotBuilder::new(1, 1, 1.0 / 60.0, 1.0 / 60.0);
    builder.push_entity_pose(bsp_runtime::SnapshotEntityPose {
        entity_index: 8,
        model_index: 1,
        transform,
        world_bounds: (glam::Vec3::ZERO, glam::Vec3::ONE),
        is_moving: true,
    });

    let updated = bridge.sync_from_snapshot(&builder.build(), &mut ast.world);
    assert_eq!(updated, 1);

    let pose = ast
        .world
        .body_pose_by_id(&PhysicsBodyId::new("bsp.entity.8"))
        .expect("synced body pose");
    assert_eq!(pose.translation, [4.0, 5.0, 6.0]);
    let expected_rotation = rotation.to_array();
    for (actual, expected) in pose.rotation.iter().zip(expected_rotation) {
        assert!(
            (*actual - expected).abs() < 1e-5,
            "pose rotation: {:?}",
            pose.rotation
        );
    }
}

#[test]
fn trigger_sensor_reports_enter_stay_and_exit_phases() {
    let mut bridge = PhysicsBridge::new();
    let trigger = make_entity_collision(10, true);
    let mut prepared = bridge
        .prepare(&empty_world(), &[trigger], &empty_lights(), &empty_behaviors())
        .unwrap();
    bridge.validate(&*prepared).unwrap();
    let mut active = bridge.activate(&mut *prepared);
    let ast: &mut PhysicsActiveState = active
        .as_any_mut()
        .downcast_mut::<PhysicsActiveState>()
        .unwrap();

    let actor_body = PhysicsBodyId::new("actor");
    let actor_collider = PhysicsColliderId::new("actor.collider");

    // Create the actor in the same private world
    ast.world
        .create_body(physics::BodyDescriptor::new(
            actor_body.clone(),
            BodyKind::Dynamic,
            [0.0, 0.0, 0.0],
        ))
        .unwrap();
    ast.world
        .create_collider(
            physics::ColliderDescriptor::new(
                actor_collider.clone(),
                actor_body.clone(),
                physics::ColliderShape::Cuboid {
                    half_extents: [0.25, 0.25, 0.25],
                },
            ),
        )
        .unwrap();

    ast.world.step(0.016).unwrap();
    assert!(ast.world.last_contact_records().iter().any(|record| {
        record.phase == PhysicsContactPhase::Enter
            && record.kind == PhysicsContactKind::Trigger
            && record.a == PhysicsColliderId::new("bsp.entity.10.piece.0")
            && record.b == actor_collider
    }));

    ast.world.step(0.016).unwrap();
    assert!(ast.world.last_contact_records().iter().any(|record| {
        record.phase == PhysicsContactPhase::Stay && record.kind == PhysicsContactKind::Trigger
    }));

    ast.world
        .set_body_position_by_id(&actor_body, [10.0, 0.0, 0.0])
        .unwrap();
    ast.world.step(0.016).unwrap();
    assert!(ast.world.last_contact_records().iter().any(|record| {
        record.phase == PhysicsContactPhase::Exit && record.kind == PhysicsContactKind::Trigger
    }));
}

#[test]
fn empty_entity_recipes_are_skipped() {
    let mut bridge = PhysicsBridge::new();
    let empty_entity = EntityCollisionRecipe {
        entity_index: 99,
        classname: "func_door".into(),
        origin: glam::Vec3::ZERO,
        is_trigger: false,
        recipes: vec![],
    };

    let prepared = bridge
        .prepare(
            &world_with_plane(),
            &[empty_entity],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    let ps: &PhysicsPreparedState = prepared
        .as_any()
        .downcast_ref::<PhysicsPreparedState>()
        .unwrap();
    assert_eq!(ps.all_body_ids.len(), 1); // only world body
}
