//! Tests for the physics bridge: app-owned bodies/colliders, trigger sensor
//! contact phases, kinematic sync, and prepare/validate/commit/rollback.

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_runtime::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeToken, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};
use physics::{
    BodyDescriptor, BodyKind, ColliderDescriptor, ColliderShape, PhysicsBodyId, PhysicsColliderId,
    PhysicsContactKind, PhysicsContactPhase, PhysicsWorld,
};

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

fn publish_bridge_with_entities(
    entities: &[EntityCollisionRecipe],
) -> (PhysicsBridge, PhysicsWorld) {
    let mut bridge = PhysicsBridge::new();
    let token = bridge
        .prepare(
            &empty_world(),
            entities,
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();
    bridge.validate(&token).unwrap();
    bridge.commit(token).unwrap();

    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, 0.0, 0.0);
    bridge.commit_to_world(&mut world).unwrap();
    (bridge, world)
}

#[test]
fn prepare_creates_world_static_body_from_collision_planes() {
    let mut bridge = PhysicsBridge::new();
    let token = bridge
        .prepare(
            &world_with_plane(),
            &[],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    assert!(!token.payload.is_empty());

    let staged = bridge.staged().unwrap();
    assert_eq!(staged.bodies.len(), 1);
    assert_eq!(staged.bodies[0].kind, BodyKind::Static);
    assert!(staged.colliders.is_empty());
}

#[test]
fn prepare_creates_kinematic_body_for_brush_entity() {
    let mut bridge = PhysicsBridge::new();
    let entity = make_entity_collision(5, false);

    bridge
        .prepare(
            &empty_world(),
            &[entity],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    let staged = bridge.staged().unwrap();
    assert_eq!(staged.bodies.len(), 1);
    let entity_body = &staged.bodies[0];
    assert_eq!(entity_body.kind, BodyKind::Kinematic);
    assert_eq!(entity_body.id, PhysicsBodyId::new("bsp.entity.5"));

    let entity_collider = &staged.colliders[0];
    assert!(matches!(
        entity_collider.shape,
        ColliderShape::ConvexHull { .. }
    ));
}

#[test]
fn prepare_creates_trigger_sensor_for_trigger_entity() {
    let mut bridge = PhysicsBridge::new();
    let trigger_entity = make_entity_collision(10, true);

    bridge
        .prepare(
            &empty_world(),
            &[trigger_entity],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    let staged = bridge.staged().unwrap();
    let trigger_body = staged
        .bodies
        .iter()
        .find(|b| b.id == PhysicsBodyId::new("bsp.entity.10"))
        .unwrap();
    assert_eq!(trigger_body.kind, BodyKind::Static);

    let trigger_collider = staged
        .colliders
        .iter()
        .find(|c| c.parent_body == PhysicsBodyId::new("bsp.entity.10"))
        .unwrap();
    assert!(trigger_collider.is_trigger);
}

#[test]
fn validate_rejects_empty_token() {
    let bridge = PhysicsBridge::new();
    let result = bridge.validate(&BridgeToken::new(vec![]));
    assert!(result.is_err());
}

#[test]
fn validate_accepts_prepared_body_references() {
    let mut bridge = PhysicsBridge::new();
    let token = bridge
        .prepare(
            &empty_world(),
            &[make_entity_collision(1, false)],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    assert!(bridge.validate(&token).is_ok());
}

#[test]
fn commit_and_rollback_lifecycle() {
    let mut bridge = PhysicsBridge::new();
    let token = bridge
        .prepare(
            &world_with_plane(),
            &[],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    bridge.validate(&token).unwrap();
    assert!(bridge.commit(token.clone()).is_ok());

    bridge.rollback(token);
    assert!(bridge.staged().is_none());
}

#[test]
fn double_commit_after_unload_is_allowed() {
    let mut bridge = PhysicsBridge::new();
    let token = bridge
        .prepare(
            &world_with_plane(),
            &[],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    bridge.validate(&token).unwrap();
    bridge.commit(token.clone()).unwrap();

    // Phase 05: double commit is allowed (resets prior state) for atomic
    // publication where the coordinator prepares a new candidate beside
    // the active world and then swaps them.
    let token2 = bridge
        .prepare(
            &world_with_plane(),
            &[],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();
    bridge.validate(&token2).unwrap();
    assert!(bridge.commit(token2).is_ok());
}

#[test]
fn commit_to_world_publishes_entity_body_and_collider() {
    let entity = make_entity_collision(42, false);
    let (bridge, world) = publish_bridge_with_entities(&[entity]);

    assert!(bridge.entity_bodies.contains_key(&42));
    assert!(world
        .body_position_by_id(&PhysicsBodyId::new("bsp.entity.42"))
        .is_some());
    assert!(world.collider_exists(&PhysicsColliderId::new("bsp.entity.42.piece.0")));
}

#[test]
fn remove_from_world_cleans_up() {
    let entity = make_entity_collision(12, false);
    let (mut bridge, mut world) = publish_bridge_with_entities(&[entity]);
    assert!(world.collider_exists(&PhysicsColliderId::new("bsp.entity.12.piece.0")));

    bridge.remove_from_world(&mut world);
    assert!(!world.collider_exists(&PhysicsColliderId::new("bsp.entity.12.piece.0")));
}

#[test]
fn sync_body_transform_updates_registered_kinematic_body() {
    let entity = make_entity_collision(7, false);
    let (bridge, mut world) = publish_bridge_with_entities(&[entity]);

    assert!(bridge.sync_body_transform(7, [1.0, 2.0, 3.0], &mut world));
    assert_eq!(
        world.body_position_by_id(&PhysicsBodyId::new("bsp.entity.7")),
        Some([1.0, 2.0, 3.0])
    );
    assert!(!bridge.sync_body_transform(999, [0.0, 0.0, 0.0], &mut world));
}

#[test]
fn sync_from_snapshot_updates_full_kinematic_pose() {
    let entity = make_entity_collision(8, false);
    let (bridge, mut world) = publish_bridge_with_entities(&[entity]);
    let rotation = glam::Quat::from_rotation_y(90.0_f32.to_radians());
    let transform = glam::Mat4::from_rotation_translation(rotation, glam::Vec3::new(4.0, 5.0, 6.0));
    let mut builder = bsp_runtime::SnapshotBuilder::new(1, 1, 1.0 / 60.0, 1.0 / 60.0);
    builder.push_entity_pose(bsp_runtime::SnapshotEntityPose {
        entity_index: 8,
        model_index: 1,
        transform,
        world_bounds: (glam::Vec3::ZERO, glam::Vec3::ONE),
        is_moving: true,
    });

    let updated = bridge.sync_from_snapshot(&builder.build(), &mut world);
    assert_eq!(updated, 1);

    let pose = world
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
    let trigger = make_entity_collision(10, true);
    let (_bridge, mut world) = publish_bridge_with_entities(&[trigger]);

    let actor_body = PhysicsBodyId::new("actor");
    let actor_collider = PhysicsColliderId::new("actor.collider");
    world
        .create_body(BodyDescriptor::new(
            actor_body.clone(),
            BodyKind::Dynamic,
            [0.0, 0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            actor_collider.clone(),
            actor_body.clone(),
            ColliderShape::Cuboid {
                half_extents: [0.25, 0.25, 0.25],
            },
        ))
        .unwrap();

    world.step(0.016).unwrap();
    assert!(world.last_contact_records().iter().any(|record| {
        record.phase == PhysicsContactPhase::Enter
            && record.kind == PhysicsContactKind::Trigger
            && record.a == PhysicsColliderId::new("bsp.entity.10.piece.0")
            && record.b == actor_collider
    }));

    world.step(0.016).unwrap();
    assert!(world.last_contact_records().iter().any(|record| {
        record.phase == PhysicsContactPhase::Stay && record.kind == PhysicsContactKind::Trigger
    }));

    world
        .set_body_position_by_id(&actor_body, [10.0, 0.0, 0.0])
        .unwrap();
    world.step(0.016).unwrap();
    assert!(world.last_contact_records().iter().any(|record| {
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

    bridge
        .prepare(
            &world_with_plane(),
            &[empty_entity],
            &empty_lights(),
            &empty_behaviors(),
        )
        .unwrap();

    let staged = bridge.staged().unwrap();
    assert_eq!(staged.bodies.len(), 1);
}
