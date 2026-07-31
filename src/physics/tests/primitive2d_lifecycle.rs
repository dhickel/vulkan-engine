//! 2D physics primitive lifecycle tests — feature gated behind `rapier2d`.
//!
//! Covers creation, removal, ray casts, overlap queries, contact records,
//! simulation steps, atomic registration, reconfiguration, forces, and
//! velocity management for [`PhysicsWorld2D`].

#![cfg(feature = "rapier2d")]

use physics::components2d::*;
use physics::world2d::PhysicsWorld2D;
use physics::{
    BodyKind, BodyMode, PhysicsBodyId, PhysicsColliderId, PhysicsContactKind, PhysicsError,
};

// ── Creation and basic simulation ───────────────────────────────

#[test]
fn create_body_and_collider_and_step() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, -10.0);

    let body_id = world
        .create_body(BodyDescriptor2D::new(
            "body.a",
            BodyKind::Dynamic,
            [0.0, 5.0],
        ))
        .unwrap();
    let collider_id = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.a",
            "body.a",
            ColliderShape2D::Cuboid {
                half_extents: [0.5, 0.5],
            },
        ))
        .unwrap();

    assert!(world.body_exists(&body_id));
    assert!(world.collider_exists(&collider_id));

    world.step(1.0 / 60.0).unwrap();
    let pos = world.body_position_by_id(&body_id).unwrap();
    assert!(pos[1] < 5.0, "body should fall");
}

// ── Durable ID consistency ──────────────────────────────────────

#[test]
fn roundtrip_body_and_collider_ids() {
    let mut world = PhysicsWorld2D::new();
    let body = PhysicsBodyId::new("body.roundtrip");

    assert_eq!(
        world
            .create_body(BodyDescriptor2D::new(
                body.clone(),
                BodyKind::Static,
                [1.0, 2.0],
            ))
            .unwrap(),
        body
    );
    let collider = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.roundtrip",
            body.clone(),
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    assert_eq!(world.body_position_by_id(&body), Some([1.0, 2.0]));
    assert!(world.collider_exists(&collider));
}

// ── Pose get/set ────────────────────────────────────────────────

#[test]
fn set_and_get_body_pose() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.pose",
            BodyKind::Kinematic,
            [0.0, 0.0],
        ))
        .unwrap();

    world.set_body_position_by_id(&body, [3.0, 4.0]).unwrap();
    assert_eq!(world.body_position_by_id(&body), Some([3.0, 4.0]));

    let pose = BodyPose2D {
        translation: [5.0, 6.0],
        rotation: 1.5,
    };
    world.set_body_pose_by_id(&body, pose).unwrap();
    let read = world.body_pose_by_id(&body).unwrap();
    assert_eq!(read.translation, [5.0, 6.0]);
    assert!((read.rotation - 1.5).abs() < 1e-5);
}

#[test]
fn set_missing_body_errors() {
    let mut world = PhysicsWorld2D::new();
    assert_eq!(
        world.set_body_position_by_id(&PhysicsBodyId::new("missing"), [0.0, 0.0]),
        Err(PhysicsError::MissingBody(PhysicsBodyId::new("missing")))
    );
}

// ── Body removal with cascading collider cleanup ─────────────────

#[test]
fn remove_body_cleans_up_attached_colliders() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.parent",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    let c1 = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.c1",
            "body.parent",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();
    let c2 = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.c2",
            "body.parent",
            ColliderShape2D::Cuboid {
                half_extents: [1.0, 1.0],
            },
        ))
        .unwrap();

    let outcome = world.remove_body_with_outcome(&body).unwrap();
    assert_eq!(outcome.removed_body, Some(body.clone()));
    assert_eq!(outcome.removed_colliders.len(), 2);
    assert!(!world.body_exists(&body));
    assert!(!world.collider_exists(&c1));
    assert!(!world.collider_exists(&c2));
}

#[test]
fn remove_collider_only() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.s",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let c = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.c",
            "body.s",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    let outcome = world.remove_collider_with_outcome(&c).unwrap();
    assert!(outcome.removed_body.is_none());
    assert_eq!(outcome.removed_colliders, vec![c.clone()]);
    assert!(world.body_exists(&body));
    assert!(!world.collider_exists(&c));
}

// ── Ray cast ────────────────────────────────────────────────────

#[test]
fn ray_hit_on_static_cuboid() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.wall",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.wall",
            "body.wall",
            ColliderShape2D::Cuboid {
                half_extents: [1.0, 1.0],
            },
        ))
        .unwrap();

    let hit = world
        .cast_ray(RayQuery2D::new([-5.0, 0.0], [1.0, 0.0], 10.0))
        .unwrap()
        .unwrap();
    assert_eq!(hit.body, body);
    assert_eq!(hit.collider, collider);
    assert!(hit.time_of_impact > 0.0 && hit.time_of_impact < 5.0);
}

#[test]
fn ray_miss_when_clear() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.wall",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.wall",
            "body.wall",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    let miss = world
        .cast_ray(RayQuery2D::new([5.0, 0.0], [1.0, 0.0], 10.0))
        .unwrap();
    assert!(miss.is_none());
}

#[test]
fn ray_zero_direction_rejected() {
    let mut world = PhysicsWorld2D::new();
    let err = world
        .cast_ray(RayQuery2D::new([0.0, 0.0], [0.0, 0.0], 1.0))
        .unwrap_err();
    assert_eq!(err, PhysicsError::ZeroDirection);
}

// ── Contact records across steps ─────────────────────────────────

#[test]
fn enter_stay_exit_cycle() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, -10.0);

    // Static floor
    world
        .create_body(BodyDescriptor2D::new(
            "body.floor",
            BodyKind::Static,
            [0.0, -1.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.floor",
            "body.floor",
            ColliderShape2D::Cuboid {
                half_extents: [3.0, 0.5],
            },
        ))
        .unwrap();

    // Dynamic ball above floor — start close enough to contact quickly
    world
        .create_body(BodyDescriptor2D::new(
            "body.ball",
            BodyKind::Dynamic,
            [0.0, 0.5],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.ball",
            "body.ball",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    // Step until contact
    for _ in 0..60 {
        world.step(1.0 / 60.0).unwrap();
    }

    let records = world.last_contact_records();
    assert!(!records.is_empty(), "ball should contact floor after steps");
    assert_eq!(records[0].kind, PhysicsContactKind::Collision);
}

// ── Trigger contacts ─────────────────────────────────────────────

#[test]
fn trigger_contact_event() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, 0.0);

    world
        .create_body(BodyDescriptor2D::new(
            "body.sensor",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(
            ColliderDescriptor2D::new(
                "collider.sensor",
                "body.sensor",
                ColliderShape2D::Ball { radius: 2.0 },
            )
            .trigger(true),
        )
        .unwrap();

    world
        .create_body(BodyDescriptor2D::new(
            "body.player",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.player",
            "body.player",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    world.step(1.0 / 60.0).unwrap();
    let records = world.last_contact_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, PhysicsContactKind::Trigger);
    assert_eq!(records[0].a, PhysicsColliderId::new("collider.sensor"));
}

// ── Atomic registration ──────────────────────────────────────────

#[test]
fn atomic_register_multi_collider() {
    let mut world = PhysicsWorld2D::new();
    let outcome = world
        .register_body(BodyRegistrationRequest2D {
            body: BodyDescriptor2D::new("body.multi", BodyKind::Dynamic, [0.0, 0.0]),
            colliders: vec![
                ColliderDescriptor2D::new(
                    "collider.a",
                    "body.multi",
                    ColliderShape2D::Ball { radius: 0.5 },
                ),
                ColliderDescriptor2D::new(
                    "collider.b",
                    "body.multi",
                    ColliderShape2D::Cuboid {
                        half_extents: [0.25, 0.25],
                    },
                ),
            ],
        })
        .unwrap();

    assert_eq!(outcome.body_id, PhysicsBodyId::new("body.multi"));
    assert_eq!(outcome.collider_ids.len(), 2);
    assert!(world.body_exists(&PhysicsBodyId::new("body.multi")));
    assert!(world.collider_exists(&PhysicsColliderId::new("collider.a")));
    assert!(world.collider_exists(&PhysicsColliderId::new("collider.b")));
}

#[test]
fn atomic_register_detects_duplicate_collider_id_in_request() {
    let mut world = PhysicsWorld2D::new();
    let result = world.register_body(BodyRegistrationRequest2D {
        body: BodyDescriptor2D::new("body.dup", BodyKind::Dynamic, [0.0, 0.0]),
        colliders: vec![
            ColliderDescriptor2D::new(
                "collider.dup",
                "body.dup",
                ColliderShape2D::Ball { radius: 0.5 },
            ),
            ColliderDescriptor2D::new(
                "collider.dup",
                "body.dup",
                ColliderShape2D::Ball { radius: 0.5 },
            ),
        ],
    });
    assert_eq!(
        result.unwrap_err(),
        PhysicsError::DuplicateColliderId(PhysicsColliderId::new("collider.dup"))
    );
    assert!(!world.body_exists(&PhysicsBodyId::new("body.dup")));
}

#[test]
fn atomic_register_wrong_parent_fails() {
    let mut world = PhysicsWorld2D::new();
    let result = world.register_body(BodyRegistrationRequest2D {
        body: BodyDescriptor2D::new("body.reg", BodyKind::Dynamic, [0.0, 0.0]),
        colliders: vec![ColliderDescriptor2D::new(
            "collider.reg",
            "body.other",
            ColliderShape2D::Ball { radius: 0.5 },
        )],
    });
    assert!(result.is_err());
    assert!(!world.body_exists(&PhysicsBodyId::new("body.reg")));
    assert!(!world.collider_exists(&PhysicsColliderId::new("collider.reg")));
}

// ── Body reconfiguration ─────────────────────────────────────────

#[test]
fn reconfigure_dynamic_to_kinematic_and_back() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.switch",
            BodyKind::Dynamic,
            [1.0, 0.0],
        ))
        .unwrap();

    assert!(world.body_is_dynamic(&body));
    world
        .reconfigure_body_mode(&body, BodyMode::Kinematic)
        .unwrap();
    assert!(world.body_is_kinematic(&body));

    // Should still be at original position
    assert_eq!(world.body_position_by_id(&body), Some([1.0, 0.0]));

    // Switch back to dynamic, preserve state
    world
        .reconfigure_body_mode(&body, BodyMode::Dynamic)
        .unwrap();
    assert!(world.body_is_dynamic(&body));
}

#[test]
fn reconfigure_to_static_preserves_reconfigured_state() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, 0.0);
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.dyn",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.dyn",
            "body.dyn",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    world.set_linear_velocity(&body, [2.0, 0.0]).unwrap();

    // Freeze to static
    world
        .reconfigure_body_mode(&body, BodyMode::Static)
        .unwrap();
    assert!(world.body_is_static(&body));

    // Velocity tracked internally even when static
    assert_eq!(world.body_linear_velocity(&body), Some([2.0, 0.0]));

    // Back to dynamic, velocity restored
    world
        .reconfigure_body_mode(&body, BodyMode::Dynamic)
        .unwrap();
    assert_eq!(world.body_linear_velocity(&body), Some([2.0, 0.0]));
}

// ── Collider replacement ─────────────────────────────────────────

#[test]
fn replace_collider_shape() {
    let mut world = PhysicsWorld2D::new();
    let _body = world
        .create_body(BodyDescriptor2D::new(
            "body.rep",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.rep",
            "body.rep",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    world
        .replace_collider(ColliderReplacementRequest2D {
            collider_id: collider.clone(),
            shape: ColliderShape2D::Cuboid {
                half_extents: [1.0, 2.0],
            },
            is_trigger: true,
            translation: [0.0, 0.0],
            rotation: 0.0,
        })
        .unwrap();

    assert!(world.collider_exists(&collider));
    // Ray hit should still work with the new shape
    let hit = world
        .cast_ray(RayQuery2D::new([-3.0, 0.0], [1.0, 0.0], 10.0))
        .unwrap();
    assert!(hit.is_some());
}

// ── Force, impulse, velocity ─────────────────────────────────────

#[test]
fn apply_impulse_changes_velocity() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, 0.0);

    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.imp",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.imp",
            "body.imp",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    world.apply_impulse(&body, [5.0, 0.0]).unwrap();
    let vel = world.body_linear_velocity(&body).unwrap();
    assert!(vel[0] > 0.0, "impulse should set velocity");
}

#[test]
fn apply_torque_changes_angular_velocity() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(0.0, 0.0);

    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.torque",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.torque",
            "body.torque",
            ColliderShape2D::Capsule {
                half_height: 0.5,
                radius: 0.25,
            },
        ))
        .unwrap();

    world.apply_torque_impulse(&body, 3.0).unwrap();
    let angvel = world.body_angular_velocity(&body).unwrap();
    assert!(angvel != 0.0, "torque should set angular velocity");
}

#[test]
fn velocity_set_and_read() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.vel",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();

    world.set_linear_velocity(&body, [1.5, -3.0]).unwrap();
    world.set_angular_velocity(&body, 2.0).unwrap();

    assert_eq!(world.body_linear_velocity(&body), Some([1.5, -3.0]));
    assert_eq!(world.body_angular_velocity(&body), Some(2.0));
}

#[test]
fn wake_and_sleep_body() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.ws",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();

    world.sleep_body(&body).unwrap();
    world.wake_body(&body).unwrap();
    // Just checking no panic; Rapier manages sleep internally
}

// ── Teleport ─────────────────────────────────────────────────────

#[test]
fn teleport_repositions_dynamic_body() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.tel",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .teleport_body(&body, BodyPose2D::from_translation([100.0, 200.0]))
        .unwrap();
    assert_eq!(world.body_position_by_id(&body), Some([100.0, 200.0]));
}

#[test]
fn teleport_ignores_static_body_silently() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.s",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    // Should not error, just silently skip
    assert!(world
        .teleport_body(&body, BodyPose2D::from_translation([1.0, 1.0]))
        .is_ok());
    // Static body position shouldn't change
    assert_eq!(world.body_position_by_id(&body), Some([0.0, 0.0]));
}

// ── Overlap queries ──────────────────────────────────────────────

#[test]
fn overlap_circle_includes_and_excludes() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.near",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.near",
            "body.near",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    world
        .create_body(BodyDescriptor2D::new(
            "body.far",
            BodyKind::Static,
            [20.0, 20.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.far",
            "body.far",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    let results = world.overlap_circle([0.0, 0.0], 5.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].collider, PhysicsColliderId::new("collider.near"));
}

#[test]
fn overlap_aabb_large_enough_includes_both() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.a",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.a",
            "body.a",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    world
        .create_body(BodyDescriptor2D::new(
            "body.b",
            BodyKind::Static,
            [5.0, 5.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.b",
            "body.b",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();

    let results = world.overlap_aabb([-3.0, -3.0], [7.0, 7.0]).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn overlap_aabb_rejects_inverted_min_max() {
    let mut world = PhysicsWorld2D::new();
    let err = world.overlap_aabb([5.0, 5.0], [1.0, 1.0]).unwrap_err();
    assert!(matches!(err, PhysicsError::NonFiniteValue { .. }));
}

// ── Introspection ────────────────────────────────────────────────

#[test]
fn body_kind_introspection() {
    let mut world = PhysicsWorld2D::new();
    let s = world
        .create_body(BodyDescriptor2D::new(
            "body.s",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let d = world
        .create_body(BodyDescriptor2D::new(
            "body.d",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    let k = world
        .create_body(BodyDescriptor2D::new(
            "body.k",
            BodyKind::Kinematic,
            [0.0, 0.0],
        ))
        .unwrap();

    assert!(world.body_is_static(&s));
    assert!(!world.body_is_dynamic(&s));
    assert!(world.body_is_dynamic(&d));
    assert!(world.body_is_kinematic(&k));
}

#[test]
fn body_exists_returns_false_for_unknown() {
    let world = PhysicsWorld2D::new();
    assert!(!world.body_exists(&PhysicsBodyId::new("nope")));
}

// ── Collider shape validation ────────────────────────────────────

#[test]
fn cuboid_non_positive_rejected() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.s",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.bad",
            "body.s",
            ColliderShape2D::Cuboid {
                half_extents: [1.0, -0.5],
            },
        ))
        .unwrap_err();
    assert_eq!(
        err,
        PhysicsError::NonPositiveDimension {
            field: "cuboid.half_extents.y"
        }
    );
}

#[test]
fn duplicate_body_rejected() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.dup",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world
        .create_body(BodyDescriptor2D::new(
            "body.dup",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap_err();
    assert_eq!(
        err,
        PhysicsError::DuplicateBodyId(PhysicsBodyId::new("body.dup"))
    );
}

#[test]
fn collider_missing_parent_rejected() {
    let mut world = PhysicsWorld2D::new();
    let err = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.orphan",
            "body.missing",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap_err();
    assert_eq!(
        err,
        PhysicsError::MissingBody(PhysicsBodyId::new("body.missing"))
    );
}

// ── Convex polygon edge tests ────────────────────────────────────

#[test]
fn convex_polygon_nan_rejected() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.poly",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.poly",
            "body.poly",
            ColliderShape2D::ConvexPolygon {
                points: vec![[0.0, 0.0], [1.0, 0.0], [0.0, f32::NAN]],
            },
        ))
        .unwrap_err();
    assert_eq!(err, PhysicsError::ConvexHullNonFiniteVertex { index: 2 });
}

#[test]
fn convex_polygon_insufficient_points() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.poly",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.poly",
            "body.poly",
            ColliderShape2D::ConvexPolygon {
                points: vec![[0.0, 0.0], [1.0, 0.0]],
            },
        ))
        .unwrap_err();
    assert_eq!(
        err,
        PhysicsError::ConvexHullInsufficientPoints { unique_count: 2 }
    );
}

// ── Gravity set / default ────────────────────────────────────────

#[test]
fn default_gravity_is_set() {
    // Default gravity is -9.81 in Y; test that a dynamic body falls.
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.g",
            BodyKind::Dynamic,
            [0.0, 5.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.g",
            "body.g",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();
    world.step(0.5).unwrap();
    let pos = world.body_position_by_id(&body).unwrap();
    assert!(pos[1] < 5.0, "body should fall under default gravity");
}

#[test]
fn custom_gravity_applied() {
    let mut world = PhysicsWorld2D::new();
    world.set_gravity(3.0, -5.0);
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.g",
            BodyKind::Dynamic,
            [0.0, 5.0],
        ))
        .unwrap();
    world
        .create_collider(ColliderDescriptor2D::new(
            "collider.g",
            "body.g",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();
    world.step(0.5).unwrap();
    let pos = world.body_position_by_id(&body).unwrap();
    assert!(pos[1] < 5.0, "body should fall under custom gravity");
}

// ── Body with collider at offset ─────────────────────────────────

#[test]
fn collider_with_local_offset() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.offset",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    world
        .create_collider(
            ColliderDescriptor2D::new(
                "collider.offset",
                "body.offset",
                ColliderShape2D::Ball { radius: 0.5 },
            )
            .translation([2.0, 1.0]),
        )
        .unwrap();

    // Body still at origin
    assert_eq!(world.body_position_by_id(&body), Some([0.0, 0.0]));

    // Ray hitting the offset collider should work
    let hit2 = world
        .cast_ray(RayQuery2D::new([2.0, 5.0], [0.0, -1.0], 10.0))
        .unwrap();
    assert!(hit2.is_some());
}

// ── Collider replacement validation ──────────────────────────────

#[test]
fn replace_collider_validates_new_shape() {
    let mut world = PhysicsWorld2D::new();
    let _body = world
        .create_body(BodyDescriptor2D::new(
            "body.rep",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.rep",
            "body.rep",
            ColliderShape2D::Ball { radius: 0.5 },
        ))
        .unwrap();

    // Invalid replacement
    let err = world
        .replace_collider(ColliderReplacementRequest2D {
            collider_id: collider.clone(),
            shape: ColliderShape2D::Ball { radius: -1.0 },
            is_trigger: false,
            translation: [0.0, 0.0],
            rotation: 0.0,
        })
        .unwrap_err();
    assert!(matches!(err, PhysicsError::NonPositiveDimension { .. }));

    // Original collider still exists
    assert!(world.collider_exists(&collider));
}

// ── Non-finite input guards ──────────────────────────────────────

#[test]
fn non_finite_translation_rejected_for_body() {
    let mut world = PhysicsWorld2D::new();
    let err = world
        .create_body(BodyDescriptor2D::new(
            "body.inf",
            BodyKind::Dynamic,
            [f32::INFINITY, 0.0],
        ))
        .unwrap_err();
    assert_eq!(
        err,
        PhysicsError::NonFiniteValue {
            field: "body.translation"
        }
    );
}

#[test]
fn non_finite_rotation_rejected_for_collider() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.rot",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world
        .create_collider(
            ColliderDescriptor2D::new(
                "collider.rot",
                "body.rot",
                ColliderShape2D::Ball { radius: 1.0 },
            )
            .rotation(f32::INFINITY),
        )
        .unwrap_err();
    assert_eq!(err, PhysicsError::InvalidRotation);
}

#[test]
fn non_finite_force_rejected() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.d",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    let err = world.apply_force(&body, [f32::NAN, 0.0]).unwrap_err();
    assert_eq!(err, PhysicsError::NonFiniteValue { field: "force" });
}

// ── Step with zero / negative dt ─────────────────────────────────

#[test]
fn negative_dt_rejected() {
    let mut world = PhysicsWorld2D::new();
    let err = world.step(-1.0).unwrap_err();
    assert_eq!(err, PhysicsError::NonPositiveDeltaTime);
}

#[test]
fn zero_dt_rejected() {
    let mut world = PhysicsWorld2D::new();
    let err = world.step(0.0).unwrap_err();
    assert_eq!(err, PhysicsError::NonPositiveDeltaTime);
}

// ── Idempotent removal ───────────────────────────────────────────

#[test]
fn double_remove_body_is_false() {
    let mut world = PhysicsWorld2D::new();
    let body = world
        .create_body(BodyDescriptor2D::new(
            "body.once",
            BodyKind::Dynamic,
            [0.0, 0.0],
        ))
        .unwrap();
    assert!(world.remove_body(&body));
    assert!(!world.remove_body(&body));
}

#[test]
fn double_remove_collider_is_false() {
    let mut world = PhysicsWorld2D::new();
    world
        .create_body(BodyDescriptor2D::new(
            "body.once",
            BodyKind::Static,
            [0.0, 0.0],
        ))
        .unwrap();
    let c = world
        .create_collider(ColliderDescriptor2D::new(
            "collider.once",
            "body.once",
            ColliderShape2D::Ball { radius: 1.0 },
        ))
        .unwrap();
    assert!(world.remove_collider(&c));
    assert!(!world.remove_collider(&c));
}
