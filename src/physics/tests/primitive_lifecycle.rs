//! Transactional primitive lifecycle tests.
//!
//! Covers atomic registration, reconfiguration, collider replacement,
//! targeted removal, force/impulse/velocity/teleport, queries, and
//! character controller integration.

use physics::{
    BodyDescriptor, BodyKind, BodyMode, BodyPose, BodyRegistrationRequest, CharacterConfig,
    CharacterController, ColliderDescriptor, ColliderReplacementRequest, ColliderShape,
    PhysicsBodyId, PhysicsColliderId, PhysicsContactPhase, PhysicsError, PhysicsWorld,
    RayQuery, RegistrationOutcome, RemovalOutcome,
};

// ── Atomic registration ─────────────────────────────────────────────

#[test]
fn atomic_body_and_collider_registration() {
    let mut world = PhysicsWorld::new();
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0, 2.0, 0.0]),
        colliders: vec![ColliderDescriptor::new(
            "collider.hero",
            "body.hero",
            ColliderShape::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
        )],
    };

    let outcome: RegistrationOutcome = world.register_body(request).unwrap();
    assert_eq!(outcome.body_id, PhysicsBodyId::new("body.hero"));
    assert_eq!(outcome.collider_ids.len(), 1);
    assert_eq!(outcome.collider_ids[0], PhysicsColliderId::new("collider.hero"));

    assert!(world.body_exists(&PhysicsBodyId::new("body.hero")));
    assert!(world.collider_exists(&PhysicsColliderId::new("collider.hero")));
}

#[test]
fn atomic_registration_duplicate_body_rejected() {
    let mut world = PhysicsWorld::new();
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![],
    };
    world.register_body(request).unwrap();

    // Duplicate body ID
    let request2 = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [1.0; 3]),
        colliders: vec![],
    };
    let err = world.register_body(request2).unwrap_err();
    assert!(matches!(err, PhysicsError::DuplicateBodyId(_)));
    assert_eq!(err, PhysicsError::DuplicateBodyId(PhysicsBodyId::new("body.hero")));
}

#[test]
fn atomic_registration_duplicate_collider_rejected() {
    let mut world = PhysicsWorld::new();
    // Create a body and collider first to establish the duplicate
    world
        .create_body(BodyDescriptor::new("body.existing", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.existing",
            "body.existing",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    // Now try to register a new body with a collider using the same ID
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![ColliderDescriptor::new(
            "collider.existing",
            "body.hero",
            ColliderShape::Sphere { radius: 1.0 },
        )],
    };

    let err = world.register_body(request).unwrap_err();
    assert!(matches!(err, PhysicsError::DuplicateColliderId(_)));
}

#[test]
fn atomic_registration_missing_parent_rejected() {
    let mut world = PhysicsWorld::new();
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![ColliderDescriptor::new(
            "collider.orphan",
            "body.nonexistent",
            ColliderShape::Sphere { radius: 1.0 },
        )],
    };
    let err = world.register_body(request).unwrap_err();
    assert!(matches!(err, PhysicsError::MissingBody(_)));
}

#[test]
fn atomic_registration_invalid_shape_rejected() {
    let mut world = PhysicsWorld::new();
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![ColliderDescriptor::new(
            "collider.bad",
            "body.hero",
            ColliderShape::Sphere { radius: 0.0 },
        )],
    };
    let err = world.register_body(request).unwrap_err();
    assert!(matches!(
        err,
        PhysicsError::NonPositiveDimension { .. }
    ));
}

#[test]
fn atomic_registration_failure_leaves_world_unchanged() {
    let mut world = PhysicsWorld::new();
    let body_before = world.body_exists(&PhysicsBodyId::new("body.hero"));

    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![ColliderDescriptor::new(
            "collider.bad",
            "body.hero",
            ColliderShape::Sphere { radius: -1.0 },
        )],
    };
    let _ = world.register_body(request).unwrap_err();

    // World must be unchanged.
    assert_eq!(world.body_exists(&PhysicsBodyId::new("body.hero")), body_before);
    assert!(!world.collider_exists(&PhysicsColliderId::new("collider.bad")));
}

#[test]
fn atomic_registration_trimesh_on_dynamic_rejected() {
    let mut world = PhysicsWorld::new();
    let request = BodyRegistrationRequest {
        body: BodyDescriptor::new("body.bad", BodyKind::Dynamic, [0.0; 3]),
        colliders: vec![ColliderDescriptor::new(
            "collider.mesh",
            "body.bad",
            ColliderShape::TriMeshStatic {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![[0, 1, 2]],
            },
        )],
    };
    let err = world.register_body(request).unwrap_err();
    assert_eq!(err, PhysicsError::TrimeshOnDynamicBody);
}

// ── Body reconfiguration ─────────────────────────────────────────────

#[test]
fn reconfigure_dynamic_to_static_preserves_pose() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new(
            "body.test",
            BodyKind::Dynamic,
            [1.0, 2.0, 3.0],
        ))
        .unwrap();
    world.set_gravity(0.0, 0.0, 0.0);
    world.step(1.0).unwrap();

    assert!(world.body_is_dynamic(&body));
    world
        .reconfigure_body_mode(&body, BodyMode::Static)
        .unwrap();
    assert!(world.body_is_static(&body));
    assert!(!world.body_is_dynamic(&body));

    let pos = world.body_position_by_id(&body).unwrap();
    assert_eq!(pos, [1.0, 2.0, 3.0]);
}

#[test]
fn reconfigure_missing_body_errors() {
    let mut world = PhysicsWorld::new();
    let err = world
        .reconfigure_body_mode(&PhysicsBodyId::new("missing"), BodyMode::Dynamic)
        .unwrap_err();
    assert!(matches!(err, PhysicsError::MissingBody(_)));
}

#[test]
fn reconfigure_static_with_trimesh_to_dynamic_rejected() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            ColliderShape::TriMeshStatic {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![[0, 1, 2]],
            },
        ))
        .unwrap();

    let err = world
        .reconfigure_body_mode(&PhysicsBodyId::new("body.floor"), BodyMode::Dynamic)
        .unwrap_err();
    assert_eq!(err, PhysicsError::TrimeshOnDynamicBody);
}

// ── Collider replacement ─────────────────────────────────────────────

#[test]
fn replace_collider_swaps_shape() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    world
        .replace_collider(ColliderReplacementRequest {
            collider_id: collider.clone(),
            shape: ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            is_trigger: false,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap();

    // Collider still exists under same ID
    assert!(world.collider_exists(&collider));
    // Can still be queried
    world.step(1.0 / 60.0).unwrap();
}

#[test]
fn replace_missing_collider_errors() {
    let mut world = PhysicsWorld::new();
    let err = world
        .replace_collider(ColliderReplacementRequest {
            collider_id: PhysicsColliderId::new("missing"),
            shape: ColliderShape::Sphere { radius: 1.0 },
            is_trigger: false,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap_err();
    assert!(matches!(err, PhysicsError::MissingCollider(_)));
}

#[test]
fn replace_collider_invalid_shape_rejected_and_unchanged() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let err = world
        .replace_collider(ColliderReplacementRequest {
            collider_id: collider.clone(),
            shape: ColliderShape::Sphere { radius: 0.0 },
            is_trigger: false,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap_err();
    assert!(matches!(err, PhysicsError::NonPositiveDimension { .. }));
    // Collider should still exist and be usable
    assert!(world.collider_exists(&collider));
}

#[test]
fn replace_collider_trimesh_on_dynamic_rejected() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let err = world
        .replace_collider(ColliderReplacementRequest {
            collider_id: collider.clone(),
            shape: ColliderShape::TriMeshStatic {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![[0, 1, 2]],
            },
            is_trigger: false,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        })
        .unwrap_err();
    assert_eq!(err, PhysicsError::TrimeshOnDynamicBody);
}

// ── Targeted removal ─────────────────────────────────────────────────

#[test]
fn remove_body_with_outcome_returns_exit_records() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, 0.0, 0.0);

    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
        ))
        .unwrap();
    world
        .create_body(BodyDescriptor::new("body.b", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.b",
            "body.b",
            ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
        ))
        .unwrap();

    world.step(1.0 / 60.0).unwrap();
    // Bodies are intersecting → should have active pairs
    assert!(!world.last_contact_records().is_empty());

    let outcome: RemovalOutcome = world
        .remove_body_with_outcome(&PhysicsBodyId::new("body.a"))
        .unwrap();

    assert_eq!(outcome.removed_body, Some(PhysicsBodyId::new("body.a")));
    assert_eq!(outcome.removed_colliders, vec![PhysicsColliderId::new("collider.a")]);
    // Exit records should be generated for the removed pairs
    assert!(!outcome.exited_pairs.is_empty());
    assert!(outcome
        .exited_pairs
        .iter()
        .all(|r| r.phase == PhysicsContactPhase::Exit));
}

#[test]
fn remove_collider_with_outcome_returns_exit_records() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, 0.0, 0.0);

    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
        ))
        .unwrap();
    world
        .create_body(BodyDescriptor::new("body.b", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.b",
            "body.b",
            ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
        ))
        .unwrap();

    world.step(1.0 / 60.0).unwrap();

    let outcome: RemovalOutcome = world
        .remove_collider_with_outcome(&PhysicsColliderId::new("collider.a"))
        .unwrap();

    assert_eq!(outcome.removed_body, None);
    assert_eq!(outcome.removed_colliders, vec![PhysicsColliderId::new("collider.a")]);
    assert!(!outcome.exited_pairs.is_empty());
}

#[test]
fn remove_with_outcome_missing_is_none() {
    let mut world = PhysicsWorld::new();
    assert!(world
        .remove_body_with_outcome(&PhysicsBodyId::new("missing"))
        .is_none());
    assert!(world
        .remove_collider_with_outcome(&PhysicsColliderId::new("missing"))
        .is_none());
}

#[test]
fn bool_removal_still_works_as_compat() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    assert!(world.remove_collider(&collider));
    assert!(!world.remove_collider(&collider)); // idempotent
    assert!(world.remove_body(&body));
    assert!(!world.remove_body(&body)); // idempotent
}

// ── Force / impulse / velocity / teleport ────────────────────────────

#[test]
fn apply_force_affects_dynamic_body() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, 0.0, 0.0);
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    world.apply_force(&body, [10.0, 0.0, 0.0]).unwrap();
    world.step(1.0 / 60.0).unwrap();

    let vel = world.body_linear_velocity(&body).unwrap();
    assert!(vel[0] > 0.0, "force should increase x velocity");
}

#[test]
fn apply_impulse_changes_velocity_immediately() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, 0.0, 0.0);
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    world.apply_impulse(&body, [0.0, 10.0, 0.0]).unwrap();
    // Impulse takes effect immediately (before stepping)
    let vel = world.body_linear_velocity(&body).unwrap();
    assert!(vel[1] > 0.0, "impulse should set y velocity immediately");
}

#[test]
fn force_and_impulse_on_static_body_noop() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0; 3]))
        .unwrap();

    world.apply_force(&body, [100.0, 0.0, 0.0]).unwrap();
    world.apply_impulse(&body, [100.0, 0.0, 0.0]).unwrap();
    world.apply_torque_impulse(&body, [100.0, 0.0, 0.0]).unwrap();

    // No error, but no movement either
    assert!(world.body_is_static(&body));
}

#[test]
fn velocity_set_on_dynamic_body() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();

    world.set_linear_velocity(&body, [5.0, 0.0, 0.0]).unwrap();
    world.set_angular_velocity(&body, [0.0, 1.0, 0.0]).unwrap();

    let lin = world.body_linear_velocity(&body).unwrap();
    assert_eq!(lin, [5.0, 0.0, 0.0]);

    let ang = world.body_angular_velocity(&body).unwrap();
    assert_eq!(ang, [0.0, 1.0, 0.0]);
}

#[test]
fn velocity_on_static_body_silent_noop() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0; 3]))
        .unwrap();

    world.set_linear_velocity(&body, [5.0, 0.0, 0.0]).unwrap();
    world.set_angular_velocity(&body, [1.0, 0.0, 0.0]).unwrap();
    // No error, but velocity is not queryable for static bodies
}

#[test]
fn wake_and_sleep_body() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();

    world.sleep_body(&body).unwrap();
    world.wake_body(&body).unwrap();
    // No observable change without stepping, but calls should succeed
}

#[test]
fn teleport_body_changes_position() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Kinematic, [1.0, 2.0, 3.0]))
        .unwrap();

    world
        .teleport_body(
            &body,
            BodyPose {
                translation: [10.0, 20.0, 30.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        )
        .unwrap();

    let pos = world.body_position_by_id(&body).unwrap();
    assert_eq!(pos, [10.0, 20.0, 30.0]);
}

#[test]
fn teleport_body_invalid_pose_rejected() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();

    let err = world
        .teleport_body(
            &body,
            BodyPose {
                translation: [f32::NAN, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        )
        .unwrap_err();
    assert!(matches!(err, PhysicsError::NonFiniteValue { .. }));
}

// ── Queries ──────────────────────────────────────────────────────────

#[test]
fn overlap_sphere_finds_intersecting_colliders() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Static, [0.0, 0.0, 0.0]))
        .unwrap();
    let collider_a = world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    world
        .create_body(BodyDescriptor::new("body.b", BodyKind::Static, [10.0, 0.0, 0.0]))
        .unwrap();
    let _collider_b = world
        .create_collider(ColliderDescriptor::new(
            "collider.b",
            "body.b",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let results = world.overlap_sphere([0.0, 0.0, 0.0], 0.5).unwrap();
    assert!(results.iter().any(|r| r.collider == collider_a));
}

#[test]
fn overlap_sphere_empty_when_no_overlap() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Static, [10.0, 0.0, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Sphere { radius: 0.5 },
        ))
        .unwrap();

    let results = world.overlap_sphere([-10.0, 0.0, 0.0], 1.0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn overlap_aabb_finds_intersecting_colliders() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Static, [0.0, 0.0, 0.0]))
        .unwrap();
    let collider_a = world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
        ))
        .unwrap();

    let results = world
        .overlap_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        .unwrap();
    assert!(results.iter().any(|r| r.collider == collider_a));
}

#[test]
fn overlap_aabb_invalid_range_rejected() {
    let mut world = PhysicsWorld::new();
    let err = world
        .overlap_aabb([1.0, 0.0, 0.0], [0.0, 1.0, 1.0])
        .unwrap_err();
    assert!(matches!(err, PhysicsError::NonFiniteValue { .. }));
}

#[test]
fn sweep_test_hits_static_collider() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0, 0.0, 0.0]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            ColliderShape::Cuboid {
                half_extents: [5.0, 0.5, 5.0],
            },
        ))
        .unwrap();

    let hit = world
        .sweep_test(
            &ColliderShape::Sphere { radius: 0.5 },
            BodyPose {
                translation: [0.0, 5.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            [0.0, -10.0, 0.0],
        )
        .unwrap()
        .unwrap();

    assert_eq!(hit.collider, collider);
    assert!(hit.time_of_impact > 0.0 && hit.time_of_impact < 1.0);
}

#[test]
fn sweep_test_miss_returns_none() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Static, [10.0, 0.0, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let hit = world
        .sweep_test(
            &ColliderShape::Sphere { radius: 0.5 },
            BodyPose {
                translation: [-10.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            [0.0, 10.0, 0.0],
        )
        .unwrap();
    assert!(hit.is_none());
}

#[test]
fn overlap_results_are_deterministically_sorted() {
    let mut world = PhysicsWorld::new();

    // Create several colliders
    for i in 0..5 {
        let body_id = PhysicsBodyId::new(format!("body.{i}"));
        let collider_id = PhysicsColliderId::new(format!("collider.{i}"));
        world
            .create_body(BodyDescriptor::new(body_id.clone(), BodyKind::Static, [0.0; 3]))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                collider_id,
                body_id,
                ColliderShape::Sphere { radius: 1.0 },
            ))
            .unwrap();
    }

    let results1 = world.overlap_sphere([0.0, 0.0, 0.0], 100.0).unwrap();
    let results2 = world.overlap_sphere([0.0, 0.0, 0.0], 100.0).unwrap();
    assert_eq!(results1, results2, "overlap results must be deterministic");
    assert!(results1.windows(2).all(|w| w[0] <= w[1]), "results must be sorted");
}

// ── Body introspection ───────────────────────────────────────────────

#[test]
fn body_kind_queries() {
    let mut world = PhysicsWorld::new();
    let s = world
        .create_body(BodyDescriptor::new("body.static", BodyKind::Static, [0.0; 3]))
        .unwrap();
    let d = world
        .create_body(BodyDescriptor::new("body.dynamic", BodyKind::Dynamic, [0.0; 3]))
        .unwrap();
    let k = world
        .create_body(BodyDescriptor::new("body.kinematic", BodyKind::Kinematic, [0.0; 3]))
        .unwrap();

    assert!(world.body_is_static(&s));
    assert!(!world.body_is_dynamic(&s));
    assert!(!world.body_is_kinematic(&s));

    assert!(!world.body_is_static(&d));
    assert!(world.body_is_dynamic(&d));
    assert!(!world.body_is_kinematic(&d));

    assert!(!world.body_is_static(&k));
    assert!(!world.body_is_dynamic(&k));
    assert!(world.body_is_kinematic(&k));
}

#[test]
fn body_velocity_none_for_missing() {
    let world = PhysicsWorld::new();
    assert!(world
        .body_linear_velocity(&PhysicsBodyId::new("missing"))
        .is_none());
    assert!(world
        .body_angular_velocity(&PhysicsBodyId::new("missing"))
        .is_none());
}

// ── Character controller ─────────────────────────────────────────────

#[test]
fn character_controller_move_and_slide_ground_detection() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, -9.81, 0.0);

    // Static floor
    world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0, -0.5, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            ColliderShape::Cuboid {
                half_extents: [10.0, 0.5, 10.0],
            },
        ))
        .unwrap();

    // Kinematic character
    let body_id = world
        .create_body(BodyDescriptor::new(
            "body.character",
            BodyKind::Kinematic,
            [0.0, 1.0, 0.0],
        ))
        .unwrap();
    let collider_id = world
        .create_collider(ColliderDescriptor::new(
            "collider.character",
            body_id.clone(),
            ColliderShape::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
        ))
        .unwrap();

    let config = CharacterConfig::default();
    let mut controller =
        CharacterController::new(&world, body_id.clone(), collider_id, config).unwrap();

    world.step(1.0 / 60.0).unwrap();

    // Move down — should land on floor
    controller
        .move_and_slide(&mut world, [0.0, -2.0, 0.0], 1.0 / 60.0)
        .unwrap();

    assert!(controller.is_on_floor(), "character should be on floor");
    let pos = world.body_position_by_id(&body_id).unwrap();
    assert!(pos[1] > -0.5, "character should rest on top of floor, got {:?}", pos);
}

#[test]
fn character_controller_horizontal_slide() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(0.0, -9.81, 0.0);

    // Floor
    world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0, -0.5, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            ColliderShape::Cuboid {
                half_extents: [10.0, 0.5, 10.0],
            },
        ))
        .unwrap();

    // Wall
    world
        .create_body(BodyDescriptor::new("body.wall", BodyKind::Static, [3.0, 0.5, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.wall",
            "body.wall",
            ColliderShape::Cuboid {
                half_extents: [0.5, 1.5, 10.0],
            },
        ))
        .unwrap();

    let body_id = world
        .create_body(BodyDescriptor::new(
            "body.character",
            BodyKind::Kinematic,
            [0.0, 1.0, 0.0],
        ))
        .unwrap();
    let collider_id = world
        .create_collider(ColliderDescriptor::new(
            "collider.character",
            body_id.clone(),
            ColliderShape::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
        ))
        .unwrap();

    let config = CharacterConfig::default();
    let mut controller =
        CharacterController::new(&world, body_id.clone(), collider_id, config).unwrap();

    world.step(1.0 / 60.0).unwrap();

    // Drop to floor first
    controller
        .move_and_slide(&mut world, [0.0, -2.0, 0.0], 1.0 / 60.0)
        .unwrap();

    // Try to walk through the wall
    let actual = controller
        .move_and_slide(&mut world, [10.0, 0.0, 0.0], 1.0 / 60.0)
        .unwrap();

    // Should not move through wall
    assert!(actual[0] < 10.0, "wall should block movement");
    let pos = world.body_position_by_id(&body_id).unwrap();
    assert!(pos[0] < 3.0, "character should not go through wall");
}

#[test]
fn character_controller_missing_body_rejected() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.floor", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.floor",
            "body.floor",
            ColliderShape::Cuboid {
                half_extents: [10.0, 0.5, 10.0],
            },
        ))
        .unwrap();

    let config = CharacterConfig::default();
    let result = CharacterController::new(
        &world,
        PhysicsBodyId::new("body.nonexistent"),
        PhysicsColliderId::new("collider.floor"),
        config,
    );
    assert!(result.is_err());
}

#[test]
fn character_controller_wrong_parent_rejected() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.a", BodyKind::Kinematic, [0.0, 1.0, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.a",
            "body.a",
            ColliderShape::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
        ))
        .unwrap();
    // Different body
    world
        .create_body(BodyDescriptor::new("body.b", BodyKind::Static, [0.0; 3]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.b",
            "body.b",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let config = CharacterConfig::default();
    // Reference body.a with collider.b (wrong parent)
    let result = CharacterController::new(
        &world,
        PhysicsBodyId::new("body.a"),
        PhysicsColliderId::new("collider.b"),
        config,
    );
    assert!(result.is_err());
}

// ── Existing API compatibility ───────────────────────────────────────

#[test]
fn existing_create_body_and_collider_still_work() {
    let mut world = PhysicsWorld::new();
    let body = world
        .create_body(BodyDescriptor::new("body.test", BodyKind::Dynamic, [1.0, 2.0, 3.0]))
        .unwrap();
    let collider = world
        .create_collider(ColliderDescriptor::new(
            "collider.test",
            body.clone(),
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();
    assert_eq!(body, PhysicsBodyId::new("body.test"));
    assert_eq!(collider, PhysicsColliderId::new("collider.test"));
}

#[test]
fn existing_ray_query_still_works() {
    let mut world = PhysicsWorld::new();
    world
        .create_body(BodyDescriptor::new("body.target", BodyKind::Static, [0.0, 0.0, 0.0]))
        .unwrap();
    world
        .create_collider(ColliderDescriptor::new(
            "collider.target",
            "body.target",
            ColliderShape::Sphere { radius: 1.0 },
        ))
        .unwrap();

    let hit = world
        .cast_ray(RayQuery::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 10.0))
        .unwrap()
        .unwrap();
    assert_eq!(hit.body, PhysicsBodyId::new("body.target"));
    assert_eq!(hit.collider, PhysicsColliderId::new("collider.target"));
}
