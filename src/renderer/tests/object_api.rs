//! Tests for Phase 04 — Object Capabilities and Grouping.
//!
//! Covers: capability matrices, finite/affine/scale/shear/reflection rejection,
//! roll-free direction round-trip, wrong-kind ops, world-space grouping invariance,
//! node-subtree detach/restore, explicit vs implicit light duplication,
//! persistent reminting, light caps, root constraints, shadow exclusivity,
//! component preservation, deterministic remaps.

use glam::{Mat4, Vec3, Vec4};
use renderer::{
    DirectionalLight, DirectionalShadowConfig, PointLight, Scene, SceneError, SpotLight,
    object::{
        identity::ObjectError,
        ObjectKind, ObjectParent, ObjectTransform,
        is_identity_basis_matrix, is_rigid_direction_only, is_rigid_matrix,
    },
};
use renderer::object::ObjectCapabilities;

// ── Helpers ─────────────────────────────────────────────────────────────

fn new_scene() -> Scene {
    Scene::new()
}

// ── Capability Matrices ────────────────────────────────────────────────

#[test]
fn node_capabilities() {
    let caps = ObjectCapabilities::for_kind(ObjectKind::Node);
    assert!(caps.supports_transform);
    assert!(caps.supports_children);
    assert!(!caps.supports_grouping);
    assert!(caps.supports_duplication);
    assert!(caps.supports_subtree_removal);
}

#[test]
fn point_light_capabilities() {
    let caps = ObjectCapabilities::for_kind(ObjectKind::PointLight);
    assert!(caps.supports_transform);
    assert!(!caps.supports_children);
    assert!(caps.supports_grouping);
    assert!(!caps.supports_subtree_removal);
}

#[test]
fn directional_light_capabilities() {
    let caps = ObjectCapabilities::for_kind(ObjectKind::DirectionalLight);
    assert!(caps.supports_transform);
    assert!(!caps.supports_children);
    assert!(caps.supports_grouping);
}

#[test]
fn spot_light_capabilities() {
    let caps = ObjectCapabilities::for_kind(ObjectKind::SpotLight);
    assert!(caps.supports_transform);
    assert!(caps.supports_grouping);
    assert!(caps.supports_duplication);
}

// ── Rigid Matrix Validation ────────────────────────────────────────────

#[test]
fn identity_is_rigid() {
    assert!(is_rigid_matrix(&Mat4::IDENTITY));
}

#[test]
fn translation_is_not_rigid() {
    // Pure translation: basis is identity, determinant is 1, but w_axis has translation.
    // Actually, translation IS rigid (affine, basis identity, det ≈ 1, last row [0,0,0,1]).
    // Let me re-check: rigid means finite affine, unit basis, orthogonal, det ≈ 1.
    // Mat4::from_translation has: x=[1,0,0,0], y=[0,1,0,0], z=[0,0,1,0], w=[tx,ty,tz,1]
    // Row 3 = [0,0,0,1] ✓, basis unit ✓, orthogonal ✓, det = 1 ✓
    // So translation IS rigid.
    let t = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    assert!(is_rigid_matrix(&t));
}

#[test]
fn scale_rejected() {
    let s = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
    assert!(!is_rigid_matrix(&s));
}

#[test]
fn shear_rejected() {
    // Construct a shear matrix manually.
    let shear = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.5, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    // Non-orthogonal basis vectors
    assert!(!is_rigid_matrix(&shear));
}

#[test]
fn reflection_rejected() {
    // Negative scale on one axis
    let r = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    // det = -1
    assert!(!is_rigid_matrix(&r));
}

#[test]
fn nan_rejected() {
    let m = Mat4::from_cols(
        Vec4::new(f32::NAN, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    assert!(!is_rigid_matrix(&m));
}

#[test]
fn identity_basis_detection() {
    assert!(is_identity_basis_matrix(&Mat4::IDENTITY));
    assert!(is_identity_basis_matrix(&Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))));
    // Non-identity basis
    assert!(!is_identity_basis_matrix(&Mat4::from_rotation_x(0.5)));
}

#[test]
fn rigid_direction_only_detection() {
    assert!(is_rigid_direction_only(&Mat4::IDENTITY));
    // Rotation with zero translation
    assert!(is_rigid_direction_only(&Mat4::from_rotation_x(0.5)));
    // Rotation with non-zero translation
    assert!(!is_rigid_direction_only(&Mat4::from_rotation_translation(
        glam::Quat::from_rotation_x(0.5),
        Vec3::new(1.0, 0.0, 0.0),
    )));
}

// ── Roll-free direction round-trip ─────────────────────────────────────

#[test]
fn roll_free_direction_roundtrip() {
    // Forward = NEG_Z
    let dir = Vec3::new(0.3, 0.8, -0.5).normalize();
    let mat = ObjectTransform::rigid_from_direction(dir);
    assert!(is_rigid_matrix(&mat));

    let recovered = ObjectTransform::direction_from_rigid(&mat);
    // The direction should be preserved (within FP tolerance)
    let diff = (dir - recovered).length();
    assert!(diff < 0.001, "direction round-trip: diff={diff}");
}

#[test]
fn direction_near_y_axis_fallback() {
    // Direction nearly collinear with Y axis
    let dir = Vec3::new(0.0, 0.99999, 0.0).normalize();
    let mat = ObjectTransform::rigid_from_direction(dir);
    assert!(is_rigid_matrix(&mat));

    let recovered = ObjectTransform::direction_from_rigid(&mat);
    assert!((dir - recovered).length() < 0.001);
}

#[test]
fn direction_near_neg_y_fallback() {
    let dir = Vec3::new(0.0, -0.99999, 0.0).normalize();
    let mat = ObjectTransform::rigid_from_direction(dir);
    assert!(is_rigid_matrix(&mat));
    let recovered = ObjectTransform::direction_from_rigid(&mat);
    assert!((dir - recovered).length() < 0.001);
}

// ── Object Enumeration ─────────────────────────────────────────────────

#[test]
fn objects_returns_all_kinds() {
    let mut scene = new_scene();
    scene.create_node_default(None).unwrap();
    scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();

    let objects = scene.objects();
    assert_eq!(objects.len(), 2);
    let kinds: Vec<ObjectKind> = objects.iter().map(|id| id.kind()).collect();
    assert!(kinds.contains(&ObjectKind::Node));
    assert!(kinds.contains(&ObjectKind::PointLight));
}

#[test]
fn objects_of_kind_filters() {
    let mut scene = new_scene();
    scene.create_node_default(None).unwrap();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();

    let nodes = scene.objects_of_kind(ObjectKind::Node);
    assert_eq!(nodes.len(), 1);

    let point_lights = scene.objects_of_kind(ObjectKind::PointLight);
    assert_eq!(point_lights.len(), 1);
    let oid = scene.object_id_for_point_light(pl).unwrap();
    assert_eq!(point_lights[0], oid);
}

// ── Object Summary ─────────────────────────────────────────────────────

#[test]
fn object_summary_has_fields() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let oid = scene.object_id(node).unwrap();

    let summary = scene.object_summary(oid).unwrap();
    assert_eq!(summary.kind, ObjectKind::Node);
    assert!(summary.name.len() > 0);
    assert_eq!(summary.mesh_count, 0);
    assert_eq!(summary.child_count, 0);
    assert_eq!(summary.tags.len(), 0);
}

// ── Wrong-kind ops ─────────────────────────────────────────────────────

#[test]
fn try_get_wrong_kind_rejected() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let oid = scene.object_id_for_point_light(pl).unwrap();

    let err = scene.try_get_node_id(oid).unwrap_err();
    match err {
        SceneError::Object(ObjectError::WrongKind { expected, actual, .. }) => {
            assert_eq!(expected, ObjectKind::Node);
            assert_eq!(actual, ObjectKind::PointLight);
        }
        _ => panic!("expected WrongKind error"),
    }
}

#[test]
fn wrong_scene_precedes_kind_validation() {
    let mut source = new_scene();
    let foreign = source.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    }).unwrap();
    let foreign_id = source.object_id_for_point_light(foreign).unwrap();

    let target = new_scene();
    let err = target.try_get_node_id(foreign_id).unwrap_err();
    assert!(matches!(err, SceneError::Object(ObjectError::WrongScene { .. })));
}

// ── Unified Transform API ──────────────────────────────────────────────

#[test]
fn node_transform_get_set() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let oid = scene.object_id(node).unwrap();

    let t = scene.get_object_transform(oid).unwrap();
    match t {
        ObjectTransform::Node(m) => assert_eq!(m, Mat4::IDENTITY),
        _ => panic!("expected Node transform"),
    }

    let new_mat = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    scene.set_object_transform(oid, &new_mat).unwrap();
    assert_eq!(scene.transform(node).unwrap(), new_mat);
}

#[test]
fn point_light_transform_rejects_scale() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let oid = scene.object_id_for_point_light(pl).unwrap();

    // Translation-only is fine
    scene.set_object_transform(
        oid,
        &Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
    ).unwrap();

    // Scale must be rejected
    let scaled = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
    let err = scene.set_object_transform(oid, &scaled).unwrap_err();
    assert!(format!("{err}").contains("translation-only"));

    // Position should be unchanged after rejection
    let t = scene.get_object_transform(oid).unwrap();
    assert!(matches!(t, ObjectTransform::PointLight(p) if (p - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001));
}

#[test]
fn spot_light_transform_rejects_shear() {
    let mut scene = new_scene();
    let sl = scene.create_spot_light(SpotLight::new(
        Vec3::ZERO,
        Vec3::NEG_Z,
        Vec3::ONE,
        1.0,
        10.0,
        0.1,
        0.5,
    )).unwrap();
    let oid = scene.object_id_for_spot_light(sl).unwrap();

    // Valid rigid transform
    let rigid = Mat4::from_rotation_translation(
        glam::Quat::from_rotation_x(0.3),
        Vec3::new(1.0, 0.0, 0.0),
    );
    scene.set_object_transform(oid, &rigid).unwrap();

    // Re-check position and direction updated
    let t = scene.get_object_transform(oid).unwrap();
    let expected_dir = ObjectTransform::direction_from_rigid(&rigid);
    match t {
        ObjectTransform::SpotLight { position, direction } => {
            assert!((position - Vec3::new(1.0, 0.0, 0.0)).length() < 0.001);
            assert!((direction - expected_dir).length() < 0.001);
        }
        _ => panic!("expected SpotLight transform"),
    }

    // Shear must be rejected
    let shear = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.5, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let err = scene.set_object_transform(oid, &shear).unwrap_err();
    assert!(format!("{err}").contains("rigid"));
}

#[test]
fn directional_light_transform_rejects_translation() {
    let mut scene = new_scene();
    let dl = scene.add_directional_light(DirectionalLight {
        direction: Vec3::NEG_Z,
        color: Vec3::ONE,
        intensity: 1.0,
    }).unwrap();
    let oid = scene.object_id_for_directional_light(dl).unwrap();

    // Rigid direction only is fine
    let rot = Mat4::from_rotation_x(0.5);
    scene.set_object_transform(oid, &rot).unwrap();

    // Translation must be rejected
    let with_trans = Mat4::from_rotation_translation(
        glam::Quat::from_rotation_x(0.5),
        Vec3::new(1.0, 0.0, 0.0),
    );
    let err = scene.set_object_transform(oid, &with_trans).unwrap_err();
    assert!(format!("{err}").contains("zero translation"));
}

// ── Typed transform getters ────────────────────────────────────────────

#[test]
fn point_light_typed_transform() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::new(3.0, 4.0, 5.0),
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();

    let mat = scene.point_light_transform(pl).unwrap();
    assert_eq!(mat.w_axis.truncate(), Vec3::new(3.0, 4.0, 5.0));
}

#[test]
fn directional_light_typed_transform_roundtrip() {
    let mut scene = new_scene();
    let dl = scene.add_directional_light(DirectionalLight {
        direction: Vec3::new(0.3, 0.8, -0.5).normalize(),
        color: Vec3::ONE,
        intensity: 1.0,
    }).unwrap();

    let mat = scene.directional_light_transform(dl).unwrap();
    assert!(is_rigid_matrix(&mat));
    // Translation should be zero
    assert!(mat.w_axis.truncate().length() < 0.001);
    // Direction should be recovered
    let recovered = ObjectTransform::direction_from_rigid(&mat);
    let expected = Vec3::new(0.3, 0.8, -0.5).normalize();
    assert!((recovered - expected).length() < 0.001, "dir round-trip");
}

// ── Object Parent API ──────────────────────────────────────────────────

#[test]
fn node_reparent_via_object_parent() {
    let mut scene = new_scene();
    let parent = scene.create_node_default(None).unwrap();
    let child = scene.create_node_default(None).unwrap();

    let child_oid = scene.object_id(child).unwrap();
    let parent_oid = scene.object_id(parent).unwrap();

    scene.set_object_parent(child_oid, ObjectParent::Node(parent_oid)).unwrap();

    let result = scene.get_object_parent(child_oid).unwrap();
    assert_eq!(result, ObjectParent::Node(parent_oid));
}

#[test]
fn light_group_does_not_change_payload() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::new(1.0, 2.0, 3.0),
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();

    let pl_oid = scene.object_id_for_point_light(pl).unwrap();
    let node_oid = scene.object_id(node).unwrap();

    // Record position before grouping
    let pos_before = match scene.get_object_transform(pl_oid).unwrap() {
        ObjectTransform::PointLight(p) => p,
        _ => panic!(),
    };

    scene.set_object_parent(pl_oid, ObjectParent::Node(node_oid)).unwrap();

    // Position unchanged
    let pos_after = match scene.get_object_transform(pl_oid).unwrap() {
        ObjectTransform::PointLight(p) => p,
        _ => panic!(),
    };
    assert_eq!(pos_before, pos_after);

    // Group parent is set
    match scene.get_object_parent(pl_oid).unwrap() {
        ObjectParent::Node(oid) => assert_eq!(oid, node_oid),
        _ => panic!("expected Node parent"),
    }
}

#[test]
fn light_ungroup_via_object_parent() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let pl_oid = scene.object_id_for_point_light(pl).unwrap();
    let node_oid = scene.object_id(node).unwrap();

    scene.set_object_parent(pl_oid, ObjectParent::Node(node_oid)).unwrap();
    scene.set_object_parent(pl_oid, ObjectParent::None).unwrap();

    assert_eq!(
        scene.get_object_parent(pl_oid).unwrap(),
        ObjectParent::None
    );
}

#[test]
fn light_group_parent_must_be_node() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let pl_oid = scene.object_id_for_point_light(pl).unwrap();

    // Try to parent a light under another light — should fail
    let pl2 = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let pl2_oid = scene.object_id_for_point_light(pl2).unwrap();

    let err = scene.set_object_parent(pl_oid, ObjectParent::Node(pl2_oid)).unwrap_err();
    match err {
        SceneError::Object(ObjectError::WrongKind { expected, actual, .. }) => {
            assert_eq!(expected, ObjectKind::Node);
            assert_eq!(actual, ObjectKind::PointLight);
        }
        _ => panic!("expected WrongKind error"),
    }
}

#[test]
fn grouping_all_light_kinds_preserves_world_payload() {
    let mut scene = new_scene();
    let group = scene.create_node_default(None).unwrap();
    let group_id = scene.object_id(group).unwrap();
    let point = scene.create_point_light(PointLight {
        position: Vec3::new(1.0, 2.0, 3.0), color: Vec3::ONE, intensity: 1.0, range: 10.0,
    }).unwrap();
    let directional = scene.add_directional_light(DirectionalLight {
        direction: Vec3::new(0.2, 0.7, -0.3).normalize(), color: Vec3::ONE, intensity: 1.0,
    }).unwrap();
    let spot = scene.create_spot_light(SpotLight::new(
        Vec3::new(4.0, 5.0, 6.0), Vec3::new(0.1, -0.4, -0.9).normalize(),
        Vec3::ONE, 1.0, 10.0, 0.1, 0.5,
    )).unwrap();

    for id in [
        scene.object_id_for_point_light(point).unwrap(),
        scene.object_id_for_directional_light(directional).unwrap(),
        scene.object_id_for_spot_light(spot).unwrap(),
    ] {
        let before = scene.get_object_transform(id).unwrap();
        scene.set_object_parent(id, ObjectParent::Node(group_id)).unwrap();
        assert_eq!(scene.get_object_transform(id).unwrap(), before);
    }
}

// ── Subtree Removal / Restoration ──────────────────────────────────────

#[test]
fn node_subtree_remove_and_restore() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let child = scene.create_node_default(Some(root)).unwrap();

    let root_oid_before = scene.object_id(root).unwrap();

    let snapshot = scene.remove_node_subtree(root).unwrap();
    assert!(!scene.is_valid_node(root));
    assert!(!scene.is_valid_node(child));

    let outcome = scene.restore_subtree(snapshot).unwrap();
    assert!(!outcome.remaps.is_empty());
    assert!(!outcome.created_roots.is_empty());

    let new_root_oid = outcome.created_roots[0];
    assert_ne!(new_root_oid, root_oid_before);
    assert_eq!(new_root_oid.kind(), ObjectKind::Node);

    // New root should be valid
    let new_root = scene.try_get_node_id(new_root_oid).unwrap();
    assert!(scene.is_valid_node(new_root));
}

#[test]
fn subtree_restore_reattaches_grouped_lights_without_payload_change() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let root_id = scene.object_id(root).unwrap();
    let light = scene.create_point_light(PointLight {
        position: Vec3::new(7.0, 8.0, 9.0), color: Vec3::ONE, intensity: 2.0, range: 20.0,
    }).unwrap();
    let light_id = scene.object_id_for_point_light(light).unwrap();
    let payload = scene.get_object_transform(light_id).unwrap();
    scene.set_object_parent(light_id, ObjectParent::Node(root_id)).unwrap();

    let snapshot = scene.remove_node_subtree(root).unwrap();
    assert_eq!(scene.get_object_parent(light_id).unwrap(), ObjectParent::None);
    let outcome = scene.restore_subtree(snapshot).unwrap();
    assert_eq!(scene.get_object_parent(light_id).unwrap(), ObjectParent::Node(outcome.created_roots[0]));
    assert_eq!(scene.get_object_transform(light_id).unwrap(), payload);
}

#[test]
fn remove_node_subtree_with_outcome() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let outcome = scene.remove_node_subtree_with_outcome(root).unwrap();
    assert!(!outcome.snapshots.is_empty());
    assert!(!scene.is_valid_node(root));
}

// ── Duplication ────────────────────────────────────────────────────────

#[test]
fn duplicate_node_mints_new_ids() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let root_oid = scene.object_id(root).unwrap();

    let outcome = scene.duplicate_node(root, None).unwrap();
    assert!(!outcome.remaps.is_empty());
    assert_eq!(outcome.created_roots.len(), 1);

    let new_oid = outcome.created_roots[0];
    assert_ne!(new_oid, root_oid);
    assert_eq!(new_oid.kind(), ObjectKind::Node);

    let new_root = scene.try_get_node_id(new_oid).unwrap();
    assert!(scene.is_valid_node(new_root));
}

#[test]
fn duplicate_node_with_children() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let _child = scene.create_node_default(Some(root)).unwrap();

    let outcome = scene.duplicate_node(root, None).unwrap();
    let new_root = scene.try_get_node_id(outcome.created_roots[0]).unwrap();

    // New root should have exactly one child
    let new_root_summary = scene.object_summary(outcome.created_roots[0]).unwrap();
    assert_eq!(new_root_summary.child_count, 1);
    assert!(scene.is_valid_node(new_root));
}

#[test]
fn duplicate_node_never_implicitly_duplicates_grouped_lights() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let node_id = scene.object_id(node).unwrap();
    let light = scene.create_point_light(PointLight {
        position: Vec3::ZERO, color: Vec3::ONE, intensity: 1.0, range: 10.0,
    }).unwrap();
    let light_id = scene.object_id_for_point_light(light).unwrap();
    scene.set_object_parent(light_id, ObjectParent::Node(node_id)).unwrap();

    scene.duplicate_node(node, None).unwrap();
    assert_eq!(scene.objects_of_kind(ObjectKind::PointLight).len(), 1);
}

#[test]
fn duplicate_point_light_preserves_world_state() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::new(5.0, 6.0, 7.0),
        color: Vec3::ONE,
        intensity: 2.0,
        range: 20.0,
    })
    .unwrap();

    let outcome = scene.duplicate_point_light(pl).unwrap();
    let new_oid = outcome.created_roots[0];
    let new_pl_id = scene.try_get_point_light_id(new_oid).unwrap();
    let light = scene.world().point_light_entry(new_pl_id).unwrap();
    assert_eq!(light.position, Vec3::new(5.0, 6.0, 7.0));
    assert_eq!(light.intensity, 2.0);
    assert_eq!(light.range, 20.0);
}

#[test]
fn duplicate_directional_light_shadow_non_owning() {
    let mut scene = new_scene();
    let dl = scene.add_directional_light(DirectionalLight {
        direction: Vec3::NEG_Z,
        color: Vec3::ONE,
        intensity: 1.0,
    }).unwrap();

    scene.set_directional_shadow_config(dl, DirectionalShadowConfig {
        enabled: true,
        ..Default::default()
    }).unwrap();
    let outcome = scene.duplicate_directional_light(dl).unwrap();
    let new_oid = outcome.created_roots[0];
    let new_dl_id = scene.try_get_directional_light_id(new_oid).unwrap();

    // Verify the duplicated light has no shadow config (non-owning)
    let record = scene.world().get_directional_light_record(new_dl_id);
    assert!(record.is_some());
    assert!(record.unwrap().directional_shadow_config.is_none());
    assert_eq!(scene.shadow_casting_directional_light_id(), Some(dl));
}

#[test]
fn duplicate_spot_light_preserves_world_state() {
    let mut scene = new_scene();
    let sl = scene.create_spot_light(SpotLight::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::NEG_Z,
        Vec3::ONE,
        3.0,
        15.0,
        0.2,
        0.6,
    )).unwrap();

    let outcome = scene.duplicate_spot_light(sl).unwrap();
    let new_oid = outcome.created_roots[0];
    let new_sl_id = scene.try_get_spot_light_id(new_oid).unwrap();
    let light = scene.world().spot_light_entry(new_sl_id).unwrap();
    assert_eq!(light.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(light.intensity, 3.0);
    assert_eq!(light.range, 15.0);
}

#[test]
fn duplicate_object_dispatches_by_kind() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let oid = scene.object_id_for_point_light(pl).unwrap();

    let outcome = scene.duplicate_object(oid).unwrap();
    assert_eq!(outcome.created_roots.len(), 1);
    assert_eq!(outcome.created_roots[0].kind(), ObjectKind::PointLight);
}

// ── Light caps respected on duplication ────────────────────────────────

#[test]
fn duplicate_point_light_outcome_has_remap() {
    let mut scene = new_scene();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();

    let outcome = scene.duplicate_point_light(pl).unwrap();
    assert!(outcome.remaps.len() > 0);
    assert_eq!(outcome.remaps[0].persistent.as_str().len(), 71); // "object." + 64 hex = 7+64=71
}

// ── Deterministic remaps ────────────────────────────────────────────────

#[test]
fn duplicate_object_outcome_has_deterministic_remap() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let root_oid = scene.object_id(root).unwrap();

    let outcome = scene.duplicate_node(root, None).unwrap();
    let remap = &outcome.remaps[0];
    assert_eq!(remap.old, root_oid);
    assert_ne!(remap.new, root_oid);
    assert!(remap.persistent.as_str().starts_with("object."));
}

// ── Root constraints on duplicate ──────────────────────────────────────

#[test]
fn duplicate_node_to_specific_parent() {
    let mut scene = new_scene();
    let root = scene.create_node_default(None).unwrap();
    let target_parent = scene.create_node_default(None).unwrap();

    let outcome = scene.duplicate_node(root, Some(target_parent)).unwrap();
    let new_root_oid = outcome.created_roots[0];
    let parent = scene.get_object_parent(new_root_oid).unwrap();

    let target_oid = scene.object_id(target_parent).unwrap();
    assert_eq!(parent, ObjectParent::Node(target_oid));
}

// ── Light group parent in summary ──────────────────────────────────────

#[test]
fn summary_reflects_light_group_parent() {
    let mut scene = new_scene();
    let node = scene.create_node_default(None).unwrap();
    let pl = scene.create_point_light(PointLight {
        position: Vec3::ZERO,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    })
    .unwrap();
    let pl_oid = scene.object_id_for_point_light(pl).unwrap();
    let node_oid = scene.object_id(node).unwrap();

    scene.set_object_parent(pl_oid, ObjectParent::Node(node_oid)).unwrap();

    let summary = scene.object_summary(pl_oid).unwrap();
    assert!(summary.group_parent.is_some());
}
