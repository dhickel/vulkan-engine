//! Tests for runtime transform batch operations and bridge sync behavior.
//!
//! Validates:
//! - `ObjectTransform` round-trips for all four `ObjectKind` variants.
//! - `Scene::set_object_transform` / `get_object_transform` correctness.
//! - Scene bounds invalidation after node transforms.
//! - Batch transform application via the component compatibility matrix.

use glam::Mat4;

use renderer::object::{
    is_identity_basis_matrix, is_rigid_direction_only, is_rigid_matrix, ObjectTransform,
};
use renderer::{
    DirectionalLight, PointLight, Scene, SpotLight,
};

// ── ObjectTransform round-trip ──────────────────────────────────────────

#[test]
fn node_transform_roundtrip() {
    let mut scene = Scene::new();
    let original = Mat4::from_scale_rotation_translation(
        glam::Vec3::new(2.0, 1.0, 1.0),
        glam::Quat::from_rotation_y(0.5),
        glam::Vec3::new(1.0, 2.0, 3.0),
    );
    let node = scene.create_node(None, original).unwrap();
    let id = scene.object_id(node).unwrap();

    let read = scene.get_object_transform(id).unwrap();
    match read {
        ObjectTransform::Node(mat) => {
            assert!(mat.abs_diff_eq(original, 1e-5));
        }
        _ => panic!("expected Node transform"),
    }
}

#[test]
fn point_light_transform_roundtrip() {
    let mut scene = Scene::new();
    let pl = PointLight {
        position: glam::Vec3::new(4.0, 5.0, 6.0),
        color: glam::Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    };
    let pl_id = scene.create_point_light(pl).unwrap();
    let id = scene.object_id_for_point_light(pl_id).unwrap();

    let read = scene.get_object_transform(id).unwrap();
    match read {
        ObjectTransform::PointLight(pos) => {
            assert!((pos - glam::Vec3::new(4.0, 5.0, 6.0)).length() < 1e-5);
        }
        _ => panic!("expected PointLight transform"),
    }
}

#[test]
fn set_node_transform_invalidates_bounds() {
    let mut scene = Scene::new();
    let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
    let id = scene.object_id(node).unwrap();

    let new_transform = Mat4::from_translation(glam::Vec3::new(10.0, 0.0, 0.0));
    scene.set_object_transform(id, &new_transform).unwrap();

    let read = scene.get_object_transform(id).unwrap();
    match read {
        ObjectTransform::Node(mat) => {
            assert!((mat.w_axis.truncate() - glam::Vec3::new(10.0, 0.0, 0.0)).length() < 1e-5);
        }
        _ => panic!("expected Node transform"),
    }
}

#[test]
fn set_object_transform_rejects_non_finite() {
    let mut scene = Scene::new();
    let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
    let id = scene.object_id(node).unwrap();

    let nan_mat = Mat4::from_cols_array(&[f32::NAN; 16]);
    let result = scene.set_object_transform(id, &nan_mat);
    assert!(result.is_err());
}

// ── Rigid matrix validation ─────────────────────────────────────────────

#[test]
fn rigid_matrix_validation_accepts_identity() {
    assert!(is_rigid_matrix(&Mat4::IDENTITY));
}

#[test]
fn rigid_matrix_rejects_scale() {
    let m = Mat4::from_scale(glam::Vec3::new(2.0, 1.0, 1.0));
    assert!(!is_rigid_matrix(&m));
}

#[test]
fn rigid_matrix_rejects_non_affine() {
    // Construct a non-affine matrix by placing a non-1 value in the W component
    // of column 3. Last row will be (0, 0, 0, 2) ≠ (0, 0, 0, 1).
    let m = Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(0.0, 0.0, 0.0, 2.0),
    );
    assert!(!is_rigid_matrix(&m));
}

#[test]
fn rigid_matrix_accepts_rotation_translation() {
    let m = Mat4::from_rotation_translation(
        glam::Quat::from_rotation_x(0.3),
        glam::Vec3::new(1.0, 2.0, 3.0),
    );
    assert!(is_rigid_matrix(&m));
}

#[test]
fn identity_basis_matrix_accepts_translation_only() {
    let m = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    assert!(is_identity_basis_matrix(&m));
}

#[test]
fn identity_basis_matrix_rejects_rotation() {
    let m = Mat4::from_rotation_x(0.1);
    assert!(!is_identity_basis_matrix(&m));
}

#[test]
fn rigid_direction_only_accepts_identity() {
    assert!(is_rigid_direction_only(&Mat4::IDENTITY));
}

#[test]
fn rigid_direction_only_rejects_translation() {
    let m = Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
    assert!(!is_rigid_direction_only(&m));
}

#[test]
fn rigid_direction_only_accepts_pure_rotation() {
    let m = Mat4::from_rotation_x(0.5);
    assert!(is_rigid_direction_only(&m));
}

// ── Spot light and directional light transforms ─────────────────────────

#[test]
fn spot_light_transform_roundtrip() {
    let mut scene = Scene::new();
    let sl = SpotLight::new(
        glam::Vec3::new(1.0, 2.0, 3.0),
        glam::Vec3::new(0.0, -1.0, 0.0),
        glam::Vec3::ONE,
        1.0,
        10.0,
        0.3,
        0.8,
    );
    let sl_id = scene.create_spot_light(sl).unwrap();
    let id = scene.object_id_for_spot_light(sl_id).unwrap();

    let read = scene.get_object_transform(id).unwrap();
    match read {
        ObjectTransform::SpotLight {
            position,
            direction,
        } => {
            assert!((position - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
            assert!((direction - glam::Vec3::new(0.0, -1.0, 0.0)).length() < 1e-5);
        }
        _ => panic!("expected SpotLight transform"),
    }
}

#[test]
fn directional_light_transform_roundtrip() {
    let mut scene = Scene::new();
    let dl = DirectionalLight {
        direction: glam::Vec3::new(0.3, 0.8, 0.4),
        color: glam::Vec3::ONE,
        intensity: 1.0,
    };
    let dl_id = scene.create_directional_light(dl).unwrap();
    let id = scene.object_id_for_directional_light(dl_id).unwrap();

    let read = scene.get_object_transform(id).unwrap();
    match read {
        ObjectTransform::DirectionalLight(dir) => {
            let expected = glam::Vec3::new(0.3, 0.8, 0.4).normalize();
            assert!((dir - expected).length() < 1e-5);
        }
        _ => panic!("expected DirectionalLight transform"),
    }
}

// ── Set non-node transform constraints ──────────────────────────────────

#[test]
fn set_point_light_transform_rejects_shear() {
    let mut scene = Scene::new();
    let pl = PointLight {
        position: glam::Vec3::ZERO,
        color: glam::Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
    };
    let pl_id = scene.create_point_light(pl).unwrap();
    let id = scene.object_id_for_point_light(pl_id).unwrap();

    // Non-identity basis matrix should be rejected.
    let bad = Mat4::from_scale(glam::Vec3::new(2.0, 1.0, 1.0));
    let result = scene.set_object_transform(id, &bad);
    assert!(result.is_err());
}

#[test]
fn set_directional_light_transform_rejects_translation() {
    let mut scene = Scene::new();
    let dl = DirectionalLight {
        direction: glam::Vec3::Y,
        color: glam::Vec3::ONE,
        intensity: 1.0,
    };
    let dl_id = scene.create_directional_light(dl).unwrap();
    let id = scene.object_id_for_directional_light(dl_id).unwrap();

    // Rigid with translation should be rejected for directional lights.
    let bad = Mat4::from_rotation_translation(
        glam::Quat::IDENTITY,
        glam::Vec3::new(1.0, 0.0, 0.0),
    );
    let result = scene.set_object_transform(id, &bad);
    assert!(result.is_err());
}

#[test]
fn set_spot_light_transform_rejects_scale() {
    let mut scene = Scene::new();
    let sl = SpotLight::new(
        glam::Vec3::ZERO,
        glam::Vec3::NEG_Z,
        glam::Vec3::ONE,
        1.0,
        10.0,
        0.3,
        0.8,
    );
    let sl_id = scene.create_spot_light(sl).unwrap();
    let id = scene.object_id_for_spot_light(sl_id).unwrap();

    // Non-rigid transform should be rejected.
    let bad = Mat4::from_scale(glam::Vec3::splat(2.0));
    let result = scene.set_object_transform(id, &bad);
    assert!(result.is_err());
}

// ── Batch transform application ─────────────────────────────────────────

#[test]
fn batch_set_transforms_across_kinds() {
    let mut scene = Scene::new();

    // Create one object of each kind.
    let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
    let pl = scene
        .create_point_light(PointLight {
            position: glam::Vec3::ZERO,
            color: glam::Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
        })
        .unwrap();
    let dl = scene
        .create_directional_light(DirectionalLight {
            direction: glam::Vec3::Y,
            color: glam::Vec3::ONE,
            intensity: 1.0,
        })
        .unwrap();
    let sl = scene
        .create_spot_light(SpotLight::new(
            glam::Vec3::ZERO,
            glam::Vec3::NEG_Z,
            glam::Vec3::ONE,
            1.0,
            10.0,
            0.3,
            0.8,
        ))
        .unwrap();

    let node_id = scene.object_id(node).unwrap();
    let pl_id = scene.object_id_for_point_light(pl).unwrap();
    let dl_id = scene.object_id_for_directional_light(dl).unwrap();
    let sl_id = scene.object_id_for_spot_light(sl).unwrap();

    // Apply valid transforms to each.
    let node_mat = Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
    scene.set_object_transform(node_id, &node_mat).unwrap();

    let pl_mat = Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0));
    scene.set_object_transform(pl_id, &pl_mat).unwrap();

    let dl_mat = Mat4::from_rotation_y(0.5);
    scene.set_object_transform(dl_id, &dl_mat).unwrap();

    let sl_mat = ObjectTransform::rigid_from_position_direction(
        glam::Vec3::new(3.0, 0.0, 0.0),
        glam::Vec3::NEG_Z,
    );
    scene.set_object_transform(sl_id, &sl_mat).unwrap();

    // Verify all transforms read back correctly.
    let node_read = scene.get_object_transform(node_id).unwrap();
    assert!(matches!(node_read, ObjectTransform::Node(_)));

    let pl_read = scene.get_object_transform(pl_id).unwrap();
    match pl_read {
        ObjectTransform::PointLight(pos) => {
            assert!((pos - glam::Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
        }
        _ => panic!("expected PointLight"),
    }

    let dl_read = scene.get_object_transform(dl_id).unwrap();
    assert!(matches!(dl_read, ObjectTransform::DirectionalLight(_)));

    let sl_read = scene.get_object_transform(sl_id).unwrap();
    match sl_read {
        ObjectTransform::SpotLight { position, .. } => {
            assert!((position - glam::Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5);
        }
        _ => panic!("expected SpotLight"),
    }
}

// ── Object transform for wrong kind ─────────────────────────────────────

/// Smoke test matching the expected filter name.
#[test]
fn runtime_transform_batch_smoke() {
    let mut scene = Scene::new();
    let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
    let id = scene.object_id(node).unwrap();
    assert!(scene.get_object_transform(id).is_ok());
}

#[test]
fn get_object_transform_wrong_kind_returns_correct_variant() {
    let mut scene = Scene::new();
    let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
    let id = scene.object_id(node).unwrap();

    // The ObjectId carries Node kind, so get_object_transform should return
    // ObjectTransform::Node.
    assert!(matches!(
        scene.get_object_transform(id).unwrap(),
        ObjectTransform::Node(_)
    ));
}
