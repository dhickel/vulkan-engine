//! Tests for Phase 07 — Selection semantics and EditorCamera.
//!
//! Covers: Selection add/remove/toggle/clear/primary/contains via
//! Scene-provided ObjectIds, remap, stale cleanup, provenance binding,
//! editor camera perspective/orthographic, focus_on, screen_to_ray.
//!
//! Note: ObjectId construction is internal to the renderer crate.
//! Integration tests get ObjectIds from Scene operations.

use glam::{Mat4, Vec3};
use renderer::{
    object::{ObjectKind, selection::Selection},
    Aabb, EditorCamera, EditorProjection, MeshBoundsEntry, PointLight, Scene, SceneBounds,
};

fn cube_aabb() -> Aabb {
    Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5))
}

#[test]
fn selection_workflow_with_scene_ids() {
    let mut scene = Scene::new();

    let node_a = scene.create_node_default(None).expect("node a");
    scene
        .set_transform(node_a, Mat4::from_translation(Vec3::new(0.0, 0.0, -3.0)))
        .expect("set transform");
    scene
        .add_mesh_with_bounds(node_a, renderer::MeshHandle::new(1, 0), SceneBounds::Known(cube_aabb()))
        .expect("mesh");

    let node_b = scene.create_node_default(None).expect("node b");
    scene
        .set_transform(node_b, Mat4::from_translation(Vec3::new(5.0, 0.0, -3.0)))
        .expect("set transform");
    scene
        .add_mesh_with_bounds(node_b, renderer::MeshHandle::new(2, 0), SceneBounds::Known(cube_aabb()))
        .expect("mesh");

    let id_a = scene.object_id(node_a).unwrap();
    let id_b = scene.object_id(node_b).unwrap();

    // create and add
    let mut sel = Selection::new();
    assert!(sel.is_empty());
    sel.add(id_a).unwrap();
    sel.add(id_b).unwrap();
    assert_eq!(sel.len(), 2);
    assert!(sel.contains(&id_a));
    assert!(sel.contains(&id_b));

    // primary
    assert_eq!(sel.primary(), Some(&id_a));

    // dedup
    let ch = sel.add(id_a).unwrap();
    assert!(!ch.changed());
    assert_eq!(sel.len(), 2);

    // toggle remove
    assert!(!sel.toggle(id_a).unwrap());
    assert_eq!(sel.len(), 1);
    assert!(sel.contains(&id_b));

    // toggle re-add
    assert!(sel.toggle(id_a).unwrap());
    assert_eq!(sel.len(), 2);

    // remove
    let ch = sel.remove(&id_b).unwrap();
    assert!(ch.changed());
    assert_eq!(sel.len(), 1);

    // set
    let ch = sel.set(id_b).unwrap();
    assert!(ch.changed());
    assert_eq!(sel.len(), 1);
    assert!(sel.contains(&id_b));
    assert!(!sel.contains(&id_a));

    // clear
    sel.clear();
    assert!(sel.is_empty());

    // from vec
    let sel2 = Selection::from(vec![id_a, id_b, id_a]);
    assert_eq!(sel2.len(), 2);

    // as_slice
    assert_eq!(sel2.as_slice()[0], id_a);

    // into_iter
    let collected: Vec<_> = sel2.into_iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn selection_remap_with_scene_ids() {
    let mut scene = Scene::new();
    let node_a = scene.create_node_default(None).expect("node a");
    let node_b = scene.create_node_default(None).expect("node b");
    let id_a = scene.object_id(node_a).unwrap();
    let id_b = scene.object_id(node_b).unwrap();

    let mut sel = Selection::new();
    sel.add(id_a).unwrap();
    sel.add(id_b).unwrap();

    // Remap: drop id_b, keep id_a
    sel.remap(|id| if *id == id_b { None } else { Some(id_a) });
    assert_eq!(sel.len(), 1);
    assert!(sel.contains(&id_a));
}

#[test]
fn selection_cleanup_stale_with_scene_ids() {
    let mut scene = Scene::new();
    let node_a = scene.create_node_default(None).expect("node a");
    let node_b = scene.create_node_default(None).expect("node b");
    let id_a = scene.object_id(node_a).unwrap();
    let id_b = scene.object_id(node_b).unwrap();

    let mut sel = Selection::new();
    sel.add(id_a).unwrap();
    sel.add(id_b).unwrap();

    // Cleanup: keep only node a's ID
    sel.cleanup_stale(|id| *id == id_a);
    assert_eq!(sel.len(), 1);
    assert!(sel.contains(&id_a));
    assert!(!sel.contains(&id_b));
}

#[test]
fn selection_provenance_from_scene() {
    let scene = Scene::new();
    let provenance = scene.provenance_token();

    let mut sel = Selection::with_provenance(provenance);
    assert_eq!(sel.provenance(), Some(provenance));

    // Verify the provenance is stable
    let provenance2 = scene.provenance_token();
    assert_eq!(provenance, provenance2);
}

#[test]
fn selection_replace_all_with_scene_ids() {
    let mut scene = Scene::new();
    let node_a = scene.create_node_default(None).expect("node a");
    let node_b = scene.create_node_default(None).expect("node b");
    let id_a = scene.object_id(node_a).unwrap();
    let id_b = scene.object_id(node_b).unwrap();

    let mut sel = Selection::new();
    sel.add(id_a).unwrap();
    sel.replace_all(vec![id_b, id_a]);
    assert_eq!(sel.len(), 2);
}

#[test]
fn selection_kind_variety() {
    let mut scene = Scene::new();

    let node = scene.create_node_default(None).expect("node");
    let pl = scene
        .create_point_light(PointLight {
            position: Vec3::new(0.0, 0.0, -3.0),
            color: Vec3::ONE,
            intensity: 10.0,
            range: 5.0,
        })
        .expect("point light");

    let node_id = scene.object_id(node).unwrap();
    let pl_id = scene.object_id_for_point_light(pl).unwrap();

    let mut sel = Selection::new();
    sel.add(node_id).unwrap();
    sel.add(pl_id).unwrap();
    assert_eq!(sel.len(), 2);
    assert_eq!(node_id.kind(), ObjectKind::Node);
    assert_eq!(pl_id.kind(), ObjectKind::PointLight);
}

// ── EditorCamera tests ─────────────────────────────────────────────────

#[test]
fn editor_camera_focus_on_many_works() {
    let mut cam = EditorCamera::default();
    let aabbs = [
        Aabb::from_min_max(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        Aabb::from_min_max(Vec3::new(3.0, 0.0, 0.0), Vec3::new(5.0, 1.0, 1.0)),
    ];
    assert!(cam.focus_on_many(&aabbs).is_ok());
    // Center should be between (-1, 1) and (3, 5) -> (1, 3) roughly
    let target = cam.orbit().target;
    assert!(target.x > -1.5 && target.x < 6.0);
}

#[test]
fn editor_camera_to_camera_view_is_finite() {
    let cam = EditorCamera::default();
    let cv = cam.to_camera_view(800, 600);
    assert!(cv.view.is_finite());
    assert!(cv.projection.is_finite());
    assert!(cv.position.is_finite());
}

#[test]
fn editor_camera_screen_to_ray_orthographic() {
    let mut cam = EditorCamera::default();
    cam.set_orthographic(5.0, 0.1, 500.0).unwrap();
    let ray = cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("should produce ray");
    assert!(ray.origin.is_finite());
    assert!(ray.direction.is_finite());
    // Orthographic rays should all be parallel (direction = camera forward).
    let forward = (cam.orbit().target - cam.eye_position()).normalize();
    assert!(
        ray.direction.dot(forward) > 0.99,
        "ortho ray should be parallel to view direction"
    );
}

#[test]
fn editor_camera_screen_to_ray_orthographic_corner_origin_differs() {
    let mut cam = EditorCamera::default();
    cam.set_orthographic(5.0, 0.1, 500.0).unwrap();
    let ray_center = cam.screen_to_ray((400.0, 300.0), (800, 600)).unwrap();
    let ray_corner = cam.screen_to_ray((0.0, 0.0), (800, 600)).unwrap();
    // Ortho rays share the same direction...
    assert!((ray_center.direction - ray_corner.direction).length() < 0.001);
    // ...but different origin points on the near plane.
    assert!((ray_center.origin - ray_corner.origin).length() > 0.1);
}

#[test]
fn editor_camera_default_is_perspective() {
    let cam = EditorCamera::default();
    assert!(matches!(
        cam.projection_mode,
        EditorProjection::Perspective { .. }
    ));
}

#[test]
fn editor_camera_perspective_rejects_invalid_fov() {
    let mut cam = EditorCamera::default();
    assert!(cam.set_perspective(0.0, 0.1, 100.0).is_err());
    assert!(cam
        .set_perspective(std::f32::consts::PI, 0.1, 100.0)
        .is_err());
}

#[test]
fn editor_camera_perspective_rejects_invalid_planes() {
    let mut cam = EditorCamera::default();
    assert!(cam.set_perspective(1.0, f32::NAN, 100.0).is_err());
    assert!(cam.set_perspective(1.0, 0.1, f32::INFINITY).is_err());
    assert!(cam.set_perspective(1.0, 5.0, 3.0).is_err());
    assert!(cam.set_perspective(1.0, -0.1, 100.0).is_err());
}

#[test]
fn editor_camera_valid_perspective_succeeds() {
    let mut cam = EditorCamera::default();
    assert!(cam.set_perspective(1.2, 0.5, 500.0).is_ok());
    assert!(matches!(
        cam.projection_mode,
        EditorProjection::Perspective { fov_y } if (fov_y - 1.2).abs() < 1e-6
    ));
}

#[test]
fn editor_camera_orthographic_rejects_invalid() {
    let mut cam = EditorCamera::default();
    assert!(cam.set_orthographic(0.0, 0.1, 100.0).is_err());
    assert!(cam.set_orthographic(-1.0, 0.1, 100.0).is_err());
    assert!(cam.set_orthographic(1.0, f32::NAN, 100.0).is_err());
    assert!(cam.set_orthographic(1.0, 5.0, 3.0).is_err());
}

#[test]
fn editor_camera_valid_orthographic_succeeds() {
    let mut cam = EditorCamera::default();
    assert!(cam.set_orthographic(50.0, 0.1, 2000.0).is_ok());
    assert!(matches!(
        cam.projection_mode,
        EditorProjection::Orthographic { half_height } if (half_height - 50.0).abs() < 1e-6
    ));
}

#[test]
fn editor_camera_view_projection_matrices_are_finite() {
    let cam = EditorCamera::default();
    let view = cam.view_matrix();
    let proj = cam.projection_matrix();
    let vp = cam.view_projection_matrix();
    assert!(view.is_finite());
    assert!(proj.is_finite());
    assert!(vp.is_finite());
}

#[test]
fn editor_camera_screen_to_ray_returns_valid_ray() {
    let cam = EditorCamera::default();
    let ray = cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("should produce ray");
    assert!(ray.origin.is_finite());
    assert!(ray.direction.is_finite());
    assert!(ray.direction.length_squared() > 0.0);
}

#[test]
fn editor_camera_screen_to_ray_center_looks_at_target() {
    let mut cam = EditorCamera::default();
    cam.orbit_mut().target = Vec3::ZERO;
    cam.orbit_mut().radius = 5.0;
    cam.orbit_mut().theta = 0.0;
    cam.orbit_mut().phi = std::f32::consts::FRAC_PI_4;

    let ray = cam
        .screen_to_ray((400.0, 300.0), (800, 600))
        .expect("ray");

    let to_target = (Vec3::ZERO - ray.origin).normalize();
    let dot = ray.direction.dot(to_target);
    assert!(dot > 0.9, "ray should point toward target, dot={dot}");
}

#[test]
fn editor_camera_focus_on_valid_aabb() {
    let mut cam = EditorCamera::default();
    let aabb = Aabb::from_min_max(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
    assert!(cam.focus_on(&aabb).is_ok());
    let target = cam.orbit().target;
    assert!((target - aabb.center()).length() < 0.001);
}

#[test]
fn editor_camera_focus_on_rejects_nonfinite_aabb() {
    let mut cam = EditorCamera::default();
    let bad = Aabb::from_min_max(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ONE);
    assert!(cam.focus_on(&bad).is_err());
}

#[test]
fn editor_camera_focus_on_rejects_degenerate_aabb() {
    let mut cam = EditorCamera::default();
    let bad = Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO);
    assert!(cam.focus_on(&bad).is_err());
}

#[test]
fn editor_camera_orbit_access() {
    let mut cam = EditorCamera::default();
    assert!(cam.orbit().radius > 0.0);
    cam.orbit_mut().radius = 10.0;
    assert!((cam.orbit().radius - 10.0).abs() < 1e-6);
}

#[test]
fn editor_camera_eye_position_changes_with_radius() {
    let mut cam = EditorCamera::default();
    let pos_before = cam.eye_position();
    cam.orbit_mut().radius *= 2.0;
    let pos_after = cam.eye_position();
    assert!((pos_after - pos_before).length() > 0.1);
}

#[test]
fn editor_camera_focus_on_clamps_radius() {
    let mut cam = EditorCamera::default();
    cam.orbit_mut().min_radius = 1.0;
    cam.orbit_mut().max_radius = 100.0;

    let small = Aabb::from_min_max(Vec3::splat(-0.001), Vec3::splat(0.001));
    assert!(cam.focus_on(&small).is_ok());
    assert!(cam.orbit().radius >= 1.0);
}

#[test]
fn editor_camera_inv_view_projection_is_usable() {
    let cam = EditorCamera::default();
    let inv_vp = cam.inv_view_projection();
    assert!(inv_vp.is_finite());
}
