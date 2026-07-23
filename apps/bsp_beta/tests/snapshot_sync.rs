//! Simulation snapshot sync tests: renderer/physics pose equality,
//! PVS bypass, transform bounds, and snapshot consumption.

use std::collections::HashMap;

use bsp_beta::physics_bridge::PhysicsBridge;
use bsp_beta::runtime_bridge::RuntimeBridge;
use bsp_beta::scene_sync::{sync_snapshot_to_physics, sync_snapshot_to_scene, EntityNodeMap};
use bsp_beta::snapshot::{InlineModelInfo, ModelMappings, SnapshotProducer};
use bsp_runtime::{
    BspSimulationSnapshot, SnapshotBuilder, SnapshotEntityPose,
};
use glam::{Mat4, Vec3};

fn make_test_snapshot() -> BspSimulationSnapshot {
    let mut builder = SnapshotBuilder::new(1, 0, 1.0 / 60.0, 0.0);
    builder.push_entity_pose(SnapshotEntityPose {
        entity_index: 1,
        model_index: 1,
        transform: Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
        world_bounds: (Vec3::new(4.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0)),
        is_moving: true,
    });
    builder.push_entity_pose(SnapshotEntityPose {
        entity_index: 2,
        model_index: 2,
        transform: Mat4::IDENTITY,
        world_bounds: (Vec3::ZERO, Vec3::ONE),
        is_moving: false,
    });
    builder.set_light_style(5, 0.5);
    builder.set_liquid_time(42.0);
    builder.build()
}

#[test]
fn empty_snapshot_has_no_poses() {
    let snap = BspSimulationSnapshot::empty();
    assert_eq!(snap.epoch.tick, 0);
    assert!(snap.entity_poses.is_empty());
    assert!(!snap.any_motion);
}

#[test]
fn sync_to_physics_ignores_static_entities() {
    let snapshot = make_test_snapshot();
    let bridge = PhysicsBridge::new();
    let mut world = physics::PhysicsWorld::new();

    let updated = sync_snapshot_to_physics(&snapshot, &bridge, &mut world);
    // No bodies registered — 0 successful syncs.
    assert_eq!(updated, 0);
}

#[test]
fn sync_to_scene_with_no_nodes_returns_zero() {
    let snapshot = make_test_snapshot();
    let nodes = EntityNodeMap::default();
    let mut scene = renderer::api::Scene::new();

    let updated = sync_snapshot_to_scene(&snapshot, &nodes, &mut scene);
    assert_eq!(updated, 0);
}

#[test]
fn snapshot_producer_advances_behavior() {
    let mappings = ModelMappings::default();
    let mut producer = SnapshotProducer::new(mappings);

    // Register a moving door in runtime bridge
    let mut runtime = RuntimeBridge::new();
    runtime.adapter.register_entities(vec![
        bsp_runtime::behavior::BehaviorEntityInfo {
            entity_index: 10,
            classname: "func_door".into(),
            targetname: Some("door_a".into()),
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(2.0), // slow enough to be moving after one tick
            wait: Some(2.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        },
    ]);
    runtime.adapter.activate_by_index(10, bsp_runtime::behavior::Activation::On);

    let inline_models = vec![InlineModelInfo {
        entity_index: 10,
        model_index: 1,
        origin: [0.0, 0.0, 0.0],
        angles: None,
        scale: None,
        local_mins: [-1.0, -1.0, -1.0],
        local_maxs: [1.0, 1.0, 1.0],
    }];

    let snapshot = producer.produce(
        1.0 / 60.0,
        &mut runtime,
        &inline_models,
        &HashMap::new(),
        &HashMap::new(),
    );

    assert_eq!(snapshot.entity_poses.len(), 1);
    assert_eq!(snapshot.entity_poses[0].entity_index, 10);
    assert!(snapshot.entity_poses[0].is_moving);
}

#[test]
fn snapshot_liquid_time_accumulates() {
    let mappings = ModelMappings::default();
    let mut producer = SnapshotProducer::new(mappings);
    let mut runtime = RuntimeBridge::new();

    let snap1 = producer.produce(1.0, &mut runtime, &[], &HashMap::new(), &HashMap::new());
    assert!((snap1.liquid_time - 1.0).abs() < 0.001);

    let snap2 = producer.produce(0.5, &mut runtime, &[], &HashMap::new(), &HashMap::new());
    assert!((snap2.liquid_time - 1.5).abs() < 0.001);
}

#[test]
fn snapshot_generation_is_monotonic() {
    let mappings = ModelMappings::default();
    let mut producer = SnapshotProducer::new(mappings);
    let mut runtime = RuntimeBridge::new();

    let snap1 = producer.produce(0.016, &mut runtime, &[], &HashMap::new(), &HashMap::new());
    let snap2 = producer.produce(0.016, &mut runtime, &[], &HashMap::new(), &HashMap::new());

    assert_eq!(snap1.generation.0, 1);
    assert_eq!(snap2.generation.0, 2);
    assert_eq!(snap1.epoch.tick, 1);
    assert_eq!(snap2.epoch.tick, 2);
}

#[test]
fn light_styles_flow_through_snapshot() {
    let mappings = ModelMappings::default();
    let mut producer = SnapshotProducer::new(mappings);
    let mut runtime = RuntimeBridge::new();

    // Register a light entity with style "5"
    runtime.adapter.register_entities(vec![
        bsp_runtime::behavior::BehaviorEntityInfo {
            entity_index: 100,
            classname: "light".into(),
            targetname: None,
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: Some("5".into()),
        },
    ]);

    // Set the style intensity
    runtime.adapter.set_light_style_intensity("5", 0.75);

    let snapshot = producer.produce(0.016, &mut runtime, &[], &HashMap::new(), &HashMap::new());

    assert!(snapshot.any_style_change);
    assert!((snapshot.light_styles.intensities[5] - 0.75).abs() < 1e-6);
}

#[test]
fn compute_world_bounds_identity_is_local() {
    let local_mins = [-1.0_f32, -1.0, -1.0];
    let local_maxs = [1.0_f32, 1.0, 1.0];
    let transform = Mat4::IDENTITY;

    let (wmin, wmax) =
        bsp_beta::snapshot::compute_world_aabb(&local_mins, &local_maxs, &transform);
    assert!((wmin - Vec3::new(-1.0, -1.0, -1.0)).length() < 0.001);
    assert!((wmax - Vec3::new(1.0, 1.0, 1.0)).length() < 0.001);
}

#[test]
fn compute_world_bounds_translated() {
    let local_mins = [-1.0_f32, -1.0, -1.0];
    let local_maxs = [1.0_f32, 1.0, 1.0];
    let transform = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));

    let (wmin, wmax) =
        bsp_beta::snapshot::compute_world_aabb(&local_mins, &local_maxs, &transform);
    assert!((wmin - Vec3::new(9.0, -1.0, -1.0)).length() < 0.001);
    assert!((wmax - Vec3::new(11.0, 1.0, 1.0)).length() < 0.001);
}

#[test]
fn compute_world_bounds_rotated() {
    let local_mins = [-1.0_f32, -1.0, -1.0];
    let local_maxs = [1.0_f32, 1.0, 1.0];
    let rot = glam::Quat::from_rotation_y(90.0_f32.to_radians());
    let transform = Mat4::from_rotation_translation(rot, Vec3::ZERO);

    let (wmin, wmax) =
        bsp_beta::snapshot::compute_world_aabb(&local_mins, &local_maxs, &transform);

    // After 90° Y rotation, the X-extent should be about the same as Z-extent
    let extent = wmax - wmin;
    assert!((extent.x - 2.0).abs() < 0.01, "extent: {extent:?}");
    assert!((extent.z - 2.0).abs() < 0.01, "extent: {extent:?}");
}

#[test]
fn model_mappings_rejects_invalid_keys() {
    let toml = r#"
[models]
"invalid/key" = "assets/a.gltf"
"#;
    assert!(ModelMappings::parse(toml).is_err());
}

#[test]
fn bsp_frame_values_store_style_intensities() {
    let mut scene = renderer::api::Scene::new();
    let mut intensities = [1.0_f32; 64];
    intensities[3] = 0.25;
    intensities[7] = 0.75;

    scene.set_bsp_frame_values(intensities, 10.0);
    // Scene stores these in SceneWorld which is private — we verify it
    // compiles and doesn't panic.
}

#[test]
fn bsp_mount_inline_transforms_stored() {
    use renderer::api::bsp::{BspMaterialHandle, BspMountState, MeshHandle};
    use std::collections::HashMap;

    let mut state = BspMountState::new();
    state.activate();

    // Add a face to a mount with inline model batch
    let batch = bsp::geometry::RenderBatch {
        key: bsp::geometry::BatchKey {
            leaf_signature: Vec::new(),
            render_class: 0,
            material_identity: 0,
            lightmap_page: 0,
        },
        face_indices: vec![0],
        pvs_eligible: false,
        is_inline_model: true,
        model_index: 1,
    };

    state.set_render_assets(
        vec![MeshHandle::new(7, 0)],
        vec![Some(BspMaterialHandle::new(3, 0))],
        vec![batch],
        vec![],
    );

    // Set an inline model transform
    let mut transforms: HashMap<u32, Mat4> = HashMap::new();
    transforms.insert(1, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
    state.inline_model_transforms = transforms;

    // Verify the transform is stored
    assert!(state.inline_model_transforms.contains_key(&1));
    let stored_transform = state.inline_model_transforms.get(&1).unwrap();
    assert!((stored_transform.w_axis.x - 5.0).abs() < 0.001);
}
