//! Integration tests for the renderer crate.
//! These test the public API surface without requiring a GPU.

use glam::{Mat4, Vec3};
use renderer::prelude::{
    AssetRegistry, CaptureTarget, FrameCaptureConfigError, FrameCaptureRequest,
    FrameCaptureSequence,
};
use renderer::{
    Aabb, AssetError, AssetKind, CommandHistory, Frustum, MeshHandle, OrbitCamera, PointLight, Ray,
    RendererError, Scene, SceneError, SceneWorld, SetTransformCommand,
};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn beginner_prelude_import_contract_compiles() {
    use renderer::prelude::*;

    let config = RendererConfig::default();
    assert!(!config.app_name.is_empty());

    let mut scene = Scene::new();
    let light = PointLight {
        position: Vec3::new(0.0, 1.0, 0.0),
        color: Vec3::ONE,
        intensity: 1.0,
        range: 4.0,
    };
    scene.create_point_light(light).expect("create point light");

    let mut registry = AssetRegistry::new();
    registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#,
            "packages/core",
        )
        .expect("manifest parses through prelude import");

    let request = FrameCaptureRequest::new(CaptureTarget::Present, "frame.png");
    assert_eq!(request.target, CaptureTarget::Present);
    let sequence = FrameCaptureSequence::new(CaptureTarget::Draw, "captures", 0, 1, 1)
        .expect("capture sequence");
    assert_eq!(sequence.remaining, 1);
    let scheduler = FrameCaptureScheduler::new("prelude-contract");
    assert!(scheduler.last_status().is_none());

    assert!(matches!(LoadStatus::<()>::Cancelled, LoadStatus::Cancelled));
    assert!(matches!(
        FrameRenderOutcome::Rendered,
        FrameRenderOutcome::Rendered
    ));
    assert!(matches!(AssetKind::Model, AssetKind::Model));

    // ── Type existence (compile-time contract covering all prelude re-exports) ──
    let _: Option<AssetManager<'_>> = None;
    let _: Option<CameraView> = None;
    let _: Option<Renderer> = None;
    let _: Option<RendererError> = None;
    let _: Option<RendererInitError> = None;
    let _: Option<RendererFrameError> = None;
    let _: Option<RendererInputRouting> = None;
    let _: Option<RendererInputSuppression> = None;
    let _: Option<AssetRegistryError> = None;
    let _: Option<SceneError> = None;
    let _: Option<SceneFragment> = None;
    let _: Option<SceneFragmentMount> = None;
    let _: Option<SceneFragmentNode> = None;
    let _: Option<SceneFragmentNodeId> = None;
    let _: Option<SceneNodeId> = None;
    let _: Option<SceneNodeSummary> = None;
    let _: Option<AssetId> = None;
    let _: Option<ActionId> = None;
    let _: Option<PointLightId> = None;
    let _: Option<MaterialHandle> = None;
    let _: Option<MeshHandle> = None;
    let _: Option<TextureHandle> = None;
    let _: Option<EnvironmentHandle> = None;
    let _: Option<ActionMap> = None;
    let _: Option<ActionMapLayer> = None;
    let _: Option<ActionBinding> = None;
    let _: Option<BindingModifiers> = None;
    let _: Option<BindingTrigger> = None;
    let _: Option<InputSystem> = None;
    let _: Option<InputRuntime> = None;
    let _: Option<InputSnapshot> = None;
    let _: Option<InputEvent> = None;
    let _: Option<InputActionEvent> = None;
    let _: Option<InputDebugFrame> = None;
    let _: Option<InputDebugSnapshot> = None;
    let _: Option<InputChord> = None;
    let _: Option<FrameInputSnapshot> = None;
    let _: Option<LayerDescriptor> = None;
    let _: Option<LayerHandle> = None;
    let _: Option<LayerId> = None;
    let _: Option<LayerPriority> = None;
    let _: Option<LayerSpec> = None;
    let _: Option<Project> = None;
    let _: Option<ProjectPackage> = None;
    let _: Option<ProjectSettings> = None;
    let _: Option<ProjectValidationOptions> = None;
    let _: Option<PackageManifest> = None;
    let _: Option<PackageValidationOptions> = None;
    let _: Option<PackageAssetRecord> = None;
    let _: Option<DurableAssetRecord> = None;
    let _: Option<SceneAssetReference> = None;
    let _: Option<SceneValidationOptions> = None;
    let _: Option<EventBus> = None;
    let _: Option<EventEnvelope> = None;
    let _: Option<EventRecorder> = None;
    let _: Option<EventSequence> = None;
    let _: Option<EventStage> = None;
    let _: Option<EngineEvent> = None;
    let _: Option<DebugTimingSnapshot> = None;
    let _: Option<DebugTimingRow> = None;
    let _: Option<DueFrameCapture> = None;
    let _: Option<FrameCaptureConfigError> = None;
    let _: Option<FrameContext> = None;
    let _: Option<VisualTuning> = None;
    let _: Option<AssetManifestMode> = None;
    let _: Option<AssetPolicyConfig> = None;
    let _: Option<PbrMaterialDesc> = None;
    let _: Option<ProceduralMeshData> = None;
    let _: Option<ProceduralVertex> = None;
    let _: Option<LoadTicket> = None;
    let _: Option<SamplerOverride> = None;
    let _: Option<ResolvedTexturePolicy> = None;
    let _: Option<TextureLoadOptions> = None;
    let _: Option<ValidationArea> = None;
    let _: Option<ValidationDiagnostic> = None;
    let _: Option<ValidationError> = None;
    let _: Option<AssetEvent> = None;
    let _: Option<FrameCaptureScheduler> = None;
    let _: Option<FrameCaptureStatus> = None;
    let _: Option<FrameCaptureRequest> = None;
    let _: Option<FilterMode> = None;
    let _: Option<WrapMode> = None;
    let _: Option<FacePattern> = None;

    // ── Free functions (compile-time existence) ──
    let _ = default_capture_root();
}

#[test]
fn renderer_input_routing_result_is_public_and_explicit() {
    use renderer::prelude::*;

    let queued = RendererInputRouting::queue();
    assert!(queued.queue_input);
    assert_eq!(queued.suppression, RendererInputSuppression::None);

    let suppressed = RendererInputRouting::suppress(RendererInputSuppression::UiMouseCapture);
    assert!(!suppressed.queue_input);
    assert_eq!(
        suppressed.suppression,
        RendererInputSuppression::UiMouseCapture
    );
}

#[test]
fn beginner_error_display_contract_is_actionable() {
    let scene_err = RendererError::from(SceneError::MissingAssetId("node asset".to_string()));
    assert_eq!(
        scene_err.to_string(),
        "scene error: missing durable asset id for node asset"
    );

    let asset_err = RendererError::from(AssetError::ManifestParse {
        path: PathBuf::from("assets/core.package.toml"),
        message: "missing field `format_version`".to_string(),
    });
    assert!(asset_err
        .to_string()
        .contains("asset error: manifest parse error for 'assets/core.package.toml'"));
    assert!(asset_err.to_string().contains("format_version"));
}

#[test]
fn capture_target_and_sequence_config_contracts_are_strict() {
    assert_eq!(CaptureTarget::parse("draw"), Some(CaptureTarget::Draw));
    assert_eq!(
        CaptureTarget::parse("present"),
        Some(CaptureTarget::Present)
    );
    assert_eq!(CaptureTarget::parse("desktop screenshot"), None);

    let request = FrameCaptureRequest::new(CaptureTarget::Draw, "captures/frame.png");
    assert_eq!(request.target, CaptureTarget::Draw);
    assert_eq!(request.output_path, PathBuf::from("captures/frame.png"));

    assert_eq!(
        FrameCaptureSequence::new(CaptureTarget::Draw, "captures", 0, 0, 1)
            .expect_err("zero interval is invalid"),
        FrameCaptureConfigError::InvalidInterval
    );
    assert_eq!(
        FrameCaptureSequence::new(CaptureTarget::Draw, "captures", 0, 1, 0)
            .expect_err("zero count is invalid"),
        FrameCaptureConfigError::InvalidCount
    );
}

#[test]
fn scene_node_create_and_transform() {
    let mut scene = Scene::new();
    let node = scene
        .create_node(None, Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)))
        .expect("create node");
    let t = scene.transform(node).expect("get transform");
    assert_eq!(t.w_axis, Vec3::new(1.0, 2.0, 3.0).extend(1.0).into());
}

#[test]
fn scene_node_remove_and_stale_reference() {
    let mut scene = Scene::new();
    let node = scene.create_node_default(None).expect("create");
    scene.remove_node(node).expect("remove");
    assert!(scene.transform(node).is_err());
}

#[test]
fn point_light_lifecycle() {
    let mut scene = Scene::new();
    let light = PointLight {
        position: Vec3::new(0.0, 5.0, 0.0),
        color: Vec3::ONE,
        intensity: 10.0,
        range: 8.0,
    };
    let id = scene.create_point_light(light).expect("create light");
    scene
        .update_point_light(
            id,
            PointLight {
                intensity: 20.0,
                ..light
            },
        )
        .expect("update light");
    scene.remove_point_light(id).expect("remove");
    assert!(scene.update_point_light(id, light).is_err());
}

#[test]
fn ray_screen_to_world_round_trip() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(1.2, 16.0 / 9.0, 0.1, 100.0);
    let inv_vp = (proj * view).inverse();

    let ray = Ray::from_screen(
        (960.0, 540.0),
        (1920, 1080),
        inv_vp,
        Vec3::new(0.0, 0.0, 5.0),
    );

    // Center of screen should point roughly along -Z
    assert!(ray.direction.z < -0.9);
    assert!(ray.origin.x.abs() < 0.01);
}

#[test]
fn aabb_intersection_hit() {
    let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    let ray = Ray {
        origin: Vec3::new(0.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
    };
    assert!(aabb.intersect_ray(&ray).is_some());
}

#[test]
fn aabb_intersection_miss() {
    let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    let ray = Ray {
        origin: Vec3::new(10.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
    };
    assert!(aabb.intersect_ray(&ray).is_none());
}

#[test]
fn frustum_culling_outside() {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(1.2, 1.0, 0.1, 100.0);
    let vp = proj * view;
    let frustum = Frustum::from_view_projection(&vp);

    // AABB far behind the camera
    let behind = Aabb::from_min_max(Vec3::new(-1.0, -1.0, 50.0), Vec3::new(1.0, 1.0, 52.0));
    assert!(!frustum.intersects_aabb(&behind));

    // AABB in front of camera
    let front = Aabb::from_min_max(Vec3::new(-0.5, -0.5, -1.0), Vec3::new(0.5, 0.5, 1.0));
    assert!(frustum.intersects_aabb(&front));
}

#[test]
fn orbit_camera_look_at() {
    let mut cam = OrbitCamera::default();
    cam.target = Vec3::new(1.0, 2.0, 3.0);
    cam.radius = 10.0;
    cam.theta = 0.0;
    cam.phi = 0.5;

    let eye = cam.eye_position();
    let view = cam.view_matrix();

    // Eye should be radius away from target
    let dist = (eye - cam.target).length();
    assert!((dist - 10.0).abs() < 0.1);

    // View matrix should be valid (non-zero determinant)
    assert!(view.determinant().abs() > 0.001);
}

#[test]
fn command_history_undo_redo() {
    let mut world = SceneWorld::new();
    let mut history = CommandHistory::new(32);

    // Create a node
    let node = world.add_node_with_parts(None, Mat4::IDENTITY, vec![]);

    // Set transform via command
    let new_t = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
    history
        .execute(Box::new(SetTransformCommand::new(node, new_t)), &mut world)
        .expect("execute");

    assert_eq!(
        world.get_node(node).unwrap().local_transform.w_axis,
        Vec3::new(5.0, 0.0, 0.0).extend(1.0).into()
    );

    // Undo
    history.undo(&mut world).expect("undo");
    assert_eq!(
        world.get_node(node).unwrap().local_transform.w_axis,
        Vec3::ZERO.extend(1.0).into()
    );

    // Redo
    history.redo(&mut world).expect("redo");
    assert_eq!(
        world.get_node(node).unwrap().local_transform.w_axis,
        Vec3::new(5.0, 0.0, 0.0).extend(1.0).into()
    );
}

#[test]
fn animation_player_evaluates_at_time() {
    use renderer::animation::{
        AnimationChannel, AnimationClip, AnimationPlayer, AnimationSampler, AnimationTarget,
        Interpolation, KeyframeValue,
    };

    let mut clip = AnimationClip::new("test", 2.0);
    clip.samplers.push(AnimationSampler {
        input: vec![0.0, 1.0, 2.0],
        output: vec![
            KeyframeValue::Translation(Vec3::new(0.0, 0.0, 0.0)),
            KeyframeValue::Translation(Vec3::new(1.0, 0.0, 0.0)),
            KeyframeValue::Translation(Vec3::new(2.0, 0.0, 0.0)),
        ],
        interpolation: Interpolation::Linear,
    });
    clip.channels.push(AnimationChannel {
        node_index: 0,
        target_path: AnimationTarget::Translation,
        sampler_index: 0,
    });

    let mut player = AnimationPlayer::new();
    player.set_clip(clip);
    player.play();

    // At t=0, should be at origin
    let transforms = player.update(0.0);
    let mat = transforms.get(&0).expect("node 0");
    assert!((mat.w_axis.x - 0.0).abs() < 0.01);

    // At t=1, should be at x=1
    let transforms = player.update(1.0);
    let mat = transforms.get(&0).expect("node 0");
    assert!((mat.w_axis.x - 1.0).abs() < 0.01);

    // At t=2, should be at x=2
    let transforms = player.update(1.0);
    let mat = transforms.get(&0).expect("node 0");
    assert!((mat.w_axis.x - 2.0).abs() < 0.01);
}

#[test]
fn package_manifest_parse_registers_durable_records() {
    let package_dir = unique_temp_dir("parse-package");
    std::fs::create_dir_all(package_dir.join("models")).expect("create package dirs");
    let manifest_path = package_dir.join("core.package.toml");
    std::fs::write(
        &manifest_path,
        r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
display_name = "Crate"
tags = ["crate", "prop"]
"#,
    )
    .expect("write manifest");

    let mut registry = AssetRegistry::new();
    let records = registry
        .load_package_manifest(&manifest_path)
        .expect("load package manifest");

    assert_eq!(records.len(), 1);
    let record = registry
        .asset_record("core.model.crate")
        .expect("durable record");
    assert_eq!(record.kind, AssetKind::Model);
    assert_eq!(record.display_name, "Crate");
    assert_eq!(
        record.package_relative_path,
        PathBuf::from("models/crate.glb")
    );
    assert_eq!(record.source_path, package_dir.join("models/crate.glb"));
}

#[test]
fn package_manifest_rejects_duplicate_asset_ids() {
    let mut registry = AssetRegistry::new();
    let err = registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"

[[assets]]
id = "core.model.crate"
kind = "texture"
path = "textures/crate.png"
"#,
            "assets/core",
        )
        .expect_err("duplicate ids should fail");

    assert!(err.to_string().contains("duplicate durable asset id"));
}

#[test]
fn package_manifest_normalizes_relative_paths() {
    let mut registry = AssetRegistry::new();
    registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/../prefabs/./crate.glb"
"#,
            "assets/core",
        )
        .expect("load package manifest");

    let record = registry
        .resolve_asset("core.model.crate")
        .expect("resolve asset");
    assert_eq!(
        record.package_relative_path,
        PathBuf::from("prefabs/crate.glb")
    );
    assert_eq!(
        record.source_path,
        PathBuf::from("assets/core/prefabs/crate.glb")
    );
}

#[test]
fn package_manifest_rejects_escape_paths_and_unsupported_versions() {
    let mut registry = AssetRegistry::new();
    let path_err = registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "../outside.glb"
"#,
            "assets/core",
        )
        .expect_err("escaping paths should fail");
    assert!(path_err.to_string().contains("invalid path"));

    let version_err = registry
        .load_package_manifest_str(
            r#"
format_version = 2
package_id = "core"
display_name = "Core Assets"
"#,
            "assets/core",
        )
        .expect_err("unsupported versions should fail");
    assert!(version_err
        .to_string()
        .contains("unsupported package manifest version"));

    let kind_err = registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.unknown"
kind = "shader"
path = "shader.wgsl"
"#,
            "assets/core",
        )
        .expect_err("unsupported kinds should fail");
    assert!(kind_err.to_string().contains("unsupported asset kind"));
}

#[test]
fn durable_asset_id_lookup_does_not_require_runtime_handles() {
    let mut registry = AssetRegistry::new();
    registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.env.indoor"
kind = "environment"
path = "sky_maps/indoor.exr"
"#,
            "assets/core",
        )
        .expect("load package manifest");

    let record = registry
        .resolve_asset("core.env.indoor")
        .expect("resolve asset");
    assert_eq!(record.kind, AssetKind::Environment);
    assert_eq!(
        record.load_path(),
        Path::new("assets/core/sky_maps/indoor.exr")
    );
    assert!(registry.find_environment(record.load_path()).is_none());
}

#[test]
fn package_asset_listing_filters_wall_chunks_deterministically() {
    let mut registry = AssetRegistry::new();
    registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "sample"
display_name = "Sample Assets"

[[assets]]
id = "sample.model.block"
kind = "model"
path = "models/block.obj"
display_name = "Block"
tags = ["prop"]

[[assets]]
id = "sample.wall.stone_2m"
kind = "wall_chunk"
path = "prefabs/wall.obj"
display_name = "Stone Wall"
tags = ["wall", "chunk"]
"#,
            "apps/editor/sample_project/assets",
        )
        .expect("load package manifest");

    let assets = registry.list_assets_matching(Some(&AssetKind::WallChunk), Some("wall"));

    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0].asset_id, "sample.wall.stone_2m");
    assert_eq!(assets[0].kind, AssetKind::WallChunk);
}

#[test]
fn path_invalidation_removes_runtime_and_durable_entries() {
    let mut registry = AssetRegistry::new();
    registry
        .load_package_manifest_str(
            r#"
format_version = 1
package_id = "core"
display_name = "Core Assets"

[[assets]]
id = "core.model.crate"
kind = "model"
path = "models/crate.glb"
"#,
            "assets/core",
        )
        .expect("load package manifest");

    let path = PathBuf::from("assets/core/models/crate.glb");
    let handle = MeshHandle::new(42, 7);
    registry.register_mesh(path.clone(), handle);
    assert_eq!(registry.find_mesh(&path), Some(handle));
    assert!(registry.asset_record("core.model.crate").is_some());

    registry.invalidate_path(&path);

    assert!(registry.find_mesh(&path).is_none());
    assert!(registry.asset_record("core.model.crate").is_none());
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("renderer-{label}-{}-{nanos}", std::process::id()))
}
