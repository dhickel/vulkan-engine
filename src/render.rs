//! Renderer facade primitives re-exported from the `renderer` support crate.

pub use renderer::api::{
    default_capture_root, default_capture_run_dir, default_manual_capture_dir,
    default_single_capture_path, parse_package_manifest, single_capture_path,
    validate_package_manifest_file, validate_package_manifest_str, validate_project_file,
    validate_project_str, validate_scene_file, validate_scene_file_with_options,
    validate_scene_str, validate_scene_str_with_options, AssetManifestMode, AssetPolicyConfig,
    AssetRegistry, AssetRegistryError, CameraView, CaptureTarget, DebugTimingRow,
    DebugTimingSnapshot, DebugUiFrameContext, DueFrameCapture, DurableAssetRecord,
    EnvironmentHandle, EnvironmentRuntimeStatus, EnvironmentSource, EnvironmentState, FacePattern,
    FilterMode, FrameCaptureConfigError, FrameCaptureRequest, FrameCaptureScheduler,
    FrameCaptureSequence, FrameCaptureSource, FrameCaptureStatus, HookError, LoadStatus,
    LoadTicket, PackageAssetRecord, PackageManifest, PackageValidationOptions, PbrMaterialDesc,
    ProceduralMeshData, ProceduralVertex, Project, ProjectPackage, ProjectSettings,
    ProjectValidationOptions, RendererFrameError, RendererInitError, ResolvedTexturePolicy,
    SamplerOverride, SceneAssetReference, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId, SceneNodeSummary, SceneValidationOptions, TextureLoadOptions,
    ValidationArea, ValidationDiagnostic, ValidationError, WrapMode,
};
pub use renderer::{
    boxed_render_hook, install_app_event_logger, AnimationPlayer, AssetError, AssetKind,
    AssetManager, BoxedRenderHook, DebugRuntimeMode, EventBus, FrameContext, FrameRenderOutcome,
    MaterialHandle, MeshHandle, PointLight, PointLightId, RenderHook, RenderHookContext, Renderer,
    RendererConfig, RendererError, RendererInputRouting, RendererInputSuppression, Scene,
    SceneError, SceneNodeId, SceneWorld, TextureHandle, VisualTuning,
};
