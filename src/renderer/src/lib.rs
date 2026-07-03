pub mod api;

pub mod animation;
mod data;
mod debug_ui;
#[cfg(not(feature = "advanced-interop"))]
mod rendergraph;
#[cfg(feature = "advanced-interop")]
pub mod rendergraph;
mod scene;
mod vulkan;

pub use api::{
    default_capture_root, default_capture_run_dir, default_manual_capture_dir,
    default_single_capture_path, editor_ui_capture_layer, parse_package_manifest,
    single_capture_path, validate_package_manifest_file, validate_package_manifest_str,
    validate_project_file, validate_project_str, validate_scene_file,
    validate_scene_file_with_options, validate_scene_str, validate_scene_str_with_options,
    AppUiCallback, AssetError, AssetKind, AssetManager, AssetManifestMode, AssetPolicyConfig,
    AssetRegistry, AssetRegistryError, CaptureLayer, CaptureTarget, DebugRuntimeMode,
    DebugTimingRow, DebugTimingSnapshot, DebugUiFrameContext, DebugViewCallback,
    DebugViewDescriptor, DebugViewId, DueFrameCapture, DurableAssetRecord, EnvironmentHandle,
    EnvironmentRuntimeStatus, EnvironmentSource, EnvironmentState, FacePattern, FilterMode,
    FrameCaptureConfigError, FrameCaptureRequest, FrameCaptureScheduler, FrameCaptureSequence,
    FrameCaptureSource, FrameCaptureStatus, FrameContext, FrameRenderOutcome, HookError,
    LoadStatus, LoadTicket, MaterialHandle, MeshHandle, PackageAssetRecord, PackageManifest,
    PackageValidationOptions, PbrMaterialDesc, PointLight, PointLightId, ProceduralMeshData,
    ProceduralVertex, Project, ProjectPackage, ProjectSettings, ProjectValidationOptions,
    RenderHook, RenderHookContext, Renderer, RendererConfig, RendererError, RendererFrameError,
    RendererInitError, ResolvedTexturePolicy, SamplerOverride, Scene, SceneAssetReference,
    SceneError, SceneFragment, SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId,
    SceneNodeId, SceneNodeSummary, SceneValidationOptions, TextureHandle, TextureLoadOptions,
    ValidationArea, ValidationDiagnostic, ValidationError, VisualTuning, WrapMode,
};

pub use animation::AnimationPlayer;
pub use data::camera::{Aabb, Camera, FPSController, Frustum, OrbitCamera, OrbitController, Ray};
pub use scene::command::{
    AddNodeCommand, Command, CommandHistory, CommandResult, PlaceAssetCommand, RemoveNodeCommand,
    SceneNodeRemap, SetTransformCommand,
};
pub use scene::scene_world::SceneWorld;
