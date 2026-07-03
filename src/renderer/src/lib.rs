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
    ActionPhase, AppUiCallback, AssetError, AssetEvent, AssetId, AssetKind, AssetManager,
    AssetManifestMode, AssetPolicyConfig, AssetRegistry, AssetRegistryError, AudioClipId,
    AudioEvent, CaptureLayer, CaptureTarget, ColliderId, ContactPhase, DebugRuntimeMode,
    DebugTimingRow, DebugTimingSnapshot, DebugUiFrameContext, DebugViewCallback,
    DebugViewDescriptor, DebugViewId, DueFrameCapture, DurableAssetRecord, EngineEvent,
    EnvironmentHandle, EnvironmentRuntimeStatus, EnvironmentSource, EnvironmentState, EventBus,
    EventEnvelope, EventRecorder, EventSequence, EventStage, FacePattern, FilterMode,
    FrameCaptureConfigError, FrameCaptureRequest, FrameCaptureScheduler, FrameCaptureSequence,
    FrameCaptureSource, FrameCaptureStatus, FrameContext, FrameId, FrameRenderOutcome, HookError,
    InputActionEvent, LifecycleEvent, ListenerError, ListenerFailure, ListenerId, LoadStatus,
    LoadTicket, MaterialHandle, MaterialId, MeshHandle, NodeId, PackageAssetRecord, PackageId,
    PackageManifest, PackageValidationOptions, PbrMaterialDesc, PhysicsBodyId, PhysicsEvent,
    PointLight, PointLightId, ProceduralMeshData, ProceduralVertex, Project, ProjectId,
    ProjectPackage, ProjectSettings, ProjectValidationOptions, RenderHook, RenderHookContext,
    Renderer, RendererConfig, RendererError, RendererFrameError, RendererInitError,
    ResolvedTexturePolicy, SamplerOverride, Scene, SceneAssetReference, SceneError, SceneFragment,
    SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId, SceneId, SceneNodeId,
    SceneNodeSummary, SceneValidationOptions, ScriptId, ScriptingEvent, TextureHandle,
    TextureLoadOptions, ValidationArea, ValidationDiagnostic, ValidationError, VisualTuning,
    WrapMode,
};

pub use animation::AnimationPlayer;
pub use data::camera::{Aabb, Camera, FPSController, Frustum, OrbitCamera, OrbitController, Ray};
pub use scene::command::{
    AddNodeCommand, Command, CommandHistory, CommandResult, PlaceAssetCommand, RemoveNodeCommand,
    SceneNodeRemap, SetTransformCommand,
};
pub use scene::scene_world::SceneWorld;
