mod assets;
pub mod config;
mod errors;
mod hooks;
mod loading;
mod renderer;
pub(crate) mod scene;

#[cfg(feature = "advanced-interop")]
pub mod advanced;

pub use crate::data::asset_manifest::{
    FilterMode, ResolvedTexturePolicy, SamplerOverride, TextureLoadOptions, WrapMode,
};
pub use crate::data::asset_registry::{
    parse_package_manifest, validate_package_manifest_file, validate_package_manifest_str,
    validate_project_file, validate_project_str, AssetKind, AssetRegistry, AssetRegistryError,
    DurableAssetRecord, PackageAssetRecord, PackageManifest, PackageValidationOptions, Project,
    ProjectPackage, ProjectSettings, ProjectValidationOptions,
};
pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
pub use crate::data::validation::{ValidationArea, ValidationDiagnostic, ValidationError};
pub use crate::debug_ui::{
    AppUiCallback, DebugTimingRow, DebugTimingSnapshot, DebugUiFrameContext, DebugViewCallback,
    DebugViewDescriptor, DebugViewId,
};
pub use crate::scene::SceneNodeId;
pub use assets::{
    AssetManager, EnvironmentSource, EnvironmentState, FacePattern, PbrMaterialDesc,
    ProceduralMeshData, ProceduralVertex,
};
pub use config::{
    default_capture_root, default_capture_run_dir, default_manual_capture_dir,
    default_single_capture_path, single_capture_path, AssetManifestMode, AssetPolicyConfig,
    CaptureTarget, DebugRuntimeMode, DueFrameCapture, FrameCaptureConfigError, FrameCaptureRequest,
    FrameCaptureScheduler, FrameCaptureSequence, FrameCaptureSource, FrameCaptureStatus,
    RendererConfig, VisualTuning,
};
pub use engine_events::{
    ActionPhase, AssetEvent, AssetId, AudioClipId, AudioEvent, ColliderId, ContactPhase,
    EngineEvent, EventBus, EventEnvelope, EventRecorder, EventSequence, EventStage, FrameId,
    InputActionEvent, LifecycleEvent, ListenerError, ListenerFailure, ListenerId, MaterialId,
    NodeId, PackageId, PhysicsBodyId, PhysicsEvent, ProjectId, SceneEvent, SceneId, ScriptId,
    ScriptingEvent,
};
pub use errors::{
    AssetError, HookError, RendererError, RendererFrameError, RendererInitError, SceneError,
};
pub use hooks::{RenderHook, RenderHookContext};
pub use input::{
    editor_ui_capture_layer, priority_bands, ActionBinding, ActionId, ActionMap, ActionMapLayer,
    BindingModifiers, BindingTrigger, CaptureLayer, FrameInputSnapshot, InputChord, InputConsume,
    InputDebugFrame, InputDebugSnapshot, InputEvent, InputLayer, InputRuntime, InputSnapshot,
    InputSystem, LayerDescriptor, LayerHandle, LayerId, LayerPriority, LayerSpec,
};
pub use loading::{LoadStatus, LoadTicket};
pub use renderer::{EnvironmentRuntimeStatus, FrameContext, FrameRenderOutcome, Renderer};
pub use scene::{
    validate_scene_file, validate_scene_file_with_options, validate_scene_str,
    validate_scene_str_with_options, PointLight, PointLightId, Scene, SceneAssetReference,
    SceneFragment, SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId, SceneNodeSummary,
    SceneValidationOptions,
};
