mod assets;
#[cfg(feature = "bsp")]
pub mod bsp;
pub mod config;
pub mod sprite;
mod errors;
pub mod event_logging;
mod frame_extensions;
mod hooks;
mod loading;
pub mod prelude;
mod renderer;
pub(crate) mod scene;
mod scene_file_tx;
mod utils;

#[cfg(feature = "advanced-interop")]
pub mod advanced;

pub use crate::data::asset_manifest::{
    FilterMode, ResolvedTexturePolicy, SamplerOverride, TextureLoadOptions, WrapMode,
};
pub use crate::data::asset_registry::{
    normalize_logical_key, parse_package_manifest, try_normalize_logical_key,
    validate_package_manifest_file, validate_package_manifest_str, validate_project_file,
    validate_project_str, AssetKind, AssetRegistry, AssetRegistryError, DurableAssetRecord,
    PackageAssetRecord, PackageManifest, PackageValidationOptions, Project, ProjectPackage,
    ProjectSettings, ProjectValidationOptions,
};
pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};

#[cfg(feature = "bsp")]
pub use crate::data::handles::{BspMaterialHandle, BspTextureHandle};
pub use crate::data::mesh_geometry::{MeshDeformation, MeshGeometryDto, MeshLocalAabb};
pub use crate::data::retirement::{
    FrameSerial, GpuRetirementQueue, RetirementClass, RetirementError, RetirementRecord,
};
pub use crate::data::validation::{ValidationArea, ValidationDiagnostic, ValidationError};
pub use crate::debug_ui::{
    AppUiCallback, DebugTimingRow, DebugTimingSnapshot, DebugUiFrameContext, DebugViewCallback,
    DebugViewDescriptor, DebugViewId,
};
pub use crate::object::query::{
    EditorPickResult, EditorProxyPolicy, ObjectQueryFilter, RayHit, UnknownBoundsPolicy,
    VolumeHit, VolumeQuery, VolumeShape,
};
pub use crate::object::selection::{Selection, SelectionChange};
pub use crate::object::ObjectId;
pub use crate::object::{ObjectLifecycleOutcome, ObjectMutationOutcome, ObjectSummary};
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
    NodeId, ObjectKind, PackageId, PhysicsBodyId, PhysicsEvent, ProjectId, SceneEvent, SceneId,
    SceneObjectId, SceneObjectLifecycleAction, SceneObjectLifecycleEvent,
    SceneObjectLifecycleSnapshot, ScriptId, ScriptingEvent,
};
pub use errors::{
    AnimationError, AssetError, CommandError, HookError, HookFailureEntry, HookFailureStage,
    HookReport, ObjectError, RendererError, RendererFrameError, RendererInitError, SceneError,
};
pub use hooks::{boxed_render_hook, BoxedRenderHook, RenderHook, RenderHookContext};
pub use input::{
    editor_ui_capture_layer, priority_bands, ActionBinding, ActionId, ActionMap, ActionMapLayer,
    BindingModifiers, BindingTrigger, CaptureLayer, FrameInputSnapshot, InputChord, InputConsume,
    InputDebugFrame, InputDebugSnapshot, InputEvent, InputLayer, InputRuntime, InputSnapshot,
    InputSystem, LayerDescriptor, LayerHandle, LayerId, LayerPriority, LayerSpec,
};
pub use loading::{LoadStatus, LoadTicket};
pub use frame_extensions::FrameExtensions;
pub use renderer::{
    CameraView, EnvironmentRuntimeStatus, FrameContext, FrameRenderOutcome, Renderer,
    RendererInputRouting, RendererInputSuppression, RetirementSerials,
};
pub use scene::{
    validate_scene_file, validate_scene_file_with_options, validate_scene_str,
    validate_scene_str_with_options, BoundsUnknownReason, DirectionalLight, DirectionalLightId,
    DirectionalShadowConfig, MeshBoundsEntry, PointLight, PointLightId, Scene, SceneAssetReference,
    SceneBounds, SceneFragment, SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId,
    SceneNodeSummary, SceneValidationOptions, SpotLight, SpotLightId,
};
pub use crate::scene::object_commands::{
    AttachComponentCommand, DuplicateObjectsCommand, RemoveComponentCommand,
    RemoveObjectsCommand, ReplaceComponentStateCommand, SetComponentPropertyCommand,
    SetObjectParentCommand, SetObjectTransformCommand,
};
