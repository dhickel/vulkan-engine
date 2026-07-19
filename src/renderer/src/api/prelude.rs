//! Curated alpha facade imports for beginner renderer examples.
//!
//! This prelude is intentionally smaller than the crate root. It includes the
//! facade types used by quickstart-style renderer, scene, asset, input, event,
//! and debug/capture flows, while leaving compatibility helpers such as
//! `SceneWorld`, command history, camera/frustum helpers, animation internals,
//! and `advanced-interop` APIs out of the beginner contract.

pub use super::{
    default_capture_root, default_capture_run_dir, default_manual_capture_dir,
    default_single_capture_path, editor_ui_capture_layer, parse_package_manifest,
    single_capture_path, validate_package_manifest_file, validate_package_manifest_str,
    validate_project_file, validate_project_str, validate_scene_file,
    validate_scene_file_with_options, validate_scene_str, validate_scene_str_with_options,
    ActionBinding, ActionId, ActionMap, ActionMapLayer, ActionPhase, AssetError, AssetEvent,
    AssetId, AssetKind, AssetManager, AssetManifestMode, AssetPolicyConfig, AssetRegistry,
    AssetRegistryError, AudioClipId, AudioEvent, BindingModifiers, BindingTrigger,
    BoundsUnknownReason, CameraView,
    CaptureLayer, CaptureTarget, ColliderId, ContactPhase, DebugRuntimeMode, DebugTimingRow,
    DebugTimingSnapshot, DebugUiFrameContext, DirectionalLight, DirectionalLightId,
    DueFrameCapture, DurableAssetRecord, EngineEvent, EnvironmentHandle, EnvironmentRuntimeStatus,
    EnvironmentSource, EnvironmentState, EventBus, EventEnvelope, EventRecorder, EventSequence,
    EventStage, FacePattern, FilterMode, FrameCaptureConfigError, FrameCaptureRequest,
    FrameCaptureScheduler, FrameCaptureSequence, FrameCaptureSource, FrameCaptureStatus,
    FrameContext, FrameInputSnapshot, FrameRenderOutcome, HookError, InputActionEvent, InputChord,
    InputConsume, InputDebugFrame, InputDebugSnapshot, InputEvent, InputLayer, InputRuntime,
    InputSnapshot, InputSystem, LayerDescriptor, LayerHandle, LayerId, LayerPriority, LayerSpec,
    LifecycleEvent, ListenerError, ListenerFailure, ListenerId, LoadStatus, LoadTicket,
    MaterialHandle, MeshBoundsEntry, MeshDeformation, MeshGeometryDto, MeshHandle, MeshLocalAabb,
    PackageAssetRecord, PackageManifest, PackageValidationOptions,
    PbrMaterialDesc, PhysicsBodyId, PhysicsEvent, PointLight, PointLightId, SpotLight, SpotLightId, ProceduralMeshData,
    ProceduralVertex, Project, ProjectPackage, ProjectSettings, ProjectValidationOptions, Renderer,
    RendererConfig, RendererError, RendererFrameError, RendererInitError, RendererInputRouting,
    RendererInputSuppression, ResolvedTexturePolicy, SamplerOverride, Scene, SceneAssetReference,
    SceneBounds, SceneError, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId, SceneNodeId, SceneNodeSummary, SceneValidationOptions, ScriptingEvent,
    TextureHandle,
    TextureLoadOptions, ValidationArea, ValidationDiagnostic, ValidationError, VisualTuning,
    WrapMode,
};
