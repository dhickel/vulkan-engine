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
    parse_package_manifest, AssetKind, AssetRegistry, AssetRegistryError, DurableAssetRecord,
    PackageAssetRecord, PackageManifest, Project, ProjectPackage, ProjectSettings,
};
pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
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
    AssetManifestMode, AssetPolicyConfig, DebugRuntimeMode, RendererConfig, VisualTuning,
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
    PointLight, PointLightId, Scene, SceneAssetReference, SceneFragment, SceneFragmentMount,
    SceneFragmentNode, SceneFragmentNodeId, SceneNodeSummary,
};
