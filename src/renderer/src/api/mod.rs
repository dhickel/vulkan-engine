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
pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
pub use crate::debug_ui::{
    DebugUiFrameContext, DebugViewCallback, DebugViewDescriptor, DebugViewId,
};
pub use crate::scene::SceneNodeId;
pub use assets::{
    AssetManager, EnvironmentSource, EnvironmentState, FacePattern, PbrMaterialDesc,
    ProceduralMeshData, ProceduralVertex,
};
pub use config::{AssetManifestMode, AssetPolicyConfig, DebugRuntimeMode, RendererConfig};
pub use errors::{
    AssetError, HookError, RendererError, RendererFrameError, RendererInitError, SceneError,
};
pub use hooks::{RenderHook, RenderHookContext};
pub use input::{
    priority_bands, ActionBinding, ActionId, ActionMap, ActionMapLayer, BindingModifiers,
    BindingTrigger, FrameInputSnapshot, InputChord, InputConsume, InputDebugFrame,
    InputDebugSnapshot, InputEvent, InputLayer, InputRuntime, InputSnapshot, InputSystem,
    LayerDescriptor, LayerHandle, LayerId, LayerPriority, LayerSpec,
};
pub use loading::{LoadStatus, LoadTicket};
pub use renderer::{EnvironmentRuntimeStatus, FrameContext, FrameRenderOutcome, Renderer};
pub use scene::{
    PointLight, PointLightId, Scene, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId,
};
