mod assets;
pub mod config;
mod errors;
mod hooks;
mod loading;
mod renderer;
pub(crate) mod scene;

#[cfg(feature = "advanced-interop")]
pub mod advanced;

pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
pub use crate::scene::SceneNodeId;
pub use input::{
    ActionBinding, ActionId, ActionMap, ActionMapLayer, InputChord, InputConsume, InputDebugSnapshot,
    InputEvent, InputLayer, InputSnapshot, InputSystem, LayerDescriptor, LayerHandle, LayerPriority,
};
pub use assets::{
    AssetManager, EnvironmentSource, EnvironmentState, FacePattern, PbrMaterialDesc,
    ProceduralMeshData, ProceduralVertex,
};
pub use crate::data::asset_manifest::{
    FilterMode, ResolvedTexturePolicy, SamplerOverride, TextureLoadOptions, WrapMode,
};
pub use config::{AssetManifestMode, AssetPolicyConfig, DebugRuntimeMode, RendererConfig};
pub use errors::{
    AssetError, HookError, RendererError, RendererFrameError, RendererInitError, SceneError,
};
pub use hooks::{RenderHook, RenderHookContext};
pub use loading::{LoadStatus, LoadTicket};
pub use renderer::{EnvironmentRuntimeStatus, FrameContext, FrameRenderOutcome, Renderer};
pub use scene::{
    PointLight, PointLightId, Scene, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId,
};
