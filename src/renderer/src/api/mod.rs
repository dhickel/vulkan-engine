mod assets;
mod config;
mod errors;
mod hooks;
mod loading;
mod renderer;
mod scene;

#[cfg(feature = "advanced-interop")]
pub mod advanced;

pub use crate::data::handles::{EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle};
pub use crate::scene::SceneNodeId;
pub use assets::{AssetManager, EnvironmentSource, EnvironmentState};
pub use config::{DebugRuntimeMode, RendererConfig};
pub use errors::{
    AssetError, HookError, RendererError, RendererFrameError, RendererInitError, SceneError,
};
pub use hooks::{RenderHook, RenderHookContext};
pub use loading::{LoadStatus, LoadTicket};
pub use renderer::{EnvironmentRuntimeStatus, FrameContext, FrameRenderOutcome, Renderer};
pub use scene::{Scene, SceneFragment, SceneFragmentMount, SceneFragmentNode, SceneFragmentNodeId};
