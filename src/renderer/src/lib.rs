pub mod api;

/// Curated alpha facade imports for quickstart-style renderer applications.
///
/// This module is the supported beginner import path. The crate root preserves
/// additional compatibility exports, but they are not part of this prelude.
pub mod prelude {
    pub use crate::api::prelude::*;
}

pub mod animation;
mod data;
mod debug_ui;

// The `advanced-interop` Cargo feature gates unstable, unsafe, or internal
// rendering surfaces behind an explicit opt-in. When disabled (the default):
// - `api::advanced` is not available.
// - `rendergraph` and its `RenderPassNode` trait are private.
//
// When enabled (`--features advanced-interop`):
// - `api::advanced` exposes `unsafe fn renderer_core_mut()` for raw backend access.
// - `rendergraph` becomes a public module, allowing expert-level custom pass
//   experimentation.
//
// **This feature is alpha/unstable.** The types and functions exposed under
// `advanced-interop` may change or be removed across alpha sprints without
// prior notice. Custom rendergraph pass registration has no resource
// declaration or synchronization validation. Misuse of raw backend access
// can break frame synchronization, descriptor lifecycle, or swapchain safety.
//
// Beginner applications and examples should remain on the default path.
#[cfg(not(feature = "advanced-interop"))]
mod rendergraph;
#[cfg(feature = "advanced-interop")]
pub mod rendergraph;
mod scene;
mod vulkan;

// Crate-root canonical re-exports. For a broader quickstart import, use
// `use renderer::prelude::*` which re-exports ~60 types for common workflows.
pub use api::event_logging::install_app_event_logger;
pub use api::{
    boxed_render_hook,
    AssetError,
    AssetKind,
    // Asset API
    AssetManager,
    BoxedRenderHook,
    CameraView,
    // Debug
    DebugRuntimeMode,
    EngineEvent,
    // Event API
    EventBus,
    // Frame API
    FrameContext,
    FrameRenderOutcome,
    // Handles (widely used)
    MaterialHandle,
    MeshHandle,
    // Lighting
    PointLight,
    PointLightId,
    // Hooks
    RenderHook,
    RenderHookContext,
    // Core runtime
    Renderer,
    RendererConfig,
    // Error root
    RendererError,
    RendererInputRouting,
    RendererInputSuppression,
    // Scene API
    Scene,
    SceneError,
    SceneNodeId,
    TextureHandle,
    VisualTuning,
};

pub use animation::AnimationPlayer;
pub use data::camera::{Aabb, Camera, FPSController, Frustum, OrbitCamera, OrbitController, Ray};
pub use scene::command::{
    AddNodeCommand, Command, CommandHistory, CommandResult, PlaceAssetCommand, RemoveNodeCommand,
    SceneNodeRemap, SetTransformCommand,
};
pub use scene::scene_world::SceneWorld;
