//! Thin root facade for the engine workspace.
//!
//! The root crate keeps raw support crates available through stable module
//! paths without taking ownership away from those crates.

pub mod camera;
pub mod events;
pub mod frame;
pub mod input;
pub mod launch;
pub mod render;
pub mod runtime;

/// Common imports for beginner app/runtime code.
pub mod prelude {
    pub use crate::camera::{Camera, FPSController, OrbitCamera, OrbitController};
    pub use crate::events::{
        runtime_event_bus, EngineEvent, EventBus, EventStage, FrameId, RuntimeEventDispatcher,
    };
    pub use crate::frame::{
        begin_app_frame, end_app_frame, AppFrameBeginReport, AppFrameEndReport, FixedStepClock,
        FixedStepConfig, FixedStepUpdate, FrameClock, FrameInfo,
    };
    pub use crate::input::{
        queue_routed_input_event, route_platform_input_to_app, ActionId, InputActionEventEmitter,
        InputEvent, InputSnapshot, InputSystem,
    };
    pub use crate::render::{
        camera_view_for_size, CameraView, FrameContext, FrameRenderOutcome, Renderer,
        RendererConfig, RendererInputRouting, RendererInputSuppression, Scene, SceneNodeId,
    };
}
