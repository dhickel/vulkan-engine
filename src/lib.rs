//! Thin root facade for the engine workspace.
//!
//! The root crate keeps raw support crates available through stable module
//! paths without taking ownership away from those crates.

pub mod camera;
pub mod command;
pub mod events;
pub mod frame;
pub mod input;
pub mod launch;
pub mod object;
pub mod render;
pub mod runtime;
pub mod time;

/// Common imports for beginner app/runtime code.
pub mod prelude {
    pub use crate::camera::{
        Camera, EditorCamera, EditorProjection, FPSController, OrbitCamera, OrbitController,
    };
    pub use crate::command::{Command, CommandHistory, CommandResult};
    pub use crate::events::{
        runtime_event_bus, EngineEvent, EventBus, EventStage, FrameId, RuntimeEventDispatcher,
    };
    pub use crate::frame::{
        begin_app_frame, end_app_frame, AppFrameBeginReport, AppFrameEndReport, FixedStepClock,
        FixedStepConfig, FixedStepUpdate, FrameClock, FrameInfo,
    };
    pub use crate::input::{
        queue_routed_input_event, route_platform_input_to_app, ActionId, Axis2D,
        AxisContributor, CompoundAxis, InputActionEventEmitter, InputEvent, InputSnapshot,
        InputSystem,
    };
    pub use crate::object::{
        object_kind, object_kind_label, ObjectId, ObjectKind, ObjectQueryFilter, RayHit,
        SceneObjectId, Selection, SelectionChange,
    };
    pub use crate::render::{
        camera_view_for_size, CameraView, FrameContext, FrameRenderOutcome, Renderer,
        RendererConfig, RendererInputRouting, RendererInputSuppression, Scene, SceneNodeId,
    };
}
