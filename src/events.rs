//! Event primitives and lightweight runtime lifecycle helpers.

pub use engine_events::{
    ActionId, ActionPhase, AssetEvent, AssetId, AudioClipId, AudioEvent, ColliderId, ContactPhase,
    DispatchReport, EngineEvent, EventBus, EventEnvelope, EventRecorder, EventSequence, EventStage,
    FrameId, InputActionEvent, LifecycleEvent, ListenerError, ListenerFailure, ListenerId,
    MaterialId, NodeId, PackageId, PhysicsBodyId, PhysicsEvent, ProjectId, SceneEvent, SceneId,
    ScriptId, ScriptingEvent,
};

/// Default event recorder capacity used by root runtime helpers.
pub const DEFAULT_RUNTIME_EVENT_RECORDER_CAPACITY: usize = 512;

/// Construct an app-owned event bus with bounded in-memory recording enabled.
///
/// This is a convenience for app/runtime code. Callers can still use
/// [`EventBus::new`] or [`EventBus::with_recorder`] directly when they want raw
/// control over recording.
pub fn runtime_event_bus() -> EventBus {
    runtime_event_bus_with_recorder_capacity(DEFAULT_RUNTIME_EVENT_RECORDER_CAPACITY)
}

/// Construct an app-owned event bus with a caller-selected recorder capacity.
pub fn runtime_event_bus_with_recorder_capacity(capacity: usize) -> EventBus {
    EventBus::with_recorder(EventRecorder::bounded(capacity))
}

/// Stateless helper for app-owned staged lifecycle dispatch.
///
/// The helper deliberately operates on a caller-provided [`EventBus`] so the app
/// can keep one bus for lifecycle, input, audio, scripting, and diagnostics.
#[derive(Copy, Clone, Debug, Default)]
pub struct RuntimeEventDispatcher;

impl RuntimeEventDispatcher {
    /// Emit one event into the caller-owned bus and immediately drain that stage.
    pub fn emit_and_drain(
        bus: &mut EventBus,
        stage: EventStage,
        frame: Option<FrameId>,
        event: EngineEvent,
    ) -> DispatchReport {
        bus.emit(stage, frame, event);
        bus.drain_stage(stage)
    }

    /// Emit one lifecycle event into the caller-owned bus and drain its stage.
    pub fn emit_lifecycle_and_drain(
        bus: &mut EventBus,
        stage: EventStage,
        frame: Option<FrameId>,
        event: LifecycleEvent,
    ) -> DispatchReport {
        Self::emit_and_drain(bus, stage, frame, EngineEvent::Lifecycle(event))
    }

    /// Emit `FrameStarted` for the provided frame on `PreUpdate`.
    pub fn frame_started(bus: &mut EventBus, frame_index: u64) -> DispatchReport {
        Self::emit_lifecycle_and_drain(
            bus,
            EventStage::PreUpdate,
            Some(FrameId(frame_index)),
            LifecycleEvent::FrameStarted,
        )
    }

    /// Drain input events after app-owned input dispatch/action emission.
    pub fn drain_input(bus: &mut EventBus) -> DispatchReport {
        bus.drain_stage(EventStage::Input)
    }

    /// Emit `FrameEnded` for the provided frame on `PostUpdate`.
    pub fn frame_ended(bus: &mut EventBus, frame_index: u64) -> DispatchReport {
        Self::emit_lifecycle_and_drain(
            bus,
            EventStage::PostUpdate,
            Some(FrameId(frame_index)),
            LifecycleEvent::FrameEnded,
        )
    }
}
