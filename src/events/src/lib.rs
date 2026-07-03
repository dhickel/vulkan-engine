//! Core event contracts for the engine alpha runtime.
//!
//! This crate is intentionally independent from the renderer, windowing, Vulkan,
//! editor, dogfood app, physics, audio, and scripting crates. It owns only typed
//! event vocabulary, staged ordering, dispatch, and recording mechanics.
//!
//! Events are emitted into an [`EventBus`] with an [`EventStage`] and optional
//! frame index. Emission assigns a monotonic [`EventSequence`]. Dispatch is
//! explicit: callers drain a specific stage with [`EventBus::drain_stage`] or
//! drain all pending events with [`EventBus::dispatch_pending`]. Subscribers see
//! events in emission order. Listener failures are collected and dispatch
//! continues for later listeners and events.

use std::collections::VecDeque;
use std::fmt;

/// Monotonic event sequence within one [`EventBus`] instance.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EventSequence(pub u64);

/// Listener handle returned by [`EventBus::subscribe`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ListenerId(pub u64);

/// Frame index associated with an event when the producer has frame context.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FrameId(pub u64);

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

string_id!(ProjectId);
string_id!(PackageId);
string_id!(SceneId);
string_id!(AssetId);
string_id!(ActionId);
string_id!(NodeId);
string_id!(MaterialId);
string_id!(PhysicsBodyId);
string_id!(ColliderId);
string_id!(AudioClipId);
string_id!(ScriptId);

/// Coarse frame-safe event stages.
///
/// Stages are ordered by their normal lifecycle position, but the bus always
/// preserves emission order during dispatch. Callers choose which stage to
/// drain so app/tool callbacks can run at explicit safe boundaries rather than
/// during renderer internals.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EventStage {
    /// Runtime initialization before project loading.
    Startup,
    /// Project and package validation/loading.
    ProjectLoad,
    /// Startup scene validation/loading.
    SceneLoad,
    /// Input/action events after input dispatch and snapshot refresh.
    Input,
    /// App systems may observe state before scene mutation.
    PreUpdate,
    /// App systems have completed frame mutation.
    PostUpdate,
    /// Render lifecycle markers only; app mutation should not happen here.
    Render,
    /// Controlled shutdown and teardown.
    Shutdown,
}

/// Event envelope metadata plus typed payload.
#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope {
    pub sequence: EventSequence,
    pub stage: EventStage,
    pub frame: Option<FrameId>,
    pub event: EngineEvent,
}

/// Top-level event family.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    Lifecycle(LifecycleEvent),
    Input(InputActionEvent),
    Scene(SceneEvent),
    Asset(AssetEvent),
    Physics(PhysicsEvent),
    Audio(AudioEvent),
    Scripting(ScriptingEvent),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleEvent {
    AppStarting { app_name: String },
    AppStarted { app_name: String },
    ProjectLoading { path: String },
    ProjectLoaded { project: ProjectId, path: String },
    SceneLoading { scene: SceneId, path: String },
    SceneLoaded { scene: SceneId, path: String },
    SceneSaved { scene: SceneId, path: String },
    FrameStarted,
    FrameEnded,
    ShutdownRequested { reason: String },
    ShutdownCompleted,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ActionPhase {
    Pressed,
    Released,
    Changed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InputActionEvent {
    pub action: ActionId,
    pub phase: ActionPhase,
    pub value: f32,
    pub source: Option<String>,
}

impl InputActionEvent {
    pub fn new(action: impl Into<ActionId>, phase: ActionPhase, value: f32) -> Self {
        Self {
            action: action.into(),
            phase,
            value,
            source: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SceneEvent {
    NodeCreated { node: NodeId },
    NodeRemoved { node: NodeId },
    NodeRenamed { node: NodeId, name: String },
    NodeTransformed { node: NodeId },
    AssetPlaced { node: NodeId, asset: AssetId },
    MaterialChanged { node: NodeId, material: MaterialId },
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssetEvent {
    PackageLoading { package: PackageId, path: String },
    PackageLoaded { package: PackageId, path: String },
    PackageFailed { package: PackageId, message: String },
    AssetLoading { asset: AssetId },
    AssetReady { asset: AssetId },
    AssetFailed { asset: AssetId, message: String },
    AssetInvalidated { asset: AssetId, reason: String },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ContactPhase {
    Enter,
    Stay,
    Exit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicsEvent {
    Collision {
        phase: ContactPhase,
        a: ColliderId,
        b: ColliderId,
    },
    Trigger {
        phase: ContactPhase,
        trigger: ColliderId,
        other: ColliderId,
    },
    QueryHit {
        body: PhysicsBodyId,
        collider: ColliderId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioEvent {
    ClipStarted { clip: AudioClipId },
    ClipStopped { clip: AudioClipId },
    ClipFinished { clip: AudioClipId },
    ClipFailed { clip: AudioClipId, message: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptingEvent {
    ScriptEmitted {
        script: ScriptId,
        name: String,
        payload: Option<String>,
    },
    ScriptError {
        script: ScriptId,
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListenerError {
    message: String,
}

impl ListenerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ListenerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for ListenerError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListenerFailure {
    pub listener: ListenerId,
    pub sequence: EventSequence,
    pub message: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DispatchReport {
    pub dispatched: usize,
    pub failures: Vec<ListenerFailure>,
}

type EventCallback = Box<dyn FnMut(&EventEnvelope) -> Result<(), ListenerError> + Send + 'static>;

struct ListenerEntry {
    id: ListenerId,
    callback: EventCallback,
}

/// Staged in-memory event bus.
///
/// Emission is append-only for pending events. Dispatch drains pending events,
/// invokes current listeners, and keeps running after individual listener
/// failures. Listeners cannot receive mutable access to the bus, which keeps
/// recursive dispatch out of the core contract for alpha.
#[derive(Default)]
pub struct EventBus {
    next_sequence: u64,
    next_listener: u64,
    pending: VecDeque<EventEnvelope>,
    listeners: Vec<ListenerEntry>,
    recorder: Option<EventRecorder>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_recorder(recorder: EventRecorder) -> Self {
        let mut bus = Self::new();
        bus.set_recorder(Some(recorder));
        bus
    }

    pub fn subscribe<F>(&mut self, callback: F) -> ListenerId
    where
        F: FnMut(&EventEnvelope) -> Result<(), ListenerError> + Send + 'static,
    {
        let id = ListenerId(self.next_listener);
        self.next_listener += 1;
        self.listeners.push(ListenerEntry {
            id,
            callback: Box::new(callback),
        });
        id
    }

    pub fn unsubscribe(&mut self, id: ListenerId) -> bool {
        let before = self.listeners.len();
        self.listeners.retain(|listener| listener.id != id);
        self.listeners.len() != before
    }

    pub fn emit(
        &mut self,
        stage: EventStage,
        frame: Option<FrameId>,
        event: EngineEvent,
    ) -> EventSequence {
        let sequence = EventSequence(self.next_sequence);
        self.next_sequence += 1;

        let envelope = EventEnvelope {
            sequence,
            stage,
            frame,
            event,
        };

        if let Some(recorder) = self.recorder.as_mut() {
            recorder.record(envelope.clone());
        }
        self.pending.push_back(envelope);
        sequence
    }

    pub fn drain_stage(&mut self, stage: EventStage) -> DispatchReport {
        let mut selected = Vec::new();
        let mut retained = VecDeque::new();

        while let Some(envelope) = self.pending.pop_front() {
            if envelope.stage == stage {
                selected.push(envelope);
            } else {
                retained.push_back(envelope);
            }
        }

        self.pending = retained;
        self.dispatch_envelopes(selected)
    }

    pub fn dispatch_pending(&mut self) -> DispatchReport {
        let selected = self.pending.drain(..).collect();
        self.dispatch_envelopes(selected)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn listener_count(&self) -> usize {
        self.listeners.len()
    }

    pub fn set_recorder(&mut self, recorder: Option<EventRecorder>) {
        self.recorder = recorder;
    }

    pub fn recorder(&self) -> Option<&EventRecorder> {
        self.recorder.as_ref()
    }

    pub fn recorder_mut(&mut self) -> Option<&mut EventRecorder> {
        self.recorder.as_mut()
    }

    fn dispatch_envelopes(&mut self, envelopes: Vec<EventEnvelope>) -> DispatchReport {
        let mut report = DispatchReport {
            dispatched: envelopes.len(),
            failures: Vec::new(),
        };

        for envelope in &envelopes {
            for listener in &mut self.listeners {
                if let Err(err) = (listener.callback)(envelope) {
                    report.failures.push(ListenerFailure {
                        listener: listener.id,
                        sequence: envelope.sequence,
                        message: err.to_string(),
                    });
                }
            }
        }

        report
    }
}

/// Bounded record of emitted events.
#[derive(Clone, Debug, PartialEq)]
pub struct EventRecorder {
    capacity: usize,
    entries: VecDeque<EventEnvelope>,
}

impl EventRecorder {
    pub fn bounded(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &EventEnvelope> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn record(&mut self, envelope: EventEnvelope) {
        if self.capacity == 0 {
            return;
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(envelope);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn lifecycle_event(name: &str) -> EngineEvent {
        EngineEvent::Lifecycle(LifecycleEvent::AppStarting {
            app_name: name.to_string(),
        })
    }

    #[test]
    fn event_families_are_constructible() {
        let events = [
            EngineEvent::Lifecycle(LifecycleEvent::ProjectLoaded {
                project: ProjectId::new("project-a"),
                path: "engine.project.toml".to_string(),
            }),
            EngineEvent::Input(InputActionEvent::new("jump", ActionPhase::Pressed, 1.0)),
            EngineEvent::Scene(SceneEvent::AssetPlaced {
                node: NodeId::new("node-a"),
                asset: AssetId::new("asset-a"),
            }),
            EngineEvent::Asset(AssetEvent::AssetReady {
                asset: AssetId::new("asset-a"),
            }),
            EngineEvent::Physics(PhysicsEvent::Collision {
                phase: ContactPhase::Enter,
                a: ColliderId::new("collider-a"),
                b: ColliderId::new("collider-b"),
            }),
            EngineEvent::Audio(AudioEvent::ClipFinished {
                clip: AudioClipId::new("clip-a"),
            }),
            EngineEvent::Scripting(ScriptingEvent::ScriptError {
                script: ScriptId::new("script-a"),
                message: "boom".to_string(),
            }),
        ];

        assert_eq!(events.len(), 7);
    }

    #[test]
    fn emission_assigns_monotonic_sequences() {
        let mut bus = EventBus::new();

        let first = bus.emit(EventStage::Startup, None, lifecycle_event("one"));
        let second = bus.emit(
            EventStage::Startup,
            Some(FrameId(7)),
            lifecycle_event("two"),
        );

        assert_eq!(first, EventSequence(0));
        assert_eq!(second, EventSequence(1));
        assert_eq!(bus.pending_len(), 2);
    }

    #[test]
    fn drain_stage_dispatches_selected_stage_in_emission_order() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);

        bus.subscribe(move |event| {
            seen_listener.lock().unwrap().push(event.sequence);
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("startup"));
        bus.emit(
            EventStage::Input,
            Some(FrameId(1)),
            lifecycle_event("input"),
        );
        bus.emit(EventStage::Startup, None, lifecycle_event("startup-2"));

        let report = bus.drain_stage(EventStage::Startup);

        assert_eq!(report.dispatched, 2);
        assert!(report.failures.is_empty());
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [EventSequence(0), EventSequence(2)]
        );
        assert_eq!(bus.pending_len(), 1);
    }

    #[test]
    fn unsubscribe_removes_listener() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(0));
        let seen_listener = Arc::clone(&seen);

        let id = bus.subscribe(move |_| {
            *seen_listener.lock().unwrap() += 1;
            Ok(())
        });

        assert_eq!(bus.listener_count(), 1);
        assert!(bus.unsubscribe(id));
        assert!(!bus.unsubscribe(id));

        bus.emit(EventStage::Startup, None, lifecycle_event("startup"));
        let report = bus.dispatch_pending();

        assert_eq!(report.dispatched, 1);
        assert_eq!(*seen.lock().unwrap(), 0);
        assert_eq!(bus.listener_count(), 0);
    }

    #[test]
    fn listener_failures_are_collected_and_dispatch_continues() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);

        let failing = bus.subscribe(|event| {
            Err(ListenerError::new(format!(
                "failed on {}",
                event.sequence.0
            )))
        });
        bus.subscribe(move |event| {
            seen_listener.lock().unwrap().push(event.sequence);
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("one"));
        bus.emit(EventStage::Startup, None, lifecycle_event("two"));

        let report = bus.dispatch_pending();

        assert_eq!(report.dispatched, 2);
        assert_eq!(report.failures.len(), 2);
        assert_eq!(report.failures[0].listener, failing);
        assert_eq!(report.failures[0].sequence, EventSequence(0));
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [EventSequence(0), EventSequence(1)]
        );
    }

    #[test]
    fn recorder_keeps_bounded_emission_order() {
        let mut bus = EventBus::with_recorder(EventRecorder::bounded(2));

        bus.emit(EventStage::Startup, None, lifecycle_event("one"));
        bus.emit(EventStage::Startup, None, lifecycle_event("two"));
        bus.emit(EventStage::Startup, None, lifecycle_event("three"));

        let sequences: Vec<_> = bus
            .recorder()
            .unwrap()
            .entries()
            .map(|event| event.sequence)
            .collect();

        assert_eq!(sequences, [EventSequence(1), EventSequence(2)]);
        assert_eq!(bus.recorder().unwrap().len(), 2);
    }

    #[test]
    fn zero_capacity_recorder_records_nothing() {
        let mut bus = EventBus::with_recorder(EventRecorder::bounded(0));

        bus.emit(EventStage::Startup, None, lifecycle_event("one"));

        assert!(bus.recorder().unwrap().is_empty());
    }
}
