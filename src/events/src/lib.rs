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

use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Monotonic event sequence within one [`EventBus`] instance.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EventSequence(pub u64);

/// Listener handle returned by [`EventBus::subscribe`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ListenerId(pub u64);

/// Frame index associated with an event when the producer has frame context.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FrameId(pub u64);

#[macro_export]
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
#[derive(Clone, Debug)]
pub struct EventEnvelope {
    pub sequence: EventSequence,
    pub stage: EventStage,
    pub frame: Option<FrameId>,
    pub event: EngineEvent,
    consumed: Cell<bool>,
}

impl PartialEq for EventEnvelope {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
            && self.stage == other.stage
            && self.frame == other.frame
            && self.event == other.event
            && self.consumed.get() == other.consumed.get()
    }
}

impl EventEnvelope {
    /// Mark this event as consumed, preventing remaining listeners from seeing it.
    pub fn consume(&self) {
        self.consumed.set(true);
    }

    /// Returns true if this event has been consumed by a prior listener.
    pub fn is_consumed(&self) -> bool {
        self.consumed.get()
    }
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

#[derive(Clone, Debug, PartialEq)]
pub enum ScriptingEvent {
    ScriptEmitted {
        script: ScriptId,
        name: String,
        payload: Option<serde_json::Value>,
    },
    ScriptError {
        script: ScriptId,
        message: String,
    },
}

/// Marker trait for event types that can be subscribed to directly via
/// [`EventBus::subscribe_to`].
///
/// Each event variant in [`EngineEvent`] implements this trait so that typed
/// subscribers receive only matching events.
pub trait EventFamily: 'static {
    /// Extract a reference to `Self` from an [`EngineEvent`] envelope, or
    /// `None` if the event is of a different variant.
    fn from_envelope(event: &EngineEvent) -> Option<&Self>
    where
        Self: Sized;
}

impl EventFamily for LifecycleEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Lifecycle(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for InputActionEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Input(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for SceneEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Scene(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for AssetEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Asset(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for PhysicsEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Physics(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for AudioEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Audio(e) => Some(e),
            _ => None,
        }
    }
}

impl EventFamily for ScriptingEvent {
    fn from_envelope(event: &EngineEvent) -> Option<&Self> {
        match event {
            EngineEvent::Scripting(e) => Some(e),
            _ => None,
        }
    }
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
    priority: i32,
    insertion_order: u64,
    poisoned: bool,
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
    insertion_counter: u64,
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

    /// Subscribe to all events with default priority 0.
    pub fn subscribe<F>(&mut self, callback: F) -> ListenerId
    where
        F: FnMut(&EventEnvelope) -> Result<(), ListenerError> + Send + 'static,
    {
        self.subscribe_with_priority(callback, 0)
    }

    /// Subscribe to all events with an explicit priority.
    ///
    /// Higher priority listeners run before lower priority listeners during
    /// dispatch. Listeners with the same priority run in insertion order.
    /// The default priority for [`subscribe`] is 0.
    pub fn subscribe_with_priority<F>(&mut self, callback: F, priority: i32) -> ListenerId
    where
        F: FnMut(&EventEnvelope) -> Result<(), ListenerError> + Send + 'static,
    {
        let id = ListenerId(self.next_listener);
        let insertion_order = self.insertion_counter;
        self.next_listener += 1;
        self.insertion_counter += 1;
        self.listeners.push(ListenerEntry {
            id,
            callback: Box::new(callback),
            priority,
            insertion_order,
            poisoned: false,
        });
        id
    }

    /// Subscribe to a specific event family.
    ///
    /// The callback only receives events that match the type `T`. Other event
    /// types are silently skipped. The universal [`subscribe`] method remains
    /// available for loggers and recorders that need all events.
    ///
    /// # Example
    ///
    /// ```ignore
    /// bus.subscribe_to::<InputActionEvent, _>(|event| {
    ///     println!("action {:?} phase {:?}", event.action, event.phase);
    ///     Ok(())
    /// });
    /// ```
    pub fn subscribe_to<T, F>(&mut self, mut callback: F) -> ListenerId
    where
        T: EventFamily + 'static,
        F: FnMut(&T) -> Result<(), ListenerError> + Send + 'static,
    {
        let untyped_callback: EventCallback = Box::new(move |envelope: &EventEnvelope| {
            if let Some(event) = T::from_envelope(&envelope.event) {
                callback(event)
            } else {
                Ok(())
            }
        });

        let id = ListenerId(self.next_listener);
        let insertion_order = self.insertion_counter;
        self.next_listener += 1;
        self.insertion_counter += 1;
        self.listeners.push(ListenerEntry {
            id,
            callback: untyped_callback,
            priority: 0,
            insertion_order,
            poisoned: false,
        });
        id
    }

    /// Subscribe to a specific event family with an explicit priority.
    ///
    /// Combines the filtering of [`subscribe_to`] with the ordering of
    /// [`subscribe_with_priority`].
    pub fn subscribe_to_with_priority<T, F>(
        &mut self,
        mut callback: F,
        priority: i32,
    ) -> ListenerId
    where
        T: EventFamily + 'static,
        F: FnMut(&T) -> Result<(), ListenerError> + Send + 'static,
    {
        let untyped_callback: EventCallback = Box::new(move |envelope: &EventEnvelope| {
            if let Some(event) = T::from_envelope(&envelope.event) {
                callback(event)
            } else {
                Ok(())
            }
        });

        let id = ListenerId(self.next_listener);
        let insertion_order = self.insertion_counter;
        self.next_listener += 1;
        self.insertion_counter += 1;
        self.listeners.push(ListenerEntry {
            id,
            callback: untyped_callback,
            priority,
            insertion_order,
            poisoned: false,
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
            consumed: Cell::new(false),
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

        // Sort listeners by priority (descending), then insertion order (ascending)
        self.listeners
            .sort_by_key(|l| (std::cmp::Reverse(l.priority), l.insertion_order));

        for envelope in &envelopes {
            for listener in &mut self.listeners {
                if listener.poisoned {
                    continue; // Skip panicked listeners
                }

                let result = catch_unwind(AssertUnwindSafe(|| {
                    (listener.callback)(envelope)
                }));

                match result {
                    Ok(Ok(())) => {
                        // Listener succeeded
                    }
                    Ok(Err(err)) => {
                        report.failures.push(ListenerFailure {
                            listener: listener.id,
                            sequence: envelope.sequence,
                            message: err.to_string(),
                        });
                    }
                    Err(panic_payload) => {
                        listener.poisoned = true;
                        let message = panic_payload_to_string(panic_payload);
                        report.failures.push(ListenerFailure {
                            listener: listener.id,
                            sequence: envelope.sequence,
                            message: format!("listener panicked: {message}"),
                        });
                    }
                }

                if envelope.is_consumed() {
                    break; // Skip remaining listeners for this event
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

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    "unknown panic payload".to_string()
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

    #[test]
    fn panicking_listener_does_not_prevent_dispatch() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);

        // First listener panics
        let panicking_id = bus.subscribe(|_event| {
            panic!("intentional panic in listener");
        });
        // Second listener should still receive events
        bus.subscribe(move |event| {
            seen_listener.lock().unwrap().push(event.sequence);
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("one"));
        bus.emit(EventStage::Startup, None, lifecycle_event("two"));

        let report = bus.dispatch_pending();

        // Both events dispatched
        assert_eq!(report.dispatched, 2);
        // Only first event caused a failure; panicking listener is poisoned and
        // skipped on the second event
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].listener, panicking_id);
        assert!(report.failures[0].message.contains("listener panicked"));
        // Non-panicking listener received both events
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [EventSequence(0), EventSequence(1)]
        );
    }

    #[test]
    fn poisoned_listener_is_skipped_on_subsequent_events() {
        let mut bus = EventBus::new();
        let call_count = Arc::new(Mutex::new(0));
        let call_count_listener = Arc::clone(&call_count);

        // Listener panics on first call
        bus.subscribe(|_event| {
            panic!("intentional panic");
        });
        // Counter listener
        bus.subscribe(move |_event| {
            *call_count_listener.lock().unwrap() += 1;
            Ok(())
        });

        // First event: panicking listener panics, gets poisoned
        bus.emit(EventStage::Startup, None, lifecycle_event("one"));
        bus.dispatch_pending();

        // Second event: panicking listener should be skipped
        bus.emit(EventStage::Startup, None, lifecycle_event("two"));
        let report = bus.dispatch_pending();

        // Only the counter listener should have received the second event
        assert_eq!(*call_count.lock().unwrap(), 2); // 2 calls: one per event
        // No new failures because the panicking listener was skipped
        assert!(report.failures.is_empty());
    }

    // ── Typed subscription tests ──

    #[test]
    fn typed_subscriber_only_receives_matching_event_type() {
        let mut bus = EventBus::new();
        let input_events = Arc::new(Mutex::new(Vec::new()));
        let input_listener = Arc::clone(&input_events);
        let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
        let lifecycle_listener = Arc::clone(&lifecycle_events);

        bus.subscribe_to::<InputActionEvent, _>(move |event| {
            input_listener
                .lock()
                .unwrap()
                .push(event.action.clone());
            Ok(())
        });
        bus.subscribe_to::<LifecycleEvent, _>(move |event| {
            lifecycle_listener
                .lock()
                .unwrap()
                .push(format!("{:?}", event));
            Ok(())
        });

        bus.emit(
            EventStage::Input,
            Some(FrameId(1)),
            EngineEvent::Input(InputActionEvent::new("jump", ActionPhase::Pressed, 1.0)),
        );
        bus.emit(
            EventStage::Startup,
            None,
            EngineEvent::Lifecycle(LifecycleEvent::AppStarting {
                app_name: "test".to_string(),
            }),
        );

        let report = bus.dispatch_pending();
        assert_eq!(report.dispatched, 2);
        assert!(report.failures.is_empty());

        let inputs = input_events.lock().unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].as_str(), "jump");

        let lifecycles = lifecycle_events.lock().unwrap();
        assert_eq!(lifecycles.len(), 1);
        assert!(lifecycles[0].contains("AppStarting"));
    }

    #[test]
    fn typed_subscriber_ignores_other_event_types() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);

        // Subscribe only to Physics events
        bus.subscribe_to::<PhysicsEvent, _>(move |event| {
            seen_listener
                .lock()
                .unwrap()
                .push(format!("{:?}", event));
            Ok(())
        });

        // Emit a non-physics event
        bus.emit(
            EventStage::Input,
            Some(FrameId(1)),
            EngineEvent::Input(InputActionEvent::new("jump", ActionPhase::Pressed, 1.0)),
        );

        let report = bus.dispatch_pending();
        assert_eq!(report.dispatched, 1);
        assert!(report.failures.is_empty());
        // Physics subscriber should not have been called for Input event
        assert!(seen.lock().unwrap().is_empty());
    }

    // ── Priority ordering tests ──

    #[test]
    fn higher_priority_listener_runs_before_lower_priority() {
        let mut bus = EventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let order_low = Arc::clone(&order);
        let order_high = Arc::clone(&order);

        bus.subscribe_with_priority(
            move |_| {
                order_low.lock().unwrap().push("low");
                Ok(())
            },
            -10,
        );
        bus.subscribe_with_priority(
            move |_| {
                order_high.lock().unwrap().push("high");
                Ok(())
            },
            10,
        );

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        bus.dispatch_pending();

        assert_eq!(order.lock().unwrap().as_slice(), &["high", "low"]);
    }

    #[test]
    fn same_priority_listeners_run_in_insertion_order() {
        let mut bus = EventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let order_a = Arc::clone(&order);
        let order_b = Arc::clone(&order);

        bus.subscribe(move |_| {
            order_a.lock().unwrap().push("a");
            Ok(())
        });
        bus.subscribe(move |_| {
            order_b.lock().unwrap().push("b");
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        bus.dispatch_pending();

        assert_eq!(order.lock().unwrap().as_slice(), &["a", "b"]);
    }

    #[test]
    fn typed_subscriber_with_priority_combines_filtering_and_ordering() {
        let mut bus = EventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let order_low = Arc::clone(&order);
        let order_high = Arc::clone(&order);

        bus.subscribe_to_with_priority::<LifecycleEvent, _>(
            move |_| {
                order_low.lock().unwrap().push("low");
                Ok(())
            },
            -5,
        );
        bus.subscribe_to_with_priority::<LifecycleEvent, _>(
            move |_| {
                order_high.lock().unwrap().push("high");
                Ok(())
            },
            5,
        );

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        bus.dispatch_pending();

        assert_eq!(order.lock().unwrap().as_slice(), &["high", "low"]);
    }

    // ── Event consumption tests ──

    #[test]
    fn consuming_event_prevents_lower_priority_listeners() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_high = Arc::clone(&seen);
        let seen_low = Arc::clone(&seen);

        // High-priority listener consumes the event
        bus.subscribe_with_priority(
            move |event| {
                seen_high.lock().unwrap().push("high");
                event.consume();
                Ok(())
            },
            10,
        );
        // Low-priority listener should never see the event
        bus.subscribe_with_priority(
            move |_| {
                seen_low.lock().unwrap().push("low");
                Ok(())
            },
            -10,
        );

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        let report = bus.dispatch_pending();

        assert_eq!(report.dispatched, 1);
        assert!(report.failures.is_empty());
        assert_eq!(seen.lock().unwrap().as_slice(), &["high"]);
    }

    #[test]
    fn consumed_event_skips_remaining_listeners_same_priority() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_a = Arc::clone(&seen);
        let seen_b = Arc::clone(&seen);

        // First listener (insertion order) consumes
        bus.subscribe(move |event| {
            seen_a.lock().unwrap().push("a");
            event.consume();
            Ok(())
        });
        // Second listener should be skipped
        bus.subscribe(move |_| {
            seen_b.lock().unwrap().push("b");
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        bus.dispatch_pending();

        assert_eq!(seen.lock().unwrap().as_slice(), &["a"]);
    }

    #[test]
    fn unconsumed_event_reaches_all_listeners() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_a = Arc::clone(&seen);
        let seen_b = Arc::clone(&seen);

        bus.subscribe(move |_| {
            seen_a.lock().unwrap().push("a");
            Ok(())
        });
        bus.subscribe(move |_| {
            seen_b.lock().unwrap().push("b");
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("test"));
        bus.dispatch_pending();

        assert_eq!(seen.lock().unwrap().as_slice(), &["a", "b"]);
    }

    #[test]
    fn consumption_is_per_event_not_global() {
        let mut bus = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);
        let seen_for_last = Arc::clone(&seen);

        // Listener that consumes every other event (sequence 0 consumed, 1 not)
        bus.subscribe(move |event| {
            let seq = event.sequence.0;
            seen_listener.lock().unwrap().push(format!("a:{}", seq));
            if seq % 2 == 0 {
                event.consume();
            }
            Ok(())
        });
        bus.subscribe(move |event| {
            let seq = event.sequence.0;
            seen.lock().unwrap().push(format!("b:{}", seq));
            Ok(())
        });

        bus.emit(EventStage::Startup, None, lifecycle_event("one")); // seq 0
        bus.emit(EventStage::Startup, None, lifecycle_event("two")); // seq 1
        bus.emit(EventStage::Startup, None, lifecycle_event("three")); // seq 2

        let report = bus.dispatch_pending();
        assert_eq!(report.dispatched, 3);
        assert!(report.failures.is_empty());

        let seen = seen_for_last.lock().unwrap();
        // seq 0: consumed, so only a sees it
        // seq 1: not consumed, both see it
        // seq 2: consumed, so only a sees it
        assert_eq!(
            seen.as_slice(),
            &["a:0", "a:1", "b:1", "a:2"]
        );
    }
}
