# 07 — Events & Lifecycle

> Provenance: `G-07`

This chapter covers the event bus, lifecycle events, staged dispatch, event recording, and the app-owned frame boundary helpers. Events provide typed, ordered notifications between subsystems without coupling them directly.

For the full API reference (every type, every enum variant, every method), see [Events & Lifecycle](../api/12-events-and-lifecycle.md).

## Architecture Overview

The event system lives in the standalone `engine_events` crate. It has zero dependencies on the renderer, Vulkan, windowing, physics, audio, or scripting crates. This independence allows any subsystem to emit and listen for events without pulling in the entire engine.

```
┌───────────────────────────────────────────────────┐
│                   EventBus                        │
│                                                   │
│  Pending Queue (emission order)                   │
│  ┌──────┬──────┬──────┬──────┬──────┐            │
│  │env 0 │env 1 │env 2 │env 3 │ ...  │            │
│  │Start │Input │PreUp │PostUp│      │            │
│  └──────┴──────┴──────┴──────┴──────┘            │
│                                                   │
│  Listeners (sorted by priority, then insertion)   │
│  ┌──────┬──────┬──────┐                          │
│  │ L1(5)│ L2(0)│ L3(0)│  ← priority in parens   │
│  └──────┴──────┴──────┘                          │
│                                                   │
│  Optional: EventRecorder (bounded ring buffer)    │
└───────────────────────────────────────────────────┘
```

## Key Types

| Type | Purpose |
|------|---------|
| `EventBus` | Staged in-memory event bus with emit, subscribe, drain_stage, dispatch_pending |
| `EventEnvelope` | Per-event metadata: sequence number, stage, optional frame ID, typed payload, consumed flag |
| `EventStage` | Coarse lifecycle stage for safe dispatch boundaries |
| `EngineEvent` | Top-level event enum with 7 families |
| `LifecycleEvent` | App start/stop, project/scene load/save, frame lifecycle |
| `InputActionEvent` | Action input with `ActionPhase` (`Pressed`, `Released`, `Changed`) |
| `SceneEvent` | Node lifecycle, transform, and asset placement |
| `AssetEvent` | Package and asset loading lifecycle |
| `PhysicsEvent` | Collision, trigger, and query events |
| `AudioEvent` | Clip start/stop/finish/fail events |
| `ScriptingEvent` | Script-emitted events and errors |
| `EventRecorder` | Bounded circular buffer of emitted events |
| `DispatchReport` | Dispatch result with dispatched count and `ListenerFailure` collection |
| `ListenerId` | Handle returned by `EventBus::subscribe` |

## Creating an Event Bus

### App-Owned Bus (recommended for custom apps)

> Provenance: `G-07-APP-BUS` — Full Match (checkpoint pattern)

```rust
use engine::prelude::{runtime_event_bus, EventBus};

let mut events = runtime_event_bus();p
```

`runtime_event_bus()` creates a caller-owned `EventBus` with a bounded recorder already attached. Use this bus for all app lifecycle, input action, and subsystem events. Custom apps should **never** interact with the renderer's internal bus.

### Renderer-Owned Bus (compatibility/diagnostic only)

The renderer maintains its own `EventBus` accessible via `renderer.events()` and `renderer.events_mut()`. This bus is used by the renderer's compatibility path (`render_scene`, `update_input`) and internal subsystems. **Do not use this bus in custom apps.** Create your own via `runtime_event_bus()`.

## The Event Families

> Provenance: `G-07-FAMILIES` — Conceptual; enum definitions in `src/events/src/lib.rs`

### LifecycleEvent

```rust
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
```

### InputActionEvent

```rust
pub struct InputActionEvent {
    pub action: ActionId,
    pub phase: ActionPhase,  // Pressed | Released | Changed
    pub value: f32,
    pub source: Option<String>,
}
```

### SceneEvent

```rust
pub enum SceneEvent {
    NodeCreated { node: NodeId },
    NodeRemoved { node: NodeId },
    NodeRenamed { node: NodeId, name: String },
    NodeTransformed { node: NodeId },
    AssetPlaced { node: NodeId, asset: AssetId },
    MaterialChanged { node: NodeId, material: MaterialId },
}
```

### AssetEvent

```rust
pub enum AssetEvent {
    PackageLoading { package: PackageId, path: String },
    PackageLoaded { package: PackageId, path: String },
    PackageFailed { package: PackageId, message: String },
    AssetLoading { asset: AssetId },
    AssetReady { asset: AssetId },
    AssetFailed { asset: AssetId, message: String },
    AssetInvalidated { asset: AssetId, reason: String },
}
```

### PhysicsEvent

```rust
pub enum PhysicsEvent {
    Collision { phase: ContactPhase, a: ColliderId, b: ColliderId },
    Trigger { phase: ContactPhase, trigger: ColliderId, other: ColliderId },
    QueryHit { body: PhysicsBodyId, collider: ColliderId },
}
```

### AudioEvent

```rust
pub enum AudioEvent {
    ClipStarted { clip: AudioClipId },
    ClipStopped { clip: AudioClipId },
    ClipFinished { clip: AudioClipId },
    ClipFailed { clip: AudioClipId, message: String },
}
```

### ScriptingEvent

```rust
pub enum ScriptingEvent {
    ScriptEmitted { script: ScriptId, name: String, payload: Option<serde_json::Value> },
    ScriptError { script: ScriptId, message: String },
}
```

## Emitting Events

> Provenance: `G-07-EMIT` — Excerpt

```rust
use engine::prelude::{EngineEvent, EventStage, FrameId, LifecycleEvent};

// Emit a lifecycle event at the Startup stage
let seq = events.emit(
    EventStage::Startup,
    None,
    EngineEvent::Lifecycle(LifecycleEvent::AppStarted {
        app_name: "my_app".to_string(),
    }),
);

// Emit with a frame context
let seq = events.emit(
    EventStage::PreUpdate,
    Some(FrameId(42)),
    EngineEvent::Lifecycle(LifecycleEvent::FrameStarted),
);
```

Events are emitted with:
- An `EventStage` indicating which lifecycle phase they belong to
- An optional `FrameId` for frame-scoped events
- The typed `EngineEvent` payload

Each emission returns a monotonic `EventSequence`. If an `EventRecorder` is attached, emitted events are also copied to the recorder before entering the pending queue.

## Subscribing to Events

### Universal Subscribers

> Provenance: `G-07-SUBSCRIBE` — Excerpt

```rust
// Subscribe to ALL events
let listener_id = events.subscribe(|envelope: &EventEnvelope| {
    match &envelope.event {
        EngineEvent::Lifecycle(lifecycle) => {
            println!("lifecycle: {:?} at stage {:?}", lifecycle, envelope.stage);
        }
        EngineEvent::Input(action) => {
            println!("action: {:?} phase {:?}", action.action, action.phase);
        }
        _ => {}
    }
    Ok(())
});
```

### Typed Subscribers

> Provenance: `G-07-TYPED-SUB` — Excerpt

```rust
use engine::prelude::{InputActionEvent, LifecycleEvent};

// Subscribe only to input action events
events.subscribe_to::<InputActionEvent, _>(|event| {
    println!("action {} = {}", event.action, event.value);
    Ok(())
});

// Subscribe only to lifecycle events with priority
events.subscribe_to_with_priority::<LifecycleEvent, _>(
    |event| {
        if matches!(event, LifecycleEvent::FrameStarted) {
            // Run expensive telemetry at high priority
        }
        Ok(())
    },
    10,  // higher priority runs first
);
```

Typed subscribers use the `EventFamily` trait. Only matching event variants reach the callback — other types are silently skipped. This is the preferred pattern: subscribe to exactly the event family you care about.

### Listener Priority

Listeners run in descending priority order, then insertion order for equal priorities. This allows you to control execution order:

```rust
// High priority: runs first, can consume events before lower listeners see them
events.subscribe_with_priority(|_| { Ok(()) }, 100);

// Default priority 0
events.subscribe(|_| { Ok(()) });

// Low priority: runs last
events.subscribe_with_priority(|_| { Ok(()) }, -100);
```

### Event Consumption

Any listener can call `envelope.consume()` to prevent remaining listeners from seeing the event:

```rust
events.subscribe_with_priority(|envelope| {
    if should_handle_exclusively(envelope) {
        envelope.consume();  // no other listener will see this event
    }
    Ok(())
}, 100);
```

Consumption is per-event, not global. Each event's consumed flag is independent.

## Staged Dispatch

> Provenance: `G-07-DRAIN` — Excerpt

Dispatch is explicit. You choose when to drain events and which stage to drain:

```rust
// Drain only the Input stage (e.g., after input dispatch):
let report = events.drain_stage(EventStage::Input);

// Drain all pending events regardless of stage:
let report = events.dispatch_pending();
```

`drain_stage` only removes events matching the requested stage — events at other stages remain in the queue. `dispatch_pending` drains everything.

### The App-Owned Frame Boundary

> Provenance: `G-07-BOUNDARY` — Full Match (checkpoint pattern)

```rust
use engine::prelude::{
    begin_app_frame, end_app_frame, FrameClock, InputActionEventEmitter, InputSystem,
};

let mut events = runtime_event_bus();
let mut input = InputSystem::new();
let mut action_events = InputActionEventEmitter::new();
let mut frame_clock = FrameClock::new();

// --- At start of each frame ---
let begin = begin_app_frame(
    &mut input,
    &mut action_events,
    &mut events,
    &mut frame_clock,
);

// ... app update, fixed-step simulation, render ...

// --- At end of each frame ---
end_app_frame(&mut events, begin.frame.index);
```

`begin_app_frame` does five things in sequence:

1. Ticks `FrameClock` (computes delta, advances frame index).
2. Calls `input.dispatch_frame()` (resets transients, dispatches queued events through layers).
3. Calls `action_events.emit_from_snapshot()` (emits `InputActionEvent`s onto the bus at `EventStage::Input`).
4. Calls `events.drain_stage(EventStage::Input)` (dispatches input events to subscribers).
5. Emits and drains `LifecycleEvent::FrameStarted` on `EventStage::PreUpdate`.

`end_app_frame` emits and drains `LifecycleEvent::FrameEnded` on `EventStage::PostUpdate`.

### App-Owned Frame Ordering

| Order | What Happens | Stage |
|------:|--------------|-------|
| 1 | Queue routed platform input (renderer side effects + app input queue) | — |
| 2 | `input.dispatch_frame()` — reset transients, dispatch layers, refresh snapshot | — |
| 3 | Action events emitted from snapshot | `Input` |
| 4 | Input stage drained — subscribers receive action events | `Input` |
| 5 | `FrameStarted` emitted and drained | `PreUpdate` |
| 6 | App update, fixed-step simulation, camera, render | — |
| 7 | `FrameEnded` emitted and drained | `PostUpdate` |

## DispatchReport and Listener Failures

> Provenance: `G-07-REPORT` — Conceptual; behavior matches `EventBus::dispatch_envelopes()`

Every drain returns a `DispatchReport`:

```rust
pub struct DispatchReport {
    pub dispatched: usize,         // total envelopes dispatched
    pub failures: Vec<ListenerFailure>,
}

pub struct ListenerFailure {
    pub listener: ListenerId,      // which listener failed
    pub sequence: EventSequence,   // which event caused the failure
    pub message: String,           // error or panic message
}
```

Key behaviors:

- **Listener errors do not abort dispatch.** If a listener returns `Err(ListenerError)`, the failure is recorded in the report and dispatch continues with remaining listeners and events.
- **Panicking listeners are poisoned.** If a listener panics, it is caught via `catch_unwind`, the panic is recorded as a failure, and the listener is marked poisoned. Poisoned listeners are skipped on all subsequent events.
- **Consumed events skip remaining listeners.** If any listener calls `envelope.consume()`, no further listeners see that event.

## EventRecorder

> Provenance: `G-07-RECORDER` — Excerpt

The `EventRecorder` is a bounded circular buffer that captures emitted events for diagnostics:

```rust
use engine::prelude::EventRecorder;

// Create a recorder with capacity 256
let recorder = EventRecorder::bounded(256);

// Attach to a bus
events.set_recorder(Some(recorder));

// Later, inspect recorded events:
if let Some(recorder) = events.recorder() {
    for envelope in recorder.entries() {
        println!("seq={:?} stage={:?} event={:?}",
            envelope.sequence, envelope.stage, envelope.event);
    }
    println!("recorded {} of {} events", recorder.len(), recorder.capacity());
}
```

`runtime_event_bus()` creates a bus with a recorder already attached.

## ID Types

The `engine_events` crate defines durable string ID types used across all subsystems:

| Type | Macro | Used By |
|------|-------|---------|
| `ProjectId` | `string_id!` | Root runtime, packaging |
| `PackageId` | `string_id!` | Asset packages |
| `SceneId` | `string_id!` | Scene files |
| `AssetId` | `string_id!` | Asset records |
| `ActionId` | `string_id!` | Input action bindings |
| `NodeId` | `string_id!` | Scene nodes |
| `MaterialId` | `string_id!` | Materials |
| `PhysicsBodyId` | `string_id!` | Physics bodies (shared with `physics` crate) |
| `ColliderId` | `string_id!` | Physics colliders (shared with `physics` crate) |
| `AudioClipId` | `string_id!` | Audio clips (shared with `audio` crate) |
| `ScriptId` | `string_id!` | Script identity |

All ID types implement `Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash`. They are created via `::new("some.id")` or `::from("some.id")`.

## Status of Deferred Event Families

The event vocabulary includes contracts for scene, asset, physics, audio, and scripting events, but broad emission by the engine runtime is staged across sprint tracks:

| Family | Emitted Today |
|--------|:---:|
| `LifecycleEvent` | Yes — app start/stop, project/scene load, frame start/end |
| `InputActionEvent` | Yes — emitted by `InputActionEventEmitter` after `dispatch_frame` |
| `SceneEvent` | Contract exists; broad scene mutation emission is not yet wired |
| `AssetEvent::Package*` | Yes — root runtime emits package load events |
| `AssetEvent::Asset*` | Contract exists; per-asset async load emission is deferred |
| `PhysicsEvent` | Via opt-in helpers — `RayHit::to_engine_event()`, `emit_contact_records()` |
| `AudioEvent` | Via opt-in bridge — see `apps/dungeon_dogfood/src/audio_bridge.rs` |
| `ScriptingEvent` | Via opt-in — `scripting` crate returns events; app emits them |

## Runnable Verification

Run the events crate test suite:

```sh
cargo test -p engine_events
```

Expected: all tests pass (emission ordering, stage drain, unsubscribe, listener failures, panicking listeners, typed subscriptions, priority ordering, consumption).

Build the checkpoint app (exercises the complete app-owned event bus + frame lifecycle):

```sh
cargo check --locked --manifest-path examples/guide_app/Cargo.toml
```

## Mutation Safety Rules

- Listener callbacks receive `&EventEnvelope`, not mutable renderer access. Mutate only app-owned state.
- If a listener needs to trigger renderer/scene work, record the intent in app-owned state and act from the app loop after dispatch completes.
- Do not call Vulkan internals, data caches, or private renderer modules from listeners.
- Do not recursively emit events from within listener callbacks. The alpha contract assumes staged drain from known boundaries.

## Next

Continue to [08 — Scene Construction](08-scene-construction.md) to learn how to build scenes programmatically with nodes, transforms, meshes, materials, lights, and environment maps.
