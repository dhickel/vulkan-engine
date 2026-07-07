# Engine Events Crate Agent Guide (`src/events`)

Use this guide for typed event vocabulary, staged ordering, dispatch, and recording mechanics.

## Crate Role

`engine_events` is the shared event vocabulary crate used by all engine subsystems:

- Typed `EngineEvent` enum covering all alpha event families
- `EventBus` with staged emission, listener subscription, and ordered dispatch
- Monotonic `EventSequence` for event ordering
- `EventRecorder` for bounded in-memory event capture
- Domain-specific ID types: `ProjectId`, `PackageId`, `SceneId`, `AssetId`, `ActionId`, `NodeId`, `MaterialId`, `PhysicsBodyId`, `ColliderId`, `AudioClipId`, `ScriptId`

## Public API

- `EngineEvent` — top-level event enum with 7 families: `Lifecycle`, `Input`, `Scene`, `Asset`, `Physics`, `Audio`, `Scripting`
- `LifecycleEvent` — app start/stop, project/scene load/save, frame lifecycle
- `InputActionEvent` — action input with `ActionPhase` (`Pressed`, `Released`, `Changed`)
- `SceneEvent` — node lifecycle, transform, and asset placement
- `AssetEvent` — package and asset loading lifecycle
- `PhysicsEvent` — collision, trigger, and query events with `ContactPhase` (`Enter`, `Stay`, `Exit`)
- `AudioEvent` — clip start/stop/finish/fail events
- `ScriptingEvent` — script-emitted events and errors
- `EventBus` — staged in-memory event bus with subscribe, emit, drain_stage, dispatch_pending
- `EventStage` — `Startup`, `ProjectLoad`, `SceneLoad`, `Input`, `PreUpdate`, `PostUpdate`, `Render`, `Shutdown`
- `EventEnvelope` — event metadata (sequence, stage, frame) plus typed payload
- `ListenerId` — handle returned by `EventBus::subscribe`
- `EventSequence` — monotonic ordering within a bus
- `FrameId` — optional frame index associated with an event
- `DispatchReport` — dispatch result with dispatched count and `ListenerFailure` collection
- `ListenerFailure` — per-listener failure detail (listener id, sequence, message)
- `ListenerError` — error type returned by listener callbacks
- `EventRecorder` — bounded circular buffer of emitted events

## Architecture

- Intentionally independent: no dependency on renderer, windowing, Vulkan, editor, dogfood, physics, audio, or scripting
- Events are emitted with `EventStage` and optional `FrameId`
- Subscribers see events in emission order
- Listener failures are collected via `DispatchReport`; dispatch continues for remaining listeners
- `drain_stage` dispatches events for a specific stage and retains others
- `dispatch_pending` drains all pending events
- `EventRecorder` provides bounded in-memory capture for diagnostics

## ID Types

All ID types are generated via the `string_id!` macro and derive `Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash`:

- `ProjectId` — project identity
- `PackageId` — package identity
- `SceneId` — scene identity
- `AssetId` — asset identity
- `ActionId` — input action identity
- `NodeId` — scene node identity
- `MaterialId` — material identity
- `PhysicsBodyId` — physics body identity shared with physics crate
- `ColliderId` — collider identity shared with physics crate
- `AudioClipId` — audio clip identity shared with audio crate
- `ScriptId` — durable string identity for scripts

## Working Rules

- New event variants should have clear producer/consumer contracts
- Do not add dependencies on renderer, Vulkan, or windowing crates
- Keep event types `Clone + Debug` for integration testing
- `DispatchReport` collects failures; do not abort dispatch on first failure
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p engine_events`
- `cargo test -p engine_events`
