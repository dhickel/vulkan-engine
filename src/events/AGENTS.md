# Engine Events Crate Agent Guide (`src/events`)

Use this guide for typed event vocabulary, staged ordering, dispatch, and recording mechanics.

## Crate Role

`engine_events` is the shared event vocabulary crate used by all engine subsystems:

- Typed `EngineEvent` enum covering all alpha event variants
- `EventBus` with staged emission, listener subscription, and ordered dispatch
- Monotonic `EventSequence` for event ordering
- Domain-specific ID types: `ScriptId`, `ScriptingEvent`, `PhysicsBodyId`, `ColliderId`, etc.

## Public API

- `EngineEvent` -- comprehensive event enum (Window, Renderer, Input, Physics, Scripting, App, Editor, Debug)
- `EventBus` -- emission, subscription, staging, and dispatch
- `EventStage` -- PreFrame, PostFrame, Immediate
- `ListenerId` -- handle returned by `EventBus::subscribe`
- `EventSequence` -- monotonic ordering within a bus

## Architecture

- Intentionally independent: no dependency on renderer, windowing, Vulkan, editor, dogfood, physics, audio, or scripting
- Events are emitted with `EventStage` and optional `FrameId`
- Subscribers see events in emission order
- Listener failures are collected; dispatch continues for remaining listeners

## Durable ID Types

- `ScriptId` -- durable string identity for scripts
- `ScriptingEvent` -- script-emitted event data
- `PhysicsBodyId` -- physics body identity shared with physics crate
- `ColliderId` -- collider identity shared with physics crate
- `ContactPhase` -- collision contact phase (start, persist, end)

## Working Rules

- New event variants should have clear producer/consumer contracts
- Do not add dependencies on renderer, Vulkan, or windowing crates
- Keep event types `Clone + Debug` for integration testing
- `DispatchFailure` collects errors; do not abort dispatch on first failure
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p engine_events`
- `cargo test -p engine_events`
