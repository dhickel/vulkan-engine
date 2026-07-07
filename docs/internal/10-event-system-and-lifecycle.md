# Event System and Lifecycle

## 1. Purpose & Audience

This page is for contributors changing event contracts, renderer/runtime emission points, or app/tool consumption. It documents ownership boundaries and validation expectations for the alpha event system.

## 2. Where This Fits in Engine Flow

`engine_events` is a standalone workspace crate. Renderer and root runtime code may depend on it at facade/runtime boundaries. Low-level renderer internals should not depend on it unless a future plan explicitly changes the architecture.

Boundary map:

```text
apps/dungeon_dogfood
  -> renderer facade reexports
      -> src/renderer/src/api/renderer.rs
          -> InputSystem snapshots
          -> VkRender only through existing facade calls

root engine runtime
  -> engine::events helpers
  -> renderer facade for project/package/scene/render work

app-owned runtime path
  -> engine::events::RuntimeEventDispatcher
      -> caller-owned EventBus
  -> renderer render-only/view APIs
```

## 3. Key Concepts

- `src/events` owns typed vocabulary and dispatch mechanics only.
- `EventBus::emit` records and queues events; stage drain is explicit.
- `Renderer` drains input/pre/post update stages at controlled frame boundaries.
- Root runtime emits startup, project, package, scene, and shutdown events while validating/loading the data-driven project path.
- Root `engine::frame::begin_app_frame`/`end_app_frame` provide the preferred app-owned lifecycle ordering over a caller-owned `EventBus`; lower-level `engine::events` helpers remain available.
- App crates consume events through `renderer::{EventBus, EventRecorder, EngineEvent, ...}` reexports.
- Listener callbacks receive immutable event envelopes and must not hold renderer internals.

## 4. Integration Points

Renderer facade:

- `Renderer::events()` exposes read-only bus inspection.
- `Renderer::events_mut()` allows subscription/unsubscription.
- `Renderer::set_event_recorder(...)` installs or clears an optional recorder.
- Legacy frame APIs `render_scene`, `render_scene_headless`, `begin_frame`, and `end_frame` emit renderer-owned `FrameStarted`/`FrameEnded`.
- Render-only/view APIs used by the new app-owned path do not emit app lifecycle events.
- `prepare_frame` and `prepare_frame_headless` dispatch input, emit action events from the refreshed snapshot, and drain `EventStage::Input`.

Root runtime:

- `RuntimeEvents` uses `engine::events::runtime_event_bus()` and `RuntimeEventDispatcher` for a bounded app-owned event bus.
- Startup emits `AppStarting`, `ProjectLoading`, `ProjectLoaded`, and `AppStarted`.
- Enabled package loads emit `PackageLoading`, `PackageLoaded`, or `PackageFailed`.
- Startup scene flow emits `SceneLoading` and `SceneLoaded`.
- Headless success emits `ShutdownCompleted`.
- Window close/Escape emits `ShutdownRequested`; do not claim windowed `ShutdownCompleted` from `winit::EventLoop::run`.

Apps:

- App-owned loops should use one caller-owned `EventBus` for lifecycle, input, audio, physics, scripting, and diagnostics.
- `begin_app_frame` ticks `FrameClock`, dispatches app input, emits snapshot-derived action events, drains input, and emits/drains `FrameStarted` in that order.
- `end_app_frame` emits/drains `FrameEnded` for the frame index returned by `begin_app_frame`.
- `RuntimeEventDispatcher::frame_started`, `drain_input`, and `frame_ended` remain the lower-level lifecycle helper path.
- `apps/dungeon_dogfood/src/events.rs` installs an app-side recorder/logger over a dogfood-owned
  bus; dogfood audio, input, and frame lifecycle events no longer require `Renderer::events_mut()`.
- These modules are examples of public facade consumption, not product UI event browsers.

## 5. Ordering Rules

Renderer frame:

```text
FrameStarted -> input dispatch -> InputActionEvent drain -> app/FPS updates -> render -> FrameEnded
```

App-owned frame:

```text
route_platform_input_to_app for platform events
  -> begin_app_frame:
       FrameClock tick
       app InputSystem dispatch exactly once
       InputActionEventEmitter emit from refreshed snapshot
       EventStage::Input drain
       FrameStarted emit/drain at PreUpdate
  -> app update / simulation / camera correction
  -> renderer render-only/view call
  -> end_app_frame:
       FrameEnded emit/drain at PostUpdate
```

Ordering guarantees: input action listeners observe the refreshed snapshot's frame index before `FrameStarted`; app update runs after `FrameStarted`; `FrameEnded` is emitted only when the caller invokes `end_app_frame`, so apps must call it after rendering or after deliberately skipping rendering.

Root runtime:

```text
AppStarting -> ProjectLoading -> ProjectLoaded -> AppStarted
  -> PackageLoading -> PackageLoaded or PackageFailed
  -> SceneLoading -> SceneLoaded
  -> ShutdownRequested or ShutdownCompleted
```

Event sequence numbers are monotonic per bus. They are diagnostic ordering, not durable IDs.

## 6. Ownership Rules

- Keep `engine_events` independent of renderer, Vulkan, windowing, editor, dogfood, physics, audio, and scripting crates.
- Keep root `engine::events` helpers as convenience over a caller-owned bus; do not make them the only way to use `engine_events`.
- Keep event imports out of `src/renderer/src/vulkan`, `data`, `scene`, and `shaders` unless a future plan adds a specific boundary.
- Do not pass `&mut Renderer`, `&mut Scene`, Vulkan handles, or cache references to listeners.
- Listener failures should be collected and logged; one listener failure must not stop later listeners/events.
- Use app-owned queues or state for work that needs mutation after observing an event.

## 7. Validation Guidance

Minimum event-system checks:

```sh
cargo test -p engine_events
cargo test -p input
cargo test -p engine
cargo test -p renderer
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
rg -n "engine_events" src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/shaders src/renderer/src/api src/runtime.rs Cargo.toml src/renderer/Cargo.toml
rg -n "EventBus|FrameStarted|FrameEnded|events_mut\\(|drain_stage|dispatch_pending" src apps tests
```

Use true headless draw-target capture only when the event change also affects visible rendering. Event contract/doc/app-consumption changes usually need compile/test evidence, not image evidence. Adopting `begin_app_frame`/`end_app_frame` without camera/render behavior changes does not by itself require capture evidence.

## 8. Cross-Module Links

- Public docs: `docs/api/12-events-and-lifecycle.md`
- Event crate: `src/events/src/lib.rs`
- Root event helpers: `src/events.rs`
- Renderer facade: `src/renderer/src/api/renderer.rs`
- Runtime launcher: `src/runtime.rs`
- Input internals: `docs/internal/09-input-winit-integration.md`
- API/backend boundary: `docs/internal/04-api-to-backend-handoff.md`

## 9. Standard References

- Rust error handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
- Rust ownership: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html

## 10. See Also

- [Internal Architecture](01-architecture.md)
- [Input winit Event Pump Integration](09-input-winit-integration.md)
- [API Events and Lifecycle](../api/12-events-and-lifecycle.md)
