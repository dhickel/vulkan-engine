# Events and Lifecycle

## 1. Purpose & Audience

This chapter documents the alpha event contract for app, tool, and runtime code that needs lifecycle or input-action observations without reaching into renderer internals.

Use events for logging, lightweight telemetry, validation recorders, editor tooling, and safe frame-boundary notifications. Do not use the alpha event bus as a cross-thread job system, a scripting VM, or a replacement for direct scene/game state ownership.

## 2. Where This Fits in Engine Flow

> **renderer compatibility path** (demos, examples, smoke testing):
`winit` event loop -> `Renderer::update_input(...)` -> `Renderer::render_scene(...)` or explicit frame API -> input dispatch -> input events -> frame lifecycle events -> render.

> **current app-owned path** (recommended for custom apps that own gameplay/input/camera state):
`winit` event loop -> `engine::input::route_platform_input_to_app(...)` -> `engine::frame::begin_app_frame(...)` -> app update -> renderer `render_scene_with_view(...)` -> `engine::frame::end_app_frame(...)`. See [15-app-owned-loop.md](15-app-owned-loop.md).

Root runtime path:
launcher startup -> project load -> package load -> scene load -> headless/windowed run -> shutdown.

The event vocabulary lives in the standalone `engine_events` crate and is re-exported through `renderer`, `renderer::api`, and the root `engine::events` facade for app users.

## 3. Key Concepts

- `EventBus` owns staged pending events, listener registration, and optional recording.
- `EventEnvelope` adds sequence number, stage, optional frame ID, and typed payload.
- `EventStage` is the safe-boundary label: `Startup`, `ProjectLoad`, `SceneLoad`, `Input`, `PreUpdate`, `PostUpdate`, `Render`, or `Shutdown`.
- `EngineEvent` groups payload families: lifecycle, input, scene, asset, physics, audio, and scripting.
- `EventRecorder::bounded(capacity)` stores emitted envelopes for diagnostics and validation.
- `Renderer::events()` and `Renderer::events_mut()` expose the bus through the public facade.
- `engine::events::runtime_event_bus()` creates a caller-owned bus with bounded recording enabled.

### Frame Lifecycle Guarantees

- Frame `FrameStarted` and `FrameEnded` events are always paired when emitted through
  `execute_frame_lifecycle`, `with_frame`, or `RuntimeEventDispatcher::frame_started` /
  `frame_ended`. Even on error, the `FrameEnded` event records the terminal outcome.
- The headless path (`render_scene_headless`) and windowed path (`render_scene`)
  converge on the same lifecycle event sequence: `PreUpdate` (FrameStarted) → input → render → `PostUpdate` (FrameEnded).
- `ScriptingEvent` now carries a `name()` accessor and `script_id()` for structured diagnostics.
- `engine::frame::begin_app_frame(...)` and `engine::frame::end_app_frame(...)` are the recommended app-owned frame lifecycle helpers.
- `engine::events::RuntimeEventDispatcher` remains the lower-level lifecycle emitter/drainer behind those helpers and is still available for direct use.
- Listener callbacks receive `&EventEnvelope`, not `&mut Renderer`. Mutate app state you own, then perform renderer/scene mutations from your normal app loop.

Currently emitted by renderer/runtime:

| Producer | Events |
|---|---|
| Legacy renderer frame APIs: `Renderer::render_scene`, `render_scene_headless`, `begin_frame`, `end_frame` | Renderer-owned `LifecycleEvent::FrameStarted`, `LifecycleEvent::FrameEnded` |
| Renderer input bridge | `InputActionEvent` with `Pressed`, `Released`, or `Changed` after `InputSystem::dispatch_frame()` |
| Root app-owned frame helpers: `begin_app_frame`, `end_app_frame` | Caller-owned input action events plus `FrameStarted`/`FrameEnded`; internally use `RuntimeEventDispatcher` over the caller-owned bus |
| Root `engine` runtime | app/project/package/scene lifecycle, package load success/failure, headless shutdown completion, windowed shutdown-requested intent |

Typed but deferred until later system sprints:

| Family | Current status |
|---|---|
| `SceneEvent` | Contract exists; broad scene mutation emission is not wired yet. |
| `AssetEvent::AssetLoading/Ready/Failed/Invalidated` | Contract exists; package load events are wired in the root runtime, broad per-asset async emission is deferred. |
| `PhysicsEvent` | Contract exists. The `physics` crate can translate ray hits and contact records into `EngineEvent::Physics` values and emit contact records into an `EventBus`; renderer/root-runtime live physics scene loading is deferred. |
| `AudioEvent` | Contract exists. `apps/dungeon_dogfood` demonstrates app-owned opt-in audio event emission; root-runtime and editor-wide audio emission are deferred. |
| `ScriptingEvent` | Contract exists. The experimental `scripting` crate can return script-emitted events and script errors with durable `ScriptId` context; app/runtime code is responsible for emitting those values at safe boundaries. Production scripting runtime scheduling and Rust hot-reload are deferred. |

## 4. Code Walkthrough

Snippet Type: Real (renderer compatibility path)
```rust
use renderer::{ActionPhase, EngineEvent, EventEnvelope, EventRecorder, LifecycleEvent, Renderer};

// Compatibility: subscribes to renderer-owned EventBus via renderer.events_mut().
// For app-owned paths, use engine::events::runtime_event_bus() and subscribe on the caller-owned bus.
fn install_event_logging(renderer: &mut Renderer) {
    renderer.set_event_recorder(Some(EventRecorder::bounded(128)));
    renderer.events_mut().subscribe(|event: &EventEnvelope| {
        match &event.event {
            EngineEvent::Lifecycle(LifecycleEvent::FrameStarted) => {
                if let Some(frame) = event.frame {
                    if frame.0 % 120 == 0 {
                        log::debug!("frame {} started at {:?}", frame.0, event.stage);
                    }
                }
            }
            EngineEvent::Input(action)
                if matches!(action.phase, ActionPhase::Pressed | ActionPhase::Released) =>
            {
                log::debug!("action={} phase={:?}", action.action, action.phase);
            }
            _ => {}
        }
        Ok(())
    });
}
```

Apps in this repository demonstrate the same pattern:

- `apps/dungeon_dogfood/src/events.rs`

`apps/dungeon_dogfood` owns one app `EventBus` for input actions, frame lifecycle, and startup audio telemetry. Its audio bridge accepts `&mut EventBus` directly, and its render loop uses `begin_app_frame`/`end_app_frame` before and after rendering with a caller-provided `CameraView`.

The standalone `physics` crate demonstrates the physics event bridge through `RayHit::to_engine_event`, `PhysicsContactRecord::to_engine_event`, `contact_records_to_engine_events`, and `emit_contact_records`. `apps/dungeon_dogfood/src/audio_bridge.rs` demonstrates the audio bridge by mapping app-owned audio outcomes into `EngineEvent::Audio`. These helpers preserve the event crate boundary: `engine_events` does not depend on `physics` or `audio`, while app code can opt in to event emission.

The experimental `scripting` crate follows the same boundary. `ScriptEngine::eval_for_script`, `eval_with_scope_for_script`, and `eval_file_for_script` return collected `ScriptingEvent` values instead of dispatching them internally. Scripts can log through `log_info`, `log_warn`, and `log_error`, and can request narrow event emission with `emit_event(name)` or `emit_event(name, payload)`. Scripts do not receive renderer, scene, Vulkan, physics, audio, editor, or app-owned mutable state bindings by default.

Snippet Type: Real
```rust
use engine::events::runtime_event_bus;
use engine::frame::{begin_app_frame, end_app_frame, FrameClock};
use engine::input::{InputActionEventEmitter, InputSystem};

let mut events = runtime_event_bus();
let mut input = InputSystem::new();
let mut action_events = InputActionEventEmitter::new();
let mut frame_clock = FrameClock::new();

let begin = begin_app_frame(&mut input, &mut action_events, &mut events, &mut frame_clock);
// update app-owned game state and render here
end_app_frame(&mut events, begin.frame.index);
```

## 5. Ordering Rules

Normal renderer frame ordering:

| Order | Stage | Notes |
|---:|---|---|
| 1 | `PreUpdate` | `FrameStarted` is emitted before frame preparation. |
| 2 | Input dispatch | `InputSystem::dispatch_frame()` refreshes snapshots. |
| 3 | `Input` | Action events are emitted from the refreshed snapshot and drained immediately. |
| 4 | App/FPS camera update | Built-in FPS controller reads the same snapshot after input events. |
| 5 | Render | Scene submission and rendergraph execution occur. |
| 6 | `PostUpdate` | `FrameEnded` is emitted after render or skipped-resize completion. |

App-owned frame ordering:

| Order | Stage | Notes |
|---:|---|---|
| 1 | Queue routed platform input | Renderer handles platform side effects; app queues uncaptured input. |
| 2 | Input dispatch | App-owned `InputSystem::dispatch_frame()` refreshes snapshots. |
| 3 | `Input` | App-owned action bridge emits input events; `RuntimeEventDispatcher::drain_input` drains them. |
| 4 | `PreUpdate` | `RuntimeEventDispatcher::frame_started` emits/drains caller-owned `FrameStarted`. |
| 5 | App update/render view build | App mutates its own state and passes render DTOs to renderer. |
| 6 | Renderer no-dispatch render | `render_scene_with_view` renders without app lifecycle emission. |
| 7 | `PostUpdate` | `RuntimeEventDispatcher::frame_ended` emits/drains caller-owned `FrameEnded`. |

Root runtime startup ordering:

| Order | Stage | Event |
|---:|---|---|
| 1 | `Startup` | `AppStarting` |
| 2 | `ProjectLoad` | `ProjectLoading` |
| 3 | `ProjectLoad` | `ProjectLoaded` |
| 4 | `Startup` | `AppStarted` |
| 5 | `ProjectLoad` | `PackageLoading`, then `PackageLoaded` or `PackageFailed` |
| 6 | `SceneLoad` | `SceneLoading`, then `SceneLoaded` |
| 7 | `Shutdown` | `ShutdownCompleted` in headless success, `ShutdownRequested` for window close/Escape intent |

## 6. Mutation Safety

- Listener callbacks must be small and non-blocking.
- Listener callbacks must not expect mutable renderer access.
- If a listener needs to trigger work, record intent into app-owned state and act from the app loop.
- Do not call into Vulkan internals, data caches, or private renderer modules from app listeners.
- Avoid recursive event dispatch from callbacks. The alpha contract assumes staged drain from renderer/runtime/app-loop boundaries.

## 7. Debugging Playbook

- Step 1: install a bounded recorder with `Renderer::set_event_recorder` for legacy renderer-owned events, or use `engine::events::runtime_event_bus()` for app-owned events.
- Step 2: subscribe before the event loop starts.
- Step 3: log only selected event families or stages to avoid frame spam.
- Step 4: on renderer-owned compatibility paths, confirm `renderer.update_input(...)` is called for every `winit` event; on app-owned paths, confirm `route_platform_input_to_app(...)` runs for platform input and `begin_app_frame(...)` runs once per app frame.
- Step 5: confirm the matching render path runs once per frame: `render_scene`/`render_scene_headless` for renderer-owned compatibility, or `render_scene_with_view`/`render_scene_headless_with_view` bracketed by `begin_app_frame`/`end_app_frame` for app-owned loops.

## 8. Cross-Module Links

- Event crate: `src/events/src/lib.rs`
- Root event helpers: `src/events.rs`
- Renderer facade integration: `src/renderer/src/api/renderer.rs`
- Input action bridge: `src/input/src/lib.rs`
- Root runtime lifecycle: `src/runtime.rs`
- Dogfood consumer: `apps/dungeon_dogfood/src/events.rs`

## 9. Standard References

- Rust closures: https://doc.rust-lang.org/book/ch13-01-closures.html
- Rust channels and shared state patterns: https://doc.rust-lang.org/book/ch16-00-concurrency.html

## 10. See Also

- [Input Polling and Layered Dispatch](06-input-polling-and-listeners.md)
- [Runtime Project Launcher](11-runtime-project-launcher.md)
- [App-Owned Loop](15-app-owned-loop.md)
- [Internal Event System and Lifecycle](../internal/10-event-system-and-lifecycle.md)
