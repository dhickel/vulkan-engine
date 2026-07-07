# Events and Lifecycle

## 1. Purpose & Audience

This chapter documents the alpha event contract for app, tool, and runtime code that needs lifecycle or input-action observations without reaching into renderer internals.

Use events for logging, lightweight telemetry, validation recorders, editor tooling, and safe frame-boundary notifications. Do not use the alpha event bus as a cross-thread job system, a scripting VM, or a replacement for direct scene/game state ownership.

## 2. Where This Fits in Engine Flow

Renderer path:
`winit` event loop -> `Renderer::update_input(...)` -> `Renderer::render_scene(...)` or explicit frame API -> input dispatch -> input events -> frame lifecycle events -> render.

App-owned facade path:
`winit` event loop -> renderer platform routing -> app `InputSystem` dispatch -> app `EventBus` input/lifecycle stages -> app update -> renderer `render_scene_with_view(...)`.

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
- `engine::events::RuntimeEventDispatcher` emits and drains lifecycle stages against a caller-owned bus without hiding raw `EventBus` access.
- Listener callbacks receive `&EventEnvelope`, not `&mut Renderer`. Mutate app state you own, then perform renderer/scene mutations from your normal app loop.

Currently emitted by renderer/runtime:

| Producer | Events |
|---|---|
| Legacy renderer frame APIs: `Renderer::render_scene`, `render_scene_headless`, `begin_frame`, `end_frame` | Renderer-owned `LifecycleEvent::FrameStarted`, `LifecycleEvent::FrameEnded` |
| Renderer input bridge | `InputActionEvent` with `Pressed`, `Released`, or `Changed` after `InputSystem::dispatch_frame()` |
| Root app-owned lifecycle helper | Caller-owned `FrameStarted`/`FrameEnded` through `RuntimeEventDispatcher::frame_started` and `frame_ended` |
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

Snippet Type: Real
```rust
use renderer::{ActionPhase, EngineEvent, EventEnvelope, EventRecorder, LifecycleEvent, Renderer};

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

- `apps/editor/src/events.rs`
- `apps/dungeon_dogfood/src/events.rs`

`apps/dungeon_dogfood` owns one app `EventBus` for input actions, frame lifecycle, and startup audio telemetry. Its audio bridge accepts `&mut EventBus` directly, and its render loop drains input/lifecycle stages through `RuntimeEventDispatcher` before and after rendering with a caller-provided `CameraView`.

The standalone `physics` crate demonstrates the physics event bridge through `RayHit::to_engine_event`, `PhysicsContactRecord::to_engine_event`, `contact_records_to_engine_events`, and `emit_contact_records`. `apps/dungeon_dogfood/src/audio_bridge.rs` demonstrates the audio bridge by mapping app-owned audio outcomes into `EngineEvent::Audio`. These helpers preserve the event crate boundary: `engine_events` does not depend on `physics` or `audio`, while app code can opt in to event emission.

The experimental `scripting` crate follows the same boundary. `ScriptEngine::eval_for_script`, `eval_with_scope_for_script`, and `eval_file_for_script` return collected `ScriptingEvent` values instead of dispatching them internally. Scripts can log through `log_info`, `log_warn`, and `log_error`, and can request narrow event emission with `emit_event(name)` or `emit_event(name, payload)`. Scripts do not receive renderer, scene, Vulkan, physics, audio, editor, or app-owned mutable state bindings by default.

Snippet Type: Real
```rust
use engine::events::{runtime_event_bus, RuntimeEventDispatcher};

let mut events = runtime_event_bus();
let frame_index = 0;

RuntimeEventDispatcher::frame_started(&mut events, frame_index);
RuntimeEventDispatcher::drain_input(&mut events);
// update app-owned game state here
RuntimeEventDispatcher::frame_ended(&mut events, frame_index);
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
- Step 4: confirm `renderer.update_input(...)` is called for every `winit` event.
- Step 5: confirm `render_scene`, `render_scene_headless`, or explicit `begin_frame`/`end_frame` runs once per frame.

## 8. Cross-Module Links

- Event crate: `src/events/src/lib.rs`
- Root event helpers: `src/events.rs`
- Renderer facade integration: `src/renderer/src/api/renderer.rs`
- Input action bridge: `src/input/src/lib.rs`
- Root runtime lifecycle: `src/runtime.rs`
- Editor consumer: `apps/editor/src/events.rs`
- Dogfood consumer: `apps/dungeon_dogfood/src/events.rs`

## 9. Standard References

- Rust closures: https://doc.rust-lang.org/book/ch13-01-closures.html
- Rust channels and shared state patterns: https://doc.rust-lang.org/book/ch16-00-concurrency.html

## 10. See Also

- [Input Polling and Layered Dispatch](06-input-polling-and-listeners.md)
- [Runtime Project Launcher](11-runtime-project-launcher.md)
- [Internal Event System and Lifecycle](../internal/10-event-system-and-lifecycle.md)
