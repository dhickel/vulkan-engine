# 04 — The App-Owned Loop

> Provenance: `CP-04` — every full code block matches `examples/guide_app/src/main.rs`

This chapter walks through the checkpoint source section by section. The complete file is at `examples/guide_app/src/main.rs` — every code block here is either the exact source (labeled **Full Match**) or an excerpt with the omitted context named (labeled **Excerpt**).

## Imports and Setup

> Provenance: `CP-04-IMPORTS` — Full Match

```rust
use std::time::Duration;

use engine::input;
use engine::prelude::*;
use engine::render::RendererError;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;

const APP_NAME: &str = "Guide App";
```

Two import paths are used:

- **`engine::prelude::*`**: All common app-owned loop types — `Camera`, `FPSController`, `EventBus`, `runtime_event_bus`, `FrameClock`, `FixedStepClock`, `begin_app_frame`, `end_app_frame`, `InputSystem`, `InputActionEventEmitter`, `Renderer`, `RendererConfig`, `Scene`, `FrameRenderOutcome`, `camera_view_for_size`, `CameraView`, and more. See the full prelude at [`src/lib.rs`](../../src/lib.rs).
- **`engine::input`**: Input-specific constructors not in the prelude — `ActionMap`, `LayerDescriptor`, `LayerPriority`, and `route_platform_input_to_app` (also available as `engine::input::route_platform_input_to_app`).

`RendererError` is imported from `engine::render` because it is not in the prelude. It is needed for matching `DeviceLost` and `BackendPoisoned` variants.

## Platform Window

> Provenance: `CP-04-WINDOW` — Full Match

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // --- Platform window ---
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;
```

The app creates the winit event loop and window. The renderer does not own window creation — the app does. The window handle is passed to `Renderer::new` and to `route_platform_input_to_app` each frame.

## Renderer Initialization

> Provenance: `CP-04-RENDERER` — Full Match

```rust
    // --- Renderer ---
    let config = RendererConfig {
        app_name: "guide_app".to_string(),
        window_width: 1280,
        window_height: 720,
        preload_startup_scene: true,
        ..Default::default()
    };
    let mut renderer = Renderer::new(config, &window)?;
    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
```

`Renderer::new` takes a config and a `&Window`. The config sets the app name (used in debug output and capture paths), the initial viewport size, and enables startup scene preloading. `preload_startup_scene: true` makes the renderer load its built-in debug/demo scene during init — `take_startup_scene()` retrieves it.

## App-Owned Runtime State

> Provenance: `CP-04-STATE` — Full Match

```rust
    // --- App-owned runtime state ---
    let mut events = runtime_event_bus();
    let mut input = InputSystem::new();
    let mut frame_clock = FrameClock::new();
    let mut action_events = InputActionEventEmitter::new();
    let mut fixed_clock = FixedStepClock::new(FixedStepConfig {
        step: Duration::from_secs_f32(1.0 / 60.0),
        max_steps_per_frame: 10,
    });

    let mut camera = Camera::default();
    let mut fps_controller = FPSController::new(0.002, 1.0);
```

Every piece of runtime state lives in the app:

| State | Type | Purpose |
|-------|------|---------|
| `events` | `EventBus` | Lifecycle, input action, and subsystem events. Created with bounded recording via `runtime_event_bus()`. |
| `input` | `InputSystem` | Frame-buffered input with priority layers, action maps, and per-frame snapshots. |
| `frame_clock` | `FrameClock` | Monotonic frame counter and delta-time measurement. Ticked by `begin_app_frame`. |
| `action_events` | `InputActionEventEmitter` | Converts input snapshots into `EngineEvent::Input` events on the bus. |
| `fixed_clock` | `FixedStepClock` | Decouples simulation rate (60 Hz) from display rate. Accumulates real time, produces fixed-step ticks. |
| `camera` | `Camera` | Position, orientation, pitch, yaw. Mutable by the FPS controller. |
| `fps_controller` | `FPSController` | Mouse-look sensitivity + movement speed. Reads `InputSnapshot`, writes to `Camera`. |

For loops that need runtime scale/pause semantics, replace the compatible `FrameClock` + `FixedStepClock` pair with one caller-owned `engine::time::Time` and use `begin_app_frame_with_time`. `Time` preserves the fixed quantum while scaling only accumulator input; it does not become renderer-owned state. This checkpoint intentionally retains the lower-level compatible pair so the documented source remains an exact match for `examples/guide_app/src/main.rs`.

## Input Action Bindings

> Provenance: `CP-04-INPUT` — Full Match

```rust
    // Install FPS action bindings in a named layer.
    {
        let mut map = input::ActionMap::new();
        map.bind_key("move.forward", KeyCode::KeyW);
        map.bind_key("move.backward", KeyCode::KeyS);
        map.bind_key("move.left", KeyCode::KeyA);
        map.bind_key("move.right", KeyCode::KeyD);
        map.bind_key("move.up", KeyCode::Space);
        map.bind_key("move.down", KeyCode::ShiftLeft);

        input.add_layer(
            input::LayerDescriptor::new("guide-fps", input::LayerPriority(10)),
            map.into_layer(),
        );
    }
```

The `InputSystem` uses priority-ordered layers. Each layer maps hardware inputs (keys, mouse buttons) to logical action IDs. The `FPSController` uses the default action names (`move.forward`, `move.backward`, etc.), so binding those actions in any layer makes them available to the controller.

`LayerPriority(10)` is an app-level priority. The renderer uses higher-priority layers for debug UI and lower-priority layers for engine-internal bindings.

## The Event Loop

> Provenance: `CP-04-LOOP-TOP` — Full Match

```rust
    let mut last_window_size = window.inner_size();
    window.request_redraw();

    // --- Event loop ---
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        // Route platform input through renderer, queue uncaptured app input.
        match engine::input::route_platform_input_to_app(&mut renderer, &window, &mut input, &event)
        {
            Ok(_) => {}
            Err(e) => {
                eprintln!("input routing failed: {e}");
                elwt.exit();
                return;
            }
        }
```

### Platform Input Routing

`route_platform_input_to_app` is the critical boundary between renderer-owned and app-owned input. It does two things:

1. Routes the raw `winit::Event` through the renderer so the renderer can handle its own platform side effects (ImGui keyboard/mouse capture, debug UI, cursor confinement, capture hotkeys).
2. For events the renderer did **not** consume, queues them into the app-owned `InputSystem`.

This means the app never calls `renderer.update_input()` — the compatibility helper that would queue input into the renderer's own `InputSystem`. Instead, the app calls the routing function and the renderer gets its side effects while the app gets its input.

The `Ok` arm matches all `RendererInputRouting` results. The `Err` arm covers genuine failures (device lost, backend poisoned) and exits.

## Window Event Dispatch

> Provenance: `CP-04-WINDOW-MATCH` — Full Match

```rust
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        last_window_size = new_size;
                        if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                            eprintln!("resize failed: {e}");
                            elwt.exit();
                        }
                    }

                    WindowEvent::ScaleFactorChanged {
                        mut inner_size_writer,
                        ..
                    } => {
                        let new_size = window.inner_size();
                        if let Err(e) = inner_size_writer.request_inner_size(new_size) {
                            eprintln!("scale-factor size request failed: {e}");
                            elwt.exit();
                            return;
                        }
                        last_window_size = new_size;
                        if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                            eprintln!("resize failed after scale change: {e}");
                            elwt.exit();
                        }
                    }
```

Three window events are handled:

- **`CloseRequested`**: Exit the event loop.
- **`Resized`**: Store the new size and request a renderer swapchain resize. The actual swapchain rebuild happens on the next `RedrawRequested` or the next `render_scene_with_view` call.
- **`ScaleFactorChanged`**: On high-DPI displays, the scale factor change must be acknowledged via `inner_size_writer.request_inner_size()`, and the new physical size must trigger a resize.

## The Redraw Path

> Provenance: `CP-04-REDRAW` — Full Match

```rust
                    WindowEvent::RedrawRequested => {
                        // Catch up to the current window size before rendering.
                        let current_size = window.inner_size();
                        if current_size != last_window_size {
                            last_window_size = current_size;
                            if let Err(e) = renderer.resize(current_size.width, current_size.height)
                            {
                                eprintln!("resize failed while redrawing: {e}");
                                elwt.exit();
                                return;
                            }
                        }
```

Before rendering, the app catches up to any resize that occurred since the last stored size. This guards against platform-specific cases where `Resized` events arrive after `RedrawRequested`.

### Begin App Frame

> Provenance: `CP-04-BEGIN-FRAME` — Full Match

```rust
                        // --- Begin app frame ---
                        let begin = begin_app_frame(
                            &mut input,
                            &mut action_events,
                            &mut events,
                            &mut frame_clock,
                        );
```

`begin_app_frame` is the frame boundary. It:

1. Ticks the `FrameClock` — computes delta time, advances the frame index.
2. Calls `input.dispatch_frame()` — finalizes this frame's input snapshot, resets per-frame transient fields (mouse delta, wheel delta, `just_pressed`/`just_released`).
3. Calls `action_events.emit_from_snapshot()` — emits `EngineEvent::Input` events for pressed, released, and changed actions onto the event bus.
4. Drains the `Input` event stage — dispatches all pending input events to subscribers.
5. Emits and drains `LifecycleEvent::FrameStarted` on the `PreUpdate` stage.

The returned `AppFrameBeginReport` contains the frame info (index, delta), action emit count, and dispatch reports.

### Fixed-Step Update

> Provenance: `CP-04-FIXED-STEP` — Full Match

```rust
                        // --- Fixed-step update ---
                        let fixed_update = fixed_clock.update(begin.frame.delta);
                        if fixed_update.dropped_time > Duration::ZERO {
                            eprintln!(
                                "dropped {:.3}ms of simulation time",
                                fixed_update.dropped_time.as_secs_f64() * 1000.0
                            );
                        }
                        let simulated_seconds = (1.0 / 60.0) * fixed_update.steps as f32;
                        fps_controller.update_from_snapshot(
                            input.snapshot(),
                            simulated_seconds,
                            &mut camera,
                        );
```

The `FixedStepClock` decouples simulation timing from display rate:

- **Accumulates** the frame's real delta time.
- **Produces** zero or more fixed-duration steps (1/60 s each).
- **Caps** catch-up at `max_steps_per_frame` (10), dropping excess time.
- **Preserves** a remainder (`alpha`) for interpolation (not used in this minimal checkpoint but available via `fixed_update.alpha`).

The `FPSController` reads the finalized input snapshot and applies mouse-look and WASD movement to the camera. It is called once per frame with `simulated_seconds` matching the accumulated fixed-step time. The controller uses `Camera::update_rotation` (mouse) and `Camera::update_position` (movement) internally.

### CameraView Construction

> Provenance: `CP-04-CAMERAVIEW` — Full Match

```rust
                        // --- Build CameraView from app-owned camera ---
                        let view =
                            camera_view_for_size(&camera, current_size.width, current_size.height);
```

`camera_view_for_size` constructs a renderer-consumable `CameraView` DTO from the app-owned `Camera` and the current viewport dimensions. It:

- Builds a view matrix from the camera's position and orientation.
- Builds a perspective projection matrix with the current aspect ratio.
- Sanitizes zero-dimension viewports (returns a square aspect when height is 0, treats width 0 as 1).

The resulting `CameraView` is Vulkan-opaque — it contains only `Mat4` matrices and a `Vec3` position. The app does not need to know how the renderer converts it into UBO data.

### Asset Pumping

> Provenance: `CP-04-PUMP` — Full Match

```rust
                        // --- Pump assets and render ---
                        if let Err(e) = renderer.pump_asset_tasks(32) {
                            eprintln!("asset pump failed: {e}");
                            elwt.exit();
                            return;
                        }
```

`pump_asset_tasks` drives async asset loading. The renderer maintains a background thread pool for texture uploads, mesh processing, and environment map loading. Calling `pump_asset_tasks(32)` drains up to 32 completed tasks and advances state machines. Without this call, async loads never complete.

### Render Submission

> Provenance: `CP-04-RENDER` — Full Match

```rust
                        match renderer.render_scene_with_view(&mut scene, view) {
                            Ok(FrameRenderOutcome::Rendered) => {
                                // Normal frame; continue.
                            }
                            Ok(FrameRenderOutcome::SkippedResizePending) => {
                                eprintln!("render skipped while swapchain resize is pending");
                            }
                            Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
                            | Ok(FrameRenderOutcome::SubmittedNotPresented)
                            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {
                                // Transient; continue.
                            }
                            Err(RendererError::DeviceLost) => {
                                eprintln!("Vulkan device lost; exiting");
                                elwt.exit();
                                return;
                            }
                            Err(RendererError::BackendPoisoned(msg)) => {
                                eprintln!("renderer backend poisoned: {msg}");
                                elwt.exit();
                                return;
                            }
                            Err(e) => {
                                eprintln!("render failed: {e}");
                                elwt.exit();
                                return;
                            }
                        }
```

`render_scene_with_view` is the single render call the app makes. It takes the app-owned scene and the app-constructed `CameraView`. The renderer handles everything else: scene CPU work, frustum culling, command recording, submission, and presentation.

Every outcome is handled:

| Outcome | Meaning | Action |
|---------|---------|--------|
| `Rendered` | Frame was recorded, submitted, and presented successfully | Continue |
| `SkippedResizePending` | Swapchain is mid-rebuild; frame was skipped | Log and continue; next frame will retry |
| `SkippedAcquireUnavailable` | Swapchain image acquire returned `VK_NOT_READY` or `VK_TIMEOUT` | Continue; transient WSI state |
| `SubmittedNotPresented` | Frame was submitted but presentation was skipped (headless/capture path) | Continue |
| `PresentedSuboptimal` | Frame presented but swapchain is suboptimal (e.g. surface changed) | Continue; resize will be triggered |
| `DeviceLost` | Vulkan device was lost (GPU hang, driver crash, physical removal) | **Exit** — must destroy and recreate renderer |
| `BackendPoisoned(msg)` | A previous terminal error has poisoned the backend | **Exit** — further operations are unsafe |
| Other `RendererError` | Unexpected failure | **Exit** |

The three transient outcomes (`SkippedAcquireUnavailable`, `SubmittedNotPresented`, `PresentedSuboptimal`) are normal WSI edge cases — they do not indicate a problem with your app or the renderer.

### End App Frame

> Provenance: `CP-04-END-FRAME` — Full Match

```rust
                        // --- End app frame ---
                        end_app_frame(&mut events, begin.frame.index);

                        window.request_redraw();
```

`end_app_frame` emits and drains `LifecycleEvent::FrameEnded` on the `PostUpdate` stage. Subscribers that need to run after rendering (e.g. telemetry, frame capture logging) receive this event with the same frame index that `begin_app_frame` used.

`window.request_redraw()` asks winit to emit another `RedrawRequested` event on the next opportunity. Combined with `ControlFlow::Poll`, this creates a continuous render loop.

### AboutToWait

> Provenance: `CP-04-ABOUT-TO-WAIT` — Full Match (end of event loop)

```rust
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
```

`AboutToWait` fires when the event loop has no more events and is about to sleep. Requesting a redraw here ensures the loop keeps running even if the platform throttles redraw events.

## Summary: The Per-Frame Sequence

```
route_platform_input_to_app()   ← every event
    │
    ├─ CloseRequested           → exit
    ├─ Resized                  → renderer.resize()
    └─ RedrawRequested
        │
        ├─ begin_app_frame()    ← tick clock, dispatch input, emit FrameStarted
        ├─ fixed_clock.update() ← accumulate delta, produce steps
        ├─ fps_controller.update_from_snapshot()
        ├─ camera_view_for_size()
        ├─ renderer.pump_asset_tasks(32)
        ├─ renderer.render_scene_with_view()  ← single render call
        │   └─ handle all FrameRenderOutcome variants + terminal errors
        ├─ end_app_frame()      ← emit FrameEnded
        └─ window.request_redraw()
```

Every type the app touches is app-owned. The renderer is called through exactly two mutating methods per frame (`pump_asset_tasks` and `render_scene_with_view`) plus the event-level `route_platform_input_to_app` and the occasional `resize`.

## Next

[Chapter 05 — Working with the Renderer](05-renderer.md) covers renderer initialization, the startup scene, resize lifecycle, asset pumping details, frame outcomes in depth, and where to find the full API reference.
