# 16 — API Compatibility Guide

> Provenance: `G-16` — derived from `src/renderer/src/api/`, `src/lib.rs`, `engine::prelude`, renderer examples, and DECISION-20260707-06

This chapter provides side-by-side ownership tables and focused code excerpts distinguishing the two supported API paths. Every public API described here is **supported** — the labeling distinguishes their intended use, not their stability or deprecation status.

> **Policy (DECISION-20260707-06)**: All renderer-owned compatibility APIs are documented with explicit labels. No APIs are removed or deprecated. App-owned guidance leads with the current path.

## Quick Decision Table

| You want to... | Use this path |
|----------------|---------------|
| Build a custom app with your own input, camera, and event system | **App-owned** (`engine::prelude`) |
| Validate the renderer, test a Vulkan feature, or run a diagnostic capture | **Renderer compatibility** (`renderer::prelude`) |
| Write a quick smoke test, prototype, or example | Either — the compatibility path is simpler but less flexible |

## Ownership Comparison

### App-Owned Path (Recommended for Custom Apps)

> Provenance: `G-16-APP` — from `examples/guide_app/src/main.rs`, `engine::prelude`, `engine::input`, `engine::render`

| Concern | Owner | Key Type / Function |
|---------|-------|---------------------|
| winit event loop + window | **App** | `EventLoop::new()`, `WindowBuilder` |
| `InputSystem` (layers, action bindings, dispatch) | **App** | `engine::prelude::InputSystem` |
| `InputActionEventEmitter` | **App** | `engine::prelude::InputActionEventEmitter` |
| `EventBus` + lifecycle events | **App** | `engine::prelude::runtime_event_bus()` |
| `FrameClock` + `FixedStepClock` | **App** | `engine::prelude::{FrameClock, FixedStepClock}` |
| `Camera` + `FPSController` | **App** | `engine::prelude::{Camera, FPSController}` |
| `CameraView` construction | **App** | `engine::render::camera_view_for_size()` |
| Platform input routing | **App calls**, Renderer handles side effects | `engine::input::route_platform_input_to_app()` |
| Frame boundary | **App** | `engine::frame::begin_app_frame()` / `end_app_frame()` |
| Render submission | **App** calls, Renderer executes | `renderer.render_scene_with_view(&mut scene, view)` |
| Vulkan device, swapchain, pipelines | **Renderer** | `Renderer::new()` |
| Asset loading, caching, GPU retirement | **Renderer** | `renderer.assets()`, `renderer.pump_asset_tasks()` |
| Resize handling | **App** detects, Renderer executes | `renderer.resize(width, height)` |
| Debug UI, ImGui platform integration | **Renderer** | Automatic when windowed |
| Frame capture output | **Renderer** | `renderer.queue_manual_frame_capture()` |
| Headless rendering | **Renderer** | `Renderer::new_headless()`, `render_scene_headless_with_view()` |

**Import**:

```rust
use engine::prelude::*;          // common app-owned types
use engine::input;                // ActionMap, LayerDescriptor, route_platform_input_to_app
use engine::render::RendererError; // DeviceLost, BackendPoisoned matching
```

**Minimal windowed loop**:

> Provenance: `G-16-APP-LOOP` — Excerpt from `examples/guide_app/src/main.rs`

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("My App")
        .build(&event_loop)?;

    let mut renderer = Renderer::new(RendererConfig::default(), &window)?;
    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

    // App-owned state
    let mut events = runtime_event_bus();
    let mut input = InputSystem::new();
    let mut frame_clock = FrameClock::new();
    let mut action_events = InputActionEventEmitter::new();
    let mut camera = Camera::default();
    let mut fps_controller = FPSController::new(0.002, 1.0);

    // ... install input bindings, then:
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        engine::input::route_platform_input_to_app(
            &mut renderer, &window, &mut input, &event,
        ).unwrap();

        if let Event::WindowEvent {
            event: WindowEvent::RedrawRequested, ..
        } = &event {
            let begin = begin_app_frame(&mut input, &mut action_events,
                                        &mut events, &mut frame_clock);

            fps_controller.update_from_snapshot(input.snapshot(), begin.frame.delta, &mut camera);
            let view = camera_view_for_size(&camera, 1280, 720);

            renderer.pump_asset_tasks(32).unwrap();
            renderer.render_scene_with_view(&mut scene, view).unwrap();

            end_app_frame(&mut events, begin.frame.index);
            window.request_redraw();
        }
    })?;
    Ok(())
}
```

### Renderer Compatibility Path (Diagnostic / Historical)

> Provenance: `G-16-COMPAT` — from `src/renderer/examples/common/mod.rs`, `src/renderer/examples/demo_pbr.rs`, `renderer::prelude`

| Concern | Owner | Key Type / Function |
|---------|-------|---------------------|
| winit event loop + window | **Example** (not app) | `EventLoop::new()`, `WindowBuilder` |
| `InputSystem` (layers, action bindings, dispatch) | **Renderer** | `renderer.update_input(&window, &event)` |
| `Camera` state | **Renderer** | Built-in camera, modified by `Renderer::install_default_fps_input()` |
| `CameraView` construction | **Renderer** | Implicit, built from internal camera |
| Frame boundary | **Renderer** | `renderer.render_scene(&window, &mut scene)` |
| `EventBus` | **Renderer** | Accessible via `renderer.events_mut()` |
| `FrameClock` | **Renderer** | Internal, not exposed |

**Import**:

```rust
use renderer::prelude::{Renderer, RendererConfig, RendererError, Scene};
```

**Minimal example loop**:

> Provenance: `G-16-COMPAT-LOOP` — Simplified from `src/renderer/examples/demo_pbr.rs`

```rust
let mut renderer = Renderer::new(config, &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_default();

event_loop.run(move |event, control_flow| {
    renderer.update_input(&window, &event)?;
    match event {
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
            renderer.render_scene(&window, &mut scene)?;
            window.request_redraw();
        }
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            control_flow.exit();
        }
        _ => {}
    }
})?;
```

> **Compatibility label**: This is a **renderer-owned compatibility/diagnostic** example. It uses the renderer's own input and camera state — the pattern used by renderer-internal tests and validation examples. Custom apps should prefer the app-owned path.

## API-by-API Migration Reference

### Input

| Renderer Compatibility | App-Owned | Notes |
|------------------------|-----------|-------|
| `renderer.update_input(&window, &event)` | `engine::input::route_platform_input_to_app(&mut renderer, &window, &mut app_input, &event)` | Compatibility path queues input into the renderer's `InputSystem`. App path routes platform events through the renderer for side effects, then queues unconsumed events into the app's `InputSystem`. |
| `renderer.install_default_fps_input()` | `app_input.add_layer(...)` with `ActionMap` bindings | Compatibility path installs a default FPS binding layer into the renderer. App path creates its own bindings in its own layers. |
| `renderer.input_snapshot()` | `app_input.snapshot()` | Compatibility path reads renderer-owned snapshot. App path reads app-owned snapshot. |

### Camera

| Renderer Compatibility | App-Owned | Notes |
|------------------------|-----------|-------|
| (Internal, not exposed) | `engine::prelude::Camera` + `FPSController` | Compatibility path owns the camera internally. App path creates and mutates its own camera. |
| (Internal) | `engine::render::camera_view_for_size(&camera, w, h)` | Compatibility path builds the `CameraView` from its internal camera. App path constructs it explicitly. |

### Render Submission

| Renderer Compatibility | App-Owned | Notes |
|------------------------|-----------|-------|
| `renderer.render_scene(&window, &mut scene)` | `renderer.render_scene_with_view(&mut scene, view)` | Compatibility path uses the renderer's internal camera/view. App path provides its own `CameraView`. |
| (Not available) | `renderer.render_scene_headless_with_view(&mut scene, view)` | Headless rendering with app-owned view. Compatibility examples use `--headless` flag which internally uses the headless path. |

### Events

| Renderer Compatibility | App-Owned | Notes |
|------------------------|-----------|-------|
| `renderer.events_mut()` | `engine::events::runtime_event_bus()` | Compatibility path borrows the renderer's event bus. App path creates its own. |
| (Not available) | `engine::frame::begin_app_frame()` / `end_app_frame()` | App path owns the frame boundary and lifecycle event emission. Compatibility path has no equivalent. |

### Frame Clock

| Renderer Compatibility | App-Owned | Notes |
|------------------------|-----------|-------|
| (Not exposed) | `engine::prelude::{FrameClock, FixedStepClock}` | Compatibility path does not expose frame timing. App path uses `FrameClock` for delta time and `FixedStepClock` for decoupled simulation. |

## Renderer Examples: Diagnostic Entry Points

Renderer examples serve specific diagnostic purposes:

| Example | Purpose | Compatibility Label |
|---------|---------|---------------------|
| `demo_pbr` | PBR material rendering validation | Renderer compatibility — validates renderer PBR pipeline |
| `demo_unlit` | Unlit rendering path validation | Renderer compatibility — validates debug/unlit shader variant |
| `demo_model_load` | Model loading (glTF via assimp) | Renderer compatibility — validates asset import pipeline |
| `demo_async_loading` | Async asset loading + progress | Renderer compatibility — validates deferred loading path |
| `api_test` | General API smoke test | Renderer compatibility — validates Renderer/Scene/Asset APIs |
| `capture_culling` | Frustum culling validation (headless) | Renderer diagnostic — `--headless --culling=on` |
| `capture_shadows` | Shadow map validation (headless) | Renderer diagnostic — `--headless` |

These examples are **not** recommended starting points for custom applications. They exist to validate the renderer itself. Use `engine_pack new-app` and the app-owned path for custom apps.

### Running Renderer Examples

```sh
# All run from repository root
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading
cargo run -p renderer --example api_test

# With custom environment map
cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
```

## Headless Capture: Both Paths

> Provenance: `G-16-HEADLESS` — from `AGENTS.md` headless smoke patterns and renderer capture examples

Both paths support headless capture. The key difference is how the `CameraView` is provided:

### Renderer Compatibility Headless

```sh
# Renderer validates itself; uses internal camera
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless
cargo run -p renderer --example capture_culling -- --headless --culling=on
cargo run -p renderer --example capture_shadows -- --headless
```

### App-Owned Headless

```sh
# Dogfood headless: app owns camera, provides CameraView
cargo run -p dungeon_dogfood -- \
  --headless --capture_target draw --capture_frames 3 \
  --capture_frame_start 5 --capture_frame_interval 5

# Voxel demo headless: app owns camera, provides CameraView
cargo run -p voxel_demo -- \
  --preset default --seed 77 --headless \
  --capture-dir .internal-dev/captures/voxel-demo-default-77
```

## Prelude Comparison

| Import | Contains | Intended For |
|--------|----------|--------------|
| `renderer::prelude::*` | `Renderer`, `RendererConfig`, `Scene`, `AssetManager`, `LoadTicket`, `InputSystem`, `FrameCaptureRequest`, `EventBus`, `Camera`, `FPSController`, `ProceduralMeshData`, `PbrMaterialDesc`, `PointLight`, `DirectionalLight`, ... | Renderer-owned examples and diagnostics |
| `engine::prelude::*` | `Renderer` (re-export), `RendererConfig`, `Scene`, `Camera`, `FPSController`, `EventBus`, `runtime_event_bus`, `FrameClock`, `FixedStepClock`, `begin_app_frame`, `end_app_frame`, `InputSystem`, `InputActionEventEmitter`, `render_scene_with_view`, `camera_view_for_size`, `CameraView`, `FrameRenderOutcome`, ... | Custom app-owned loops |

## Deferred Gaps

These features are not yet available in either path:

- Larger project runtime (campaign, save/load)
- Material override system
- Custom rendergraph pass registration
- Generated app templates with renderer-window integration
- Production audio mixing/spatialization/streaming
- Runtime scene-to-physics loading
- Editor collision/audio authoring UI

## Next

[Chapter 17 — Troubleshooting](17-troubleshooting.md) covers common errors, Vulkan setup issues, diagnostic workflows, and when to use headless vs WSI evidence.
