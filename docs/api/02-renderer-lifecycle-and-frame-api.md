# Renderer Lifecycle and Frame API

## 1. Purpose & Audience
This chapter is for students, hobbyists, and indie developers using the facade renderer API. It explains how to initialize `Renderer`, drive per-frame rendering, and handle frame outcomes safely.

## 2. Where This Fits in Engine Flow

> **renderer compatibility path** (good for demos, examples, and smoke testing):
`Renderer::new(...)` -> `take_startup_scene()` -> event loop `update_input(...)` -> `render_scene(...)` or explicit `begin_frame(...)` -> `render_scene_in_frame(...)` -> `end_frame(...)`.

> **current app-owned path** (recommended for custom apps):
`Renderer::new(...)` -> event loop `route_platform_input(...)` -> app update/camera -> `render_scene_with_view(...)`. See [15-app-owned-loop.md](15-app-owned-loop.md).

## 3. Key Concepts
- `Renderer::new(config, window)` is the canonical initialization entrypoint.
- Data-driven projects run through the root launcher: `cargo run -- --project apps/dungeon_dogfood/engine.project.toml`.
- Renderer examples remain facade lifecycle diagnostics: `cargo run -p renderer --example api_test`.
- Custom Rust apps run through their app crate: `cargo run -p <app>`.
- `take_startup_scene()` transfers the optional preloaded startup scene out of the renderer once; after that, the app owns scene construction and mutation.
- Two frame APIs exist (renderer compatibility path):
  - single-call path: `render_scene(...)`
  - explicit frame path: `begin_frame(...)`, `render_scene_in_frame(...)`, `end_frame(...)`
- App-owned camera/render paths can use `render_scene_with_view(...)` or
  `render_scene_headless_with_view(...)` to render with a caller-provided `CameraView` without
  dispatching renderer-owned input/camera/lifecycle state.
- `FrameRenderOutcome` must be handled exhaustively:
  - `Rendered`
  - `SkippedAcquireUnavailable`
  - `SkippedResizePending`
  - `SubmittedNotPresented`
  - `PresentedSuboptimal`
- `RendererError::DeviceLost` is terminal: destroy and recreate the renderer. A renderer whose
  backend previously panicked or returned a terminal Vulkan failure rejects later backend work with
  `RendererError::BackendPoisoned`.
- Rendering auto-pumps deferred asset work each frame; explicit pumping is available via `pump_asset_tasks(...)`.
- Resize handling is explicit: call `resize(width, height)` from window resize events, and use `resize_requested()` to observe a pending swapchain rebuild request.
- Render hooks are safe facade callbacks registered with `set_pre_render_hook(...)` and `set_post_render_hook(...)`; they do not expose raw Vulkan handles and hook failures are logged without failing the frame. See [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md).
- Debug UI is now renderer-managed and can be controlled via:
  - `toggle_debug_ui()`, `set_debug_ui_visible(...)`, `is_debug_ui_visible()`
  - `register_debug_view(...)`, `unregister_debug_view(...)`, `set_debug_view_enabled(...)`
- App-owned imgui chrome can be registered with `register_app_ui(...)` for native tools that
  need always-visible panels instead of optional debug windows. If an app intercepts its UI hotkey
  before renderer input routing, call `refresh_cursor_capture(window)` after callback registration
  changes so cursor confinement/visibility updates immediately.
- `update_input(...)` remains the compatibility path that handles renderer platform side effects and
  queues into renderer-owned input. Apps that own input dispatch can call `route_platform_input(...)`
  and then `engine::input::queue_routed_input_event(...)` for uncaptured gameplay/app input.
- Legacy renderer frame APIs emit renderer-owned lifecycle events. Apps using app-owned input/camera
  and `render_scene_with_view(...)` should prefer `engine::frame::begin_app_frame(...)` and
  `engine::frame::end_app_frame(...)` over relying on renderer-owned lifecycle state.
- `camera_position()`, `set_camera_position(...)`, and `set_camera_look_at(...)` operate on the renderer-owned compatibility camera. Custom apps should prefer app-owned camera state and caller-provided `CameraView` values.
- `environment_runtime_status()` reports requested, active, and transitioning environment-map runtime state for diagnostics and debug UI.
- Default runtime toggle for debug UI visibility is backquote (`\``) in `update_input(...)`.
- Headless validation uses `Renderer::new_headless(config)` and `render_scene_headless(...)`; do not set `config.headless = true` on the windowed `Renderer::new(config, window)` path.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/examples/api_test.rs (facade bootstrap)
let mut renderer = Renderer::new(config.clone(), &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
```

Snippet Type: Real
```rust
// Explicit frame API (src/renderer/examples/api_test.rs)
let mut frame = renderer.begin_frame(&window)?;
let outcome = renderer.render_scene_in_frame(&mut frame, &mut scene)?;
renderer.end_frame(frame)?;

match outcome {
    FrameRenderOutcome::Rendered => {
        // normal frame bookkeeping
    }
    FrameRenderOutcome::SkippedAcquireUnavailable => {
        // no image became available within the bounded acquire budget; retry next frame
    }
    FrameRenderOutcome::SkippedResizePending => {
        // wait for resize handling and keep loop alive
    }
    FrameRenderOutcome::SubmittedNotPresented => {
        // GPU submission completed, but the out-of-date swapchain was not presented
    }
    FrameRenderOutcome::PresentedSuboptimal => {
        // frame was presented; service the pending swapchain resize/rebuild
    }
}
```

Snippet Type: Real (renderer compatibility path)
```rust
// Single-call API (src/renderer/src/api/renderer.rs)
let outcome = renderer.render_scene(&window, &mut scene)?;
```

Snippet Type: Real
```rust
// Window resize path (src/renderer/src/api/renderer.rs)
renderer.resize(size.width, size.height)?;
if renderer.resize_requested() {
    // keep the event loop alive; a later frame or resize call will complete rebuild work
}
```

Snippet Type: Real
```rust
// Render hook registration API
renderer.set_pre_render_hook(Some(renderer::boxed_render_hook(|ctx| {
    log::trace!("pre-render frame {}", ctx.frame_index);
    Ok(())
})));
renderer.set_post_render_hook(None);
```

Snippet Type: Real
```rust
// Debug view registration API
let view_id = renderer.register_debug_view(
    renderer::DebugViewDescriptor::new("app.stats", "App Stats"),
    Box::new(|ui, ctx| {
        ui.window("App Stats").build(|| {
            ui.text(format!("frame {}", ctx.frame_index));
        });
    }),
)?;
renderer.set_debug_view_enabled(&view_id, true);
```

Snippet Type: Real
```rust
// App UI registration API
renderer.register_app_ui(
    "editor.workspace",
    Box::new(|ui, ctx| {
        ui.window("Viewport").build(|| {
            ui.text(format!("frame {}", ctx.frame_index));
        });
    }),
)?;
```

Snippet Type: Pseudocode
```text
App owns Window + Renderer + Scene.
Every loop tick:
  pass winit events into update_input
  or route_platform_input + queue_routed_input_event for app-owned input
  if resize event: call renderer.resize
  render one frame (single-call, explicit API, or caller-view API)
  handle Rendered vs transient acquire skips vs SkippedResizePending
```

## 5. Best Practices
- Start from `src/renderer/examples/api_test.rs` for canonical ownership and loop structure.
- Handle `FrameRenderOutcome` with a full `match`; do not assume every frame renders.
- Use one frame style per loop (`render_scene` or explicit frame API), not both in the same tick.
- For custom game/app loops that own camera, input, and lifecycle events, prefer the caller-view
  render APIs plus `begin_app_frame`/`end_app_frame` on the app-owned bus.
- Keep render and load orchestration separate in your app architecture.
- Use `register_app_ui(...)` for always-present app chrome; use debug views for optional runtime
  diagnostics.

## 6. Gotchas & Failure Modes
- Calling `render_scene(...)` while an explicit frame is open returns `RendererError::InvalidState`.
- Calling `begin_frame(...)` twice without `end_frame(...)` returns invalid state.
- Calling `render_scene_in_frame(...)` twice for one `FrameContext` is invalid.
- Calling `resize(...)` while an explicit frame is open returns `RendererError::InvalidState`.
- `SkippedAcquireUnavailable` is transient and does not request a resize or rebuild; keep the loop alive and retry a later frame.
- Resize flow can produce repeated `SkippedResizePending` outcomes until resize is serviced.
  Zero requested or capability-selected extents remain pending without Vulkan swapchain creation.
  `RendererFrameError::Resize` means a transient zero/empty capability state was detected before
  retirement and the existing generation is safe to retry; device loss, backend poisoning,
  compatibility violations, and other render errors remain terminal.
- `SubmittedNotPresented` means queue submission succeeded but presentation did not reach the
  presentation engine because the swapchain was out of date.
- `PresentedSuboptimal` means the frame reached presentation, but image acquisition or presentation
  reported a suboptimal swapchain and requested a rebuild.
- After `DeviceLost`, or after a backend panic caught by host code, do not retry on the same
  renderer. Later backend operations return `BackendPoisoned`; recreate the renderer.
- Running the wrong target is a common startup trap: use root `cargo run -- --project ...` for project manifests, renderer examples for facade diagnostics, and app crates for custom Rust behavior.
- Registering app UI marks imgui/app chrome as active for input capture, cursor release, and
  built-in FPS-controller suppression.
- `take_startup_scene()` is one-shot; subsequent calls return `None`.

## 7. Debugging Playbook
- Step 1: choose the right runtime path: `cargo run -- --project apps/dungeon_dogfood/engine.project.toml` for project launcher issues, or `cargo run -p renderer --example api_test` for renderer facade diagnostics.
- Step 2: verify call order (`begin_frame` -> `render_scene_in_frame` -> `end_frame`) if using explicit API.
- Step 3: inspect resize path; check whether frames are intentionally skipped due to `resize_requested`.
- Step 4: if init fails and shader compile is enabled, verify shader toolchain availability (`glslc` or `glslangValidator`).
- Step 5: if frame calls fail, print `RendererError` variant to separate init/frame/scene/asset errors.
  Treat `DeviceLost` and `BackendPoisoned` as terminal for that renderer instance.

## 8. Cross-Module Links
- Facade API surface: `src/renderer/src/api/mod.rs`
- Renderer lifecycle implementation: `src/renderer/src/api/renderer.rs`
- Render hook contract: `docs/api/05-render-hooks-and-extension-points.md`
- Debug UI manager internals: `src/renderer/src/debug_ui/mod.rs`
- Renderer facade diagnostic example: `src/renderer/examples/api_test.rs`
- Runtime launcher docs: `docs/api/11-runtime-project-launcher.md`
- Internal render path mental model: `docs/internal/01-rendering-pipeline-mental-model.md`

## 9. Standard References
- Vulkan initialization guide: https://github.khronos.org/Vulkan-Site/guide/latest/initialization.html
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/
- winit docs: https://docs.rs/winit/latest/winit/
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
