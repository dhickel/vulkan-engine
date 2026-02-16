# Renderer Lifecycle and Frame API

## 1. Purpose & Audience
This chapter is for students, hobbyists, and indie developers using the facade renderer API. It explains how to initialize `Renderer`, drive per-frame rendering, and handle frame outcomes safely.

## 2. Where This Fits in Engine Flow
Facade lifecycle path:
`Renderer::new(...)` -> `take_startup_scene()` -> event loop `update_input(...)` -> `render_scene(...)` or explicit `begin_frame(...)` -> `render_scene_in_frame(...)` -> `end_frame(...)`.

## 3. Key Concepts
- `Renderer::new(config, window)` is the canonical initialization entrypoint.
- Root `cargo run` does not start runtime rendering; use `cargo run -p renderer --example ...`.
- Two frame APIs exist:
  - single-call path: `render_scene(...)`
  - explicit frame path: `begin_frame(...)`, `render_scene_in_frame(...)`, `end_frame(...)`
- `FrameRenderOutcome` must be handled exhaustively:
  - `Rendered`
  - `SkippedResizePending`
- Rendering auto-pumps deferred asset work each frame; explicit pumping is available via `pump_asset_tasks(...)`.
- Debug UI is now renderer-managed and can be controlled via:
  - `toggle_debug_ui()`, `set_debug_ui_visible(...)`, `is_debug_ui_visible()`
  - `register_debug_view(...)`, `unregister_debug_view(...)`, `set_debug_view_enabled(...)`
- Default runtime toggle for debug UI visibility is backquote (`\``) in `update_input(...)`.
- Current alpha limitation: headless mode is not implemented (`Renderer::new` returns unsupported when `config.headless = true`).

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
    FrameRenderOutcome::SkippedResizePending => {
        // wait for resize handling and keep loop alive
    }
}
```

Snippet Type: Real
```rust
// Single-call API (src/renderer/src/api/renderer.rs)
let outcome = renderer.render_scene(&window, &mut scene)?;
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

Snippet Type: Pseudocode
```text
App owns Window + Renderer + Scene.
Every loop tick:
  pass winit events into update_input
  if resize event: call renderer.resize
  render one frame (single-call or explicit API)
  handle Rendered vs SkippedResizePending
```

## 5. Best Practices
- Start from `src/renderer/examples/api_test.rs` for canonical ownership and loop structure.
- Handle `FrameRenderOutcome` with a full `match`; do not assume every frame renders.
- Use one frame style per loop (`render_scene` or explicit frame API), not both in the same tick.
- Keep render and load orchestration separate in your app architecture.

## 6. Gotchas & Failure Modes
- Calling `render_scene(...)` while an explicit frame is open returns `RendererError::InvalidState`.
- Calling `begin_frame(...)` twice without `end_frame(...)` returns invalid state.
- Calling `render_scene_in_frame(...)` twice for one `FrameContext` is invalid.
- Resize flow can produce repeated `SkippedResizePending` outcomes until resize is serviced.
- Expecting runtime output from root `cargo run` is a common migration trap.

## 7. Debugging Playbook
- Step 1: reproduce with `cargo run -p renderer --example api_test`.
- Step 2: verify call order (`begin_frame` -> `render_scene_in_frame` -> `end_frame`) if using explicit API.
- Step 3: inspect resize path; check whether frames are intentionally skipped due to `resize_requested`.
- Step 4: if init fails and shader compile is enabled, verify shader toolchain availability (`glslc` or `glslangValidator`).
- Step 5: if frame calls fail, print `RendererError` variant to separate init/frame/scene/asset errors.

## 8. Cross-Module Links
- Facade API surface: `src/renderer/src/api/mod.rs`
- Renderer lifecycle implementation: `src/renderer/src/api/renderer.rs`
- Debug UI manager internals: `src/renderer/src/debug_ui/mod.rs`
- Canonical runtime usage: `src/renderer/examples/api_test.rs`
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
