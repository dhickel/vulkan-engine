# 01 - Quickstart: Facade Bootstrap

This chapter gets a new app rendering frames as fast as possible with the current public API.

## Minimum Integration Path

Public crates/types to import:
- `RendererConfig`
- `Renderer`
- `Scene`
- `FrameRenderOutcome`

Minimal flow:
1. Create your `winit` `Window`.
2. Create `Renderer::new(config, &window)`.
3. Pull startup scene via `take_startup_scene()` or create `Scene::new()`.
4. In redraw path, call `renderer.render_scene(&window, &mut scene)` and handle `FrameRenderOutcome`.

Example:
```rust
use renderer::{FrameRenderOutcome, Renderer, RendererConfig, Scene};

let mut config = RendererConfig::default();
config.app_name = "my game".to_string();
config.validation_layer = true; // recommended during development

let mut renderer = Renderer::new(config, &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

match renderer.render_scene(&window, &mut scene)? {
    FrameRenderOutcome::Rendered => {}
    FrameRenderOutcome::SkippedResizePending => {}
}
```

## Recommended Config Defaults (Alpha)

Keep these unless you have a reason to change:
- `validation_layer = true` for development builds.
- `compile_shaders = false` unless you are editing shader source and have toolchain installed.
- `shader_debug_mode = DebugRuntimeMode::Default` for normal behavior.
- `headless = false` (`headless=true` is currently unsupported and returns `RendererError::Unsupported`).

## Event Loop Skeleton

Use `update_input` for all events and wire resize/redraw handlers:
```rust
renderer.update_input(&window, &event)?;

match event {
    WindowEvent::Resized(size) => renderer.resize(size.width, size.height)?,
    WindowEvent::RedrawRequested => {
        match renderer.render_scene(&window, &mut scene)? {
            FrameRenderOutcome::Rendered => {}
            FrameRenderOutcome::SkippedResizePending => {}
        }
    }
    _ => {}
}
```

Reference implementation:
- `src/renderer/examples/api_test.rs`

## First-Day Troubleshooting

- `Renderer::new` fails at startup:
  - Check Vulkan runtime/driver installation.
  - Enable `validation_layer=true` for clearer errors.
- No runtime from `cargo run` at repo root:
  - This is expected. Use `cargo run -p renderer --example ...`.
- Stalls on first heavy load:
  - Expected in alpha for synchronous startup/model loads.

## Learn More

- Lifecycle details: `02_renderer_lifecycle_and_frame_api.md`
- Asset loading: `04_assets_sync_deferred_and_handles.md`
- Vulkan initialization background:
  - https://github.khronos.org/Vulkan-Site/guide/latest/initialization.html
