# Quickstart — First Frame

> Source: examples at [`src/renderer/examples/`](../src/renderer/examples/) — no legacy docs consulted.

## 1. Dependencies

```toml
[dependencies]
renderer = { path = "src/renderer" }
winit = "0.30"
```

## 2. Create a Window and Renderer

```rust
use renderer::prelude::{Renderer, RendererConfig, Scene};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::dpi::PhysicalSize;

let event_loop = EventLoop::new()?;
let config = RendererConfig {
    app_name: "My App".into(),
    ..RendererConfig::default()
};

let window = WindowBuilder::new()
    .with_title("My App")
    .with_inner_size(PhysicalSize::new(config.window_width, config.window_height))
    .build(&event_loop)?;

let mut renderer = Renderer::new(config, &window)?;
```

`RendererConfig::default()` provides sensible defaults: 1920×1080, PBR shading mode, validation layers off, startup scene preloaded. See [07-config.md](07-config.md) for all options.

## 3. Get a Scene

The renderer can preload a startup scene (a default PBR environment with a skybox and test geometry):

```rust
let mut scene = renderer.take_startup_scene().unwrap_or_default();
```

Or build your own:

```rust
let mut scene = Scene::new();
let mut assets = renderer.assets();
let fragment = assets.load_model("path/to/model.glb")?;
scene.merge_fragment(None, fragment)?;
```

`take_startup_scene()` consumes the preloaded scene; after that, you manage scenes yourself.

## 4. The Render Loop

Feed window events to the renderer, then render each frame:

```rust
renderer.install_default_fps_input();  // WASD + mouse look

event_loop.run(move |event, control_flow| {
    if let Err(err) = renderer.update_input(&window, &event) {
        eprintln!("Input error: {err}");
        control_flow.exit();
        return;
    }

    match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => control_flow.exit(),
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    control_flow.exit();
                }
            }
            WindowEvent::Resized(size) => {
                renderer.resize(size.width, size.height)?;
            }
            WindowEvent::RedrawRequested => {
                renderer.render_scene(&window, &mut scene)?;
                window.request_redraw();
            }
            _ => {}
        },
        _ => {}
    }
})?;
```

Key points:
- `update_input()` must be called for **every** event — it routes to imgui and the input system
- `install_default_fps_input()` registers WASD movement + mouse look as an input layer
- `render_scene()` is the one-shot convenience; for explicit control use `begin_frame` / `render_scene_in_frame` / `end_frame`

## 5. Full Minimal Example

See [`src/renderer/examples/demo_pbr.rs`](../src/renderer/examples/demo_pbr.rs) for the complete PBR demo. See [`src/renderer/examples/demo_model_load.rs`](../src/renderer/examples/demo_model_load.rs) for a model-loading example that builds its own scene.

The renderer examples use `renderer::prelude` for the alpha beginner facade.
Root-level compatibility exports still exist, but new quickstart-style code
should prefer the prelude unless a chapter explicitly labels an API as
compatibility or advanced.

## 6. Debug UI

Press **F1** to toggle the debug UI overlay (performance graphs, timing). Press **F2** to toggle the in-engine console. These are registered by `update_input()` — if you bypass it, debug toggles won't work.

## See Also

- [02-renderer.md](02-renderer.md) — renderer lifecycle deep dive
- [03-scene.md](03-scene.md) — scene construction
- [04-assets.md](04-assets.md) — loading models and environments
- [src/renderer/examples/common/mod.rs](../src/renderer/examples/common/mod.rs) — shared example harness
