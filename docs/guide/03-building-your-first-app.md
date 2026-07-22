# 03 — Building Your First App

> Provenance: `CP-03`

The engine provides a two-stage onboarding path:

1. **`engine_pack new-app`**: A renderer-free scaffold that compiles and runs with zero GPU dependencies
2. **`examples/guide_app`**: The maintained checkpoint that adds the renderer and a full windowed loop

## Stage 1: The Renderer-Free Scaffold

`engine_pack new-app` generates a standalone Rust application that depends on the engine's public support crates (`engine_events`, `input`, `physics`) but **not** the renderer. This gives you a compile-and-run starting point without requiring a GPU or Vulkan SDK:

```sh
cargo run -p engine_pack -- new-app /tmp/my_engine_app --id my_app --name "My Engine App"
```

This creates:

```
/tmp/my_engine_app/
├── Cargo.toml
├── README.md
└── src/
    └── main.rs
```

The generated `Cargo.toml` references the engine crates by relative path. It is a standalone Cargo project — **not** a workspace member:

```toml
[package]
name = "my_app"
version = "0.1.0"
edition = "2021"

[dependencies]
engine_events = { path = "/path/to/vulkan-engine/src/events" }
input = { path = "/path/to/vulkan-engine/src/input" }
physics = { path = "/path/to/vulkan-engine/src/physics" }
```

Verify it compiles and runs:

```sh
cd /tmp/my_engine_app
cargo check
cargo run
```

Expected output: a line showing the app initialized with one pending lifecycle event and a physics world with default gravity.

### Why No Renderer?

The scaffold is intentionally renderer-free because:

- You can verify your Rust toolchain and the engine's support crates work without installing Vulkan
- You can write event-driven, input-aware, physics-backed logic before adding graphics
- The `new-app` template stays fast to generate and fast to compile

The scaffold is a valid first step. When you are ready to add a window and rendering, continue to Stage 2.

## Stage 2: The Maintained Checkpoint

The `examples/guide_app/` directory contains a minimal complete app that owns a winit window, an input system, event bus, frame clock, camera, FPS controller, and the rendering loop. It is the executable specification for the concepts in [Chapter 04](04-app-owned-loop.md) and [Chapter 05](05-renderer.md).

### Checkpoint Manifest

> Provenance: `CP-03-MANIFEST` — matches `examples/guide_app/Cargo.toml`

```toml
[package]
name = "guide_app"
version = "0.1.0"
edition = "2021"
publish = false

[workspace]
resolver = "2"

[dependencies]
engine = { path = "../.." }
env_logger = "0.11"
log = "0.4"
winit = "0.29"

[patch.crates-io]
imgui-rs-vulkan-renderer = { git = "https://github.com/dhickel/imgui-rs-vulkan-renderer", branch = "dev" }
```

Key points:

- **`[workspace]` with `resolver = "2"`**: This makes `guide_app` a nested workspace. It is **not** added to the root `Cargo.toml` members — it compiles independently.
- **`engine = { path = "../.." }`**: Depends on the root `engine` crate (which transitively brings in `renderer`, `input`, `engine_events`).
- **`winit = "0.29"`**: The windowing library. Version 0.29 is pinned to match the renderer's winit dependency.
- **`[patch.crates-io]`**: Required to match the root workspace's imgui renderer patch. Without this, the nested workspace resolves a different version.
- **`Cargo.lock` is source-controlled**: The lockfile in `examples/guide_app/Cargo.lock` ensures reproducible builds.

### Run the Checkpoint

```sh
# From the repository root
cargo run --manifest-path examples/guide_app/Cargo.toml
```

Or with logging:

```sh
RUST_LOG=info cargo run --manifest-path examples/guide_app/Cargo.toml
```

> **Working directory must be the repository root.** The renderer locates shaders, assets, and startup scenes relative to the current directory. Running from any other directory will fail with asset-not-found errors.

Expected: A 1280×720 window opens showing the engine's built-in startup scene. Use **WASD** to move, **Space/Shift** for up/down, and the **mouse** to look around.

### What the Checkpoint Owns

| Concern | Owner |
|---------|-------|
| winit event loop + window | App |
| `InputSystem` (layers, action bindings, dispatch) | App |
| `InputActionEventEmitter` | App |
| `EventBus` + lifecycle events (`FrameStarted`/`FrameEnded`) | App |
| `FrameClock` + `FixedStepClock` | App |
| `Camera` + `FPSController` | App |
| `CameraView` construction via `camera_view_for_size` | App |
| `begin_app_frame` / `end_app_frame` boundary | App |
| Platform input routing via `route_platform_input_to_app` | App calls, Renderer handles side effects |
| Vulkan device, swapchain, pipeline, descriptor lifecycle | Renderer |
| `render_scene_with_view` frame submission | Renderer |
| Asset loading, texture/mesh caching, GPU retirement | Renderer |
| Resize (swapchain rebuild) | Renderer, called by App via `renderer.resize()` |
| Debug UI, ImGui platform integration | Renderer |
| Frame capture output | Renderer |

### Startup Scene

The checkpoint uses the renderer's built-in startup scene (`renderer.take_startup_scene()`). This scene is preloaded during renderer initialization when `preload_startup_scene` is `true` (the default). No external asset paths are required — everything ships inside the renderer crate.

If `take_startup_scene()` returns `None` (unusual but possible if preloading was disabled), the checkpoint falls back to an empty scene: `Scene::new`.

## Next

[Chapter 04 — The App-Owned Loop](04-app-owned-loop.md) walks through every section of the checkpoint source with inline explanations.
