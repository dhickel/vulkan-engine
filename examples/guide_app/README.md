# Guide App — Maintained Checkpoint

Executable companion for the `docs/guide/` end-user documentation. This is a nested Cargo workspace; it is **NOT** a member of the root `Cargo.toml`.

## Working Directory

All commands must be run from the repository root (`vulkan-engine/`). The renderer locates shaders, assets, and startup scenes relative to the working directory.

## Locked Commands

```sh
# Format check
cargo fmt --check --manifest-path examples/guide_app/Cargo.toml

# Compile check
cargo check --locked --manifest-path examples/guide_app/Cargo.toml

# Run (requires Vulkan-capable GPU + driver with WSI support)
cargo run --manifest-path examples/guide_app/Cargo.toml

# Run with logging
RUST_LOG=info cargo run --manifest-path examples/guide_app/Cargo.toml
```

## Ownership Map

| Concern | Owner |
|---------|-------|
| winit event loop + window | App |
| `InputSystem` dispatch, action maps, FPS bindings | App |
| `InputActionEventEmitter` action emission | App |
| `EventBus` + lifecycle events (`FrameStarted`, `FrameEnded`) | App |
| `FrameClock` + `FixedStepClock` frame timing | App |
| `Camera` + `FPSController` camera state | App |
| `CameraView` construction via `camera_view_for_size` | App |
| `begin_app_frame` / `end_app_frame` | App |
| Platform input routing | App calls `route_platform_input_to_app` |
| Vulkan device, swapchain, pipeline, descriptor lifecycle | Renderer |
| `render_scene_with_view` frame submission | Renderer |
| Asset loading, texture/mesh caching, GPU resource retirement | Renderer |
| Resize handling (swapchain rebuild) | Renderer called by App via `renderer.resize()` |
| Debug UI, ImGui platform integration | Renderer |
| Frame capture output | Renderer |
| Startup scene geometry, materials, skybox | Renderer (preloaded) |

## Startup Scene

The checkpoint uses the renderer's built-in startup scene accessed via `renderer.take_startup_scene()`. No external asset paths are required.

## WSI Expectation

The checkpoint requires a Vulkan-capable GPU with active windowing system integration (WSI). Headless/offscreen-only Vulkan installations will fail at swapchain creation. For headless validation use the renderer's capture examples (see `docs/api/00-index.md`).

## Building From Clean

This is a nested workspace. Run all commands with `--manifest-path examples/guide_app/Cargo.toml` from the repository root. Do NOT add `examples/guide_app` to the root workspace members.
