# 02 — Architecture Overview

> Provenance: `CP-02`

Before writing code, understand which crates exist, what they own, and how to navigate the documentation.

## Workspace Crates

The root workspace (`Cargo.toml`) lists these members:

| Crate | Path | Role |
|-------|------|------|
| `engine` | root | Data-driven launcher binary + thin library facade over support crates |
| `renderer` | `src/renderer/` | Vulkan renderer runtime, scene graph, assets, debug UI, captures |
| `input` | `src/input/` | Frame-buffered input stack with layers, actions, and snapshots |
| `engine_events` | `src/events/` | Typed event vocabulary, `EventBus`, staged dispatch, recorders |
| `audio` | `src/audio/` | Alpha audio crate (clip metadata, device-backed playback) |
| `physics` | `src/physics/` | Alpha physics crate (colliders, ray queries, contact records) |
| `scripting` | `src/scripting/` | Alpha scripting crate |
| `launch_shared` | `src/launch_shared/` | Shared launch infrastructure |
| `dungeon_dogfood` | `apps/dungeon_dogfood/` | Dogfood application — real-app proof of the app-owned loop |
| `voxel_demo` | `apps/voxel_demo/` | Configurable procedural-cave application |
| `engine_pack` | `tools/engine_pack/` | Packaging CLI and `new-app` scaffold generator |

> **Important**: Workspace membership does **not** imply production readiness. `audio`, `physics`, `scripting`, and `launch_shared` are alpha-stage crates whose APIs may change without notice.

## The Launcher and Facade

The root `engine` crate serves two roles:

1. **Launcher binary** (`src/main.rs`): Data-driven project runtime. You point it at an `engine.project.toml` and it bootstraps the full engine. This is the path for project-manifest-based workflows.

2. **Library facade** (`src/lib.rs`): Thin re-exports of support-crate types through stable module paths (`engine::camera`, `engine::events`, `engine::frame`, `engine::input`, `engine::render`, `engine::prelude`). Custom apps depend on this facade.

You can also depend on the support crates directly (`renderer`, `input`, `engine_events`) — the facade is a convenience, not a requirement.

## App-Owned vs Renderer-Owned

The central architectural contract:

| Who Owns It | Examples |
|-------------|----------|
| **App** | winit event loop and window, `InputSystem`, `EventBus`, `FrameClock`, `FixedStepClock`, `Camera`, `FPSController`, gameplay state, scene mutation |
| **Renderer** | Vulkan device/swapchain lifecycle, descriptor/pipeline management, frame submission, asset loading/caching, GPU resource retirement, debug UI, platform input side effects, frame capture output |

The app tells the renderer **what** to render by constructing a `CameraView` and calling `render_scene_with_view`. The renderer decides **how** to render it — which pipelines, what order, when to present.

### Renderer Compatibility Path (historical)

Renderer-owned examples (`demo_pbr`, `api_test`, etc.) use a simpler compatibility pattern where the renderer also owns input dispatch and camera state:

```rust
renderer.update_input(&window, &event)?;
renderer.render_scene(&window, &mut scene)?;
```

This pattern is labeled **compatibility/diagnostic** in these guides — it exists for validating the renderer itself, not for building custom applications. Custom apps use the app-owned loop described in [Chapter 04](04-app-owned-loop.md).

## Three-Tier Documentation

When you need more detail than this guide provides:

| Tier | Go To | For |
|------|--------|-----|
| **Guide** | `docs/guide/` | Step-by-step tutorials, concepts, architecture |
| **API Reference** | [`docs/api/00-index.md`](../api/00-index.md) | Function signatures, error variants, configuration fields, feature flags |
| **Internal** | [`docs/internal/00-index.md`](../internal/00-index.md) | Rendergraph pass order, Vulkan descriptor layouts, shader contracts, cache internals |

The guide links to API docs for reference material and to internal docs for maintainer-level explanations. Do not read internal docs unless you are modifying the engine itself.

## Dependency Flow

```
guide_app / your app
    └── engine (root facade)
            ├── renderer ──┬── input (re-exported)
            │              ├── engine_events (re-exported)
            │              └── Vulkan + assets
            ├── input
            ├── engine_events
            ├── audio (alpha)
            ├── physics (alpha)
            └── scripting (alpha)
```

The root `engine` crate depends on the support crates. Your app depends only on `engine` (and `winit`). The renderer depends on `input` and `engine_events` but the root facade hides this — you import from `engine::prelude` and `engine::input`, never from `renderer` directly.

## Renderer Examples vs Your App

| | Renderer Examples | Your App |
|---|---|---|
| **Purpose** | Validate the renderer | Build your application |
| **Input** | Renderer-owned `InputSystem` | App-owned `InputSystem` |
| **Camera** | Renderer-owned `Camera` | App-owned `Camera` + `FPSController` |
| **Frame boundary** | `renderer.render_scene()` | `begin_app_frame` / `render_scene_with_view` / `end_app_frame` |
| **Events** | Renderer-owned `EventBus` | App-owned `EventBus` via `runtime_event_bus()` |
| **Import** | `renderer::prelude::*` | `engine::prelude::*` |
| **Cargo.toml** | In workspace | Depends on `engine = { path = "..." }` |

## Next

[Chapter 03 — Building Your First App](03-building-your-first-app.md) walks through the two-stage onboarding: `engine_pack new-app` for the renderer-free scaffold, then the maintained `examples/guide_app` checkpoint for the complete windowed loop.
