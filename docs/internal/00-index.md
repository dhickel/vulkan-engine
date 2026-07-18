# Engine Internals Reference

> All citations trace to source code. Generated from a fresh codebase audit — no legacy docs consulted.

## Audience

Contributors working inside the renderer internals — Vulkan orchestration, data caches, pass execution, and scene flattening. Assumes Rust proficiency and basic Vulkan familiarity.

## Workspace Context

Root `Cargo.toml` currently declares `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, and `apps/marching_terrain`. These internals docs primarily cover the renderer/input path; support crates and apps should not be inferred production-ready from workspace membership alone.

## Architecture at a Glance

```
Renderer (public API)
  ├── AssetManager → DataCache (handles, GPU uploads)
  ├── SceneWorld → RenderSubmission (flattened draw commands)
  ├── InputSystem (layered event dispatch)
  ├── Debug UI (imgui)
  └── VkRender (Vulkan frame loop)
        └── RenderGraph (pass orchestration)
              ├── PrepareTargetsPass
              ├── ShadowPass
              ├── SkyboxPass
              ├── GeometryPass
              ├── PresentCopyPass
              ├── ImguiPass
              ├── DebugCapturePass
              └── TerminalPresentPass
```

## Reading Order

| Order | Document | What It Covers |
|-------|----------|----------------|
| 1 | [01-architecture.md](01-architecture.md) | Module map, data flow, subsystem boundaries |
| 2 | [02-renderer-internals.md](02-renderer-internals.md) | API→backend handoff, frame lifecycle, synchronization |
| 3 | [03-asset-pipeline.md](03-asset-pipeline.md) | Disk→GPU asset pipeline, caches, staging |
| 4 | [04-vulkan-subsystem.md](04-vulkan-subsystem.md) | Vulkan init, descriptors, pipelines, memory |
| 5 | [05-scene-internals.md](05-scene-internals.md) | Scene flattening, render submission, culling |
| 6 | [06-input-internals.md](06-input-internals.md) | Input dispatch, priority groups, action resolution |
| 7 | [07-rendergraph.md](07-rendergraph.md) | Pass traits, dependencies, attachment aliasing |
| 8 | [08-shaders.md](08-shaders.md) | Shader contracts, compilation, PBR pipeline |
| 9 | [09-input-winit-integration.md](09-input-winit-integration.md) | Winit ingestion, input dispatch, snapshot bridge |
| 10 | [10-event-system-and-lifecycle.md](10-event-system-and-lifecycle.md) | Event ownership boundaries, emission ordering, validation |
| 11 | [11-physics-and-collision.md](11-physics-and-collision.md) | Physics crate boundaries, collision metadata validation, event bridge |
| 12 | [12-audio-foundation.md](12-audio-foundation.md) | Audio crate boundaries, package/scene validation, event bridge |

## Key Source Files

| File | Role |
|------|------|
| [`src/renderer/src/api/renderer.rs`](../../src/renderer/src/api/renderer.rs) | Public API facade |
| [`src/renderer/src/vulkan/vk_render.rs`](../../src/renderer/src/vulkan/vk_render.rs) | Vulkan frame transactions, rendergraph orchestration, submit/present, and terminal-error classification |
| [`src/renderer/src/data/data_cache.rs`](../../src/renderer/src/data/data_cache.rs) | Mesh/texture/material caches (~2475 lines) |
| [`src/renderer/src/scene/scene_world.rs`](../../src/renderer/src/scene/scene_world.rs) | Scene graph and submission builder |
| [`src/renderer/src/rendergraph/mod.rs`](../../src/renderer/src/rendergraph/mod.rs) | Fixed rendergraph order and pass trait |
| [`src/renderer/src/vulkan/vk_shadow.rs`](../../src/renderer/src/vulkan/vk_shadow.rs) | Frame-local directional shadow resources and light-space fitting |
| [`src/input/src/lib.rs`](../../src/input/src/lib.rs) | Input system (single file) |
| [`src/events/src/lib.rs`](../../src/events/src/lib.rs) | Event contracts, staged bus, recorder |
| [`src/physics/src/lib.rs`](../../src/physics/src/lib.rs) | Renderer-independent alpha physics API, Rapier wrapper, event bridge |
| [`src/audio/src/lib.rs`](../../src/audio/src/lib.rs) | Renderer-independent alpha audio clip/probe/playback facade |

## Distributed Knowledge

Module-level guides provide subsystem detail:
- Vulkan: [`src/renderer/src/vulkan/AGENTS.md`](../../src/renderer/src/vulkan/AGENTS.md)
- Data/caches: [`src/renderer/src/data/AGENTS.md`](../../src/renderer/src/data/AGENTS.md)
- Shaders: [`src/renderer/src/shaders/AGENTS.md`](../../src/renderer/src/shaders/AGENTS.md)
- Renderer: [`src/renderer/AGENTS.md`](../../src/renderer/AGENTS.md)

## See Also

- [API Reference](../api/00-index.md) — public API surface
- [API Events and Lifecycle](../api/12-events-and-lifecycle.md) — public event consumption contract
- [Physics and Collision](11-physics-and-collision.md) — current alpha physics/collision implementation boundary
- [Audio Foundation](12-audio-foundation.md) — current alpha audio implementation boundary
- [Alpha Readiness Baseline](../gap-report.md) — current readiness and residual-classification routing
