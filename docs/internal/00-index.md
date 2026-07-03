# Engine Internals Reference

> All citations trace to source code. Generated from a fresh codebase audit — no legacy docs consulted.

## Audience

Contributors working inside the renderer internals — Vulkan orchestration, data caches, pass execution, and scene flattening. Assumes Rust proficiency and basic Vulkan familiarity.

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
              ├── SkyboxPass
              ├── GeometryPass
              ├── PresentCopyPass
              └── ImguiPass
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

## Key Source Files

| File | Role |
|------|------|
| [`src/renderer/src/api/renderer.rs`](../src/renderer/src/api/renderer.rs) | Public API facade |
| [`src/renderer/src/vulkan/vk_render.rs`](../src/renderer/src/vulkan/vk_render.rs) | Vulkan frame orchestration (~3862 lines) |
| [`src/renderer/src/data/data_cache.rs`](../src/renderer/src/data/data_cache.rs) | Mesh/texture/material caches (~2475 lines) |
| [`src/renderer/src/scene/scene_world.rs`](../src/renderer/src/scene/scene_world.rs) | Scene graph and submission builder |
| [`src/renderer/src/rendergraph/mod.rs`](../src/renderer/src/rendergraph/mod.rs) | Render graph and pass trait |
| [`src/input/src/lib.rs`](../src/input/src/lib.rs) | Input system (single file) |

## Distributed Knowledge

Module-level guides provide subsystem detail:
- Vulkan: [`src/renderer/src/vulkan/AGENTS.md`](../src/renderer/src/vulkan/AGENTS.md)
- Data/caches: [`src/renderer/src/data/AGENTS.md`](../src/renderer/src/data/AGENTS.md)
- Shaders: [`src/renderer/src/shaders/AGENTS.md`](../src/renderer/src/shaders/AGENTS.md)
- Renderer: [`src/renderer/AGENTS.md`](../src/renderer/AGENTS.md)

## See Also

- [API Reference](../api/00-index.md) — public API surface
- [Gap Report](../gap-report.md) — known limitations
