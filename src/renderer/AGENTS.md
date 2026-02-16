# Renderer Package Agent Guide (`src/renderer`)

Use this guide for package-level renderer work. Use child module guides for subsystem implementation details.

## Package Role

`renderer` owns:

- public facade API (`src/renderer/src/api/`)
- runtime examples (`src/renderer/examples/`)
- scene ownership and submission build
- Vulkan frame/render orchestration
- rendergraph pass execution

Primary entrypoint type exports are in `src/renderer/src/lib.rs` and `src/renderer/src/api/mod.rs`.

## Current Runtime Path

1. Example constructs `Renderer`.
2. App updates input and scene each frame.
3. Scene emits `RenderSubmission`.
4. `VkRender` executes rendergraph and submits/presents.

## Documentation Routing

- API index: `docs/api/00-index.md`
- Internal index: `docs/internal/00-index.md`
- Facade lifecycle and frame API: `docs/api/02-renderer-lifecycle-and-frame-api.md`
- Scene workflows: `docs/api/03-scene-graph-and-fragment-workflows.md`
- Asset workflows: `docs/api/04-assets-sync-deferred-and-handles.md`
- Render hooks: `docs/api/05-render-hooks-and-extension-points.md`
- API-to-backend handoff: `docs/internal/04-api-to-backend-handoff.md`
- Rendergraph internals: `docs/internal/07-rendergraph-dependencies-and-aliasing.md`

Module guides:

- Data/cache/scene internals: `src/renderer/src/data/AGENTS.md`
- Vulkan internals: `src/renderer/src/vulkan/AGENTS.md`
- Shader lineage/contracts: `src/renderer/src/shaders/AGENTS.md`

## High-Risk Areas

- `src/renderer/src/vulkan/vk_render.rs`: highest blast radius orchestration.
- `src/renderer/src/data/data_cache.rs`: handle validity and lifetime-sensitive caches.
- render-path correctness is sensitive to descriptor/pipeline binding order.
- some destroy paths remain incomplete (`todo!()`).

## Working Rules

- Keep stable handle contracts (slot + generation) intact unless deliberately migrating all consumers.
- Treat `.spv` artifacts and GLSL sources as paired assets.
- Prefer small scoped edits and validate with `cargo check -p renderer --examples`.
- If docs and code disagree, treat code as logical truth and record the divergence.

## Runtime Commands

- `cargo run -p renderer --example api_test`
- `cargo run -p renderer --example api_test -- --env <path>`
- `cargo run -p renderer --example demo_pbr`
- `cargo run -p renderer --example demo_unlit`
- `cargo run -p renderer --example demo_model_load`
- `cargo run -p renderer --example demo_async_loading`
