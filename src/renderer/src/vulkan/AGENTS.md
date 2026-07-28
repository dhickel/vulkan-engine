# Vulkan Module Agent Guide (`src/renderer/src/vulkan`)

Use this guide for Vulkan lifecycle, sync, resource ownership, and frame execution behavior.

## Module Role

This module owns:

- Vulkan initialization (`vk_init.rs`)
- core ownership/destroy/frame structs (`vk_types.rs`)
- render loop orchestration (`vk_render.rs`)
- directional shadow resources and light-volume fitting (`vk_shadow.rs`)
- descriptor systems (`vk_descriptor.rs`)
- pipeline creation (`vk_pipeline.rs`)
- memory/suballocation (`vk_storage.rs`)

## Core Execution Path

`VkRender::new(...)` builds device/swapchain/frame resources/caches.

`VkRender::render(...)` performs:

1. transfer polling and completion handling
2. frame-slot wait/cleanup and image acquisition
3. rendergraph execution, including the frame-local directional shadow pass
4. submit/present, or a drain transaction when recording fails after acquisition

## Documentation Routing

- Internal index: `docs/internal/00-index.md`
- Rendering pipeline model: `docs/internal/01-rendering-pipeline-mental-model.md`
- Synchronization/fencing: `docs/internal/02-synchronization-and-fencing.md`
- API-to-backend handoff: `docs/internal/04-api-to-backend-handoff.md`
- Vulkan frame lifecycle: `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- Rendergraph dependencies: `docs/internal/07-rendergraph-dependencies-and-aliasing.md`
- BSP runtime and lifetime: `docs/internal/18-bsp-runtime-and-lifetime.md` — BSP pipeline variants, descriptor layouts, and frame-value UBO
- Package context: `src/renderer/AGENTS.md`

## Current Risks

- `VkDestroyable` trait is implemented for all active resource types.
- Geometry pass and pipeline binding order are correctness-sensitive.
- Swapchain rebuild still has explicit cleanup FIXME areas.
- Queue usage assumptions should be revisited if family strategy changes.

## Working Rules

- Keep synchronization/transition ordering explicit when modifying frame logic. Any post-acquire failure after fence reset must retire the fence and acquired semaphore/image state.
- Keep descriptor set ordering aligned with shader and pipeline contracts; scene set 0 binding 5 is the per-frame comparison shadow map.
- Validate major changes with renderer checks and bounded runtime smoke.
- If docs and code diverge, treat code as logical truth and record the divergence.

## Validation

- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- Optional headless smoke: `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test`
- For frame output, synchronization, rendergraph, or readback changes that need visual proof, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and prefer timeout-bound headless captures under `.internal-dev/captures/`.
