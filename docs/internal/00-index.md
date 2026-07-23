# Engine Internals Reference

> All citations trace to source code. Generated from a fresh codebase audit — no legacy docs consulted.

## Audience

Contributors working inside the renderer internals — Vulkan orchestration, data caches, pass execution, and scene flattening. Assumes Rust proficiency and basic Vulkan familiarity.

## Workspace Context

Root `Cargo.toml` currently declares the root `engine` package plus `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `src/events` (`engine_events`), `src/launch_shared`, `apps/dungeon_dogfood`, `apps/voxel_demo`, and `tools/engine_pack`. These internals docs primarily cover renderer/input and selected app-owned architecture; support crates and apps should not be inferred production-ready from workspace membership alone.

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
| 1 | [01-rendering-pipeline-mental-model.md](01-rendering-pipeline-mental-model.md) | Renderer frame data flow, pass order, command-recording ownership |
| 2 | [02-synchronization-and-fencing.md](02-synchronization-and-fencing.md) | Synchronization primer, frame timeline, fence ownership |
| 3 | [03-asset-lifecycle-and-io.md](03-asset-lifecycle-and-io.md) | Disk→GPU asset lifecycle, caches, staging, package identity resolution |
| 4 | [04-api-to-backend-handoff.md](04-api-to-backend-handoff.md) | API→backend handoff, copied draw records, CameraView ownership boundary |
| 5 | [05-vulkan-sync-and-frame-lifecycle.md](05-vulkan-sync-and-frame-lifecycle.md) | Vulkan frame state machines, descriptor pools, swapchain/drain transactions |
| 6 | [06-data-suballocation-and-transfer.md](06-data-suballocation-and-transfer.md) | Suballocation, storage buffers, transfer completion, fence-observed retirement |
| 7 | [07-rendergraph-dependencies-and-aliasing.md](07-rendergraph-dependencies-and-aliasing.md) | Fixed pass order, dependencies, attachment transitions, aliasing caveats |
| 8 | [08-scene-flattening-and-culling.md](08-scene-flattening-and-culling.md) | Scene flattening, render submission, light caps, conservative culling |
| 9 | [09-input-winit-integration.md](09-input-winit-integration.md) | Winit ingestion, input dispatch, snapshot bridge |
| 10 | [10-event-system-and-lifecycle.md](10-event-system-and-lifecycle.md) | Event ownership boundaries, emission ordering, validation |
| 11 | [11-physics-and-collision.md](11-physics-and-collision.md) | Physics crate boundaries, collision metadata validation, event bridge |
| 12 | [12-audio-foundation.md](12-audio-foundation.md) | Audio crate boundaries, package/scene validation, event bridge |
| 13 | [13-engine-integration-contracts.md](13-engine-integration-contracts.md) | Phase 0 contracts: frame/swapchain/descriptor state machines, compatibility map, CSM camera proof, evidence index |
| 14 | [14-renderer-descriptor-abi.md](14-renderer-descriptor-abi.md) | Descriptor ABI manifest: every set, binding, type, size, stage, pipeline consumer, and shader pair |
| 15 | [15-visual-regression.md](15-visual-regression.md) | Decoded-pixel visual-regression harness, per-test tolerances, baseline-update workflow |
| 16 | [16-voxel-demo-config-regeneration.md](16-voxel-demo-config-regeneration.md) | Voxel-demo config identities, deterministic v2 generation, CPU scene packages, editor lifecycle, and regeneration commit/retirement |
| 17 | [17-safety-refactor-remediation-ledger.md](17-safety-refactor-remediation-ledger.md) | Phase 10 closeout: 21-row finding ledger, cross-reference map, status taxonomy, and companion to `tests/remediation_ledger.rs` |

## Key Source Files

| File | Role |
|------|------|
| [`src/renderer/src/api/renderer.rs`](../../src/renderer/src/api/renderer.rs) | Public API facade |
| [`src/renderer/src/vulkan/vk_render.rs`](../../src/renderer/src/vulkan/vk_render.rs) | Vulkan ownership, construction/teardown, environment generation, and frame coordination |
| [`src/renderer/src/vulkan/vk_frame.rs`](../../src/renderer/src/vulkan/vk_frame.rs) | Frame transaction, acquire/drain/submit/present, and timing lifecycle |
| [`src/renderer/src/vulkan/vk_commands.rs`](../../src/renderer/src/vulkan/vk_commands.rs) | Pass-specific command recording and draw-list policy |
| [`src/renderer/src/data/data_cache.rs`](../../src/renderer/src/data/data_cache.rs) | Mesh/texture/material caches (~2475 lines) |
| [`src/renderer/src/scene/scene_world.rs`](../../src/renderer/src/scene/scene_world.rs) | Scene graph and submission builder |
| [`src/renderer/src/rendergraph/mod.rs`](../../src/renderer/src/rendergraph/mod.rs) | Fixed rendergraph order and pass trait |
| [`src/renderer/src/vulkan/vk_shadow.rs`](../../src/renderer/src/vulkan/vk_shadow.rs) | Frame-local directional shadow resources and light-space fitting |
| [`src/input/src/lib.rs`](../../src/input/src/lib.rs) | Input system (single file) |
| [`src/events/src/lib.rs`](../../src/events/src/lib.rs) | Event contracts, staged bus, recorder |
| [`src/physics/src/lib.rs`](../../src/physics/src/lib.rs) | Renderer-independent alpha physics API, Rapier wrapper, event bridge |
| [`src/audio/src/lib.rs`](../../src/audio/src/lib.rs) | Renderer-independent alpha audio clip/probe/playback facade |
| [`apps/voxel_demo/src/config.rs`](../../apps/voxel_demo/src/config.rs) | Strict preset documents, resolution, canonical identities, and save/load rules |
| [`apps/voxel_demo/src/scene_package.rs`](../../apps/voxel_demo/src/scene_package.rs) | Renderer-free CPU generation/mesh/partition package boundary |
| [`apps/voxel_demo/src/regeneration.rs`](../../apps/voxel_demo/src/regeneration.rs) | Latest-wins worker, main-thread replacement commit, material cache, and deferred retirement |
| [`apps/voxel_demo/src/editor.rs`](../../apps/voxel_demo/src/editor.rs) | imgui draft model and event-loop-owned command queue |

## Distributed Knowledge

Module-level guides provide subsystem detail:
- Vulkan: [`src/renderer/src/vulkan/AGENTS.md`](../../src/renderer/src/vulkan/AGENTS.md)
- Data/caches: [`src/renderer/src/data/AGENTS.md`](../../src/renderer/src/data/AGENTS.md)
- Shaders: [`src/renderer/src/shaders/AGENTS.md`](../../src/renderer/src/shaders/AGENTS.md)
- Renderer: [`src/renderer/AGENTS.md`](../../src/renderer/AGENTS.md)

## Conceptual Deep Dives

These eight chapters are the canonical conceptual reference for contributors working inside the renderer internals. Each file has been verified against live code at Phase 05 and is current. These replace the older numbered internal docs (01-architecture through 08-shaders) for conceptual learning; the older files remain as historical references for topics not yet migrated.

**Historical disposition (Phase 05):** The original 01-architecture.md, 02-renderer-internals.md, 03-asset-pipeline.md, 04-vulkan-subsystem.md, 05-scene-internals.md, 06-input-internals.md, 07-rendergraph.md, and 08-shaders.md remain in the tree as read-only historical artifacts. They are superseded by the eight Conceptual Deep Dive chapters below and should not receive new content. The Reading Order table above points at canonical deep dives; use historical files only when explicitly researching legacy context.

| Chapter | Document | Purpose |
|---------|----------|--------|
| 1 | [01-rendering-pipeline-mental-model.md](01-rendering-pipeline-mental-model.md) | End-to-end pass list, command-recording ownership, terminal-present stage, both facade entry paths |
| 2 | [02-synchronization-and-fencing.md](02-synchronization-and-fencing.md) | Primer: VkFrameSync transaction/fence ownership, frame timeline, barrier patterns |
| 3 | [03-asset-lifecycle-and-io.md](03-asset-lifecycle-and-io.md) | Loader/cache/retirement facts, Assimp ingest, handle validation, package identity resolution |
| 4 | [04-api-to-backend-handoff.md](04-api-to-backend-handoff.md) | Copied material draw records, CameraView boundary, ownership table, no raw-pointer design |
| 5 | [05-vulkan-sync-and-frame-lifecycle.md](05-vulkan-sync-and-frame-lifecycle.md) | Deep dive: state machines, descriptor pool lifecycle, swapchain states, drain transactions, RetirementClass taxonomy |
| 6 | [06-data-suballocation-and-transfer.md](06-data-suballocation-and-transfer.md) | VkSubAllocator, VkStorageBuffer, free-list coalescing, fence-observed completion, latch semantics |
| 7 | [07-rendergraph-dependencies-and-aliasing.md](07-rendergraph-dependencies-and-aliasing.md) | Fixed sequential pass order, transition ownership matrix, DAG/aliasing explicitly not promised |
| 8 | [08-scene-flattening-and-culling.md](08-scene-flattening-and-culling.md) | Multiple point lights (clamped to MAX_POINT_LIGHTS_GPU), one directional shadow owner, conservative bounds/culling, vk_commands.rs recording ownership |

## See Also

- [API Reference](../api/00-index.md) — public API surface
- [API Events and Lifecycle](../api/12-events-and-lifecycle.md) — public event consumption contract
- [Physics and Collision](11-physics-and-collision.md) — current alpha physics/collision implementation boundary
- [Audio Foundation](12-audio-foundation.md) — current alpha audio implementation boundary
- [Alpha Readiness Baseline](../gap-report.md) — current readiness and residual-classification routing
