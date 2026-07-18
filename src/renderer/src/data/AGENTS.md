# Data Module Agent Guide (`src/renderer/src/data`)

Use this guide for scene/caches/asset-ingest behavior in the renderer data layer.

## Module Role

This module owns:

- camera/controller data (`camera.rs`)
- CPU/GPU transfer-facing structs (`gpu_data.rs`)
- mesh/material/texture/environment caches (`data_cache.rs`)
- active model ingest path (`assimp_util.rs`)
- data helpers (`data_util.rs`)

`gltf_util.rs` is legacy commented code, not active runtime path.

## Critical Contracts

- Stable handles use slot + generation semantics.
- Reserved default slots are contract-critical for texture/material/mesh defaults.
- Cache access must validate handle generation and loaded state.
- Submission boundary types avoid direct Vulkan handle ownership.

## Documentation Routing

- API index: `docs/api/00-index.md`
- Internal index: `docs/internal/00-index.md`
- API scene workflows: `docs/api/03-scene-graph-and-fragment-workflows.md`
- API assets and handles: `docs/api/04-assets-sync-deferred-and-handles.md`
- Internal asset lifecycle: `docs/internal/03-asset-lifecycle-and-io.md`
- Internal data suballocation/transfer: `docs/internal/06-data-suballocation-and-transfer.md`
- Internal flattening/culling: `docs/internal/08-scene-flattening-and-culling.md`

Related runtime files:

- `src/renderer/src/scene/scene_world.rs`
- `src/renderer/src/scene/render_submission.rs`
- `src/renderer/src/vulkan/vk_render.rs`

## Current Risks

- Draw-path material bindings are copied into `CopiedMaterialDrawRecord` while the texture-cache lock is held; adding fields without preserving that copy boundary can reintroduce a lifetime hazard.
- Render bucketing correctness remains sensitive to ordering assumptions.
- Unchecked deallocation paths can violate default-slot/handle invariants.

## Working Rules

- Preserve stable handle invariants unless migration updates all consumers.
- Keep cache-owned addresses out of frame draw objects; copy only required material binding data under the cache guard.
- Keep default-slot contracts explicit in any allocation/deallocation edits.
- If docs and code diverge, treat code as logical truth and record the divergence.

## Validation

- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- For asset, material, environment, or scene-cache changes with visible render impact, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and validate with deterministic headless captures when practical.
