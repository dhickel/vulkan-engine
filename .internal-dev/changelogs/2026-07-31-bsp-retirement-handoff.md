# BSP Detached-Mount Retirement Handoff

## Date

2026-07-31

## Change Summary

Added an ownership-preserving handoff from `BspCoordinator` scene detachment to renderer fence-aware retirement. Replacement, unload, rollback, stale or duplicate upload completion, quarantine teardown, and terminal teardown now retain each `DetachedBspMount` in a pending queue until the caller drains it. Renderer retirement rejections can be reconstructed and requeued without losing the lease.

## Files

- `src/bsp_runtime/src/coordinator.rs`
- `src/bsp_runtime/src/candidate.rs`
- `src/bsp_runtime/tests/retirement_handoff.rs`
- `src/bsp_runtime/tests/transaction_failures.rs`
- `src/bsp_runtime/AGENTS.md`
- `src/renderer/src/api/bsp.rs`
- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/scene/scene_world.rs`
- `.internal-dev/specifications/bsp-transaction-ownership.md`

## Behavioral Impact

`BspCoordinator` exposes pending-count, drain, and requeue APIs for detached mounts. `Scene::clear_bsp_mount` now returns its detached receipt rather than discarding it. `BspRetirementRejection::into_detached` preserves the intact resource lease for retry. Existing retirement diagnostics remain cumulative scene-detachment counts, not renderer acknowledgements.

## Specification Impact

Updated `.internal-dev/specifications/bsp-transaction-ownership.md` to require explicit detached-mount custody through renderer acknowledgement and to record the new handoff API. This resolves the coordinator-side ownership gap while keeping application submission as an explicit downstream obligation.

## Risks

Callers that drain a receipt must still submit it to `Renderer::retire_bsp_mount` or requeue a reconstructed rejection. The API preserves ownership but cannot force a caller to complete retirement after taking the move-only value.

## Follow-up Items

- Wire `bsp_beta` regeneration and shutdown paths to drain, submit, acknowledge, and requeue rejected retirements.
- Keep generic committed-bridge teardown tracking aligned with GitHub #60.
