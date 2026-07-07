# Engine Alpha Readiness Baseline

Date: 2026-07-03

This file keeps the historical `docs/gap-report.md` path stable, but it is no longer a raw defect list. Sprint 01 Phase 02 rewrote it as the current alpha readiness baseline so future planning does not treat stale missing-subsystem claims as source truth.

## Source Of Truth

- Code is the logical source of truth.
- Documentation is intended truth.
- Root `Cargo.toml` currently declares these workspace members:
  - `src/input`
  - `src/renderer`
  - `src/audio`
  - `src/physics`
  - `src/scripting`
  - `apps/dungeon_dogfood`
  - `apps/marching_terrain`
- Workspace membership means the crate or app exists in the build graph. It does not prove alpha completeness, runtime readiness, editor readiness, or feature parity.

## Current Runtime Orientation

- `engine` (`src/main.rs`) is the alpha data-driven runtime launcher for project manifests.
- Renderer example behavior is still exercised through `src/renderer/examples/*.rs`, but examples are diagnostics/API references rather than the primary project runtime path.
- Custom Rust application behavior lives in workspace app crates under `apps/<name>`.
- Public API docs start at `docs/api/00-index.md`.
- Contributor/internal docs start at `docs/internal/00-index.md`.
- Phase 01 restored the active `.internal-dev/AGENTS.md` process guide.

## Superseded Historical Claims

The earlier report contained current-tense claims that several subsystems did not exist. Those claims are stale after comparison with root `Cargo.toml`: audio, physics, scripting, editor, and dungeon dogfood workspace members are present. Their remaining readiness gaps must be reclassified in Sprint 01 Phase 03 using live source evidence instead of copying the old report forward.

Historical findings that may still need verification include:

- scene serialization and save/load support;
- ray picking and object selection;
- undo/redo infrastructure;
- headless/offscreen renderer support;
- swapchain rebuild cleanup;
- animation/skinning runtime completeness;
- editor camera workflows;
- project and asset-browser readiness;
- asset hot reload behavior;
- production-path error propagation;
- depth buffer access;
- public rendergraph extension access;
- critical test coverage;
- push constant sizing;
- example asset-path ergonomics;
- legacy scratch modules;
- input profile end-user schema documentation;
- frustum and occlusion culling.

These are residual candidates, not validated current blockers in this file. Phase 03 owns classification as verified, stale, unknown, accepted debt, or blocked validation.

## Current Readiness Assessment

The workspace has moved beyond a renderer/input-only layout, but the alpha baseline is not yet validated as general-purpose-engine-ready. The defensible current position is:

- the root `engine` binary launches data-driven project manifests;
- renderer examples remain canonical renderer diagnostics;
- custom Rust apps run as app crates under `apps/<name>`;
- input has a documented crate contract and validation commands;
- audio, physics, scripting, editor, and dogfood app presence is confirmed only at the workspace-manifest level in this phase;
- stale historical claims must not drive planning until Phase 03 validates and classifies them against live source;
- visual or runtime readiness requires later compile/runtime/capture evidence, not this docs repair.

## Residual Tracking Route

Use this file for current alpha-readiness orientation. Use the Sprint 01 residual register produced by Phase 03 for verified open gaps and deferred work. Until that register exists, treat historical gap items above as candidate inputs only.
