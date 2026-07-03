# Sprint 03: Editor Packaged Asset Placement Hardening

Status: planned

## Objective

Harden the editor path for package-backed model, prefab, and wall chunk placement so an alpha author can place a packaged asset, select the created scene node, save it, reload it, validate the saved scene through `engine_pack`, and prove the visible result with deterministic engine-owned capture evidence.

## User-Visible Outcome

The default editor sample project can be used as a durable placement workflow:

1. Load `apps/editor/sample_project/engine.project.toml`.
2. Select package assets such as `editor_sample.model.block` and `editor_sample.wall.stone_2m`.
3. Place them into the scene through the command/history path.
4. See the placed node selected and represented in the hierarchy/inspector/status UI.
5. Save to a scene copy, reload it, and see durable asset references survive.
6. Run `engine_pack validate-project` and `engine_pack validate-scene` against the saved scene.
7. Produce headless capture evidence showing the placed packaged assets.

## Classification

Large. This sprint crosses editor state/UI action flow, renderer scene command and scene persistence contracts, `engine_pack` validation, sample project fixtures, docs, and visual proof.

## Scope

- Editor package asset selection, placement confirmation, selection after place, status messages, undo/redo behavior, and focused tests.
- Save/reload scene behavior for package-authored nodes, using temp/copied scenes unless a worker intentionally updates the canonical sample scene with justification.
- `engine_pack` validation of saved scene data against the sample project/package registry.
- Deterministic headless capture proof for packaged model and wall chunk placement.
- API docs and sprint evidence closeout.

## Out Of Scope

- Binary packaging/archive format.
- Thumbnail rendering.
- Material graph editing, PBR factor editing, texture authoring, or material asset document authoring.
- Broad editor UI redesign or global editor architecture rewrite.
- Runtime application launcher work from Sprint 04.
- Closing Sprint 01.

## Target Surfaces

- `apps/editor/src/main.rs`
- `apps/editor/src/app_state.rs`
- `apps/editor/src/panels.rs`
- `apps/editor/src/launch.rs`
- `apps/editor/sample_project/**`
- `src/renderer/src/scene/command.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/data/asset_registry.rs`
- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/**`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- Optional focused capture fixture/example under `src/renderer/examples/capture_tests/`
- Evidence under `.internal-dev/headless_capture_tests/`, `.internal-dev/captures/`, and this sprint's `artifacts/`

## Primary Gate

Packaged assets place, select, save, reload, validate, and visually prove.

## Phase List

1. Phase 01: State and command hardening.
2. Phase 02: Save/reload and validation hardening.
3. Phase 03: Headless capture proof.
4. Phase 04: Docs, final validation, and closeout.

## Evidence Index

Canonical evidence index:

`artifacts/validation-summary.json`

It starts conservative and must not claim `fully_validated` until all phase validators pass, headless capture evidence is reconciled, final quality review passes, and main-thread commit/push/email closeout evidence is recorded.

## Closeout Responsibilities

The main thread owns scoped commits, pushes, and email reports after each phase and final closeout. Planning artifacts mention these as evidence gates only; they do not prescribe live waiting or email mechanics.
