# Phase 02 Renderer View Path Worker Report

Date: 2026-07-07
Status: complete

## Changed Files

- `src/renderer/src/api/renderer.rs`: added public `CameraView` DTO, view constructors/helpers, `render_scene_with_view`, `render_scene_headless_with_view`, and a shared submission helper that applies caller-provided view/projection/position before `Scene::build_submission`. Legacy render path now builds a `CameraView` from the renderer-owned camera and continues through the same submission helper.
- `src/renderer/src/api/mod.rs`: re-exported `CameraView` from `renderer::api`.
- `src/renderer/src/api/prelude.rs`: re-exported `CameraView` from `renderer::prelude`.
- `src/renderer/src/lib.rs`: re-exported `CameraView` from the renderer crate root.
- `src/renderer/tests/integration.rs`: extended prelude import contract with `CameraView`.
- `src/render.rs`: re-exported `CameraView` through the root engine render facade.
- `src/lib.rs`: re-exported `CameraView` through the root engine prelude.
- `tests/facade_imports.rs`: extended root facade import tests to prove `engine::render::CameraView`, `engine::prelude::CameraView`, and `renderer::CameraView` import surfaces compile.

## Criteria Satisfied

- Renderer exposes a renderer-owned/lower `CameraView` DTO with `view`, `projection`, and `position`.
- New public render paths accept caller view data: `Renderer::render_scene_with_view` and `Renderer::render_scene_headless_with_view`.
- New no-dispatch path does not call `prepare_frame`, `prepare_frame_headless`, `InputSystem::dispatch_frame`, `emit_input_action_events_from_snapshot`, `dispatch_events_for_stage`, or FPS camera update. It uses `InputDebugSnapshot::default()` for debug UI frame context instead of renderer-owned input.
- Legacy renderer-owned camera APIs and legacy render paths remain available and still compile.
- `CameraView` is re-exported through `renderer::api`, `renderer::prelude`, renderer crate root, `engine::render`, and `engine::prelude`.
- Focused renderer unit test proves caller-provided view/projection/position reaches scene/submission camera data.
- Root facade import test proves the root re-export surfaces compile.

## Criteria Not Satisfied

- None known for Phase 02.

## Validation

- `cargo fmt --check`: passed.
- `cargo check -p renderer`: passed with existing renderer dead-code warnings.
- `cargo test -p renderer`: passed; 166 lib tests, 20 integration tests, 5 ignored doctests.
- `cargo check -p renderer --examples`: passed with existing renderer dead-code warnings.
- `cargo check -p engine`: passed with existing renderer dead-code warnings.
- `cargo test -p engine`: passed; 22 lib tests, 2 facade integration tests.
- `rg -n "render_scene_with|CameraView|RenderView|dispatch_frame\\(|prepare_frame\\(|prepare_frame_headless\\(|self\\.camera" src/renderer/src/api/renderer.rs src/renderer/src/api src/renderer/src/lib.rs src/lib.rs`: passed as inspection evidence. Matches show the new `render_scene_with_view` path and DTO exports; `prepare_frame`, `prepare_frame_headless`, `dispatch_frame`, and `self.camera` remain confined to legacy paths/tests or legacy camera facade methods.

## Safe Adjacent Hygiene

- Ran `cargo fmt`, which formatted touched Rust files.
- Added `cargo test -p engine` beyond the requested validation list because Phase 02 changed root facade import tests.

## `.internal-dev` Artifacts Touched

- Added this worker report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-02-renderer-view-path-worker-report.md`.

## Blockers And Risks

- No blockers.
- Existing renderer dead-code warnings remain. They predate this phase and were not in scope.
- The worktree contains pre-existing Phase 01/root facade and other local edits outside this phase. They were preserved and not reverted.
