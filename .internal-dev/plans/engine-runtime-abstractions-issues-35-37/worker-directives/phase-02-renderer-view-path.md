# Phase 02 Worker Directive: Renderer View DTO And No-Dispatch Render Path

Status: ready after Phase 01 validation
Validation report: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-02-validation-report.md`

## Objective

Introduce renderer-owned/lower `CameraView` or `RenderView` DTO and a new renderer render path that uses caller-provided view/projection/position without dispatching input, emitting/draining app events, or updating renderer-owned camera.

## User-Visible Outcome

Apps can render a scene with explicit per-frame camera view data. Legacy renderer-owned camera APIs still compile for old examples.

## Direct Editable Targets

- `src/renderer/src/api/renderer.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/src/api/prelude.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/scene.rs` only if scene camera API support is needed
- `src/renderer/tests/integration.rs` or renderer unit tests
- `src/renderer/examples/*` only for optional compatibility compile fixes
- root facade re-export files from Phase 01 if adding `CameraView` re-export

## Forbidden Scope

- Do not migrate dogfood yet.
- Do not move input dispatch ownership yet.
- Do not move event bus ownership yet.
- Do not remove legacy renderer camera methods.
- Do not require `winit::Window` for headless no-dispatch rendering.

## Supporting Docs To Read

- `02-target-design.md`
- `shared/implementation-notes.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/data/AGENTS.md` if editing data internals

## Ordered Steps

1. Add `CameraView` or `RenderView` in renderer API surface with `view`, `projection`, and `position`.
2. Add constructors/helpers if useful:
   - from matrices;
   - from renderer `Camera` plus aspect ratio;
   - perspective helper with explicit FOV/near/far.
3. Refactor internal render submission so one helper updates `Scene` from supplied view DTO before `build_submission`.
4. Keep legacy `render_scene_internal` creating a view DTO from renderer-owned `Camera`.
5. Add new public no-dispatch/no-camera-ownership path for windowed and headless use.
6. Ensure new path does not call:
   - `prepare_frame`;
   - `prepare_frame_headless`;
   - `InputSystem::dispatch_frame`;
   - `emit_input_action_events_from_snapshot`;
   - `dispatch_events_for_stage` for app lifecycle/input events;
   - internal FPS camera update.
7. Add tests proving caller-provided view is used, at least at scene update/submission level. Use unit tests where possible; avoid Vulkan-heavy tests unless already established.
8. Re-export the DTO from `renderer::api`, `renderer::prelude`, renderer crate root, and root `engine::render`/`prelude`.

## Senior-Engineer Guidance

- The core bug is renderer-owned camera overwrite. The new path must not call the old internal camera calculation.
- It is acceptable for legacy paths to keep current overwrite behavior until closeout labels them compatibility.
- Keep projection ownership explicit. Do not silently compute projection from internal defaults unless caller asks for a helper.
- Preserve capture/debug UI frame context where practical, but do not fake app-owned input by reading renderer-owned `InputSystem`.

## Acceptance Criteria

- Renderer exposes a view DTO from renderer/lower crate.
- New render path accepts caller view data and compiles.
- New render path is mechanically no-dispatch/no-camera-ownership.
- Legacy renderer render/camera APIs still compile.
- Root facade re-exports the new view DTO.

## Negative Checks

- No renderer dependency on root `engine`.
- New path does not call `dispatch_frame`.
- New path does not use `self.camera` for render view unless caller explicitly passed a view created from it.
- New path does not drain renderer-owned app events.
- Headless no-dispatch path does not require `Window`.

## Validation Commands

```sh
cargo check -p renderer
cargo test -p renderer
cargo check -p renderer --examples
cargo check -p engine
rg -n "render_scene_with|CameraView|RenderView|dispatch_frame\\(|prepare_frame\\(|prepare_frame_headless\\(|self\\.camera" src/renderer/src/api/renderer.rs src/renderer/src/api src/renderer/src/lib.rs src/lib.rs
```

If visual/camera proof is claimed, run the headless capture skill with output under `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-02/`.

## Evidence Expectations

- Worker notes identify the new API names.
- Validator inspects code to prove separation from legacy dispatch/camera calls.
- Tests or code inspection prove caller-provided matrices reach `scene.update_camera`.

## Stop Conditions

- Stop if implementing the new path requires changing Vulkan backend ownership broadly.
- Stop if the only feasible implementation still calls legacy frame prep.
- Stop if view DTO placement would require renderer depending on root `engine`.

## Do Not Close Unless

- New path exists before dogfood migration.
- Legacy paths remain available.
- Phase 02 validation report is written.
