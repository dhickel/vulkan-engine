# Phase 05 Validation Report

Status: passed
Date: 2026-07-07
Validator role: phase validation / quality review

## Directive And Evidence Read

- `AGENTS.md`
- `.internal-dev/specifications/AGENTS.md`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-05-dogfood-migration.md`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-05-dogfood-migration-worker-report.md`
- Supporting plan docs: `02-target-design.md`, `shared/implementation-notes.md`
- Package guides: `src/renderer/AGENTS.md`, `src/input/AGENTS.md`, `src/events/AGENTS.md`
- Relevant knowledge: `.internal-dev/knowledge/renderer-camera-override-behavior.md`, `.internal-dev/knowledge/dungeon-alpha-validation-matrix.md`

## Findings

No blocking findings.

Non-blocking residuals:

- Existing renderer and dogfood dead-code warning noise remains during `cargo check` and `cargo test -p dungeon_dogfood`. This matches the worker report and does not contradict Phase 05 behavior.
- The canonical final evidence index `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json` is not present yet. Phase 06 explicitly owns populating it, so this is not a Phase 05 failure.
- I did not rerun the 60-second windowed runtime smoke. I inspected the worker report, source migration, compile/test gates, forbidden-call grep, and headless capture artifacts. Runtime swapchain acquire retry warnings remain recorded in the worker report and changelog as environment-specific residual.

## Pass/Fail By Criterion

| Criterion | Result | Evidence |
| --- | --- | --- |
| `dungeon_dogfood` owns input/events/camera on active path. | PASS | Windowed setup constructs `app_events`, `app_input`, `frame_clock`, `app_camera`, and `fps_controller` in `apps/dungeon_dogfood/src/main.rs:271-308`; headless setup mirrors this in `apps/dungeon_dogfood/src/main.rs:718-755`. |
| Active path does not call `renderer.events_mut()`. | PASS | Forbidden-call grep over `apps/dungeon_dogfood/src` returned no matches. Startup audio probe receives `&mut app_events` at `apps/dungeon_dogfood/src/main.rs:274-278` and `apps/dungeon_dogfood/src/main.rs:721-725`. |
| Active path does not use renderer-owned camera state for gameplay. | PASS | Forbidden-call grep returned no `camera_position(` or `set_camera_position(` matches in dogfood source. App camera is updated through `FPSController` and corrected from `PlayerState` before view construction at `apps/dungeon_dogfood/src/main.rs:485-517`. |
| Dogfood renders using caller-provided `CameraView`. | PASS | `camera_view(...)` builds `CameraView::from_camera` in `apps/dungeon_dogfood/src/main.rs:451-458`; render path calls `render_scene_headless_with_view` / `render_scene_with_view` at `apps/dungeon_dogfood/src/main.rs:517-522`. |
| App-owned input dispatch occurs exactly once per redraw/app frame. | PASS | `render_frame(...)` calls `input.dispatch_frame()` once at `apps/dungeon_dogfood/src/main.rs:475-478`; redraw and headless loops call `render_frame(...)` once per frame at `apps/dungeon_dogfood/src/main.rs:374-388` and `apps/dungeon_dogfood/src/main.rs:806-821`. Forbidden-call grep returned no legacy renderer frame/input dispatch calls. |
| Audio bridge emits into caller-owned event bus. | PASS | `run_startup_audio_probe` accepts `events: &mut EventBus` at `apps/dungeon_dogfood/src/audio_bridge.rs:57-61`; audio events emit into that bus at `apps/dungeon_dogfood/src/audio_bridge.rs:164-169` and drain startup stage at `apps/dungeon_dogfood/src/audio_bridge.rs:182-192`. |
| Forbidden active-path calls absent from `apps/dungeon_dogfood/src`. | PASS | `rg -n "events_mut\\(|camera_position\\(|set_camera_position\\(|install_default_fps_input\\(|begin_frame\\(|render_scene_in_frame\\(|render_scene\\(|render_scene_headless\\(|update_input\\(" apps/dungeon_dogfood/src` returned no matches. |
| Docs/spec/changelog closeout accurate and not overstated. | PASS | Dogfood docs describe app-owned runtime and caller-view rendering at `docs/api/14-dogfood-vertical-slice.md:66-89`; API/spec entries state the Phase 05 contract at `.internal-dev/specifications/api.md:20-22`, `.internal-dev/specifications/architecture.md:17-19`, and `.internal-dev/specifications/services.md:17-19`; changelog records validation and residual swapchain warnings at `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-05-dogfood-migration.md:22-35`. |
| Validation evidence supports worker report. | PASS | Local compile/test/static gates passed. Capture sidecar reports `status: "succeeded"`, draw target, frame 0, and 1280x720 extent; PNG exists as 1280x720 RGBA and visual inspection showed a nonblank dungeon corridor. |

## Commands Run

```sh
sed -n '1,220p' AGENTS.md
sed -n '1,240p' .internal-dev/specifications/AGENTS.md
sed -n '1,260p' .internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-05-dogfood-migration.md
sed -n '1,260p' .internal-dev/plans/engine-runtime-abstractions-issues-35-37/work-units/phase-05-dogfood-migration-worker-report.md
find .internal-dev/knowledge -maxdepth 2 -type f | sort
sed -n '1,260p' .internal-dev/plans/engine-runtime-abstractions-issues-35-37/02-target-design.md
sed -n '1,260p' .internal-dev/plans/engine-runtime-abstractions-issues-35-37/shared/implementation-notes.md
sed -n '1,220p' src/renderer/AGENTS.md
sed -n '1,180p' src/input/AGENTS.md
sed -n '1,180p' src/events/AGENTS.md
sed -n '1,220p' .internal-dev/knowledge/renderer-camera-override-behavior.md
sed -n '1,220p' .internal-dev/knowledge/dungeon-alpha-validation-matrix.md
rg -n "events_mut\\(|camera_position\\(|set_camera_position\\(|install_default_fps_input\\(|begin_frame\\(|render_scene_in_frame\\(|render_scene\\(|render_scene_headless\\(|update_input\\(" apps/dungeon_dogfood/src
cargo check -p dungeon_dogfood
cargo test -p dungeon_dogfood
cargo check
find .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
file .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.png .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.json
```

## Command Results

- `cargo check -p dungeon_dogfood`: passed with existing renderer dead-code warnings and 5 dogfood warnings.
- `cargo test -p dungeon_dogfood`: passed, 53 tests passed.
- `cargo check`: passed with existing renderer dead-code warnings.
- Forbidden-call grep over `apps/dungeon_dogfood/src`: passed, no matches.
- Capture sidecar inspection: passed; JSON reports successful draw capture at 1280x720.
- Capture PNG inspection: passed; file is a 1280x720 RGBA PNG and visual inspection showed a nonblank dungeon corridor.

## Evidence Inspected

- `apps/dungeon_dogfood/src/main.rs:271-308`
- `apps/dungeon_dogfood/src/main.rs:320-328`
- `apps/dungeon_dogfood/src/main.rs:374-388`
- `apps/dungeon_dogfood/src/main.rs:451-522`
- `apps/dungeon_dogfood/src/main.rs:718-755`
- `apps/dungeon_dogfood/src/main.rs:806-821`
- `apps/dungeon_dogfood/src/audio_bridge.rs:57-61`
- `apps/dungeon_dogfood/src/audio_bridge.rs:164-192`
- `docs/api/14-dogfood-vertical-slice.md:66-89`
- `.internal-dev/specifications/api.md:20-22`
- `.internal-dev/specifications/architecture.md:17-19`
- `.internal-dev/specifications/services.md:17-19`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-05-dogfood-migration.md:22-35`
- `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.json`
- `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.png`

## Remediation Routing

None required for Phase 05.

## Residual Risk

- Windowed runtime smoke was not rerun during this validation pass; the worker report and changelog preserve the observed swapchain acquire warning residual.
- Phase 06 still needs to finish compatibility labeling, stale-reference sweep, whole-plan evidence indexing, and final closeout.
