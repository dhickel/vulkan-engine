# Sprint 03 Phase 03 Validation Report

## Verdict

Worker-complete pending independent validation.

Phase 03 now has deterministic engine-owned visual proof from the editor headless path. The accepted capture command loaded the Phase 02 saved scene copy through the sample project, loaded package records, rendered with `Renderer::new_headless`, captured the offscreen draw target, wrote three PNG/JSON capture pairs under `.internal-dev/captures/`, and exited successfully after the requested captures completed.

The earlier present-target capture directory is superseded and is not used as Phase 03 proof. The accepted evidence is the draw-target capture directory listed below.

## Scope

Implemented Sprint 03 Phase 03 only on branch `sprint/alpha-03-editor-packaged-placement`.

Changed code:

- `apps/editor/src/main.rs`: added a narrow `--headless` editor execution path that uses `Renderer::new_headless`, loads the same project package registry and saved scene path as the windowed editor, renders frames through `render_scene_headless`, and exits once requested capture artifacts have succeeded.

Changed evidence:

- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/`: generated durable draw-target PNG and JSON sidecar artifacts.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json`: updated Phase 03 evidence conservatively.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/validation/phase-03-validation-report.md`: this report.

No fallback renderer capture harness was added because the preferred editor path now produces deterministic evidence.

## Capture Input

Project:

```text
apps/editor/sample_project/engine.project.toml
```

Saved scene:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json
```

Expected visible assets:

- `editor_sample.model.block`, node `node.placed.editor_sample_model_block.000001`, display name `Block Prop`, transform translation `[-1.5, 0.0, -2.0]`, scale `[1.25, 1.0, 1.25]`.
- `editor_sample.wall.stone_2m`, node `node.placed.editor_sample_wall_stone_2m.000002`, display name `Stone Wall 2m`, transform translation `[1.5, 0.0, -2.0]`, scale `[1.0, 1.0, 1.0]`.

## Capture Command

```text
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw
```

Result: passed. The command exited with status 0 after all three requested captures completed. Logs reported `Loaded 2 package-backed asset record(s) for headless editor capture` and `Headless editor capture completed: 3 capture(s) written`.

## Capture Artifacts

Capture directory:

```text
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw
```

PNG files:

```text
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-5-draw-seq-0000.png
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-10-draw-seq-0001.png
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-15-draw-seq-0002.png
```

Sidecar JSON files:

```text
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-5-draw-seq-0000.json
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-10-draw-seq-0001.json
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-15-draw-seq-0002.json
```

Sidecar summary:

- All three sidecars report `status: succeeded`.
- Capture target is `draw`.
- Source format is `R16G16B16A16_SFLOAT`, copied from the offscreen draw image path and converted to RGBA8 PNG.
- Frames captured: 5, 10, and 15.
- Extent is 1440 x 900.
- Source is `Sequence`.

## Visual Observation

Inspected:

```text
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-15-draw-seq-0002.png
```

Observation: the image clearly shows two placed package-backed assets against the sky environment. The left asset is a cuboid block prop with visible depth/side shading. The right asset is a larger rectangular wall chunk. Their left/right placement matches the saved scene transforms, with the block at negative X and the wall chunk at positive X.

Visual status: pass, pending independent validation.

Pixel sanity check across all three accepted PNGs:

```text
size = 1440 x 900
mean RGB = 84.27, 89.89, 104.15
channel extrema = R 17..231, G 24..218, B 41..207
```

This confirms the accepted draw-target captures are nonblank and stable across frames 5, 10, and 15.

## Validation Commands

```text
cargo fmt --check
Result: passed

cargo check -p editor
Result: passed with existing renderer/editor dead-code warnings

RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw
Result: passed, wrote three PNG/JSON capture pairs

cargo check -p renderer --examples
Result: passed with existing renderer warnings

cargo test -p renderer scene
Result: passed, 38 renderer lib tests and 2 integration tests matching filter

git diff --check
Result: passed

jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json
Result: passed after Phase 03 summary edits
```

## Residual Risk

- Phase 03 is worker-complete but still needs independent validation before being marked fully passed.
- Existing renderer/editor dead-code warnings remain and were not part of this phase.
- The first probe of the old windowed `--headless` path wrote two captures but exited with a timeout/core dump; those partial files were removed before the clean headless editor capture run.
- `.idea/engine.iml` and `.reasonix/` remain unrelated dirty state and were not touched.

## Independent Validation - 2026-07-03

Validator verdict: pass, pending commit/push/reporting. No Phase 03 code defects found.

Primary risk checked: the accepted proof is truly engine-owned headless draw-target capture, not a compositor screenshot or a windowed event-loop capture. Code inspection confirms `run()` returns to `run_headless_editor()` before constructing `EventLoop` or `WindowBuilder` when `--headless` is set. The headless path creates `Renderer::new_headless`, loads the sample project package registry and Phase 02 saved scene copy, renders with `render_scene_headless`, and exits after the requested captures succeed.

Accepted evidence checked:

- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-5-draw-seq-0000.{png,json}`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-10-draw-seq-0001.{png,json}`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/engine-editor-frame-15-draw-seq-0002.{png,json}`

All accepted sidecars report:

- `status: succeeded`
- `capture_target: draw`
- `source: Sequence`
- `format: R16G16B16A16_SFLOAT`
- `extent: 1440 x 900`

Visual and pixel evidence:

- Visually inspected frame 15. It shows the block prop on the left and the larger wall chunk on the right against the sky environment, matching the saved scene left/right placement.
- `file` reports all accepted PNGs as 1440 x 900 RGBA PNGs.
- ImageMagick stats for all accepted and validator-rerun PNGs: `mean=0.522856`, `min=0.0666667`, `max=1`, `unique=8007`.
- RGB means for all accepted and validator-rerun PNGs: `R=0.330484`, `G=0.352507`, `B=0.408433`, `A=1`.

Independent rerun command:

```text
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator
```

Result: passed with exit code 0. Logs showed `Loaded 2 package-backed asset record(s) for headless editor capture`, loaded the Phase 02 scene path, saved draw-target captures for frames 5, 10, and 15, and reported `Headless editor capture completed: 3 capture(s) written`.

Validator rerun artifacts:

- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator/engine-editor-frame-5-draw-seq-0000.{png,json}`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator/engine-editor-frame-10-draw-seq-0001.{png,json}`
- `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator/engine-editor-frame-15-draw-seq-0002.{png,json}`

Criterion results:

| Criterion | Result | Evidence |
| --- | --- | --- |
| Accepted command uses `--headless --capture_target draw` and durable draw directory | Pass | Reported command and independent rerun both use the required flags and `.internal-dev/captures/...headless-draw*` paths. |
| Accepted proof does not rely on windowed event loop or compositor screenshot | Pass | `apps/editor/src/main.rs` dispatches to `run_headless_editor()` before `EventLoop::new()`/`WindowBuilder`, and the renderer side uses headless/offscreen paths. |
| Uses `Renderer::new_headless` plus `render_scene_headless` | Pass | Code inspection of `run_headless_editor()`. |
| Sidecars report draw target, non-present source format, success | Pass | Accepted and rerun sidecars report `capture_target: draw`, `format: R16G16B16A16_SFLOAT`, and `status: succeeded`. |
| PNGs exist and are nonblank | Pass | All PNGs exist, are 1440 x 900 RGBA, have 8,007 unique colors, and frame 15 was visually inspected. |
| Package-backed saved scene path is exercised | Pass | Logs and code path load the sample project/package registry and Phase 02 saved scene copy. |
| Validation summary is valid JSON and reconciled | Pass | `jq empty` passed; summary was reconciled through Phase 03 commit, push, and reporting evidence. |

Validator commands:

```text
jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json
Result: passed

for f in .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/*.json; do jq -e '.status == "succeeded" and .capture_target == "draw" and .format == "R16G16B16A16_SFLOAT" and .extent.width == 1440 and .extent.height == 900 and .source == "Sequence"' "$f"; done
Result: passed for all accepted sidecars

file .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/*.png
Result: all accepted PNGs are 1440 x 900 RGBA PNGs

magick <accepted-and-validator-png> -format '%f %wx%h mean=%[fx:mean] min=%[fx:minima] max=%[fx:maxima] unique=%k\n' info:
Result: all accepted and validator-rerun PNGs are nonblank and stable

cargo fmt --check
Result: passed

cargo check -p editor
Result: passed with existing dead-code warnings

cargo check -p renderer --examples
Result: passed with existing dead-code warnings

cargo test -p renderer scene
Result: passed; 38 renderer lib tests and 2 integration tests matched the filter

RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --headless --capture_target draw --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw-validator
Result: passed; wrote three validator PNG/JSON draw-target capture pairs

git diff --check
Result: passed
```

Validator-updated files:

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/validation/phase-03-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json`

Unrelated local state preserved:

- `.idea/engine.iml`
- `.reasonix/`
