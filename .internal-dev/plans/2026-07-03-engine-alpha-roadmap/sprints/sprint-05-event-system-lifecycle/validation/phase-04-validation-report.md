# Phase 04 Validation Report: Closeout Evidence

Date: 2026-07-03

Branch: `sprint/alpha-05-event-system-lifecycle`

Status: validation passed, final quality passed

## Scope Validated

- Confirmed Phase 01, Phase 02, and Phase 03 validation reports and validator reviews exist.
- Ran the full compile/test matrix required by the Phase 04 directive.
- Ran root runtime headless debug timing smoke and recorded JSONL evidence.
- Ran true engine headless draw-target capture with `--capture_target=draw`.
- Inspected PNG/JSON capture evidence and viewed the draw capture output.
- Reconciled stale status wording in prior phase reports and sprint README.
- Kept `fully_validated: false` during Phase 04 handoff; final quality later passed and the summary can now be finalized.

## Commands

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Passed | Reran after formatting the Phase 04 warning cleanup. |
| `cargo check` | Passed | Existing renderer warnings remain; Phase 04 removed a new root unused-import warning. |
| `cargo test -p engine_events` | Passed | 7 tests passed. |
| `cargo test -p input` | Passed | 10 tests passed. |
| `cargo test -p renderer` | Passed | 152 lib tests, 17 integration tests, 5 ignored doctests. |
| `cargo test -p engine` | Passed | 20 tests passed. |
| `cargo check -p renderer --examples` | Passed | Existing renderer warnings remain. |
| `cargo check -p editor` | Passed | Existing renderer/editor warnings remain. |
| `cargo check -p dungeon_dogfood` | Passed | Existing renderer/dogfood warnings remain. |
| `cargo check -p engine_pack` | Passed | Existing renderer warnings remain. |
| `RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl` | Passed | Wrote one timing JSONL snapshot. |
| `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_target=draw --capture_dir=.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw` | Passed | Wrote three draw-target PNGs and sidecars. |
| `rg -n "engine_events" apps docs src/renderer/src/vulkan src/renderer/src/data src/renderer/src/scene src/renderer/src/shaders src/renderer/src/api src/runtime.rs Cargo.toml src/renderer/Cargo.toml` | Passed | Direct `engine_events` imports remain at root runtime/facade/docs boundaries. |
| `rg -n "/tmp|pending|planned|not implemented|TODO|agent id|desktop screenshot|playwright" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-05-event-system-lifecycle` | Reviewed | Remaining hits are directive/checklist text, existing packaging CLI `/tmp` examples, unrelated rendergraph future-direction docs, and deliberate final-quality pending references. |
| `git diff --check` | Passed | No whitespace errors. |

## Runtime Smoke Evidence

Debug report:

- `.internal-dev/debug_reports/sprint-05-event-system-lifecycle/root-runtime-events-timing.jsonl`

Observed:

- File exists.
- Contains one `timing_snapshot` row.
- Mode is `CPU-only fallback (GPU timestamps unsupported)`.
- Runtime exited successfully.

## Headless Draw Capture Evidence

Capture command used the engine headless path and `--capture_target=draw`.

PNG evidence:

- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-5-draw-seq-0000.png`
- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-10-draw-seq-0001.png`
- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-15-draw-seq-0002.png`

Sidecar evidence:

- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-5-draw-seq-0000.json`
- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-10-draw-seq-0001.json`
- `.internal-dev/captures/sprint-05-event-system-lifecycle-headless-draw/editor-sample-project-frame-15-draw-seq-0002.json`

Sidecar facts:

- `status`: `succeeded`
- `capture_target`: `draw`
- `format`: `R16G16B16A16_SFLOAT`
- `extent`: `1440 x 900`
- `sequence_index`: `0`, `1`, `2`

PNG facts:

- All PNGs are `1440 x 900`, 8-bit/color RGBA, non-interlaced.
- Pixel stats are nonblank with RGB extrema spanning approximately `R 17-214`, `G 24-210`, `B 41-207`.
- Visual inspection shows the editor sample scene draw output: sky/environment background and a central rendered block.

## Cleanup Performed

- Removed a new root runtime unused-import warning by gating `EventEnvelope` import with `#[cfg(test)]`.
- Ran `cargo fmt`.
- Updated prior phase report status lines to match committed/pushed/reported reality.

## Residuals

- Existing renderer/app warning noise remains outside Sprint 05 scope.
- Final quality review passed after Phase 04 handoff.
- `.idea/engine.iml` and `.reasonix/` remain unrelated local state and must stay out of commits.

## Validator Handoff

Final quality review passed. Validate future closeout edits against the recorded evidence paths and true headless draw-target capture proof.
