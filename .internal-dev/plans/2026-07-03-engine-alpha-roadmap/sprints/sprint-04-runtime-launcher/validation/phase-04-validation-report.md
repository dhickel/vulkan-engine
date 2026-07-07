# Phase 04 Validation Report: Capture Closeout

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Scope: Sprint 04 Phase 04 final validation and capture closeout

## Summary

Phase 04 ran the full required compile/test matrix, produced final root-launcher true headless draw-target capture proof, produced debug timing JSONL through the root launcher, inspected representative PNGs, and reconciled stale-reference sweep hits.

Status: local validation passed, pending independent validator review.

## Commands

| Command | Result | Notes |
| --- | --- | --- |
| `cargo fmt --check` | Passed | Exit 0 |
| `cargo check` | Passed | Existing renderer dead-code warnings remain |
| `cargo check -p renderer --examples` | Passed | Existing renderer warnings remain |
| `cargo check -p editor` | Passed | Existing renderer warnings plus existing `set_active_scene_text` warning |
| `cargo check -p engine_pack --locked` | Passed | Existing renderer warnings remain |
| `cargo test -p engine` | Passed | 17 tests passed |
| `cargo test -p renderer` | Passed | 150 lib tests, 17 integration tests, 5 ignored doctests |
| `cargo test -p engine_pack --locked` | Passed | 13 CLI validation tests passed |
| Final root draw capture command | Passed | 3 true headless draw-target captures |
| Debug-record smoke command | Passed | 1 true headless draw-target capture plus debug JSONL |
| `git diff --check` | Passed | Exit 0 after report/summary update |
| JSON summary parse | Passed | `python -m json.tool .../validation-summary.json >/dev/null` exit 0 after report/summary update |
| Stale-reference sweep | Passed with accepted context hits | See classifications below |

Final root draw capture command:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

Debug-record smoke command:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 1 \
  --capture_frame_start 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/debug-smoke \
  --record_debug 10 \
  --record_debug_interval 50 \
  --record_debug_path .internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl
```

## Capture Proof

Final capture directory:

- `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/`

Sidecars:

| Sidecar | Status | Target | Format | Extent | PNG |
| --- | --- | --- | --- | --- | --- |
| `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-5-draw-seq-0000.json` | `succeeded` | `draw` | `R16G16B16A16_SFLOAT` | `1440x900` | `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-5-draw-seq-0000.png` |
| `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-10-draw-seq-0001.json` | `succeeded` | `draw` | `R16G16B16A16_SFLOAT` | `1440x900` | `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-10-draw-seq-0001.png` |
| `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-15-draw-seq-0002.json` | `succeeded` | `draw` | `R16G16B16A16_SFLOAT` | `1440x900` | `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw/editor-sample-project-frame-15-draw-seq-0002.png` |

Debug-smoke capture directory:

- `.internal-dev/captures/sprint-04-runtime-launcher/debug-smoke/`

Debug-smoke sidecar:

| Sidecar | Status | Target | Format | Extent | PNG |
| --- | --- | --- | --- | --- | --- |
| `.internal-dev/captures/sprint-04-runtime-launcher/debug-smoke/editor-sample-project-frame-5-draw-seq-0000.json` | `succeeded` | `draw` | `R16G16B16A16_SFLOAT` | `1440x900` | `.internal-dev/captures/sprint-04-runtime-launcher/debug-smoke/editor-sample-project-frame-5-draw-seq-0000.png` |

PNG existence checks passed for all sidecar `png_path` values. Representative PNGs from `headless-draw` frame 15 and `debug-smoke` frame 5 were visually inspected; both are nonblank and show the sample scene against the skybox.

This is true root-launcher headless draw-target evidence. No desktop screenshots, compositor screenshots, or present-target captures were used.

## Debug Timing Evidence

Debug timing path:

- `.internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl`

The JSONL file exists, is non-empty, and contains 2 timing snapshots:

- start snapshot at frame 0;
- interval snapshot at frame 5 with CPU and GPU timing data.

## Stale-Reference Sweep

Command:

```bash
rg -n "migration stub|runtime project launcher.*deferred|present-target proof|desktop screenshot|dynamic Rust hot reload implemented|scripting implemented|physics implemented|audio implemented|TODO|not implemented" README.md docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher
```

Classifications:

| Hit class | Classification |
| --- | --- |
| Sprint planning files mentioning the pre-sprint migration stub or deferred launcher | Accepted historical/planning context |
| Sprint directives and matrices containing the sweep command itself | Accepted validation criteria context |
| Sprint specs stating desktop screenshots/present-target captures are invalid | Accepted proof-gate wording |
| `docs/internal/07-rendergraph-dependencies-and-aliasing.md` `not implemented` rendergraph roadmap notes | Accepted unrelated internal roadmap wording |
| Validation reports mentioning the stale terms as checks or resolved criteria | Accepted evidence context |

No current public-facing API/readme docs claim that the root binary is still a migration stub, that the runtime project launcher is still deferred, that dynamic Rust hot reload/scripting/physics/audio are implemented, or that desktop/present proof was used.

## Residuals

- Existing renderer dead-code warnings remain visible during compile, test, and root launcher runs.
- Existing editor `set_active_scene_text` dead-code warning remains visible during `cargo check -p editor`.
- Phase 04 still requires independent validator review before closeout commit/push/report.

## Conclusion

Phase 04 local validation passes. The root launcher runs the sample project outside the editor and produces true headless draw-target capture evidence with passing sidecar predicates. Debug timing capture also works through the root launcher.
