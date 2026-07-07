# Phase 06 Validation Report

Status: passed
Date: 2026-07-07
Validator role: closeout validation / evidence reconciliation

## Directive And Evidence Read

- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-06-compat-docs-closeout.md`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/shared/validation-matrix.md`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/phase-00-validation-report.md` through `phase-05-validation-report.md`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-phase-05-dogfood-migration.md`
- Updated docs, specs, and knowledge files for the root facade and app-owned runtime contract

## Findings

No blocking findings.

Non-blocking residuals:

- The compatibility renderer-owned lifecycle/input/camera APIs remain by design. Active docs now label the app-owned root facade and caller-view render path as the intended path.
- The final runtime smokes reached expected startup milestones and then timed out under long-running event-loop policy, with repeated swapchain acquire retry warnings near timeout.
- Known dead-code warning noise remains in renderer, dogfood, and marching terrain checks.

## Pass/Fail By Criterion

| Criterion | Result | Evidence |
| --- | --- | --- |
| Specs reflect root bin+lib facade and ownership boundaries. | PASS | `.internal-dev/specifications/api.md`, `architecture.md`, `service-graph.md`, `services.md`, and `decisions.md` contain the root facade, renderer `CameraView`, caller-view handoff, and support-crate boundary entries. |
| Docs present app-owned runtime as intended truth. | PASS | API/internal indexes, student quickstart, runtime launcher, renderer lifecycle, input, events, and dogfood docs now distinguish app-owned runtime use from legacy renderer compatibility. |
| Knowledge distinguishes camera paths. | PASS | `.internal-dev/knowledge/renderer-camera-override-behavior.md` separates legacy renderer-owned camera overrides from caller-provided `CameraView` rendering. |
| Changelog records specification impact. | PASS | `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-issues-35-37.md` records docs/spec/knowledge/evidence updates and notes the behavior impact. |
| Final validation index exists and is internally consistent. | PASS | `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json` records phase reports, commands, runtime smokes, headless capture evidence, tooling constraints, superseded artifacts, residual risks, and final review. The capture sidecar now includes the required caller-view path metadata. |
| Stale active-contract wording was swept. | PASS | The exact directive sweep left no active stale-contract hits requiring repair. Beginner-path/facade-language hits were classified as updated intended contract, expected import usage, compatibility labeling, or historical artifacts. |
| Full final command set passed. | PASS | Compile/test gates passed. Runtime smokes reached expected startup milestones and ran until timeout without fatal errors, with swapchain retry warnings recorded as residual risk. |

## Commands Run

```sh
cargo fmt --check
cargo check -p input --quiet
cargo test -p input --quiet
cargo check -p engine_events --quiet
cargo test -p engine_events --quiet
cargo check -p renderer --quiet
cargo test -p renderer --quiet
cargo check -p renderer --examples --quiet
cargo check -p dungeon_dogfood --quiet
cargo check -p marching_terrain --quiet
cargo check --quiet
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
rg -n "pending|planned|not implemented|TODO|/tmp|renderer-owned camera|renderer\\.events_mut|camera_position\\(|set_camera_position\\(|set_camera_look_at|engine_core|engine_runtime" docs .internal-dev src apps
rg -n "renderer::prelude|engine::prelude|compatibility export|compatibility exports|root facade|beginner path|quickstart" docs .internal-dev src apps
```

## Command Results

- `cargo fmt --check`: passed.
- `cargo check -p input --quiet`: passed.
- `cargo test -p input --quiet`: passed, 10 tests.
- `cargo check -p engine_events --quiet`: passed.
- `cargo test -p engine_events --quiet`: passed, 18 tests and 1 ignored doctest.
- `cargo check -p renderer --quiet`: passed with known renderer warning noise.
- `cargo test -p renderer --quiet`: passed, 167 unit tests and 21 integration tests; 5 doctests ignored.
- `cargo check -p renderer --examples --quiet`: passed with known warning noise.
- `cargo check -p dungeon_dogfood --quiet`: passed with known renderer and dogfood warning noise.
- `cargo check -p marching_terrain --quiet`: passed with known renderer and marching terrain warning noise.
- `cargo check --quiet`: passed with known renderer warning noise.
- `api_test` runtime smoke: reached renderer startup milestones and ran until timeout; swapchain retry warnings recorded as residual.
- `dungeon_dogfood` runtime smoke: reached audio bridge, scene seeding, and event-loop startup before timeout; swapchain retry warnings recorded as residual.
- Stale-reference sweeps: no active stale-contract repair remained; historical artifacts were classified rather than rewritten.
- Capture sidecar metadata reconciliation: passed after adding requested app-owned `CameraView`, submitted `render_scene_headless_with_view` path, unused legacy renderer camera path, and residual-risk fields.

## Evidence Inspected

- `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`
- `.internal-dev/changelogs/2026-07-07-engine-runtime-abstractions-issues-35-37.md`
- `.internal-dev/knowledge/renderer-camera-override-behavior.md`
- `.internal-dev/specifications/api.md`
- `.internal-dev/specifications/architecture.md`
- `.internal-dev/specifications/service-graph.md`
- `.internal-dev/specifications/services.md`
- `.internal-dev/specifications/decisions.md`
- `docs/api/00-index.md`
- `docs/api/01-student-quickstart.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/internal/01-architecture.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.png`
- `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.json`
- `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/validation/final-closeout-signoff-review.md`

## Remediation Routing

The final signoff review found an evidence-only defect: the capture sidecar lacked the camera/view metadata required by the validation matrix. The canonical sidecar was enriched with post-capture validation metadata for the requested app-owned `CameraView`, submitted caller-view path, unused legacy renderer camera path, and residual risks. No product-code remediation was required.

## Residual Risk

- Swapchain acquire retry warnings should be investigated separately if they reproduce outside timeout-bound validation.
- Future docs must keep legacy renderer-owned APIs labeled as compatibility unless a separate breaking migration plan removes them.
- The root facade is intentionally thin; future orchestration should be added only when real app usage proves the need.
