# Validation Matrix

Date: 2026-07-07
Status: canonical validation map

## Evidence Index

Final implementation evidence index:

`artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`

Required fields:

- `status`
- `phase_reports`
- `commands`
- `runtime_smokes`
- `headless_captures`
- `tooling_constraints`
- `superseded_artifacts`
- `residual_risks`
- `final_quality_review`

Allowed interim statuses:

- `baseline_drift_recorded`
- `implementation_checks_passed`
- `validator_failed`
- `repair_in_progress`
- `runtime_smoke_pending`
- `blocked_tooling_constraint`
- `fully_validated`

Do not use `fully_validated` until every required gate is reconciled.

## Phase Gates

| Phase | Primary commands | Behavioral assertions | Evidence path |
| --- | --- | --- | --- |
| 00 preflight drift | `cargo check -p audio`; `cargo check -p renderer --examples`; `cargo check -p dungeon_dogfood` | Baseline blockers are fixed or classified without regression claims. | `validation/phase-00-validation-report.md` |
| 01 root facade | `cargo check`; mandatory root facade/raw import proof; root crate tests | `engine` lib facade compiles, facade modules import from outside the module that defines them, raw crates remain directly importable, launcher behavior preserved, no forbidden lower-crate -> root `engine` edge. | `validation/phase-01-validation-report.md` |
| 02 renderer view path | `cargo check -p renderer`; `cargo test -p renderer`; `cargo check -p renderer --examples` | New view path uses caller view and avoids renderer input/event/camera ownership. Legacy path still works. | `validation/phase-02-validation-report.md` |
| 03 input migration | `cargo check -p input`; `cargo test -p input`; root runtime tests; `cargo check -p renderer` | One dispatch per app frame; transients and action events preserved; UI capture routing explicit. | `validation/phase-03-validation-report.md` |
| 04 event migration | `cargo check -p engine_events`; `cargo test -p engine_events`; root runtime tests | Caller-owned event bus owns lifecycle/input stages and preserves monotonic order. | `validation/phase-04-validation-report.md` |
| 05 dogfood migration | `cargo check -p dungeon_dogfood`; `cargo check`; runtime smoke when compile clean | Dogfood active path owns input/events/camera and renders with caller view. | `validation/phase-05-validation-report.md` |
| 06 closeout | full check suite; stale-reference sweep | Specs/docs/changelog/knowledge reflect new contracts; legacy APIs labeled compatibility. | `validation/phase-06-validation-report.md` |

## Full Final Command Set

```sh
cargo check -p input
cargo test -p input
cargo check -p engine_events
cargo test -p engine_events
cargo check -p renderer
cargo test -p renderer
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
cargo check -p marching_terrain
cargo check
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
```

## Headless Capture Criteria

Use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` if implementation claims visual/camera proof or materially changes visible camera output.

Task-scoped capture output:

`.internal-dev/captures/engine-runtime-abstractions-issues-35-37/`

Required checks:

- PNG and JSON sidecar exist.
- Sidecar identifies target, frame, source, format, extent, requested camera/view, applied camera/view path, and residual risks.
- New caller-provided view path is distinguished from legacy renderer-owned camera path.

## Stale-Reference Sweep

Before final validation, search docs and `.internal-dev` for:

```sh
rg -n "pending|planned|not implemented|TODO|/tmp|renderer-owned camera|renderer.events_mut|camera_position\\(|set_camera_position\\(|set_camera_look_at|engine_core|engine_runtime" docs .internal-dev src apps
```

Also classify beginner-path and facade-language references:

```sh
rg -n "renderer::prelude|engine::prelude|compatibility export|compatibility exports|root facade|beginner path|quickstart" docs .internal-dev src apps
```

The validator must classify hits as:

- expected legacy compatibility reference;
- updated intended contract;
- stale reference requiring repair;
- unrelated historical artifact.
