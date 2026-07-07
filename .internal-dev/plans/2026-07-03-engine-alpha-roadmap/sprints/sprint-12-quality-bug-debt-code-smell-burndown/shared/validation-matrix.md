# Validation Matrix

| Gate | Phase | Required Commands | Conditional Commands | Evidence |
| --- | --- | --- | --- | --- |
| Residual inventory | 01 | targeted `rg` scans, `cargo check -p renderer`, `cargo check -p renderer --examples` | `cargo test -p renderer --no-run` if needed to separate compile from doctest failures | `reports/phase-01-residual-inventory.md`, `validation/phase-01-validation-report.md` |
| Vulkan lifecycle hardening | 02 | `cargo fmt --check`, `cargo check -p renderer`, `cargo check -p renderer --examples`, targeted Vulkan tests if added | renderer runtime smokes for touched examples, shutdown reproduction, headless capture only if output changes | `reports/phase-02-vulkan-lifecycle-hardening.md`, `validation/phase-02-validation-report.md`, debug reports |
| Runtime panic/stall hardening | 03 | `cargo fmt --check`, `cargo check`, `cargo test -p renderer`, `cargo check -p renderer --examples` | package/app tests for touched crates, debug-record timing smokes, capture only if visible behavior changes | `reports/phase-03-runtime-panic-stall-hardening.md`, `validation/phase-03-validation-report.md` |
| Docs/examples/test drift | 04 | `cargo fmt --check`, `cargo check -p renderer --examples`, `cargo test -p renderer`, stale doc scans | `cargo doc -p renderer --no-deps`, `cargo test -p engine_pack`, `cargo test -p editor`, `cargo test -p dungeon_dogfood` if touched | `reports/phase-04-docs-examples-test-drift.md`, `validation/phase-04-validation-report.md` |
| Final residual acceptance | 05 | full validation set, stale-reference sweep, evidence consistency check | rerun failed/nearby checks after remediation, capture reconciliation if any capture exists | `reports/phase-05-final-residual-acceptance.md`, `validation/phase-05-validation-report.md`, `validation/final-quality-review.md` |

## Full Validation Set

```sh
cargo fmt --check
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo test -p renderer
```

## Runtime Smoke Set

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_pbr-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_unlit-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_model_load-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_async_loading-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-env-timing.jsonl
```

## Stale Sweep

```sh
rg -n "pending|planned|not implemented|/tmp|desktop screenshot|fully_validated|final_quality_review_passed|TOOLING_CONSTRAINT" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
rg -n "gap-report|old image views|todo!\\(\\)|destroy paths|VkSubAllocator::destroy|fence\\[0\\]|double free" docs/api docs/internal .internal-dev/bugs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
```

## Evidence Status Rules

- `planning_locked`: plan exists, no implementation yet.
- `phase_XX_implementation_complete_validation_pending`: worker finished, validator not done.
- `phase_XX_validated`: validator passed the phase.
- `phase_XX_failed`: validator found defects.
- `final_quality_pending`: implementation and phase validation are not enough for closeout.
- `final_quality_review_passed_with_residuals`: final validator accepts named residuals.
- `fully_validated`: all required checks and validators pass with no unresolved residuals.
- `blocked_tooling_constraint`: required model/tool/capture unavailable and no approved fallback exists.

## Validator Minimum Duties

Every validator must check:

- plan criteria;
- application architecture fit;
- protected local state;
- command evidence;
- test quality;
- docs drift;
- evidence index consistency;
- residual classification honesty.
