# Validation Matrix

| Gate | Phase | Required Commands | Conditional Commands | Evidence |
| --- | --- | --- | --- | --- |
| Export audit | 01 | `cargo check -p renderer`, targeted `rg` export scans | `cargo doc -p renderer --no-deps` if rustdoc annotations change | `validation/phase-01-validation-report.md`, `reports/phase-01-export-audit.md` |
| Prelude/examples | 02 | `cargo fmt --check`, `cargo check -p renderer --examples`, `cargo test -p renderer` | `cargo doc -p renderer --no-deps` if rustdoc/prelude docs added | `validation/phase-02-validation-report.md`, `reports/phase-02-example-contract.md` |
| Targeted hardening | 03 | `cargo fmt --check`, `cargo check`, `cargo test -p renderer`, `cargo check -p renderer --examples` | `cargo test -p input`, `cargo test -p engine_pack`, runtime debug smoke, headless capture if visible behavior changes | `validation/phase-03-validation-report.md`, `reports/phase-03-friction-hardening.md` |
| Docs/evidence finalization | 04 | `cargo fmt --check`, `cargo check`, `cargo test -p renderer`, `cargo check -p renderer --examples`, stale scans | `cargo test -p engine_pack`, `cargo doc -p renderer --no-deps`, capture reconciliation if needed | `validation/phase-04-validation-report.md`, `reports/phase-04-final-docs-validation.md`, `artifacts/validation-summary.json` |
| Final quality review | final | Review all phase reports, code diff, docs, evidence index, residuals | Rerun failed/nearby checks only after first full pass | final validator report path chosen by orchestrator |

## Required Stale Scans

Run before final validation:

```sh
rg -n "TODO|pending|planned|not implemented|/tmp|sprint-08|Sprint 08|sprint-04|headless-draw" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract
rg -n "stable public surface|Everything below api|advanced-interop|prelude|SceneWorld|CommandHistory|AnimationPlayer" docs/api src/renderer/src src/renderer/examples src/renderer/tests
```

## Evidence Status Rules

- Use `planning_locked` before implementation.
- Use `phase_N_implementation_complete_validation_pending` only after a worker finishes but before validation.
- Use `phase_N_validated` only after the validator report passes.
- Use `final_quality_review_passed` only after final quality review passes with residuals.
- Use `fully_validated` only if all required checks, validators, and capture requirements pass with no unresolved residual risks.
- Use `blocked_tooling_constraint` when required tooling/capture/model behavior is unavailable.
