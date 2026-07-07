# Final Orchestration Plan

## Execution Order

1. Dispatch Phase 01 worker with `worker-directives/phase-01-state-and-command-hardening.md`.
2. Run Phase 01 validator and write `validation/phase-01-validation-report.md`.
3. If passed, main thread performs scoped commit/push/email report and records evidence.
4. Dispatch Phase 02 worker.
5. Run Phase 02 validator and record report/evidence.
6. If passed, main thread performs scoped commit/push/email report and records evidence.
7. Dispatch Phase 03 worker.
8. Run Phase 03 validator. Validator must inspect capture artifacts before passing.
9. If passed, main thread performs scoped commit/push/email report and records evidence.
10. Dispatch Phase 04 worker.
11. Run Phase 04 validator.
12. Run final quality validator after all phase reports pass.
13. Main thread records final commit/push/email/changelog evidence and reports conservative final status to the user.

## Required Reports

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`
- `validation/final-quality-review.md`
- `artifacts/validation-summary.json`

## Remediation

- Failed implementation behavior: fresh scoped repair worker for that phase.
- Failed docs/evidence: scoped docs/evidence repair worker unless it is an obvious validator-safe one-place edit.
- Failed capture harness: repair capture harness/evidence first; change product code only if evidence proves product behavior is wrong.
- Ambiguous or flawed criteria: return to planning before more coding.
- Same targeted issue failing twice: escalate to a fresh high-reasoning repair worker.

## Closeout Gates

- All required validation reports exist and pass.
- `artifacts/validation-summary.json` has no contradiction between top-level status, phase statuses, capture status, residual risks, and commit/push/email evidence.
- Headless capture PNG/sidecar paths are recorded and inspected.
- Docs and `.internal-dev` stale-reference sweep is complete.
- Changelog is created under `.internal-dev/changelogs/` when main thread confirms closeout timing.
- Sprint tracker can move Sprint 03 to closed only after implementation, validation, changelog, commit, push, and report gates are complete.
- Sprint 01 remains untouched unless separately instructed.
