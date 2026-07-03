# Final Orchestration Plan

## Dispatch Sequence

1. Dispatch Phase 01 worker with `worker-directives/phase-01-core-event-crate-api.md`.
2. Run Phase 01 validator and write `validation/phase-01-validation-report.md`.
3. If passed, main thread commits and pushes Phase 01.
4. Dispatch Phase 02 worker with `worker-directives/phase-02-renderer-runtime-integration.md`.
5. Run Phase 02 validator and write `validation/phase-02-validation-report.md`.
6. If passed, main thread commits and pushes Phase 02.
7. Dispatch Phase 03 worker with `worker-directives/phase-03-apps-samples-docs.md`.
8. Run Phase 03 validator and write `validation/phase-03-validation-report.md`.
9. If passed, main thread commits and pushes Phase 03.
10. Dispatch Phase 04 validation/closeout worker with `worker-directives/phase-04-validation-closeout.md`.
11. Run final quality validator after Phase 04 evidence is present.
12. If final quality passes, main thread handles changelog timing, tracker closeout, commit/push, and final email/report.

## Remediation Routing

- Code defect: fresh scoped repair worker using the selected worker model for the failed target.
- Docs/evidence defect: fresh scoped repair worker unless the validator can safely make a trivial one-place correction.
- Headless capture harness defect: repair harness/evidence path first; product code changes only after proof of a real product bug.
- Plan defect: return to planning for revised criteria/directives before more coding.
- Validator error: correct checklist or use a fresh validator before dispatching product repair.
- Same targeted issue failing twice: escalate to a fresh high-reasoning repair worker.

## Required Validation Reports

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`
- `validation/final-quality-review.md`

## Required Evidence Index

`artifacts/validation-summary.json` must record:

- top-level status;
- phase validation status;
- command results;
- runtime smoke evidence;
- draw capture evidence;
- superseded artifacts if any;
- model/tooling constraints;
- residual risks;
- final validator result.

## Closeout Gates

- All phase validators pass.
- Final quality validator passes.
- True headless draw capture proof is present and inspected.
- Docs stale sweep is clean or residuals are tracked.
- No unrelated `.idea/engine.iml` or `.reasonix/` changes are included.
- Main thread confirms whether to create changelog now under `.internal-dev/changelogs/`.
- Main thread updates `SPRINT-TRACKER.md` to `closed` only after evidence, changelog decision, commit/push, and email/report gates.

## Main Thread Email/Report Responsibilities

The main thread, not workers, should send any final email/report. The report should include:

- branch name: `sprint/alpha-05-event-system-lifecycle`;
- phase commit/push summary;
- validation commands and pass/fail status;
- draw capture evidence path;
- debug timing evidence path;
- final status from `validation-summary.json`;
- residual risks or follow-up bug/note paths.

## Stop Rules For Orchestrator

- Stop before implementation if a required worker/validator model cannot be selected and no user-approved fallback exists.
- Stop between phases when a validator fails until remediation is validated.
- Stop final closeout if `validation-summary.json` status is more optimistic than evidence.
- Stop if headless draw capture cannot be produced and ask the user/main thread before accepting fallback evidence.
