# Phase 04 Worker Directive: Final Docs, Evidence, And Quality Review Prep

## Objective

Reconcile Sprint 10 documentation, validation evidence, residual risks, and final status so the main thread can dispatch final validation without guessing what happened.

## User-Visible Outcome

Sprint 10 has a conservative final evidence package showing which advanced rendering contract work passed, what remains deferred, and whether runtime/capture proof was required and satisfied.

## Editable Targets

- `artifacts/validation-summary.json`
- `reports/README.md`
- `reports/final-closeout-notes.md`
- Changed docs only for stale-reference or wording corrections found during closeout.
- Phase validation reports if the validator permits simple docs/evidence corrections.

## Forbidden Scope

- Do not add new product behavior in Phase 04.
- Do not edit `SPRINT-TRACKER.md`; main thread owns tracker reconciliation.
- Do not create changelog unless the user/main thread confirms it is time.
- Do not edit `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All phase reports and validation reports.
- `shared/validation-matrix.md`
- `final-orchestration-plan.md`
- `.internal-dev/AGENTS.md`

## Senior Engineer Guidance

- Closeout is evidence reconciliation, not another implementation phase.
- Use conservative status language. Accepted residuals mean no `fully_validated`.
- If Phase 03 deferred advanced features, record that as an explicit product residual, not a validation failure by itself.
- If command results are missing, the status must say pending or blocked.

## Ordered Steps

1. Inspect phase reports, validation reports, command logs, and capture evidence if present.
2. Run final validation commands from the matrix unless already run after the last code change.
3. Run stale-reference sweep over changed docs and this sprint directory.
4. Update `artifacts/validation-summary.json` with command results, phase statuses, residual risks, tooling constraints, and final conservative status.
5. Write `reports/final-closeout-notes.md`.
6. Prepare handoff for final quality validator.

## Acceptance Criteria

- Validation summary is cross-field consistent.
- Every required phase has a report path and status.
- Missing or blocked checks are explicit.
- Residual risks are named and routed.
- No stale docs claim unsupported advanced features are complete.

## Negative Checks

- No `fully_validated` if any phase validator failed, any required capture is missing, or residuals remain unresolved.
- No `/tmp` paths as canonical durable evidence.
- No desktop screenshot proof.
- No stale `TODO` or "planned" language in final docs when work is actually complete.

## Validation Commands

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p renderer --features advanced-interop
cargo check -p renderer --examples --features advanced-interop
```

Run focused tests and runtime/capture commands from Phase 03 if they were required by implementation.

## Stop Conditions

- Stop if evidence is contradictory.
- Stop if a required command fails from a likely Sprint 10 regression.
- Stop if final status would require a model/tool substitution not approved by the user.

## Evidence Expectations

- `reports/final-closeout-notes.md`
- `validation/phase-04-validation-report.md`
- Updated `artifacts/validation-summary.json`

## Do Not Close Unless

- Final status is conservative and justified.
- The main thread can reconcile tracker/changelog decisions from the closeout notes.
- Final validator has enough evidence paths to review without rerunning discovery.
