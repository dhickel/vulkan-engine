# Validation Plan

## Phase Reports

Each phase validator writes:

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`

Final quality review writes:

- `validation/final-quality-review.md`

## Validator Responsibilities

Validators must check:

- Phase directive criteria.
- Application contract fit.
- Architecture fit.
- Robustness and regression risk.
- Test quality and command output.
- Docs drift.
- `.internal-dev` evidence consistency.
- Unrelated dirty state preservation, specifically `.idea/engine.iml` and `.reasonix/`.

## Capture Reconciliation

Phase 03 and final validation must reconcile headless capture evidence from `.internal-dev/captures/`:

- command run;
- capture directory;
- PNG file paths;
- sidecar JSON paths;
- expected visual result;
- actual observed result;
- pass/fail/inconclusive judgment.

Do not mark visual proof passed if the PNG is missing, blank, unrelated to package placement, or not inspected.

## Final Validation

After all phase validators pass, run a final quality validator with high rigor. It must compare:

- plan suite;
- phase validation reports;
- code/test/docs changes;
- `artifacts/validation-summary.json`;
- capture evidence;
- changelog;
- sprint tracker/status updates;
- commit/push/email evidence recorded by the main thread.

Before final validation, run a stale-reference sweep over docs and `.internal-dev` for stale artifact paths, `/tmp` evidence paths, stale agent IDs, `pending`, `planned`, `not implemented`, `TODO`, and outdated phase wording. Do not remove honest residual-risk notes merely because they include pending status; reconcile them into final status language.
