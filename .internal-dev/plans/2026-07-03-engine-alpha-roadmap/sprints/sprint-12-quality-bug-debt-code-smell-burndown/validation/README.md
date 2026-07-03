# Validation README

## Validator Report Paths

- Phase 01: `validation/phase-01-validation-report.md`
- Phase 02: `validation/phase-02-validation-report.md`
- Phase 03: `validation/phase-03-validation-report.md`
- Phase 04: `validation/phase-04-validation-report.md`
- Phase 05: `validation/phase-05-validation-report.md`
- Final quality review: `validation/final-quality-review.md`

## Required Report Contents

Each phase validation report must include:

- phase directive reviewed;
- changed files inspected;
- commands run and results;
- evidence artifacts inspected;
- criteria pass/fail table;
- architecture and contract fit assessment;
- protected local state check;
- docs drift check;
- residual risks and acceptance state;
- remediation handoff if failed;
- final phase status.

## Browser/Playwright

Browser validation does not apply to this sprint. Renderer/editor visual behavior must use engine-owned headless capture through `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.

## Capture Reconciliation

If capture exists, validators must inspect:

- command line used;
- captured image/output path;
- sidecar metadata;
- logs;
- whether the capture proves the stated visual behavior.

Capture success alone is not a pass if the image/logs show a broken or irrelevant state.

## Evidence Index Reconciliation

Before passing any phase, update or verify:

`artifacts/validation-summary.json`

The validator must fail docs/evidence quality if the top-level status claims more than the phase reports support.

## Final Quality Review

Final quality review runs only after all phase validators pass or residuals are explicitly accepted. It must compare:

- all plan files;
- all worker reports;
- all validator reports;
- changed code/docs/tests;
- command and runtime evidence;
- capture evidence if present;
- residual risks;
- bug artifacts created or updated;
- evidence index consistency.

The final quality validator may pass with `final_quality_review_passed_with_residuals` only when residuals are non-critical or explicitly user-accepted with mitigation.
