# Validation README

## Required Reports

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`
- `validation/final-quality-review.md`

## Validator Responsibilities

Each phase validator must check:

- plan criteria and phase directive compliance;
- architecture boundaries and dependency hygiene;
- test quality, not only command pass/fail;
- docs/code drift;
- evidence index consistency;
- residual risk and inherited blocker handling;
- `.idea/engine.iml` and `.reasonix/` untouched.

## Capture Validation

Capture is not required for non-visual Sprint 08 work. If visible renderer/editor behavior changes, validators must require true engine-owned headless capture with `--headless --capture_target draw`. Desktop screenshots fail the capture criterion.

## Final Quality Review

After all phase validators pass, run a final quality validator. It must compare:

- all plan files;
- worker reports;
- validation reports;
- final code/docs/tests;
- `artifacts/validation-summary.json`;
- capture/debug artifacts if any;
- stale-reference sweep results.

The final review either passes the suite or writes a remediation handoff.
