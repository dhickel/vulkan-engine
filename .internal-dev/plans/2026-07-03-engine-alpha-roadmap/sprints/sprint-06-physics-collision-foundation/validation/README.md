# Validation README

## Required Reports

- `phase-01-validation-report.md`: physics crate API and dependency boundary.
- `phase-02-validation-report.md`: package/scene collision metadata and CLI validation.
- `phase-03-validation-report.md`: event bridge and dogfood proof/debt gate.
- `phase-04-validation-report.md`: docs, full validation, evidence reconciliation.
- `final-quality-review.md`: final large-suite quality validator report after all phase validators pass.

## Validator Requirements

Each phase validator must check:

- plan criteria and phase directive criteria;
- application contract and architecture fit;
- dependency boundaries;
- tests added and commands run;
- docs drift and stale claims;
- protected path hygiene for `.idea/engine.iml` and `.reasonix/`;
- `artifacts/validation-summary.json` consistency if touched.

## Browser/Capture Policy

This sprint does not require browser validation. It also does not require visual capture for pure physics or metadata work.

If implementation changes visible renderer/editor behavior, the validator must require true engine headless draw capture using `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` with `--headless --capture_target draw`. Desktop screenshots are not acceptable.

## Final Quality Review

After all phase validators pass, run a final quality validator with `gpt-5.5` xhigh unless the main thread has an explicit model override. The final validator compares:

- plan suite;
- phase directives;
- validation reports;
- code/tests/docs changes;
- `artifacts/validation-summary.json`;
- dogfood proof or migration debt artifact;
- stale-reference sweep results.
