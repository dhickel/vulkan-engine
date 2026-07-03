# Validation README

## Validator Role

Validators are non-mutating unless correcting an obvious one-place validator artifact defect. They must inspect plan criteria, code changes, command output evidence, capture metadata, docs consistency, and release risk.

## Phase Reports

Each phase validation writes:

- `phase-01-validation-report.md`
- `phase-02-validation-report.md`
- `phase-03-validation-report.md`
- `phase-04-validation-report.md`
- `phase-05-validation-report.md`

Required headings:

- `Scope`
- `Criteria`
- `Commands Run`
- `Evidence Inspected`
- `Findings`
- `Pass/Fail`
- `Residual Risk`
- `Remediation Handoff`

## Capture Validation

For Phase 04 and final review, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.

Desktop screenshots are invalid evidence for Sprint 11.

Required capture command properties:

- `--headless`
- `--capture_target draw`
- timeout-bound command
- capture output under `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`

Required inspection:

- PNG exists and is visually inspected;
- sidecar JSON exists and reports draw target;
- extent is positive;
- result is not blank or fully black;
- observed scene matches the visual contract.

## Final Quality Review

After phase validators pass, run a final `gpt-5.5` xhigh quality review unless user/model instructions override. The reviewer must reconcile:

- plan suite;
- phase validation reports;
- product diffs;
- docs;
- capture sidecars and images;
- debug reports;
- `.internal-dev/bugs/` entries created by the sprint;
- `artifacts/validation-summary.json`.

If any required evidence is missing, final status is `final_quality_pending` or `blocked_tooling_constraint`, not `fully_validated`.
