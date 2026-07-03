# Validation Plan

Each mutating phase must receive a separate validator report:

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`

Validators must inspect:

- phase directive acceptance criteria;
- code/docs architecture fit;
- public API compatibility;
- examples compile contract;
- docs/code drift;
- validation command output;
- evidence index consistency;
- protected local state preservation;
- capture policy compliance when applicable.

## Validator Rules

- Validators are non-mutating unless fixing an obvious validator-side typo or stale reference in a validation artifact.
- If a code defect, API contract defect, docs/evidence defect, or capture harness defect is found, return it to the main thread for scoped repair routing.
- If criteria are ambiguous or flawed, return to planning before more coding.
- Use a fresh validator after criteria change, after remediation touches a new domain, after more than two failed cycles, or after the validator misses an obvious issue.

## Browser/Capture Applicability

No browser validation is expected. Renderer visual capture is only required if visible renderer/editor behavior changes. If required, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and true engine-owned headless draw capture.

## Final Review

After all phase validators pass or accepted residuals are explicitly recorded, run a final quality review. It must compare the plan suite, reports, validation reports, code diff, docs diff, command evidence, and `artifacts/validation-summary.json`.
