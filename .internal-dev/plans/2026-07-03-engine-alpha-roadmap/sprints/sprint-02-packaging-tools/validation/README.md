# Validation README

Each phase validator must write one report:

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`

Final validator writes:

- `validation/final-quality-review.md`

## Validator Scope

Validators must check:

- plan criteria and negative criteria;
- application contract fit for project/package/scene identity;
- architecture fit and shared validation reuse;
- test quality and fixture coverage;
- docs drift;
- evidence completeness;
- branch, commit, push, and AgentMail gates;
- capture decision correctness.

## Required Report Sections

- Phase target and status.
- Files created/changed/deleted.
- Changed files/line counts/git links matrix.
- Commands run and results.
- Validation/capture evidence paths.
- Commit hash.
- Pushed ref.
- GitHub commit and compare links or explanation if unavailable.
- AgentMail HTML report evidence.
- Findings, residuals, and remediation handoff.
- Conservative final status.

## Capture Reconciliation

Default status is:

```text
not_required_cli_schema_only
```

If implementation changes renderer-visible behavior, scene runtime loading semantics, asset placement/readiness, Vulkan behavior, camera/material/shader output, or claims visual proof, the validator must require `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` evidence before passing.

## Evidence Summary

Phase 04 owns `artifacts/validation-summary.json`. It must not say `fully_validated` unless all required checks, validators, pushes, emails, and capture decisions passed with no residuals.
