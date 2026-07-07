# Validation Plan

Each phase validator writes one report:

- `phase-01-validation-report.md`
- `phase-02-validation-report.md`
- `phase-03-validation-report.md`
- `phase-04-validation-report.md`
- `phase-05-validation-report.md`
- `final-quality-review.md`

## Validator Responsibilities

- Compare worker output against `00-specification-lock.md`, `02-target-design.md`, and the phase directive.
- Check application contract fit, architecture fit, regression risk, docs drift, and `.internal-dev` evidence quality.
- Re-run or independently verify critical commands where practical.
- Inspect capture PNG and JSON sidecars when a phase claims visual proof.
- Reconcile `artifacts/validation-summary.json` status with actual evidence.
- Fail validation on protected-path edits, tracker mutation, unearned final status, or desktop screenshot proof.

## Required Report Sections

- Scope
- Criteria Checked
- Commands Run
- Evidence Inspected
- Capture Proof Status
- Findings
- Remediation Handoff
- Residual Risk
- Verdict

## Final Quality Review

Run only after all phase validators pass or accepted residuals are explicitly recorded. The final quality validator must compare:

- all plan suite files;
- worker reports;
- phase validation reports;
- release docs and known issues;
- changed code/tests/docs;
- capture evidence;
- clean validation evidence;
- `artifacts/validation-summary.json`.

Final pass is not a release publication. It means the release-candidate branch is ready for main-thread review, tracker reconciliation, optional changelog creation, commit/push, and any external release process the user requests.

