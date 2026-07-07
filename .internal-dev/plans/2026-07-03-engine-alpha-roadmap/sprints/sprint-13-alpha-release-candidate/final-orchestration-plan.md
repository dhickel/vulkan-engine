# Final Orchestration Plan

## Preconditions

- Main thread must confirm whether Sprint 13 may execute before Sprints 10-12 have planned/closed artifacts, or wait until those predecessor sprints are reconciled.
- Start execution from `sprint/alpha-13-alpha-release-candidate` or the user-approved integrated release base.
- Do not use dirty local Sprint 09 state as release proof.

## Dispatch Order

1. Phase 01 worker: `worker-directives/phase-01-release-inventory-docs-lock.md`
2. Phase 01 validator: `validation/phase-01-validation-report.md`
3. Phase 02 worker: `worker-directives/phase-02-fresh-clone-validation.md`
4. Phase 02 validator: `validation/phase-02-validation-report.md`
5. Phase 03 worker: `worker-directives/phase-03-sample-editor-runtime-proof.md`
6. Phase 03 validator: `validation/phase-03-validation-report.md`
7. Phase 04 worker: `worker-directives/phase-04-dogfood-run-visual-proof.md`
8. Phase 04 validator: `validation/phase-04-validation-report.md`
9. Phase 05 worker: `worker-directives/phase-05-release-notes-known-issues-final.md`
10. Phase 05 validator: `validation/phase-05-validation-report.md`
11. Final quality validator: `validation/final-quality-review.md`

## Model Defaults

Use current orchestration defaults unless the user overrides them:

- implementation worker: `gpt-5.3`, high reasoning;
- targeted repair worker: fresh `gpt-5.3`, high reasoning;
- second failure escalation repair: fresh `gpt-5.5`, high reasoning;
- phase validator: `gpt-5.5`, high reasoning;
- final quality validator: `gpt-5.5`, xhigh reasoning.

If a required model/tool cannot be selected, record `TOOLING_CONSTRAINT` in `artifacts/validation-summary.json` and stop for user approval before fallback.

## Phase Gates

Every phase must have:

- worker report under `reports/`;
- validator report under `validation/`;
- updated `artifacts/validation-summary.json`;
- command evidence;
- protected-path and tracker checks;
- residual classification.

Phases 03 and 04 additionally require capture evidence under `.internal-dev/captures/sprint-13-alpha-release-candidate/`.

## Remediation Routing

- Code defect: fresh scoped repair worker using the selected worker model unless the issue is trivial and mechanical.
- Docs/evidence defect: fresh scoped repair worker unless the validator can safely correct a one-place validation artifact typo/stale link.
- Capture harness defect: repair capture harness/evidence first; change product code only after evidence proves a product bug.
- Plan defect: return to planning for revised criteria/directives.
- Validator error: correct checklist or use a fresh validator before product repair.

If the same targeted issue fails validation twice after repair attempts, dispatch a fresh escalation repair worker with the selected escalation model.

## Final Quality Review Gate

Final validator must compare:

- this plan suite;
- all worker reports;
- all validation reports;
- changed code/docs/tests;
- release notes and known issues;
- clean validation evidence;
- capture evidence and sidecars;
- `artifacts/validation-summary.json`.

Final pass requires:

- no missing required phase reports;
- no protected-path edits;
- no tracker mutation by workers;
- no desktop screenshot visual proof;
- no unearned `fully_validated` status;
- all residuals classified as release-blocking or accepted alpha debt.

## Closeout Gates

Before final user report:

- Ask the user whether it is time to create `.internal-dev/changelogs/` and any `.internal-dev/knowledge/`, `.internal-dev/notes/`, or `.internal-dev/bugs/` entries required by repo policy.
- Leave `SPRINT-TRACKER.md` reconciliation to the main thread.
- If publishing/tagging/PR/email is needed, main thread handles it after final validation.

