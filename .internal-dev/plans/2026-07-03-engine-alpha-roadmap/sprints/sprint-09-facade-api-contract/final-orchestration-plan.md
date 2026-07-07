# Final Orchestration Plan

## Dispatch Order

Run phases sequentially. Each phase depends on the prior phase's implementation report and validation report.

1. Phase 01 worker: `worker-directives/phase-01-facade-surface-audit.md`
2. Phase 01 validator: write `validation/phase-01-validation-report.md`
3. Main thread branch/push and HTML email summary for Phase 01.
4. Phase 02 worker: `worker-directives/phase-02-alpha-prelude-and-example-contract.md`
5. Phase 02 validator: write `validation/phase-02-validation-report.md`
6. Main thread branch/push and HTML email summary for Phase 02.
7. Phase 03 worker: `worker-directives/phase-03-error-input-camera-material-docs-hardening.md`
8. Phase 03 validator: write `validation/phase-03-validation-report.md`
9. Main thread branch/push and HTML email summary for Phase 03.
10. Phase 04 worker: `worker-directives/phase-04-docs-final-validation.md`
11. Phase 04 validator: write `validation/phase-04-validation-report.md`
12. Main thread branch/push and HTML email summary for Phase 04.
13. Final quality validator after all phase validators pass or explicitly accepted residuals are recorded.

## Model Defaults

Use the currently configured planning/orchestration defaults unless the user overrides them at dispatch time:

- implementation worker: `gpt-5.3`, high reasoning;
- targeted repair worker: fresh `gpt-5.3`, high reasoning;
- second failure escalation repair: fresh `gpt-5.5`, high reasoning;
- phase validator: `gpt-5.5`, high reasoning;
- final quality validator: `gpt-5.5`, xhigh reasoning.

If a required model/tool cannot be selected, record `TOOLING_CONSTRAINT` and stop for user approval before fallback.

## Phase Gates

Each phase must have:

- worker report under `reports/`;
- phase validation report under `validation/`;
- updated `artifacts/validation-summary.json`;
- command evidence and residuals;
- protected local state check;
- main-thread branch push;
- main-thread HTML email summary.

Email sending, waiting, and branch push mechanics are out-of-band main-thread responsibilities, not worker tasks.

## Remediation Routing

- Code defect: fresh scoped repair worker using the selected worker model unless the issue is a trivial mechanical miss.
- Docs/evidence defect: fresh scoped repair worker unless the validator can safely make a one-place validator artifact correction.
- Capture harness defect: repair harness/evidence first; change product code only after evidence proves a product bug.
- Plan defect: return to planning for revised criteria/directives.
- Validator error: correct checklist or use a fresh validator before product repair.

If the same targeted issue fails validation twice after repair attempts, dispatch a fresh escalation repair worker with the selected escalation model.

## Final Quality Review Gate

The final validator must compare:

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- all phase directives;
- all reports;
- all phase validation reports;
- changed code/docs/tests;
- command outputs summarized in evidence;
- capture artifacts if applicable;
- `artifacts/validation-summary.json`.

Final pass requires:

- docs and examples agree on the supported beginner facade;
- public export classification is visible and truthful;
- examples compile or residual failures are accepted and not caused by Sprint 09;
- no unearned final status in evidence;
- no hidden desktop screenshot proof;
- no protected local state edits.

## Closeout Gates

Before final user report:

- Ask the user whether it is time to create `.internal-dev/changelogs/` and any `.internal-dev/knowledge/`, `.internal-dev/notes/`, or `.internal-dev/bugs/` entries required by repo policy.
- Do not create out-of-scope future consideration notes without user approval.
- Keep accepted residuals explicit, especially Sprint 08 residuals and any pre-existing renderer doctest/prose issues.
- If a GitHub PR or issue mirroring is needed, main thread handles it after final validation.

## Capture Gate

No capture is expected for docs/API/examples-only work. If any phase changes visible renderer/editor behavior:

- use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`;
- use true engine-owned capture with `--headless --capture_target draw`;
- record artifacts under `.internal-dev/captures/`;
- reconcile capture results in the relevant phase validator and final quality review;
- do not use desktop screenshots as proof.
