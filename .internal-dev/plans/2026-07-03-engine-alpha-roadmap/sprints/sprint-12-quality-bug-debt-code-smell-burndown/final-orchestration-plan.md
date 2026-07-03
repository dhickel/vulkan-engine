# Final Orchestration Plan

## Dispatch Order

Run phases sequentially. Do not start Sprint 12 execution until the main thread reconciles Sprint 09 state and prepares `sprint/alpha-12-quality-bug-debt-code-smell-burndown`.

1. Phase 01 worker: `worker-directives/phase-01-residual-inventory-triage.md`
2. Phase 01 validator: `validation/phase-01-validation-report.md`
3. Main thread reviews whether inventory requires plan revision or sprint split.
4. Phase 02 worker: `worker-directives/phase-02-vulkan-lifecycle-hardening.md`
5. Phase 02 validator: `validation/phase-02-validation-report.md`
6. Phase 03 worker: `worker-directives/phase-03-runtime-panic-stall-hardening.md`
7. Phase 03 validator: `validation/phase-03-validation-report.md`
8. Phase 04 worker: `worker-directives/phase-04-docs-examples-test-drift.md`
9. Phase 04 validator: `validation/phase-04-validation-report.md`
10. Phase 05 worker: `worker-directives/phase-05-final-residual-acceptance.md`
11. Phase 05 validator: `validation/phase-05-validation-report.md`
12. Final quality validator: `validation/final-quality-review.md`

## Model Defaults

Use the configured planning/orchestration defaults unless the user overrides them at dispatch time:

- implementation worker: `gpt-5.3`, high reasoning;
- targeted repair worker: fresh `gpt-5.3`, high reasoning;
- second failure escalation repair: fresh `gpt-5.5`, high reasoning;
- phase validator: `gpt-5.5`, high reasoning;
- final quality validator: `gpt-5.5`, xhigh reasoning.

If a required model/tool cannot be selected, record `TOOLING_CONSTRAINT` and stop for user approval before fallback.

## Phase Gates

Each phase must have:

- worker report under `reports/`;
- validator report under `validation/`;
- updated `artifacts/validation-summary.json`;
- command evidence;
- residual disposition;
- protected local state check.

Main-thread branch push, commit, email, and tracker reconciliation are out-of-band responsibilities. This plan does not update `SPRINT-TRACKER.md`.

## Remediation Routing

- `code_defect`: fresh scoped repair worker using selected worker model unless the issue is a trivial mechanical miss.
- `docs_or_evidence_defect`: fresh scoped repair worker unless the validator can safely make a one-place validator artifact correction.
- `browser_harness_defect`: not expected. For renderer capture harness defects, repair capture/evidence first; change product code only after evidence proves a product bug.
- `plan_defect`: return to planning for revised criteria/directives.
- `validator_error`: correct checklist or use fresh validator before product repair.

If the same targeted issue fails validation twice after repair attempts, dispatch a fresh escalation repair worker with the selected escalation model.

## Stop And Split Gates

Return to planning or user decision if:

- Phase 01 identifies multiple critical defect families that cannot fit safely in one burn-down sprint.
- Vulkan lifecycle remediation requires broad ownership redesign.
- A fix requires public API breakage.
- Required engine-owned headless capture is blocked and no approved fallback exists.
- Critical residual acceptance is requested without mitigation.

## Final Quality Review Gate

The final validator must compare:

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- shared guidance and validation matrix;
- every worker directive;
- every worker report;
- every phase validation report;
- changed source/docs/tests;
- runtime debug reports;
- capture evidence if applicable;
- bug artifacts;
- `artifacts/validation-summary.json`.

Final pass requires:

- no unresolved critical residuals without explicit acceptance;
- validation matrix green or named residuals;
- no unearned `fully_validated`;
- docs/examples aligned with current alpha behavior;
- protected local state untouched;
- Sprint tracker untouched.

## Closeout Gates

Before final user report:

- Ask the user whether it is time to create `.internal-dev/changelogs/` entries required by repo guidance.
- Ask before creating `.internal-dev/notes/` future-consideration entries.
- Ensure any out-of-scope bugs discovered during execution have focused `.internal-dev/bugs/` reports.
- Leave `SPRINT-TRACKER.md` reconciliation to the main thread as requested.

## Capture Gate

If visible renderer/editor behavior changed:

- read `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`;
- run true engine-owned headless capture;
- store artifacts under `.internal-dev/captures/sprint-12-quality-burndown/`;
- reconcile screenshots/logs/metadata in validator reports;
- do not use desktop screenshots as proof.
