# Final Orchestration Plan

## Dispatch Order

1. Phase 01 worker: `worker-directives/phase-01-capture-contracts-and-cli.md`
2. Phase 01 validator writes `validation/phase-01-validation-report.md`
3. Phase 02 worker: `worker-directives/phase-02-vulkan-capture-windowed.md`
4. Phase 02 validator writes `validation/phase-02-validation-report.md`
5. Phase 03 worker: `worker-directives/phase-03-headless-offscreen-capture.md`
6. Phase 03 validator writes `validation/phase-03-validation-report.md`
7. Phase 04 worker: `worker-directives/phase-04-manual-and-editor-integration.md`
8. Phase 04 validator writes `validation/phase-04-validation-report.md`
9. Phase 05 worker: `worker-directives/phase-05-docs-validation-evidence.md`
10. Phase 05 validator writes `validation/phase-05-validation-report.md`
11. Final quality validator writes `validation/final-quality-review.md`

No implementation phase should start until the prior mutating phase is validated or an approved remediation/gate decision is recorded.

## Required Models

- Implementation workers: default `gpt-5.5`, high reasoning unless user overrides.
- Phase validators: default `gpt-5.5`, high reasoning.
- Final large-suite quality validator: default `gpt-5.5`, xhigh reasoning.
- No browser/Playwright agent applies.

If a required model/tool is unavailable, record `TOOLING_CONSTRAINT` and stop for user approval before using a fallback.

## Validation Gates

Every phase validator must check:

- phase directive acceptance criteria;
- locked spec compatibility;
- architecture fit;
- regression risk;
- tests/commands and evidence;
- stale docs or artifact references introduced by the phase.

Final quality validator must compare:

- this plan suite;
- all phase validation reports;
- final code/tests/docs;
- `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`;
- PNG artifacts under `.internal-dev/debug_reports/`;
- compile/check command evidence;
- residual risks and user-decision gates.

## Remediation Routing

- `code_defect`: fresh scoped repair worker with selected implementation model unless the issue is a trivial one-place correction.
- `docs_or_evidence_defect`: fresh scoped repair worker unless a validator can safely make a simple local correction.
- `browser_harness_defect`: not applicable; no browser proof.
- `validator_error`: correct checklist or use a fresh validator.
- `plan_defect`: return to planning before more coding.

If the same targeted issue fails validation twice after repair attempts, escalate to a fresh scoped repair worker using the default escalation model `gpt-5.5`, high reasoning, unless user overrides.

## Headless Gate

If phase 03 proves true headless/offscreen requires broader architecture work than the bounded target abstraction, orchestration must stop and ask the user. The handoff must include:

- exact blocker;
- files/functions causing the blocker;
- whether hidden-window/windowed capture is the proposed fallback;
- what validation would and would not prove;
- estimate of extra architecture work if continuing true headless.

Do not proceed with windowed-only fallback as final scope without user approval.

## Final Required Evidence

The final evidence index must be:

- `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`

It must record:

- top-level conservative status;
- compile gates;
- parser/scheduler tests;
- runtime capture matrix rows;
- N-frame validation;
- headless validation or approved gate/fallback;
- manual capture validation;
- phase validator pass/fail;
- final quality review result;
- model/tooling constraints;
- superseded artifacts;
- residual risks.

## Closeout Gates

Before final user report:

- all phase reports exist;
- final quality review exists;
- evidence index is internally consistent;
- stale-reference sweep completed;
- no required PNG proof relies on desktop screenshot tooling;
- repo instruction about `.internal-dev` closeout artifacts is honored by asking the user whether to create changelog/knowledge/notes/bug records.

Email/remote updates are out-of-band main-thread responsibilities and are intentionally not part of worker directives.
