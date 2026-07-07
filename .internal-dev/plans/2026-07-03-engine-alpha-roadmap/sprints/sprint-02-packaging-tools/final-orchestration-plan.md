# Final Orchestration Plan

## Required Branch

Use `sprint/alpha-02-packaging-tools`. Stop if not on this branch.

Preserve unrelated dirty state:

- `.idea/engine.iml`
- `.reasonix/`

## Dispatch Order

1. Phase 01: `worker-directives/phase-01-shared-validation-contract.md`
2. Validate Phase 01 and write `validation/phase-01-validation-report.md`.
3. Commit scoped Phase 01 changes, push, and send Phase 01 AgentMail HTML report.
4. Phase 02: `worker-directives/phase-02-cli-validation-commands.md`
5. Validate Phase 02 and write `validation/phase-02-validation-report.md`.
6. Commit scoped Phase 02 changes, push, and send Phase 02 AgentMail HTML report.
7. Phase 03: `worker-directives/phase-03-authoring-and-pack-commands.md`
8. Validate Phase 03 and write `validation/phase-03-validation-report.md`.
9. Commit scoped Phase 03 changes, push, and send Phase 03 AgentMail HTML report.
10. Phase 04: `worker-directives/phase-04-docs-final-validation-closeout.md`
11. Validate Phase 04 and write `validation/phase-04-validation-report.md`.
12. Run final quality review and write `validation/final-quality-review.md`.
13. Commit/push closeout artifacts if changed, send final AgentMail HTML report, and report status to the user.

## Phase Commit/Push Gate

Every phase must have one or more scoped commits containing only in-scope files. After validation passes:

```bash
git push origin sprint/alpha-02-packaging-tools
```

Record in the phase validation report:

- commit hash;
- pushed ref;
- `git status --short --branch` after push;
- GitHub commit link or unavailable reason;
- GitHub compare link or unavailable reason;
- changed files/line counts/git links matrix.

## AgentMail Gate

After each pushed phase, send an HTML progress report. Use `email-report-template.html` and include:

- sprint target;
- phase status;
- files created/changed/deleted;
- line counts;
- commands run;
- validation/capture evidence paths;
- commit hash;
- pushed ref;
- GitHub commit and compare links;
- residuals.

Do not proceed to the next phase until the email is sent or a stop condition is recorded.

## Validation Routing

- Normal phase validation: fresh phase validation/red-team agent.
- Code defect: fresh scoped repair worker unless the issue is a trivial mechanical miss.
- Docs/evidence defect: fresh scoped repair worker unless validator can safely make an obvious one-place correction.
- Browser/Playwright: not applicable.
- Headless capture: conditional. Required only if renderer/scene/asset/Vulkan visual behavior changes or visual proof is claimed.
- Browser/capture harness defect: repair evidence/harness first; product code changes only after evidence proves a product bug.
- Plan defect: return to advanced planning for revised criteria/directives.
- Same targeted issue fails twice: stop and escalate to a fresh higher-reasoning repair pass.

## Final Quality Review

Final validator must compare:

- this plan suite;
- all worker directives;
- all phase validation reports;
- `artifacts/validation-summary.json`;
- code/tests/docs changed in the sprint;
- commit/push evidence;
- AgentMail evidence;
- capture decision/evidence.

Final status rules:

- `fully_validated`: only when all required checks, validators, pushes, emails, and capture decisions pass and no residuals remain.
- `final_quality_review_passed_with_residuals`: checks and evidence are acceptable but tracked residuals remain.
- `blocked_tooling_constraint`: required tooling, push, AgentMail, cargo, runtime, or capture evidence could not be completed.
- `final_quality_pending`: final validator did not run or has unresolved findings.

## Closeout Gates

- Tracker reflects actual Sprint 02 status.
- Sprint 01 blocked changelog timing remains visible and is not resolved by this sprint.
- Changelog creation is handled only after user confirmation per repo guidance.
- No unrelated dirty files are staged or committed.
- Final user report includes phase list, validation status, pushed ref, and residuals.
