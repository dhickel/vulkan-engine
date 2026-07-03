# Final Orchestration Plan

## Dispatch Order

1. Phase 01: `worker-directives/phase-01-process-baseline-audit.md`
2. Validate Phase 01 and write `validation/phase-01-validation-report.md`.
3. Commit, push, and send Phase 01 AgentMail HTML report.
4. Phase 02: `worker-directives/phase-02-docs-gap-report-repair.md`
5. Validate Phase 02 and write `validation/phase-02-validation-report.md`.
6. Commit, push, and send Phase 02 AgentMail HTML report.
7. Phase 03: `worker-directives/phase-03-residual-register-validation-matrix.md`
8. Validate Phase 03 and write `validation/phase-03-validation-report.md`.
9. Commit, push, and send Phase 03 AgentMail HTML report.
10. Phase 04: `worker-directives/phase-04-final-baseline-validation-closeout.md`
11. Validate Phase 04 and write `validation/phase-04-validation-report.md`.
12. Run final quality review and write `validation/final-quality-review.md`.
13. Commit, push, send final AgentMail HTML report, and report status to the user.

## Required Branch

Use `sprint/alpha-01-baseline-audit`. Stop if branch state or unrelated dirty changes make safe phase commits impossible.

## Phase Commit/Push Gate

Every phase must have one or more scoped commits containing only in-scope files. After validation passes:

```bash
git push origin sprint/alpha-01-baseline-audit
```

Record:

- commit hash;
- pushed ref;
- GitHub commit link;
- GitHub compare link when base branch can be formed;
- `git status --short --branch` after push.

## Email Gate

After each pushed phase, send Dwight an HTML email report via AgentMail. Use `email-report-template.html` and include:

- phase name/status;
- files created/changed;
- line counts where practical;
- commands run;
- validation/capture evidence paths;
- commit hash;
- pushed branch/ref;
- GitHub compare/commit links when remote URL can be formed;
- residuals/blockers.

Do not proceed to the next phase until the email is sent or a stop condition is recorded.

## Validation Routing

- Normal validation: fresh phase validation/red-team agent after each worker.
- Docs/evidence defect: fresh scoped repair worker unless validator can safely make an obvious one-place correction.
- Product-code defect: record residual/blocker; do not repair in Sprint 01 without user expansion.
- Browser/Playwright: not applicable.
- Headless capture: only if visual behavior changes or a phase explicitly requires visual proof.
- Plan defect: return to planning agent before more coding.
- Same targeted issue fails twice: stop and escalate to a fresh high-reasoning repair worker.

## Final Quality Review

Required after all phase validations pass or record accepted residuals. The final validator must compare:

- `00-specification-lock.md`;
- `01-current-state-analysis.md`;
- `02-target-design.md`;
- all worker directives;
- all phase validation reports;
- `artifacts/validation-summary.json`;
- current docs;
- register/review artifacts;
- git commit/push evidence;
- AgentMail evidence.

Final status must be conservative:

- `fully_validated` only when all required checks pass and no accepted residuals remain;
- `final_quality_review_passed_with_residuals` when checks pass or blockers are acceptable and residuals are tracked;
- `blocked_tooling_constraint` when required tooling, push, AgentMail, cargo, runtime, or capture evidence cannot be completed.

## Closeout Gates

- Tracker updated to the correct status.
- Changelog created only after required user confirmation or explicit orchestration authorization.
- Known residuals tracked in `.internal-dev/reviews/` or `.internal-dev/bugs/`.
- No stale gap-report current-truth references remain.
- No unrelated dirty changes staged or committed.
- Final user report summarizes created/changed files, validation status, pushed ref, and residuals.
