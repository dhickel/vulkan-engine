# Work Units

Execute phases in order. Every phase has its own worker directive, validator report, commit/push gate, and post-phase email report requirement.

| Phase | Directive | Objective | Depends On |
|---|---|---|---|
| 01 | `worker-directives/phase-01-process-baseline-audit.md` | Restore active `.internal-dev` process guidance and create the verified baseline inventory. | Planning suite |
| 02 | `worker-directives/phase-02-docs-gap-report-repair.md` | Repair current docs and retire stale gap-report truth. | Phase 01 |
| 03 | `worker-directives/phase-03-residual-register-validation-matrix.md` | Create consolidated residual register and reusable alpha validation matrix/evidence index. | Phase 02 |
| 04 | `worker-directives/phase-04-final-baseline-validation-closeout.md` | Run baseline validation, reconcile evidence, update tracker, and prepare closeout. | Phase 03 |

## Execution Rules

- Do not combine phases into one large commit.
- Do not proceed to the next phase until validator pass, pushed commit, and phase email are complete.
- Route plan defects back to the planning agent.
- Route docs/evidence defects to a fresh scoped repair worker unless the validator can safely make a one-place correction.
- Route product-code defects to residual tracking, not repair, unless the user explicitly expands Sprint 01.
