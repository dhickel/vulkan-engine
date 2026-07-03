# Work Units

Run phases sequentially. This is a release-candidate suite, so downstream phases depend on earlier contract and evidence decisions.

| Phase | Directive | Purpose | Depends On |
|---:|---|---|---|
| 01 | `worker-directives/phase-01-release-inventory-docs-lock.md` | Audit current release docs/contracts and lock the public release documentation shape | Sprint 09-12 status decision |
| 02 | `worker-directives/phase-02-fresh-clone-validation.md` | Prove clean checkout/worktree commands and collect baseline validation | Phase 01 docs/contract lock |
| 03 | `worker-directives/phase-03-sample-editor-runtime-proof.md` | Prove sample package/editor edit-save/root runtime workflow | Phase 02 clean source gate |
| 04 | `worker-directives/phase-04-dogfood-run-visual-proof.md` | Prove dogfood full-content run and true headless draw visual baseline | Phase 02 clean source gate; Phase 01 dogfood contract |
| 05 | `worker-directives/phase-05-release-notes-known-issues-final.md` | Draft release notes/known issues/workflow notes and prepare final review | Phases 01-04 validated |

## Dispatch Notes

- Do not dispatch Phase 02 until the main thread confirms the intended base branch and predecessor sprint status.
- Do not dispatch Phase 05 until Phase 03 and Phase 04 capture evidence has been reconciled by validators.
- If a phase finds a plan defect, return to planning rather than letting a worker invent new release criteria.

