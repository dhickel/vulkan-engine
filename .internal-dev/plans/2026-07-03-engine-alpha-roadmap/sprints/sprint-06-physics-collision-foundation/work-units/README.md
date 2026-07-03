# Work Units

Execute phases in order. Each phase must be validated before the next dependent phase starts.

| Phase | Directive | Purpose | Depends On |
|---:|---|---|---|
| 01 | `worker-directives/phase-01-physics-crate-alpha-contract.md` | Build renderer-free physics IDs, descriptors, queries, and event records | None |
| 02 | `worker-directives/phase-02-package-scene-collision-metadata.md` | Add package/scene collision metadata and CLI validation | Phase 01 contract concepts |
| 03 | `worker-directives/phase-03-event-bridge-dogfood-proof.md` | Bridge physics outcomes to Sprint 05 events and make dogfood proof/debt decision | Phases 01-02 |
| 04 | `worker-directives/phase-04-docs-final-validation.md` | Update docs, run full validation, reconcile evidence, prepare final quality review | Phases 01-03 |

## Commit/Report Boundary

Main thread owns phase commits, pushes, sprint tracker updates, changelog timing, and email/report sending after validation. Workers and validators should write evidence only to the paths specified in their directives/reports.

