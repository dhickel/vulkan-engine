# Work Units

Date: 2026-07-07
Status: execution-ready

## Dispatch Order

1. `worker-directives/phase-00-preflight-drift.md`
2. `worker-directives/phase-01-root-facade.md`
3. `worker-directives/phase-02-renderer-view-path.md`
4. `worker-directives/phase-03-app-owned-input.md`
5. `worker-directives/phase-04-app-owned-events.md`
6. `worker-directives/phase-05-dogfood-migration.md`
7. `worker-directives/phase-06-compat-docs-closeout.md`

Each phase is mutating and requires validation before the next dependent phase starts.

## Phase Dependency Map

- Phase 00 unlocks clean or classified baseline gates.
- Phase 01 creates root facade helpers used by later app migrations.
- Phase 02 creates no-dispatch renderer view path required before dogfood migration.
- Phase 03 migrates input semantics and action-event bridge before event bus ownership changes.
- Phase 04 moves app event bus ownership once input emission source is stable.
- Phase 05 proves the model in dogfood.
- Phase 06 updates docs/specs/changelog and runs final evidence closeout.

## Parallelism

Do not parallelize implementation phases. The phase boundaries intentionally protect input/event/camera ownership semantics.

Validators can run only after their phase worker reports completion and evidence.

## Commit Strategy

If main-thread policy requires commits, use one rollback-friendly commit per validated phase plus a final closeout commit if needed. Include code and related `.internal-dev` updates together for that phase.
