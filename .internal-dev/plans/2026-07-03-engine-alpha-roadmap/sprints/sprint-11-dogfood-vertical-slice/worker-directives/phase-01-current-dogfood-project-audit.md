# Phase 01 Worker Directive: Current Dogfood And Project Contract Audit

## Objective

Audit live dogfood, package/project/runtime, active sprint overlap, and validation contracts before product changes. Produce a precise implementation map and API friction inventory.

## User-Visible Outcome

Later workers know exactly what dogfood content can migrate to package/project/scene contracts, what must remain custom Rust app behavior, and what is blocked by Sprint 09/Sprint 10 or existing engine limitations.

## Editable Files And Artifacts

Editable:

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/api-friction.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/migration-debt.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/validation/phase-01-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/artifacts/validation-summary.json`

Read-only:

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- `.internal-dev/skills/engine-alpha-sprint/SKILL.md`
- `apps/dungeon_dogfood/**`
- `apps/editor/sample_project/**`
- `docs/api/**`
- `tools/engine_pack/**`
- `src/runtime.rs`
- `src/launch.rs`
- renderer package/project/scene validation modules

Forbidden:

- Product code changes.
- `SPRINT-TRACKER.md`.
- `.idea/engine.iml`.
- `.reasonix/`.

## Supporting Docs To Read

- This suite's `00-specification-lock.md`, `01-current-state-analysis.md`, `02-target-design.md`.
- `docs/api/10-packaging-cli.md`.
- `docs/api/11-runtime-project-launcher.md`.
- `apps/dungeon_dogfood/README.md`.

## Ordered Steps

1. Run `git status --short` and record dirty files relevant to Sprint 09/Sprint 11.
2. Inspect dogfood manifests, README, source modules, and asset tree.
3. Inspect editor sample project/package/scene as the live format reference.
4. Inspect `engine_pack` command behavior and renderer validation APIs.
5. Compare current dogfood `content_pack.toml` fields to package/project/scene expressiveness.
6. Identify exact unsupported concepts that need either schema/API work or migration debt.
7. Create `reports/api-friction.md` with contract/API pain points and recommended owner sprint.
8. Create `reports/migration-debt.md` with dogfood-only behavior allowed to remain after Sprint 11, if any.
9. Update `artifacts/validation-summary.json` phase 01 status to `audit_complete_pending_validation` only after the report artifacts exist.
10. Write `validation/phase-01-validation-report.md`.

## Senior-Engineer Guidance

- Fact: current dogfood README says broad project-manifest migration is deferred. Logic: this sprint must either change that truth or document a deliberate exception.
- Fact: project/package schema already exists in renderer validators. Logic: do not invent a dogfood validator.
- Fact: Sprint 09 is active. Logic: classify shared renderer API/example files as sequencing hazards.
- Fact: this phase is non-mutating for product code. Logic: stop if you need code edits to answer the audit.

## Acceptance Criteria

- API friction report exists and distinguishes bugs, missing contracts, docs drift, and accepted app-owned behavior.
- Migration debt report exists and states whether `content_pack.toml` should be removed, replaced, or retained transitionally.
- Phase 02/03/04 directives are still executable after audit; if not, document required plan revision.
- No product code, tracker, IDE, or `.reasonix` files changed.

## Negative Checks

- No command output claims validation of code that was not run.
- No vague "needs cleanup" entries without target file/module and next action.
- No recommendation to use desktop screenshots for visual proof.

## Validation Commands

```sh
git status --short
rg -n "content_pack|engine.project|PackageManifest|validate-project|--headless|capture_target|draw|record_debug" apps docs/api tools src -g '!target'
test -f .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/api-friction.md
test -f .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/migration-debt.md
```

## Stop Conditions

- Stop if Sprint 09 active changes make the dogfood contract impossible to audit from current files.
- Stop if `.internal-dev/AGENTS.md` or required sprint files are missing.
- Stop and ask main thread if the audit shows package/project contracts cannot represent required dogfood content without a schema decision.

## Evidence Expectations

- Record file paths inspected.
- Record git status snapshot.
- Record whether each dogfood content category maps to package/project/scene now.
- Validation report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/validation/phase-01-validation-report.md`.

## Do Not Close Unless

- Reports are written.
- Validation summary is updated conservatively.
- Sequencing hazards are named.
- Validator can reproduce the audit from cited files and commands.
