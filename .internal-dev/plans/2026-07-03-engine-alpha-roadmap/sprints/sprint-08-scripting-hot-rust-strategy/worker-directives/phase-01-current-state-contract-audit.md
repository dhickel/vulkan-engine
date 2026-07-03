# Phase 01 Worker Directive: Current-State Contract Audit

## Objective

Produce a precise audit of existing app-crate, app-template, scripting, script-asset, script-event, and hot-reload claims before implementation begins.

## User-Visible Outcome

Sprint 08 starts from verified live source truth, not stale roadmap language.

## Editable Targets

- `artifacts/phase-01-current-state-contract-audit.md`
- `reports/phase-01-email.md`

## Forbidden Scope

- Do not edit product code, tests, schemas, runtime config, docs outside this plan directory, `.idea/engine.iml`, or `.reasonix/`.
- Do not implement template generation or script bindings in this phase.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- Top-level `AGENTS.md`
- `docs/api/00-index.md`
- `docs/api/01-student-quickstart.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/api/12-events-and-lifecycle.md`
- `src/scripting/src/lib.rs`
- `src/events/src/lib.rs`
- `tools/engine_pack/src/main.rs`

## Senior-Engineer Guidance

- Separate verified facts from recommendations.
- Treat code as logical source of truth and docs as intended truth.
- Identify claims that are still deferred versus claims Sprint 08 can safely promote.
- Call out any needed target-design correction before Phase 02.

## Ordered Steps

1. Inspect relevant code/docs/tests for current app/template/scripting/hot-reload state.
2. Confirm `src/scripting` public API and tests.
3. Confirm event vocabulary and dependency boundaries.
4. Confirm current `engine_pack` command surface and test layout.
5. Search docs for stale terms: `scripting runtime`, `generated app templates`, `hot reload`, `dynamic Rust`, `app crates`.
6. Write `artifacts/phase-01-current-state-contract-audit.md` with verified facts, drift, implementation opportunities, and blockers.
7. Draft `reports/phase-01-email.md` summarizing audit outcome and Phase 02 readiness.

## Acceptance Criteria

- Audit names exact files and claims to update.
- Audit confirms whether `script` asset kind exists or not.
- Audit confirms whether app template tooling exists or not.
- Audit records existing Sprint 07 residual handling.
- Phase report is ready for main-thread HTML email conversion.

## Negative Checks

- No product code/docs outside this plan directory changed.
- No broad `.internal-dev` scans.
- No new future-consideration note without user approval.

## Validation Commands

```bash
git status --short
rg -n "scripting runtime|generated app templates|hot reload|dynamic Rust|app crates|script" docs src tools apps -g '*.md' -g '*.rs' -g '*.toml'
```

No compile commands are required unless the worker needs to verify an audit claim.

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-01-validation-report.md`
- Phase report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md`

## Stop Conditions

- Stop and request planning revision if live code contradicts the Sprint 08 target design materially.
- Stop if the audit reveals a required user decision before app-template or scripting work can proceed.

## Do Not Close Unless

- Audit artifact exists.
- Phase report exists.
- No product behavior changed.
- Validator has enough evidence to approve Phase 02 dispatch.
