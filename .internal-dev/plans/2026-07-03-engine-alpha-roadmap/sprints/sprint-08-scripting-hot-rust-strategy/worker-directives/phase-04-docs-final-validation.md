# Phase 04 Worker Directive: Docs And Final Validation

## Objective

Align docs with implemented Sprint 08 behavior, run final validation, reconcile residuals, and prepare closeout evidence without overstating support.

## User-Visible Outcome

The repo tells users the correct alpha extension story and includes validation evidence for the claims Sprint 08 makes.

## Editable Targets

- Relevant docs under `docs/api/` and `docs/internal/`
- `README.md`
- `artifacts/validation-summary.json`
- `reports/phase-04-email.md`
- Optional `reports/final-email.md`
- Validation artifacts under `validation/`

## Forbidden Scope

- Do not add new product behavior in this phase except trivial docs/evidence fixes.
- Do not mark Sprint 08 closed in the tracker unless main thread explicitly owns that closeout.
- Do not create changelog unless user/main thread confirms timing.
- Do not use desktop screenshots.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All prior phase reports and validation reports.
- `shared/validation-matrix.md`
- `final-orchestration-plan.md`
- Current docs touched by phases 02 and 03.
- Top-level `AGENTS.md` and `.internal-dev/AGENTS.md`.

## Senior-Engineer Guidance

- Docs must say less rather than more when support is partial.
- "Experimental" and "deferred" are meaningful status words; use them deliberately.
- The evidence index is a contract. If residuals remain, the top-level status must reflect that.
- Capture is not a ritual. Only require it for visible renderer/editor behavior changes.

## Ordered Implementation Steps

1. Review Phase 01-03 implementation and validation artifacts.
2. Update docs to match implemented app-template and scripting support.
3. Remove or rewrite stale claims that still say implemented features are deferred, or vice versa.
4. Run stale-reference sweep over docs and this sprint directory for old paths, `/tmp` evidence, stale "pending/planned/not implemented" wording, TODOs, and phase wording that no longer matches.
5. Run required validation commands from `shared/validation-matrix.md`.
6. Run true headless draw capture only if visible renderer/editor behavior changed.
7. Update `artifacts/validation-summary.json` conservatively.
8. Draft `reports/phase-04-email.md` and optional final email draft.

## Acceptance Criteria

- Docs clearly state Rust app crates primary, scripts experimental, hot Rust reload deferred/tooling-only.
- Docs and code agree on app-template status.
- Docs and code agree on script asset/event support.
- Required validation commands are run or inherited blockers are recorded.
- Evidence summary is internally consistent.
- Final quality validator can review without reconstructing missing context.

## Negative Checks

- No `fully_validated` status unless all required validators and capture gates pass and residuals are accepted consistently.
- No stale docs claiming generated templates/scripting are both implemented and deferred.
- No desktop screenshots.
- No product scope creep.

## Validation Commands

```bash
cargo fmt --check
cargo check
cargo test -p scripting
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
rg -n "/tmp|pending|planned|not implemented|TODO|desktop screenshot|generated app templates|scripting runtime|hot Rust|dynamic Rust" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy
```

Conditional:

```bash
cargo test -p dungeon_dogfood
```

Run only if Sprint 08 changed dogfood expectations.

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-04-validation-report.md`
- Final review path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/final-quality-review.md`
- Evidence index path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/validation-summary.json`
- Phase report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-04-email.md`

## Stop Conditions

- Stop for repair if docs and code disagree on implemented status.
- Stop for planning revision if prior phases changed scope materially.
- Stop if capture is applicable but the engine-owned capture path is unavailable; record tooling constraint.

## Do Not Close Unless

- Required reports exist.
- Evidence summary is conservative.
- Final quality review has enough context.
- Main thread has clear push/email/changelog handoff.
