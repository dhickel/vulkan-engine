# Phase 05 Worker Directive: Docs, Evidence, And Final Validation Prep

## Objective

Align public docs, sprint reports, evidence index, and final validation inputs after implementation phases pass.

## User-Visible Outcome

The dogfood vertical slice has accurate clean-checkout instructions, validation commands, residuals, capture evidence references, and a final report draft for main-thread closeout.

## Editable Files

Likely editable:

- `apps/dungeon_dogfood/README.md`
- `docs/api/00-index.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- optional `docs/api/14-dogfood-vertical-slice.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/api-friction.md`
- `reports/migration-debt.md`
- `reports/final-report.md`
- `artifacts/validation-summary.json`
- `validation/phase-05-validation-report.md`

Forbidden:

- `SPRINT-TRACKER.md`.
- Changelog entry unless main thread confirms timing.
- `.idea/engine.iml`.
- `.reasonix/`.
- Product behavior changes except trivial docs/examples command typo fixes.

## Ordered Steps

1. Read all phase validation reports and implementation notes.
2. Update dogfood README with current commands, full-content mode, headless capture, known limitations, and clean-checkout validation.
3. Update API docs/index so packaging/runtime docs link to dogfood vertical slice where appropriate.
4. Ensure docs do not claim unsupported migration, production game completeness, or full validation.
5. Update `reports/api-friction.md` and `reports/migration-debt.md` to match final code.
6. Create `reports/final-report.md` with changed files summary, validation evidence, residuals, and closeout instructions.
7. Update `artifacts/validation-summary.json` with command results and evidence paths, using conservative final status such as `code_validation_passed_capture_pending`, `final_quality_pending`, or `ready_for_final_quality_review`.
8. Run stale-reference sweep and docs checks.
9. Write phase validation report.

## Senior-Engineer Guidance

- Docs are part of the contract for alpha users. If a command in docs cannot run, docs fail.
- Do not close residuals by wording them away. Link them to reports or bugs.
- Email/report closeout belongs to the main thread; provide draft content and evidence paths.
- Tracker update is explicitly out of scope.

## Acceptance Criteria

- Public docs include package/project validation, windowed run, debug timing, and headless draw capture commands.
- Docs state custom Rust app versus data-driven root launcher boundary accurately.
- Evidence summary JSON is internally consistent.
- Final report draft exists.
- Stale-reference sweep is clean or has named residuals.

## Negative Checks

- No `/tmp` evidence paths in final docs/evidence unless explicitly marked non-authoritative.
- No stale `pending`, `planned`, `not implemented`, or `TODO` claims in final user docs unless part of known limitations.
- No desktop screenshot references as validation.
- No `fully_validated` if final quality review has not passed.

## Validation Commands

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo check -p engine_pack
cargo test -p engine_pack
cargo check -p dungeon_dogfood
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
rg -n "/tmp|desktop screenshot|present-target|fully_validated|TODO|not implemented|pending|planned" docs/api apps/dungeon_dogfood/README.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice
```

Use judgment on the sweep: plan files may legitimately contain initial planned/pending history, but final reports and public docs must not misstate current status.

## Stop Conditions

- Stop if required phase reports are missing.
- Stop if docs require behavior that implementation did not provide.
- Stop if validation summary contradicts phase reports or capture evidence.
- Stop and ask before writing changelog.

## Evidence Expectations

- Docs files changed.
- Final report path.
- Validation summary diff/status.
- Stale-reference sweep results.
- Validation report path: `validation/phase-05-validation-report.md`.

## Do Not Close Unless

- Final quality validator has enough evidence to review without guessing.
- Main-thread closeout responsibilities are clearly listed.
- Tracker and changelog are untouched unless explicitly approved.
