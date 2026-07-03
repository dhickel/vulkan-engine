# Phase 04 Worker Directive: Docs And Final Validation

## Objective

Update public/internal documentation for the implemented alpha physics/collision contracts, run the full validation suite, reconcile evidence, and prepare final quality review.

## User-Visible Outcome

The sprint has clear docs, conservative evidence, and a final validator handoff that states what is supported now and what remains deferred.

## Editable Targets

- `docs/api/00-index.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/api/12-events-and-lifecycle.md`
- `docs/internal/00-index.md`
- New or updated internal physics/collision doc, likely `docs/internal/11-physics-and-collision.md`
- This sprint's `validation/phase-04-validation-report.md`
- This sprint's `artifacts/validation-summary.json`
- Optional `reports/phase-04-final-email.md` as a main-thread email draft only if useful

## Forbidden Scope

- Do not implement product code in Phase 04 except trivial docs/evidence corrections.
- Do not claim full editor collision authoring, full dogfood migration, generated mesh bounds, or production gameplay physics unless already implemented and validated.
- Do not add desktop screenshots.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All current sprint plan files.
- Phase 01-03 validation reports and implementation notes.
- `docs/api/00-index.md`
- `docs/internal/00-index.md`
- `.internal-dev/AGENTS.md`

## Senior-Engineer Guidance

- Docs are intended truth. If implementation supports less than the plan hoped, document the smaller truth and list residuals.
- Make deferred dogfood/editor/generated-bounds limits explicit.
- Evidence status must match reality. Do not set `fully_validated` until final validator passes.
- Stale-reference sweep should catch `/tmp`, stale agent ids, pending/planned/not implemented claims that no longer match, TODO markers, desktop screenshot references, and outdated phase wording.

## Ordered Implementation Steps

1. Read phase validation reports and implementation handoffs.
2. Update API docs for physics/collision usage, scene/package metadata, packaging validation, runtime limitations, and events.
3. Update internal docs/index for architecture, boundary ownership, and future migration slices.
4. If dogfood migration was deferred, link or summarize the debt artifact without pretending it is implemented.
5. Run the full validation commands from `shared/validation-matrix.md`.
6. Run runtime smoke only if runtime/app behavior changed.
7. Run true headless draw capture with `--headless --capture_target draw` only if visible renderer/editor behavior changed.
8. Run stale-reference sweep over docs and this sprint directory.
9. Update `artifacts/validation-summary.json` conservatively with commands and evidence statuses.
10. Write `validation/phase-04-validation-report.md` and prepare for final quality validator.

## Acceptance Criteria

- Public/internal docs match implemented behavior and deferred limits.
- Full validation commands pass or blockers are recorded conservatively.
- Runtime/capture evidence is present only when required by actual behavior changes.
- `artifacts/validation-summary.json` is internally consistent and not overstated.
- Final quality validator has enough evidence to pass or produce a targeted remediation plan.

## Negative Checks

- No docs claim full physics gameplay, editor authoring, or dogfood migration unless true and validated.
- No stale `/tmp` evidence paths.
- No desktop screenshot evidence.
- No `fully_validated: true` before final quality review.

## Validation Commands

```bash
cargo fmt --check
cargo check
cargo test -p physics
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p physics
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

If runtime/app behavior changed:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-06-physics-collision-foundation/root-runtime-physics-timing.jsonl
```

If visible renderer/editor behavior changed, use true headless draw capture through `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` with `--headless --capture_target draw`.

Stale sweep:

```bash
rg -n "/tmp|desktop screenshot|screenshot|TODO|pending|planned|not implemented|agent id|fully_validated|TOOLING_CONSTRAINT" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/validation/phase-04-validation-report.md`
- Final quality report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/validation/final-quality-review.md`
- Canonical evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/artifacts/validation-summary.json`

## Stop Conditions

- Stop if docs and code disagree in a way that requires implementation repair.
- Stop if validation fails outside a trivial docs/evidence correction.
- Stop if capture becomes required but the capture skill/tool is unavailable; record `TOOLING_CONSTRAINT` and ask the main thread.

## Do Not Close Unless

- Docs and evidence match the implemented contract.
- Full validation results are recorded.
- Final quality review is ready to run.
- Main-thread-only responsibilities are clearly left to the main thread.
