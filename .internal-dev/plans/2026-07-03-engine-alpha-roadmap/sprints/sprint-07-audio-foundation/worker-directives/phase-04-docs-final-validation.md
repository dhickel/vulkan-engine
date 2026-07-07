# Phase 04 Worker Directive: Docs And Final Validation

## Objective

Document the implemented alpha audio contract, reconcile validation evidence, run final checks, and prepare closeout artifacts without overstating device-dependent support.

## User-Visible Outcome

Engine users have honest docs for packaging, validating, loading, and optionally playing audio clips, plus a final evidence package that separates core support from optional device smoke.

## Editable Targets

- `docs/api/00-index.md`
- Relevant docs under `docs/api/`
- `docs/internal/00-index.md`
- Relevant docs under `docs/internal/`
- Sprint artifacts under `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/`
- `artifacts/validation-summary.json`
- `validation/phase-04-validation-report.md`
- `validation/final-quality-review.md` is for the final validator, not the implementation worker
- `reports/phase-04-email.md`

## Forbidden Scope

- Do not implement new product behavior in this phase except tiny docs/evidence fixes.
- Do not change schemas/API contracts unless a validator sends a scoped repair back through orchestration.
- Do not claim editor placement, production mixing, spatialization, streaming, or guaranteed device support unless prior phases implemented and validated them.
- Do not use desktop screenshots.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All Sprint 07 phase validation reports.
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/validation-matrix.md`
- `docs/api/00-index.md`
- `docs/internal/00-index.md`
- Top-level `AGENTS.md`
- `.internal-dev/AGENTS.md`

## Senior-Engineer Guidance

- Docs should be boring and exact: "packaged audio can be referenced and optionally played when a device is available" is the correct level.
- Avoid device-independent tests becoming hidden promises of audible output.
- Make unsupported areas visible as alpha limits.
- Keep evidence status conservative until validators pass.
- Stale-reference sweep must include old phase wording, `/tmp`, desktop screenshot mentions, `TODO`, `pending`, `planned`, `not implemented`, stale agent IDs, and contradictory validation statuses.

## Ordered Implementation Steps

1. Review what Phases 01-03 actually implemented and validated.
2. Update public API docs to explain audio package/reference/load/play flow and device-dependent runtime caveats.
3. Update internal docs to explain audio subsystem boundary, event bridge, validation strategy, and deferred features.
4. Update docs indexes if new pages are added.
5. Update `artifacts/validation-summary.json` with command results, phase statuses, device smoke status, dogfood gate status, residual risks, and superseded artifacts.
6. Run required final commands from `shared/validation-matrix.md`.
7. Run stale-reference sweep over docs and this sprint directory.
8. Draft `reports/phase-04-email.md`.
9. Leave `validation/final-quality-review.md` for the final quality validator to write after phase validation passes.

## Acceptance Criteria

- Public/internal docs match implemented behavior and limits.
- Final command set passes or blockers are recorded conservatively.
- Validation summary is internally consistent and does not claim `fully_validated` before final review.
- Optional device smoke status is explicit.
- No visual/capture evidence is requested unless visible behavior changed.

## Negative Checks

- No docs overclaim full device support, mixer, spatialization, streaming, editor authoring, or platform matrix.
- No stale `/tmp` evidence paths as authoritative artifacts.
- No "desktop screenshot" validation for audio.
- No final status contradictions in validation summary.

## Validation Commands

```bash
cargo fmt --check
cargo check
cargo test -p audio
cargo check -p audio
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

If runtime/dogfood behavior changed:

```bash
cargo test -p dungeon_dogfood
```

Stale-reference sweep:

```bash
rg -n "/tmp|desktop screenshot|screenshot|TODO|pending|planned|not implemented|agent id|fully_validated|TOOLING_CONSTRAINT" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation
```

Triaged intentional hits are acceptable only if explained in the phase report.

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/phase-04-validation-report.md`
- Final quality review path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/validation/final-quality-review.md`
- Phase report draft path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-04-email.md`
- Canonical evidence index: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/artifacts/validation-summary.json`

## Stop Conditions

- Stop if docs reveal a contract mismatch that requires product repair.
- Stop if validation summary cannot be made consistent with actual reports.
- Stop if required final commands fail in a way that is not an existing known residual.

## Do Not Close Unless

- Docs are honest about device dependence.
- Final checks have been run or blockers are recorded.
- Validation summary is conservative and consistent.
- The suite is ready for final quality validator review.
