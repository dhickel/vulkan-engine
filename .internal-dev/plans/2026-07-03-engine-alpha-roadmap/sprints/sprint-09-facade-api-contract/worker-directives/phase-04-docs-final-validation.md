# Phase 04 Worker Directive: Docs Final Validation

## Objective

Reconcile Sprint 09 docs, reports, stale references, and evidence so the final validator can determine whether the facade API alpha contract is truly ready.

## User-Visible Outcome

The sprint closes with a coherent API contract, compile evidence, phase reports suitable for HTML email summaries, and a conservative validation summary that does not hide residual risks.

## Editable Targets

- `docs/api/00-index.md`
- Other touched `docs/api/*.md` files from prior phases.
- Plan reports under `reports/`
- `shared/validation-matrix.md` only if criteria must be corrected.
- `validation/README.md` only if validator routing must be corrected.
- `artifacts/validation-summary.json`
- Optional changelog draft under this plan directory only if the main thread asks; do not create `.internal-dev/changelogs/` entry without confirmation.

## Forbidden Scope

- Do not add new product features.
- Do not make API export changes unless repairing a clear docs/evidence defect from previous phases.
- Do not broaden docs into full API reference rewrite.
- Do not archive or move plan files.
- Do not send email or push branches; main thread owns that.

## Supporting Docs To Read

- All prior phase reports and validator reports.
- `00-specification-lock.md`
- `shared/validation-matrix.md`
- `validation/README.md`
- `artifacts/validation-summary.json`
- All docs changed in phases 01-03.

## Senior Engineer Guidance

- This is a reconciliation phase. Bias toward correcting claims, links, and evidence consistency rather than inventing new behavior.
- Stale wording is a real defect for this sprint because the sprint's product is the public contract.
- The final evidence status must match the weakest required gate.
- Keep residuals visible. Do not turn accepted residuals into "done" language.
- If docs still say every public export is stable beginner surface, validation must fail.

## Implementation Steps

1. Read all phase reports and validation reports.
2. Run stale-reference scans over docs and this plan directory.
3. Fix stale paths, outdated sprint references, contradictory status words, and unsupported API promises.
4. Ensure docs consistently describe:
   - beginner-supported facade;
   - compatibility exports;
   - advanced interop feature gate;
   - deferred Sprint 10 advanced rendering opt-in.
5. Ensure `reports/README.md` has all expected report files or records missing reports as defects.
6. Update `artifacts/validation-summary.json` with actual command/report/capture/email/branch status, preserving conservative top-level status.
7. Write `reports/phase-04-final-docs-validation.md`.

## Acceptance Criteria

- Stale scans are run and findings are repaired or recorded.
- Docs have one coherent alpha-supported facade story.
- Evidence index references real reports and does not claim unearned final status.
- Any residual risks are explicit.
- Final validator has enough evidence to review without replanning.

## Negative Checks

- No product feature work.
- No unsupported final success wording.
- No desktop screenshot evidence.
- No unapproved `.internal-dev/changelogs/`, `.internal-dev/notes/`, or `.internal-dev/bugs/` entries.

## Validation Commands

```sh
cargo fmt --check
cargo check
cargo test -p renderer
cargo check -p renderer --examples
rg -n "TODO|pending|planned|not implemented|/tmp|sprint-08|Sprint 08|sprint-04|headless-draw" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract
rg -n "stable public surface|Everything below api|advanced-interop|prelude|SceneWorld|CommandHistory|AnimationPlayer" docs/api src/renderer/src src/renderer/examples src/renderer/tests
```

Run conditionally:

```sh
cargo test -p engine_pack
cargo test -p input
cargo doc -p renderer --no-deps
```

## Stop Conditions

- Stop if earlier reports are missing or contradictory enough that a validator cannot judge the sprint.
- Stop if final docs require new product/API behavior to become truthful.
- Stop if evidence index cannot be made consistent with actual validation state.

## Evidence Expectations

- Worker report: `reports/phase-04-final-docs-validation.md`
- Validator report path: `validation/phase-04-validation-report.md`
- Updated `artifacts/validation-summary.json`

## Do Not Close Unless

- All stale scans are recorded.
- Validation summary is internally consistent.
- Final quality review prerequisites are ready.
