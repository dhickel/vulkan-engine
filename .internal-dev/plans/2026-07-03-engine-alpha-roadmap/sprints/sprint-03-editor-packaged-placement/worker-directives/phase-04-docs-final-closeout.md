# Phase 04 Worker Directive: Docs Final Closeout

## Objective

Update docs and sprint evidence to reflect the implemented packaged placement workflow, reconcile all validation/capture evidence, prepare changelog closeout, and leave the sprint ready for final quality review.

## User-Visible Outcome

The API guide accurately documents editor packaged placement, save/reload, validation, and capture proof. Sprint evidence is internally consistent and ready for the main thread's final commit/push/email closeout.

## Editable Targets

- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `docs/api/00-index.md` only if an index update is required
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- `artifacts/validation-summary.json`
- `validation/final-quality-review.md` only if the final validator writes it; worker may prepare evidence, not self-pass
- `.internal-dev/changelogs/2026-07-03-sprint-03-editor-packaged-placement.md` when main thread confirms closeout timing

## Forbidden Scope

- Do not reopen product implementation unless validation found a scoped docs/evidence defect.
- Do not close Sprint 01.
- Do not claim `fully_validated` without all phase validators, capture reconciliation, final quality review, and main-thread commit/push/email evidence.
- Do not include `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- All phase validation reports.
- `artifacts/validation-summary.json`.
- Capture artifacts and Phase 03 report.
- `.internal-dev/AGENTS.md`.
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`.

## Senior Engineer Guidance

- Documentation should teach the alpha workflow without overselling maturity.
- Keep deferred features explicit: binary archives, thumbnails, CSG/brush editing, runtime launcher, material graph/PBR editing.
- Evidence summary must be conservative. Contradictory statuses are validation failures.
- Changelog is for finalized work only and may require main-thread timing confirmation.

## Ordered Implementation Steps

1. Read phase reports and capture evidence.
2. Update API docs to match implemented behavior, validation commands, and known limitations.
3. Update sprint tracker from proposed/planned toward the correct orchestration status only when implementation has actually reached that gate.
4. Update `artifacts/validation-summary.json` with phase command results, capture artifacts, validation statuses, residual risks, and main-thread evidence placeholders or recorded links.
5. Run stale-reference sweep over docs and `.internal-dev` for old artifact paths, `/tmp` evidence, stale agent IDs, false pending/planned wording, TODOs, and outdated phase wording.
6. Prepare changelog entry under `.internal-dev/changelogs/` only when main thread confirms it is time.
7. Run final validation commands.
8. Hand off to final quality validator.

## Acceptance Criteria

- Docs match implemented Sprint 03 behavior and do not describe planned-only behavior as complete.
- Validation summary is internally consistent.
- Capture evidence is linked and reconciled.
- Sprint tracker status is accurate and does not close Sprint 01.
- Changelog exists if closeout timing was confirmed.
- Final quality review is ready to compare plan, code, docs, tests, phase reports, capture evidence, and closeout evidence.

## Negative Checks

- No stale `/tmp` evidence as canonical proof.
- No unsupported binary/archive packaging claims.
- No visual proof claim without Phase 03 artifacts.
- No broad docs rewrite unrelated to Sprint 03.

## Validation Commands

```bash
cargo fmt --check
git diff --check
cargo check
cargo check -p editor
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo check -p engine_pack --locked
cargo test -p editor
cargo test -p renderer scene
cargo test -p renderer asset_registry
cargo test -p engine_pack --locked
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene <saved-scene-copy> --project apps/editor/sample_project/engine.project.toml
```

Also verify Phase 03 capture artifacts remain present and inspectable.

## Evidence Expectations

- Write or prepare `validation/phase-04-validation-report.md`.
- Update `artifacts/validation-summary.json`.
- Record docs files changed and stale-reference sweep findings.
- Main thread records final commit/push/email/changelog evidence.

## Stop Conditions

- Stop if phase validation reports are missing or failed.
- Stop if capture evidence is missing, blank, or unreconciled.
- Stop if validation summary would need to claim final status while residual blockers remain.
- Stop if changelog timing is not confirmed; leave changelog as pending rather than inventing closeout.

## Do Not Close Unless

- Final quality review can run against complete, consistent evidence.
- Docs and evidence are aligned with code.
- Sprint 03 status is conservative and Sprint 01 remains untouched.
