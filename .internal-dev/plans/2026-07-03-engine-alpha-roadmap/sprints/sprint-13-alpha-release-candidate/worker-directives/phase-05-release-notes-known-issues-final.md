# Phase 05 Worker Directive: Release Notes, Known Issues, Final Evidence

## Objective

Draft the alpha release notes, known issues, supported platform/driver expectations, contributor/agent workflow notes, and prepare the suite for final quality review.

## User-Visible Outcome

The release-candidate branch has a clear release narrative and a conservative pass/no-release evidence package.

## Editable Targets

- Release docs selected in Phase 01.
- `README.md` and API docs only for final link/wording corrections.
- `reports/phase-05-release-notes-known-issues-final.md`
- `artifacts/validation-summary.json`
- `validation/final-quality-review.md` is written by the final validator, not the worker.

## Forbidden Scope

- Do not implement product features.
- Do not hide or delete residuals to make the release look cleaner.
- Do not edit `SPRINT-TRACKER.md`.
- Do not create changelogs/knowledge/notes/bugs unless the main thread confirms timing.

## Supporting Docs To Read

- All phase reports and validation reports.
- `artifacts/validation-summary.json`
- Release docs from Phase 01.
- Current `README.md`, quickstart, runtime launcher, packaging CLI, editor, and dogfood docs.

## Senior-Engineer Guidance

- Release notes should be boring and precise: what works, what does not, how it was validated.
- Known issues should include inherited residuals if they affect users, even if they are not Sprint 13 regressions.
- The final evidence summary must not claim `fully_validated` until validators and capture proof agree.
- Keep final docs aligned with exact commands used in validation.

## Ordered Steps

1. Read every phase report and validation report.
2. Draft/update release notes with:
   - release name/date/branch/commit placeholder;
   - supported workflows;
   - validation summary;
   - known issues;
   - no-release criteria;
   - platform/driver/toolchain expectations.
3. Draft/update contributor/agent workflow notes:
   - clean validation;
   - true headless draw capture;
   - residual logging;
   - protected paths and tracker ownership.
4. Run final docs stale/overclaim sweep.
5. Run final command baseline if Phase 03/04 changed product code or docs materially:
   ```sh
   cargo fmt --check
   cargo check
   cargo check -p renderer --examples
   cargo check -p editor
   cargo check -p dungeon_dogfood
   cargo check -p engine_pack --locked
   cargo test -p input
   cargo test -p engine
   cargo test -p engine_pack --locked
   ```
6. Validate capture sidecars for sample runtime, editor, and dogfood.
7. Update `artifacts/validation-summary.json` to `final_quality_pending` at most.
8. Write `reports/phase-05-release-notes-known-issues-final.md`.

## Acceptance Criteria

- Release notes and known issues reflect actual evidence.
- Supported platform/driver/toolchain expectations are visible.
- Contributor/agent workflow notes include fresh validation and capture rules.
- Evidence index is complete and conservative.
- Final quality review can run without reconstructing missing phase state.

## Negative Checks

```sh
rg -n "/tmp|desktop screenshot|present-target proof|pending|planned|not implemented|TODO|release TBD|unknown status" README.md docs apps/dungeon_dogfood .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate
```

Expected matches must be explained in the phase report.

## Validation Commands

```sh
cargo fmt --check
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json >/dev/null
```

Capture sidecar predicate:

```sh
for dir in \
  .internal-dev/captures/sprint-13-alpha-release-candidate/sample-runtime-draw \
  .internal-dev/captures/sprint-13-alpha-release-candidate/editor-sample-draw \
  .internal-dev/captures/sprint-13-alpha-release-candidate/dogfood-draw
do
  for f in "$dir"/*.json; do
    jq -e '.status == "succeeded" and .capture_target == "draw"' "$f"
  done
done
```

## Stop Conditions

- Stop if any required phase validation report is missing or failed.
- Stop if capture evidence is missing/inconclusive.
- Stop if known issues omit release-blocking failures.
- Stop if evidence index and reports disagree.

## Evidence Expectations

- Worker report: `reports/phase-05-release-notes-known-issues-final.md`
- Validator report: `validation/phase-05-validation-report.md`
- Final validator report: `validation/final-quality-review.md`

## Do Not Close Unless

- The release candidate can honestly be marked pass, pass-with-residuals, or blocked.
- User decision gates are listed for main-thread closeout.
- Changelog/tracker/publish actions are left to the main thread.

