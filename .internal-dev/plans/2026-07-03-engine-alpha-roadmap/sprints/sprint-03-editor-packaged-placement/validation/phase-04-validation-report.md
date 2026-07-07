# Sprint 03 Phase 04 Validation Report

## Verdict

Phase 04 docs and evidence closeout passed local validation, independent final quality review, and changelog creation.

Sprint 03 is not marked closed in this phase. It remains in final validation until final commit, push, and final report gates complete.

## Scope

Updated documentation and sprint evidence for Sprint 03 editor packaged-asset placement hardening.

Changed files:

- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/validation/phase-03-validation-report.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/validation/phase-04-validation-report.md`

## Documentation Changes

The editor asset browser API page now documents:

- the Sprint 03 saved-scene artifact path used for save/reload validation;
- `engine_pack validate-project` and `engine_pack validate-scene` commands for the sample project, canonical startup scene, and saved-scene copy;
- runtime-handle and path-only identity constraints;
- the true editor headless capture command using `--headless --capture_target draw`;
- accepted draw-target sidecar expectations: `capture_target = "draw"`, `format = "R16G16B16A16_SFLOAT"`, `status = "succeeded"`, and `extent = 1440 x 900`;
- current limitations: no binary package archives, thumbnails, CSG/brush editing, material graph/PBR authoring, packaged audio placement, physics/collision authoring, scripting, or runtime project launcher.

The sprint tracker marks Sprint 03 as `validating`, not `closed`, and leaves Sprint 01 untouched.

## Evidence Reconciliation

`artifacts/validation-summary.json` was reconciled so that:

- top-level status is `phase_04_passed_final_quality_review_changelogged_pending_final_commit_push_email_evidence`;
- Phase 01, Phase 02, and Phase 03 are recorded as passed, committed, pushed, and reported;
- Phase 04 is recorded as passed with final quality review and changelog complete, pending final commit, push, and report evidence;
- accepted capture artifacts are recorded as independently validated, committed, pushed, and reported;
- accepted capture proof points at `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw`;
- the earlier present-target capture directory is explicitly superseded;
- residual risks state that final commit, push, and report evidence remain active.

One stale Phase 03 report phrase was corrected from the temporary `phase_03_passed_pending_commit` wording to commit/push/reporting-reconciled wording.

## Stale-Reference Sweep

Command:

```text
rg -n "present-target|present-seq|sprint-03-editor-packaged-placement/engine-editor|pending_commit|phase_03_passed_pending|phase_03_worker|required_pending|/tmp|TODO|fully_validated|binary archive|thumbnail|CSG|brush|runtime launcher|material graph" docs/api/09-editor-asset-browser-and-wall-chunks.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement
```

Result: no blocking stale references after the Phase 03 report correction.

Remaining matches are intentional:

- docs and plan directives explicitly list deferred features as out of scope;
- Phase 03 report and summary intentionally state that earlier present-target captures are superseded;
- validation instructions mention stale-reference terms as things to search for;
- historical phase reports preserve the status wording that was true at the time of that phase.

## Capture Evidence

Accepted capture directory:

```text
.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw
```

Accepted files remain present:

```text
engine-editor-frame-5-draw-seq-0000.png
engine-editor-frame-5-draw-seq-0000.json
engine-editor-frame-10-draw-seq-0001.png
engine-editor-frame-10-draw-seq-0001.json
engine-editor-frame-15-draw-seq-0002.png
engine-editor-frame-15-draw-seq-0002.json
```

Sidecar predicate check:

```text
for f in .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/*.json; do jq -e '.status == "succeeded" and .capture_target == "draw" and .format == "R16G16B16A16_SFLOAT" and .extent.width == 1440 and .extent.height == 900 and .source == "Sequence"' "$f" >/dev/null || exit 1; done
```

Result: passed.

## Validation Commands

```text
cargo fmt --check
Result: passed

git diff --check
Result: passed

cargo check
Result: passed

cargo check -p editor
Result: passed with existing renderer/editor dead-code warnings

cargo check -p renderer
Result: passed with existing renderer dead-code warnings

cargo check -p renderer --examples
Result: passed with existing renderer dead-code warnings

cargo check -p input
Result: passed

cargo check -p engine_pack --locked
Result: passed with existing renderer dead-code warnings

cargo test -p editor
Result: passed, 17 tests

cargo test -p renderer scene
Result: passed, 38 renderer lib tests and 2 integration tests matching filter

cargo test -p renderer asset_registry
Result: passed, 8 renderer lib tests matching filter

cargo test -p engine_pack --locked
Result: passed, 13 CLI validation tests

cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
Result: passed, valid[project]

cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
Result: passed, valid[scene]

cargo run -p engine_pack -- validate-scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --project apps/editor/sample_project/engine.project.toml
Result: passed, valid[scene]

jq empty .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/validation-summary.json
Result: passed
```

## Residual Risk

- Existing renderer dead-code warnings remain and were not part of Sprint 03.
- Sprint 03 still requires final commit, push, and final email evidence before it can be closed.
- `.idea/engine.iml` and `.reasonix/` remain unrelated dirty/untracked local state and were not touched.
