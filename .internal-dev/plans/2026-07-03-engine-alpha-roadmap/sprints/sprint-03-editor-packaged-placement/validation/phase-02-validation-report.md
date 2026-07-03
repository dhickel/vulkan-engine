# Sprint 03 Phase 02 Validation Report

## Findings

No blocking findings for Phase 02.

Non-blocking worktree note: `.idea/engine.iml` is modified and `.reasonix/` is untracked in the current checkout. They are listed in the sprint summary as unrelated dirty state to preserve, were not required for Phase 02, and were not touched by this validator.

## Verdict

Passed independent validation. Phase 02 is ready for commit/push/reporting and is recorded as `passed_pending_commit` in the sprint validation summary.

## Scope

Validated Phase 02 save/reload persistence hardening on branch `sprint/alpha-03-editor-packaged-placement`.

Reviewed governance and task inputs:

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `worker-directives/phase-02-save-reload-validation.md`
- prior `validation/phase-02-validation-report.md`
- changed files: `apps/editor/src/main.rs`, `src/renderer/src/api/scene.rs`
- canonical summary: `artifacts/validation-summary.json`
- generated scene evidence: `artifacts/phase-02-saved-scene-copy.engine.scene.json`

## Criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Save/reload preserves durable package asset references for one model and one wall chunk | Pass | `src/renderer/src/api/scene.rs:2232` test places `editor_sample.model.block` and `editor_sample.wall.stone_2m` through `PlaceAssetCommand`, saves, reloads, and asserts asset IDs/path hints in loaded summaries. |
| Saved scene copy validates with `engine_pack validate-scene --project apps/editor/sample_project/engine.project.toml` | Pass | Independent CLI run returned `valid[scene]` for `.internal-dev/.../phase-02-saved-scene-copy.engine.scene.json`. |
| Project validates with `engine_pack validate-project` | Pass | Independent CLI run returned `valid[project]: apps/editor/sample_project/engine.project.toml (project.editor_sample)`. |
| Loading clears selection and command history or proves equivalent stale-runtime-ID defense | Pass | `apps/editor/src/main.rs:585` calls `reset_editor_runtime_after_scene_load`; `apps/editor/src/main.rs:973` proves undo/redo, selection, and transform edit state are cleared. |
| Canonical sample scene unchanged unless approved | Pass | `git diff -- apps/editor/sample_project` produced no diff; canonical sample scene CLI validation passed. |
| No runtime handles in saved scene JSON | Pass | Test asserts no `"slot"`, `"generation"`, or `mesh_handle`; independent `rg` over the generated scene found no runtime/handle strings. Renderer scene validation also has structured runtime-handle diagnostics. |
| No path-only identity for package-backed nodes | Pass | Generated artifact stores `asset.id` for both package-backed nodes alongside `path_hint`; test and CLI validation reject missing/unknown IDs. |
| No accidental product fixture/evidence placement | Pass | Generated scene lives under sprint `artifacts/`, not under `apps/editor/sample_project`; test writes a deterministic `.internal-dev` evidence path. |
| No visual capture claims | Pass | Phase summary and report keep visual proof pending for Phase 03; no capture artifacts are claimed for Phase 02. |

## Evidence Details

The save/reload proof exercises the placement path rather than handcrafted JSON. `place_test_asset` builds a `SceneFragment` and executes `PlaceAssetCommand::new(...)` through `Scene::execute_command`, then `scene.save(...)` writes the evidence artifact. The reload proof uses `Scene::load_with_loader`, which shares the normal file read and JSON deserialization path with `Scene::load` before injecting the fake test loader.

The saved scene artifact contains:

- model node `node.placed.editor_sample_model_block.000001` with `asset.id = editor_sample.model.block`, `path_hint = models/block_prop.obj`, phase tags, transform, and material override `mat_override.phase02_block`
- wall node `node.placed.editor_sample_wall_stone_2m.000002` with `asset.id = editor_sample.wall.stone_2m`, `path_hint = prefabs/wall_straight_2m.obj`, wall tags, and transform

Test side effects are acceptable for this phase: the renderer test creates/overwrites one deterministic evidence file under the sprint-local `.internal-dev/.../artifacts/` directory, which the directive allowed. It does not mutate the canonical sample scene.

## Commands Run

```text
cargo test -p editor scene_load_reset_clears_selection_and_command_history
Result: passed, 1 test; existing renderer warnings emitted

cargo test -p renderer editor_packaged_scene_save_copy_round_trips_model_and_wall_chunk
Result: passed, 1 renderer lib test; generated/updated the sprint saved scene artifact; existing renderer warnings emitted

cargo run -q -p engine_pack -- validate-scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --project apps/editor/sample_project/engine.project.toml
Result: passed, valid[scene]; existing renderer warnings emitted

cargo run -q -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
Result: passed, valid[project]; existing renderer warnings emitted

cargo run -q -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
Result: passed, valid[scene]; existing renderer warnings emitted

rg -n '"slot"|"generation"|mesh_handle|runtime|handle' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json
Result: no matches

git diff -- apps/editor/sample_project/scenes/start.engine.scene.json apps/editor/sample_project/engine.project.toml apps/editor/sample_project
Result: no diff

git diff --check
Result: passed
```

## Residual Risk

- Existing renderer dead-code warnings remain and were not part of this phase.
- Phase 03 still owns visible editor runtime/capture proof.
- Current dirty state includes `.idea/engine.iml` and `.reasonix/`; this validator preserved them as unrelated/out-of-scope state.
