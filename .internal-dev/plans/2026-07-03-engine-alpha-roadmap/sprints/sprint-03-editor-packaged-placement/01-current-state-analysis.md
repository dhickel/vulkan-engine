# Current State Analysis

## Verified Repo Facts

- Sprint tracker lists Sprint 03 as proposed with target "Editor packaged-asset placement hardening" and primary gate "Packaged assets place, select, save, reload, and visually prove".
- Sprint 02 is recorded as closed on branch `sprint/alpha-02-packaging-tools`; it added `tools/engine_pack` and validation/authoring/pack commands.
- Sprint 01 remains blocked and must not be closed by this sprint.
- `docs/api/09-editor-asset-browser-and-wall-chunks.md` documents the intended editor placement, inspector, save/load, and wall chunk behavior.
- Existing editor files are `apps/editor/src/main.rs`, `apps/editor/src/app_state.rs`, `apps/editor/src/panels.rs`, and `apps/editor/src/launch.rs`.
- Existing sample project files include `apps/editor/sample_project/engine.project.toml`, `assets/editor_sample.package.toml`, OBJ model/prefab files, and `scenes/start.engine.scene.json`.
- `tools/engine_pack` is a workspace package and includes CLI validation tests.

## Current Behavior Shape

- `EditorSession` tracks project path/name, active scene path, dirty state, package asset records, selected asset, placement state, selection, hierarchy, transform edits, and status messages.
- Placement UI exposes package-backed asset filters and `Place Selected`, `Confirm`, and `Cancel` actions.
- `ConfirmPlacement` loads the durable asset ID through `renderer.assets().load_model_asset`, builds a `SceneAssetReference`, and executes `PlaceAssetCommand`.
- `PlaceAssetCommand` mounts a `SceneFragment`, applies the placement transform to the fragment root, stamps stable IDs, name, tags, and asset reference on the placed root, and returns the created runtime node.
- Save/load actions call `scene.save(path)` and `Scene::load(path, &mut assets)`, clear selection on load, and reset command history.
- Docs already describe that load clears selection and undo history to avoid stale runtime handles.

## Risks And Gaps

- The editor placement workflow may be present but under-tested as an end-to-end package/save/reload contract.
- Stable placement ID generation uses an in-memory monotonic index; workers must verify save/reload and repeated placement do not collide in realistic sessions.
- Undo/redo selection behavior may preserve selection for some commands but should be tested for placement-created nodes and restored/remapped nodes.
- `Scene::save` may assign or serialize stable IDs differently from editor placement; tests must assert durable strings, not runtime IDs.
- Headless editor capture can load a scene, but may not automate UI placement. Phase 03 should avoid inventing a broad UI automation framework merely to obtain proof.
- Canonical sample scene mutation can create review noise and cross-sprint drift. Use copied scene fixtures unless the product intentionally needs the sample updated.
- Visual proof needs actual PNG inspection and sidecar metadata; compile/check success is not visual validation.

## Architecture Fit

- The correct durable identity boundary is package/project/scene strings and paths, not runtime handles.
- The command/history boundary is already the right place for editor placement mutations.
- The renderer scene API should remain the owner of persistence and runtime scene loading; editor code should compose it rather than fork a parallel serializer.
- `engine_pack` should validate persisted scene data, not become the editor's runtime scene mutator.
- Capture validation should use engine-owned headless capture instead of desktop screenshots.

## Known Unrelated State To Preserve

- `.idea/engine.iml`
- `.reasonix/`

Do not include these in phase commits, diff summaries, validation reports, or closeout claims unless the user explicitly directs otherwise.
