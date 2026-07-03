# Phase 02 Worker Directive: Save Reload Validation

## Objective

Harden save/reload behavior for editor-authored package-backed scene nodes and prove the saved scene validates through `engine_pack` without mutating the canonical sample scene by accident.

## User-Visible Outcome

An alpha author can place packaged assets, save the authored scene to a deliberate path, reload it, and validate it with `engine_pack` while preserving durable asset references and stable node IDs.

## Editable Targets

- `apps/editor/src/main.rs`
- `apps/editor/src/app_state.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/data/asset_registry.rs`
- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/cli_validation.rs`
- Test fixtures under `tools/engine_pack/fixtures/` if useful
- Sprint-local/temp scene evidence under `.internal-dev/headless_capture_tests/` or this sprint's `artifacts/`

## Forbidden Scope

- Do not change the canonical sample scene unless approved by the main thread.
- Do not introduce runtime handles as durable identity.
- Do not implement binary archive packaging.
- Do not perform visual capture work; Phase 03 owns that.
- Do not close Sprint 01 or touch unrelated `.idea/engine.iml` / `.reasonix/`.

## Supporting Docs To Read

- Phase 01 validation report and changes.
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `shared/implementation-notes.md`
- `shared/validation-matrix.md`
- Sprint 02 validation summary for `engine_pack` command expectations.

## Senior Engineer Guidance

- Save/reload proof should exercise the same durable asset IDs the editor places, not handcrafted JSON that bypasses the placement path.
- Use temp copies or generated scenes under `.internal-dev` for mutation-heavy tests.
- `engine_pack validate-scene` is the persistence gate. It should catch unknown asset IDs and runtime handle identity regressions.
- Loading must reset selection/history to prevent stale runtime node IDs from being reused.

## Ordered Implementation Steps

1. Review Phase 01 output and confirm placement identity is stable enough for persistence tests.
2. Build a deterministic test/helper flow that creates or copies a scene, places `editor_sample.model.block` and `editor_sample.wall.stone_2m`, saves it, and reloads it.
3. Assert saved JSON includes durable asset IDs and stable node IDs, and excludes runtime slot/generation handle shapes.
4. Assert reload restores placed nodes with asset references, tags, transforms, and material override metadata where applicable.
5. Ensure load clears selection and command history or add tests confirming current behavior.
6. Run `engine_pack validate-project` and `validate-scene` on the sample project and saved scene copy.
7. Add negative tests or fixtures if a regression risk is exposed, such as unknown asset IDs or runtime-handle-shaped node IDs.
8. Update phase evidence and conservative validation summary fields.

## Acceptance Criteria

- Save/reload round trip preserves durable package asset references for at least one model and one wall chunk.
- Saved scene copy validates with `engine_pack validate-scene --project apps/editor/sample_project/engine.project.toml`.
- Project validates with `engine_pack validate-project`.
- Loading a scene clears selection and command history, or an equivalent stale-runtime-ID defense is proven.
- Canonical sample scene is unchanged unless intentionally approved and documented.

## Negative Checks

- No runtime handles in saved scene JSON.
- No path-only identity: `SceneAssetReference.id` must be present for package-backed nodes.
- No accidental updates to sample project fixtures.
- No capture/visual proof claims yet.

## Validation Commands

```bash
cargo fmt --check
cargo check -p editor
cargo check -p renderer
cargo check -p engine_pack --locked
cargo test -p editor
cargo test -p renderer scene
cargo test -p engine_pack --locked
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene <saved-scene-copy> --project apps/editor/sample_project/engine.project.toml
git diff --check
```

Record the actual `<saved-scene-copy>` path in the phase report.

## Evidence Expectations

- Save copied scene path and validation command.
- Summarize JSON identity assertions.
- Record whether canonical sample scene changed.
- Write or prepare `validation/phase-02-validation-report.md`.
- Main thread records commit/push/email report evidence after validation passes.

## Stop Conditions

- Stop if reload can only work by serializing runtime handles.
- Stop if package registry loading cannot be made available before `Scene::load` without broader architecture work.
- Stop if tests would overwrite canonical sample scene without explicit approval.

## Do Not Close Unless

- Saved scene copy round trip and `engine_pack` validation are proven.
- Runtime handle negative checks are covered.
- Phase evidence names exact scene copy and commands.
