# Phase 02 Worker Directive: Packaged Content And Sample Scene

## Objective

Create or normalize dogfood package/project/scene content so durable engine contracts, not dogfood-only manifests, describe the vertical slice content wherever practical.

## User-Visible Outcome

Users and validators can run `engine_pack` validation against dogfood project/package/scene files and see the same core content that the dogfood app uses.

## Editable Files

Likely editable:

- `apps/dungeon_dogfood/engine.project.toml`
- `apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml`
- `apps/dungeon_dogfood/scenes/start.engine.scene.json`
- `apps/dungeon_dogfood/assets/content_pack.toml` only if the Phase 01 migration decision allows a transitional bridge.
- `apps/dungeon_dogfood/assets/content_manifest.md`
- `tools/engine_pack/tests/cli_validation.rs`
- `tools/engine_pack/fixtures/**` only for focused dogfood/package validation fixtures if needed.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/reports/migration-debt.md`
- `validation/phase-02-validation-report.md`
- `artifacts/validation-summary.json`

Read before editing:

- Phase 01 reports.
- `apps/editor/sample_project/engine.project.toml`.
- `apps/editor/sample_project/assets/editor_sample.package.toml`.
- `apps/editor/sample_project/scenes/start.engine.scene.json`.
- `src/renderer/src/data/asset_registry.rs`.
- `src/renderer/src/api/scene.rs`.

Forbidden:

- Do not edit active Sprint 09 files.
- Do not use absolute local paths in package/project/scene files.
- Do not serialize runtime handles.
- Do not update `SPRINT-TRACKER.md`.

## Ordered Steps

1. Read Phase 01 audit and confirm the migration decision.
2. Add dogfood `engine.project.toml` with stable project ID, startup scene, asset root, enabled package, and sane window settings.
3. Add dogfood package manifest with durable asset IDs for models, materials/textures where supported, environment, and audio clip.
4. Add a startup scene that references durable asset IDs for visible baseline content and includes lights/environment where schema supports it.
5. If `content_pack.toml` remains, reduce it to gameplay-only/transitional data where possible and record the debt.
6. Add or adjust tests/fixtures so package/project/scene validation covers dogfood data or a representative subset.
7. Run validation commands and fix only in-scope data/schema issues.
8. Update migration-debt report and validation summary.
9. Write phase validation report.

## Senior-Engineer Guidance

- Durable IDs are the contract. File paths are diagnostics/load locations.
- Prefer a small, clear startup scene over trying to serialize the whole procedural dungeon.
- If material/environment metadata cannot be expressed cleanly, use existing metadata fields only if validators already accept them; otherwise record debt.
- The dogfood app can still generate gameplay geometry. This phase proves packaged content and scene baseline, not complete gameplay serialization.

## Acceptance Criteria

- `engine_pack validate-package` passes for dogfood package.
- `engine_pack validate-project` passes for dogfood project.
- `engine_pack validate-scene` passes for dogfood startup scene.
- Dogfood content manifest names canonical package/project/scene files.
- Any remaining `content_pack.toml` responsibility is documented as transitional.
- Tests cover at least one negative or regression-prone validation case if schema/tooling changed.

## Negative Checks

- No runtime handles in TOML/JSON.
- No absolute paths or parent traversal.
- No duplicate asset IDs.
- No silent removal of existing dogfood assets needed by later runtime phases.

## Validation Commands

```sh
cargo check -p engine_pack
cargo test -p engine_pack
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

If paths differ, update this directive's downstream references through reports and `artifacts/validation-summary.json`.

## Stop Conditions

- Stop if schema changes are required beyond a focused validation/tooling addition.
- Stop if Phase 01 identifies unresolved user decision gates.
- Stop if the current package/project validators reject content that should be valid but fixing it would touch active Sprint 09 files.

## Evidence Expectations

- Include exact command output summary.
- Include list of created/changed dogfood data files.
- Include migration debt summary.
- Validation report path: `validation/phase-02-validation-report.md`.

## Do Not Close Unless

- Data files validate or a blocker is recorded.
- Migration debt is explicit.
- Validation summary does not claim runtime/capture success yet.
