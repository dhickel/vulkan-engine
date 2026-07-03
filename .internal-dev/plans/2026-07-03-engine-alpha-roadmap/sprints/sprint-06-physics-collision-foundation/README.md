# Sprint 06: Physics And Collision Foundation

Status: planning_ready

## Objective

Turn the current thin Rapier wrapper and dogfood-local collision into an alpha physics/collision foundation with clear engine-level contracts.

## User-Visible Outcome

After this sprint, authored collision metadata can exist in package and scene contracts, basic physics queries and contact/trigger events are available and testable without Vulkan, and dungeon dogfood either has a small proof path against the new contracts or an explicit migration debt record.

## Work Classification

Large roadmap sprint. The work spans the physics crate API, durable physics/collision descriptors, renderer package/scene persistence, packaging CLI validation, event-contract integration, docs, and dogfood migration decision gates.

## In Scope

- Expand `src/physics` into a renderer-independent alpha physics API around durable wrapper IDs, descriptors, colliders, queries, and contact/trigger event extraction.
- Reuse Sprint 05 `engine_events` physics event contracts without making physics depend on renderer or Vulkan.
- Add typed collision metadata to package and scene persistence contracts with validation that rejects runtime handles and invalid shape/body data.
- Extend `engine_pack` validation coverage for collision metadata through existing package/scene validation paths.
- Add focused non-render tests and a small sample/proof path for event/query behavior.
- Evaluate dungeon dogfood collision migration through an explicit decision gate: narrow adapter/proof only if low risk, otherwise record migration debt under this sprint.
- Update public/internal docs to describe supported alpha contracts and deferred limits.
- Produce phase validation reports and a conservative `artifacts/validation-summary.json`.

## Out Of Scope

- Full gameplay physics migration for dungeon dogfood.
- Editor UI authoring for collision components.
- CPU mesh-bound generation or renderer scene graph ownership of CPU mesh bounds unless a minimal placeholder contract needs naming.
- Serializing Rapier runtime handles.
- Vulkan, renderer, or window dependencies in the physics crate.
- Broad cleanup of existing warning noise unless this sprint introduces a warning.
- Visual proof unless a phase changes visible renderer/editor behavior; if needed, use true engine headless draw capture only with `--headless --capture_target draw`.

## Target Surfaces

- Physics: `src/physics/Cargo.toml`, `src/physics/src/lib.rs`, optional focused modules/tests under `src/physics/src/`.
- Events: `src/events/src/lib.rs` only if contract gaps are found; prefer bridge code outside `engine_events`.
- Renderer data/scene contracts: `src/renderer/src/data/asset_registry.rs`, `src/renderer/src/api/scene.rs`, validation tests in the same modules.
- Packaging CLI: `tools/engine_pack/src/main.rs`, `tools/engine_pack/tests/cli_validation.rs`.
- Dogfood: `apps/dungeon_dogfood/src/collision.rs`, `player.rs`, `main.rs`, `layout.rs`, `scene_seed.rs` only for a narrow proof or debt-linked adapter.
- Docs: `docs/api/00-index.md`, `docs/api/03-scene-graph-and-fragment-workflows.md`, `docs/api/04-assets-sync-deferred-and-handles.md`, `docs/api/10-packaging-cli.md`, `docs/api/11-runtime-project-launcher.md`, `docs/api/12-events-and-lifecycle.md`, `docs/internal/00-index.md`, new/update internal physics/collision docs.
- Artifacts: this sprint directory, `.internal-dev/debug_reports/sprint-06-physics-collision-foundation/`, optional `.internal-dev/captures/sprint-06-physics-collision-foundation/` only if visible behavior changes.

## Acceptance Criteria

- `src/physics` exposes a small durable alpha API for body/collider IDs, body kinds, collider shapes, trigger flags, transforms, stepping, queries, and contact/trigger extraction.
- Physics crate tests prove core behavior without renderer/Vulkan.
- Collision metadata is representable in package and scene contracts using durable IDs and typed descriptors.
- Package/scene validators reject runtime handles, invalid dimensions, invalid body/shape kinds, duplicate collision IDs where applicable, and unknown collision asset references where applicable.
- `engine_pack` validation sees the same metadata failures as renderer validation.
- Physics contact/query/trigger outcomes can bridge to `EngineEvent::Physics` with Sprint 05 IDs and contact phases.
- Dogfood has either a minimal low-risk proof path or a written migration debt artifact explaining why full migration is deferred.
- Docs state supported alpha behavior and deferred editor/gameplay limits without claiming full physics gameplay.
- Final validation evidence is reconciled in `artifacts/validation-summary.json` and final quality review.

## Negative Criteria

- Do not serialize or expose Rapier runtime handles as durable file IDs.
- Do not make `physics` depend on `renderer`, `ash`, `winit`, `imgui`, editor, or dogfood crates.
- Do not require Vulkan startup for physics or metadata tests.
- Do not replace dogfood's ramp/floor stepping unless a narrow adapter can preserve tests and behavior.
- Do not claim CPU mesh collision generation is complete if only a placeholder descriptor exists.
- Do not mark `fully_validated` until every phase validator and final quality validator pass.
- Do not touch unrelated `.idea/engine.iml` or `.reasonix/`.

## Validation Plan

- Core: `cargo fmt --check`, `cargo test -p physics`, `cargo check -p physics`, `cargo check`.
- Metadata/CLI: `cargo test -p renderer`, `cargo test -p engine_pack`, `cargo check -p engine_pack`.
- Events/apps: `cargo test -p engine_events`, `cargo test -p physics`, `cargo check -p dungeon_dogfood`, `cargo check -p editor`.
- Full closeout: commands listed in `shared/validation-matrix.md`.
- Runtime smoke only if runtime/app integration changes require it; use timeout-bound engine commands and write output under `.internal-dev/debug_reports/sprint-06-physics-collision-foundation/`.
- Capture proof only if visible renderer/editor behavior changes; use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and true engine-owned headless draw capture with `--headless --capture_target draw`, never desktop screenshots.

## Advanced-Planner Handoff

Execute phases in order from `work-units/README.md`. Validate each phase before dependent work proceeds. Main thread owns commits, pushes, sprint tracker closeout, changelog timing, and email/report responsibilities after validation.
