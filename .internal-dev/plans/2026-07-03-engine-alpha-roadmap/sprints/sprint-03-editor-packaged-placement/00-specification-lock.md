# Specification Lock

## Locked Objective

Sprint 03 hardens editor packaged-asset placement from package registry selection through durable scene persistence and visual proof. The sprint is successful only when package-backed model/wall assets can be placed, selected, saved, reloaded, validated with `engine_pack`, and shown in deterministic headless capture evidence.

## Acceptance Criteria

- The editor default sample project loads package records from `apps/editor/sample_project/engine.project.toml`.
- Placeable assets include `AssetKind::Model`, `AssetKind::Prefab`, and `AssetKind::WallChunk`.
- Confirming placement uses `PlaceAssetCommand`; it does not directly mutate the scene outside the command/history contract.
- The placed root node receives a durable stable node ID, display name, tags, and `SceneAssetReference` with durable asset ID.
- The editor selects the created node after placement and preserves or remaps selection after undo/redo when possible.
- Placement, undo, redo, save, load, and validation failures push visible status messages without silently corrupting state.
- Save/reload tests prove durable scene JSON contains asset IDs and stable node IDs, not runtime handles.
- `engine_pack validate-project apps/editor/sample_project/engine.project.toml` passes.
- `engine_pack validate-scene <saved-scene-copy> --project apps/editor/sample_project/engine.project.toml` passes for a saved scene containing at least one packaged model and one wall chunk.
- Headless capture evidence shows package-backed placed assets through the same package/project/scene data path or a clearly justified capture harness that exercises equivalent package-backed scene loading.
- Docs describe only implemented behavior and clearly keep binary archives, thumbnails, and advanced material authoring out of scope.

## Validation Criteria

- Compile/check commands:
  - `cargo fmt --check`
  - `git diff --check`
  - `cargo check`
  - `cargo check -p editor`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p engine_pack --locked`
- Focused tests:
  - `cargo test -p editor` if editor tests exist or are added.
  - `cargo test -p renderer scene`
  - `cargo test -p renderer asset_registry`
  - `cargo test -p engine_pack --locked`
- CLI validation:
  - `cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml`
  - `cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml`
  - A sprint-local saved-scene-copy validation command recorded by Phase 02.
- Visual validation:
  - Use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
  - Prefer `cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene <saved-scene-copy> --headless --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement`.
  - If editor automation cannot deterministically create or display the placement, create a focused renderer capture scene/example that loads the same sample project package and saved scene data path, then capture through renderer headless capture commands.

## Negative Criteria

- No runtime `SceneNodeId`, `MeshHandle`, `TextureHandle`, `MaterialHandle`, `EnvironmentHandle`, `LoadTicket`, slot/generation pair, or other runtime handle may be serialized as durable scene identity.
- No binary packaging/archive claims.
- No broad UI redesign.
- No global editor architecture rewrite.
- No visual validation claim without capture PNG and sidecar evidence.
- No mutation of the canonical sample scene unless the worker explicitly justifies that a source-controlled sample update is the intended product artifact; otherwise use temp copies under `.internal-dev/headless_capture_tests/` or sprint artifacts.
- Do not close Sprint 01.
- Do not include unrelated `.idea/engine.iml` or `.reasonix/` changes in Sprint 03 evidence, commits, or summaries.

## Constraints

- Code is the logical source of truth; docs are intended truth.
- `.internal-dev` is untracked and is the durable sprint document/evidence store.
- Headless capture validation is required because this sprint changes and claims visible editor/runtime behavior.
- Main thread handles commits, pushes, and email reports. Workers and validators record evidence expectations and changed files, but do not own out-of-band coordination.

## Assumptions To Verify

- `apps/editor/src/main.rs` already supports `--headless`, frame capture flags, package loading, `SaveScene`, and `LoadScene`.
- `Scene::save`, `Scene::load`, `PlaceAssetCommand`, and `engine_pack validate-scene` already preserve most durable identity behavior, but need focused tests around editor-generated placement.
- Editor UI automation may not be sufficient for deterministic placement from a headless run; Phase 03 may need a small capture-focused harness.

## User-Decision Gates

- If a worker believes the canonical sample scene should be changed, stop and ask the main thread to approve that product artifact change.
- If editor headless automation cannot drive placement without adding a broader automation system, prefer a small capture-focused fixture/example and record why.
- If a required validator/model/tool is unavailable, record `TOOLING_CONSTRAINT` and stop for main-thread/user approval before substituting.

## Stop Rules

- Stop feature work if runtime handles appear in saved durable scene JSON.
- Stop closeout if capture evidence is missing or inconclusive.
- Stop phase progression when a phase validator fails until remediation is routed and revalidated.
- Stop if implementation requires unrelated renderer/Vulkan rewrites to satisfy this sprint.
