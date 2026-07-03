# Sprint 02: Asset Package Authoring And Validation Tools

Status: planned

## Objective

Create an alpha Rust packaging CLI and shared validation path for project, package, and scene authoring workflows.

## User-Visible Outcome

An alpha project author can run a Rust CLI to validate the editor sample project, validate package and scene files, create starter project/package files, add or scan assets, and produce a folder-based package output without hand-editing every manifest.

## Scope

- Add a workspace CLI crate, expected path `tools/engine_pack`, and include it in the root Cargo workspace.
- Add or expose shared Rust validation functions around existing renderer project/package/scene schemas so the CLI, editor, and runtime do not diverge.
- Support Sprint 02 gate commands: `validate-package`, `validate-project`, `validate-scene`, `new-project`, `new-package`, `scan-assets`, `add-asset`, and `pack` where feasible within the phase constraints.
- Validate `apps/editor/sample_project/engine.project.toml` and its package/scene files.
- Add valid and invalid fixtures with stable, readable error assertions.
- Update only needed docs/API references for the CLI and schema behavior.

## Out Of Scope

- Binary archive/package format.
- Thumbnail rendering or material graph import.
- Editor UI placement hardening.
- Runtime project launcher.
- Resolving Sprint 01 changelog timing. Sprint 01 remains blocked until the user confirms changelog timing.

## Target Surfaces

- Workspace: `Cargo.toml`, new `tools/engine_pack/Cargo.toml`, new `tools/engine_pack/src/`.
- Shared validation candidates: `src/renderer/src/data/asset_registry.rs`, `src/renderer/src/api/scene.rs`, possibly renderer-owned validation modules or test helpers.
- Tests and fixtures: `tools/engine_pack/tests/`, `tools/engine_pack/fixtures/`, focused renderer unit tests where shared validation is exposed.
- Existing sample project: `apps/editor/sample_project/`.
- Docs: `docs/api/04-assets-sync-deferred-and-handles.md`, `docs/api/03-scene-graph-and-fragment-workflows.md`, optional CLI doc if the worker creates one.
- Sprint evidence: this sprint directory, `artifacts/validation-summary.json`, phase validation reports, AgentMail HTML reports.

## Acceptance Criteria

- The CLI is Rust-based and a Cargo workspace member.
- CLI validation succeeds for `apps/editor/sample_project/engine.project.toml`.
- Invalid package/project/scene fixtures fail with stable, readable messages that identify file, field/record, and reason.
- Shared validation preserves durable identity: project IDs, package IDs, asset IDs, scene IDs, node stable IDs, and durable asset references stay as strings/paths, never runtime handles.
- The editor/runtime package loading path either calls shared validation directly or remains covered by a compatibility test that prevents CLI/editor disagreement.
- Each phase produces a validation report, scoped commit, pushed branch, AgentMail HTML progress report, and changed files/line counts/git links matrix.
- Final status is conservative. Do not record `fully_validated` if any required validation, push, email, or capture decision remains unresolved.

## Negative Criteria

- No Python canonical validator unless a hard blocker is documented and approved.
- No serialization of `SceneNodeId`, `PointLightId`, `MeshHandle`, `TextureHandle`, `MaterialHandle`, `EnvironmentHandle`, or `LoadTicket` as durable identity.
- No broad renderer/Vulkan behavior change unless the phase explicitly proves it and uses headless capture validation.
- No committing unrelated dirty state in `.idea/engine.iml` or `.reasonix/`.
- No final claim that visual asset placement/render readiness was proven unless headless capture evidence exists.

## Validation Plan

- Compile/test: `cargo check`, `cargo check -p renderer`, `cargo check -p renderer --examples`, `cargo check -p input`, `cargo check -p engine_pack`, `cargo test -p engine_pack`, and focused renderer tests for shared validators.
- CLI smoke: validate sample package, sample project, sample scene, and invalid fixtures.
- Runtime smoke: not required by default unless implementation changes runtime/render behavior.
- Visual/capture proof: default decision is `not_required_cli_schema_only`. If a worker claims rendered asset placement/readiness or changes scene/asset loading in a way that affects rendered output, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Docs/process: stale reference sweep, tracker update, validation summary consistency, per-phase email/report artifacts.

## Phase List

1. Phase 01: Shared validation contract and schema hardening.
2. Phase 02: CLI validation commands and negative fixtures.
3. Phase 03: Authoring, scanning, add-asset, and folder pack commands.
4. Phase 04: Docs, sample project proof, final validation, and closeout evidence.

## Closeout Checklist

- All phase reports exist under `validation/`.
- `artifacts/validation-summary.json` reconciles phase status, commands, capture decision, commits, pushes, email reports, and residuals.
- Sprint tracker reflects final status after orchestration.
- Changelog timing is handled by the main thread after user confirmation, per repo guidance.
