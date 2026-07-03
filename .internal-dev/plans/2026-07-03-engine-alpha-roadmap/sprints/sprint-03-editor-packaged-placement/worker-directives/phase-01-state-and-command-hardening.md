# Phase 01 Worker Directive: State And Command Hardening

## Objective

Inspect and harden the editor packaged asset selection/placement command flow so placing a package-backed asset produces a durable scene node, selects it, supports undo/redo coherently, and reports clear status messages.

## User-Visible Outcome

In the editor, selecting a package asset and confirming placement creates a visible scene hierarchy/inspector selection with durable asset metadata. Undo/redo does not leave stale selections or silent failures.

## Editable Targets

- `apps/editor/src/app_state.rs`
- `apps/editor/src/main.rs`
- `apps/editor/src/panels.rs` only for minimal affordance/status fixes
- `src/renderer/src/scene/command.rs`
- `src/renderer/src/api/scene.rs` only for focused command/selection support
- Tests in the same modules or existing package test locations
- Phase evidence under `validation/phase-01-validation-report.md`

## Forbidden Scope

- Do not change binary packaging/archive behavior.
- Do not redesign the editor layout.
- Do not mutate `apps/editor/sample_project/scenes/start.engine.scene.json`.
- Do not close Sprint 01.
- Do not include `.idea/engine.iml` or `.reasonix/` in changes.

## Supporting Docs To Read

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `shared/senior-engineer-guidance.md`
- `shared/implementation-notes.md`

## Senior Engineer Guidance

- Placement must go through `PlaceAssetCommand`; this preserves undo/redo and keeps editor state separate from scene mutation.
- Selection is runtime-node-based while durability is stable-ID-based. Tests should cover the transition points where runtime IDs change.
- Status messages are part of the alpha UX contract. A failed placement, stale selection, undo, redo, or missing asset should explain itself.
- Keep fixes local. If you feel pulled into a general editor architecture rewrite, stop and report the needed smaller interface seam instead.

## Ordered Implementation Steps

1. Inspect existing `EditorAction::SelectAsset`, `StartPlacement`, `CancelPlacement`, `ConfirmPlacement`, `Undo`, `Redo`, selection cleanup, and `PlaceAssetCommand` behavior.
2. Add or harden focused tests for `EditorSession` asset selection, placement state, stable placement ID generation, cancel behavior, and status limits.
3. Add or harden focused command tests proving `PlaceAssetCommand` stamps root stable ID, asset reference, name, tags, transform, and returns `created_node`.
4. Harden editor action processing so successful placement selects the created node and failed placement leaves a clear status without clearing unrelated useful state.
5. Harden undo/redo selection cleanup or remapping for placed nodes.
6. Ensure new placement after undo clears redo according to command history contract.
7. Run focused and package checks.
8. Update `artifacts/validation-summary.json` phase 01 fields only if implementation occurs in this phase; keep status conservative until validator passes.

## Acceptance Criteria

- Confirm placement of a loaded durable asset creates a node through `PlaceAssetCommand`.
- Created root has durable stable ID, `SceneAssetReference.id`, path hint, display name, tags, and expected transform.
- Editor selection points at the created node after placement.
- Undo removes the placed node and does not leave an invalid active selection.
- Redo recreates or reselects the placed node when command result exposes it.
- Missing/unloaded asset and no-active-placement paths push status messages and do not panic.
- Tests cover the critical state/command behavior.

## Negative Checks

- No runtime handle serialization.
- No canonical sample scene mutation.
- No broad UI redesign.
- No final visual proof claim in this phase.

## Validation Commands

Run as applicable and record exact output summaries:

```bash
cargo fmt --check
cargo check -p editor
cargo test -p editor
cargo test -p renderer scene
cargo test -p renderer asset_registry
git diff --check
```

If `cargo test -p editor` has no tests or cannot run because the package is bin-only, record that fact and compensate with focused renderer/editor module tests where practical.

## Evidence Expectations

- Write or prepare evidence for `validation/phase-01-validation-report.md`.
- Record commands, results, changed files, tests added, and residual risks.
- Note whether `.idea/engine.iml` or `.reasonix/` were present and explicitly excluded.
- Main thread will handle commit, push, and email report evidence after validation passes.

## Stop Conditions

- Stop if placement hardening requires a global editor architecture rewrite.
- Stop if a saved durable scene would require runtime handle identity.
- Stop if tests require mutating the canonical sample scene.
- Stop and route to planning if acceptance criteria conflict with current renderer scene API contracts.

## Do Not Close Unless

- Focused tests prove placement command identity and editor selection behavior.
- Required validation commands have run or blockers are documented.
- Phase report path and summary evidence are ready for validation.
