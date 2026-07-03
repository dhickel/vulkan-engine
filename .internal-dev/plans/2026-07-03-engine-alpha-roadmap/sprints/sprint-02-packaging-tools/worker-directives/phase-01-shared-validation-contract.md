# Phase 01 Worker Directive: Shared Validation Contract

## Objective

Create or expose shared Rust validation for project, package, and scene files so the future CLI and editor/runtime paths do not diverge.

## User-Visible Outcome

The engine has one Rust validation contract for alpha package/project/scene files, with clear diagnostics and tests for durable identity rules.

## Editable Targets

- `src/renderer/src/data/asset_registry.rs`
- possible new renderer data/api validation modules under `src/renderer/src/data/` or `src/renderer/src/api/`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/mod.rs`
- renderer tests under `src/renderer/tests/` or module tests in touched files
- sprint evidence under this sprint directory only

## Forbidden Scope

- Do not create the CLI crate in this phase unless a tiny test harness is absolutely required.
- Do not change Vulkan, rendergraph, shader, or visual runtime behavior.
- Do not stage `.idea/engine.iml` or `.reasonix/`.
- Do not close Sprint 01.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`

## Senior Guidance

- Package parsing is already strong; project validation is the obvious gap.
- Scene validation is currently private and load-oriented. Expose a narrow validation API or add a validation-only parser without making the CLI the source of truth.
- Use typed error/diagnostic structures where possible; string-only errors make stable CLI assertions brittle.
- Keep validation IO-light and Vulkan-free.
- Preserve current editor behavior unless tests prove a bug and the fix is narrow.

## Implementation Steps

1. Confirm branch and dirty state.
2. Add shared validation entrypoints for package, project, and scene files.
3. Strengthen project validation for format version, empty IDs, invalid paths, missing package manifests when validating from a file, duplicate enabled package IDs, startup scene existence, and settings sanity.
4. Add scene validation that checks format version, graph roots, duplicate stable node IDs, missing parents, durable asset references, and runtime handle-shaped identity.
5. Ensure package validation can report missing source files when called in file-validation mode.
6. Add focused tests for valid sample-like inputs and the invalid cases named in `shared/implementation-notes.md`.
7. Re-export only the minimal validation surface intended for CLI/editor use.
8. Write phase evidence and leave commit/push/email gates for the orchestrator after validation.

## Acceptance Criteria

- Shared validation functions exist for project, package, and scene validation.
- Renderer tests cover durable identity rejection and major invalid fixtures.
- Validators can run without constructing `Renderer` or initializing Vulkan.
- Existing package load APIs keep working.

## Negative Checks

- No runtime handle serialization accepted as identity.
- No duplicate validation logic owned only by a future CLI.
- No visual/runtime claim.

## Validation Commands

```bash
cargo check -p renderer
cargo check -p renderer --examples
cargo test -p renderer asset_registry
cargo test -p renderer scene
```

Adjust focused test names to match implementation. Record exact commands.

## Evidence Expectations

- Validation report path: `validation/phase-01-validation-report.md`
- Include files/line counts/git links matrix.
- Include commands and results.
- Include capture decision `not_required_cli_schema_only` unless this phase unexpectedly changes visual behavior.

## Commit/Push/AgentMail Gate

After phase validation passes, orchestrator must commit scoped changes, push `sprint/alpha-02-packaging-tools`, and send an AgentMail HTML progress report using `email-report-template.html`.

## Stop Conditions

- Stop if exposing scene validation requires broad scene loader redesign.
- Stop if project/package validation requires deciding new package dependency semantics.
- Stop if tests reveal editor sample project is invalid and the fix is not obviously schema-aligned.

## Do Not Close Unless

- Validation report exists.
- Required checks ran or blockers are recorded.
- Durable identity checks are explicit.
- Commit hash, pushed ref, GitHub links, and AgentMail evidence are recorded after orchestration gates.
