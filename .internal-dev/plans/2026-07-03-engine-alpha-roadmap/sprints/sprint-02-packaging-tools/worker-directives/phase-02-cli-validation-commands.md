# Phase 02 Worker Directive: CLI Validation Commands

## Objective

Create the Rust workspace CLI and implement read-only validation commands for packages, projects, and scenes.

## User-Visible Outcome

Users can run `engine_pack validate-package`, `engine_pack validate-project`, and `engine_pack validate-scene` against real and fixture files and receive stable readable diagnostics.

## Editable Targets

- root `Cargo.toml`
- `tools/engine_pack/Cargo.toml`
- `tools/engine_pack/src/**`
- `tools/engine_pack/tests/**`
- `tools/engine_pack/fixtures/**`
- minimal renderer validation exports only if Phase 01 left a narrow gap
- sprint evidence under this sprint directory

## Forbidden Scope

- Do not implement authoring or pack commands in this phase.
- Do not change editor UI or runtime rendering.
- Do not add Python tooling.
- Do not stage `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `shared/validation-matrix.md`
- Phase 01 validation report and commits

## Senior Guidance

- Keep command behavior deterministic before making output pretty.
- Prefer stable diagnostic codes or stable message prefixes that integration tests can assert.
- The sample project validation is the sprint's main real-world smoke test.
- `validate-scene` should resolve durable asset IDs through project/package metadata when given a project.

## Implementation Steps

1. Add `tools/engine_pack` as a workspace member.
2. Implement command parsing for validation commands.
3. Implement `validate-package <path> [--expected-package-id <id>]`.
4. Implement `validate-project <path>` to validate project shape, package manifests, package asset files, startup scene path, and referenced scene/package asset IDs.
5. Implement `validate-scene <path> --project <path>`; allow narrower flags only if useful and tested.
6. Add fixtures for valid and invalid project/package/scene cases.
7. Add CLI tests that assert exit codes and stable diagnostics.
8. Run sample project validation through the CLI.

## Acceptance Criteria

- `cargo check -p engine_pack` passes.
- `cargo test -p engine_pack` passes.
- Sample editor project validates successfully.
- At least one invalid fixture per major category fails with a stable diagnostic.
- CLI validation uses shared Rust validators from Phase 01.

## Negative Checks

- CLI must not call Vulkan or open a renderer window.
- CLI must not accept path-only asset identity as valid durable identity.
- CLI must not disagree with editor package loading for enabled package IDs.

## Validation Commands

```bash
cargo check -p engine_pack
cargo test -p engine_pack
cargo run -p engine_pack -- validate-package apps/editor/sample_project/assets/editor_sample.package.toml --expected-package-id editor_sample
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

## Evidence Expectations

- Validation report path: `validation/phase-02-validation-report.md`
- Include files/line counts/git links matrix.
- Include command outputs or summarized key lines for every CLI smoke.
- Include capture decision `not_required_cli_schema_only`.

## Commit/Push/AgentMail Gate

After phase validation passes, orchestrator must commit scoped changes, push `sprint/alpha-02-packaging-tools`, and send an AgentMail HTML progress report using `email-report-template.html`.

## Stop Conditions

- Stop if `engine_pack` cannot depend on renderer without unacceptable compile or dependency consequences; return to planning for a crate-boundary revision.
- Stop if validation requires undefined package dependency semantics.
- Stop if sample project fails due to nontrivial schema mismatch.

## Do Not Close Unless

- CLI validation commands exist and are tested.
- Sample project validates.
- Validation report records commit, pushed ref, GitHub links, and AgentMail evidence after orchestration gates.
