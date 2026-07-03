# Phase 02 Validation Report

Date: 2026-07-03

Branch: `sprint/alpha-02-packaging-tools`

Sprint target: asset package authoring and validation tools.

Phase target: read-only Rust CLI validation commands for package, project, and scene files.

Phase status: `passed_red_team_remediation_pending_commit`

Capture decision: `not_required_cli_schema_only`

## Scope

Implemented the `engine_pack` Rust CLI crate as a workspace member with read-only validation commands:

- `validate-package <path> [--expected-package-id <id>] [--project-root <path>]`
- `validate-project <path>`
- `validate-scene <path> --project <path>`

Out of scope by directive: package authoring, generated manifests, asset copying, editor placement UI, Vulkan/rendergraph/shader/runtime visual changes, and Sprint 01 closeout.

## Changed Files Matrix

Phase 02 implementation commit: pending at report creation.

Branch link: `https://github.com/dhickel/vulkan-engine/tree/sprint/alpha-02-packaging-tools`

| File | Created/Changed/Deleted | Lines After | GitHub Link |
|---|---:|---:|---|
| `Cargo.toml` | Changed | 14 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/Cargo.toml` |
| `Cargo.lock` | Changed | 4867 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/Cargo.lock` |
| `tools/engine_pack/Cargo.toml` | Created | 7 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/Cargo.toml` |
| `tools/engine_pack/src/main.rs` | Created | 245 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/src/main.rs` |
| `tools/engine_pack/tests/cli_validation.rs` | Created | 170 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/tests/cli_validation.rs` |
| `tools/engine_pack/fixtures/packages/duplicate-id.package.toml` | Created | 13 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/packages/duplicate-id.package.toml` |
| `tools/engine_pack/fixtures/packages/missing-version.package.toml` | Created | 7 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/packages/missing-version.package.toml` |
| `tools/engine_pack/fixtures/packages/runtime-handle.package.toml` | Created | 11 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/packages/runtime-handle.package.toml` |
| `tools/engine_pack/fixtures/packages/valid.package.toml` | Created | 8 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/packages/valid.package.toml` |
| `tools/engine_pack/fixtures/packages/models/crate.obj` | Created | 1 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/packages/models/crate.obj` |
| `tools/engine_pack/fixtures/projects/valid/engine.project.toml` | Created | 17 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/projects/valid/engine.project.toml` |
| `tools/engine_pack/fixtures/projects/valid/assets/fixture.package.toml` | Created | 8 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/projects/valid/assets/fixture.package.toml` |
| `tools/engine_pack/fixtures/projects/valid/assets/models/crate.obj` | Created | 0 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/projects/valid/assets/models/crate.obj` |
| `tools/engine_pack/fixtures/projects/valid/scenes/start.engine.scene.json` | Created | 24 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/projects/valid/scenes/start.engine.scene.json` |
| `tools/engine_pack/fixtures/projects/invalid_missing_scene/engine.project.toml` | Created | 13 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/projects/invalid_missing_scene/engine.project.toml` |
| `tools/engine_pack/fixtures/scenes/unknown-asset.engine.scene.json` | Created | 24 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/fixtures/scenes/unknown-asset.engine.scene.json` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-02-validation-report.md` | Created | 153 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-02-validation-report.md` |

## CLI Behavior Evidence

- Successful commands print stable `valid[...]` status lines to stdout.
- Validation failures print stable `error[code]: ...` diagnostics to stderr and exit with code 1.
- CLI usage failures exit with code 2.
- `validate-scene` composes project package metadata and rejects unknown durable asset IDs.
- `validate-project` verifies enabled package manifests, source files, and startup scene existence.
- The CLI depends on the renderer crate validators and does not duplicate schema parsing logic.

## Red-Team Remediation

Validation agent `019f26ec-d3e4-76d0-82f4-36964a0bf300` initially failed Phase 02 on three findings:

- `fixtures/packages/valid.package.toml` was named as a valid fixture but used an escaping source path.
- The custom argument parser mishandled `--flag value <path>` ordering for package and scene validation.
- The report's dirty-state section did not distinguish in-scope pending sprint evidence from unrelated local dirty state.

Remediation:

- Made the standalone valid package fixture self-contained at `fixtures/packages/models/crate.obj`.
- Consumed command options before selecting positional paths so both `path --flag value` and `--flag value path` work.
- Added regression coverage for option-before-path order and the standalone valid package fixture.
- Clarified dirty-state reporting below.

## Validation Commands

```bash
cargo fmt --check
```

Result: passed.

```bash
cargo check -p engine_pack
```

Result: passed with existing renderer warnings.

```bash
cargo test -p engine_pack
```

Result: passed. 6 CLI integration tests passed; 0 failed.

```bash
cargo run -p engine_pack -- validate-package apps/editor/sample_project/assets/editor_sample.package.toml --expected-package-id editor_sample
```

Result: passed. Output: `valid[package]: apps/editor/sample_project/assets/editor_sample.package.toml (2 assets)`.

```bash
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
```

Result: passed. Output: `valid[project]: apps/editor/sample_project/engine.project.toml (project.editor_sample)`.

```bash
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

Result: passed. Output: `valid[scene]: apps/editor/sample_project/scenes/start.engine.scene.json`.

```bash
cargo run -p engine_pack -- validate-package tools/engine_pack/fixtures/packages/valid.package.toml --project-root tools/engine_pack/fixtures/packages --expected-package-id fixture
```

Result: passed. Output: `valid[package]: tools/engine_pack/fixtures/packages/valid.package.toml (1 assets)`.

```bash
cargo run -p engine_pack -- validate-package --expected-package-id editor_sample apps/editor/sample_project/assets/editor_sample.package.toml
```

Result: passed. Output: `valid[package]: apps/editor/sample_project/assets/editor_sample.package.toml (2 assets)`.

```bash
cargo run -p engine_pack -- validate-scene --project apps/editor/sample_project/engine.project.toml apps/editor/sample_project/scenes/start.engine.scene.json
```

Result: passed. Output: `valid[scene]: apps/editor/sample_project/scenes/start.engine.scene.json`.

## Dirty State

In-scope pending Phase 02 changes before commit:

```text
 M .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/validation-summary.json
 M Cargo.lock
 M Cargo.toml
?? tools/engine_pack/
```

Unrelated dirty state intentionally excluded from Phase 02 staging:

```text
 M .idea/engine.iml
?? .reasonix/
```

## Notes And Residual Risk

- This phase is intentionally read-only. It gives Sprint 02 a validation executable before adding authoring commands in Phase 03.
- Visual capture was not required because no renderer output, scene placement, shader, material, Vulkan, camera, or editor visual behavior changed.
- Existing renderer dead-code warning profile remains unchanged by this phase.
