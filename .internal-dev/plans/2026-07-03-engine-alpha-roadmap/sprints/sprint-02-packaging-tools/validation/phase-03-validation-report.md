# Phase 03 Validation Report

Date: 2026-07-03

Branch: `sprint/alpha-02-packaging-tools`

Sprint target: asset package authoring and validation tools.

Phase target: authoring commands and folder-based pack output for alpha packaging.

Phase status: `passed_red_team_remediation_committed_pushed_pending_report`

Capture decision: `not_required_cli_schema_only`

## Scope

Implemented alpha authoring and folder-pack commands in the Rust `engine_pack` CLI:

- `new-project <dir> --id <project_id> --name <name>`
- `new-package <path> --id <package_id> --name <display_name>`
- `scan-assets <asset-root> [--package-id <id>]`
- `add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]`
- `pack <engine.project.toml> --out <dir>`

Out of scope by directive: binary archives, thumbnail generation, editor UI import/placement, runtime launcher, audio/script/collision API expansion, and visual renderer changes.

## Changed Files Matrix

Phase 03 implementation commit: `e7f5557412c05dd7b851e04ba5fdaf9cffc08c49`

Phase 03 implementation commit link: `https://github.com/dhickel/vulkan-engine/commit/e7f5557412c05dd7b851e04ba5fdaf9cffc08c49`

Branch link: `https://github.com/dhickel/vulkan-engine/tree/sprint/alpha-02-packaging-tools`

AgentMail report: `pending`

| File | Created/Changed/Deleted | Lines After | GitHub Link |
|---|---:|---:|---|
| `Cargo.lock` | Changed | 4870 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/Cargo.lock` |
| `tools/engine_pack/Cargo.toml` | Changed | 10 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/Cargo.toml` |
| `tools/engine_pack/src/main.rs` | Changed | 710 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/src/main.rs` |
| `tools/engine_pack/tests/cli_validation.rs` | Changed | 456 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/tools/engine_pack/tests/cli_validation.rs` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/phase-03-smoke.log` | Created | 43 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/phase-03-smoke.log` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/engine.project.toml` | Created | 13 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/engine.project.toml` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/smoke.package.toml` | Created | 10 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/smoke.package.toml` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/PACK_REPORT.json` | Created | 12 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/PACK_REPORT.json` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-03-validation-report.md` | Created | 187 | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-03-validation-report.md` |

## CLI Behavior Evidence

- `new-project` creates `assets/`, `scenes/`, `engine.project.toml`, and a starter scene that validates through the Phase 02 project and scene validators.
- `new-package` creates a minimal package manifest and validates it with an expected package ID.
- `scan-assets` walks known model, texture, and environment extensions and prints deterministic `[[assets]]` TOML suggestions without mutating the source tree.
- `add-asset` appends one asset table to an existing package manifest, preserves existing content order, accepts repeated tags, and validates source files after mutation.
- `pack` validates the project/startup scene/package manifests/source files, copies project-relative files into a folder output, and writes `PACK_REPORT.json`.
- `pack` rejects absolute and parent-traversing project package paths.
- `pack` rejects raw parent-traversing package asset paths even when the shared runtime validator would normalize them.
- failed repacks remove stale `PACK_REPORT.json` before validation, preventing an old successful report from surviving a failed run.

## Generated Smoke Artifacts

Sprint-local smoke output path:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/
```

Generated project/package:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/engine.project.toml
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/smoke.package.toml
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/models/smoke.obj
```

Generated pack output:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/PACK_REPORT.json
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/assets/editor_sample.package.toml
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/assets/models/block_prop.obj
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/assets/prefabs/wall_straight_2m.obj
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/engine.project.toml
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed/scenes/start.engine.scene.json
```

## Red-Team Remediation

Validation agent `019f26fa-33dc-7e41-9e16-ac28bc159fc0` initially found two high-severity pack bugs and one evidence gap:

- raw parent-traversing package asset paths could be normalized and copied;
- failed repacks could leave a stale successful `PACK_REPORT.json`;
- Phase 03 report and validation summary were not yet written.

Remediation:

- Added strict raw package manifest parsing in `pack` so package asset `path` values containing parent traversal are rejected before copy.
- Added `pack_rejects_parent_traversing_asset_paths`.
- Moved stale `PACK_REPORT.json` cleanup before validation can abort a repack.
- Added `failed_repack_removes_stale_success_report`.
- Wrote this Phase 03 validation report and updated sprint validation summary.

## Validation Commands

```bash
cargo fmt --check
```

Result: passed.

```bash
cargo check -p engine_pack --locked
```

Result: passed with existing renderer warnings.

```bash
cargo test -p engine_pack --locked
```

Result: passed. 13 CLI integration tests passed; 0 failed.

```bash
cargo run -p engine_pack -- new-project .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project --id project.engine_pack_smoke --name "Engine Pack Smoke"
```

Result: passed. Output: `created[project]`.

```bash
cargo run -p engine_pack -- new-package .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/smoke.package.toml --id smoke --name "Smoke Assets"
```

Result: passed. Output: `created[package]`.

```bash
cargo run -p engine_pack -- add-asset .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets/smoke.package.toml --id smoke.model.smoke --kind model --path models/smoke.obj --tag smoke --tag generated
```

Result: passed. Output: `added[asset]`.

```bash
cargo run -p engine_pack -- scan-assets .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/project/assets --package-id smoke
```

Result: passed. Output included deterministic TOML suggestion for `smoke.model.models.smoke`.

```bash
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
```

Result: passed. Output: `valid[project]: apps/editor/sample_project/engine.project.toml (project.editor_sample)`.

```bash
cargo run -p engine_pack -- pack apps/editor/sample_project/engine.project.toml --out .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/engine-pack-smoke/packed
```

Result: passed. Output: `packed[project]` with 5 copied files and `PACK_REPORT.json`.

## Dirty State

In-scope pending Phase 03 changes before commit:

```text
 M Cargo.lock
 M tools/engine_pack/Cargo.toml
 M tools/engine_pack/src/main.rs
 M tools/engine_pack/tests/cli_validation.rs
```

In-scope ignored sprint evidence to force-add:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/phase-03/
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-03-validation-report.md
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/validation-summary.json
```

Unrelated dirty state intentionally excluded from Phase 03 staging:

```text
 M .idea/engine.iml
?? .reasonix/
```

## Notes And Residual Risk

- Visual capture was not required because this phase changed only CLI/schema/filesystem behavior and did not alter renderer output, scene placement UI, shaders, materials, camera, Vulkan behavior, or editor visuals.
- The pack output is intentionally folder-based. Binary archives and dependency semantics remain out of scope.
- The `toml = "=0.8.23"` dependency is an exact direct dependency on an already locked workspace crate version, used only to inspect raw package manifest asset paths before folder copy.
