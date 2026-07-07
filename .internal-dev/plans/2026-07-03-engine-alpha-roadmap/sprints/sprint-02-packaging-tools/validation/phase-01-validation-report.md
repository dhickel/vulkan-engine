# Phase 01 Validation Report

Date: 2026-07-03

Branch: `sprint/alpha-02-packaging-tools`

Sprint target: asset package authoring and validation tools.

Phase target: shared renderer-owned validation contract for package, project, and scene files.

Phase status: `passed_local_validation_committed_pending_push`

Capture decision: `not_required_cli_schema_only`

## Scope

Implemented shared Rust validation entrypoints for renderer-owned package, project, and scene schemas. Validation is Vulkan-free and does not construct `Renderer`.

Out of scope by directive: no CLI crate, no Vulkan/rendergraph/shader/visual runtime behavior changes, no Sprint 01 closeout.

## Changed Files Matrix

Phase 01 implementation commit: `fa7abc33ea76c7f81edd3b04518a37d680bf695a`

Branch link: `https://github.com/dhickel/vulkan-engine/tree/sprint/alpha-02-packaging-tools`

| File | Created/Changed/Deleted | Added Lines | Removed Lines | Lines After | Commit | GitHub Link |
|---|---:|---:|---:|---:|---|---|
| `src/renderer/src/data/validation.rs` | Created | 121 | 0 | 121 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/data/validation.rs` |
| `src/renderer/src/data/asset_registry.rs` | Changed | 935 | 3 | 1566 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/data/asset_registry.rs` |
| `src/renderer/src/data/mod.rs` | Changed | 1 | 0 | 16 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/data/mod.rs` |
| `src/renderer/src/api/scene.rs` | Changed | 451 | 3 | 2545 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/api/scene.rs` |
| `src/renderer/src/api/mod.rs` | Changed | 9 | 4 | 56 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/api/mod.rs` |
| `src/renderer/src/lib.rs` | Changed | 13 | 9 | 42 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/src/renderer/src/lib.rs` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md` | Changed | 2 | 2 | 51 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/SPRINT-TRACKER.md` |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-01-validation-report.md` | Created | 134 | 0 | 134 | `fa7abc33` | `https://github.com/dhickel/vulkan-engine/blob/sprint/alpha-02-packaging-tools/.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/validation/phase-01-validation-report.md` |

## Validation Surface

- `ValidationDiagnostic`, `ValidationError`, and `ValidationArea` provide stable diagnostic codes plus optional file path and durable ID context.
- Package validation entrypoints:
  - `validate_package_manifest_str`
  - `validate_package_manifest_file`
  - `PackageValidationOptions`
- Project validation entrypoints:
  - `validate_project_str`
  - `validate_project_file`
  - `ProjectValidationOptions`
- Scene validation entrypoints:
  - `validate_scene_str`
  - `validate_scene_file`
  - `validate_scene_str_with_options`
  - `validate_scene_file_with_options`
  - `SceneValidationOptions`

## Criteria Evidence

- Shared validation functions exist for project, package, and scene validation: satisfied.
- Validators run without constructing `Renderer` or initializing Vulkan: satisfied by string/file validators and module tests.
- Durable identity rejection is explicit:
  - package asset IDs still reject empty/path-shaped IDs through existing asset registry logic.
  - package validator rejects runtime handle-shaped asset fields with `asset.runtime_handle_identity`.
  - package validator now recursively rejects nested runtime handle-shaped TOML values under asset metadata.
  - scene validator rejects node/runtime handle-shaped identity with `scene.runtime_handle_identity`.
  - scene validator rejects missing durable asset IDs with `scene.missing_asset_id`.
- Project validation strengthened:
  - format version, empty IDs/names, project-relative path rules, duplicate enabled package IDs, missing enabled package manifests, startup scene existence, duplicate enabled asset IDs, and window size sanity.
- Package validation strengthened:
  - file-validation mode can report missing source files with `asset.missing_source_path`.
  - tests cover missing/unsupported package versions, duplicate IDs, path-shaped IDs, escaping paths, package ID mismatch, and nested runtime handles.
- Scene validation strengthened/exposed:
  - format version, empty scene/node IDs, duplicate stable node IDs, missing parents, root graph mismatch/disconnected graph, missing durable asset IDs, optional unknown asset IDs from caller-supplied package metadata, and runtime handle-shaped identity.
- Existing package load APIs keep working:
  - Existing `AssetRegistry::load_package_manifest*` paths retained.
  - `Project::load` now delegates to shared project validation in parse-only mode to preserve existing editor behavior.

## Red-Team Remediation

Validation agent `019f26e0-1880-7b10-8e44-bf39a0002151` initially failed Phase 01 on two issues:

- Package validation did not recursively catch nested runtime handles under metadata.
- Package/project invalid-case tests did not cover enough of the locked cases.

Remediation:

- Added recursive TOML runtime-handle scanning for every value under each `[[assets]]` table.
- Added tests for nested runtime handles, missing/unsupported package versions, duplicate package asset IDs, path-shaped asset IDs, escaping asset paths, package ID mismatch, and missing startup scenes.

## Validation Commands

```bash
cargo fmt --check
```

Result: passed.

```bash
cargo test -p renderer asset_registry
```

Result: passed. 8 renderer asset registry tests passed; 0 failed.

```bash
cargo test -p renderer scene
```

Result: passed. 36 lib scene-filtered tests and 2 integration scene-filtered tests passed; 0 failed.

```bash
cargo check -p renderer
```

Result: passed. Existing renderer warning profile remains.

```bash
cargo check -p renderer --examples
```

Result: passed. Existing renderer warning profile remains.

## Dirty State

Unrelated dirty state remains intentionally excluded from Phase 01 staging:

```text
 M .idea/engine.iml
?? .reasonix/
```

Sprint tracker changes are in scope for Sprint 02 orchestration and will be included with the phase commit.

## Notes And Residual Risk

- API docs describe `default_environment` as an object, while the current Rust `Project` schema remains `Option<PathBuf>`. This phase did not migrate that schema because the directive constrained work to shared validation and preserving current editor behavior.
- Scene validation can report unknown durable asset IDs when a caller supplies known package asset IDs. The later CLI phase still needs to compose project/package loading with scene validation to pass those IDs in.
- No visual behavior changed; headless capture was not required.
