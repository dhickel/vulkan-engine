# Target Design

## CLI Shape

Preferred workspace member:

```text
tools/engine_pack/
  Cargo.toml
  src/main.rs
  src/commands.rs
  src/output.rs
  src/fs.rs
  tests/
  fixtures/
```

Expected binary/package name: `engine_pack`.

Use a small explicit argument parser. `clap` is acceptable if the worker judges the dependency worth it; otherwise a hand-rolled parser is acceptable for Sprint 02. The key contract is stable commands and deterministic diagnostics.

## Commands

- `validate-package <package.toml> [--expected-package-id <id>] [--project-root <path>]`
- `validate-project <engine.project.toml>`
- `validate-scene <scene.engine.scene.json> --project <engine.project.toml>` or `--package <package.toml>` when useful.
- `new-project <dir> --id <project_id> --name <name>`
- `new-package <path> --id <package_id> --name <display_name>`
- `scan-assets <asset-root> [--package-id <id>]`
- `add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]`
- `pack <engine.project.toml> --out <dir>`

Workers may add aliases only if the canonical commands above remain documented and tested.

## Shared Validation Boundary

Keep the first shared boundary close to existing renderer schemas:

- Package/project parsing and validation in `src/renderer/src/data/asset_registry.rs` or a child module under `src/renderer/src/data/`.
- Public facade exports through `src/renderer/src/lib.rs` or `src/renderer/src/api/mod.rs` only for alpha-supported validator types/functions.
- Scene schema validation exposed through a narrow function that validates JSON and returns diagnostics without loading real meshes/textures/environments.

If exposing current private scene structs is too invasive, create a small validation-only mirror in the renderer crate and require compatibility tests against `Scene::load`/save behavior. Do not let CLI-owned duplicated schema become canonical.

## Diagnostic Contract

Use stable diagnostics with:

- file path;
- logical area: `project`, `package`, `asset`, `scene`, `node`, `environment`, or `pack`;
- durable ID when available;
- reason;
- exit code `0` for valid, nonzero for invalid or IO failure.

Example format:

```text
error[package.duplicate_asset_id]: apps/editor/sample_project/assets/editor_sample.package.toml: duplicate durable asset id 'editor_sample.model.block'
```

Exact code names may change during implementation, but tests must lock the final shape.

## Identity Contract

Durable identity remains:

- `project_id`;
- `package_id`;
- `asset.id`;
- `scene_id`;
- scene node `id`;
- scene asset reference `asset.id` plus optional `path_hint`;
- manifest-relative `path` as a load location only.

Forbidden durable identity:

- slot/generation runtime handles;
- `MeshHandle`, `TextureHandle`, `MaterialHandle`, `EnvironmentHandle`;
- `SceneNodeId`, `PointLightId`, `LoadTicket`;
- raw paths as sole asset IDs.

## Pack Output

Alpha `pack` output is a copied folder layout:

```text
<out>/
  engine.project.toml
  assets/
  scenes/
  PACK_REPORT.json
```

`PACK_REPORT.json` should include source project, copied files, skipped disabled packages, warnings, and validation status. Do not invent a binary archive.

## Capture Policy

Default Sprint 02 validation is schema/CLI only:

```text
capture_decision: not_required_cli_schema_only
```

If a worker changes runtime scene loading semantics or claims rendered asset placement/readiness, require headless capture validation with deterministic scene/camera setup and evidence paths in the phase report.
