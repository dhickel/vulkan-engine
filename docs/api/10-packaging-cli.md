# Packaging CLI

## 1. Purpose & Audience

This chapter documents the alpha `engine_pack` Rust CLI used to validate, author, and folder-pack project/package/scene data. It is for engine users and contributors who need a repeatable asset packaging path before editor import and release packaging are complete.

## 2. Where This Fits in Engine Flow

Packaging flow:

`new-project` or existing project -> `new-package` or existing package -> `scan-assets`/`add-asset` -> validation commands -> `pack` folder output -> editor/root runtime consumption.

The CLI is intentionally backed by the renderer crate validators. It should not become a second schema implementation or the project launcher. The root `engine` binary consumes project/package/scene data with `cargo run -- --project <path>`.

## 3. Command Reference

Run commands from the workspace root:

```sh
cargo run -p engine_pack -- <command>
```

Validate a package manifest:

```sh
cargo run -p engine_pack -- validate-package apps/editor/sample_project/assets/editor_sample.package.toml --expected-package-id editor_sample
```

Validate a project and its enabled package/source files:

```sh
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
```

Validate a scene against a project asset registry:

```sh
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

Create a starter project:

```sh
cargo run -p engine_pack -- new-project /tmp/engine-project --id project.example --name "Example Project"
```

Create a starter package manifest:

```sh
cargo run -p engine_pack -- new-package /tmp/engine-project/assets/example.package.toml --id example --name "Example Assets"
```

Scan a directory and print deterministic TOML asset suggestions without mutating files:

```sh
cargo run -p engine_pack -- scan-assets /tmp/engine-project/assets --package-id example
```

Append an asset record to a package manifest:

```sh
cargo run -p engine_pack -- add-asset /tmp/engine-project/assets/example.package.toml --id example.model.crate --kind model --path models/crate.obj --tag model --tag crate
```

Create a validated folder pack:

```sh
cargo run -p engine_pack -- pack apps/editor/sample_project/engine.project.toml --out /tmp/editor-sample-pack
```

The pack command writes a folder tree plus `PACK_REPORT.json`. It does not create a binary archive, thumbnail cache, import database, or editor placement transaction. Runtime launch is handled by the root `engine` binary after project/package/scene validation.

## 4. Durable Identity Rules

The CLI follows the same durable identity rules as the renderer facade and editor scene formats:

- project IDs, package IDs, asset IDs, scene IDs, node IDs, and light IDs are authored identity;
- paths are load locations and diagnostics, not identity;
- runtime handles such as `MeshHandle`, `TextureHandle`, `EnvironmentHandle`, `SceneNodeId`, `PointLightId`, and `LoadTicket` must not appear in project, package, or scene files;
- durable asset IDs should be stable across file moves and package-relative path changes;
- generated scan IDs are deterministic suggestions, not mandatory final names.

`scan-assets` currently recognizes:

| Kind | Extensions |
|---|---|
| `model` | `.gltf`, `.glb`, `.obj` |
| `texture` | `.png`, `.jpg`, `.jpeg`, `.ktx`, `.ktx2` |
| `environment` | `.hdr`, `.exr` |

`scan-assets` does not infer audio, script, collision, or material records. Collision metadata may be authored manually in package manifests and is validated by `validate-package`, `validate-project`, `validate-scene`, and `pack` through the renderer validators.

## 5. Pack Output Shape

Folder packs preserve project-relative paths. A successful pack of the sample editor project contains:

```text
PACK_REPORT.json
engine.project.toml
assets/editor_sample.package.toml
assets/models/block_prop.obj
assets/prefabs/wall_straight_2m.obj
scenes/start.engine.scene.json
```

`PACK_REPORT.json` records:

- source project path;
- copied project-relative files;
- disabled packages skipped by package ID;
- warnings;
- validation status.

The packer rejects absolute paths and parent-traversing paths before copying. Failed repacks remove stale `PACK_REPORT.json` before validation can abort, so an old success report does not survive a failed pack attempt.

## 6. Validation Exit Behavior

Successful commands print stable `valid[...]`, `created[...]`, `added[...]`, `scan[...]`, or `packed[...]` status lines.

Validation failures exit with code `1` and print stable `error[code]` diagnostics.

Usage failures exit with code `2`.

## 7. Deferred Work

The alpha CLI deliberately does not yet provide:

- binary package/archive format;
- asset thumbnails;
- editor drag-and-drop imports;
- editor placement UI;
- package dependency semantics;
- hot-reload or reimport pipeline;
- dynamic Rust hot reload;
- scripting runtime;
- runtime physics scene loading or gameplay collision integration;
- audio integration;
- generated Rust app templates;
- broad dogfood migration to project manifests.

Use the runtime launcher for the current alpha project run path:

```sh
cargo run -- --project apps/editor/sample_project/engine.project.toml
```
