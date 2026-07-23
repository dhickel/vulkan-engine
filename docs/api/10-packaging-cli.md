# Packaging CLI

## 1. Purpose & Audience

This chapter documents the alpha `engine_pack` Rust CLI used to validate, author, and folder-pack project/package/scene data. It is for engine users and contributors who need a repeatable asset packaging path before editor import and release packaging are complete.

## 2. Where This Fits in Engine Flow

Packaging flow:

`new-project` or existing project -> `new-package` or existing package -> `scan-assets`/`add-asset` -> validation commands -> `pack` folder output -> editor/root runtime consumption.

App-template flow:

`new-app` -> standalone Rust support-crate scaffold -> `cargo check --manifest-path <app>/Cargo.toml` -> app-owned iteration.

The CLI is intentionally backed by the renderer crate validators for project/package/scene data. It should not become a second schema implementation or the project launcher. The root `engine` binary consumes project/package/scene data with `cargo run -- --project <path>`.

## 3. Command Reference

Run commands from the workspace root:

```sh
cargo run -p engine_pack -- <command>
```

Validate a package manifest:

```sh
# Dogfood dungeon package (10 assets: models, textures, environment, audio)
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
```

Validate a project and its enabled package/source files:

```sh
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
```

Validate a scene against a project asset registry:

```sh
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

Create a starter project:

```sh
cargo run -p engine_pack -- new-project /tmp/engine-project --id project.example --name "Example Project"
```

Create a standalone Rust app support-crate scaffold without mutating the root workspace:

```sh
cargo run -p engine_pack -- new-app /tmp/engine-app --id app.example --name "Example App"
cargo check --manifest-path /tmp/engine-app/Cargo.toml
```

The generated app depends on public support crates by absolute path: `engine_events`, `input`, and `physics`. It is compile-first and deliberately avoids renderer internals, root `Cargo.toml` edits, dynamic Rust reload, plugin ABI loading, and runtime hot reload. Renderer window-loop integration still belongs in app crates and renderer examples; this template is the minimal off-workspace app-control starting point.

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
cargo run -p engine_pack -- pack apps/dungeon_dogfood/engine.project.toml --out /tmp/dogfood-pack
```

The pack command writes a folder tree plus `PACK_REPORT.json`. It does not create a binary archive, thumbnail cache, import database, or editor placement transaction. Runtime launch is handled by the root `engine` binary after project/package/scene validation.

## 4. Fail-Closed Transaction Semantics (Phase 10)

All mutating `engine_pack` commands (`new-app`, `new-project`, `new-package`, `add-asset`, `pack`) now use transactional staging:

- **Preflight**: All inputs are validated before any output is written.
- **Staging**: Content is built in a sibling staging directory (`.<name>.staging.<pid>.<n>`).
- **Publication**: The staging directory is atomically renamed to the target path. If the target already exists, the operation fails with an error (no silent overwrite).
- **Rollback**: `add-asset` uses a backup/rename sequence for manifest mutation. On validation failure after write, the original manifest is restored from backup. If rollback itself fails, the error includes recovery paths.
- **Cleanup**: On any failure before publication, the staging directory is removed.

`pack` refuses to overwrite an existing output directory. `new-app` and `new-project` refuse existing target directories.

`scan-assets` uses:
- `symlink_metadata` to detect and silently skip symlinks (never follows them)
- Per-entry canonicalization with root containment checks
- Visited-set tracking to reject filesystem cycles

## 5. Durable Identity Rules

The CLI follows the same durable identity rules as the renderer facade and scene formats:

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
| `audio` | `.wav`, `.ogg`, `.flac`, `.mp3` |

Path validation rejects non-UTF-8 file names at the scan boundary. All asset
registry output (TOML manifests, scan suggestions, pack reports) is generated
through serializer-backed paths that cannot emit invalid UTF-8 or escape
sequences. The former ad-hoc `sanitize_id_component` and `toml_escape` helpers
have been removed; identity components that are not valid UTF-8 are rejected
rather than silently mangled.

`scan-assets` infers basic audio asset records by extension. Audio metadata such as declared format, usage, and default gain may be authored manually under `[assets.metadata.audio]` and is validated by `validate-package`, `validate-project`, `validate-scene`, and `pack` through the renderer validators. Collision and material records are still manually authored; collision metadata is also validated through the same renderer-backed path. Package-level script asset records are deferred in this alpha and `script` is not currently accepted as an asset kind.

## 6. Pack Output Shape

Folder packs preserve project-relative paths. A successful pack of the dogfood project contains:

```text
PACK_REPORT.json
engine.project.toml
assets/dogfood_dungeon.package.toml
assets/models/crate_a.obj
assets/models/torch_sconce.obj
scenes/start.engine.scene.json
```

`PACK_REPORT.json` records:

- source project path;
- copied project-relative files;
- disabled packages skipped by package ID;
- warnings;
- validation status.

The packer rejects absolute paths and parent-traversing paths before copying. Failed repacks remove stale `PACK_REPORT.json` before validation can abort, so an old success report does not survive a failed pack attempt.

## 7. Validation Exit Behavior

Successful commands print stable `valid[...]`, `created[...]`, `added[...]`, `scan[...]`, or `packed[...]` status lines.

Validation failures exit with code `1` and print stable `error[code]` diagnostics.

Usage failures exit with code `2`.

## 8. Deferred Work

The alpha CLI deliberately does not yet provide:

- binary package/archive format;
- asset thumbnails;
- editor drag-and-drop imports;
- editor placement UI;
- package dependency semantics;
- hot-reload or reimport pipeline;
- dynamic Rust hot reload;
- package-level script asset scanning/validation;
- production scripting runtime;
- runtime physics scene loading or gameplay collision integration;
- production audio mixing/spatialization/streaming or editor audio placement;
- renderer-window generated app templates;
- broad dogfood migration to project manifests.

Use the runtime launcher for the current alpha project run path:

```sh
cargo run -- --project apps/dungeon_dogfood/engine.project.toml
```
