# 13 — Packaging & Distribution

> Provenance: `G-13`

This chapter covers the `engine_pack` CLI tool: scaffolding new apps, projects, and packages; validating manifests, scenes, and asset references; scanning directories for recognized assets; adding assets to manifests; and producing validated packed output for distribution.

For the API reference, see [Packaging CLI](../api/10-packaging-cli.md). The tool source lives at [`tools/engine_pack/src/main.rs`](../../tools/engine_pack/src/main.rs).

## Architecture

`engine_pack` is a standalone CLI binary in the engine workspace. It uses the renderer crate's validators (project, package, scene) for schema enforcement but does not start a GPU runtime, open a window, or depend on Vulkan. All commands are pure CPU operations — filesystem reads/writes, JSON/TOML parsing, schema validation, and directory scanning.

```
engine_pack
├── Scaffolding
│   ├── new-app          Standalone Rust app crate
│   ├── new-project      Project directory + starter scene
│   └── new-package      Package manifest (TOML)
│
├── Validation
│   ├── validate-package Package manifest + source file checks
│   ├── validate-project Project manifest + scene + asset cross-refs
│   └── validate-scene   Scene JSON + asset ID resolution
│
├── Asset Discovery
│   ├── scan-assets      Directory scan → TOML suggestions
│   └── add-asset        Append asset record to manifest
│
└── Distribution
    └── pack              Validated folder pack → PACK_REPORT.json
```

All commands run from the workspace root:

```sh
cargo run -p engine_pack -- <command>
```

## Scaffolding

### new-app — Renderer-Free App Scaffold

> Provenance: `G-13-NEW-APP` — Excerpt

```sh
cargo run -p engine_pack -- new-app /tmp/my_engine_app --id my_app --name "My Engine App"
```

Creates a standalone Rust application that depends on the engine's **public support crates only** (`engine_events`, `input`, `physics`). It does **not** depend on the renderer, does **not** mutate the root workspace `Cargo.toml`, and does **not** implement dynamic Rust reload or plugin ABI loading.

Output structure:

```
/tmp/my_engine_app/
├── Cargo.toml      # Standalone crate (not a workspace member)
├── README.md
└── src/
    └── main.rs     # Init event bus + physics world + input snapshot
```

The generated `Cargo.toml` references engine crates by absolute path:

```toml
[package]
name = "my_app"
version = "0.1.0"
edition = "2021"

[dependencies]
engine_events = { path = "/path/to/vulkan-engine/src/events" }
input = { path = "/path/to/vulkan-engine/src/input" }
physics = { path = "/path/to/vulkan-engine/src/physics" }
```

Verify the scaffold compiles and runs:

```sh
cd /tmp/my_engine_app
cargo check
cargo run
```

Expected output: a line showing the app name, pending event count, and physics world initialized with default gravity.

**This scaffold is deliberately renderer-free.** It validates your toolchain and the support crates without requiring a GPU or Vulkan SDK. When you are ready to add a window and rendering, add the `engine` crate dependency and follow the checkpoint pattern from [Chapter 04](04-app-owned-loop.md).

### new-project — Project Directory

```sh
cargo run -p engine_pack -- new-project /tmp/engine-project --id project.example --name "Example Project"
```

Creates:

```
/tmp/engine-project/
├── engine.project.toml          # Project manifest
├── assets/                       # Empty asset directory
└── scenes/
    └── start.engine.scene.json  # Minimal valid starter scene
```

The generated `engine.project.toml`:

```toml
format_version = 1
project_id = "project.example"
name = "Example Project"
project_version = "0.1.0"
asset_root = "assets"
startup_scene = "scenes/start.engine.scene.json"
packages = []

[settings]
window_width = 1280
window_height = 720
fullscreen = false
vsync = true
```

The command also runs `validate-project` on the result, confirming the starter scene and project manifest are valid immediately after creation.

### new-package — Package Manifest

```sh
cargo run -p engine_pack -- new-package /tmp/engine-project/assets/example.package.toml \
  --id example --name "Example Assets"
```

Creates a minimal package manifest:

```toml
format_version = 1
package_id = "example"
display_name = "Example Assets"
package_version = "0.1.0"
```

The command validates the result with `--expected-package-id` to ensure the generated package ID matches the requested ID.

## Validation

### validate-package — Package Manifest Validation

> Provenance: `G-13-VAL-PKG` — Excerpt

```sh
# Create a temp package and validate:
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-package "$TMPDIR/package.toml" --id pkg --name "Pkg"
cargo run -p engine_pack -- validate-package "$TMPDIR/package.toml" --expected-package-id pkg

# With source file checks (verifies every asset path exists on disk)
cargo run -p engine_pack -- validate-package "$TMPDIR/package.toml" --expected-package-id pkg
rm -rf "$TMPDIR"
```

The `--expected-package-id` flag asserts that the manifest ID matches. The `--project-root` flag resolves relative asset paths for source file existence checks.

Output on success:

```
valid[package]: /tmp/.../package.toml (0 assets)
```

### validate-project — Project Validation

```sh
# Validate a project with a temp scaffold (dogfood uses a custom manifest format):
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-project "$TMPDIR/project" --id test.valid --name "Valid Test"
cargo run -p engine_pack -- validate-project "$TMPDIR/project/engine.project.toml"
rm -rf "$TMPDIR"
```

This validates:
1. The project manifest schema (`format_version`, `project_id`, `startup_scene`)
2. Referenced package manifests exist and are valid
3. The startup scene file exists and is valid JSON
4. Scene asset references resolve against the project's enabled packages
5. All asset source files exist on disk

Output on success:

```
valid[project]: /tmp/.../project/engine.project.toml (test.valid)
```

### validate-scene — Scene Validation

```sh
cargo run -p engine_pack -- validate-scene "$TMPDIR/scenes/start.engine.scene.json" \
  --project "$TMPDIR/engine.project.toml"
```

Where `$TMPDIR` is a project directory created by `new-project`. This validates:
1. The scene JSON schema (`format_version: 1`)
2. Node hierarchy integrity (no orphan children, no cycles)
3. Asset references resolve against the project's known asset IDs
4. Light definitions are valid

Output on success:

```
valid[scene]: /tmp/.../scenes/start.engine.scene.json
```

## Asset Discovery

### scan-assets — Directory Scan

> Provenance: `G-13-SCAN` — Excerpt

Scan a directory for recognized asset files and output TOML asset record suggestions:

```sh
cargo run -p engine_pack -- scan-assets /tmp/engine-project/assets --package-id example
```

Output (deterministic, sorted by path):

```toml
[[assets]]
id = "example.model.crate"
kind = "model"
path = "models/crate.obj"
display_name = "crate"

[[assets]]
id = "example.texture.brick_albedo"
kind = "texture"
path = "textures/brick_albedo.png"
display_name = "brick albedo"
```

Recognized extensions:

| Kind | Extensions |
|------|------------|
| `model` | `.gltf`, `.glb`, `.obj` |
| `texture` | `.png`, `.jpg`, `.jpeg`, `.ktx`, `.ktx2` |
| `environment` | `.hdr`, `.exr` |
| `audio` | `.wav`, `.ogg`, `.flac`, `.mp3` |

Generated asset IDs are deterministic suggestions, not mandatory final names. Edit them before adding to a package manifest.

### add-asset — Append to Manifest

```sh
cargo run -p engine_pack -- add-asset /tmp/engine-project/assets/example.package.toml \
  --id example.model.crate --kind model --path models/crate.obj --tag model --tag crate
```

Appends an `[[assets]]` record to the package manifest and re-validates the result. The `--tag` flag is repeatable. Asset paths are validated for relative-path safety (no `..`, no absolute paths).

Output on success:

```
added[asset]: example.model.crate -> /tmp/engine-project/assets/example.package.toml
```

## pack — Validated Folder Pack

> Provenance: `G-13-PACK` — Excerpt

```sh
cargo run -p engine_pack -- pack /tmp/my_project/engine.project.toml --out /tmp/my_project_packed
```

The `pack` command:
1. Validates the project (manifest + startup scene + all enabled packages)
2. Copies the project manifest, startup scene, and all package manifests into the output directory
3. Copies all referenced asset files (preserving directory structure)
4. Writes `PACK_REPORT.json` with a full manifest of copied files
5. Skips disabled packages (recorded in the report)

Output structure (example with a temp project):

```
/tmp/my_project_packed/
├── PACK_REPORT.json           # Pack manifest
├── engine.project.toml        # Project manifest
├── scenes/
│   └── start.engine.scene.json
└── assets/
    └── core.package.toml
```

### PACK_REPORT.json

```json
{
  "source_project": "/tmp/my_project/engine.project.toml",
  "copied_files": [
    "engine.project.toml",
    "scenes/start.engine.scene.json",
    "assets/core.package.toml"
  ],
  "skipped_disabled_packages": [],
  "warnings": [],
  "validation_status": "passed"
}
```

### What pack Does NOT Do

- Binary archive creation (no `.zip`, `.tar`, or custom archive format)
- Thumbnail cache generation
- Import database creation
- Editor placement transactions
- Runtime hot-reload manifest

The `pack` command produces a folder tree ready for distribution alongside the runtime binary. Runtime launch is handled by the root `engine` binary:

```sh
cargo run -- --project /tmp/dogfood-pack/engine.project.toml
```

## Durable Identity Rules

All `engine_pack` commands follow the same identity rules as the renderer facade and scene format:

| What | Rule |
|------|------|
| Project/package/asset/scene/node IDs | Authored identity — stable across file moves |
| File paths | Load locations and diagnostics, not identity |
| Runtime handles (`MeshHandle`, `TextureHandle`, `LoadTicket`) | Must not appear in manifests or scene files |
| Generated scan IDs | Deterministic suggestions, not mandatory |
| Asset ID format | Alphanumeric + `.`, `_`, `-` preferred |

## Complete Workflow

A typical project onboarding sequence:

```sh
# 1. Scaffold
cargo run -p engine_pack -- new-project /tmp/my_project --id project.my_game --name "My Game"
cargo run -p engine_pack -- new-package /tmp/my_project/assets/core.package.toml --id core --name "Core Assets"
cargo run -p engine_pack -- new-app /tmp/my_game_app --id my_game --name "My Game App"

# 2. Add assets to the project directory
cp /some/models/*.glb /tmp/my_project/assets/models/
cp /some/textures/*.png /tmp/my_project/assets/textures/

# 3. Discover and register assets
cargo run -p engine_pack -- scan-assets /tmp/my_project/assets --package-id core
# (copy the output into /tmp/my_project/assets/core.package.toml manually,
#  or use add-asset repeatedly)

# 4. Validate everything
cargo run -p engine_pack -- validate-package /tmp/my_project/assets/core.package.toml --expected-package-id core
cargo run -p engine_pack -- validate-project /tmp/my_project/engine.project.toml

# 5. Pack for distribution
cargo run -p engine_pack -- pack /tmp/my_project/engine.project.toml --out /tmp/my_project_packed

# 6. Verify the scaffold app compiles independently
cd /tmp/my_game_app && cargo check
```

## Runnable Verification

### Scaffolding (temp directories, no GPU)

```sh
# Create and validate a project in /tmp
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-project "$TMPDIR/project" --id test.pack --name "Pack Test"
cargo run -p engine_pack -- new-package "$TMPDIR/project/assets/core.package.toml" --id core --name "Core"
cargo run -p engine_pack -- new-app "$TMPDIR/app" --id test.app --name "Test App"

# Validate the scaffold
cargo run -p engine_pack -- validate-package "$TMPDIR/project/assets/core.package.toml" --expected-package-id core
cargo run -p engine_pack -- validate-project "$TMPDIR/project/engine.project.toml"
cd "$TMPDIR/app" && cargo check
```

Expected: all commands succeed.

### Validating a Project (temp directory, no GPU)

```sh
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-project "$TMPDIR/project" --id test.valid --name "Valid Test"
cargo run -p engine_pack -- new-package "$TMPDIR/project/assets/core.package.toml" --id core --name "Core"
cargo run -p engine_pack -- validate-package "$TMPDIR/project/assets/core.package.toml" --expected-package-id core
cargo run -p engine_pack -- validate-project "$TMPDIR/project/engine.project.toml"
cargo run -p engine_pack -- validate-scene "$TMPDIR/project/scenes/start.engine.scene.json" \
  --project "$TMPDIR/project/engine.project.toml"
rm -rf "$TMPDIR"
```

### Build Check

```sh
cargo check -p engine_pack
cargo test -p engine_pack
```

### Pack (write to temp)

```sh
OUT=$(mktemp -d)
TMPDIR=$(mktemp -d)
cargo run -p engine_pack -- new-project "$TMPDIR/project" --id test.pack --name "Pack Test"
cargo run -p engine_pack -- pack "$TMPDIR/project/engine.project.toml" --out "$OUT"
cat "$OUT/PACK_REPORT.json"
ls -R "$OUT"
rm -rf "$OUT" "$TMPDIR"
```

## Output Reference

| Command | Output Pattern | Exit 0 Means |
|---------|---------------|--------------|
| `new-app` | `created[app]: <path>` | App scaffold written and directories created |
| `new-project` | `created[project]: <path>` | Project + starter scene written and validated |
| `new-package` | `created[package]: <path>` | Package manifest written and validated |
| `validate-package` | `valid[package]: <path> (N assets)` | Schema valid, source files (if checked) exist |
| `validate-project` | `valid[project]: <path> (id)` | Schema valid, packages+scene resolve |
| `validate-scene` | `valid[scene]: <path>` | Schema valid, asset IDs resolve in project |
| `scan-assets` | TOML `[[assets]]` blocks | Valid directory, recognized files found |
| `add-asset` | `added[asset]: <id> -> <path>` | Record appended and manifest re-validated |
| `pack` | `packed[project]: <src> -> <out> (N files)` | All validation passed, files copied |

## Next

Case study, compatibility, and troubleshooting chapters are planned as later additions to the guide.
