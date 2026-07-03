# Assets: Sync, Deferred, and Handle Semantics

## 1. Purpose & Audience
This chapter documents facade-level asset loading for `AssetManager`, including synchronous loads, deferred loads with `LoadTicket`, and practical handle lifecycle behavior.

## 2. Where This Fits in Engine Flow
Asset flow at API level:
`renderer.assets()` -> sync load (`load_*`) or deferred request (`request_*_load`) -> polling (`poll_*_load`) -> scene integration -> frame rendering.

## 3. Key Concepts
- `AssetManager` is borrowed from `Renderer` (`renderer.assets()`) and scoped to mutable renderer access.
- Sync loading APIs block until upload/prepare is complete:
  - `load_model`, `load_mesh`, `load_texture`, `load_environment`
- Deferred loading APIs return `LoadTicket` immediately:
  - `request_model_load`, `request_texture_load`, `request_texture_load_with_options`
- Deferred completion is observed through `LoadStatus<T>`:
  - `Pending { queued_at }`
  - `Uploaded { value }`
  - `Failed { error }`
  - `Cancelled`
- Deferred progress requires pumping (`Renderer::pump_asset_tasks`) and/or regular render loop activity.
- Handle types (`MeshHandle`, `TextureHandle`, `EnvironmentHandle`) use slot+generation; stale handles are expected after unload/reuse.

### Project, Package, and Durable Asset ID Contract

Status: implemented through the Phase 06 editor asset-browser and prefab/wall-chunk placement slice.

Implemented:

- package manifest structs and TOML loading with `format_version = 1`;
- durable asset records keyed by stable asset ID in `AssetRegistry`;
- duplicate ID, invalid path, unsupported kind, package ID mismatch, and unsupported version failures;
- facade-level package loading, asset listing, record lookup, and durable ID resolution through `AssetManager`;
- deterministic kind/search listing through `AssetManager::list_assets_matching`;
- ID-based model/prefab/wall chunk, texture, and environment load entrypoints that call the existing runtime loaders after resolving metadata;
- editor project/package loading for enabled project packages;
- editor asset browser listing and placement of package records.

Deferred:

- drag-and-drop imports and thumbnail generation;
- complete hot-reload/reimport tooling beyond registry/path invalidation;
- shipping package/export pipeline.

`engine.project.toml` is the project entrypoint. It must live at the project root unless a caller explicitly opens another path. Paths are project-relative unless a field explicitly says otherwise.

Required project manifest fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `format_version` | integer | yes | Must be `1` for the first editor project manifest. Unknown newer versions must fail with a clear unsupported-version error unless migration exists. |
| `project_id` | string | yes | Stable project identity. Human-readable IDs are preferred; UUID-like IDs are valid for generated/imported projects. |
| `name` | string | yes | Display name, not identity. |
| `project_version` | string | recommended | User/project semantic version. Missing value defaults to `"0.1.0"` for display only. |
| `asset_root` | string | yes | Project-relative root used to resolve package manifest and asset paths. |
| `startup_scene` | string or null | yes | Project-relative scene file path or `null`. |
| `default_environment` | object or null | yes | Durable asset reference for default environment or `null`. |
| `packages` | array of object | yes | Package manifest records. Empty array is valid but editor asset browsing will be empty. |
| `editor` | object | yes | Project-level editor defaults that are not authored scene content. |
| `settings` | object | recommended | Runtime/editor defaults such as window size, vsync, and asset manifest mode. Missing values use documented engine defaults. |

Sample `engine.project.toml`:

```toml
format_version = 1
project_id = "project.dungeon_dogfood"
name = "Dungeon Dogfood"
project_version = "0.1.0"
asset_root = "assets"
startup_scene = "scenes/blockout.engine.scene.json"

[default_environment]
id = "core.env.indoor_4k"
path_hint = "sky_maps/indoor_4k.exr"

[[packages]]
package_id = "core"
manifest = "packages/core.package.toml"
enabled = true

[[packages]]
package_id = "dungeon"
manifest = "packages/dungeon.package.toml"
enabled = true

[editor]
recent_scene = "scenes/blockout.engine.scene.json"
grid_spacing = 0.5
snap_rotation_degrees = 90.0

[settings]
window_width = 1920
window_height = 1080
fullscreen = false
vsync = true
asset_manifest_mode = "best_effort"
```

Package manifests are TOML files referenced by `engine.project.toml`. They define durable asset records, not loaded runtime handles.

Required package manifest fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `format_version` | integer | yes | Must be `1` for the first editor package manifest. Unknown newer versions must fail unless migration exists. |
| `package_id` | string | yes | Stable namespace for contained records. Must match the project package record that references this file. |
| `display_name` | string | yes | Editor-facing name. |
| `package_version` | string | recommended | Package semantic/content version. Missing value defaults to `"0.1.0"` for display only. |
| `assets` | array of table records | yes | Asset records keyed by durable `id`. |

Required asset record fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `id` | string | yes | Durable asset ID serialized by scenes. Must be globally unique after package loading. Recommended form is `package.category.name`. |
| `kind` | string | yes | One of the initial known kinds: `model`, `prefab`, `wall_chunk`, `texture`, `material`, `environment`, `scene_fragment`. Unknown kinds fail until explicitly supported. |
| `path` | string | yes | Package-manifest-relative path to the source asset. It is load location, not identity. |
| `display_name` | string | recommended | Editor name. Defaults to `id` for display. |
| `tags` | array of string | recommended | Editor search/filter tags. Defaults to empty. |
| `material` / `materials` | string or array | optional | Durable material IDs applied by default. |
| `metadata` | table | optional | Kind-specific placement/import metadata. |

Sample package manifest:

```toml
format_version = 1
package_id = "core"
display_name = "Core Assets"
package_version = "0.1.0"

[[assets]]
id = "core.wall.stone_2m"
kind = "prefab"
path = "prefabs/wall_stone_2m.glb"
display_name = "Stone Wall 2m"
tags = ["wall", "chunk", "stone"]

[assets.metadata.wall_chunk]
version = 1
grid_size = [2.0, 2.0, 0.25]
connectors = ["north", "south"]
snap_grid = 0.5
snap_rotation_degrees = 90.0

[[assets]]
id = "core.env.indoor_4k"
kind = "environment"
path = "sky_maps/indoor_4k.exr"
display_name = "Indoor 4k"
tags = ["environment", "lighting"]

[[assets]]
id = "core.material.stone"
kind = "material"
path = "materials/stone.material.toml"
display_name = "Stone"
tags = ["material", "stone"]
```

Stable asset ID rules:

- Durable asset IDs must be strings and must be unique across all enabled packages in a project.
- Human-readable IDs are preferred for authored assets. Imported assets may use a UUID-like suffix, for example `imported.model.550e8400-e29b-41d4-a716-446655440000`.
- IDs must be stable across asset moves, file renames, and package-relative path changes.
- Paths are load locations and diagnostics. A `path_hint` may be serialized next to an ID to improve error messages or import migration, but it must not be the only identity.
- The resolver must report duplicate IDs, missing package manifests, unknown package versions, unknown asset kinds, and missing asset paths before the editor presents assets as placeable.
- Runtime handles are resolution outputs. `MeshHandle`, `MaterialHandle`, `TextureHandle`, `EnvironmentHandle`, `SceneNodeId`, `PointLightId`, and `LoadTicket` must never be written as durable project, package, or scene identity.

Current facade APIs:

```rust
let mut assets = renderer.assets();
let records = assets.load_package_manifest("assets/packages/core.package.toml")?;
let all_assets = assets.list_assets();
let wall_assets = assets.list_assets_matching(Some(renderer::AssetKind::WallChunk), Some("wall"));
let crate_record = assets.resolve_asset("core.model.crate")?;
let fragment = assets.load_model_asset("core.model.crate")?;
```

`load_package_manifest` only records CPU-side metadata. It does not upload meshes, textures, environments, or materials. Runtime handles are still produced by `load_model_asset`, `load_texture_asset`, `load_environment_asset`, `request_model_asset_load`, or `request_texture_asset_load`.

Material/settings persistence:

- Package material assets should define reusable defaults by durable material ID.
- Scene `materials` should store overrides keyed by stable override ID and refer back to a durable base material ID when one exists.
- Editor/project settings belong in `engine.project.toml` when they are workspace defaults, and in scene `editor` metadata only when they are authored scene/editor state.
- Renderer-only cache settings and runtime upload state must not be persisted as authored material/settings data.

Negative examples that must fail editor-ready validation:

```toml
format_version = 1
package_id = "bad"

[[assets]]
id = "bad.crate"
kind = "model"
path = "models/crate.glb"
mesh_handle = { slot = 3, generation = 1 }
```

```toml
package_id = "bad"
display_name = "Missing Version"
```

```toml
format_version = 1
package_id = "bad"

[[assets]]
kind = "model"
path = "models/crate.glb"
```

```toml
format_version = 1
package_id = "bad"

[[assets]]
id = "models/crate.glb"
kind = "model"
path = "models/crate.glb"
```

The last example treats the path as identity. It is invalid even though the ID string happens to look like a path.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// Sync model load and mount
let mut assets = renderer.assets();
let fragment = assets.load_model("src/renderer/src/assets/DamagedHelmet.glb")?;
let mount = scene.merge_fragment(None, fragment)?;
let root = mount.mounted_root;
```

Snippet Type: Real
```rust
// Deferred model load ticket polling
let ticket = renderer
    .assets()
    .request_model_load("src/renderer/src/assets/BoomBox.glb")?;

loop {
    let status = renderer.assets().poll_model_load(ticket);
    match status {
        renderer::LoadStatus::Pending { .. } => {
            let _ = renderer.pump_asset_tasks(32)?;
        }
        renderer::LoadStatus::Uploaded { value: fragment } => {
            scene.merge_fragment(None, fragment)?;
            break;
        }
        renderer::LoadStatus::Failed { error } => return Err(error.into()),
        renderer::LoadStatus::Cancelled => break,
    }
}
```

Snippet Type: Real
```rust
// Environment load + scene skybox
let env = renderer.assets().load_environment(EnvironmentSource::Auto(
    "src/renderer/src/assets/sky_maps/indoor_4k.exr".into(),
))?;
scene.set_skybox(env);
```

Snippet Type: Pseudocode
```text
Use two loops/concepts:
  render loop: draws every frame
  load service: polls tickets and mounts ready fragments
The load service can run each tick but should remain logically separate from draw submission code.
```

## 5. Best Practices
- Start with sync loads for first integrations, then adopt deferred tickets for larger content.
- Handle `LoadStatus` exhaustively and log terminal failures.
- Keep loading orchestration and rendering orchestration decoupled for simpler debugging.
- Document current alpha constraints in your app docs (ticket retention limits, cancellation behavior, pending requirements).

## 6. Gotchas & Failure Modes
- Forgetting to pump deferred work can leave tickets stuck in `Pending`.
- Polling a wrong/expired ticket returns `UnknownTicket` in `LoadStatus::Failed`.
- `cancel_load` rejects tickets that are already running or completed.
- Unloading reserved/default resources can fail (`ReservedHandle`).
- Using stale handles after resource lifecycle changes produces stale/invalid errors.

## 7. Debugging Playbook
- Step 1: print ticket IDs and status transitions with timestamps.
- Step 2: confirm that `pump_asset_tasks` is called regularly during deferred workflows.
- Step 3: distinguish `Load`, `Decode`, `Io`, `Sync`, and handle errors in logs.
- Step 4: if content is loaded but not visible, verify fragment merge and scene skybox assignment.
- Step 5: if environment appears inactive, inspect `renderer.environment_runtime_status()` and `assets.environment_state(env)`.

## 8. Cross-Module Links
- Asset facade implementation: `src/renderer/src/api/assets.rs`
- Loading types (`LoadTicket`, `LoadStatus`): `src/renderer/src/api/assets.rs`
- Renderer pump integration: `src/renderer/src/api/renderer.rs`
- Asset internals and cache transitions: `docs/internal/03-asset-lifecycle-and-io.md`

## 9. Standard References
- Rust `Result` and error handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
- Vulkan Guide memory allocation: https://github.khronos.org/Vulkan-Site/guide/latest/memory_allocation.html
- Vulkan Guide synchronization: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/internal/03-asset-lifecycle-and-io.md`
