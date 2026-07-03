# Editor Asset Browser and Wall Chunk Placement

## Purpose

This page documents the editor placement and inspector/save-load slice. The editor can load a project package registry, list durable package assets, place model/prefab/wall chunk records into the scene, select nodes, edit supported inspector metadata, transform the selected node, and save/reload the durable authored state through scene persistence.

## Launch

The editor loads `apps/editor/sample_project/engine.project.toml` by default when no `--project` flag is provided and that sample exists. The sample project declares `startup_scene = "scenes/start.engine.scene.json"`, so Save/Load has an active path on default launch. To choose a project explicitly:

```sh
cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml
```

Scene save/load uses the project startup scene path by default or the active `--scene` path when supplied:

```sh
cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene apps/editor/sample_project/scenes/start.engine.scene.json
```

## Sample Package

The built-in sample package is repo-authored and uses simple OBJ fixtures:

- project: `apps/editor/sample_project/engine.project.toml`
- package manifest: `apps/editor/sample_project/assets/editor_sample.package.toml`
- model asset ID: `editor_sample.model.block`
- wall chunk asset ID: `editor_sample.wall.stone_2m`

The wall chunk is discoverable as a package record:

```toml
[[assets]]
id = "editor_sample.wall.stone_2m"
kind = "wall_chunk"
path = "prefabs/wall_straight_2m.obj"
display_name = "Stone Wall 2m"
tags = ["wall", "chunk", "prefab", "sample"]
```

## Browser Behavior

The asset browser is backed by `AssetManager` package records. Records are sorted deterministically by durable asset ID. The UI exposes a text search plus kind filters for all/model/prefab/wall assets. Search matches asset ID, display name, kind, and tags.

Selecting a model, prefab, or wall chunk enables placement. Placement is explicit:

1. Select a placeable asset.
2. Click `Place Selected`.
3. Optionally edit placement translation/rotation/scale.
4. Click `Confirm` to execute the command, or `Cancel` to leave placement mode.

Placement errors are pushed to the editor status area.

## Persistence Contract

Placement uses `PlaceAssetCommand`, not direct scene mutation. The command mounts the loaded `SceneFragment`, stamps the mounted root with `SceneAssetReference`, stable scene node ID, display name, and tags, then returns the created runtime node for selection. Undo removes the placed subtree. Redo recreates equivalent content with fresh runtime IDs but the same durable asset reference.

Saved scene JSON persists durable IDs such as:

```json
{
  "asset": {
    "id": "editor_sample.wall.stone_2m",
    "path_hint": "prefabs/wall_straight_2m.obj"
  },
  "tags": ["wall", "chunk", "prefab", "sample"]
}
```

The durable asset ID is the identity. `path_hint` is diagnostic/fallback data and must not be the only identity.

## Inspector And Save/Load

The inspector shows the selected node name, runtime ID, stable scene node ID, asset reference, numeric local transform, editable tags, and material slot override metadata. Supported editable fields in this phase are:

- `name`
- `tags`
- local transform translation/rotation/scale
- material slot `0` override ID, stored in the node `material_overrides` map

The material field is scene metadata only. It records a stable override ID such as `mat_override.sample_block`; it does not edit PBR factors, textures, shader graphs, material asset documents, or GPU material cache state.

Save and Load are available from the top menu. The active scene path is visible and editable in the scene hierarchy panel. Successful edits mark the session `Unsaved`; successful save/load marks it `Saved`. Failed save/load attempts push visible status messages and do not clear the dirty flag. Loading a scene clears current selection and resets undo history so stale runtime node handles cannot affect the new scene.

## Wall Chunk V1

Wall chunk v1 is prefab placement metadata only. It does not implement CSG, brush editing, or polygon editing. Metadata such as grid size, connectors, and snap values lives in the package manifest so wall chunks remain searchable package assets instead of hardcoded file paths.
