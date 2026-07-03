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

Sprint 03 validation keeps the canonical sample scene unchanged by writing a saved-scene copy under the sprint artifacts directory:

```text
.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json
```

That copy contains one package-backed model node and one package-backed wall chunk node created through `PlaceAssetCommand`, then saved and reloaded through the scene persistence path.

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

## Validation Commands

Validate the sample project and canonical startup scene with `engine_pack`:

```sh
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

Validate the Sprint 03 saved-scene copy with the same project registry:

```sh
cargo run -p engine_pack -- validate-scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

The persistence tests also assert that saved package-backed nodes do not serialize runtime handles such as slots, generations, mesh handles, or path-only identity.

## Headless Capture Proof

The editor supports a true headless capture path for validation. With `--headless`, the editor returns into the headless path before constructing a window/event loop, creates `Renderer::new_headless`, loads the project package registry and scene, renders with `render_scene_headless`, and exits after requested captures complete.

Use `--capture_target draw` for visual proof so the sidecars identify the offscreen draw target instead of a present-target path:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- \
  --project apps/editor/sample_project/engine.project.toml \
  --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-03-editor-packaged-placement/artifacts/phase-02-saved-scene-copy.engine.scene.json \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw
```

Sprint 03 accepted capture sidecars report:

- `capture_target = "draw"`
- `format = "R16G16B16A16_SFLOAT"`
- `status = "succeeded"`
- `extent = 1440 x 900`

The accepted frame 15 PNG shows the block prop on the left and the wall chunk on the right, matching the saved scene transforms.

## Current Limitations

The alpha editor placement slice does not yet include binary package archives, asset thumbnails, CSG/brush wall editing, polygon editing, material graph editing, PBR material authoring, packaged audio placement, physics/collision authoring, scripting, or a runtime project launcher. Wall chunks are currently package-backed prefab/model placements with searchable metadata and durable scene identity.
