# Scene Graph and Fragment Workflows

## 1. Purpose & Audience
This chapter explains how facade users build, mutate, and extend scene content with `Scene` and `SceneFragment`. It is aimed at Rust developers new to scene graph workflows in engines.

## 2. Where This Fits in Engine Flow
Scene authoring path:
asset load (`AssetManager`) -> scene creation/mutation (`Scene`) -> optional fragment merge (`Scene::merge_fragment`) -> renderer submission (`Renderer::render_scene*`).

The event crate defines typed scene event contracts, but broad scene mutation emission is deferred. Current scene mutation APIs remain direct facade calls and command-history transactions.

## 3. Key Concepts
- `Scene` is the runtime-owned mutable graph used for rendering.
- `SceneNodeId`, `PointLightId`, and `DirectionalLightId` are slot+generation handles; they can become stale.
- `SceneFragment` is detached content meant for merge/mount workflows.
- `SceneFragmentMount` returns:
  - `mounted_root`: root node in destination scene
  - `node_mapping`: fragment node IDs mapped to scene node IDs
- `Scene::set_skybox(env)` sets scene skybox environment handle.
- Frustum culling is enabled by default. Use `Scene::set_frustum_culling(false)` for diagnostics or compatibility and `Scene::frustum_culling_enabled()` to inspect the current setting.

Culling uses authoritative scene bounds from `SceneBounds`. Authoritative local AABBs are carried into scene nodes via `SceneNode.mesh_bounds`, transformed through eight corners for correct world bounds under rotation, shear, and negative scale, and aggregated into subtree unions post-order. `SceneBounds::Known(Aabb)` / `Proxy(Aabb)` trust the bound for pruning; `SceneBounds::ConservativeVisible(BoundsUnknownReason)` forces the node and subtree to always traverse. Descendants are tested independently when subtree pruning is unavailable. Imported meshes are classified from Assimp bone/animation-mesh metadata before GPU promotion: rigid meshes receive exact CPU-position AABBs, while skinned/deformed meshes remain conservative-visible. Use `Scene::add_mesh_with_bounds` for procedural content; `Scene::add_mesh` (without bounds) marks the mesh `ConservativeVisible(MissingGeometry)`. `AssetManager::mesh_scene_bounds(mesh)` converts a DTO into a `SceneBounds` for callers. Explicit node proxies may replace only empty or `MissingGeometry` nodes; they cannot override known, stale, skinned, deformed, or invalid bounds.

### Editor Scene File Contract

Status: implemented for the Phase 03 editor-roadmap persistence slice. `Scene::save` writes the versioned `.engine.scene.json` shape below, and `Scene::load` validates the document before allocating runtime nodes.

Scene files must use versioned JSON with the recommended extension `.engine.scene.json`. Runtime handles are transient resolution outputs and must not be serialized as durable identity. A loader must build explicit maps from serialized stable IDs to runtime `SceneNodeId`, `PointLightId`, `MeshHandle`, `MaterialHandle`, `TextureHandle`, and `EnvironmentHandle` values.

Asset-backed editor nodes should record durable identity with `Scene::set_node_asset_reference`. Runtime mesh handles alone cannot be reverse-mapped into package asset IDs, so nodes created only with `Scene::add_mesh` serialize as organizational/runtime nodes without an `asset` reference.

Environment persistence follows the same rule. Use `Scene::set_skybox_asset_reference` when the skybox was resolved from a durable package asset; `Scene::set_skybox` only sets the runtime environment handle.

Supported material/settings persistence for this phase is scene-local material override IDs attached to node material slots via `Scene::set_node_material_override`, plus node names and tag lists through the scene facade. The editor inspector currently exposes material slot `0` as an override ID string because that is the durable metadata the renderer can persist without mutating GPU material cache state. PBR factor editing, texture assignment, shader graphs, and material asset documents remain deferred.

Current material override limits:

- `Scene::set_node_material_override` validates that the slot and override ID are non-empty strings, then stores the mapping on the node.
- `Scene::clear_node_material_override` removes one stored slot mapping.
- `SceneNodeSummary::material_overrides` and scene save/load preserve those strings as durable metadata.
- The renderer does not yet resolve these strings into live GPU material mutations. Treat them as editor/package identity that later material tooling can interpret.

Common scene error messages are intentionally beginner-readable. A stale or invalid `SceneNodeId` names the slot and generation so callers can find old handles. Missing durable asset references report the context, for example `missing durable asset id for node asset`. Unsupported scene files report the found and expected format versions. Package/project/asset failures that happen while resolving scene content surface as `RendererError::Asset(...)`; scene graph shape failures surface as `RendererError::Scene(...)`.

Required top-level fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `format_version` | integer | yes | Must be `1` for the first editor scene format. Unknown newer versions must fail with a clear unsupported-version error unless an explicit migration exists. Missing versions must fail validation. |
| `scene_id` | string | yes | Stable document identity, human-readable when authored by the editor and UUID-like when imported/generated. |
| `display_name` | string | recommended | Editor-facing scene name. Missing value defaults to `scene_id` for display only. |
| `root_nodes` | array of string | yes | Stable node IDs for parentless nodes. Flat-only validation is insufficient; nested parent/child relationships must be represented and restored. |
| `nodes` | array of object | yes | Runtime graph content using stable node IDs, not `SceneNodeId` slot/generation values. |
| `lights` | array of object | yes | Light records with stable IDs and authored settings. Empty array is valid. |
| `environment` | object or null | yes | Scene environment reference by durable asset ID, path fallback, or explicit `null`. |
| `materials` | object | recommended | Scene-local material instances and overrides keyed by stable material override ID. Missing object means no scene-authored overrides. |
| `editor` | object | yes | Editor-only metadata such as active camera, grid, visibility, lock state, expanded hierarchy nodes, and selection hints. Runtime code may ignore fields it does not understand. |

Required node fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `id` | string | yes | Stable scene node ID unique within the scene file. It must not be a serialized `SceneNodeId`. |
| `parent` | string or null | yes | Stable parent node ID or `null`. Parent must exist unless `null`. |
| `name` | string | yes | Editor/user-facing node name. Names are not identity. |
| `transform` | object | yes | Authored local transform. Version 1 stores `translation`, `rotation`, and `scale`; loaders may convert to `Mat4` after validation. |
| `asset` | object or null | yes | Durable asset reference for asset-backed nodes. Empty/null means an organizational node. |
| `material_overrides` | object | recommended | Per-mesh/material-slot override references. Missing object means package/default materials. |
| `visibility` | object | recommended | Authoring visibility, layer, and lock metadata. Missing object defaults to visible, unlocked, and runtime default layer. |
| `tags` | array of string | recommended | Editor/game tags. Missing array defaults to empty. |
| `prefab` | object | optional | Placement metadata for prefab-backed nodes, including wall chunk v1 records. |
| `collision` | object | optional | Durable collision metadata for future runtime physics loading. Validation is implemented; editor UI authoring and live physics binding are deferred. |

Asset references inside a scene must use durable IDs first:

```json
{
  "id": "node.wall_north_001",
  "parent": null,
  "name": "North wall",
  "transform": {
    "translation": [0.0, 1.0, -4.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0]
  },
  "asset": {
    "id": "core.wall.stone_2m",
    "path_hint": "prefabs/wall_stone_2m.glb"
  },
  "material_overrides": {
    "0": "mat_override.damp_stone"
  },
  "visibility": {
    "visible": true,
    "locked": false,
    "layer": "world"
  },
  "tags": ["wall", "chunk"],
  "collision": {
    "body": {
      "id": "body.wall_north_001",
      "kind": "static"
    },
    "colliders": [
      {
        "id": "collider.wall_north_001",
        "shape": { "kind": "box", "half_extents": [1.0, 1.0, 0.125] },
        "trigger": false,
        "asset": "core.collision.wall",
        "offset": [0.0, 0.0, 0.0]
      }
    ]
  },
  "prefab": {
    "kind": "wall_chunk",
    "version": 1,
    "package_id": "core",
    "asset_id": "core.wall.stone_2m",
    "grid_size": [2.0, 2.0, 0.25],
    "snap": {
      "grid": 0.5,
      "rotation_degrees": 90.0
    },
    "connectors": ["north", "south"]
  }
}
```

Use `engine_pack validate-scene <scene.engine.scene.json> --project <engine.project.toml>` to validate scene asset references against the enabled project packages before treating a scene file as editor-ready. The CLI path is documented in [Packaging CLI](10-packaging-cli.md).

Wall chunk v1 is prefab asset placement metadata. It identifies a prefab mesh asset, placement size/snap/connectors, and editor categorization. It must not encode editable polygon, CSG, or brush geometry; true polygon/brush editing is deferred beyond the current package-backed prefab placement and persistence slice.

Collision metadata is a durable authoring contract, not a live physics binding yet. Scene validation accepts `collision.body.kind` values `static`, `dynamic`, and `kinematic`; collider shape kinds `box`/`cuboid`, `sphere`, `capsule`, and `capsule_y`; finite offsets; positive shape dimensions; unique durable body/collider IDs; and optional durable collision asset IDs known to the project registry. Validators reject serialized runtime handles, duplicate collision IDs, invalid dimensions, missing colliders, path-only IDs, and unknown collision asset references. The current alpha does not automatically instantiate a `physics::PhysicsWorld` from scene collision metadata.

Sample scene:

```json
{
  "format_version": 1,
  "scene_id": "scene.dungeon_blockout",
  "display_name": "Dungeon Blockout",
  "root_nodes": ["node.room_root"],
  "nodes": [
    {
      "id": "node.room_root",
      "parent": null,
      "name": "Room Root",
      "transform": {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0]
      },
      "asset": null,
      "material_overrides": {},
      "visibility": { "visible": true, "locked": false, "layer": "world" },
      "tags": []
    },
    {
      "id": "node.wall_north_001",
      "parent": "node.room_root",
      "name": "North wall",
      "transform": {
        "translation": [0.0, 1.0, -4.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0]
      },
      "asset": { "id": "core.wall.stone_2m", "path_hint": "prefabs/wall_stone_2m.glb" },
      "material_overrides": { "0": "mat_override.damp_stone" },
      "visibility": { "visible": true, "locked": false, "layer": "world" },
      "tags": ["wall", "chunk"],
      "collision": {
        "body": { "id": "body.wall_north_001", "kind": "static" },
        "colliders": [
          {
            "id": "collider.wall_north_001",
            "shape": { "kind": "box", "half_extents": [1.0, 1.0, 0.125] },
            "trigger": false,
            "asset": "core.collision.wall",
            "offset": [0.0, 0.0, 0.0]
          }
        ]
      },
      "prefab": {
        "kind": "wall_chunk",
        "version": 1,
        "package_id": "core",
        "asset_id": "core.wall.stone_2m",
        "grid_size": [2.0, 2.0, 0.25],
        "snap": { "grid": 0.5, "rotation_degrees": 90.0 },
        "connectors": ["north", "south"]
      }
    }
  ],
  "lights": [
    {
      "id": "light.torch_001",
      "kind": "point",
      "parent": "node.room_root",
      "position": [0.0, 2.0, 0.0],
      "color": [1.0, 0.82, 0.55],
      "intensity": 8.0,
      "range": 12.0
    }
  ],
  "environment": {
    "asset": { "id": "core.env.indoor_4k", "path_hint": "sky_maps/indoor_4k.exr" }
  },
  "materials": {
    "mat_override.damp_stone": {
      "base": "core.material.stone",
      "parameters": { "roughness": 0.92, "metallic": 0.0 }
    }
  },
  "editor": {
    "active_camera": "editor.camera.default",
    "grid": { "enabled": true, "spacing": 0.5 },
    "selection": ["node.wall_north_001"]
  }
}
```

Negative examples that must fail editor-ready validation:

```json
{ "nodes": [{ "id": { "slot": 4, "generation": 2 }, "asset": { "mesh_handle": { "slot": 7, "generation": 1 } } }] }
```

```json
{ "scene_id": "scene.no_version", "nodes": [] }
```

```json
{ "format_version": 1, "scene_id": "scene.flat_only", "nodes": [{ "id": "child", "parent": "missing_parent" }] }
```

```json
{ "format_version": 1, "scene_id": "scene.path_identity", "nodes": [{ "id": "crate", "asset": { "path": "models/crate.glb" } }] }
```

The last example may be useful as an import fallback, but it is invalid as durable editor identity because an asset path alone cannot survive moves, package remaps, or duplicate filenames.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/scene.rs (typical scene-node workflow)
let root = scene.create_node_default(None)?;
let child = scene.create_node_default(Some(root))?;
scene.set_transform(child, glam::Mat4::IDENTITY)?;
scene.add_mesh(child, mesh_handle)?;
```

### Command-backed editor mutations

Editor transform/delete workflows should use `CommandHistory` through the scene facade instead of mutating transforms directly for user-authored transactions. `SetTransformCommand` records the previous local transform, clears redo on a new execute, and supports undo/redo as one completed transaction.

`RemoveNodeCommand` preserves the removed subtree data and keeps old `SceneNodeId` values stale after deletion. Undo restores equivalent nodes with fresh runtime IDs and returns a node remap through `CommandResult`; editor selection should remap by that result and/or by stable scene node ID.

`PlaceAssetCommand` is the editor placement path for model, prefab, and wall chunk assets. It mounts a loaded `SceneFragment`, stamps the mounted root with a durable `SceneAssetReference`, name, tags, and stable scene node ID, and returns `CommandResult::created_node` so the editor can select the placed asset. Undo removes the placed subtree. Redo recreates equivalent scene nodes with fresh runtime IDs while preserving durable asset identity for scene save/load.

Snippet Type: Real
```rust
let mut history = renderer::CommandHistory::new(128);
let node = scene.create_node_default(None)?;

scene.execute_command(
    &mut history,
    Box::new(renderer::SetTransformCommand::new(node, glam::Mat4::IDENTITY)),
)?;

scene.undo_command(&mut history)?;
scene.redo_command(&mut history)?;
```

Snippet Type: Real
```rust
let fragment = renderer.assets().load_model_asset("editor_sample.wall.stone_2m")?;

let result = scene.execute_command(
    &mut history,
    Box::new(renderer::PlaceAssetCommand::new(
        scene.root(),
        glam::Mat4::IDENTITY,
        fragment,
        renderer::SceneAssetReference::new(
            "editor_sample.wall.stone_2m",
            Some("prefabs/wall_straight_2m.obj".into()),
        ),
        "Stone Wall 2m",
        vec!["wall".to_string(), "chunk".to_string()],
        "node.placed.wall.000001",
    )),
)?;
let placed_node = result.created_node.expect("placement created a node");
```

### Picking contract

`Scene::pick_last_camera` casts from screen coordinates using the last camera matrices supplied by `Renderer::render_scene`. Picking takes mutable scene access so it can rebuild world transforms and bounds before the query. Picking uses authoritative world bounds when known (`SceneBounds::Known`) or explicit proxy when set (`SceneBounds::Proxy`). Conservative-visible mesh-bearing nodes skip exact hit to avoid false positives. Empty group nodes still receive a small editor-origin proxy for selection.

Snippet Type: Real
```rust
// Scene fragment assembly + merge
let mut fragment = SceneFragment::new();
let frag_root = fragment.add_node_default(None)?;
let frag_child = fragment.add_node_default(Some(frag_root))?;
fragment.set_root(frag_root)?;

let mount = scene.merge_fragment(None, fragment)?;
let mounted_child = mount.node_mapping[&frag_child];
```

Snippet Type: Real
```rust
// Point-light lifecycle
let light_id = scene.create_point_light(PointLight {
    position: glam::Vec3::new(0.0, 2.0, 0.0),
    color: glam::Vec3::new(1.0, 0.95, 0.9),
    intensity: 8.0,
    range: 15.0,
})?;

scene.update_point_light(light_id, PointLight {
    position: glam::Vec3::new(0.0, 2.5, 0.0),
    color: glam::Vec3::new(1.0, 0.9, 0.8),
    intensity: 10.0,
    range: 18.0,
})?;
```

`Scene::reserve_point_light_storage(total_slots)` is available for transaction coordinators that validate all light payloads before a later publication-only commit. Ordinary apps usually do not need it; use `create_point_light` directly.

A scene may also own one directional light. Its direction points from a shaded surface toward the light (positive Y means the light is above the scene). PBR materials receive direct lighting and a per-frame 2048² D32 shadow map; unlit materials remain lighting-independent. The alpha facade intentionally exposes no shadow-resolution, filtering, cascade, or per-light shadow toggle yet: one live directional light selects the fixed 3×3-PCF shadow path.

Snippet Type: Real
```rust
let sun = scene.create_directional_light(DirectionalLight {
    direction: glam::Vec3::new(-0.4, 1.0, 0.3),
    color: glam::Vec3::new(1.0, 0.95, 0.85),
    intensity: 3.0,
})?;

scene.update_directional_light(sun, DirectionalLight {
    direction: glam::Vec3::new(-0.2, 1.0, 0.5),
    color: glam::Vec3::ONE,
    intensity: 4.0,
})?;
```

Snippet Type: Pseudocode
```text
Keep app-level scene edits in one stage of your frame loop:
  1) apply gameplay/editor mutations
  2) mount any newly completed fragments
  3) render with renderer facade
Avoid mutating and removing the same node handle in unrelated systems without ownership rules.
```

Frustum culling can be controlled explicitly when investigating proxy-bound content:

Snippet Type: Real
```rust
assert!(scene.frustum_culling_enabled()); // default for a new Scene
scene.set_frustum_culling(false);         // diagnostic/content compatibility opt-out
```

## 5. Best Practices
- Validate graph ownership in your app: decide which system can create/remove nodes.
- Keep fragment merge boundaries clean: build fragments detached, then mount once.
- Store returned handle IDs from `SceneFragmentMount` rather than re-querying by index assumptions.
- Treat scene handles as opaque IDs; never encode assumptions about slot values.

## 6. Gotchas & Failure Modes
- Passing an invalid parent to `create_node` returns `SceneError::InvalidParent`.
- Using a node handle after removal can return `SceneError::StaleNode`.
- Merging an empty fragment fails (`MergeFailed`).
- Fragment roots can be ambiguous if not set and multiple parentless nodes exist.
- Invalid point light values (non-finite color, non-positive range, negative intensity) are rejected.
- Directional-light directions must be finite and non-zero; a second live directional light is rejected.
- Non-finite transforms are rejected at mutation time (`SceneError::InvalidMutation`).
- Stale animation targets return `AnimationError::StaleTarget` during playback.
- Command history at zero capacity cannot undo or redo; execute succeeds but stores nothing.

### Canonical Scene Persistence (Phase 08)

`Scene::save` writes a failure-atomic file: serialization and validation happen first, then a staged temporary file is written, flushed, and renamed over the target. Partial writes are never published.

All three light kinds round-trip through the canonical DTO:
- Point lights serialize under `lights` with `kind: "point"`
- Directional lights serialize under `directional_lights` with `kind: "directional"`
- Spot lights serialize under `spot_lights` with `kind: "spot"`

Excluded from serialization (reconstructed at load):
- Runtime handles (`SceneNodeId`, `MeshHandle`, `EnvironmentHandle`)
- World transforms (recomputed from local transform + hierarchy)
- Dirty flags, child indexes, bounds caches, descriptors, GPU payloads

Validation at load time checks:
- Unique stable IDs, parent existence and acyclicity
- Finite transforms, valid asset references
- Finite/ranged lights, valid cone angles for spots
- Collision/audio schema integrity

### Animation Targets (Phase 08)

Animation channels reference scene nodes by durable `SceneNodeId` instead of by volatile index. Clips are validated at acceptance (duration, sampler indices, monotonic timestamps, interpolation cardinality, target-path type consistency).

**Fallible API**: `AnimationPlayer` provides both legacy infallible methods (`set_clip`, `set_speed`, `update`) and new fallible methods (`try_set_clip`, `try_set_speed`, `try_update`). Fallible methods validate inputs before mutating playback state and return typed `AnimationError` variants. Invalid animation inputs are rejected before altering play time, play state, speed, or the lazy-resolved target cache.

Interpolation modes:
- **Step**: previous key value.
- **Linear**: lerp for translation/scale, shortest-path normalized quaternion slerp for rotation.
- **CubicSpline**: glTF Hermite with three output elements per key (in-tangent, value, out-tangent). Tangents are scaled by the key interval. Quaternion output is normalized and finite-checked.

### Directional Light Persistence (Phase 08)

Directional lights with shadow ownership survive scene save/load round-trips. The v2 scene format serializes the shadow-owning directional's configuration under `directional_lights` with `kind: "directional"`. A scene loaded from v2 with a shadow owner restores the identical shadow configuration. The legacy single-light shadow path (v1 format) remains loadable through forward-compatible migration.

### Command History (Phase 08)

Zero-capacity `CommandHistory::new(0)`:
- `execute` succeeds, clears redo, discards the command (stores nothing)
- `undo` returns `CommandError::NothingToUndo`
- `redo` returns `CommandError::NothingToRedo`

`PlaceAssetCommand` clones every authoritative `SceneFragmentNode` field including `mesh_bounds` so geometry metadata survives placement.

## 7. Debugging Playbook
- Step 1: print node IDs (`slot`, `generation`) for failing operations.
- Step 2: if merge fails, inspect fragment root choice and parent links.
- Step 3: verify mutation ordering: remove operations should invalidate downstream users of that handle.
- Step 4: isolate by building a minimal fragment with one root and one child.
- Step 5: if rendering looks wrong, confirm merged fragment nodes actually received meshes/transforms.

## 8. Cross-Module Links
- Scene facade implementation: `src/renderer/src/api/scene.rs`
- Scene node IDs and world internals: `src/renderer/src/scene/scene_world.rs`
- API usage entrypoint: `src/renderer/examples/api_test.rs`
- Internal scene flattening notes: `docs/internal/01-rendering-pipeline-mental-model.md`
- Event lifecycle contract: `docs/api/12-events-and-lifecycle.md`

## 9. Standard References
- glTF node transform model: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#nodes-and-hierarchy
- glam crate docs (`Mat4`, `Vec3`): https://docs.rs/glam/latest/glam/
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/api/12-events-and-lifecycle.md`
- `docs/api/18-editor-object-system.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/20-scene-object-lifecycle.md`
