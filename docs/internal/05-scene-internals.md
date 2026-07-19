# Scene Internals — Flattening & Submission

> Source: [`src/renderer/src/scene/scene_world.rs`](../src/renderer/src/scene/scene_world.rs), [`src/renderer/src/scene/render_submission.rs`](../src/renderer/src/scene/render_submission.rs) — no legacy docs consulted.

## Scene Data Model

Status note: this document describes the current runtime scene internals and the Phase 03 editor persistence boundary implemented by `Scene::save` / `Scene::load`.

### SceneNode

```rust
// scene_world.rs:35-45
pub struct SceneNode {
    pub stable_id: Option<String>,     // serialized scene document ID
    pub name: String,                  // editor-facing label
    pub parent: Option<SceneNodeId>,
    pub children: Vec<SceneNodeId>,
    pub meshes: Vec<MeshHandle>,       // a node can reference multiple meshes
    pub asset: Option<SceneAssetReference>,
    pub material_overrides: BTreeMap<String, String>,
    pub local_transform: Mat4,
    pub world_transform: Mat4,         // computed during build_submission
    pub dirty: bool,                   // transform changed since last build
    pub layer_mask: u64,              // visibility layers
    pub tags: Vec<String>,
}
```

The scene graph stores nodes in `Vec<SceneNodeEntry>` (slot + generation wrapper around `Option<SceneNode>`) indexed by `SceneNodeId`. Parent-child relationships are stored as indices, not pointers.

### SceneWorld

```rust
// scene_world.rs:87-95
pub struct SceneWorld {
    nodes: Vec<SceneNodeEntry>,        // slot + generation wrapped nodes
    free_slots: Vec<u32>,              // recycled slots
    root: Option<SceneNodeId>,         // single root node
    camera: SceneDataUBO,              // camera uniforms
    skybox_env_id: EnvironmentHandle,  // active environment map
    point_lights: Vec<PointLightEntry>, // slot + generation wrapped point lights
    free_point_light_slots: Vec<u32>,
    directional_lights: Vec<DirectionalLightEntry>, // public facade permits one live entry
    free_directional_light_slots: Vec<u32>,
}
```

## Scene Persistence Boundary

Editor scene JSON is a persistence/import boundary. Runtime scene internals are a render/update boundary. The loader translates between them explicitly:

| Serialized concept | Runtime concept | Required behavior |
|---|---|---|
| Stable scene node string ID | `SceneNodeId` | Allocate runtime nodes during load and store a loader-local map. Never reuse serialized strings as slots, and never persist `SceneNodeId`. |
| Stable light string ID | `PointLightId` | Allocate runtime lights during load and store a loader-local map if editor commands need it. |
| Durable asset ID | `MeshHandle` / `MaterialHandle` / `TextureHandle` / `EnvironmentHandle` | Resolve through project/package asset registry. Runtime handles are outputs only. |
| `path_hint` | import fallback/diagnostic | Used only after a durable asset ID fails to resolve. It must not be the only identity. |
| Scene-local material override ID | runtime material/settings metadata | Preserved as per-node material-slot override IDs. Full material graph editing is deferred. |

Version behavior:

- Missing `format_version` must fail.
- `format_version == 1` is the initial supported scene contract.
- Older supported versions must pass through explicit migrations before runtime node allocation.
- Unknown newer versions must fail clearly unless a migration is explicitly registered.
- Unknown optional `editor` metadata may be preserved or ignored by runtime code, but unknown required runtime fields must fail validation.

Scene load ordering:

1. Parse JSON and validate top-level `format_version`.
2. Validate all stable IDs, parent references, `root_nodes`, asset references, lights, environment, material overrides, and prefab metadata before mutating a live runtime scene.
3. Resolve durable asset IDs through the project/package registry and prepare load requests.
4. Allocate runtime scene nodes and build a stable-ID to `SceneNodeId` map.
5. Restore parent/child relationships deterministically. Flat-only validation is not editor-ready.
6. Resolve materials, environment, and prefab placement metadata.
7. Allocate lights and editor selection/camera metadata that the runtime/editor layer supports.

Phase 03 behavior: `Scene::save` writes stable string node IDs, parent string IDs, durable node asset references, durable environment references, point lights, and supported material override IDs. It does not serialize runtime handles or the old `model_path` / `skybox_path` placeholders. `Scene::load` validates unsupported versions, duplicate node IDs, missing durable asset IDs, bad parents, cycles, and disconnected graphs before building runtime nodes.

Current limit: the runtime still has a single submitted root. Phase 03 load validation rejects multi-root editor documents until the runtime submission path can render multiple roots or a deliberate synthetic-root strategy is designed.

## Editor Commands and Picking

Phase 05 editor mutations use `CommandHistory` around scene commands. `SetTransformCommand` stores the old local transform and treats each inspector edit as a completed undo transaction. `RemoveNodeCommand` snapshots the removed subtree, then removes it through normal slot/generation invalidation. Undo restores equivalent nodes with fresh `SceneNodeId` values and returns a remap, so old handles remain stale and editor selection can move to the restored runtime ID without weakening handle safety.

Picking uses the scene's last renderer-updated camera matrices and current scene transforms. Known meshes use exact eight-corner-transformed world AABBs, explicitly declared proxies remain tagged, and empty/group nodes use a small editor-origin helper. Conservative-visible mesh-bearing nodes are skipped rather than reporting a false proxy hit.

Persistence negative examples:

- Serialized runtime handles: `{"mesh_handle":{"slot":7,"generation":1}}`.
- Flat-only validation: a file with nodes but no deterministic parent/root restoration.
- Unversioned scene files: any JSON scene without `format_version`.
- Asset path as only identity: `{"asset":{"path":"models/crate.glb"}}` with no durable `id`.

## Build Submission: The Hot Path

`SceneWorld::build_submission()` runs every frame. It produces a `RenderSubmission`:

```rust
// render_submission.rs
pub struct RenderSubmission {
    pub render_objects: Vec<RenderObject>,
    pub point_lights: Vec<PointLight>,
    pub directional_light: Option<FrameDirectionalLight>,
    pub environment: Option<EnvironmentHandle>,
    pub camera_position: Vec3,
    pub camera_view: Mat4,
    pub camera_projection: Mat4,
}
```

### Flattening Algorithm

1. For each root node, recursively traverse children in depth-first order
2. For each node, compute `world_transform = parent.world_transform * node.local_transform`
3. If the node has a mesh + material, push a `RenderObject`:

```rust
// gpu_data.rs:307
pub struct RenderObject {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub world_transform: Mat4,
    pub vertex_buffer_address: u64,     // buffer device address
    pub index_buffer_address: u64,
    pub vertex_count: u32,
    pub index_count: u32,
    pub material_data: *const VkLoadedMaterial,  // raw pointer, stable for frame
}
```

4. Collect active point lights with their world positions (from the transform lookup)
5. Package camera matrices
6. Return the submission

### Optional Culling

`SceneWorld` has frustum culling enabled by default, with `Scene::set_frustum_culling(false)` as a diagnostic opt-out. Known/proxy node bounds are tested against the frustum, known subtree unions may prune complete branches, and conservative-visible members prevent pruning. There is still no distance-based LOD or occlusion culling.

### Material Pointer Convention

The `material_data: *const VkLoadedMaterial` field is a **raw pointer** into the `MaterialCache`. This is documented as stable for the current frame's duration — the cache is not mutated during rendering.

## SceneFragment System

### Fragment Structure

A `SceneFragment` is an immutable, self-contained subtree:

```rust
pub struct SceneFragment {
    pub nodes: Vec<SceneFragmentNode>,
    pub root_node_indices: Vec<usize>,
}

pub struct SceneFragmentNode {
    pub name: String,
    pub local_transform: Mat4,
    pub mesh: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub children: Vec<usize>,  // indices into nodes[]
}
```

### Merge Process

`Scene::merge_fragment(parent, fragment)`:
1. Allocates `SceneNodeId`s for each fragment node
2. Copies transforms, mesh/material handles
3. Re-wires children from usize indices to `SceneNodeId`
4. Attaches root nodes under `parent` (or as scene roots if `parent` is `None`)
5. Returns `SceneFragmentMount` with the mapping

### Immutability

Fragments are consumed on merge — you cannot re-merge the same fragment. To instance the same model twice, load it twice (each load produces a fresh fragment).

## Light Systems

Point lights are stored in a parallel flat array with slot+generation handles (`PointLightId`). The `EnvironmentUBO` supports up to 16 point lights in std140 layout (32 bytes each). Exceeding 16 submitted point lights clamps.

The directional-light store uses the same slot+generation lifecycle, while the public facade rejects a second live light. `direction` is surface-to-light. Submission carries one optional `FrameDirectionalLight`; the renderer writes its direction, color/intensity, and fitted light view-projection matrix into `EnvironmentUBO` before PBR draws.

## See Also

- [../api/03-scene.md](../api/03-scene.md) — public scene API
- [02-renderer-internals.md](02-renderer-internals.md) — how submission feeds the rendergraph
- [src/renderer/src/scene/scene_world.rs](../src/renderer/src/scene/scene_world.rs) — implementation
