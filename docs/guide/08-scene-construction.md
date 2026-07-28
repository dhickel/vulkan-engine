# 08 — Scene Construction

> Provenance: `G-08`

This chapter covers building scenes programmatically using the public `Scene` facade. You will learn how to create node hierarchies, set transforms, attach meshes with bounds, configure materials, place lights, set up environment maps, import scene fragments, and control frustum culling.

For the full API reference (every method signature, every validation rule, serialization format), see [Scene Graph & Fragment Workflows](../api/03-scene-graph-and-fragment-workflows.md). The API source lives at [`src/renderer/src/api/scene.rs`](../../src/renderer/src/api/scene.rs).

## The Scene Facade

`Scene` is the public, renderer-owned container for all scene data: nodes, transforms, meshes, material overrides, lights, environment, and camera data. It is not Vulkan-specific — the public API uses `glam` math types and durable string IDs. Vulkan types (handles, buffers, descriptors) are internal to the renderer.

```rust
use engine::prelude::Scene;

let mut scene = Scene::new();
// Or, from the renderer's built-in startup scene:
let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
```

## Nodes and Transforms

> Provenance: `G-08-NODES` — Excerpt

### Creating Nodes

```rust
use engine::prelude::{Scene, SceneNodeId};
use glam::Mat4;

let mut scene = Scene::new();

// Create a root node (no parent, identity transform)
let root = scene.create_node_default(None)?;

// Create a child with a specific transform
let child = scene.create_node(
    Some(root),
    Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
)?;
```

### Modifying Transforms

```rust
// Set absolute local transform
scene.set_transform(child, Mat4::from_rotation_x(0.5))?;

// Read current transform
let transform = scene.transform(child)?;
```

### Reparenting

```rust
// Move a node to a different parent (or to root)
scene.reparent_node(child, Some(new_parent))?;
scene.reparent_node(child, None)?; // Move to root
```

### Node Names and Tags

```rust
scene.set_node_name(child, "Player Mesh")?;
scene.set_node_tags(child, vec!["player".into(), "animated".into()])?;
```

### Removing Nodes

```rust
scene.remove_node(child)?;
// Descendants are removed recursively.
```

### Inspecting Nodes

```rust
// Check validity (generation-aware)
if scene.is_valid_node(child) {
    // ...
}

// Get stable ID (persisted across save/load)
if let Ok(Some(stable_id)) = scene.node_stable_id(child) {
    println!("stable id: {stable_id}");
}

// Find by stable ID
if let Some(node) = scene.find_node_by_stable_id("node.000001") {
    // ...
}

// List all nodes with summaries
for summary in scene.node_summaries() {
    println!(
        "{} parent={:?} children={} meshes={}",
        summary.name, summary.parent, summary.child_count, summary.mesh_count
    );
}
```

## Mesh Attachment with Bounds

> Provenance: `G-08-MESH` — Excerpt

Meshes are attached to nodes via `MeshHandle`. The simplest form adds a mesh without explicit bounds (treated as always-visible for culling):

```rust
use engine::prelude::SceneBounds;
use renderer::prelude::MeshHandle;

// Requires AssetManager to obtain a MeshHandle (see Chapter 09)
// let mesh_handle: MeshHandle = asset_manager.load_mesh("models/crate.obj")?;

scene.add_mesh(node, mesh_handle)?;
```

For proper frustum culling, attach meshes with explicit model-space bounds:

```rust
// Use bounds from the geometry DTO
let bounds = asset_manager.mesh_scene_bounds(mesh_handle)?;
scene.add_mesh_with_bounds(node, mesh_handle, bounds)?;
```

### SceneBounds

| Variant | Meaning | Culling |
|---------|---------|---------|
| `Known(Aabb)` | Authoritative bound from trusted rigid geometry | Eligible for pruning |
| `Proxy(Aabb)` | Explicit stand-in bound when geometry is unavailable | Eligible for pruning |
| `ConservativeVisible(reason)` | Bound is unreliable (skinned, deformed, missing, stale) | Always visible |

`mesh_scene_bounds()` converts the geometry DTO into the appropriate variant:
- Rigid meshes with valid AABBs → `Known(Aabb)`
- Skinned meshes → `ConservativeVisible(Skinned)`
- Deformed meshes → `ConservativeVisible(Deformed)`
- Unknown/missing → `ConservativeVisible(MissingGeometry)`

### Proxy Bounds

When a node has no mesh geometry (or you want to explicitly bound an empty group node), set a proxy:

```rust
use renderer::prelude::Aabb;

let proxy = Aabb::from_min_max(
    glam::Vec3::new(-1.0, 0.0, -1.0),
    glam::Vec3::new(1.0, 2.0, 1.0),
);
scene.set_node_proxy_bounds(node, proxy)?;
scene.clear_node_proxy_bounds(node)?;
```

Proxy bounds cannot override existing known mesh bounds — attempting to do so returns an error.

## Frustum Culling

Frustum culling is **enabled by default**. When enabled, mesh-backed nodes whose transform-aware AABB falls outside the camera frustum are skipped during render submission, reducing GPU draws.

```rust
// Check current state
let culling_on = scene.frustum_culling_enabled();

// Disable (e.g., for tools or skybox-only views)
scene.set_frustum_culling(false);

// Re-enable
scene.set_frustum_culling(true);
```

Nodes with `ConservativeVisible` bounds are never culled, even with frustum culling enabled. Descendants are tested independently — a parent being out of frustum does not automatically cull all children.

## Materials

### Material Overrides

> Provenance: `G-08-MATERIALS` — Excerpt

Scene nodes can override materials per slot. Override IDs reference entries in the scene-level materials map:

```rust
// Set a material override on slot "0" of a node
scene.set_node_material_override(node, "0", "my_custom_material")?;

// Clear the override
scene.clear_node_material_override(node, "0")?;
```

Material override parameters are stored at the scene level:

```rust
use std::collections::BTreeMap;

let mut params = BTreeMap::new();
params.insert("base_color_factor".into(), serde_json::json!([1.0, 0.0, 0.0, 1.0]));
params.insert("metallic_factor".into(), serde_json::json!(0.0));
params.insert("roughness_factor".into(), serde_json::json!(0.5));

scene.set_material_parameters("my_custom_material".to_string(), params);

// Read back
if let Some(params) = scene.material_parameters("my_custom_material") {
    // ...
}
```

## Lights

The scene supports three light types, each with GPU caps enforced at creation time.

### Directional Lights

> Provenance: `G-08-DIR-LIGHT` — Excerpt

```rust
use engine::prelude::{DirectionalLight, DirectionalShadowConfig};

// The API supports multiple directional lights (up to the GPU cap).
let sun = scene.add_directional_light(DirectionalLight {
    direction: glam::Vec3::new(-0.5, -1.0, -0.3).normalize(),
    color: glam::Vec3::new(1.0, 0.95, 0.8),
    intensity: 5.0,
})?;

// Or use create_directional_light (legacy single-light method).
// Prefer add_directional_light for new code.

// Shadow configuration: at most one directional light may cast shadows.
scene.set_directional_shadow_config(
    sun,
    DirectionalShadowConfig { enabled: true },
)?;

// Query shadow caster
if let Some(shadow_id) = scene.shadow_casting_directional_light_id() {
    // ...
}

// Query all directional lights
for light in scene.directional_lights() {
    println!("dir={:?} color={:?}", light.direction, light.color);
}
```

The scene supports at most one shadow-casting directional light. Enabling shadows on a second directional returns `UnsupportedLightFeature`.

### Point Lights

```rust
use engine::prelude::PointLight;

let light = scene.create_point_light(PointLight {
    position: glam::Vec3::new(2.0, 3.0, 0.0),
    color: glam::Vec3::new(1.0, 0.5, 0.2),
    intensity: 50.0,
    range: 10.0,
})?;

// Update later
scene.update_point_light(light, PointLight {
    position: glam::Vec3::new(3.0, 3.0, 0.0),
    color: glam::Vec3::new(0.8, 0.4, 0.1),
    intensity: 30.0,
    range: 8.0,
})?;

// Remove
scene.remove_point_light(light)?;
```

Point-light shadows are not yet supported. Calling `set_point_light_shadow_config(light, true)` returns `UnsupportedLightFeature`.

### Spot Lights

```rust
use engine::prelude::SpotLight;

let spot = scene.create_spot_light(SpotLight::new(
    glam::Vec3::new(0.0, 5.0, 0.0),         // position
    glam::Vec3::new(0.0, -1.0, 0.0),        // direction (toward scene)
    glam::Vec3::new(1.0, 1.0, 1.0),         // color
    20.0,                                     // intensity
    15.0,                                     // range
    0.3,                                      // inner cone angle (radians)
    0.6,                                      // outer cone angle (radians)
))?;

scene.update_spot_light(spot, /* new params */)?;
scene.remove_spot_light(spot)?;
```

Spot-light shadows are not yet supported. Calling `set_spot_light_shadow_config(spot, true)` returns `UnsupportedLightFeature`.

### GPU Light Caps

| Light Type | GPU Cap |
|------------|---------|
| Directional | `MAX_DIRECTIONAL_LIGHTS_GPU` |
| Point | `MAX_POINT_LIGHTS_GPU` |
| Spot | `MAX_SPOT_LIGHTS_GPU` |

Exceeding these caps at creation time returns an error.

### Light Validation

All lights are validated at creation and update:
- Direction must be finite and non-zero
- Intensity must be finite and ≥ 0.0
- Color is clamped to non-negative
- Range (point/spot) must be finite and > 0.0
- Cone angles (spot) must be in [0, π] with inner ≤ outer

## Environment Maps

> Provenance: `G-08-ENV` — Excerpt

```rust
use renderer::prelude::EnvironmentHandle;

// Set skybox from a handle obtained via AssetManager
scene.set_skybox(skybox_handle);

// Check if a skybox is set
if scene.has_skybox() {
    // ...
}

// Set with a durable asset reference for serialization
scene.set_skybox_asset_reference(
    skybox_handle,
    SceneAssetReference::new("env.sky.indoor", Some(PathBuf::from("sky_maps/indoor_4k.exr"))),
)?;
```

Environment maps provide IBL (image-based lighting) for PBR materials. The startup scene includes a default environment. See [Chapter 09](09-asset-pipeline.md) for loading custom environments.

## Scene Fragments

> Provenance: `G-08-FRAGMENTS` — Excerpt

`SceneFragment` is a detached hierarchy that can be mounted into a `Scene`. Fragments are produced by model loading (Chapter 09) and can be constructed programmatically.

### Building a Fragment

```rust
use renderer::prelude::{SceneFragment, SceneFragmentNodeId};

let mut fragment = SceneFragment::new();

// Add nodes (no meshes yet — just hierarchy)
let root = fragment.add_node_default(None)?;
let child_a = fragment.add_node(
    Some(root),
    Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
    vec![],  // no meshes at this node
)?;

// Optionally set a skybox on the fragment
fragment.set_skybox(env_handle);

// Or clear it
fragment.clear_skybox();
```

### Merging a Fragment into a Scene

```rust
let mount = scene.merge_fragment(
    Some(parent_node),  // mount under this node, or None for root
    fragment,
)?;

println!("mounted root: {:?}", mount.mounted_root);
for (fragment_id, scene_id) in &mount.node_mapping {
    println!("fragment node {} -> scene node {:?}", fragment_id.index, scene_id);
}
```

The `SceneFragmentMount` contains the new root node ID and the full mapping from fragment-local IDs to scene node IDs.

## Scene Pick (Ray Cast)

> Provenance: `G-08-PICK` — Excerpt

```rust
// Pick using the last camera matrices set on the scene
if let Some(hit_node) = scene.pick_last_camera(
    mouse_x, mouse_y,
    window_width, window_height,
) {
    println!("hit node: {:?}", hit_node);
}

// Pick with explicit camera matrices
if let Some(hit_node) = scene.pick(
    mouse_x, mouse_y,
    window_width, window_height,
    view_matrix, projection_matrix, camera_position,
) {
    println!("hit node: {:?}", hit_node);
}
```

Picking uses transform-aware editor proxy AABBs. Mesh-backed nodes use a one-unit local proxy; empty group nodes use a smaller origin proxy. This is an editor-oriented convenience, not a precise triangle-level ray cast.

## Scene Serialization

### Save

```rust
scene.save("scenes/my_scene.engine.scene.json")?;
```

The serialized format uses durable asset IDs (not runtime handles), versioned JSON (`format_version: 1`), and editor metadata.

### Load

```rust
// Requires an AssetManager (see Chapter 09)
let scene = Scene::load("scenes/start.engine.scene.json", &mut asset_manager)?;
```

Durable asset IDs in the file are resolved through the provided `AssetManager`. Path hints serve as fallback when IDs are not found in the registry.

## Command System (Undo/Redo)

The scene supports a command pattern for undo/redo:

```rust
use renderer::prelude::CommandHistory;

let mut history = CommandHistory::new();

// Execute a command
scene.execute_command(&mut history, my_command)?;

// Undo
scene.undo_command(&mut history)?;

// Redo
scene.redo_command(&mut history)?;
```

Commands are implemented via the `Command` trait in `src/renderer/src/scene/command.rs`. This is an advanced feature for editor/tool workflows.

## Runnable Verification

Build and check the renderer (scene module):

```sh
cargo check -p renderer
cargo check -p renderer --examples
```

Render the startup scene (WSI required) — exercises the full scene → submission → render path:

```sh
timeout --signal=INT 60s cargo run -p renderer --example demo_pbr
```

The checkpoint app also exercises the complete scene construction + render path:

```sh
cargo check --locked --manifest-path examples/guide_app/Cargo.toml
```

## What Not to Expose

The public `Scene` facade intentionally hides:
- Vulkan handles, buffers, descriptors
- `RenderSubmission` internals
- GPU cache handles (`VkDataCache`)
- Internal `SceneWorld` slot/generation mechanics

Your app should interact with the scene exclusively through the public `Scene` methods and the `AssetManager` for handle creation (Chapter 09).

## Next

Continue to [09 — Asset Pipeline](09-asset-pipeline.md) to learn how to load models, textures, and environments, manage handle lifetimes, and drive async loading.

> For the unified editor object system (`ObjectId`, `SceneObjectId`, `Selection`, `ObjectSummary`, commands), see the [Editor Object System API Reference](../api/18-editor-object-system.md).
