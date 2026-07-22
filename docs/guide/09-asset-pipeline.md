# 09 — Asset Pipeline

> Provenance: `G-09`

This chapter covers loading models, textures, environments, and packages through the renderer's asset pipeline. It explains synchronous and deferred loading, load tickets, handle lifetime management, procedural mesh upload, geometry queries, and asset retirement.

For the full API reference (every method signature, every error variant, every handle type), see [Assets: Sync, Deferred & Handles](../api/04-assets-sync-deferred-and-handles.md). The API source lives at [`src/renderer/src/api/assets.rs`](../../src/renderer/src/api/assets.rs).

## Architecture Overview

The asset pipeline has two modes:

| Mode | Blocks? | Returns | Use Case |
|------|:-------:|---------|----------|
| **Synchronous** | Yes (pumps transfers) | `Result<Handle, AssetError>` | Startup loading, editor import |
| **Deferred** | No | `LoadTicket` → poll for result | Background streaming, async prefetch |

Both modes go through the same GPU upload path. Synchronous loads block the calling thread until upload completes; deferred loads return a ticket immediately and complete on background threads.

```
AssetManager (public facade)
    │
    ├── load_model() ─────────► SceneFragment
    ├── load_texture() ───────► TextureHandle
    ├── load_environment() ───► EnvironmentHandle
    ├── upload_procedural_mesh() ──► MeshHandle
    ├── create_material_pbr() ───► MaterialHandle
    │
    ├── request_model_load() ──► LoadTicket ─► poll_model_load() ─► SceneFragment
    ├── request_texture_load() ─► LoadTicket ─► poll_texture_load() ─► TextureHandle
    │
    └── pump_asset_tasks(max_steps) ← called every frame by the app
```

## Getting an AssetManager

> Provenance: `G-09-AM` — Excerpt

The `AssetManager` is obtained from the renderer:

```rust
use renderer::prelude::AssetManager;

let mut asset_manager = renderer.asset_manager();
```

The `AssetManager` borrows the renderer's internal GPU upload infrastructure (`VkRenderCore`) and load tracker. It is short-lived — recreate it each time you need to load assets.

## Handle Types

All runtime asset references use stable handles with **slot + generation** semantics:

| Handle | Identifies | Created By |
|--------|-----------|------------|
| `MeshHandle` | GPU mesh (vertex/index buffers) | `load_mesh`, `upload_procedural_mesh` |
| `TextureHandle` | GPU texture image | `load_texture` |
| `MaterialHandle` | PBR material parameters + texture bindings | `create_material_pbr` |
| `EnvironmentHandle` | Skybox + IBL environment map | `load_environment` |

Handles are copyable, comparable, and can be stored. Generation validation prevents use-after-free: if you hold a stale handle from a retired mesh, the cache will reject it with a generation mismatch error.

### Reserved Default Slots

Low-numbered slots are reserved for engine defaults (skybox mesh, default PBR material, default textures). Attempting to unload these returns `AssetError::ReservedHandle`.

## Loading Models

### Synchronous

> Provenance: `G-09-SYNC-MODEL` — Excerpt

```rust
use renderer::prelude::SceneFragment;

// Load a model file. Returns a SceneFragment ready to merge into a Scene.
let fragment: SceneFragment = asset_manager.load_model("models/crate.obj")?;

// Merge into the scene
let mount = scene.merge_fragment(Some(parent_node), fragment)?;
```

Supported formats: glTF (`.gltf`, `.glb`), Wavefront OBJ (`.obj`), and others via assimp. The returned `SceneFragment` contains the full node hierarchy, meshes, materials, and texture references extracted from the model file.

### Deferred (Async)

```rust
use renderer::prelude::LoadTicket;

// Request background loading
let ticket: LoadTicket = asset_manager.request_model_load("models/large_level.glb")?;

// ... in your frame loop, before rendering:
asset_manager.pump_asset_tasks(32);

// Poll for completion
match asset_manager.poll_model_load(ticket) {
    LoadStatus::Pending { queued_at } => {
        // Still loading; try again next frame.
    }
    LoadStatus::Uploaded { value: fragment } => {
        let mount = scene.merge_fragment(Some(parent_node), fragment)?;
    }
    LoadStatus::Failed { error } => {
        eprintln!("model load failed: {error}");
    }
    LoadStatus::Cancelled => {
        // Load was cancelled before completion.
    }
}
```

Deferred loading uses a bounded thread pool (`max_in_flight_jobs: 4`). Queued tasks are started as slots become available. Completed tasks are collected by `pump_asset_tasks`.

### Cancelling Deferred Loads

```rust
// Only works while the task is still queued (not yet in-flight)
asset_manager.cancel_load(ticket)?;
```

In-flight tasks cannot be cancelled — they must complete (or fail) naturally.

## Loading Textures

### Synchronous

```rust
use renderer::prelude::TextureHandle;

let texture: TextureHandle = asset_manager.load_texture("textures/brick_albedo.png")?;
```

With explicit policy overrides:

```rust
use renderer::prelude::TextureLoadOptions;

let options = TextureLoadOptions {
    srgb: true,
    generate_mipmaps: true,
    ..Default::default()
};
let texture = asset_manager.load_texture_with_options("textures/brick_albedo.png", options)?;
```

Batch loading:

```rust
let handles = asset_manager.load_textures_with_options(vec![
    ("textures/a_albedo.png".into(), TextureLoadOptions::default()),
    ("textures/a_normal.png".into(), TextureLoadOptions { srgb: false, ..Default::default() }),
])?;
```

Supported formats: PNG, JPEG, KTX, KTX2.

### Deferred

```rust
let ticket = asset_manager.request_texture_load("textures/large_terrain.ktx2")?;

// With options:
let ticket = asset_manager.request_texture_load_with_options(
    "textures/large_terrain.ktx2",
    TextureLoadOptions { srgb: false, generate_mipmaps: true, ..Default::default() },
)?;

// Poll:
match asset_manager.poll_texture_load(ticket) {
    LoadStatus::Uploaded { value: handle } => { /* use texture */ }
    // ...
}
```

## Loading Environments

> Provenance: `G-09-ENV` — Excerpt

```rust
use renderer::prelude::EnvironmentSource;

// Auto-detect format from extension (.hdr, .exr)
let env: EnvironmentHandle = asset_manager.load_environment(
    EnvironmentSource::Auto("sky_maps/outdoor_4k.exr")
)?;

// Attach to scene
scene.set_skybox(env);
```

Environment maps are used for image-based lighting (IBL) and skybox rendering. The engine supports HDR (`.hdr`) and OpenEXR (`.exr`) formats.

### Environment State

```rust
use renderer::prelude::EnvironmentState;

match asset_manager.environment_state(env)? {
    EnvironmentState::Unloaded   => { /* not yet loaded */ }
    EnvironmentState::Loading    => { /* skybox loaded, IBL maps still processing */ }
    EnvironmentState::Ready      => { /* fully ready for rendering */ }
    EnvironmentState::Failed(e) => { /* load/processing failed */ }
}
```

The default environment (startup scene) is available at:

```rust
let default_env = asset_manager.default_environment();
```

## Procedural Mesh Upload

> Provenance: `G-09-PROC` — Excerpt

Upload procedurally generated geometry without going through a model file:

```rust
use renderer::prelude::{ProceduralMeshData, ProceduralVertex};

let mesh_data = ProceduralMeshData {
    name: "procedural_quad".to_string(),
    vertices: vec![
        ProceduralVertex {
            position: glam::Vec3::new(-0.5, -0.5, 0.0),
            normal: glam::Vec3::Z,
            uv0: glam::Vec2::new(0.0, 0.0),
            uv1: glam::Vec2::ZERO,
            tangent: glam::Vec4::ZERO,
        },
        // ... three more vertices
    ],
    indices: vec![0, 1, 2, 0, 2, 3],
    material: Some(material_handle), // optional: must be a loaded MaterialHandle
};

let mesh_handle = asset_manager.upload_procedural_mesh(mesh_data)?;
scene.add_mesh_with_bounds(node, mesh_handle, SceneBounds::Known(my_aabb))?;
```

Validation at upload time:
- Non-empty vertices and indices
- Valid triangle indices (in-bounds, non-degenerate)
- Finite position/normal values
- Material handle (if provided) must be valid and loaded

## Creating PBR Materials

```rust
use renderer::prelude::PbrMaterialDesc;

let material = asset_manager.create_material_pbr(PbrMaterialDesc {
    name: "red_plastic".to_string(),
    base_color_factor: [1.0, 0.2, 0.2, 1.0],
    metallic_factor: 0.0,
    roughness_factor: 0.5,
    base_color_texture: None,
    metallic_roughness_texture: None,
    normal_texture: None,
    occlusion_texture: None,
    emissive_texture: None,
    emissive_factor: [0.0, 0.0, 0.0],
    alpha_mode: renderer::prelude::AlphaMode::Opaque,
    alpha_cutoff: 0.5,
    double_sided: false,
})?;
```

Parameter values are clamped to safe ranges (metallic [0,1], roughness [0.02,1], etc.).

## Geometry Queries

> Provenance: `G-09-GEO` — Excerpt

Query the Vulkan-free mesh geometry DTO for loaded meshes:

```rust
use renderer::prelude::MeshGeometryDto;

let dto: MeshGeometryDto = asset_manager.mesh_geometry(mesh_handle)?;
println!(
    "mesh has {} vertices, {} triangles, deformation={:?}",
    dto.positions.len(),
    dto.indices.len(),
    dto.deformation,
);

// Query just the local AABB
if let Some(aabb) = asset_manager.mesh_local_aabb(mesh_handle)? {
    println!("mesh AABB: min={:?} max={:?}", aabb.min, aabb.max);
}

// Get SceneBounds directly
let bounds = asset_manager.mesh_scene_bounds(mesh_handle)?;
match bounds {
    SceneBounds::Known(aabb) => { /* safe for culling */ }
    SceneBounds::ConservativeVisible(reason) => { /* always visible: {reason:?} */ }
    SceneBounds::Proxy(aabb) => { /* explicit proxy bound */ }
}
```

Mesh geometry DTOs contain:
- Model-space positions, normals, UVs
- Triangle indices
- Conservative local AABB (computed from positions, validated for finiteness)
- Deformation classification (`Rigid`, `Skinned`, `Deformed`, `Unknown`)

## Unloading and Retirement

> Provenance: `G-09-UNLOAD` — Excerpt

Assets can be unloaded to reclaim GPU memory:

```rust
// Unload a mesh. The GPU resources are retired after in-flight frames complete.
asset_manager.unload_mesh(mesh_handle)?;

// Unload a material.
asset_manager.unload_material(material_handle)?;

// Unload a texture.
asset_manager.unload_texture(texture_handle)?;
```

Unloading uses a **fence-based retirement** mechanism. Resources are not immediately freed — they are queued for retirement with the latest submitted frame serial. When the GPU signals that all frames referencing the resource have completed, the resource is actually freed.

### Reserved Handles

Attempting to unload a reserved handle (default skybox mesh, default PBR material, default textures) returns `AssetError::ReservedHandle`:

```rust
match asset_manager.unload_mesh(some_handle) {
    Err(AssetError::ReservedHandle { slot, .. }) => {
        eprintln!("slot {slot} is reserved and cannot be unloaded");
    }
    // ...
}
```

## Package Manifests

> Provenance: `G-09-PACKAGE` — Excerpt

Package manifests (`.package.toml`) define durable asset records that map asset IDs to file paths:

```toml
format_version = 1
package_id = "my_package"
display_name = "My Package"
package_version = "0.1.0"

[[assets]]
id = "my_package.model.crate"
kind = "model"
path = "models/crate.obj"

[[assets]]
id = "my_package.tex.brick"
kind = "texture"
path = "textures/brick_albedo.png"
```

Load a package manifest into the asset registry:

```rust
// Load and record all assets from a package
let records = asset_manager.load_package_manifest("assets/my_package.package.toml")?;

for record in &records {
    println!("asset: {} kind={} path={}", record.asset_id, record.kind, record.source_path.display());
}
```

### Loading by Durable Asset ID

Once a package is registered, load assets by durable ID instead of path:

```rust
let fragment = asset_manager.load_model_asset("my_package.model.crate")?;
let texture = asset_manager.load_texture_asset("my_package.tex.brick")?;

// Deferred by ID:
let ticket = asset_manager.request_model_asset_load("my_package.model.crate")?;
```

### Listing and Searching

```rust
// All registered assets
for record in asset_manager.list_assets() {
    println!("{}", record.asset_id);
}

// Filter by kind and search query
for record in asset_manager.list_assets_matching(Some(AssetKind::Model), Some("crate")) {
    println!("{}", record.asset_id);
}

// Look up a specific record
if let Some(record) = asset_manager.asset_record("my_package.model.crate") {
    println!("path: {}", record.source_path.display());
}
```

## Asset Pumping

> Provenance: `G-09-PUMP` — Full Match (checkpoint excerpt)

Every frame, before rendering:

```rust
renderer.pump_asset_tasks(32)?;
```

This call:
1. Pumps GPU transfer submissions (upload completion)
2. Polls texture upload finalization
3. Collects completed background tasks (model/texture loads)
4. Starts queued deferred loads (up to `max_in_flight_jobs`)
5. Cleans up terminal ticket records (capped at 2048)

**Without this call, async loads never complete and GPU resources are never retired.** Call it every frame.

## Asset Policy Configuration

The `AssetPolicyConfig` in `RendererConfig` controls asset discovery and loading behavior. The default configuration uses best-effort manifest mode with filename heuristics enabled and no compression. See [`docs/api/04-assets-sync-deferred-and-handles.md`](../api/04-assets-sync-deferred-and-handles.md) for the full policy reference.

## Runnable Verification

Build and check the renderer (asset module):

```sh
cargo check -p renderer
cargo check -p renderer --examples
```

Run the model-loading demo (exercises sync model load → fragment merge → render):

```sh
timeout --signal=INT 60s cargo run -p renderer --example demo_model_load
```

Run the async loading demo (exercises deferred loading + ticket polling):

```sh
timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading
```

Run the API test with a custom environment (exercises environment loading + IBL):

```sh
timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
```

Working directory must be the repository root for asset paths to resolve.

## Working Directory Requirement

All asset paths are relative to the **current working directory**. The renderer locates shaders, default assets, and startup scene data relative to where the binary is launched. Run all examples and your app from the repository root:

```sh
# Correct:
cd /path/to/vulkan-engine
cargo run -p renderer --example demo_model_load

# Incorrect (asset-not-found errors):
cargo run --manifest-path /path/to/vulkan-engine/src/renderer/Cargo.toml --example demo_model_load
```

## Next

Continue to [10 — Physics](10-physics.md) to learn about the alpha physics crate: rigid bodies, colliders, collision detection, ray casting, and event bridging.
