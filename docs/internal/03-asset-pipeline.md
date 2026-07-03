# Asset Pipeline — Disk to GPU

> Source: [`src/renderer/src/api/assets.rs`](../src/renderer/src/api/assets.rs), [`src/renderer/src/data/`](../src/renderer/src/data/), [`src/renderer/src/vulkan/`](../src/renderer/src/vulkan/) — no legacy docs consulted.

## End-to-End Flow: Loading a glTF Model

### 1. API Entry

`AssetManager::load_model(path)` at [`api/assets.rs`](../src/renderer/src/api/assets.rs) → delegates to `AssetManager::load_model_impl()`.

### 2. Assimp Parsing

`assimp_util::load_model()` at [`data/assimp_util.rs:157`](../src/renderer/src/data/assimp_util.rs:157) calls russimp_sys directly (no higher-level crate wrapper).

**Post-processing flags**: `GenSmoothNormals | JoinIdenticalVertices | Triangulate | FlipUVs | FixInfacingNormals | CalcTangentSpace`.

Returns `ModelMeta { scene_world, material_ids, mesh_ids }` where `scene_world` is an Assimp scene graph with node hierarchy.

### 3. Material Processing

`assimp_util::process_materials()` converts Assimp material texture types to engine `MaterialMeta`:

| Assimp Type | Engine Mapping |
|-------------|---------------|
| `BASE_COLOR` | base color texture |
| `METALNESS` | metallic channel |
| `DIFFUSE_ROUGHNESS` | roughness channel |
| `NORMAL_CAMERA` | normal map |
| `AMBIENT_OCCLUSION` | occlusion map |
| `EMISSIVE_COLOR` | emissive map |

Handles combined metallic-roughness textures (metalness in B, roughness in G) and split sources, normalizing to engine convention (R=roughness, G=metalness).

### 4. Texture Loading

For each texture reference in the material:
1. Load image from disk (PNG, JPG, HDR via image crate)
2. Generate mipmap chain
3. Determine format (sRGB for color, linear for data textures)
4. Resolve sampler parameters via the manifest chain: API override → `.meta` file → filename heuristics → defaults (resolution at [`asset_manifest.rs:390`](../src/renderer/src/data/asset_manifest.rs:390))
5. Upload to GPU via staging buffer

### 5. Mesh Data Upload

Vertex and index data from assimp → `VkMeshBuffers`:

```rust
// data_cache.rs
struct VkMeshBuffers {
    vertex_buffer: VkBufferSlice,  // sub-allocated from VkSubAllocator
    index_buffer: VkBufferSlice,
    vertex_count: u32,
    index_count: u32,
}
```

Upload path: CPU staging buffer (`VkHostBuffer`) → `vkCmdCopyBuffer` on transfer queue → `VkFenceQueue` tracks completion.

### 6. Material GPU Data

Materials are stored in a **shader storage buffer object (SSBO)** via `VkSubAllocator`. Each material entry includes PBR factors (base color, metallic, roughness) and texture descriptor set indices. The geometry pass accesses materials via `bufferDeviceAddress`.

### 7. Handle Issuance

Each loaded asset gets a `MeshHandle` / `MaterialHandle` / `TextureHandle` with `slot + generation` from [`data/handles.rs`](../src/renderer/src/data/handles.rs). The slot indexes into the cache; generation invalidates stale references.

Default/reserved slots in `TextureCache`:
- Slot 0: white (base color fallback)
- Slot 1: white (metallic-roughness fallback)
- Slot 2: blue (normal map fallback)
- Slot 3: white (occlusion fallback)
- Slot 4: black (emissive fallback)
- Slot 5: pink (error/missing texture)

### 8. SceneFragment Packaging

The assimp scene hierarchy is converted to `SceneFragment`:
- Each assimp node → `SceneFragmentNode` with local transform, mesh/material references
- Hierarchy preserved as parent-child relationships
- Returned to user for `Scene::merge_fragment()`

## Environment Map Pipeline

Environment maps follow a multi-stage compute pipeline:

1. **Source load** — HDR/EXR image decoded to RGBA32F, NaN/Inf sanitized
2. **Equirect-to-cubemap** — compute shader (`env_equirect_to_cube.frag`) projects the equirectangular image onto 6 cube faces
3. **Irradiance map** — compute shader (`env_irradiance_cube.frag`) computes diffuse IBL via cosine-weighted hemisphere sampling
4. **Prefiltered environment** — compute shader (`env_prefilter_cube.frag`) generates specular IBL at multiple roughness mip levels
5. **BRDF LUT** — compute shader (`gen_brd_flut.frag`) generates a 2D lookup table for the split-sum approximation

Progress is tracked via `EnvironmentState` enum; current stage queryable via `Renderer::environment_runtime_status()`.

## Async Loading (LoadTicket)

`AssetManager::load_model_async()` spawns the assimp parse + GPU upload on a background task. The `LoadTicket` is polled each frame (automatically by `render_scene()` or manually via `poll_ticket()`). Status transitions: `Pending` → `Complete` (or `Failed`).

## Data Caches

### TextureCache

`Vec<CachedTexture>` at [`data_cache.rs:218`](../src/renderer/src/data/data_cache.rs:218). Each entry is `Unloaded(TextureMeta) | Loaded(VkLoadedTexture)`. `load_texture(slot)` transitions Unloaded → Loaded by allocating GPU image, uploading, creating descriptor.

### MeshCache

`Vec<CachedMesh>` at [`data_cache.rs:1454`](../src/renderer/src/data/data_cache.rs:1454). Vertex/index data in `VkSubAllocator` blocks. Buffer device addresses enable bindless vertex pulling in the geometry shader.

## See Also

- [04-vulkan-subsystem.md](04-vulkan-subsystem.md) — transfer queue, staging, memory allocation
- [08-shaders.md](08-shaders.md) — PBR material evaluation
- [../api/04-assets.md](../api/04-assets.md) — public asset API
