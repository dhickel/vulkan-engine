# Data Module Agent Guide (`src/renderer/src/data`)

This is the deep guide for scene data, caches, asset loading, and camera/input coupling.

## Module Map

- `camera.rs`: FPS camera and controller.
- `gpu_data.rs`: GPU-facing structs, scene graph (`Node`), draw context.
- `data_cache.rs`: texture/material/mesh/environment caches and shader/pipeline/descriptor caches.
- `assimp_util.rs`: active model loader path.
- `gltf_util.rs`: currently commented-out legacy code.
- `data_util.rs`: math/image/utility helpers and synchronization primitives.

## Core Data Flow

### Model Ingest (Current Active Path)

1. `assimp_util::load_model(...)` parses model file.
2. Produces `MaterialMeta`, `MeshMeta`, and root `Node` hierarchy.
3. Materials and meshes are added into caches (`TextureCache`, `MeshCache`) as unloaded entries.
4. Allocation phase uploads/allocates GPU resources and transitions entries to loaded state.

### Frame Render Prep

1. `VkRender::update_scene()` reads camera transforms.
2. Root `Node::draw(...)` traverses scene graph.
3. Traversal emits `RenderObject`s into `DrawContext`, grouped by `VkPipelineType`.

## Important Types

### Scene and Draw

- `Node`
- parent: `Option<Weak<RefCell<Node>>>`
- children: `Vec<Rc<RefCell<Node>>>`
- meshes: cache IDs
- local/world transforms + dirty flag

- `DrawContext`
- `active_pipelines: HashSet<VkPipelineType>`
- `render_objects: [Vec<RenderObject>; VkPipelineType::COUNT]`

- `RenderObject`
- index metadata
- index buffer handle
- vertex buffer device address
- raw pointer to loaded material

### Material and Texture

- `MaterialMeta` + `MaterialValues`: CPU-side material metadata and packed values.
- `TextureMeta`: CPU image bytes + format/mip metadata.
- `VkLoadedMaterial`: descriptor set + metadata sub-allocation + pipeline classification.
- `VkLoadedTexture`: `VkImageAlloc` + sampler.

### Cache Systems

- `TextureCache`
- default textures/materials at fixed indices
- lazy style `CachedTexture`/`CachedMaterial` (unloaded/loaded)
- uploads texture data via host staging + queue submits

- `MeshCache`
- stores `CachedMesh` states
- allocates vertex/index data through `VkSubAllocator`
- includes built-in skybox mesh at ID 0

- `EnvironmentCache`
- stores cubemap skybox state and generated env maps

## Defaults and Reserved IDs (Critical)

`TextureCache` reserves:
- textures `0..=5` for default/fallback resources
- materials `0..=1` for default/error materials

`MeshCache` reserves:
- mesh `0` for skybox geometry (`SKYBOX_MESH`)

Many systems assume these IDs. Preserve this contract unless performing a deliberate migration.

## Current Gotchas and Risks

1. Cache ID invalidation risk on deallocation.
- `deallocate_materials`, `deallocate_textures`, and mesh deallocation paths use `Vec::remove`.
- This shifts later indices and can break external IDs still in use.

2. Incomplete cleanup implementation.
- `impl VkDestroyable for TextureCache` currently `todo!()`.

3. Raw material pointer lifetime assumptions.
- `RenderObject.material` is a raw pointer into `TextureCache` storage.
- Safe only if no cache mutation/reallocation occurs during draw consumption.

4. DrawContext-to-pipeline coupling still requires attention.
- `DrawContext.render_objects` now derives from `VkPipelineType::COUNT`.
- This prevents OOB on enum growth, but geometry pass logic should still avoid assuming all pipeline variants are draw-path compatible.

5. Legacy glTF path is non-operational.
- `gltf_util.rs` is effectively commented-out code and should not be treated as live.

6. Format conversion fallback bug hazard.
- `TextureCache::add_texture` fallback path returns `DEFAULT_ERROR_MAT` on conversion failure; this is a material ID constant used in a texture-ID context.

## Camera/Input Integration Notes

- `FPSController` implements input listener traits.
- Movement uses yaw-only translation (quake-like strafing).
- Pitch is clamped to avoid singularities.
- Mouse delta is consumed each frame via `InputManager::update()`.

## Editing Strategy

When changing this module:
- Preserve cache ID invariants unless you also migrate all ID consumers.
- Avoid mutating texture/material vectors during frame consumption.
- Be explicit about loaded/unloaded transitions and ownership.
- If you replace raw pointers with safer handles, update render hot paths accordingly.

## Suggested Next Hardening Tasks

1. Replace ID-shifting `Vec::remove` cache semantics with tombstone/slot reuse.
2. Implement `TextureCache::destroy` and complete resource cleanup path.
3. Keep geometry pass ordering explicit (opaque/mask/blend) and avoid relying on `HashSet` iteration order.
4. Restore or delete `gltf_util.rs` legacy code path to reduce ambiguity.

## Related Files

- `src/renderer/src/data/gpu_data.rs`
- `src/renderer/src/data/data_cache.rs`
- `src/renderer/src/data/assimp_util.rs`
- `src/renderer/src/data/camera.rs`
- `src/renderer/src/vulkan/vk_render.rs`
