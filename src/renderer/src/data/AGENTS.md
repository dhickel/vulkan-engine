# Data Module Agent Guide (`src/renderer/src/data`)

This is the deep guide for scene data, caches, asset loading, and camera/input coupling.

## Module Map

- `camera.rs`: FPS camera and controller.
- `gpu_data.rs`: GPU-facing structs and draw-context payloads.
- `data_cache.rs`: texture/material/mesh/environment caches and shader/pipeline/descriptor caches.
- `assimp_util.rs`: active model loader path.
- `gltf_util.rs`: currently commented-out legacy code.
- `data_util.rs`: math/image/utility helpers and synchronization primitives.

## Core Data Flow

### Model Ingest (Current Active Path)

1. `assimp_util::load_model(...)` parses model file.
2. Produces `MaterialMeta`, `MeshMeta`, and `SceneWorld` node hierarchy.
3. Materials and meshes are added into caches (`TextureCache`, `MeshCache`) as unloaded entries.
4. Allocation phase uploads/allocates GPU resources and transitions entries to loaded state.

### Frame Render Prep

1. Example/facade frame loop updates camera transforms and writes them into `SceneWorld`.
2. `SceneWorld::build_submission(...)` traverses nodes and emits draw items.
3. `VkRender` resolves submission mesh handles into internal Vulkan draw buckets per frame.

## Important Types

### Scene and Draw

- `SceneWorld`/`SceneNode` (`src/renderer/src/scene/scene_world.rs`)
- stable slot+generation parent/children links (`SceneNodeId`)
- meshes: `MeshHandle` stable handles (slot + generation)
- local/world transforms + dirty flag

- `RenderSubmission::draw_items`
- scene-facing per-frame payload of mesh handles + transforms
- no Vulkan handles or raw pointers in scene boundary types

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

## Defaults and Reserved Handles (Critical)

`TextureCache` reserves:
- texture slots `0..=5` (generation `0`) for default/fallback resources
- material slots `0..=1` (generation `0`) for default/error materials

`MeshCache` reserves:
- mesh slot `0` (generation `0`) for skybox geometry (`SKYBOX_MESH`)

`EnvironmentCache` uses `EnvironmentHandle` with the same slot+generation validation model.

Many systems assume these reserved handles. Preserve this contract unless performing a deliberate migration.

## Current Gotchas and Risks

1. Tombstone semantics are now the stable-handle contract.
- `TextureCache` and `MeshCache` deallocation paths tombstone and bump generation instead of shifting slots.
- Safe deallocation APIs preserve reserved default slots; bypassing this contract requires explicit `unsafe` unchecked methods.

2. Raw material pointer lifetime assumptions.
- `RenderObject.material` is a raw pointer into `TextureCache` storage.
- Safe only if no cache mutation/reallocation occurs during draw consumption.

3. Submission-to-renderer bucketing still requires attention.
- Renderer-side handle resolution currently creates pipeline buckets every frame.
- Geometry pass logic should still avoid assuming all pipeline variants are draw-path compatible.

4. Legacy glTF path is non-operational.
- `gltf_util.rs` is effectively commented-out code and should not be treated as live.

5. Handle validation is now the API boundary.
- Public cache getters now validate generation and return `CacheError` (`InvalidHandle`, `StaleHandle`, `NotLoaded`, `OutOfBounds`).

## Camera/Input Integration Notes

- `FPSController` implements input listener traits.
- Movement uses yaw-only translation (quake-like strafing).
- Pitch is clamped to avoid singularities.
- Mouse delta is consumed each frame via `InputManager::update()`.

## Editing Strategy

When changing this module:
- Preserve stable handle invariants unless you also migrate all handle consumers.
- Avoid mutating texture/material vectors during frame consumption.
- Be explicit about loaded/unloaded transitions and ownership.
- If you replace raw pointers with safer handles, update render hot paths accordingly.

## Suggested Next Hardening Tasks

1. Add optional tombstone slot-reuse/compaction strategy (without changing stable-ID behavior).
2. Add targeted tests for unchecked deallocation APIs and default-slot invariants.
3. Keep geometry pass ordering explicit (opaque/mask/blend) and avoid relying on `HashSet` iteration order.
4. Fix `TextureCache::add_texture` fallback return constant to texture-ID domain.
5. Restore or delete `gltf_util.rs` legacy code path to reduce ambiguity.

## Related Files

- `src/renderer/src/data/gpu_data.rs`
- `src/renderer/src/data/data_cache.rs`
- `src/renderer/src/data/assimp_util.rs`
- `src/renderer/src/data/camera.rs`
- `src/renderer/src/vulkan/vk_render.rs`
