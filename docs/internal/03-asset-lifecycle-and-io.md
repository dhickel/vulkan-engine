# Asset Lifecycle and I/O

## 1. Purpose & Audience
This chapter is for contributors working on model/texture ingest, cache state transitions, deferred loading, and GPU upload handoff behavior.

## 2. Where This Fits in Engine Flow
Asset flow starts at API load requests and ends when frame submission can resolve loaded handles:
request -> decode/import -> cache insert (unloaded) -> GPU transfer -> loaded cache entries -> render-time handle resolution.

## 3. Key Concepts
- Handles are `slot + generation` and must be validated before access.
- Cache states are explicit: `Unloaded`, `Loaded`, or tombstone (`_NULL`) slots.
- CPU ownership (bytes/meta) and GPU ownership (buffers/images/descriptors) are separate phases.
- Deferred loads return `LoadTicket`; completion is observable via polling (`LoadStatus`).
- Template contract reference: `docs/internal/00-index.md` (mandatory 10-section order).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/data/assimp_util.rs
let model = assimp_util::load_model(path, data_cache, has_animation, policy_config)?;
// model.scene_world + model.material_ids + model.mesh_ids
```

Snippet Type: Real
```rust
// src/renderer/src/data/data_cache.rs (handle validation pattern)
fn validate_mesh_slot(&self, handle: MeshHandle) -> Result<usize, CacheError> {
    let slot = handle.slot as usize;
    let Some(generation) = self.mesh_generations.get(slot) else {
        return Err(CacheError::OutOfBounds);
    };
    if *generation != handle.generation {
        return Err(CacheError::StaleHandle);
    }
    Ok(slot)
}
```

Snippet Type: Real
```rust
// src/renderer/src/data/data_cache.rs (sync wrapper)
pub fn allocate_textures(&mut self, texture_ids: Vec<TextureHandle>) -> bool {
    loop {
        if texture_ids.iter().all(|id| self.is_texture_loaded(*id)) {
            return true;
        }
        if self.submit_texture_uploads(&texture_ids).is_err() {
            return false;
        }
        let finalized = self.poll_texture_uploads();
        if finalized == 0 { std::thread::sleep(Duration::from_millis(1)); }
    }
}
```

Lifecycle chart (current alpha path):

| Stage | Ownership | Primary code |
|---|---|---|
| Load request | API thread | `src/renderer/src/api/assets.rs` |
| Disk decode/import | CPU importer | `src/renderer/src/data/assimp_util.rs` |
| Cache insertion (unloaded) | CPU cache | `TextureCache::add_*`, `MeshCache::add_*` |
| Transfer submit | staging + transfer queue | `submit_texture_uploads`, `VkHostBuffer` |
| Transfer completion poll | render thread | `poll_texture_uploads`, `VkFenceQueue::check_fences` |
| Promotion to loaded | CPU metadata + GPU handles | `Cached*::Unloaded -> Cached*::Loaded` |
| Draw consumption | render thread | `SceneWorld::build_submission`, `get_loaded_id` |

Snippet Type: Pseudocode
```text
ticket = request_model_load(path)
while poll(ticket) is Pending:
  pump_asset_tasks()
on Ready(fragment):
  scene.mount_fragment(fragment)
```

## 5. Best Practices
- Keep each ownership handoff explicit in docs and code (CPU bytes -> GPU resource).
- Preserve reserved default slots for textures/materials/meshes unless intentionally migrating.
- Treat unchecked deallocation APIs as dangerous and isolate their use.
- Document blocking behavior separately from deferred behavior so users know current alpha constraints.

## 6. Gotchas & Failure Modes
- `NotLoaded` means a handle is valid but promotion has not completed.
- `StaleHandle` means generation mismatch after slot reuse.
- `OutOfBounds` means invalid slot index.
- Mutating cache storage during render consumption can invalidate raw material pointers in `RenderObject`.
- Assuming the legacy `src/renderer/src/data/gltf_util.rs` path is active is incorrect; Assimp path is current.

## 7. Debugging Playbook
- Step 1: verify ticket state transitions (`Pending` -> `Ready` or `Failed`) in API load tracker.
- Step 2: verify transfer completion by checking whether pending texture batches are finalizing.
- Step 3: if assets remain unloaded, validate handle (`slot`, `generation`) before deeper debugging.
- Step 4: confirm fallback/default resources were not destroyed by unchecked deallocation.
- Step 5: for stuck loads, run bounded smoke (`timeout`) and inspect transfer/fence logs.

## 8. Cross-Module Links
- Import path: `src/renderer/src/data/assimp_util.rs`
- Handle contracts: `src/renderer/src/data/handles.rs`
- Cache state machines: `src/renderer/src/data/data_cache.rs`
- Render consumption: `src/renderer/src/vulkan/vk_render.rs`

## 9. Standard References
- Vulkan memory allocation guide: https://github.khronos.org/Vulkan-Site/guide/latest/memory_allocation.html
- Vulkan synchronization guide: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Assimp docs: https://assimp-docs.readthedocs.io/
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/

## 10. See Also
- `docs/internal/02-synchronization-and-fencing.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `src/renderer/src/data/AGENTS.md`
