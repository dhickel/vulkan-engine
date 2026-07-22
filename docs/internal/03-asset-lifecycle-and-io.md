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

### Asset Identity Resolution Layer

Status: implemented for the editor-roadmap package identity, editor listing, and prefab/wall-chunk placement slice.

Implemented:

- `AssetRegistry` keeps existing path-to-runtime-handle maps and now also stores package records keyed by durable asset ID.
- Package manifests are TOML with `format_version = 1`, `package_id`, package display/version fields, and `[[assets]]` records.
- Package loading validates duplicate IDs, invalid or escaping paths, unsupported kinds, mismatched package IDs, and unsupported versions before records are visible.
- Package-relative paths are normalized deterministically and resolved against the manifest directory or caller-provided package base.
- `AssetManager` exposes package loading, asset listing, asset record lookup, ID resolution, and ID-based model/prefab/texture/environment load requests.
- The editor opens project manifests, loads enabled package manifests, lists package records, and places model/prefab/wall chunk records through command-backed scene mutation.
- `AssetRegistry::invalidate_path` clears path-handle maps and durable records associated with a changed path. The obsolete standalone `FileWatcher` module was removed; automatic hot-reload orchestration remains deferred.

Deferred:

- project creation, package authoring, and project-manifest migration tooling;
- editor import tooling, thumbnails, and drag-and-drop;
- material asset document loading beyond manifest metadata and existing runtime material APIs;
- complete hot reload/reimport after invalidation;
- shipping package/export behavior.

The editor asset lifecycle has two identity layers:

1. Durable file identity: `project_id`, `package_id`, asset `id`, stable scene node IDs, and material override IDs.
2. Runtime resolution outputs: `MeshHandle`, `MaterialHandle`, `TextureHandle`, `EnvironmentHandle`, `SceneNodeId`, `PointLightId`, and `LoadTicket`.

The asset registry implementation for the editor must resolve durable IDs into runtime handles without making handles durable. The minimum resolution record is:

| Field | Contract |
|---|---|
| `package_id` | Package namespace from the loaded package manifest. |
| `asset_id` | Durable asset ID. This is the scene/package serialized identity. |
| `kind` | Validated asset kind that selects the import/load path. |
| `source_path` | Canonical project/package-relative load path. This is not identity. |
| `path_hint` | Optional serialized diagnostic/migration hint from scene files. |
| `metadata` | Kind-specific authored metadata, including wall chunk v1 placement data. |
| `runtime_state` | In-memory state such as unloaded/loading/loaded and any runtime handles. This must not be serialized as identity. |

Package loading must validate these before assets become visible to the editor:

- project `format_version == 1`;
- package `format_version == 1`;
- package record `package_id` matches the loaded package manifest;
- no duplicate durable asset IDs across enabled packages;
- every record has `id`, `kind`, and `path`;
- every path resolves inside the package/project asset root after normalization;
- unknown asset kinds, unknown newer versions, and missing manifests fail with clear diagnostics;
- older supported versions must migrate through explicit migration code, not best-effort field guessing.

Wall chunk v1 metadata is stored on prefab asset records and copied into scene placement metadata when placed. It includes dimensions/snap/connectors/category for the editor, plus the durable prefab asset ID. It does not store brush planes, editable vertices, or CSG operations.

The existing `.meta` texture sidecar manifest remains a texture import policy input. It is not the package manifest and must not be used as a durable scene asset identity source.

Current implementation note: path-based handle lookup still exists for current runtime and hot-reload needs. Durable package IDs supplement that behavior for editor scene references; they do not replace runtime slot/generation handles inside caches.

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
- `RenderObject` now owns a `CopiedMaterialDrawRecord` captured while the texture-cache lock is held. New draw-time material fields must be copied under that same guard; storing cache-owned addresses would reintroduce the removed lifetime hazard.
- Assuming the legacy `src/renderer/src/data/gltf_util.rs` path is active is incorrect; Assimp path is current.
- **Mesh handle retirement**: `AssetManager::unload_mesh` invalidates the handle immediately (generation bump and geometry lookup removal) but retains the copied `VkMeshBuffers`, GPU suballocations, and neutral geometry DTO until the fence for `max(last_referenced_serial, latest_submitted_serial)` signals. Querying a retired handle returns `StaleHandle`. The cache slot is not reusable until destruction and retirement-queue reaping complete. Double-unload is idempotently stale. Reserved default slots (e.g. skybox mesh at slot 0) are never retired and return `AssetError::ReservedHandle`; generation exhaustion rejects invalidation without moving or destroying the live payload.
- **Material handle retirement**: `AssetManager::unload_material` invalidates the handle immediately (generation bump, `_NULL` tombstone) but retains the SSBO suballocation (`meta_alloc`) and image descriptor set until all referencing GPU frames complete. Texture ownership is NOT cascaded — shared/default textures survive unrelated material unload. Material metadata retirement is tracked in `material_retirement_queue` and reaped from fence observations in `acquire_frame_slot`. Descriptor sets are returned to their pool via `vkFreeDescriptorSets` (pools carry `FREE_DESCRIPTOR_SET`). Reserved default material slots (indices 0-1) are never retired.
- **Texture handle retirement**: `AssetManager::unload_texture` invalidates the handle immediately but retains the Vulkan image/view payload and sampled handle metadata until all referencing GPU frames complete. Texture payloads are tracked in `texture_retirement_queue` and reaped from fence observations. On reap, `VkImageAlloc` destroys the image view before the image/VMA allocation; sampler handles remain owned by `VkSamplerCache` and are destroyed once by that cache. Reserved default texture slots (indices 0-5) are never retired.
- **Reference tracking**: During scene flattening (`build_draw_buckets` in `vk_commands.rs`), every material and its texture references are marked against the prospective submission serial via `mark_material_referenced` / `mark_texture_referenced`. The serial is published only after `queue_submit2` succeeds. Retirement `retire_after` = `max(last_referenced_serial, latest_submitted_serial)`, ensuring cleanup waits for the greater of the last observed frame fence or the latest successful submission.
- **Rollback vs retirement**: Failed uploads and unsubmitted staged resources use the existing immediate-deallocation path (`deallocate_materials` / `deallocate_texture`) since no queue submission owns them. Only successfully submitted work transfers cleanup responsibility to fence observation.

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
