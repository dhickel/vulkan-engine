# API Review & Documentation Plan - February 12, 2026

## 1. Executive Summary
The engine is nearing an Alpha state with a functional Vulkan renderer and a simplified concurrency model. However, several "Alpha-blocking" issues remain, specifically around synchronization stalls and API safety.

## 2. Identified Bugs & Issues

### 2.1 Synchronous Environment Hitch
- **Location:** `VkRenderCore::ensure_environment_ready`
- **Issue:** This function is called during the frame loop when an environment switch is detected. It triggers `upload_skybox` and `generate_environment`, both of which eventually perform `device_wait_idle()` or block on fences.
- **Impact:** Switching a skybox causes a 50ms-200ms frame spike.
- **Recommendation:** Implement the "Staged Promotion" or "Async Asset" pattern where environments are prepared in the background, and the renderer only switches when the new environment's GPU resources (cubemaps, descriptors) are fully ready.

### 2.2 Scene Node Handle Fragility
- **Location:** `SceneWorld`, `SceneNodeId`
- **Issue:** Nodes are identified by raw `u32` indices. Deleting a node in the middle of the `nodes` vector will shift indices or leave "holes," leading to stale references or crashes.
- **Impact:** Difficult to manage dynamic scenes (e.g., spawning/despawning enemies).
- **Recommendation:** Implement a `SceneNodeHandle` using a **SlotMap** (Index + Generation) to ensure that stale handles can be detected and ignored safely.

### 2.3 Blocking Transfer "Await"
- **Location:** `TextureCache::allocate_textures`
- **Issue:** Uses `host_buffer.await_done(10000)`, which blocks the calling thread (often the main thread during startup or a background thread during runtime).
- **Impact:** Runtime hitches during lazy loading.
- **Recommendation:** Transition to a non-blocking "Poll & Notify" pattern where the cache checks if the transfer fence is signaled and only "Promotes" the asset to `Loaded` once done.

### 2.4 Error Handling & Robustness
- **Issue:** Numerous `unwrap()` calls in `vk_util` and `vk_render` (e.g., `device.create_image(...).unwrap()`).
- **Impact:** The engine will crash on any resource allocation failure (OOM, device lost) rather than failing gracefully.
- **Recommendation:** Convert critical path `unwrap()` calls to `Result` or `log::error!` with fallback.

## 3. Logic Refactoring Opportunities

### 3.1 Consolidated Frame Resource Management
- **Current:** Command pools and buffers for each frame are created manually in `VkRenderCore::new` and managed in `init_present_pools`.
- **Proposal:** Create a `VkFrame` struct that encapsulates its own `CommandPool`, `CommandBuffer`, `DescriptorSetAllocator`, and `Fence/Semaphore`. This simplifies `VkRenderCore` and makes adding more "frames in flight" trivial.

### 3.2 Staged Transfer Utility
- **Current:** Texture and Mesh caches implement their own "Host -> Device" copy logic.
- **Proposal:** Extract a `TransferOrchestrator` that handles the "Fill Staging -> Record Copy -> Submit -> Wait/Poll" lifecycle.

### 3.3 Shader Initialization Helper
- **Current:** `init_caches` manually lists every shader path.
- **Proposal:** Move shader manifest to a config file or a more data-driven approach.

## 4. Documentation Strategy (GPU Programming Focus)

To support beginners and intermediates, the following documentation pass will be applied:

1.  **"The Why" Comments:** Explain *why* we transition layouts (e.g., "Must be TRANSFER_DST for the GPU to write to it from a buffer").
2.  **Order of Operations:** Explicitly comment where order is critical (e.g., "Barrier must happen BEFORE the copy").
3.  **Vulkan Concepts:** Briefly explain concepts like "Descriptors" (GPU pointers), "Barriers" (GPU execution sync), and "Stages" (where in the pipeline we are).
4.  **System Overviews:** Every major file will have a `//!` module-level doc comment explaining its role in the engine.

## 5. Implementation Status - Pass 01 (February 12, 2026)

### Completed in this pass
- **2.2 Scene Node Handle Fragility:** Completed.
  - `SceneWorld` migrated from raw index IDs to stable slot+generation `SceneNodeId`.
  - Added stale-handle validation helpers, safe traversal guards, and recursive node removal.
  - Added unit tests for stale handle detection, recursive removal, and traversal safety.
- **3.1 Consolidated Frame Resource Management:** Partially completed (low-risk consolidation).
  - Added `VkFrame::new(...)` constructor to centralize per-frame resource assembly.
  - Refactored `VkPresent::new(...)` to deterministic zip-based assembly (removed head-removal pattern).
  - Refactored `init_present_pools(...)` to reusable queue-pool helper and `Result`-based error propagation.
- **3.3 Shader Initialization Helper:** Completed.
  - Added data-driven core shader manifest: `src/renderer/src/shaders/core_shader_manifest.txt`.
  - Added manifest loader/validation (`load_core_shader_manifest`) and wired `init_caches(...)` to use it.

### Deferred by scope decision
- **2.1 Synchronous Environment Hitch:** Deferred intentionally.
- **2.3 Blocking Transfer "Await":** Deferred intentionally.
- **2.4 Error Handling & Robustness (`unwrap` migration):** Deferred intentionally.

### Documentation strategy status
- Implemented for this pass:
  - Added/expanded module-level `//!` overviews across previously undocumented renderer modules.
  - Added explicit order/why comments in critical image transition/copy paths.
  - Synced `AGENTS.md` files with new scene-handle and shader-manifest architecture.

---
*End of Review*
