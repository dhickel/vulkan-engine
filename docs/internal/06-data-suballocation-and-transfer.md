# Data Sub-allocation and Transfer Queue

## 1. Purpose & Audience
This chapter is for contributors working on GPU data lifetime for meshes/material metadata and texture upload handoff. It targets Rust-proficient contributors who are new to engine memory systems and need a concrete mental model of allocation, upload, and loaded-state promotion.

## 2. Where This Fits in Engine Flow
Primary paths:
`MeshCache::allocate*` / `TextureCache::allocate_materials` -> `VkSubAllocator::allocate_bytes` -> `VkStorageBuffer::add_items` -> `VkHostBuffer::submit_*_commands` -> `VkTransfer` channel -> `VkRender::pump_transfer_submissions` / `VkFenceQueue::check_fences`.

For textures, the promotion path is:
`TextureCache::submit_texture_uploads` -> pending batch map -> `TextureCache::poll_texture_uploads` -> `CachedTexture::Unloaded -> CachedTexture::Loaded`.

## 3. Key Concepts

### 3.1 Backing Identity
- Primary buffer always has `buffer_index == 0`.
- Each overflow buffer gets a unique `buffer_index` from a monotonic counter (`next_buffer_id`), starting at 1.
- An overflow buffer is inserted into `extra_buffers` **before** any `VkSubAlloc` referencing it is returned to the caller. This ensures `deallocate` can always find the owning buffer.
- If allocation on a freshly-created overflow buffer fails before any `VkSubAlloc` is returned (pre-commit failure), the buffer's backing `VkBuffer` is destroyed immediately.

### 3.2 Checked Arithmetic
- All size/address/alignment calculations use `checked_add`, `checked_sub`, `checked_mul` in the `checked_arith` module.
- Alignment controls **destination spacing** only: `total_chunk_allot` advances by aligned stride, but only exactly `payload_size` bytes are copied from source slices.
- Overflow or underflow returns a domain error (`VkBufferResult::Error`) instead of panicking.

### 3.3 Mesh Allocation Transaction
- Vertex allocations are staged through `vertex_storage` first.
- Index allocations are staged through `index_storage` second (distinct ownership domain).
- If vertex allocation fails: partial vertex allocs are rolled back through `vertex_storage`.
- If index allocation fails: all staged vertex allocs are rolled back through `vertex_storage`, then partial index allocs through `index_storage`.
- Cache metadata (handle state) is published only after both stages succeed.

### 3.4 Cache Publication Atomicity
- `allocate_materials` stages all work before publishing any cache handle:
  1. Validate handles and collect unloaded materials.
  2. Load all referenced textures.
  3. Allocate meta-buffer storage.
  4. Stage descriptor writes (no cache mutation).
  5. Commit all staged materials atomically.
- If any step in phase 4 fails, all meta allocs from successfully staged, current, and not-yet-staged materials are deallocated, and no handles are updated.
- Texture upload recording owns a rollback boundary: partial image allocations are destroyed on record/setup failure and host command buffers are reset before the error is returned.

### 3.5 Descriptor Budget
- `VkDynamicDescriptorAllocator` enforces a `total_set_budget` ceiling (default 4092).
- Before creating any new pool, a pre-commit check verifies that the sum of all existing pool capacities plus the new pool's capacity does not exceed the budget.
- Budget exhaustion returns `DescriptorAllocError::BudgetExhausted { current, ceiling }` without mutating state.
- Runtime allocators use `new_with_total_set_budget(...)` with a ceiling derived from device descriptor limits, clamped to the allocator cap and at least the initial pool size. Tests may still override via `set_total_set_budget()`.

### 3.6 Lock Order and Guard Lifetime
- **Lock DAG**: `mesh_cache` → `texture_cache` (when both needed); cache lock → `host_buffer` or allocator lock only for bounded local work.
- Render-thread asset pumping uses `try_lock()` for texture finalization and returns a structured error on poison.
- `TextureCache::allocate_textures` no longer sleeps; it exits through bounded idle-poll backpressure if the caller is not pumping transfer work.
- `VkHostBuffer` lock IS held during synchronous buffer-transfer operations; this is by design because the host buffer is the staging resource being used.
- Poisoned locks return `AssetError::Sync`, `String` errors, or logged destroy-path errors instead of process panics in the cache/asset paths touched by this phase.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_storage.rs — Backing identity assignment
let overflow_id = self.take_next_buffer_id()?;
let overflow = Self::allocate_buffer(..., overflow_id)?;
let overflow_slot = self.extra_buffers.len();
self.extra_buffers.push(overflow);

// Only after insertion may add_items return VkSubAlloc values that reference overflow_id.
let overflow = &mut self.extra_buffers[overflow_slot];
match overflow.add_items(...) {
    VkBufferResult::Success(items) => VkAllocResult::Success(items),
    VkBufferResult::Error { successful_allocs, .. } if successful_allocs.is_empty() => {
        let mut failed = self.extra_buffers.pop().unwrap();
        self.destroy_storage_buffer(&mut failed)?;
        VkAllocResult::Failure { ... }
    }
    /* partial successes keep the owner inserted so caller rollback can route correctly */
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_storage.rs — Checked arithmetic in add_items
let payload_size = bytes.len() as u64;
let aligned_stride = checked_arith::aligned_size(payload_size, self.alignment)
    .ok_or_else(|| "aligned stride overflow".to_string())?;

// Source copy uses payload_size bytes; allocation/free-list accounting uses aligned_stride.
curr_allot = curr_allot.checked_add(aligned_stride)
    .ok_or_else(|| "curr_allot overflow".to_string())?;
total_chunk_allot = total_chunk_allot.checked_add(aligned_stride)
    .ok_or_else(|| "total_chunk_allot overflow".to_string())?;
```

Snippet Type: Real
```rust
// src/renderer/src/data/data_cache.rs — Mesh allocation transaction
// Stage 1: allocate vertices
let vertex_allocs = match self.vertex_storage.allocate_bytes(...) {
    VkAllocResult::Success(allocs) => allocs,
    VkAllocResult::Failure { error_msg, successful_allocs } => {
        for alloc in successful_allocs {
            self.vertex_storage.deallocate(alloc);
        }
        return LoadResult::Failed(None);
    }
};

// Stage 2: allocate indices
let index_allocs = match self.index_storage.allocate_bytes(...) {
    VkAllocResult::Success(allocs) => allocs,
    VkAllocResult::Failure { error_msg, successful_allocs } => {
        // Roll back ALL staged vertex allocs first.
        for alloc in vertex_allocs {
            self.vertex_storage.deallocate(alloc);
        }
        // Then deallocate partial index allocs through the correct owner.
        for alloc in successful_allocs {
            self.index_storage.deallocate(alloc);
        }
        return LoadResult::Failed(None);
    }
};
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_descriptor.rs — Descriptor budget pre-commit
let current_total = self.total_allocated_sets();
if current_total.saturating_add(next_sets) > self.total_set_budget {
    return Err(DescriptorAllocError::BudgetExhausted {
        current: current_total,
        ceiling: self.total_set_budget,
    });
}
```

Snippet Type: Pseudocode
```text
# Parking-lot allocation mental model (one VkStorageBuffer)
tail = [unallocated contiguous bytes at end]
free_list = [holes from deallocation]

allocate(request):
  if placement == EndOnly:
    use tail or fail
  else:
    try contiguous chunk (free_list best/largest or tail)
    if contiguous unavailable and placement == ContiguousPreferred:
      split across multiple chunks
  record uploads in batches <= max_upload_bytes
  submit transfer + graphics barrier work
  wait for latch/fence completion before reusing staging buffer
  advance tail / shrink used free chunk entries
  All arithmetic uses checked_add/checked_sub — no silent overflow.

deallocate(sub_alloc):
  route to owning buffer via sub_buffer_index
  push freed range into free_list
  merge neighbors where [left.end == new.start] or [new.end == right.start]
  All arithmetic uses checked operations.
```

## 5. Best Practices
- Preserve allocator invariants when editing:
  - Free chunks must never overlap.
  - `sub_buffer_index` must always identify the owning `VkStorageBuffer`.
  - Tail and free-list updates must consume exactly the uploaded byte ranges.
  - Every overflow buffer must be in `extra_buffers` before any `VkSubAlloc` referencing it is returned.
- Keep upload completion semantics explicit:
  - Enqueue (`submit_*`) is not completion.
  - Treat data as draw-safe only after fence/latch completion and cache-state promotion.
- Keep partial-allocation failure handling in sync with caller rollback:
  - Mesh path: vertex allocs roll back through `vertex_storage`, index allocs through `index_storage`.
  - Material path: meta allocs are deallocated on any staging failure; no partial cache mutation.
- Prefer handle validation before mutating loaded/unloaded state in caches.
- Never hold cache locks across transfer-latch blocking, file I/O, callbacks, or retries.

## 6. Gotchas & Failure Modes
- Free-list corruption risk:
  - Any wrong `offset/size/address` arithmetic can create overlapping `FreeChunk` entries.
  - `deallocate` trusts the incoming `VkSubAlloc`; invalid or stale allocations can poison allocator state.
- Fragmentation surprises:
  - `ContiguousOnly` may fail even if total free bytes are large.
  - `ContiguousPreferred` may split allocations across chunks; this can increase bookkeeping and future fragmentation pressure.
- Transfer visibility timing:
  - Submitting transfer commands does not make data immediately visible to draw code.
  - Skipping poll/pump paths leaves latches unresolved and batches stuck in pending state.
- Queue-ordering dependencies:
  - `VkRender::pump_transfer_submissions` must run frequently (startup loader loop and per-frame render path).
  - If transfer submissions are not drained, `await_done()` callers can block until timeout.
- Budget exhaustion:
  - Descriptor pool growth is bounded by `total_set_budget`. Heavy material/scene loading may hit this ceiling.
  - Callers receive `BudgetExhausted { current, ceiling }` and can adjust batch sizes or defer work.
- Alignment vs payload:
  - Alignment controls destination spacing in the GPU buffer, not source read length.
  - Only exactly `payload_size` bytes are copied from each source slice.

## 7. Debugging Playbook
- Step 1: confirm transfer progress first.
  - Inspect whether `pump_transfer_submissions` is being called and whether `VkFenceQueue` queue length is decreasing.
- Step 2: check latch state.
  - If `countdown_latch.get_count() != 0` indefinitely, inspect submitted command count versus fence signaling.
- Step 3: validate cache-state transitions.
  - For textures, verify `pending_batches` entries move from `WaitingFence` to promotion in `poll_texture_uploads`.
  - For meshes/materials, verify `Cached*::Unloaded` entries are replaced with `Cached*::Loaded` only after successful allocations.
- Step 4: inspect allocator integrity.
  - Log/inspect `sub_buffer_index`, allocation offsets, and free chunk boundaries after deallocation-heavy workloads.
  - If large allocations fail unexpectedly, inspect fragmentation (`max_free` vs `total_free`).
  - Verify overflow buffer identities are unique and monotonic.
- Step 5: reproduce in bounded headless smoke run and collect first error.
  - Use `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_model_load`.
  - Use `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_async_loading`.

## 8. Cross-Module Links
- Sub-allocator core: `src/renderer/src/vulkan/vk_storage.rs`
- Transfer channel, host buffer, fence queue: `src/renderer/src/vulkan/vk_types.rs`
- Transfer servicing in render loop: `src/renderer/src/vulkan/vk_render.rs`
- Cache state transitions and pending batches: `src/renderer/src/data/data_cache.rs`
- GPU-facing allocation payloads: `src/renderer/src/data/gpu_data.rs`
- Descriptor pool management: `src/renderer/src/vulkan/vk_descriptor.rs`
- Asset loading and cache locks: `src/renderer/src/api/assets.rs`

## 9. Standard References
- Vulkan Guide memory allocation: https://github.khronos.org/Vulkan-Site/guide/latest/memory_allocation.html
- Vulkan spec memory chapter: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#memory
- Vulkan buffer device address: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-bufferDeviceAddress
- AMD Vulkan memory recommendations: https://gpuopen.com/learn/vulkan-device-memory/
- Vulkan Guide synchronization overview: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/internal/03-asset-lifecycle-and-io.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `src/renderer/src/vulkan/AGENTS.md`
