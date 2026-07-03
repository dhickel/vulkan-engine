# Data Sub-allocation and Transfer Queue

## 1. Purpose & Audience
This chapter is for contributors working on GPU data lifetime for meshes/material metadata and texture upload handoff. It targets Rust-proficient contributors who are new to engine memory systems and need a concrete mental model of allocation, upload, and loaded-state promotion.

## 2. Where This Fits in Engine Flow
Primary paths:
`MeshCache::allocate*` / `TextureCache::allocate_materials` -> `VkSubAllocator::allocate_bytes` -> `VkStorageBuffer::add_items` -> `VkHostBuffer::submit_*_commands` -> `VkTransfer` channel -> `VkRender::pump_transfer_submissions` / `VkFenceQueue::check_fences`.

For textures, the promotion path is:
`TextureCache::submit_texture_uploads` -> pending batch map -> `TextureCache::poll_texture_uploads` -> `CachedTexture::Unloaded -> CachedTexture::Loaded`.

## 3. Key Concepts
- `VkSubAllocator` is a buffer-of-buffers manager:
- It owns one primary `VkStorageBuffer` plus `extra_buffers` allocated on demand.
- Every `VkSubAlloc` includes `sub_buffer_index`, so deallocation can route back to the correct backing buffer.
- `VkStorageBuffer` uses a tail allocator plus free-list reuse:
- `buffer_tail` is fast O(1) bump allocation for new contiguous writes.
- `free_chunks` tracks freed regions and coalesces adjacent chunks.
- Placement policy controls behavior:
- `EndOnly`: allocate only from tail.
- `ContiguousOnly`: require one contiguous region.
- `ContiguousPreferred`: prefer contiguous, fall back to disjoint chunking.
- Upload completion is split into submission and completion:
- Submission happens when `submit_transfer_commands`/`submit_graphics_commands` enqueue `VkCmdSubmitInfo` into `VkTransfer`.
- Completion happens later when render thread submits queued work and `VkFenceQueue` observes signaled fences.
- Cache states are separate from allocation success:
- Allocation/upload may partially succeed.
- Cache entries become `Loaded` only after promotion steps (`self.cached_*[slot] = ...Loaded(...)`).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_storage.rs
pub fn allocate_bytes(
    &mut self,
    data: &[&[u8]],
    buffer_placement: BufferPlacement,
) -> VkAllocResult {
    let host_buffer = match self.transfer_buffer.lock() {
        Ok(buffer) => buffer,
        Err(error_msg) => {
            return VkAllocResult::Failure {
                error_msg: format!("Error acquiring host buffer lock: {:?}", error_msg),
                successful_allocs: vec![],
            };
        }
    };

    match self.buffer.add_items(data, buffer_placement, &self.device, &host_buffer) {
        VkBufferResult::OutOfSpace(mut partial_alloc) => {
            // grows by allocating one extra VkStorageBuffer and retrying remaining items
            /* ... */
        }
        VkBufferResult::Success(allocs) => VkAllocResult::Success(allocs),
        VkBufferResult::Error { error_msg, .. } => VkAllocResult::Failure {
            error_msg: format!("Error encountered during allocation: {:?}", error_msg),
            successful_allocs: vec![],
        },
    }
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_storage.rs
pub fn deallocate(&mut self, sub_alloc: VkSubAlloc) {
    if sub_alloc.sub_buffer_index == 0 {
        self.buffer.delete_item(sub_alloc);
    } else if let Some(buffer) = self.extra_buffers.get_mut(sub_alloc.sub_buffer_index as usize) {
        buffer.delete_item(sub_alloc)
    }
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_storage.rs
pub fn insert(&mut self, mut new_chunk: FreeChunk) {
    self.total_free += new_chunk.size;
    let mut front_adj: Option<usize> = None;
    let mut back_adj: Option<usize> = None;

    for (i, iter_chunk) in self.chunks.iter_mut().enumerate() {
        if (iter_chunk.start_addr + iter_chunk.size) == new_chunk.start_addr {
            front_adj = Some(i)
        }
        if (new_chunk.start_addr + new_chunk.size) == iter_chunk.start_addr {
            back_adj = Some(i)
        }
    }

    // Coalesce adjacent neighbors, then swap_remove old entries.
    /* ... */
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_types.rs
pub fn submit_transfer_commands(
    &self,
    submit_params: VkSubmitParam,
) -> Result<(), SendError<VkCmdSubmitInfo>> {
    let submit_info = VkCmdSubmitInfo {
        cmd_buffer: self.transfer_pool.buffers[0],
        fence: [self.fence[0]],
        semaphore: self.semaphore,
        submit_params,
        queue_type: VkQueueType::Transfer,
        latch_guard: self.countdown_latch.create_guard(),
    };
    self.render_sender.send(submit_info)
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
pub fn pump_transfer_submissions(&mut self, max_submissions: usize) -> usize {
    self.fence_await_queue.check_fences(&self.device);
    if max_submissions == 0 {
        return 0;
    }
    self.drain_transfer_submissions(max_submissions)
}
```

Snippet Type: Real
```rust
// src/renderer/src/data/data_cache.rs
pub fn poll_texture_uploads(&mut self) -> usize {
    if self.pending_batches.is_empty() {
        return 0;
    }
    let host_buffer = self.host_buffer.lock().unwrap();
    if host_buffer.countdown_latch.get_count() != 0 {
        return 0;
    }
    host_buffer.reset_buffers(&self.device);
    drop(host_buffer);
    // WaitingFence -> promote Unloaded to Loaded
    /* ... */
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

deallocate(sub_alloc):
  push freed range into free_list
  merge neighbors where [left.end == new.start] or [new.end == right.start]
```

## 5. Best Practices
- Preserve allocator invariants when editing:
- Free chunks must never overlap.
- `sub_buffer_index` must always identify the owning `VkStorageBuffer`.
- Tail and free-list updates must consume exactly the uploaded byte ranges.
- Keep upload completion semantics explicit:
- Enqueue (`submit_*`) is not completion.
- Treat data as draw-safe only after fence/latch completion and cache-state promotion.
- Keep partial-allocation failure handling in sync with caller rollback:
- Mesh path currently rolls back vertex allocations if index allocation fails.
- Material path deallocates successful metadata allocations on partial failure.
- Prefer handle validation before mutating loaded/unloaded state in caches.

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
- Sharp edges worth tracking:
- `VkHostBuffer::destroy` now destroys both fences (`fence[0]` transfer, `fence[1]` graphics) as of Sprint 12.
- `VkStorageBuffer::select_best_chunk` calls `find_best_fit(self.max_size)`, which is stricter than `find_best_fit(total_bytes)` and biases toward non-best-fit fallback behavior.

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
- Step 5: reproduce in bounded headless smoke run and collect first error.
- Use `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_model_load`.
- Use `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_async_loading`.

## 8. Cross-Module Links
- Sub-allocator core: `src/renderer/src/vulkan/vk_storage.rs`
- Transfer channel, host buffer, fence queue: `src/renderer/src/vulkan/vk_types.rs`
- Transfer servicing in render loop: `src/renderer/src/vulkan/vk_render.rs`
- Cache state transitions and pending batches: `src/renderer/src/data/data_cache.rs`
- GPU-facing allocation payloads: `src/renderer/src/data/gpu_data.rs`

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
