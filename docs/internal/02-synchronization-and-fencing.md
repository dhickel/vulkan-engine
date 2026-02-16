# Synchronization and Fencing

## 1. Purpose & Audience
This chapter is for contributors changing frame orchestration, submission order, swapchain handling, transfer completion, or image/buffer barriers in `src/renderer/src/vulkan`.

## 2. Where This Fits in Engine Flow
Synchronization spans the full frame path in `VkRenderCore::render_with_hooks(...)`:
1. service async transfers
2. wait/reset frame fence
3. acquire swapchain image
4. record passes
5. submit (wait on acquire semaphore, signal render semaphore + fence)
6. present (wait on render semaphore)

## 3. Key Concepts
- `VkFrameSync` owns per-frame binary semaphores and one frame fence.
- Frame ring (`VkPresent`) prevents CPU overwrite of GPU-owned per-frame resources.
- `VkFenceQueue` polls async transfer fences and unblocks background workers.
- Barriers in `vk_util::record_image_barrier(...)` encode layout/access/stage contracts.
- Template contract reference: `docs/internal/00-index.md` (mandatory 10-section order).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
unsafe fn wait_and_reset_frame_fence(&self, frame_sync: VkFrameSync) {
    let fence = [frame_sync.render_fence];
    self.device.wait_for_fences(&fence, true, u64::MAX).unwrap();
    self.device.reset_fences(&fence).unwrap();
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
let wait_info = [vk_util::semaphore_submit_info(
    vk_util::frame_acquire_wait_stage_mask(),
    frame.frame_sync.swap_semaphore,
)];
let signal_info = [vk_util::semaphore_submit_info(
    vk_util::frame_render_complete_signal_stage_mask(),
    frame.frame_sync.render_semaphore,
)];
```

Snippet Type: Real
```rust
// src/renderer/src/data/data_cache.rs + src/renderer/src/vulkan/vk_storage.rs
host_buffer.submit_transfer_commands(VkSubmitParam::signaling(
    vk_util::async_transfer_signal_stage_mask(),
))?;
host_buffer.submit_graphics_commands(VkSubmitParam::waiting(
    vk_util::async_texture_upload_wait_stage_mask(), // texture upload path
))?;
host_buffer.submit_graphics_commands(VkSubmitParam::waiting(
    vk_util::async_buffer_upload_wait_stage_mask(), // storage buffer upload path
))?;
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_util.rs (barrier tuple pattern)
record_image_barrier(
    device,
    cmd,
    image,
    None,
    (old_layout, new_layout),
    (src_stage, dst_stage),
    Some((src_access, dst_access)),
    None,
);
```

Frame timeline (current behavior):

| Step | CPU action | GPU dependency object |
|---|---|---|
| 1 | Poll async transfer queue and fence queue | transfer fences (`VkFenceQueue`) |
| 2 | Wait/reset current frame fence | `render_fence` |
| 3 | Acquire next swapchain image | signals `swap_semaphore` |
| 4 | Record rendergraph passes | command buffer only |
| 5 | Submit graphics | waits `swap_semaphore`, signals `render_semaphore` + `render_fence` |
| 6 | Present | waits `render_semaphore` |

Snippet Type: Pseudocode
```text
for each frame slot:
  wait(frame_fence)
  reset(frame_fence)
  image = acquire(signal swap_semaphore)
  record_passes(image)
  submit(wait swap_semaphore, signal render_semaphore + frame_fence)
  present(wait render_semaphore)
```

## 5. Best Practices
- Keep wait/reset fence operations before touching frame-owned resources.
- Tie every barrier to a concrete hazard (write->read, write->write, ownership transfer).
- Prefer named stage-mask helpers in `vk_util` over inline bitmasks so sync intent stays explicit.
- Keep queue-family transfer logic explicit when transfer and graphics queues differ.
- Treat validation-layer warnings as correctness bugs until proven otherwise.

## 6. Gotchas & Failure Modes
- Fence wait/reset in wrong order can race CPU frame resource reuse.
- Stage/access mismatches can appear to work on one GPU and fail on another.
- Upload waits at the wrong stage (for example waiting at shader stages when work starts in transfer/vertex-input) can cause hidden stalls or hazards.
- Assuming transfer completion before fence/latch completion causes stale uploads.
- Swapchain rebuild failures can surface as acquire/present errors and trigger resize loops.

## 7. Debugging Playbook
- Step 1: reproduce with validation layers enabled and capture first sync warning.
- Step 2: map warning to exact resource transition (image/buffer, old/new layout, stage/access).
- Step 3: verify frame timeline order in `VkRenderCore::render_with_hooks(...)`.
- Step 4: inspect transfer completion path (`pump_transfer_submissions`, `VkFenceQueue::check_fences`).
- Step 5: if present/acquire fails, verify resize handling and swapchain target rebinding.

## 8. Cross-Module Links
- Frame loop orchestration: `src/renderer/src/vulkan/vk_render.rs`
- Frame sync types: `src/renderer/src/vulkan/vk_types.rs`
- Barrier helpers: `src/renderer/src/vulkan/vk_util.rs`
- Pass sequencing context: `src/renderer/src/rendergraph/mod.rs`

## 9. Standard References
- Vulkan synchronization chapter: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization
- Vulkan implicit synchronization: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#sync-implicit
- Vulkan Guide sync examples: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html
- Vulkan Guide layout transitions: https://github.khronos.org/Vulkan-Site/guide/latest/layout_transitions.html
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

## 10. See Also
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/03-asset-lifecycle-and-io.md`
- `src/renderer/src/vulkan/AGENTS.md`
