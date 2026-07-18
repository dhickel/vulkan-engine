# Synchronization and Fencing

## 1. Purpose & Audience
This chapter is for contributors changing frame orchestration, submission order, swapchain handling, transfer completion, or image/buffer barriers in `src/renderer/src/vulkan`.

## 2. Where This Fits in Engine Flow
Synchronization spans the full frame path in `VkRenderCore::render_with_hooks(...)`:
1. service async transfers
2. wait for the frame-slot fence and clean frame-local resources
3. acquire/bind the swapchain image (or select the headless slot)
4. reset the fence and record passes
5. submit (wait on acquire semaphore, signal render semaphore + fence)
6. present (wait on render semaphore), or drain/submit/present when recording fails

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
unsafe fn wait_for_frame_fence(&self, frame_sync: VkFrameSync) -> Result<(), String> {
    let fence = [frame_sync.render_fence];
    let result = self.device.wait_for_fences(&fence, true, u64::MAX);
    if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
        return Err("Vulkan device lost during fence wait".to_string());
    }
    result.map_err(|err| format!("wait_for_fences failed: {err:?}"))
}
```

Fence reset is a separate fallible operation performed only after image acquisition/binding succeeds. This prevents an acquire retry from leaving an unsignaled fence with no submission.

Snippet Type: Pseudocode
```text
windowed submit:
  wait swap_semaphore at ALL_COMMANDS
  signal render_semaphore at ALL_GRAPHICS
  signal render_fence on completion
headless submit:
  omit swap/render binary semaphores
  still signal render_fence on completion
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
| 2 | Wait current frame fence; clean descriptor/deferred state | `render_fence` |
| 3 | Acquire and bind next swapchain image | signals `swap_semaphore` |
| 4 | Reset fence; record rendergraph passes | command buffer + `render_fence` transaction invariant |
| 5 | Submit graphics | waits `swap_semaphore`, signals `render_semaphore` + `render_fence` |
| 6 | Present | waits `render_semaphore` |

Snippet Type: Pseudocode
```text
for each frame slot:
  wait(frame_fence)
  clean_frame_local_resources()
  image = acquire(signal swap_semaphore)
  if no image: rewind frame slot; keep fence signaled
  bind(image)
  reset(frame_fence)
  if record_passes(image) fails:
    replace partial recording with drain commands
  submit(wait swap_semaphore, signal render_semaphore + frame_fence)
  present(wait render_semaphore)
```

## 5. Best Practices
- Wait before touching frame-owned resources; reset only after successful image acquisition/binding and immediately before the guaranteed record-or-drain submission transaction.
- Tie every barrier to a concrete hazard (write->read, write->write, ownership transfer).
- Prefer named stage-mask helpers in `vk_util` over inline bitmasks so sync intent stays explicit.
- Keep queue-family transfer logic explicit when transfer and graphics queues differ.
- Treat validation-layer warnings as correctness bugs until proven otherwise.

## 6. Gotchas & Failure Modes
- Fence wait/cleanup/reset in the wrong order can race CPU frame resource reuse or leave an unsignaled fence with no submit.
- A windowed post-acquire abort that only signals the fence is incomplete: the acquire semaphore, render semaphore, and image must also be retired through the drain submit/present path.
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
