# Vulkan Sync and Frame Lifecycle

## 1. Purpose & Audience
This chapter is for contributors editing Vulkan frame orchestration in `vk_render`, especially fence/semaphore ordering, image layout transitions, descriptor pool lifetime, and rendergraph pass sequencing.

## 2. Where This Fits in Engine Flow
Per-frame backend flow:
`Renderer::render_scene(...)` -> `VkRender::render_with_hooks(...)` -> `VkRenderCore::render_with_hooks(...)` -> acquire frame slot -> rendergraph pass recording -> submit -> present.

## 3. Key Concepts
- `VkFrameSync` defines one frame slot's synchronization contract:
  - `swap_semaphore`: signaled by acquire, waited by graphics submit.
  - `render_semaphore`: signaled by graphics submit, waited by present.
  - `render_fence`: waited/reset by CPU before reusing frame-local resources.
- `VkPresent` is the frame-ring owner (`Vec<VkFrame>`) and maps acquired swapchain image index to the current frame's `present_image`/`present_image_view`.
- Descriptor lifetime is frame-scoped: `VkFrame.descriptors.clear_pools(...)` runs at frame-slot reuse after fence wait.
- Two barrier helper styles exist:
  - `transition_image` / `transition_image_layered`: broad `ALL_COMMANDS` + memory read/write masks (`vk_sync2` path).
  - `record_image_barrier`: explicit `layout/stage/access` (legacy-style helper, heavily used in texture upload/mip generation).
- Semaphore stage masks are centralized in `vk_util` helpers:
  - frame submit: `frame_acquire_wait_stage_mask()`, `frame_render_complete_signal_stage_mask()`
  - async upload submit: `async_transfer_signal_stage_mask()`, `async_texture_upload_wait_stage_mask()`, `async_buffer_upload_wait_stage_mask()`
- Frame timing now records `frame_fence_wait` so CPU await spikes are isolated from broader `acquire_frame`.
- Rendergraph pass order is explicit and semantic: `PrepareTargets -> Skybox -> Geometry -> PresentCopy -> Imgui`.

Synchronization primitive role table:

| Primitive | Producer | Consumer | Used in code | Purpose |
|---|---|---|---|---|
| `swap_semaphore` | `acquire_next_image2` | `queue_submit2` wait list | `acquire_swapchain_image_index`, `submit_frame` | Prevent rendering until swapchain image is available |
| `render_semaphore` | `queue_submit2` signal list | `queue_present` wait list | `submit_frame`, `present_frame` | Prevent present before rendering completes |
| `render_fence` | `queue_submit2` | CPU (`wait_for_fences`) | `submit_frame`, `wait_and_reset_frame_fence` | Prevent CPU frame-slot reuse while GPU still owns slot resources |
| transfer fences (`VkFenceQueue`) | async transfer submits | render thread polling | `pump_transfer_submissions`, `service_async_transfers` | Avoid consuming incomplete async uploads |

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
fn acquire_frame_slot(&mut self) -> Option<FrameAcquire> {
    let frame_data = self.presentation.get_next_frame();
    let frame_sync = frame_data.sync;
    let cmd_pool = frame_data.cmd_pools.get(VkQueueType::Graphics);
    let cmd_buffer = cmd_pool.buffers[0];
    let queue = self.vulkan_cache.queues.get_queue(VkQueueType::Graphics);

    unsafe {
        self.wait_and_reset_frame_fence(frame_sync);
        self.cleanup_curr_frame_resources(); // includes curr_frame.descriptors.clear_pools(...)
    }

    let image_index = unsafe { self.acquire_swapchain_image_index(frame_sync) };
    let Some(image_index) = image_index else {
        self.resize_requested = true;
        return None;
    };

    if let Err(err) = self.presentation.bind_acquired_present_target(image_index) {
        error!(
            "Failed to bind acquired present target {}: {:?}",
            image_index, err
        );
        self.resize_requested = true;
        return None;
    }
    Some(FrameAcquire { queue, cmd_buffer, frame_sync, image_index })
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
fn submit_frame(&self, frame: FrameAcquire) {
    let cmd_info = [vk_util::command_buffer_submit_info(frame.cmd_buffer)];
    let wait_info = [vk_util::semaphore_submit_info(
        vk_util::frame_acquire_wait_stage_mask(),
        frame.frame_sync.swap_semaphore,
    )];
    let signal_info = [vk_util::semaphore_submit_info(
        vk_util::frame_render_complete_signal_stage_mask(),
        frame.frame_sync.render_semaphore,
    )];
    let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];
    unsafe { self.device.queue_submit2(frame.queue, &submit, frame.frame_sync.render_fence).unwrap(); }
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_descriptor.rs
pub fn clear_pools(&mut self, device: &ash::Device) -> Result<(), String> {
    for &pool in &self.ready_pools { unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())?; } }
    for &pool in &self.full_pools {
        unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())?; }
        self.ready_pools.push(pool);
    }
    self.full_pools.clear();
    Ok(())
}
```

Barrier transition cookbook (current engine patterns):

| Case | Layout transition | Stage mask | Access mask | Where used |
|---|---|---|---|---|
| Color target prep | `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` (via `GENERAL` in draw path) | `ALL_COMMANDS -> ALL_COMMANDS` (`transition_image`) | `MEMORY_WRITE -> MEMORY_WRITE|MEMORY_READ` (`transition_image`) | `prepare_draw_targets`, `copy_draw_to_present` |
| Depth target prep | `UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL` | `ALL_COMMANDS -> ALL_COMMANDS` (`transition_image`) | `MEMORY_WRITE -> MEMORY_WRITE|MEMORY_READ` (`transition_image`) | `prepare_draw_targets` |
| Transfer upload/mip finalization | `TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL` | `TRANSFER -> FRAGMENT_SHADER` | `TRANSFER_READ -> SHADER_READ` | `record_mip_maps_generation`, compressed texture upload path |
| Transfer queue ownership handoff | `TRANSFER_DST_OPTIMAL -> TRANSFER_SRC_OPTIMAL` | `TRANSFER -> TRANSFER` | `TRANSFER_WRITE -> TRANSFER_READ` + queue family indices | `upload_texture` transfer->graphics ownership barrier |

Snippet Type: Pseudocode
```text
frame_slot = presentation.get_next_frame()
wait(frame_slot.render_fence)
reset(frame_slot.render_fence)
process_deferred_deletes(frame_slot)
reset_dynamic_descriptor_pools(frame_slot)

acquired = acquire_next_image(signal=swap_semaphore)
bind_acquired_present_target(acquired.index)
record_rendergraph_passes_in_fixed_order()

submit(wait=swap_semaphore, signal=render_semaphore, fence=render_fence)
present(wait=render_semaphore, image_index=acquired.index)
```

## 5. Best Practices
- Keep acquire -> record -> submit -> present order explicit; avoid hidden side effects between steps.
- Wait and reset the frame fence before any frame-local resource reuse (descriptor pools, deferred deletion queue, command buffer reset).
- Document producer/consumer for every semaphore/fence when changing submission paths.
- Prefer explicit `record_image_barrier` when queue ownership or narrow stage/access scopes matter.
- Keep rendergraph pass ordering assumptions written next to transition changes.

## 6. Gotchas & Failure Modes
- Wrong fence timing: resetting/clearing per-frame state before fence completion can cause use-after-free style GPU faults.
- Stage masks too broad or too narrow:
  - Too broad (`ALL_COMMANDS`) may hide performance issues.
  - Too narrow can create intermittent hazards across GPUs/drivers.
- Descriptor pool exhaustion and churn if `clear_pools` no longer runs on frame-slot reuse.
- Swapchain rebuild lifecycle: `rebuild_swapchain` calls `replace_present_images` which invokes `destroy_present_views` before rebinding. This edge was addressed; monitor for regressions around frame slot desync on resize.
- `VkPresent::get_next_frame`/`get_curr_frame_mut` ring semantics depend on counter ordering; changing this can silently desync acquired image binding.

## 7. Debugging Playbook
- Step 1: enable validation layers and capture the first synchronization warning (later warnings are often cascades).
- Step 2: trace warning to exact transition in `prepare_draw_targets`, `copy_draw_to_present`, or upload barrier path.
- Step 3: verify frame-slot sequence in `acquire_frame_slot`, `submit_frame`, and `present_frame`.
- Step 4: if async uploads appear stale, inspect `pump_transfer_submissions` and fence queue completion status.
- Step 5: if resize/present loops occur, inspect `rebuild_swapchain` and acquired present target rebinding (`bind_acquired_present_target`).

## 8. Cross-Module Links
- Frame loop core: `src/renderer/src/vulkan/vk_render.rs`
- Frame-ring and sync types: `src/renderer/src/vulkan/vk_types.rs`
- Descriptor allocators: `src/renderer/src/vulkan/vk_descriptor.rs`
- Barrier helpers and upload transitions: `src/renderer/src/vulkan/vk_util.rs`
- Rendergraph ordering: `src/renderer/src/rendergraph/mod.rs`
- Present/UI pass boundaries: `src/renderer/src/rendergraph/passes/present_copy_pass.rs`, `src/renderer/src/rendergraph/passes/imgui_pass.rs`

## 9. Standard References
- Vulkan synchronization chapter: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization
- Vulkan descriptor sets chapter: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets
- Vulkan Guide synchronization overview: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- Vulkan Guide synchronization examples: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html
- Vulkan Guide layout transitions: https://github.khronos.org/Vulkan-Site/guide/latest/layout_transitions.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/internal/02-synchronization-and-fencing.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `src/renderer/src/vulkan/AGENTS.md`
