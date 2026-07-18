# Vulkan Sync and Frame Lifecycle

## 1. Purpose & Audience
This chapter is for contributors editing Vulkan frame orchestration in `vk_render`, `vk_frame` and `vk_commands`, especially fence/semaphore ordering, image layout transitions, descriptor pool lifetime, and rendergraph pass sequencing.

## 2. Where This Fits in Engine Flow
Per-frame backend flow:
`Renderer::render_scene(...)` -> `VkRender::render_with_hooks(...)` -> `VkRenderCore::render_with_hooks(...)` -> acquire frame slot (`vk_frame`) -> rendergraph pass recording (`vk_commands`) -> submit -> present.

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
- Frame timing records `frame_fence_wait` so CPU await spikes are isolated from broader `acquire_frame`.
- Frame acquisition waits and cleans the slot first, acquires/binds an image, and resets the fence only after acquisition succeeds.
- Once the fence is reset, a recording failure uses a drain transaction: replace partial commands, submit to signal the fence, and, for windowed acquisition, present to consume/release the semaphore and image state.
- Rendergraph pass order is explicit and semantic: `PrepareTargets -> Shadow -> Skybox -> Geometry -> PresentCopy -> Imgui -> DebugCapture -> TerminalPresent`.

Synchronization primitive role table:

| Primitive | Producer | Consumer | Used in code | Purpose |
|---|---|---|---|---|
| `swap_semaphore` | `acquire_next_image2` | `queue_submit2` wait list | `acquire_swapchain_image_index`, `submit_frame` | Prevent rendering until swapchain image is available |
| `render_semaphore` | `queue_submit2` signal list | `queue_present` wait list | `submit_frame`, `present_frame` | Prevent present before rendering completes |
| `render_fence` | `queue_submit2` | CPU (`wait_for_fences`) | `submit_frame`, `wait_for_frame_fence`, `reset_frame_fence` | Prevent CPU frame-slot reuse while GPU still owns slot resources |
| transfer fences (`VkFenceQueue`) | async transfer submits | render thread polling | `pump_transfer_submissions`, `service_async_transfers` | Avoid consuming incomplete async uploads |

## 4. Code Walkthrough
Snippet Type: Pseudocode
```rust
// Simplified from src/renderer/src/vulkan/vk_frame.rs
fn acquire_frame_slot(&mut self) -> Result<Option<FrameAcquire>, String> {
    let frame = self.presentation.get_next_frame();
    let frame_sync = frame.sync;
    let expected_frame_serial = self.presentation.frame_epoch();

    // Wait for the frame-slot fence and create a completion token.
    let mut token = unsafe { self.wait_for_frame_fence(frame_sync, frame.index)?; }
    // Descriptor cleanup consumes the token; rejects stale/duplicate tokens.
    unsafe { self.cleanup_curr_frame_resources(&mut token, expected_frame_serial)?; }
    debug_assert!(token.is_consumed());

    let acquired = /* acquire/bind swapchain image, or choose headless slot */;
    let Some(acquired) = acquired else {
        self.presentation.rewind_frame();
        self.resize_requested = true;
        return Ok(None); // fence remains signaled
    };

    unsafe { self.reset_frame_fence(frame_sync)?; }
    Ok(Some(acquired))
}
```

Snippet Type: Pseudocode
```rust
// Simplified from src/renderer/src/vulkan/vk_frame.rs
fn submit_frame(&self, frame: FrameAcquire) -> Result<(), String> {
    // Windowed: wait swap semaphore, signal render semaphore, signal frame fence.
    // Headless: no binary semaphores; the queue submission still signals the fence.
    let result = unsafe { self.device.queue_submit2(/* ... */, frame.frame_sync.render_fence) };
    if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
        return Err("Vulkan device lost during frame submission".into());
    }
    result.map_err(|err| format!("queue_submit2 failed: {err:?}"))
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_descriptor.rs
pub fn clear_pools(
    &mut self,
    device: &ash::Device,
    token: &mut CompletedFrameSlot,
    expected_frame_serial: u64,
) -> Result<(), DescriptorAllocError> {
    // Token validation before any Vulkan call.
    let (slot_index, epoch) = token.take()
        .ok_or_else(|| DescriptorAllocError::ResetRejected("token already consumed".into()))?;
    if slot_index != self.frame_slot_index
        || epoch != expected_frame_serial
        || epoch <= self.last_reset_epoch
    {
        return Err(DescriptorAllocError::ResetRejected("slot/serial mismatch".into()));
    }
    // Reset every unique pool exactly once. On partial failure, quarantine all
    // records as exhausted so no physically-reset subset is claimed reusable.
    for handle in unique_handles(&self.ready_pools, &self.full_pools) {
        if let Err(error) = self.adapter.reset_pool(device, handle) {
            self.quarantine_all_pools();
            return Err(DescriptorAllocError::ResetFailed(format!("{error:?}")));
        }
    }
    // All Vulkan resets succeeded; update state and begin fresh frame counters.
    for record in self.ready_pools.iter_mut() { record.allocated_sets = 0; }
    for record in self.full_pools.drain(..) { self.ready_pools.push(record); }
    self.last_reset_epoch = epoch;
    self.stats.reset_count += 1;
    Ok(())
}
```

Barrier transition cookbook (current engine patterns):

| Case | Layout transition | Stage mask | Access mask | Where used |
|---|---|---|---|---|
| Color target prep | `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` (via `GENERAL` in draw path) | `ALL_COMMANDS -> ALL_COMMANDS` (`transition_image`) | `MEMORY_WRITE -> MEMORY_WRITE|MEMORY_READ` (`transition_image`) | `prepare_draw_targets`, `copy_draw_to_present` |
| Depth target prep | `UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL` | `ALL_COMMANDS -> ALL_COMMANDS` (`transition_image`) | `MEMORY_WRITE -> MEMORY_WRITE|MEMORY_READ` (`transition_image`) | `prepare_draw_targets` |
| Directional shadow write/read | `UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL` | top-of-pipe → early/late fragment tests → fragment shader | none → depth write → sampled read | `ShadowPass` |
| Transfer upload/mip finalization | `TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL` | `TRANSFER -> FRAGMENT_SHADER` | `TRANSFER_READ -> SHADER_READ` | `record_mip_maps_generation`, compressed texture upload path |
| Transfer queue ownership handoff | `TRANSFER_DST_OPTIMAL -> TRANSFER_SRC_OPTIMAL` | `TRANSFER -> TRANSFER` | `TRANSFER_WRITE -> TRANSFER_READ` + queue family indices | `upload_texture` transfer->graphics ownership barrier |

Snippet Type: Pseudocode
```text
frame_slot = presentation.get_next_frame()
wait(frame_slot.render_fence)
process_deferred_deletes(frame_slot)
reset_dynamic_descriptor_pools(frame_slot)

acquired = acquire_next_image(signal=swap_semaphore)
if no_image: rewind_slot_and_leave_fence_signaled()
bind_acquired_present_target(acquired.index)
reset(frame_slot.render_fence)

if record_rendergraph_passes_in_fixed_order() fails:
  replace_partial_commands_with_drain()
  submit(wait=swap_semaphore, signal=render_semaphore, fence=render_fence)
  present_to_retire_windowed_image_and_semaphores()
else:
  submit(wait=swap_semaphore, signal=render_semaphore, fence=render_fence)
  present(wait=render_semaphore, image_index=acquired.index)
```

## 5. Best Practices
- Keep acquire -> record -> submit -> present order explicit; avoid hidden side effects between steps.
- Wait for the frame fence before frame-local resource reuse. Reset it only after acquisition/binding succeeds and immediately before entering the record/submit transaction.
- Document producer/consumer for every semaphore/fence when changing submission paths.
- Prefer explicit `record_image_barrier` when queue ownership or narrow stage/access scopes matter.
- Keep rendergraph pass ordering assumptions written next to transition changes.

## 6. Gotchas & Failure Modes
- Wrong fence timing: clearing per-frame state before fence completion can cause use-after-free style GPU faults; resetting the fence before a path that may return without submit can deadlock slot reuse.
- Stage masks too broad or too narrow:
  - Too broad (`ALL_COMMANDS`) may hide performance issues.
  - Too narrow can create intermittent hazards across GPUs/drivers.
- Descriptor pool exhaustion and churn if `clear_pools` no longer runs on frame-slot reuse.
- Descriptor reset without a valid `CompletedFrameSlot` token is rejected. The token must be created by the fence-wait path and is single-use. Duplicate or mismatched tokens produce `ResetRejected`.
- `CompletedFrameSlot` carries two distinct serials: the descriptor-reset epoch consumed by `clear_pools`, and the slot's `last_submitted_serial` used for GPU retirement. Conflating them would either reject valid descriptor cleanup or retire resources against the wrong submission.
- **Handle retirement**: Every slot+generation resource store must use fence-aware retirement. `GpuRetirementQueue<T>` delays payload destruction and slot reuse until `completed_serial >= retire_after`. `FrameSerial` is committed only after successful queue submission and completion advances only from successful fence observations; submit failure cannot fabricate either serial. Mesh references are marked while immutable draw records are built. Mesh unload uses `retire_after = max(last_referenced_serial, latest_submitted_serial)`, immediately bumps generation/removes lookup visibility, and retains `VkMeshBuffers`, suballocations, and the neutral geometry DTO through `GpuRetirementQueue` until `reap_mesh_retirement` destroys the payload and releases the slot. Reserved default slots are non-retirable. Dependent phases must use the same `RetirementClass` taxonomy for bounds entries, collider recipes, BVH leaves, LOD chains, and instance records.
- Fragmentation metric means observed `ERROR_FRAGMENTED_POOL` events and affected pool counts, not an unsupported claim about driver-internal fragmentation percentage.
- Swapchain rebuild lifecycle: `rebuild_swapchain` follows an explicit Nascent → Current → Retired → Absent state machine (see `src/renderer/src/vulkan/vk_swapchain.rs`). Once `vkCreateSwapchainKHR` is called with a non-null `oldSwapchain`, that generation is permanently Retired even if the new creation fails. A Retired generation is never rendered through, restored as current, or passed again as `oldSwapchain`. `SwapchainOwner` solely owns window-system present views; `VkPresent` only references them. Views are destroyed before their swapchain handle, partial view creation is rolled back, and the retired handle remains alive until replacement publication commits. `device_wait_idle` is still called before retirement to drain in-flight work but is not a substitute for the ownership model — state tests must not depend on it.
- Resize requests are coalesced to the latest event, including zero extents. A zero extent remains pending and is deferred without capability queries or replacement calls. A successful non-zero rebuild consumes only the request generation it installed, so a newer concurrent request remains pending.
- Acquire and present results are classified structurally via `AcquireClass` / `PresentClass` enums without string parsing. Surface-lost is distinct from out-of-date; both trigger explicit terminal or rebuild paths rather than silent loops.
- `VkPresent::get_next_frame`/`get_curr_frame_mut` ring semantics depend on counter ordering; changing this can silently desync acquired image binding.

## 7. Debugging Playbook
- Step 1: enable validation layers and capture the first synchronization warning (later warnings are often cascades).
- Step 2: trace warning to exact transition in `prepare_draw_targets`, `copy_draw_to_present`, or upload barrier path.
- Step 3: verify frame-slot sequence in `acquire_frame_slot`, `submit_frame`, and `present_frame`.
- Step 4: if async uploads appear stale, inspect `pump_transfer_submissions` and fence queue completion status.
- Step 5: if resize/present loops occur, inspect `rebuild_swapchain` and acquired present target rebinding (`bind_acquired_present_target`).

## 8. Cross-Module Links
- Frame loop core: `src/renderer/src/vulkan/vk_render.rs` (coordinator)
- Frame lifecycle: `src/renderer/src/vulkan/vk_frame.rs`
- Command recording: `src/renderer/src/vulkan/vk_commands.rs`
- Swapchain lifecycle state machine: `src/renderer/src/vulkan/vk_swapchain.rs`
- Frame-ring and sync types: `src/renderer/src/vulkan/vk_types.rs`
- Descriptor allocators: `src/renderer/src/vulkan/vk_descriptor.rs`
- Barrier helpers and upload transitions: `src/renderer/src/vulkan/vk_util.rs`
- Rendergraph ordering: `src/renderer/src/rendergraph/mod.rs`
- Shadow pass: `src/renderer/src/rendergraph/passes/shadow_pass.rs`, `src/renderer/src/vulkan/vk_shadow.rs`
- Present/UI/capture boundaries: `src/renderer/src/rendergraph/passes/present_copy_pass.rs`, `imgui_pass.rs`, `debug_capture_pass.rs`, `terminal_present_pass.rs`

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
