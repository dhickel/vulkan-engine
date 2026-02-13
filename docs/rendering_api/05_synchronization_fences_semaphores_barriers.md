# 05 - Synchronization: Fences, Semaphores, and Barriers

## Why synchronization is central in this renderer

This engine overlaps CPU and GPU work using frames in flight. Correctness requires explicit sync.

Current building blocks:
- Fences for CPU reuse safety.
- Semaphores for GPU queue ordering.
- Image barriers for layout and memory visibility transitions.

Best practice:
- Assume sync first when debugging flicker, frame stalls, or transient corruption.

Learn more:
- Vulkan synchronization guide: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html

## Fence contract (per-frame)

Frame lifecycle contract:
1. Wait for frame fence.
2. Reset frame fence.
3. Reuse per-frame command/descriptors/deferred deletions.

Code example (in-tree/internal):
```rust
let fence = [frame_sync.render_fence];
self.device.wait_for_fences(&fence, true, u64::MAX)?;
self.device.reset_fences(&fence)?;
```

Best practice:
- Never reset/recycle per-frame resources before fence wait completes.

Learn more:
- Frame sync type: `src/renderer/src/vulkan/vk_types.rs` (`VkFrameSync`)

## Semaphore contract (acquire -> render -> present)

Current chain:
- `swap_semaphore`: acquire signals, render submit waits.
- `render_semaphore`: render submit signals, present waits.

Code example (in-tree/internal shape):
```rust
let wait_info = [vk_util::semaphore_submit_info(vk::PipelineStageFlags2::ALL_COMMANDS, frame_sync.swap_semaphore)];
let signal_info = [vk_util::semaphore_submit_info(vk::PipelineStageFlags2::ALL_GRAPHICS, frame_sync.render_semaphore)];
let submit = [vk_util::submit_info_2(&cmd_info, &signal_info, &wait_info)];
```

Best practice:
- Keep semaphore ownership frame-local; cross-frame mixing is a common source of intermittent issues.

Learn more:
- Synchronization examples: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html

## Image barriers/layout transitions contract

Typical per-frame transitions:
- draw image: `UNDEFINED -> GENERAL -> COLOR_ATTACHMENT_OPTIMAL`
- depth image: `UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL`
- copy path: `COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL`
- present image for copy: `UNDEFINED -> TRANSFER_DST_OPTIMAL`
- present image for UI: `TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL`
- final present handoff: `COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR`

Code example (in-tree/internal):
```rust
vk_util::transition_image(device, cmd, frame.draw.image,
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
);
```

Best practice:
- Document transition intent when editing pass flow; layout names alone are not enough context.

Learn more:
- Layout transitions: https://github.khronos.org/Vulkan-Site/guide/latest/layout_transitions.html

## Deferred loading pipes/channels

Current deferred-loading system is internal:
- `VkHostBuffer` records transfer/graphics command buffers.
- `VkCmdSubmitInfo` is sent over channel (`VkTransfer`).
- Render thread drains and submits.
- `VkFenceQueue` polls completion and releases latches.

Code example (in-tree/internal):
```rust
host_buffer.submit_transfer_commands(VkSubmitParam::signaling(vk::PipelineStageFlags2::ALL_TRANSFER))?;
host_buffer.submit_graphics_commands(VkSubmitParam::waiting(vk::PipelineStageFlags2::VERTEX_SHADER))?;
```

Can users touch this today?
- External API users: no (not exposed as stable public API surface).
- Engine contributors/fork users: yes (by editing in-tree runtime/renderer internals).

Best practice:
- Keep all queue submits centralized on render thread unless you redesign queue ownership/synchronization model end-to-end.

Learn more:
- Internal transfer types: `src/renderer/src/vulkan/vk_types.rs` (`VkHostBuffer`, `VkTransfer`, `VkFenceQueue`)

## Practical debug checklist for sync regressions

1. Validate fence wait/reset order.
2. Verify acquire/present semaphore pairing.
3. Verify image transitions for each optional pass-flag combination.
4. Re-test resize + environment switching.

Best practice:
- Fix first validation-layer sync error before touching multiple areas; cascading errors are common.

Learn more:
- Khronos validation docs: https://github.khronos.org/Vulkan-Site/guide/latest/validation_overview.html
