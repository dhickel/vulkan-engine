# Renderer Internals — API-to-Backend Handoff

> Source: [`src/renderer/src/api/renderer.rs`](../src/renderer/src/api/renderer.rs), [`src/renderer/src/vulkan/vk_render.rs`](../src/renderer/src/vulkan/vk_render.rs) — no legacy docs consulted.

## Frame Lifecycle (Detailed)

The renderer supports 2-3 frames in flight via a `VkPresent` ring buffer. Each frame slot owns:

- A command buffer (from a per-frame command pool)
- A fence (signaled when GPU completes the frame)
- A set of semaphores (image-acquired, render-complete)

### Step 1: Acquire Swapchain Image

```rust
// vk_render.rs — acquire path
let (image_index, suboptimal) = acquire_next_image_with_retry(
    &self.swapchain,
    timeout_ns,
    &self.frame_data[frame_idx].image_available_semaphore,
)?;
```

The acquire uses a timeout-based retry loop. If the swapchain is out of date, it sets a `resize_requested` flag.

### Step 2: Wait on Fence

```rust
// vk_render.rs — fence wait
device.wait_for_fences(&[self.frame_data[frame_idx].fence], true, u64::MAX)?;
device.reset_fences(&[self.frame_data[frame_idx].fence])?;
```

Ensures the GPU is done with this frame slot's resources before reusing them.

### Step 3: Reset Pools

The per-frame command pool and descriptor pool are reset. This invalidates all command buffers and descriptor sets from the previous use of this slot.

### Step 4: Scene → RenderSubmission

```rust
// Called from render_scene_in_frame()
let submission = scene.build_submission(&self.camera, viewport_size);
```

`SceneWorld::build_submission()` at [`scene/scene_world.rs`](../src/renderer/src/scene/scene_world.rs) traverses the scene tree, computes world transforms, and packages flat arrays:
- `Vec<RenderObject>` — per-draw data (mesh handle, material handle, transform)
- `Vec<PointLight>` — active point lights
- `EnvironmentHandle` — active environment map

Frustum culling is enabled by default. Mesh-backed nodes whose transform-aware proxy AABBs are outside the Vulkan `[0, 1]` camera frustum are omitted; descendants are tested independently. The public `Scene` facade can disable culling for diagnostics or compatibility.

### Step 5: Rendergraph Execution

```rust
// vk_render.rs
self.render_graph.execute(
    &self.core,
    &submission,
    frame_idx,
    swapchain_image_view,
    &self.frame_data[frame_idx],
);
```

The default pass order: `PrepareTargetsPass` → `SkyboxPass` → `GeometryPass` → `PresentCopyPass` → `ImguiPass`. Each pass records Vulkan commands into the frame's command buffer.

### Step 6: Submit + Present

```rust
// vk_render.rs
let submit_info = vk::SubmitInfo2::default()
    .command_buffer_infos(&[command_buffer_info])
    .wait_semaphore_infos(&[wait_info])
    .signal_semaphore_infos(&[signal_info]);

queue.submit2(&[submit_info], fence)?;
queue.present_khr(&present_info)?;
```

Uses Vulkan 1.3 `vkQueueSubmit2` with timeline semaphore support. The fence signals when the GPU completes the frame; the render-complete semaphore gates presentation.

### Step 7: Deferred Cleanup

After present, the renderer processes a deletion queue — Vulkan resources marked for destruction are actually destroyed once the fence confirms the GPU is done with them.

## Synchronization Model

| Barrier | Purpose |
|---------|---------|
| Image-acquired semaphore | Swapchain → first usage (color attachment) |
| Render-complete semaphore | Last usage → present |
| Per-frame fence | GPU frame completion → CPU pool reset |

Additional barriers inside passes:
- **Image layout transitions**: `RenderPassNode` trait provides `input_attachment_transitions()` and `output_attachment_transitions()` at [`rendergraph/mod.rs:31`](../src/renderer/src/rendergraph/mod.rs:31)
- **Buffer barriers**: vertex/index upload → vertex shader read
- **Descriptor set updates**: batched via `VkDescriptorWriter` after pool reset

## Swapchain Rebuild

Triggered by `resize()` or `VK_ERROR_OUT_OF_DATE_KHR`:

1. `device.device_wait_idle()` — drain GPU
2. Destroy old swapchain
3. Re-query surface capabilities
4. Create new swapchain with updated extent
5. Rebuild depth buffer and framebuffer attachments

**Resolved**: `VkPresent::replace_present_images` (in `vk_types.rs`) calls `destroy_present_views` before reassignment, so old image views are properly destroyed on swapchain rebuild.

## Render Hooks

Pre-hook fires after command buffer is acquired but before rendergraph execution. Post-hook fires after present but before frame counter advance. Both receive `RenderHookContext { frame_index, viewport_size }` — no direct access to Vulkan resources.

For internal access, the `advanced-interop` feature gate ([`api/advanced.rs`](../src/renderer/src/api/advanced.rs)) exposes `Renderer::raw_core_mut() -> &mut VkRender` — documented as unsafe, internal-use-only.

## See Also

- [03-asset-pipeline.md](03-asset-pipeline.md) — asset loading and GPU upload
- [07-rendergraph.md](07-rendergraph.md) — pass execution order
- [src/renderer/src/vulkan/vk_render.rs](../src/renderer/src/vulkan/vk_render.rs) — frame orchestration (~3862 lines)
