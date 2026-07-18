# Renderer Internals — API-to-Backend Handoff

> Source: [`src/renderer/src/api/renderer.rs`](../../src/renderer/src/api/renderer.rs), [`src/renderer/src/vulkan/vk_render.rs`](../../src/renderer/src/vulkan/vk_render.rs) — no legacy docs consulted.

## Frame Lifecycle (Detailed)

The renderer supports 2-3 frames in flight via a `VkPresent` ring buffer. Each frame slot owns:

- A command buffer (from a per-frame command pool)
- A fence (signaled when GPU completes the frame)
- A set of semaphores (image-acquired, render-complete)

### Step 1: Wait and Clean the Frame Slot

The backend waits on the selected slot's render fence before reusing descriptor pools, timing queries, command buffers, or deferred resources. Fence waits classify `VK_ERROR_DEVICE_LOST` instead of panicking.

### Step 2: Acquire, Bind, Then Reset the Fence

Windowed rendering acquires with a bounded retry loop and binds the acquired swapchain image to the current frame slot. Headless rendering selects that slot's offscreen present image. Out-of-date/retry paths rewind the ring and leave the already-signaled fence unchanged.

Only after acquisition/binding succeeds does the backend reset the fence. From that point, normal recording or a failure drain submission must queue a signal for it.

### Step 3: Record-or-Drain Transaction

The command buffer records the fixed rendergraph. If recording fails after acquisition, partial commands are replaced with a drain recording; windowed mode submits and presents that drain so the acquire semaphore, render semaphore, and acquired image are retired as well as the fence.

### Step 4: Scene → RenderSubmission

```rust
// Called from render_scene_in_frame()
let submission = scene.build_submission(&self.camera, viewport_size);
```

`SceneWorld::build_submission()` at [`scene/scene_world.rs`](../../src/renderer/src/scene/scene_world.rs) traverses the scene tree, computes world transforms, and packages flat arrays:
- `Vec<FrameDrawItem>` — per-draw mesh handle and world transform
- `Vec<FramePointLight>` — active point lights
- `Option<FrameDirectionalLight>` — the scene's single surface-to-light directional source
- `EnvironmentHandle` — active environment map

Frustum culling is enabled by default. Mesh-backed nodes whose transform-aware proxy AABBs are outside the Vulkan `[0, 1]` camera frustum are omitted; descendants are tested independently. The public `Scene` facade can disable culling for diagnostics or compatibility.

### Step 5: Rendergraph Execution

```rust
// src/renderer/src/rendergraph/mod.rs
let mut context = RenderGraphContext {
    submission,
    frame,
    renderer: core,
};
let report = rendergraph.execute(&mut context)?;
```

The default pass order: `PrepareTargetsPass` → `ShadowPass` → `SkyboxPass` → `GeometryPass` → `PresentCopyPass` → `ImguiPass` → `DebugCapturePass` → `TerminalPresentPass`. `ShadowPass` clears and records the current frame slot's 2048² D32 map, then transitions it to shader-read layout before PBR geometry samples scene descriptor set 0 binding 5.

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

Uses Vulkan 1.3 `vkQueueSubmit2` with per-frame binary acquire/render semaphores in windowed mode. The fence signals when the GPU completes the frame; the render-complete semaphore gates presentation. Headless submissions omit the binary semaphores but still signal the fence.

### Step 7: Frame Outcome and Deferred Cleanup

Presentation preserves not-presented and suboptimal outcomes for the facade. Deferred frame-local cleanup occurs the next time the slot is selected, after its fence has signaled.

## Synchronization Model

| Barrier | Purpose |
|---------|---------|
| Image-acquired semaphore | Swapchain → first usage (color attachment) |
| Render-complete semaphore | Last usage → present |
| Per-frame fence | GPU frame completion → CPU pool reset |

Additional synchronization inside passes:
- **Image layout transitions**: each pass records its own explicit transition; the current `RenderPassNode` trait does not declare resources or derive barriers.
- **Directional shadow dependency**: `ShadowPass` transitions D32 writes to fragment shader reads before `GeometryPass`.
- **Buffer barriers**: upload paths synchronize transfer writes with graphics reads.
- **Descriptor set updates**: batched via `VkDescriptorWriter` after pool reset.

## Swapchain Rebuild

Triggered by `resize()` or `VK_ERROR_OUT_OF_DATE_KHR`:

1. `device.device_wait_idle()` — drain GPU
2. Destroy old swapchain
3. Re-query surface capabilities
4. Create new swapchain with updated extent
5. Rebuild depth buffer and framebuffer attachments

**Resolved**: `VkPresent::replace_present_images` (in `vk_types.rs`) calls `destroy_present_views` before reassignment, so old image views are properly destroyed on swapchain rebuild.

## Render Hooks

Pre-hook fires after command buffer is acquired but before rendergraph execution. The pre-hook fires before rendergraph recording and the post-hook fires after rendergraph recording but before command-buffer end/submit. Both receive `RenderHookContext { frame_index, viewport_size }` and no direct Vulkan resources.

For internal access, the `advanced-interop` feature gate (`src/renderer/src/api/advanced.rs`) exposes unsafe `renderer::api::advanced::renderer_core_mut(&mut Renderer) -> &mut VkRenderCore` for expert diagnostics.

## See Also

- [03-asset-pipeline.md](03-asset-pipeline.md) — asset loading and GPU upload
- [07-rendergraph.md](07-rendergraph.md) — pass execution order
- [src/renderer/src/vulkan/vk_render.rs](../../src/renderer/src/vulkan/vk_render.rs) — frame transactions and backend orchestration
