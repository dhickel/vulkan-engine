# Vulkan Subsystem

> Source: [`src/renderer/src/vulkan/`](../../src/renderer/src/vulkan/) — no legacy docs consulted.

## Module Layout

| File | Role | Lines |
|------|------|-------|
| `mod.rs` | Module declarations | — |
| `vk_init.rs` | Instance, device, swapchain, queues, allocators, surface | — |
| `vk_render.rs` | Frame orchestration, `VkRenderCore`, `VkRender` | ~3862 |
| `vk_descriptor.rs` | Descriptor layout builder, dynamic allocator, writer | — |
| `vk_pipeline.rs` | Pipeline creation, shader stages, pipeline layout | — |
| `vk_storage.rs` | Buffer/image allocation, sub-allocation, staging | — |
| `vk_types.rs` | Wrapper types for Vulkan handles | — |
| `vk_util.rs` | Command buffer helpers, barrier utilities | — |
| `vk_debug.rs` | Validation layer setup, debug utils | — |

## Initialization Sequence

`vk_init.rs` documents the init order at its header:

```
Entry → Instance → Surface → Physical Device → Logical Device →
Queues → Swapchain → Allocators → Command Pools → Sync Primitives
```

### Device Features

Requested at [`vk_init.rs:33-45`](../../src/renderer/src/vulkan/vk_init.rs#L33-L45):

| Feature | Purpose |
|---------|---------|
| `bufferDeviceAddress` | Bindless vertex pulling in geometry pass |
| `dynamicRendering` | No `VkRenderPass` — direct dynamic rendering |
| `synchronization2` | `vkQueueSubmit2`, pipeline barriers 2 |
| `descriptorIndexing` | Partially bound descriptors, non-uniform indexing |
| `scalarBlockLayout` | Tighter SSBO packing |

### Queue Selection

`queue_indices_with_preferences()` prefers dedicated transfer and compute queues, falling back to the graphics queue. The transfer queue is used for async asset uploads via `VkTransfer` channel.

### Validation

Debug builds enable `VK_LAYER_KHRONOS_validation` with `VK_EXT_debug_utils` callback. Production builds (or `validation_layer: false`) skip validation layers entirely.

## Frame Orchestration (`vk_render.rs`)

### `VkRenderCore`

Owns all Vulkan state: device, queues, swapchain, allocators, command pools, descriptor pools, sync primitives. Created once at init; lives for the `Renderer` lifetime.

### `VkRender`

Wraps `VkRenderCore` with a `RenderGraph`. This is the type that `api/renderer.rs` interacts with:

```rust
pub fn render_with_hooks(
    &mut self,
    submission: &RenderSubmission,
    swapchain_image_view: vk::ImageView,
    pre_hook: Option<&mut RenderHook>,
    post_hook: Option<&mut RenderHook>,
    frame_index: u64,
    viewport_size: (u32, u32),
) -> Result<(), RendererError>
```

### Frame Ring Buffer

`VkPresent` struct manages frame overlap. Configurable 2-3 frames in flight. Each slot has:

```
FrameData {
    command_pool: VkCommandPool,
    command_buffer: VkCommandBuffer,
    fence: VkFence,
    image_available_semaphore: VkSemaphore,
    render_finished_semaphore: VkSemaphore,
}
```

### Async Transfers

`VkHostBuffer` — CPU-visible staging buffer for uploads.
`VkTransfer` — background transfer channel with its own command pool.
`VkFenceQueue` — tracks transfer completion, polled each frame.

## Descriptor System (`vk_descriptor.rs`)

**Not bindless** — uses traditional descriptor sets allocated per frame.

| Component | Role |
|-----------|------|
| `DescriptorLayoutBuilder` | Fluent builder for `VkDescriptorSetLayout` — add bindings, build layout |
| `VkDynamicDescriptorAllocator` | Auto-growing pool allocator. 1.5x growth factor, capped at 4092 sets per pool. Ready/Full pool separation. Reset per frame |
| `VkDescriptorWriter` | Batched image/buffer descriptor updates. Writes to a `Vec<VkWriteDescriptorSet>` and flushes with `update_descriptor_sets` |

### Descriptor Set Layouts

Three main layouts:
1. **Material** — combined image samplers for PBR textures + uniform buffer for material params
2. **Environment** — combined image sampler for skybox + uniform buffer for `EnvironmentUBO`
3. **Per-frame** — uniform buffer for camera matrices, SSBO for joint transforms

## Pipeline Management (`vk_pipeline.rs`)

Pipeline creation is shader-driven. Each shader pair (`.vert` + `.frag` SPIR-V) defines a pipeline with:
- Vertex input state (matches `Vertex` struct at [`gpu_data.rs:36`](../../src/renderer/src/data/gpu_data.rs#L36))
- Dynamic rendering state (color + depth attachment formats)
- Depth/stencil state (depth test enabled, depth write enabled, `LESS` compare op)
- Pipeline layout (derived from descriptor set layouts + push constant ranges)

## Memory Allocation (`vk_storage.rs`)

Uses `VkSubAllocator` — a bump allocator within larger `VkBuffer`/`VkImage` allocations:
- Vertex/index buffers sub-allocated from shared GPU-local buffers
- Material SSBO sub-allocated from a device-local buffer
- Staging buffers use `VkHostBuffer` (host-visible, coherent)

The sub-allocator grows by allocating new backing `VkBuffer`s when full. Individual sub-allocations are not freed until the parent buffer is destroyed (arena-style).

## Known Sharp Edges

- **Swapchain image view leak**: old views not destroyed on resize ([`vk_render.rs:1154`](../../src/renderer/src/vulkan/vk_render.rs#L1154))
- **`todo!()` destroy paths**: several Vulkan wrapper types have incomplete `Drop` implementations
- **`unwrap()` in production paths**: descriptor allocation, command buffer recording, and window handle extraction use `unwrap()` where errors should propagate; this remains a residual candidate for alpha readiness classification in the [alpha readiness baseline](../gap-report.md).
- **Push constant size**: skybox push constants may exceed 128-byte minimum (`// FIXME` at [`gpu_data.rs:628`](../../src/renderer/src/data/gpu_data.rs#L628))

## See Also

- [02-renderer-internals.md](02-renderer-internals.md) — frame lifecycle
- [03-asset-pipeline.md](03-asset-pipeline.md) — GPU upload and transfer
- [src/renderer/src/vulkan/AGENTS.md](../../src/renderer/src/vulkan/AGENTS.md) — Vulkan contributor guide
