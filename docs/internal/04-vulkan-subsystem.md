# Vulkan Subsystem

> Source: [`src/renderer/src/vulkan/`](../../src/renderer/src/vulkan/) — no legacy docs consulted.

## Module Layout

| File | Role | Lines |
|------|------|-------|
| `mod.rs` | Module declarations | — |
| `vk_init.rs` | Instance, device, swapchain, queues, allocators, surface | — |
| `vk_render.rs` | Frame transactions, `VkRenderCore`, rendergraph execution, submit/present | — |
| `vk_shadow.rs` | Per-frame D32 directional shadow resources and light-volume fitting | — |
| `vk_descriptor.rs` | Descriptor layout builder, dynamic allocator, writer | — |
| `vk_pipeline.rs` | Geometry/skybox/environment/shadow pipeline creation | — |
| `vk_storage.rs` | Buffer/image allocation, sub-allocation, staging | — |
| `vk_types.rs` | Vulkan ownership wrappers and scene descriptor sets | — |
| `vk_util.rs` | Command buffer, allocation, upload, and barrier helpers | — |
| `vk_debug.rs` | Validation callback support | — |

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

Validation is controlled by `RendererConfig::validation_layer`, not build profile. When enabled, the renderer requests `VK_LAYER_KHRONOS_validation` and a debug-utils callback. The opt-in ignored GPU smoke and validation-enabled dogfood smoke exercise this path; CPU CI does not require a GPU.

## Frame Orchestration (`vk_render.rs`)

### `VkRenderCore`

Owns all Vulkan state: device, queues, swapchain, allocators, command pools, descriptor pools, sync primitives. Created once at init; lives for the `Renderer` lifetime.

### `VkRender`

Wraps `VkRenderCore` with the default `RenderGraph` and a shared backend-health state. Facade operations enter through an unwind-aware guard. The first classified device-loss operation maps to `RendererError::DeviceLost`; later backend operations after terminal failure or unwind map to `BackendPoisoned`.

The frame path returns a typed internal `VkFrameRenderOutcome`, preserving rendered, resize-skipped, submitted-not-presented, and presented-suboptimal states for facade mapping.

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

Active draw layouts are kept distinct:
1. **Scene set (set 0)** — camera UBO, environment/light UBO, irradiance/prefilter/BRDF samplers, and per-frame comparison shadow sampler at binding 5.
2. **Skin set (set 1)** — joint-matrix UBO.
3. **Material set (set 2)** — PBR texture samplers; material metadata is supplied through the draw push-constant record.

`VkSceneDescriptors` allocates one scene set and aligned scene/environment UBO slots per frame. Each set references the shadow image owned by the same frame slot.

## Pipeline Management (`vk_pipeline.rs`)

Pipeline creation is shader-driven. Geometry uses buffer-device-address vertex pulling, scene/skin/material descriptor layouts, dynamic rendering, depth testing, and material push constants. The directional shadow pipeline is depth-only, uses the D32 attachment format, and receives an 80-byte light-model-view-projection plus vertex-address push-constant block.

## Memory Allocation (`vk_storage.rs`)

Uses `VkSubAllocator` — a bump allocator within larger `VkBuffer`/`VkImage` allocations:
- Vertex/index buffers sub-allocated from shared GPU-local buffers
- Material SSBO sub-allocated from a device-local buffer
- Staging buffers use `VkHostBuffer` (host-visible, coherent)

The sub-allocator grows by allocating new backing `VkBuffer`s when full. Individual sub-allocations are not freed until the parent buffer is destroyed (arena-style).

## Known Sharp Edges

- **Frame transaction boundary**: after fence reset, every recording failure must record and submit a drain; windowed acquisition also has to retire the acquired image and both binary semaphores.
- **Terminal backend policy**: device loss is classified but not recovered. Hosts must recreate the renderer after `DeviceLost` or `BackendPoisoned`.
- **Shadow ABI**: scene binding 5, the 656-byte Rust/GLSL `EnvironmentUBO`, D32 layout transitions, and the fixed 3×3 PCF comparison contract must change together.
- **Material lifetime**: `RenderObject` owns a copied material draw record. Reintroducing cache-owned raw pointers would make cache mutation a frame lifetime hazard.
- **Push constant portability**: skybox push constants exceed Vulkan's 128-byte minimum guarantee; current target desktop devices expose sufficient capacity, but this remains a portability caveat.

## See Also

- [05-vulkan-sync-and-frame-lifecycle.md](05-vulkan-sync-and-frame-lifecycle.md) — frame lifecycle
- [06-data-suballocation-and-transfer.md](06-data-suballocation-and-transfer.md) — GPU upload and transfer
- [src/renderer/src/vulkan/AGENTS.md](../../src/renderer/src/vulkan/AGENTS.md) — Vulkan contributor guide
