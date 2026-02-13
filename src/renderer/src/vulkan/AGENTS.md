# Vulkan Module Agent Guide (`src/renderer/src/vulkan`)

This is the deep guide for Vulkan lifecycle, synchronization, resource ownership, and frame execution.

## Module Map

- `vk_init.rs`: Vulkan setup (instance/device/queues/swapchain/features).
- `vk_types.rs`: foundational structs and ownership traits (`VkDestroyable`, frame types, transfer types).
- `vk_render.rs`: main renderer orchestration and frame loop.
- `vk_storage.rs`: custom sub-allocator for packed buffers.
- `vk_descriptor.rs`: descriptor layout/writer/pool allocators.
- `vk_pipeline.rs`: graphics pipeline construction and cache init.
- `vk_util.rs`: utility creation and command-record helper functions.
- `vk_debug.rs`: image capture/debug helpers.

## Core Architecture

### Startup

`VkRender::new(...)` performs:
1. entry/instance/surface creation
2. physical device selection + queue family planning
3. logical device creation with Vulkan 1.1/1.2/1.3 features where supported
4. swapchain and per-frame resource construction
5. allocator + caches + pipelines + descriptor layouts
6. transfer subsystem + host staging buffers
7. scene/model preload and initial allocations (returned to runtime as `SceneWorld`)

### Frame Model

`VkPresent` owns `Vec<VkFrame>` as ring-buffered frame resources.
Each `VkFrame` contains:
- sync primitives (`VkFrameSync`)
- draw/depth images
- references to present image/view
- command pools
- dynamic descriptor allocator
- deferred deletion queue

### Render Loop

`VkRender::render(...)` performs:
1. async transfer fence polling and queued submit handling
2. consumes immutable `RenderSubmission` from runtime
3. wait/reset frame fence
4. acquire swapchain image
5. reset and begin command buffer
6. execute rendergraph pass list over command buffer
7. transition to present and submit
8. submit and present

## PBR and IBL Entry Points

If working specifically on radiance/PBR behavior, start here:

- `src/renderer/src/vulkan/vk_render.rs`
  - `generate_environment(...)`
  - `draw_skybox_from_submission(...)`
  - `draw_geometry_from_submission(...)`
  - `copy_draw_to_present(...)`
  - `draw_imgui_to_present(...)`
- `src/renderer/src/rendergraph/*`
  - pass execution order and pass node wiring
- `src/renderer/src/vulkan/vk_pipeline.rs`
  - `init_met_rough_pipelines(...)`
  - `init_irradiance_pipeline(...)`
  - `init_pre_filter_pipeline(...)`
  - `init_brd_flut_pipeline(...)`
- `src/renderer/src/vulkan/vk_descriptor.rs`
  - `init_descriptor_cache(...)`

Shader files are mapped in `src/renderer/src/shaders/AGENTS.md`.
Core shader manifest path mapping for cache init lives in `src/renderer/src/shaders/core_shader_manifest.txt`.

External lineage reference:
- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

## Descriptor and Pipeline Conventions

- Descriptor strategy is traditional sets, not bindless indexing architecture.
- Material draw path expects pipeline layouts with set order:
- set 0: scene descriptors
- set 1: skin/joint descriptors
- set 2: material samplers
- Push constants carry model matrix, vertex buffer address, material metadata address.

## Memory and Allocation

### `vk_mem` + `VkSubAllocator`

- Images and some standalone buffers use `vk_mem` directly.
- Geometry/material metadata use `VkSubAllocator` (`vk_storage.rs`).
- Sub-allocator uses bump tail + free list with coalescing.
- Allocation placement policies: `ContiguousPreferred`, `ContiguousOnly`, `EndOnly`.

### Async Transfer

- `VkHostBuffer` records transfer/graphics command buffers and sends submit info over channel.
- `VkTransfer` receiver is polled from render thread.
- `VkFenceQueue` checks completion and resolves latches.

## Current Gotchas and Risks

1. Incomplete destroy coverage.
- `impl VkDestroyable for VkSubAllocator` currently `todo!()`.

2. Geometry pass ordering and descriptor binding correctness risk.
- `draw_geometry_from_submission` resolves handle-based submission draws into internal pipeline buckets.
- Small changes to bucket ordering can break alpha behavior.

3. Swapchain rebuild cleanup concern.
- `rebuild_swapchain` has FIXME about old present image views and lifecycle.

4. Queue usage assumptions.
- Render path submits/presents on graphics queue path; queue family strategy currently seeks shared graphics/present, but this should be revisited if queue policy changes.

5. `VkWindowState::update_window_scale` is marked broken.
- It mutates current extent directly and is flagged FIXME.

6. Descriptor allocator sizing and reset behavior are dynamic.
- Be careful when changing per-frame clear timing or pool growth policy.

7. Heavy unwrap/panic patterns in Vulkan hot paths.
- Harden with structured error propagation cautiously to avoid masking original API errors.

## File-by-File Focus

### `vk_render.rs`

- Highest complexity and highest blast radius file.
- Touch here only with clear validation and smoke-test strategy.
- Keep command ordering and transitions explicit.

### `vk_types.rs`

- Defines lifetime and destruction model contracts.
- If changing frame ownership, update both render loop and destroy paths.

### `vk_storage.rs`

- Allocation correctness depends on address math and chunk bookkeeping.
- Add assertions/tests before changing chunk selection or coalescing behavior.

### `vk_descriptor.rs`

- Dynamic allocator behavior controls frame memory pressure.
- Keep layout binding order in sync with shaders and draw code.

### `vk_pipeline.rs`

- Pipeline layout set order must match descriptor binding in `draw_geometry`.
- Vertex input is empty because shaders fetch vertex data by address.

## Validation and Debugging Workflow

- First line check: `cargo check -p renderer`.
- Runtime issues:
- enable validation layers path through `VkRender::new(..., with_validation=true, ...)`.
- inspect descriptor layout and pipeline layout consistency.
- verify image layout transitions around draw/copy/present.

## Recommended Hardening Queue

1. Fix pipeline-switch bind bug in `draw_geometry`.
2. Implement missing destroy paths (`VkSubAllocator`, dependent cache destroy flows).
3. Replace risky cache ID-shifting operations in data layer to protect render-time pointers.
4. Add targeted tests or debug assertions for pipeline enum indexing and frame-ring invariants.

## Related Files

- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/vulkan/vk_storage.rs`
- `src/renderer/src/vulkan/vk_descriptor.rs`
- `src/renderer/src/data/data_cache.rs`
