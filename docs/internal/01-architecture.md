# Architecture Overview

> Source: module structure at [`src/renderer/src/lib.rs`](../../src/renderer/src/lib.rs), subsystem mod files.

## Module Map

| Module | Path | Role | Key Dependencies |
|--------|------|------|-----------------|
| root `engine` facade | `src/` | App-owned runtime helpers and data-driven launcher: camera, events, frame, input, render reexports | `renderer`, `input`, `engine_events` |
| `api` | `src/renderer/src/api/` | Public renderer facade: `Renderer`, `AssetManager`, `Scene`, hooks, config | `data`, `vulkan`, `scene`, `input` |
| `data` | `src/renderer/src/data/` | CPU-side caches, handles, GPU data structs, model/texture ingest | `vulkan::vk_types`, `vulkan::vk_storage`, `vulkan::vk_descriptor` |
| `vulkan` | `src/renderer/src/vulkan/` | Vulkan init, frame transactions, directional shadow resources, descriptors, pipelines, memory | `data::*`, `ash` |
| `scene` | `src/renderer/src/scene/` | Scene graph hierarchy, render submission flattening | `data::handles`, `data::gpu_data` |
| `rendergraph` | `src/renderer/src/rendergraph/` | Fixed pass orchestration (prepare → shadow → skybox → geometry → copy → imgui → capture → terminal present) | `vulkan::vk_render`, `scene` |
| `debug_ui` | `src/renderer/src/debug_ui/` | ImGui debug overlay, timing/spike tracking | `imgui`, `data::handles` |
| `texture` | `src/renderer/src/texture.rs` | **Legacy stub** — single comment line, retained for experimentation | — |
| `input` | `src/input/src/lib.rs` | Layered input dispatch, action maps, polling snapshots | `winit` |

The workspace also includes `apps/dungeon_dogfood/` — a dogfood application that exercises the engine with level loading, collision, marker-based content, app-owned runtime primitives, and caller-view rendering.

## Data Flow: Init → Per-Frame → Shutdown

### Initialization

1. **`Renderer::new()`** (`src/renderer/src/api/renderer.rs`):
   - Consumes an app-owned winit window reference; `Renderer::new_headless()` uses offscreen present images
   - `VkRenderCore::new()` → Vulkan instance, device, swapchain/offscreen targets, allocators, command pools, and frame-local shadow maps
   - `AssetManager::new()` → handle allocators, default textures (white, blue normal, black, pink error)
   - `InputSystem::new()` → empty renderer-owned layer registry for compatibility/example paths
   - Debug UI (imgui context, font atlas)
   - Optional: loads startup scene (default PBR environment + test geometry)

### Per-Frame

2. **Input routing**:
   - Compatibility renderer-owned path: `update_input()` feeds winit events to imgui + renderer-owned `InputSystem`
   - App-owned path: `route_platform_input()` handles platform/UI side effects and returns routing data for app-owned `InputSystem`
3. **`render_scene()`** (or explicit frame trio):
   - Pumps async asset tasks (transfers, load completions)
   - `begin_frame()`: acquires swapchain image, waits on fence, resets command pool
   - `render_scene_in_frame()`:
     - `SceneWorld::build_submission()` → flattens hierarchy into `RenderSubmission`
     - `VkRender::render_with_hooks()`:
       - Runs pre-render hook
       - `RenderGraph::execute()`: prepares targets, records the directional shadow map, draws skybox/geometry/UI, captures due frames, and performs the terminal present transition
       - Runs post-render hook
     - `vkQueueSubmit2` with fence + semaphores
     - `vkQueuePresentKHR`; a recording failure after acquire records and submits a drain transaction so the fence/semaphores/image remain retireable
   - `end_frame()`: advances frame counter, processes deferred deletions

App-owned custom loops can instead build a `CameraView` from app camera/gameplay state and call
`render_scene_with_view(...)` or `render_scene_headless_with_view(...)`. That path avoids renderer-owned input/camera/lifecycle dispatch for app state while keeping Vulkan submission inside the renderer.

### Shutdown

4. **Drop `Renderer`**: device wait idle → stop worker threads → destroy frame-local shadow and presentation resources, descriptors/caches, command pools/synchronization, and allocator/device objects in ownership order. Teardown-sensitive validation is exercised by the isolated ignored GPU smoke test.

## Key Design Decisions

- **Vulkan 1.3** with dynamic rendering — no `VkRenderPass` objects
- **Buffer device address** for bindless-style vertex pulling in the geometry pass
- **Slot + generation handles** throughout — safe invalidation for all GPU resource references
- **Descriptor set per-frame ring buffer** — pools reset each frame, not recycled across frames
- **2-3 frames in flight** via `VkPresent` ring buffer with fence-per-frame
- **Async transfer channel** — `VkHostBuffer` + `VkFenceQueue` for background asset uploads

## See Also

- [02-renderer-internals.md](02-renderer-internals.md) — frame lifecycle deep dive
- [src/renderer/AGENTS.md](../../src/renderer/AGENTS.md) — renderer package guide
- [src/renderer/src/vulkan/AGENTS.md](../../src/renderer/src/vulkan/AGENTS.md) — Vulkan subsystem guide
