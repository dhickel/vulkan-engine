# Architecture Overview

> Source: module structure at [`src/renderer/src/lib.rs`](../src/renderer/src/lib.rs), subsystem mod files.

## Module Map

| Module | Path | Role | Key Dependencies |
|--------|------|------|-----------------|
| `api` | `src/renderer/src/api/` | Public facade: `Renderer`, `AssetManager`, `Scene`, hooks, config | `data`, `vulkan`, `scene`, `input` |
| `data` | `src/renderer/src/data/` | CPU-side caches, handles, GPU data structs, model/texture ingest | `vulkan::vk_types`, `vulkan::vk_storage`, `vulkan::vk_descriptor` |
| `vulkan` | `src/renderer/src/vulkan/` | Vulkan init, frame loop, descriptors, pipelines, memory | `data::*`, `ash` |
| `scene` | `src/renderer/src/scene/` | Scene graph hierarchy, render submission flattening | `data::handles`, `data::gpu_data` |
| `rendergraph` | `src/renderer/src/rendergraph/` | Pass orchestration (prepare → skybox → geometry → copy → imgui) | `vulkan::vk_render`, `scene` |
| `debug_ui` | `src/renderer/src/debug_ui/` | ImGui debug overlay, timing/spike tracking | `imgui`, `data::handles` |
| `texture` | `src/renderer/src/texture.rs` | **Legacy stub** — single comment line, retained for experimentation | — |
| `input` | `src/input/src/lib.rs` | Layered input dispatch, action maps, polling snapshots | `winit` |

The workspace also includes `apps/dungeon_dogfood/` — a dogfood application that exercises the engine with level loading, collision, and marker-based content.

## Data Flow: Init → Per-Frame → Shutdown

### Initialization

1. **`Renderer::new()`** ([`api/renderer.rs:80`](../src/renderer/src/api/renderer.rs:80)):
   - Creates winit window
   - `VkRenderCore::new()` → Vulkan instance, device, swapchain, allocators, command pools
   - `AssetManager::new()` → handle allocators, default textures (white, blue normal, black, pink error)
   - `InputSystem::new()` → empty layer registry
   - Debug UI (imgui context, font atlas)
   - Optional: loads startup scene (default PBR environment + test geometry)

### Per-Frame

2. **`update_input()`** → feeds winit events to imgui + `InputSystem::queue_winit_window_event()`
3. **`render_scene()`** (or explicit frame trio):
   - Pumps async asset tasks (transfers, load completions)
   - `begin_frame()`: acquires swapchain image, waits on fence, resets command pool
   - `render_scene_in_frame()`:
     - `SceneWorld::build_submission()` → flattens hierarchy into `RenderSubmission`
     - `VkRender::render_with_hooks()`:
       - Runs pre-render hook
       - `RenderGraph::execute()`: each pass records commands
       - Runs post-render hook
     - `vkQueueSubmit2` with fence + semaphores
     - `vkQueuePresentKHR`
   - `end_frame()`: advances frame counter, processes deferred deletions

### Shutdown

4. **Drop `Renderer`**: device wait idle → destroy resources in reverse order. Some Vulkan destroy paths are incomplete (`todo!()` markers in wrapper types).

## Key Design Decisions

- **Vulkan 1.3** with dynamic rendering — no `VkRenderPass` objects
- **Buffer device address** for bindless-style vertex pulling in the geometry pass
- **Slot + generation handles** throughout — safe invalidation for all GPU resource references
- **Descriptor set per-frame ring buffer** — pools reset each frame, not recycled across frames
- **2-3 frames in flight** via `VkPresent` ring buffer with fence-per-frame
- **Async transfer channel** — `VkHostBuffer` + `VkFenceQueue` for background asset uploads

## See Also

- [02-renderer-internals.md](02-renderer-internals.md) — frame lifecycle deep dive
- [src/renderer/AGENTS.md](../src/renderer/AGENTS.md) — renderer package guide
- [src/renderer/src/vulkan/AGENTS.md](../src/renderer/src/vulkan/AGENTS.md) — Vulkan subsystem guide
