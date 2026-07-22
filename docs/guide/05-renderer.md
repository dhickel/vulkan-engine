# 05 — Working with the Renderer

> Provenance: `CP-05`

This chapter covers the renderer's public API surface from the app's perspective: initialization, the startup scene, resize handling, asset pumping, frame outcomes, and terminal errors. For complete API reference (every function signature, every error variant, every config field), see the [Renderer Lifecycle & Frame API](../api/02-renderer-lifecycle-and-frame-api.md).

## Renderer Initialization

> Provenance: `CP-05-INIT` — Excerpt (setup code omitted; see Chapter 04 for full context)

```rust
    let config = RendererConfig {
        app_name: "guide_app".to_string(),
        window_width: 1280,
        window_height: 720,
        preload_startup_scene: true,
        ..Default::default()
    };
    let mut renderer = Renderer::new(config, &window)?;
```

`Renderer::new` performs Vulkan instance creation, physical device selection, logical device creation, swapchain setup, pipeline compilation, descriptor pool allocation, and (if `preload_startup_scene: true`) startup scene loading. This takes 5–30 seconds depending on GPU, driver, and shader compilation cache.

### RendererConfig Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `app_name` | `String` | `"engine"` | Used in debug output, window title (compatibility path), and capture directory names |
| `window_width` | `u32` | `1920` | Initial framebuffer width. Must match the window size or a resize will follow. |
| `window_height` | `u32` | `1080` | Initial framebuffer height. |
| `validation_layer` | `bool` | `false` | Enables `VK_LAYER_KHRONOS_validation`. Set to `true` for debugging Vulkan issues; significantly reduces performance. |
| `shader_debug_mode` | `DebugRuntimeMode` | `Default` | Selects shader debug variants (`Default`, `TestPbr`, `TestUnlit`). Leave at `Default` for normal use. |
| `compile_shaders` | `bool` | `false` | Compiles GLSL to SPIR-V at runtime. Leave `false` — the renderer ships pre-compiled `.spv` files. |
| `preload_startup_scene` | `bool` | `true` | When `true`, loads the built-in startup scene during `Renderer::new`. Disable to reduce init time when building your own scene from scratch. |
| `visual_tuning` | `VisualTuning` | `exposure: 4.5, gamma: 2.2, ibl_ambient_scale: 1.0` | App-controlled tonemapping and IBL parameters. See [`VisualTuning`](../api/02-renderer-lifecycle-and-frame-api.md) for details. |
| `headless` | `bool` | `false` | Reserved for headless operation. Use `Renderer::new_headless` instead. |
| `asset_policy` | `AssetPolicyConfig` | Best-effort manifest mode, heuristics enabled, no compression | Controls how assets are discovered and loaded. See [Asset Loading](../api/04-assets-sync-deferred-and-handles.md). |

> The full `RendererConfig` documentation is at [`docs/api/02-renderer-lifecycle-and-frame-api.md`](../api/02-renderer-lifecycle-and-frame-api.md) and the source at [`src/renderer/src/api/config.rs`](../../src/renderer/src/api/config.rs).

## The Startup Scene

```rust
    let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);
```

When `preload_startup_scene` is `true`, the renderer loads a built-in scene during initialization. `take_startup_scene()` extracts it — after this call, the renderer no longer holds a reference to the scene. If preloading was disabled, or if `take_startup_scene()` was already called, it returns `None`.

The startup scene contains:

- A skybox (default environment map)
- Several PBR material sample meshes (spheres, cubes)
- A directional light with shadow mapping
- A ground plane
- Default camera position

Your app is free to modify the scene after taking it — add/remove nodes, change materials, add lights, import fragments. The startup scene is a convenient default, not a required baseline.

> **Note for guide readers**: The startup scene's exact contents are not a stable API contract. Future engine versions may change the default meshes, materials, or layout. For reproducible scenes, build your own programmatically or load from asset files.

## Resize Handling

The app is responsible for detecting window size changes and calling `renderer.resize()`:

```rust
                        // Catch up to the current window size before rendering.
                        let current_size = window.inner_size();
                        if current_size != last_window_size {
                            last_window_size = current_size;
                            if let Err(e) = renderer.resize(current_size.width, current_size.height)
                            {
                                eprintln!("resize failed while redrawing: {e}");
                                elwt.exit();
                                return;
                            }
                        }
```

`Renderer::resize(width, height)` does the following:

1. If the requested extent matches the current swapchain extent, returns `Ok(())` immediately (no-op).
2. If a resize was already requested but not yet applied, it is coalesced — only the latest request matters.
3. Sets a "resize pending" flag. The actual swapchain rebuild happens inside the next `render_scene_with_view` call.
4. If `width == 0 || height == 0` (e.g. window minimized), the resize request is recorded but the swapchain is not immediately rebuilt. Rendering will skip with `SkippedResizePending` until the window has positive dimensions again.

The checkpoint catches up to the current size before every redraw. The platform may deliver `Resized` events in any order relative to `RedrawRequested`, so the guard `if current_size != last_window_size` prevents stale-size renders.

### Scale Factor Changes

On high-DPI systems, the window's scale factor can change (e.g. moving the window from a 1× to a 2× display). The `ScaleFactorChanged` event provides an `inner_size_writer` that must be used to acknowledge the change before the new physical size can be applied:

```rust
                    WindowEvent::ScaleFactorChanged {
                        mut inner_size_writer,
                        ..
                    } => {
                        let new_size = window.inner_size();
                        if let Err(e) = inner_size_writer.request_inner_size(new_size) {
                            eprintln!("scale-factor size request failed: {e}");
                            elwt.exit();
                            return;
                        }
                        last_window_size = new_size;
                        if let Err(e) = renderer.resize(new_size.width, new_size.height) {
                            eprintln!("resize failed after scale change: {e}");
                            elwt.exit();
                        }
                    }
```

## Asset Pumping

```rust
                        if let Err(e) = renderer.pump_asset_tasks(32) {
                            eprintln!("asset pump failed: {e}");
                            elwt.exit();
                            return;
                        }
```

`pump_asset_tasks(max_steps)` drives the renderer's async asset pipeline. It:

- Collects completed background tasks (texture uploads, mesh processing, environment map decoding)
- Advances load state machines (pending → uploading → ready)
- Drains retirement queues for GPU resources whose fence has signaled
- Returns the number of tasks processed, or an error if the asset system is in an unrecoverable state

**Call this every frame before `render_scene_with_view`.** Without it, async asset loads never complete and GPU resources are never reclaimed. The `max_steps` parameter (32 in the checkpoint) limits per-frame work to prevent frame time spikes from bulk retirement.

> For full asset API details — `AssetManager`, handles, load tickets, packages — see [Asset Loading](../api/04-assets-sync-deferred-and-handles.md).

## Frame Outcomes

`render_scene_with_view` returns `Result<FrameRenderOutcome, RendererError>`. The five outcome variants:

### `FrameRenderOutcome::Rendered`

The frame was recorded, submitted to the GPU queue, and presented to the swapchain. This is the normal, expected result.

### `FrameRenderOutcome::SkippedResizePending`

The swapchain is mid-rebuild (zero-area window, or a resize was requested but the new swapchain hasn't been created yet). The frame was skipped entirely — no command recording or submission. Continue the loop; the next frame will retry.

### `FrameRenderOutcome::SkippedAcquireUnavailable`

The swapchain image acquire returned `VK_NOT_READY` or `VK_TIMEOUT`. This is a transient WSI condition (the compositor hasn't released an image yet). The frame was skipped. Continue the loop.

### `FrameRenderOutcome::SubmittedNotPresented`

The frame was recorded and submitted to the GPU, but presentation was not performed. This occurs during headless rendering or when frame capture is configured without a present target. Continue the loop.

### `FrameRenderOutcome::PresentedSuboptimal`

The frame was presented, but the swapchain is in a suboptimal state (e.g. the window surface properties changed). Presentation succeeded, but a resize should be triggered. The next `RedrawRequested` will catch up and call `resize()`.

> These outcomes are enumerated in [`src/renderer/src/api/renderer.rs`](../../src/renderer/src/api/renderer.rs). For their exact definitions and the internal state transitions, see the [renderer lifecycle API docs](../api/02-renderer-lifecycle-and-frame-api.md).

## Terminal Errors

Two `RendererError` variants require the app to exit:

### `RendererError::DeviceLost`

The Vulkan device was lost. Causes include:

- GPU hang detected by the driver
- Physical GPU removal (eGPU disconnect)
- Driver crash or reset
- `VK_ERROR_DEVICE_LOST` from any Vulkan call

After `DeviceLost`, the renderer's internal state is invalid. The only safe action is to drop the renderer and either exit or create a new one. The checkpoint exits.

### `RendererError::BackendPoisoned(msg)`

A previous operation encountered a terminal error (device lost, unexpected Vulkan error, or internal invariant violation). The backend was marked poisoned to prevent further unsafe operations. The message describes the original failure.

After `BackendPoisoned`, all further renderer calls will also return `BackendPoisoned`. The app must exit and recreate the renderer.

> For the complete error taxonomy, see [`RendererError`](../api/02-renderer-lifecycle-and-frame-api.md) and the source at [`src/renderer/src/api/errors.rs`](../../src/renderer/src/api/errors.rs).

### Other Errors

Any other `RendererError` variant (e.g. `InvalidState`, `Frame(Resize(...))`, `CaptureConfig(...)`) in the checkpoint is treated as fatal and exits. A more robust app might handle specific errors differently (e.g. retry on transient errors, log and continue on capture failures), but the checkpoint exits to keep error handling explicit and unambiguous.

## Renderer API Reference Links

The guide covers the app's interaction with the renderer at a conceptual level. For full API documentation:

| Topic | Document |
|-------|----------|
| `Renderer` lifecycle, frame API, resize, hooks | [`docs/api/02-renderer-lifecycle-and-frame-api.md`](../api/02-renderer-lifecycle-and-frame-api.md) |
| `Scene`, scene graph, fragments, culling, lights | [`docs/api/03-scene-graph-and-fragment-workflows.md`](../api/03-scene-graph-and-fragment-workflows.md) |
| `AssetManager`, packages, handles, async loading | [`docs/api/04-assets-sync-deferred-and-handles.md`](../api/04-assets-sync-deferred-and-handles.md) |
| Render hooks and extension points | [`docs/api/05-render-hooks-and-extension-points.md`](../api/05-render-hooks-and-extension-points.md) |
| Input system (app-owned path) | [`docs/api/06-input-polling-and-listeners.md`](../api/06-input-polling-and-listeners.md) |
| Engine arguments, config, launch flags | [`docs/api/07-engine-arguments.md`](../api/07-engine-arguments.md) |
| Events and lifecycle | [`docs/api/12-events-and-lifecycle.md`](../api/12-events-and-lifecycle.md) |
| App-owned loop primitives (full reference) | [`docs/api/15-app-owned-loop.md`](../api/15-app-owned-loop.md) |

### Internal Deep Dives

For engine maintainers and those modifying the renderer itself:

| Topic | Document |
|-------|----------|
| API-to-backend handoff (`render_scene_with_view` → Vulkan) | [`docs/internal/04-api-to-backend-handoff.md`](../internal/04-api-to-backend-handoff.md) |
| Rendergraph pass order, dependencies, aliasing | [`docs/internal/07-rendergraph-dependencies-and-aliasing.md`](../internal/07-rendergraph-dependencies-and-aliasing.md) |
| Vulkan descriptor ABI, pipeline layouts, shader contracts | [`docs/internal/14-renderer-descriptor-abi.md`](../internal/14-renderer-descriptor-abi.md) |
| winit input integration, cursor confinement, platform quirks | [`docs/internal/09-input-winit-integration.md`](../internal/09-input-winit-integration.md) |

## Compatibility Callouts

### Renderer-Owned Input (not for custom apps)

The renderer provides `Renderer::update_input(&window, &event)` and `Renderer::render_scene(&window, &mut scene)` for backward compatibility with renderer-internal examples and tests. These methods use the renderer's own `InputSystem` and `Camera` state. **Do not use these in custom apps** — they couple your app to renderer-owned state and prevent you from owning input dispatch, event emission, and camera control.

The guide's app-owned path (`route_platform_input_to_app` + `render_scene_with_view`) is the supported pattern for custom applications.

### Renderer-Owned Event Bus (not for custom apps)

The renderer maintains its own `EventBus` accessible via `renderer.events()` and `renderer.events_mut()`. This bus is used by the renderer's compatibility path and internal subsystems. Custom apps should create their own `EventBus` via `runtime_event_bus()` and never interact with the renderer's bus.

## Next

Continue to the independently adoptable subsystem chapters:

- [06 — Input System](06-input.md)
- [07 — Events & Lifecycle](07-events-and-lifecycle.md)
- [08 — Scene Construction](08-scene-construction.md)
- [09 — Asset Pipeline](09-asset-pipeline.md)

Real-world case studies for Dungeon Dogfood and Voxel Demo are planned as later guide chapters.
