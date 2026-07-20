# Renderer Lifecycle & Frame API

> Source: [`src/renderer/src/api/renderer.rs`](../src/renderer/src/api/renderer.rs) — no legacy docs consulted.

## Overview

`Renderer` is the entry point for everything: it owns the Vulkan device, swapchain, asset manager, input system, debug UI, and scene rendering. You create one per application window.

## Constructor

```rust
pub fn new(config: RendererConfig, window: &Window) -> Result<Self, RendererError>
```

Defined at [`renderer.rs:80`](../src/renderer/src/api/renderer.rs:80). Initializes:
- Vulkan instance, device, swapchain (via `vk_init.rs`)
- `AssetManager` (handles, caches, default textures)
- `InputSystem` (layered dispatch)
- Debug UI (imgui context)
- Optional startup scene (if `config.preload_startup_scene` is true)

## Frame Loop: Two APIs

### Convenience: `render_scene()` — renderer compatibility path

> **Compatibility note:** `render_scene` is the **renderer compatibility path** convenience method. For custom apps, prefer `render_scene_with_view` with a caller-provided `CameraView`. See [15-app-owned-loop.md](15-app-owned-loop.md).

```rust
pub fn render_scene(&mut self, window: &Window, scene: &mut Scene)
    -> Result<FrameRenderOutcome, RendererError>
```

Defined at [`renderer.rs:145`](../src/renderer/src/api/renderer.rs:145). Pumps asset tasks, calls `begin_frame` + `render_scene_in_frame` + `end_frame` internally. Returns:

```rust
pub enum FrameRenderOutcome {
    Rendered,                    // frame completed (and presented when windowed)
    SkippedAcquireUnavailable,   // bounded acquire timed out; retry a later frame
    SkippedResizePending,        // skipped because swapchain needs rebuild
    SubmittedNotPresented, // GPU submit succeeded; out-of-date swapchain was not presented
    PresentedSuboptimal,   // presented, but acquire/present requested a rebuild
}
```

### Explicit: `begin_frame` / `render_scene_in_frame` / `end_frame`

```rust
pub fn begin_frame(&mut self, window: &Window) -> Result<FrameContext, RendererError>
pub fn render_scene_in_frame(&mut self, frame: &mut FrameContext, scene: &mut Scene)
    -> Result<FrameRenderOutcome, RendererError>
pub fn end_frame(&mut self, frame: FrameContext) -> Result<(), RendererError>
```

Defined at [`renderer.rs:170-210`](../src/renderer/src/api/renderer.rs:170). Use this when you need to interleave custom Vulkan work between begin and end. `FrameContext` is opaque:

```rust
pub struct FrameContext {
    frame_number: u32,       // opaque, internal use only
    render_attempted: bool,  // prevents double-render
}
```

**Rules:**
- `render_scene_in_frame` must be called exactly once per `FrameContext`
- Calling `render_scene()` while an explicit frame is open returns `InvalidState`
- Calling `render_scene_in_frame` twice returns `InvalidState`

## Resize

```rust
pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererError>
pub fn resize_requested(&self) -> bool
```

Defined at [`renderer.rs:124`](../src/renderer/src/api/renderer.rs:124). Rebuilds the swapchain. Fails if an explicit frame is open. `resize_requested()` polls whether the Vulkan runtime has flagged a pending resize (from `VK_ERROR_OUT_OF_DATE_KHR`).

## Input Integration — renderer compatibility path

> **Compatibility note:** These methods use renderer-owned `InputSystem` and camera state. For custom apps, prefer the **current app-owned path**: `route_platform_input_to_app` for platform routing, app-owned `InputSystem` with `InputActionEventEmitter`, and app-owned `FPSController` for camera. See [15-app-owned-loop.md](15-app-owned-loop.md) and [06-input-polling-and-listeners.md](06-input-polling-and-listeners.md).

```rust
pub fn update_input(&mut self, window: &Window, event: &Event<()>) -> Result<(), RendererError>
pub fn install_default_fps_input(&mut self) -> LayerHandle
pub fn uninstall_default_fps_input(&mut self)
pub fn input(&self) -> &InputSystem
pub fn input_mut(&mut self) -> &mut InputSystem
```

Defined at [`renderer.rs:97-108`](../src/renderer/src/api/renderer.rs:97). `update_input()` routes winit events to both imgui and the `InputSystem`. It also handles F1 (debug UI toggle) and F2 (console toggle) internally.

## Render Hooks

```rust
pub fn set_pre_render_hook(&mut self, hook: Option<RenderHook>)
pub fn set_post_render_hook(&mut self, hook: Option<RenderHook>)
```

Defined at [`renderer.rs:237-240`](../src/renderer/src/api/renderer.rs:237). Pre-hook fires before rendergraph execution; post-hook fires after. `RenderHook` is `Box<dyn FnMut(&mut RenderHookContext) -> Result<(), HookError> + Send>`. See [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md).

## Camera Access — renderer compatibility path

> **Compatibility note:** These methods read/write the renderer-owned camera. Still used by capture tests and marching_terrain compatibility path. For custom apps, manage camera state app-side and pass a `CameraView` via `render_scene_with_view`. See [15-app-owned-loop.md](15-app-owned-loop.md).

```rust
pub fn camera_position(&self) -> Vec3
pub fn set_camera_position(&mut self, position: Vec3)
```

Defined at [`renderer.rs:296`](../src/renderer/src/api/renderer.rs:296). Direct read/write of the internal camera world position. The default FPS input layer updates this automatically.

## Debug UI & Timing

```rust
// Debug views
pub fn register_debug_view(&mut self, descriptor: DebugViewDescriptor,
    callback: DebugViewCallback) -> DebugViewId
pub fn unregister_debug_view(&mut self, id: DebugViewId)
pub fn set_debug_view_enabled(&mut self, id: DebugViewId, enabled: bool)

// Global visibility
pub fn toggle_debug_ui(&mut self)
pub fn set_debug_ui_visible(&mut self, visible: bool)
pub fn is_debug_ui_visible(&self) -> bool
pub fn toggle_console_ui(&mut self)
pub fn toggle_debug_overlay_ui(&mut self)

// Timing recording
pub fn configure_debug_timing_recording(&mut self, duration_secs: Option<u64>,
    interval_ms: Option<u64>, output_path: Option<String>) -> Result<(), RendererError>
pub fn start_debug_timing_recording(&mut self) -> Result<String, RendererError>
```

Defined at [`renderer.rs:244-298`](../src/renderer/src/api/renderer.rs:244). See [08-debug.md](08-debug.md).

## Environment Status

```rust
pub fn environment_runtime_status(&self) -> EnvironmentRuntimeStatus
```

Defined at [`renderer.rs:305`](../src/renderer/src/api/renderer.rs:305). Returns the current state of environment map transitions (e.g., during progressive IBL precomputation).

## Startup Scene

```rust
pub fn take_startup_scene(&mut self) -> Option<Scene>
```

Defined at [`renderer.rs:223`](../src/renderer/src/api/renderer.rs:223). Returns the pre-built startup scene if `preload_startup_scene` was true. Call this once to take ownership; subsequent calls return `None`. If you build your own scene, always call this first to avoid wasted resources.

## See Also

- [03-scene.md](03-scene.md) — scene construction API
- [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md) — render hook extension points
- [08-debug.md](08-debug.md) — debug UI and timing capture
- [Internal: API-to-backend handoff](../internal/02-renderer-internals.md)
