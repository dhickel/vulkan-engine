# Render Hooks & Extension Points

> Source: [`src/renderer/src/api/hooks.rs`](../src/renderer/src/api/hooks.rs) — no legacy docs consulted.

## Overview

Render hooks let you inject custom logic before and after the rendergraph executes. They're the primary extension mechanism for users who need to run custom Vulkan commands, update GPU data, or perform post-processing without modifying engine internals.

Use events for lifecycle/input observation and hooks for render-thread extension points. Event listeners receive immutable envelopes and should not perform renderer mutation directly.

## Hook Types

```rust
pub type RenderHook = Box<dyn FnMut(&mut RenderHookContext<'_>) -> Result<(), HookError> + Send>;

pub struct RenderHookContext<'a> {
    pub frame_index: u64,
    pub viewport_size: (u32, u32),
}
```

Defined at [`hooks.rs:11-15`](../src/renderer/src/api/hooks.rs:11).

### Pre-Render Hook

Fires **after** the Vulkan frame is prepared (command buffer acquired, descriptor pool reset) but **before** the rendergraph executes. Use for: updating per-frame uniforms, dispatching compute work, recording custom commands into the frame's command buffer.

```rust
renderer.set_pre_render_hook(Some(Box::new(|ctx: &mut RenderHookContext| {
    // ctx.frame_index, ctx.viewport_size available
    Ok(())
})));
```

### Post-Render Hook

Fires **after** the rendergraph completes (all passes executed, frame submitted) but **before** the renderer advances the frame counter. Use for: reading back GPU data, screenshot capture, custom present logic.

```rust
renderer.set_post_render_hook(Some(Box::new(|ctx: &mut RenderHookContext| {
    Ok(())
})));
```

## Removing Hooks

Pass `None` to clear a hook:

```rust
renderer.set_pre_render_hook(None);
renderer.set_post_render_hook(None);
```

## Error Handling

```rust
pub enum HookError {
    Fatal(String),
    Transient(String),
}
```

Defined at [`hooks.rs`](../src/renderer/src/api/hooks.rs). `Fatal` errors propagate up through `render_scene()` as `RendererError::Hook(...)`. `Transient` errors are logged but don't abort the frame.

## Limitations

- The `RenderHookContext` exposes only `frame_index` and `viewport_size` — no access to the Vulkan command buffer, descriptor sets, or rendergraph state
- For deeper integration, use the `advanced-interop` feature gate at [`api/advanced.rs`](../src/renderer/src/api/advanced.rs), which exposes `raw_core_mut()` (documented as unsafe, internal-use-only)
- Hooks run synchronously on the render thread; long-running hooks will block rendering

## See Also

- [02-renderer.md](02-renderer.md) — where hooks fit in the frame lifecycle
- [12-events-and-lifecycle.md](12-events-and-lifecycle.md) — event observation and mutation safety
- [Internal: API-to-backend handoff](../internal/02-renderer-internals.md)
- [src/renderer/src/api/hooks.rs](../src/renderer/src/api/hooks.rs) — implementation
