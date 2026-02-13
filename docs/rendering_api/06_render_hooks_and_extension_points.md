# 06 - Render Hooks and Extension Points

This chapter explains what render hooks are, why they exist, and how to use them safely.

## What Hooks Are

Render hooks are optional callbacks attached to the renderer:

- `Renderer::set_pre_render_hook(Option<RenderHook>)`
- `Renderer::set_post_render_hook(Option<RenderHook>)`
- `RenderHook = Box<dyn FnMut(&mut RenderHookContext) -> Result<(), HookError> + Send>`

Each hook receives `RenderHookContext<'_>`:
- `frame_index: u64`
- `viewport_size: (u32, u32)`

`FnMut` means the closure can keep mutable state across frames (counters, timers, rolling stats).

## Hook API Surface

- `RenderHookContext<'_>`
  - `frame_index: u64`
  - `viewport_size: (u32, u32)`
- `RenderHook = Box<dyn FnMut(&mut RenderHookContext) -> Result<(), HookError> + Send>`
- `Renderer::set_pre_render_hook(Option<RenderHook>)`
- `Renderer::set_post_render_hook(Option<RenderHook>)`

## Why Hooks Exist (Problem Being Solved)

Hooks provide a stable extension point for per-frame instrumentation and app-side coordination
without exposing internal Vulkan objects.

Use hooks when you need:
- frame-stage tracing/logging,
- lightweight profiling markers,
- frame-indexed telemetry,
- viewport-aware app bookkeeping at render boundaries.

Hooks are not meant for custom draw submission or rendergraph mutation.

## Invocation Timing

When hooks are enabled, render flow is:

1. transfer/environment prep,
2. frame acquire + command begin,
3. `pre_render` hook,
4. rendergraph execution,
5. `post_render` hook,
6. command end + submit + present.

Important timing caveats:
- If resize is pending and frame render is skipped, hooks are not run.
- If frame acquire fails for that call, hooks are not run.
- If rendergraph execution fails, `post_render` is not run.

## Failure Policy

- Hook failures are non-fatal by default.
- Returned hook errors are wrapped as `HookError::Invocation` and logged.
- Panics inside hooks are caught and converted to `HookError::Invocation`.
- Frame execution continues.

Implication:
- Hooks are safe for instrumentation/debug overlays.
- Do not rely on hook error returning from `render_scene`; failures are logged, not escalated.

## How To Use Hooks

Register once during setup, then render normally (one-shot or explicit frame API).  
Pass `None` to clear a hook.

## Example: Frame Boundary Timing

```rust
use renderer::RenderHookContext;
use std::sync::{Arc, Mutex};
use std::time::Instant;

let frame_start = Arc::new(Mutex::new(None::<Instant>));

let pre_state = Arc::clone(&frame_start);
renderer.set_pre_render_hook(Some(Box::new(move |ctx: &mut RenderHookContext<'_>| {
    *pre_state.lock().unwrap() = Some(Instant::now());
    log::trace!("pre_render frame={} viewport={:?}", ctx.frame_index, ctx.viewport_size);
    Ok(())
})));

let post_state = Arc::clone(&frame_start);
renderer.set_post_render_hook(Some(Box::new(move |ctx: &mut RenderHookContext<'_>| {
    if let Some(start) = post_state.lock().unwrap().take() {
        let elapsed = start.elapsed();
        log::trace!("post_render frame={} elapsed={:?}", ctx.frame_index, elapsed);
    }
    Ok(())
})));

// Later, if needed:
renderer.set_pre_render_hook(None);
renderer.set_post_render_hook(None);
```

## Extension Limits

Current public hook context is metadata-only.

No public safe API currently exposes:
- Vulkan command buffer mutation.
- Descriptor injection.
- Rendergraph pass registration.

Design implication:
- Keep hooks short, non-blocking, and side-effect-limited.
- Put required game logic in your normal update/render flow, not in hooks.

For advanced internal interop, see feature-gated `advanced-interop` API and use with caution.

## Learn More

- Hook implementation: `src/renderer/src/api/hooks.rs`
- Renderer hook integration: `src/renderer/src/api/renderer.rs`
