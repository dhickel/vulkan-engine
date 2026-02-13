# 06 - Render Hooks and Extension Points

This chapter documents the safe hook API for frame-stage customization.

## Hook API Surface

- `RenderHookContext<'_>`
  - `frame_index: u64`
  - `viewport_size: (u32, u32)`
- `RenderHook = Box<dyn FnMut(&mut RenderHookContext) -> Result<(), HookError> + Send>`
- `Renderer::set_pre_render_hook(Option<RenderHook>)`
- `Renderer::set_post_render_hook(Option<RenderHook>)`

## Stage Boundaries

Current invocation points:
- `pre_render`:
  - After acquire + command buffer begin.
  - Before rendergraph pass execution.
- `post_render`:
  - After rendergraph execution.
  - Before command end/submit/present.

## Failure Policy

- Hook failures are non-fatal by default.
- Returned hook errors are wrapped as `HookError::Invocation` and logged.
- Panics inside hooks are caught and converted to `HookError::Invocation`.
- Frame execution continues.

Implication:
- Hooks are safe for instrumentation/debug overlays.
- Do not rely on hook error returning from `render_scene`; failures are logged, not escalated.

## Example Hook Registration

```rust
use renderer::{HookError, RenderHookContext};

renderer.set_pre_render_hook(Some(Box::new(|ctx: &mut RenderHookContext<'_>| {
    if ctx.viewport_size.0 == 0 || ctx.viewport_size.1 == 0 {
        return Err(HookError::Invocation("viewport is zero".to_string()));
    }
    Ok(())
})));
```

## Extension Limits

Current public hook context is metadata-only.

No public safe API currently exposes:
- Vulkan command buffer mutation.
- Descriptor injection.
- Rendergraph pass registration.

For advanced internal interop, see feature-gated `advanced-interop` API and use with caution.

## Learn More

- Hook implementation: `src/renderer/src/api/hooks.rs`
- Renderer hook integration: `src/renderer/src/api/renderer.rs`
