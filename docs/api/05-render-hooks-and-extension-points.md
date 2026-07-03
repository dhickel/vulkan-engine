# Render Hooks and Extension Points

## 1. Purpose & Audience
This chapter is for students, hobbyists, and indie developers using the facade API who need lightweight custom per-frame behavior without taking direct ownership of Vulkan internals.

## 2. Where This Fits in Engine Flow
Hook flow in the current facade path:
`Renderer::set_pre_render_hook(...)` / `Renderer::set_post_render_hook(...)` -> `Renderer::render_scene(...)` (or explicit frame API) -> `Renderer::render_scene_internal(...)` -> `VkRender::render_with_hooks(...)` -> `VkRenderCore::render_with_hooks(...)`.

## 3. Key Concepts
- Public, safe extension surface:
  - `Renderer::set_pre_render_hook(Option<RenderHook>)`
  - `Renderer::set_post_render_hook(Option<RenderHook>)`
  - `RenderHookContext` (`frame_index`, `viewport_size`)
  - **Limitation**: The optional `depth` field in `RenderHookContext` is currently always `None`. Depth texture plumbing from the rendergraph into hooks is not yet implemented. Hooks that need depth must use the `advanced-interop` feature gate to access raw Vulkan resources.
- Public debug UI extension surface (separate from hooks):
  - `Renderer::register_debug_view(...)`
  - `Renderer::unregister_debug_view(...)`
  - `Renderer::set_debug_view_enabled(...)`
  - `Renderer::toggle_debug_ui()` / `set_debug_ui_visible(...)`
- Hook ordering is fixed:
  - pre-hook runs before rendergraph execution
  - post-hook runs after successful rendergraph execution
- Hook callbacks are API-level only. They do not expose command buffers, descriptors, or rendergraph internals.
- Hook failures are isolated:
  - panic is caught and converted to `HookError::Invocation`
  - returned hook errors are wrapped as `HookError::Invocation`
  - renderer logs errors and continues frame execution
- Internal-only / unstable extension path:
  - `api::advanced::renderer_core_mut(...)` is `unsafe`
  - it is feature-gated behind `advanced-interop`
  - this bypasses facade invariants and is not a stable beginner-facing contract

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs
pub fn set_pre_render_hook(&mut self, hook: Option<RenderHook>) {
    self.pre_render_hook = hook;
}

pub fn set_post_render_hook(&mut self, hook: Option<RenderHook>) {
    self.post_render_hook = hook;
}
```

Snippet Type: Real
```rust
// Typical registration pattern in app code
use renderer::{HookError, Renderer};

fn install_hooks(renderer: &mut Renderer) {
    renderer.set_pre_render_hook(Some(Box::new(|ctx| {
        if ctx.viewport_size.0 == 0 || ctx.viewport_size.1 == 0 {
            return Err(HookError::Registration("viewport is zero-sized".to_string()));
        }
        Ok(())
    })));

    renderer.set_post_render_hook(Some(Box::new(|ctx| {
        if ctx.frame_index % 300 == 0 {
            log::info!("Rendered frame {}", ctx.frame_index);
        }
        Ok(())
    })));
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs (ordering boundary)
self.reset_and_begin_frame_cmd(frame.cmd_buffer);
pre_render_hook();

let graph_result = unsafe { self.execute_rendergraph_for_frame(submission, rendergraph) };
if let Err(err) = graph_result {
    error!("RenderGraph execution failed: {err}");
    self.resize_requested = true;
    return;
}

post_render_hook();
```

Snippet Type: Pseudocode
```text
Future custom pass strategy (not stable API yet):
  keep facade hooks for app-side logic only
  add internal rendergraph node registration behind internal API
  validate node dependencies and resource hazards internally
  expose a higher-level safe facade only after ordering/resource contracts are stable
```

## 5. Best Practices
- Keep hooks narrowly scoped and side-effect-light.
- Use hooks for app-level orchestration, telemetry, and lightweight state checks.
- Prefer debug views for ongoing UI tooling; use hooks for non-UI orchestration boundaries.
- Return explicit `HookError` messages so logs are actionable.
- Keep heavy GPU/resource work out of hooks; use asset APIs and normal frame flow instead.
- Document expected hook order in your app (`pre` before graph, `post` after successful graph).

## 6. Gotchas & Failure Modes
- Pre/post hooks run inside frame execution; expensive work here will hurt frame time.
- `post_render` is skipped if rendergraph execution fails early in backend.
- Hook errors are logged, not escalated to `RendererError` from render calls.
- Assuming hook ownership over backend resources is incorrect at facade level.
- Advanced interop (`renderer_core_mut`) can violate synchronization and lifecycle guarantees if misused.

## 7. Debugging Playbook
- Step 1: run `cargo run -p renderer --example api_test` and add temporary pre/post hook logs.
- Step 2: confirm order in logs (`pre` before rendergraph-related logs, then `post`).
- Step 3: if hook failures are silent in app flow, inspect logger output for `pre_render hook failed` / `post_render hook failed`.
- Step 4: if post-hook never triggers, check for rendergraph failure logs and resize-pending behavior.
- Step 5: if using advanced interop, reproduce with it disabled to isolate unsafe backend mutations.

## 8. Cross-Module Links
- Hook types and panic/error wrapping: `src/renderer/src/api/hooks.rs`
- Public renderer hook setters and invocation site: `src/renderer/src/api/renderer.rs`
- Hook-capable backend frame execution: `src/renderer/src/vulkan/vk_render.rs`
- Unsafe advanced interop boundary: `src/renderer/src/api/advanced.rs`
- Internal frame handoff context: `docs/internal/04-api-to-backend-handoff.md`

## 9. Standard References
- Vulkan render pass dependency model: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#renderpass
- Vulkan Guide (sync overview): https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- Rust trait objects (closure/object boundary context): https://doc.rust-lang.org/book/ch17-02-trait-objects.html
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
