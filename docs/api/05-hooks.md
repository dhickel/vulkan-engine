# Render Hooks & Extension Points

> **This page has been superseded.** The canonical hook and extension-point documentation is at
> [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md).
> This page is kept as a compatibility redirect. The old content claimed hooks could record custom Vulkan
> commands, listed obsolete `HookError` variants, and misrepresented error propagation behavior.
> Those claims were incorrect — see the canonical page for the live-code-accurate contract.

## Quick Summary (Live Code)

- `Renderer::set_pre_render_hook(Option<RenderHook>)` and `set_post_render_hook(Option<RenderHook>)` accept safe, API-level callbacks.
- `RenderHookContext` exposes `frame_index: u64`, `viewport_size: (u32, u32)`, and `depth_texture: Option<TextureHandle>`.
- Hooks do **not** expose command buffers, descriptor sets, rendergraph state, or any raw Vulkan handles.
- `HookError` variants: `Unsupported(String)`, `Registration(String)`, `Invocation(String)` — see [`src/renderer/src/api/errors.rs`](../../src/renderer/src/api/errors.rs).
- Hook errors are **logged** and the frame continues; they are **not** escalated to `RendererError`.
- Panics inside hooks are caught and converted to `HookError::Invocation`.
- Pre-hook fires before rendergraph execution; post-hook fires after successful rendergraph execution.

## Extension Points Beyond Hooks

- **Debug views:** `Renderer::register_debug_view(...)` — custom imgui panels rendered by the engine UI manager. See [08-debug.md](08-debug.md).
- **App UI:** `Renderer::register_app_ui(...)` — always-rendered imgui chrome for editor shells.
- **Frame capture:** `Renderer::request_frame_capture(...)` — present-target or draw-target frame captures through the facade.
- **Timing capture:** `Renderer::configure_debug_timing_recording(...)` — JSONL timing reports for offline analysis.
- **Events:** `EventBus`, `EventRecorder` — lifecycle and input observation without renderer mutation. See [12-events-and-lifecycle.md](12-events-and-lifecycle.md).

## Advanced Interop (Feature-Gated)

For internal-engine experiments and expert diagnostics, the `advanced-interop` Cargo feature (opt-in, **alpha/unstable**) exposes:

- `renderer::api::advanced::renderer_core_mut()` — `unsafe` access to `&mut VkRenderCore`. Bypasses all facade invariants. Misuse can break synchronization, descriptor lifecycle, or swapchain safety.
- `renderer::rendergraph` — the pass graph and `RenderPassNode` trait become public. Custom pass registration has no resource/synchronization validation and is **not stable**. See [07-rendergraph-dependencies-and-aliasing.md](../internal/07-rendergraph-dependencies-and-aliasing.md).

These paths are not beginner-stable and do not imply API compatibility across alpha sprints.

## See Also

- [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md) — canonical documentation
- [02-renderer-lifecycle-and-frame-api.md](02-renderer-lifecycle-and-frame-api.md) — frame lifecycle
- [08-debug.md](08-debug.md) — debug views and timing capture
- [12-events-and-lifecycle.md](12-events-and-lifecycle.md) — event observation
- [Internal: Rendergraph Dependencies](../internal/07-rendergraph-dependencies-and-aliasing.md) — pass ordering and resource contracts
