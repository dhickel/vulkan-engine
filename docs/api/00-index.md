# Engine API Reference

> All citations trace to source code. Generated from a fresh codebase audit — no legacy docs consulted.

## Audience

Rust developers building applications on this Vulkan rendering engine. Familiarity with Rust is assumed; graphics knowledge is explained inline.

## Workspace Context

The root workspace currently contains `engine`, `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, and `apps/editor`. The root `engine` binary is the alpha data-driven project runtime launcher. Custom Rust behavior belongs in app crates under `apps/<name>`. Renderer examples remain diagnostics/API references. Support crates and apps are alpha-stage workspace members unless their own docs say otherwise.

## Quick Navigation

| Topic | Document | What You'll Learn |
|-------|----------|-------------------|
| First frame | [01-quickstart.md](01-quickstart.md) | Window → Renderer → Scene → loop |
| Renderer lifecycle | [02-renderer.md](02-renderer.md) | `Renderer::new`, frame API, resize, hooks |
| Scene construction | [03-scene-graph-and-fragment-workflows.md](03-scene-graph-and-fragment-workflows.md) | `Scene`, fragments, transforms, lights, persistence, editor commands |
| Asset loading | [04-assets-sync-deferred-and-handles.md](04-assets-sync-deferred-and-handles.md) | `AssetManager`, packages, durable IDs, load tickets, models, environments |
| Render hooks | [05-hooks.md](05-hooks.md) | `RenderHook`, `RenderHookContext`, extension points |
| Input system | [06-input.md](06-input.md) | `InputSystem`, layers, action maps, snapshots |
| Configuration | [07-config.md](07-config.md) | `RendererConfig`, `VisualTuning`, asset policies |
| Debug & timing | [08-debug.md](08-debug.md) | Debug UI, timing capture, custom views |
| Editor placement | [09-editor-asset-browser-and-wall-chunks.md](09-editor-asset-browser-and-wall-chunks.md) | Project package loading, asset browser, wall chunk prefab placement |
| Packaging CLI | [10-packaging-cli.md](10-packaging-cli.md) | Rust CLI validation, authoring, and folder pack output |
| Runtime launcher | [11-runtime-project-launcher.md](11-runtime-project-launcher.md) | Root `engine` launcher, project manifests, headless draw capture, app-crate loop |
| Events and lifecycle | [12-events-and-lifecycle.md](12-events-and-lifecycle.md) | `EventBus`, lifecycle/action events, recorder usage, safe mutation rules |
| Audio foundation | [13-audio-foundation.md](13-audio-foundation.md) | Packaged audio metadata, device-independent clips, opt-in playback, audio events |

## Top-Level Re-exports

Everything a user needs is re-exported from the `renderer` crate (see [`src/renderer/src/lib.rs`](../../src/renderer/src/lib.rs)):

```rust
// Core facade
pub use api::{Renderer, RendererConfig, RendererError, Scene, ...};

// Handles
pub use data::handles::{MeshHandle, TextureHandle, MaterialHandle, EnvironmentHandle};

// Input (re-exported from the input crate)
pub use input::{InputSystem, InputSnapshot, InputLayer, ActionMap, ...};

// Events (re-exported from engine_events)
pub use engine_events::{EventBus, EventRecorder, EngineEvent, EventStage, ...};

// Debug
pub use debug_ui::{DebugViewId, DebugViewDescriptor, DebugTimingRow, ...};
```

The full re-export list is at [`src/renderer/src/api/mod.rs`](../../src/renderer/src/api/mod.rs). Everything below `api::*` in `lib.rs` is the stable public surface.

## Runtime Launcher

Data-driven projects run through the root `engine` launcher:

```sh
cargo run -- --project apps/editor/sample_project/engine.project.toml
```

Headless visual validation uses the offscreen draw target:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

Renderer examples remain useful diagnostics and API references. Custom Rust applications live under `apps/<name>` and run with `cargo run -p <app>`. For an off-workspace compile-first starting point, `engine_pack new-app` generates a standalone support-crate scaffold that uses public `engine_events`, `input`, and `physics` dependencies without editing the root workspace. The alpha event contract is available through the renderer facade. The standalone `physics` crate provides durable ID descriptors, basic collider shapes, ray queries, contact records, and helpers that translate physics records into `engine_events` payloads. The standalone `audio` crate provides durable clip IDs, device-independent load/probe paths, explicit device-backed playback, package/scene validation, and an opt-in dogfood proof. Runtime scene-to-physics loading, editor collision/audio authoring UI, dynamic Rust hot reload, production scripting runtime scheduling, package-level script assets, production audio mixing/spatialization/streaming, broad dogfood migration to project manifests, and renderer-window generated app templates are deferred.

## Canonical Renderer Example

Every demo follows the same pattern (see [`src/renderer/examples/common/mod.rs`](../../src/renderer/examples/common/mod.rs)):

1. Create a `winit` event loop + window
2. Construct `Renderer::new(config, &window)`
3. Build a `Scene` (or use `renderer.take_startup_scene()`)
4. In the event loop: feed events to `renderer.update_input()`, call `renderer.render_scene()`, request redraw

```rust
// src/renderer/examples/demo_pbr.rs (simplified)
let mut renderer = Renderer::new(config, &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_default();

event_loop.run(move |event, control_flow| {
    renderer.update_input(&window, &event)?;
    match event {
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
            renderer.render_scene(&window, &mut scene)?;
            window.request_redraw();
        }
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            control_flow.exit();
        }
        _ => {}
    }
})?;
```

## Running Renderer Examples

```sh
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading
cargo run -p renderer --example api_test
```

## See Also

- [Internal Architecture Reference](../internal/00-index.md) — implementation details
- [Events and Lifecycle](12-events-and-lifecycle.md) — event subscriptions, recorders, and lifecycle ordering
- [Audio Foundation](13-audio-foundation.md) — packaged audio metadata and opt-in playback
- [Alpha Readiness Baseline](../gap-report.md) — current readiness and residual-classification routing
- [Renderer AGENTS.md](../../src/renderer/AGENTS.md) — contributor guide
