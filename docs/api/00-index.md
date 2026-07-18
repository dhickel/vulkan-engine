# Engine API Reference

> All citations trace to source code. Generated from a fresh codebase audit — no legacy docs consulted.

## Audience

Rust developers building applications on this Vulkan rendering engine. Familiarity with Rust is assumed; graphics knowledge is explained inline.

## Workspace Context

The root workspace currently contains `engine`, `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, `apps/marching_terrain`, and `tools/engine_pack`. The root `engine` crate is both the alpha data-driven project runtime launcher and a thin app facade over raw support crates. Custom Rust behavior belongs in app crates under `apps/<name>`. Renderer examples remain diagnostics/API references. Support crates and apps are alpha-stage workspace members unless their own docs say otherwise.

## Quick Navigation

| Topic | Document | What You'll Learn |
|-------|----------|-------------------|
| First frame | [01-quickstart.md](01-quickstart.md) | Window → Renderer → Scene → loop |
| Renderer lifecycle | [02-renderer.md](02-renderer.md) | `Renderer::new`, frame API, resize, hooks |
| Scene construction | [03-scene-graph-and-fragment-workflows.md](03-scene-graph-and-fragment-workflows.md) | `Scene`, fragments, transforms, default-on culling, directional shadows, persistence, editor commands |
| Asset loading | [04-assets-sync-deferred-and-handles.md](04-assets-sync-deferred-and-handles.md) | `AssetManager`, packages, durable IDs, load tickets, models, environments |
| Render hooks | [05-render-hooks-and-extension-points.md](05-render-hooks-and-extension-points.md) | `RenderHook`, `RenderHookContext`, extension points |
| Input system | [06-input.md](06-input.md) | `InputSystem`, layers, action maps, snapshots |
| Configuration | [07-config.md](07-config.md) | `RendererConfig`, `VisualTuning`, asset policies |
| Debug & timing | [08-debug.md](08-debug.md) | Debug UI, timing capture, custom views |
| Editor placement | [09-editor-asset-browser-and-wall-chunks.md](09-editor-asset-browser-and-wall-chunks.md) | Project package loading, asset browser, wall chunk prefab placement |
| Packaging CLI | [10-packaging-cli.md](10-packaging-cli.md) | Rust CLI validation, authoring, and folder pack output |
| Runtime launcher | [11-runtime-project-launcher.md](11-runtime-project-launcher.md) | Root `engine` launcher, project manifests, headless draw capture, app-crate loop |
| Events and lifecycle | [12-events-and-lifecycle.md](12-events-and-lifecycle.md) | `EventBus`, lifecycle/action events, recorder usage, safe mutation rules |
| Audio foundation | [13-audio-foundation.md](13-audio-foundation.md) | Packaged audio metadata, device-independent clips, opt-in playback, audio events |
| Dogfood vertical slice | [14-dogfood-vertical-slice.md](14-dogfood-vertical-slice.md) | Alpha demo app walkthrough, headless capture, package/project contracts |
| App-owned loop | [15-app-owned-loop.md](15-app-owned-loop.md) | Root `engine` input/frame/render helpers for custom app-owned loops |

## Public API Contract

The renderer crate is in alpha. Public symbols are split into tiers so beginner
docs do not imply that every reachable export is the supported beginner facade.
Existing root exports are preserved for compatibility, but the alpha beginner
path is the small set used by the quickstart, lifecycle, scene, asset, input,
debug, and runtime-launcher chapters.

`renderer::prelude` is the supported beginner import path for renderer-owned
example loops. It is intentionally smaller than the crate root. `engine::prelude`
is the thin app-owned runtime facade for custom app loops that own input,
events, frame clock, and camera state. `renderer::api` remains the explicit
renderer facade namespace, and the renderer crate root re-exports that facade
plus a few older helper groups used by current tests and examples. Those
root-only renderer helper groups are compatibility public, not beginner-stable
API.

```rust
// Alpha beginner facade examples
use renderer::prelude::{Renderer, RendererConfig, RendererError, Scene};

// Handles
pub use data::handles::{MeshHandle, TextureHandle, MaterialHandle, EnvironmentHandle};

// Input (re-exported from the input crate)
pub use input::{InputSystem, InputSnapshot, InputLayer, ActionMap, ...};

// Events (re-exported from engine_events)
pub use engine_events::{EventBus, EventRecorder, EngineEvent, EventStage, ...};

// Debug
pub use debug_ui::{DebugViewId, DebugViewDescriptor, DebugTimingRow, ...};
```

The full facade re-export list is at
[`src/renderer/src/api/mod.rs`](../../src/renderer/src/api/mod.rs). Extra
root-only compatibility exports are listed in
[`src/renderer/src/lib.rs`](../../src/renderer/src/lib.rs).

| Tier | Feature gate | Current exposure | Intended use | Stability |
|------|-------------|------------------|--------------|-----------|
| Alpha beginner facade | default | `renderer::prelude::{Renderer, RendererConfig, Scene, AssetManager, LoadTicket, InputSystem, FrameCaptureRequest, EventBus, ...}` with the same names still available under `renderer::api` and, for compatibility, the crate root. See [`src/renderer/src/api/prelude.rs`](../../src/renderer/src/api/prelude.rs). | Supported alpha path for opening a renderer, creating or loading scenes, loading assets, updating input, rendering frames, and using debug/capture controls. | alpha supported |
| App-owned runtime facade | default | `engine::{camera, events, frame, input, render}` modules plus `engine::prelude` for common app imports. | Custom Rust app loops that own input dispatch, lifecycle events, camera/controller state, and submit caller-provided `CameraView` data to renderer. Raw support crates remain directly usable. | alpha supported |
| Safe extension points | default | Public but not in prelude: `RenderHook`, `RenderHookContext`, `DebugViewDescriptor`, `DebugViewCallback`, `DebugTimingSnapshot`, `FrameCaptureScheduler`, `Camera`, `FPSController`, `SceneWorld`, `CommandHistory`, scene commands, `Aabb`, `Frustum`, `Ray`, camera controllers, `AnimationPlayer`. | Intermediate users needing app logic, telemetry, debug UI, lightweight frame observation. | alpha supported with constraints |
| Advanced interop | `advanced-interop` (opt-in) | `renderer::api::advanced` (unsafe `renderer_core_mut`), `renderer::rendergraph` (`RenderGraph`, `RenderPassNode` trait, `RenderGraphContext`). | Explicit opt-in for engine-internal experiments and expert diagnostics. | **alpha unstable** — no API compatibility guarantee across sprints |
| Raw backend escape hatch | `advanced-interop` + `unsafe` | `renderer_core_mut()` returns `&mut VkRenderCore`. `RenderGraphContext` no longer exposes unrestricted core recording access. | Internal escape hatch. Bypasses all facade invariants (synchronization, descriptor lifecycle, swapchain safety). Not a normal user path. | unstable escape hatch |
| Internal implementation detail | N/A | Private renderer modules and public-looking implementation concepts reached only through compatibility paths. | Do not document as new user-facing APIs in beginner docs. | N/A |
| Deferred gaps | N/A | Larger project runtime, material override, custom rendergraph pass registration, and generated app-template work not implemented in this alpha surface. | Document as future work only when needed; do not imply current support. | N/A |

## Runtime Launcher

Data-driven projects run through the root `engine` launcher:

```sh
cargo run -- --project apps/dungeon_dogfood/engine.project.toml
```

Headless visual validation uses the offscreen draw target:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/runtime-launcher/headless-draw
```

Renderer examples remain useful diagnostics and API references. Custom Rust applications live under `apps/<name>` and run with `cargo run -p <app>`. `apps/dungeon_dogfood` is the current app-owned runtime proof: it owns input/events/frame/camera state and renders with caller-provided views. For an off-workspace compile-first starting point, `engine_pack new-app` generates a standalone support-crate scaffold that uses public `engine_events`, `input`, and `physics` dependencies without editing the root workspace. The alpha event contract is available through the renderer facade and the root app facade. The standalone `physics` crate provides durable ID descriptors, basic collider shapes, ray queries, contact records, and helpers that translate physics records into `engine_events` payloads. The standalone `audio` crate provides durable clip IDs, device-independent load/probe paths, explicit device-backed playback, package/scene validation, and an opt-in dogfood proof. Runtime scene-to-physics loading, editor collision/audio authoring UI, dynamic Rust hot reload, production scripting runtime scheduling, package-level script assets, production audio mixing/spatialization/streaming, and renderer-window generated app templates are deferred.

## Canonical Renderer Example

> **Compatibility note:** This section describes the **renderer compatibility path** (renderer-owned input/camera/frame state). For custom apps, prefer the **current app-owned path** (app-owned `InputSystem`, `FPSController`, and `render_scene_with_view`). See [15-app-owned-loop.md](15-app-owned-loop.md) for the full app-owned loop guide.

Every renderer-owned demo follows the same compatibility pattern (see [`src/renderer/examples/common/mod.rs`](../../src/renderer/examples/common/mod.rs)):

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

## Culling and Directional Shadow Defaults

New `Scene` values enable frustum culling by default. Call
`scene.set_frustum_culling(false)` only for diagnostics or content whose geometry
extends beyond the current node proxy bounds. A scene may own one
`DirectionalLight`; PBR materials use it with one fixed 2048² frame-local shadow
map. Shadow resolution, filtering, cascades, and per-light shadow toggles are not
publicly configurable in this gate.

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
