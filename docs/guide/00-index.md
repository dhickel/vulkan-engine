# Engine Guide

> End-user documentation for building applications with the Vulkan rendering engine.

## Audience

Rust developers who want to build real-time 3D applications using this engine. You should be comfortable with Rust but do not need prior Vulkan or graphics programming experience — graphics concepts are explained as they appear.

## Three-Tier Documentation Architecture

| Tier | Directory | Audience | Purpose |
|------|-----------|----------|---------|
| **Guide** | `docs/guide/` (you are here) | App builders | Cumulative learning path, tutorials, how-to |
| **API Reference** | [`docs/api/`](../api/00-index.md) | Runtime contract readers | Facade contracts, type signatures, feature flags |
| **Internal** | [`docs/internal/`](../internal/00-index.md) | Engine maintainers | Architecture decisions, rendergraph, Vulkan internals |

When a guide chapter needs the exact signature of a function or the complete list of error variants, it links to the API reference rather than duplicating it. When it mentions an engine-internal design choice, it links to the internal docs.

## Chapter Map

### Part I — Cumulative Path (read in order)

Each chapter builds on the previous one. Start here if you are new to the engine.

| # | Chapter | What You'll Do |
|---|---------|----------------|
| [01](01-getting-started.md) | Getting Started | Install prerequisites, clone the repo, verify with a renderer compatibility smoke test |
| [02](02-architecture-overview.md) | Architecture Overview | Understand workspace crates, the launcher + facade, app-owned vs renderer-owned responsibilities, and how to navigate the three doc tiers |
| [03](03-building-your-first-app.md) | Building Your First App | Run `engine_pack new-app`, inspect the renderer-free scaffold, then meet the maintained windowed checkpoint |
| [04](04-app-owned-loop.md) | The App-Owned Loop | Walk through the complete checkpoint source: platform routing, frame boundary, fixed-step update, `CameraView`, render submission, error handling |
| [05](05-renderer.md) | Working with the Renderer | Renderer init, startup scene, resize, asset pumping, frame outcomes, terminal errors, and where to find API-level details |

### Part II — Independently Adoptable Subsystems

Use these chapters when you need a specific capability. They do not depend on reading Part I in order.

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| [06](06-input-and-actions.md) | Input & Actions | Action maps, layer priority, snapshots, `InputActionEventEmitter` |
| [07](07-scene-graph.md) | Scene Graph | Nodes, transforms, meshes, materials, fragments, culling |
| [08](08-asset-loading.md) | Asset Loading | `AssetManager`, handles, load tickets, packages, environments |
| [09](09-events-and-lifecycle.md) | Events & Lifecycle | `EventBus`, stages, lifecycle events, recorders, subscribers |
| [10](10-camera-and-view.md) | Camera & View | `Camera`, `FPSController`, `CameraView`, `camera_view_for_size` |
| [11](11-frame-timing.md) | Frame Timing | `FrameClock`, `FixedStepClock`, fixed-step simulation vs display rate |
| [12](12-debug-and-capture.md) | Debug & Capture | Debug UI, timing capture, frame capture for visual validation |
| [13](13-renderer-configuration.md) | Renderer Configuration | `RendererConfig`, asset policies, visual tuning, validation layers |

### Part III — Real-World Reference

| # | Chapter | What It Covers |
|---|---------|----------------|
| [14](14-dogfood-case-study.md) | Case Study: Dungeon Dogfood | Full walkthrough of the dogfood app — architecture, collision, audio bridge, headless capture |
| [15](15-voxel-demo-case-study.md) | Case Study: Voxel Demo | Procedural cave generation, MC33 meshing, configurable presets, regeneration |

### Part IV — Compatibility & Troubleshooting

| # | Chapter | What It Covers |
|---|---------|----------------|
| [16](16-compatibility.md) | Renderer Compatibility Paths | Distinguishing renderer-owned `update_input`/`render_scene` examples from app-owned loops; when to use each |
| [17](17-troubleshooting.md) | Troubleshooting | Common errors, Vulkan setup issues, resize problems, asset loading failures |

## Conventions

- **Checkpoint provenance**: Every code block in Part I chapters matches `examples/guide_app/src/main.rs` exactly (provenance ID prefix `CP-`). Excerpt blocks are labeled with the omitted context.
- **Renderer examples are diagnostic**: References to `demo_pbr`, `api_test`, etc. are labeled as **compatibility/diagnostic** at first use — they test the renderer, not your app.
- **Compilation is not visual proof**: A snippet that compiles may still look wrong at runtime. Visual claims require GPU evidence (headless captures or WSI observation), not compilation alone.
- **WSI requires a display**: The checkpoint app needs a real GPU with windowing system integration. Headless capture examples are documented separately.
