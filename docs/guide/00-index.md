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
| [06](06-input.md) | Input System | Action maps, layer priority, snapshots, `InputActionEventEmitter`, platform routing |
| [07](07-events-and-lifecycle.md) | Events & Lifecycle | `EventBus`, stages, lifecycle events, typed subscribers, recorders, dispatch reports |
| [08](08-scene-construction.md) | Scene Construction | Nodes, transforms, meshes with bounds, materials, lights, fragments, culling |
| [09](09-asset-pipeline.md) | Asset Pipeline | `AssetManager`, sync/deferred loading, handles, tickets, procedural upload, geometry queries |
| [10](10-physics.md) | Physics (Alpha) | Rigid bodies, colliders, ray casting, contact events, convex hull validation |
| [11](11-audio.md) | Audio (Alpha) | Clip loading, device-backed playback, device-independent probe, event bridging |
| [12](12-debug-and-diagnostics.md) | Debug & Diagnostics | Logging, timing capture, headless capture, debug UI, validation layers, asset/collider validation |
| [13](13-packaging-and-distribution.md) | Packaging & Distribution | `engine_pack`: scaffolding, validation, asset scanning, packed distribution |

### Part III — Real-World Reference

Case studies of two complete workspace applications. Read these to see how the concepts from Parts I and II compose into real apps.

| # | Chapter | What It Covers |
|---|---------|----------------|
| [14](14-dungeon-dogfood-walkthrough.md) | Case Study: Dungeon Dogfood | App-owned loop, procedural dungeon, AABB collision, mesh-collider bridge, audio telemetry, headless capture |
| [15](15-voxel-demo-walkthrough.md) | Case Study: Voxel Demo | v2 presets/config, deterministic generation, MC33 partition, PBR materials, imgui editor, latest-wins regeneration |

### Part IV — Compatibility & Troubleshooting

Reference chapters for distinguishing supported API paths and diagnosing common issues.

| # | Chapter | What It Covers |
|---|---------|----------------|
| [16](16-api-compatibility-guide.md) | API Compatibility Guide | Side-by-side ownership tables: app-owned vs renderer compatibility paths, migration labels, prelude comparison |
| [17](17-troubleshooting.md) | Troubleshooting | Compile errors, init failures, runtime outcomes, terminal errors, subsystem failures, headless vs WSI evidence |

## Conventions

- **Checkpoint provenance**: Every code block in Part I chapters matches `examples/guide_app/src/main.rs` exactly (provenance ID prefix `CP-`). Excerpt blocks are labeled with the omitted context.
- **Renderer examples are diagnostic**: References to `demo_pbr`, `api_test`, etc. are labeled as **compatibility/diagnostic** at first use — they test the renderer, not your app.
- **Compilation is not visual proof**: A snippet that compiles may still look wrong at runtime. Visual claims require GPU evidence (headless captures or WSI observation), not compilation alone.
- **WSI requires a display**: The checkpoint app needs a real GPU with windowing system integration. Headless capture examples are documented separately.
