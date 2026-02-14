# Engine Repository Agent Guide

This guide is the top-level orientation for agents maintaining this repository.
Use it to understand where to work, then switch to module-level `AGENTS.md` files for deep package detail.

## Scope and Intent

- Language/runtime: Rust 2021, desktop Vulkan renderer.
- Primary runtime entrypoints: facade-first renderer examples under `src/renderer/examples/`.
- Root binary crate: `engine` (`src/main.rs`) is a migration stub that points to renderer examples.
- Internal package crates:
  - `src/renderer` (`renderer`) - rendering runtime and most engine logic.
  - `src/input` (`input`) - input event broadcast/listener layer.
- Internal engineering docs: `.internal-dev/*.md`.

## Repository Layout

- `Cargo.toml`: workspace root (`engine`, `src/input`, `src/renderer`).
- `src/main.rs`: migration stub that prints canonical renderer example commands.
- `src/renderer/`: rendering crate (Vulkan, asset loading, frame loop).
- `src/input/`: reusable input system crate.
- `.internal-dev/`: architecture notes, reviews, and maintenance docs.
- `debug_textures/`: optional debug image output.

## External Baseline

- This engine's glTF/PBR direction was developed with significant guidance from:
  - `https://github.com/SaschaWillems/Vulkan-glTF-PBR`
- Treat that repository as a reference baseline for shading model expectations, IBL flow, and material conventions when behavior is ambiguous.

## Reality Check (Current State)

- The codebase is functional but not fully hardened.
- There are known TODO/FIXME areas and a few high-risk patterns.
- Do not assume every cleanup path is implemented.
- `gltf_util.rs` is currently commented-out legacy code; Assimp path is active.

## Runtime Flow

1. Launch one of the facade-first renderer examples (`cargo run -p renderer --example ...`).
2. Example creates window/event loop and `Renderer`.
3. `Renderer::new()` initializes `VkRender`, caches, transfer system, pipelines, ImGui, and startup resources.
4. Event loop calls `Renderer::update_input(...)` then `Renderer::render_scene(...)` (or explicit frame API).
5. `VkRender::render()` performs per-frame acquire/update/record/submit/present.

## High-Level Module Insights

- `src/input`: broadcaster/listener input layer used by camera controller.
- `src/renderer/src/data`: scene graph, camera, asset metadata, caches, loaders.
- `src/renderer/src/vulkan`: Vulkan initialization, resource wrappers, allocators, frame synchronization, render orchestration.

## Known High-Risk Areas to Keep in Mind

- ID stability risk in data caches:
  - Core mesh/material/texture and scene-node paths now use slot+generation handles.
  - Remaining risk is concentrated in unchecked/deallocation edge paths and any code that bypasses handle validation contracts.
- Incomplete destructors:
  - Some `VkDestroyable` impls still contain `todo!()`.
- Render-path sharp edges:
  - Pipeline-switch logic in geometry draw path has known correctness risk.
  - Swapchain rebuild has explicit FIXME about old present image views.
- Heavy `unwrap()/panic!()` usage in hot paths and init paths.

Use module-level docs for exact files/lines and safe editing strategy.

## Documentation Contract

- Keep `.internal-dev/` and `AGENTS.md` docs in sync with major architecture changes.
- Add changelog entries to `.internal-dev/changelogs/` when requested.
- Keep top-level guide concise and architectural.
- Keep nested guides implementation-focused and opinionated about gotchas.
- `.internal-dev/` is intentionally gitignored in this repository; it is acceptable for planning/review docs there to remain local and untracked unless explicitly force-added for sharing.

## Where to Start for a Task

- Input or controls: `src/input/AGENTS.md`.
- Renderer-wide behavior: `src/renderer/AGENTS.md`.
- Scene/assets/materials/caches: `src/renderer/src/data/AGENTS.md`.
- Vulkan lifecycle/render/sync/memory: `src/renderer/src/vulkan/AGENTS.md`.
- Shader authoring and source lineage: `src/renderer/src/shaders/AGENTS.md`.

## Quick Commands

- Check compile: `cargo check`
- Check renderer crate: `cargo check -p renderer`
- Check renderer examples: `cargo check -p renderer --examples`
- Check input crate: `cargo check -p input`
- Runtime example entrypoints:
  - `cargo run -p renderer --example api_test`
  - `cargo run -p renderer --example demo_pbr`
  - `cargo run -p renderer --example demo_unlit`
  - `cargo run -p renderer --example demo_model_load`
  - `cargo run -p renderer --example demo_async_loading`
  - `cargo run` (prints migration guidance and exits; runtime removed from root binary)

## Headless Runtime Smoke (Terminal Agents)

- In headless agent sessions, renderer examples can still be launched from terminal.
- You will not see display output, but verbose startup/runtime logs are still visible.
- Always run runtime examples with bounded execution and auto-close after startup load completes.
- Preferred pattern:
  - `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_pbr`
  - `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_unlit`
  - `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_model_load`
  - `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_async_loading`
  - `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test`
- Treat successful startup logs plus no fatal errors before timeout as headless runtime smoke pass.

## Runtime Example Scenarios

- Canonical scenarios:
  - `demo_pbr`: startup PBR path.
  - `demo_unlit`: startup Unlit path.
  - `demo_model_load`: model load + scene fragment merge path.
  - `demo_async_loading`: deferred ticket polling + mount path.
  - `api_test`: explicit `begin_frame`/`render_scene_in_frame`/`end_frame` flow.
- Example-per-scenario binaries are the canonical runtime validation path.
- Do not leave long-running render loops active; terminate after load/smoke capture.

Note: runtime validation of rendering features requires a Vulkan-capable environment.
