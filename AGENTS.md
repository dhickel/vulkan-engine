# Engine Repository Agent Guide

This guide is the top-level orientation for agents maintaining this repository.
Use it to understand where to work, then switch to module-level `AGENTS.md` files for deep package detail.

## Scope and Intent

- Language/runtime: Rust 2021, desktop Vulkan renderer.
- Primary binary crate: `engine` (`src/main.rs`), which calls into `renderer::run()`.
- Internal package crates:
  - `src/renderer` (`renderer`) - rendering runtime and most engine logic.
  - `src/input` (`input`) - input event broadcast/listener layer.
- Internal engineering docs: `.internal-dev/*.md`.

## Repository Layout

- `Cargo.toml`: root crate plus path dependency on `renderer`.
- `src/main.rs`: minimal entrypoint.
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

1. `src/main.rs` calls `renderer::run()`.
2. `renderer::run()` creates window/event loop and `InputManager`.
3. `VkRender::new()` initializes Vulkan, caches, transfer system, pipelines, ImGui, default scene.
4. Event loop updates input/controller and calls `app.render(frame_number)`.
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
- Check input crate: `cargo check -p input`
- Runtime debug selector (startup material/pipeline testing):
  - `cargo run -- debug_runtime testpbr`
  - `cargo run -- debug_runtime testunlit`
  - `cargo run -- --debug-runtime=testunlit`

## Runtime Debug Scenarios

- `debug_runtime` is the runtime test selector entrypoint for controlled render-path validation.
- Current scenarios:
  - `testpbr`: keep startup scene on PBR material/pipeline path.
  - `testunlit`: force startup scene materials to Unlit pipeline path.
- This selector is intentionally extensible for future test scenarios; when adding new modes, keep this file and `src/renderer/AGENTS.md` in sync.

Note: runtime validation of rendering features requires a Vulkan-capable environment.
