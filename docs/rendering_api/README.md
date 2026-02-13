# Rendering API Docs

## Audience and scope

This folder documents the renderer as it exists today in this repository.

Audience:
- Hobbyists and students comfortable with Rust and general engine code.
- Developers newer to Vulkan/graphics pipeline details.

Scope:
- Integration path.
- Data contracts.
- Frame flow.
- Pipelines/descriptors/shaders.
- Synchronization.
- Assets and environment maps.
- Gotchas and debugging.
- Practical recipes with code examples.

Best practice:
- Read these docs with the source open, especially `src/renderer/src/lib.rs` and `src/renderer/src/vulkan/vk_render.rs`.

Learn more:
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/

## Public API status (important)

Current external API surface is minimal:
- Stable external entrypoint: `renderer::run()`.

Most scene/asset/submission APIs shown here are currently **in-tree/internal usage patterns** (for engine contributors or fork users), not polished external crate APIs yet.

Best practice:
- Treat examples in this folder as current implementation contracts and dogfood guidance.
- If you need stronger external APIs, track the gap list in `.internal-dev/reviews`.

Learn more:
- Entry point source: `src/main.rs`, `src/renderer/src/lib.rs`

## Current rendering model (alpha)

- Vulkan 1.3 style dynamic rendering (`vkCmdBeginRendering`, no legacy render pass objects).
- Forward-style frame with explicit pass order.
- Offscreen draw + depth targets, then copy/blit into swapchain image.
- ImGui overlay on the present image.
- PBR metallic-roughness and Unlit material paths.
- Skybox + generated IBL maps (irradiance + prefilter + BRDF LUT).

Best practice:
- Treat the current model as explicit and deterministic. Avoid hidden state transitions when extending it.

Learn more:
- Dynamic rendering overview: https://github.khronos.org/Vulkan-Site/guide/latest/dynamic_rendering.html

## Document map

1. `01_quick_start_and_integration.md`
2. `02_renderer_api_contracts.md`
3. `03_frame_execution_flow.md`
4. `04_pipelines_descriptors_and_shaders.md`
5. `05_synchronization_fences_semaphores_barriers.md`
6. `06_assets_environments_and_uploads.md`
7. `07_gotchas_best_practices_and_debugging.md`
8. `08_learning_resources.md`
9. `09_examples_and_recipes.md`

Recommended reading order for new users:
1. Quick Start
2. API Contracts
3. Frame Execution Flow
4. Examples and Recipes
5. Pipelines/Descriptors/Shaders
6. Synchronization
