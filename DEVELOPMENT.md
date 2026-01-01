# Development Documentation

## Current Focal Point
**Refactoring `VkRender` Initialization**: The `VkRender::new` function is excessively large and complex. We are in the process of breaking it down into smaller, reusable helper functions to improve readability and maintainability.
-   **Completed:** Created `vk_init_helpers` and refactored command pool/buffer creation.
-   **Completed:** Extracted synchronization objects (fences/semaphores) and host buffer initialization into `vk_init_helpers.rs`.
-   **Next Steps:** Continue extracting logical blocks from `VkRender::new` (e.g., descriptor allocators, ImGui init) into helper functions.

## Recent Changes
-   **Refactor:** Introduced `src/renderer/src/vulkan/vk_init_helpers.rs` to consolidate command pool and buffer creation logic.
-   **Refactor:** Updated `VkRender::new` and `init_present_pools` in `src/renderer/src/vulkan/vk_render.rs` to use the new helper, significantly reducing boilerplate.
-   **Refactor:** Added `create_sync_objects` and `create_host_buffer` to `vk_init_helpers.rs` and integrated them into `VkRender::new`.

## Architecture Overview
The engine uses a `VkRender` struct as the central hub for all Vulkan state.
-   **Render Graph:** Rendering logic is decoupled into a `RenderGraph` (in `render_graph.rs`) managing `RenderPass` traits.
-   **Initialization:** Core Vulkan setup is handled in `vk_init.rs` and the new `vk_init_helpers.rs`.
-   **Data Management:** `VkDataCache` manages assets (textures, meshes).

## Known Issues
-   `VkRender::new` is still too large.
-   `generate_environment` function contains duplicated code for Irradiance and Prefilter map generation.

## Agent Handover
Please continue the refactoring of `VkRender::new`. Look for distinct initialization steps that can be extracted into `vk_init_helpers.rs` or similar utility modules. Verify changes with `cargo check`.
