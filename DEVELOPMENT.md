# Development Documentation

## Current Focal Point
**Refactoring `VkRender` Initialization**: The `VkRender::new` function is excessively large and complex. We are in the process of breaking it down into smaller, reusable helper functions to improve readability and maintainability.
-   **Completed:** Created `vk_init_helpers` and refactored command pool/buffer creation.
-   **Completed:** Extracted synchronization objects (fences/semaphores) and host buffer initialization into `vk_init_helpers.rs`.
-   **Completed:** Extracted `init_imgui` and `create_descriptor_allocators` into `vk_init_helpers.rs` and updated `VkRender::new` to use them.
-   **Next Steps:** Continue extracting logical blocks from `VkRender::new`. Identify remaining large blocks (e.g., render pass/frame buffer setup if not already done, or pipeline creation) and move them to helpers.

## Recent Changes
-   **Refactor:** Cleaned up `vk_init_helpers.rs` to remove duplicate imports and definitions.
-   **Refactor:** Extracted descriptor allocator initialization into `vk_init_helpers::create_descriptor_allocators`.
-   **Refactor:** Updated `VkRender::new` to use `vk_init_helpers::init_imgui` and `vk_init_helpers::create_descriptor_allocators`, significantly simplifying the function.
-   **Refactor:** Updated `VkRender::new` to use `vk_init_helpers::create_sync_objects`, `vk_init_helpers::create_host_buffer`, and `vk_init_helpers::init_imgui`.
-   **Refactor:** Introduced `create_sync_objects`, `create_host_buffer`, and `init_imgui` in `src/renderer/src/vulkan/vk_init_helpers.rs`.
-   **Refactor:** Updated `VkRender::new` and `init_present_pools` in `src/renderer/src/vulkan/vk_render.rs` to use the new helper, significantly reducing boilerplate.

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
