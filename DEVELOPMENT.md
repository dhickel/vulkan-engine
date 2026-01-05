# Development Documentation

## Current Focal Point
**Refactoring `VkRender` Initialization**: The `VkRender::new` function is excessively large and complex. We are in the process of breaking it down into smaller, reusable helper functions to improve readability and maintainability.
-   **Completed:** Created `vk_init_helpers` and refactored command pool/buffer creation.
-   **Completed:** Extracted synchronization objects (fences/semaphores) and host buffer initialization into `vk_init_helpers.rs`.
-   **Completed:** Extracted descriptor allocator initialization and ImGui initialization into `vk_init_helpers.rs`.
-   **Next Steps:** Continue extracting logical blocks from `VkRender::new`. Specifically, look at `init_caches`, `init_descriptors` (which is already a function but outside impl), and the swapchain/surface creation logic if applicable. Also consider consolidating `init_present_pools` into the helper or keeping it if it's specific enough.

## Recent Changes
-   **Refactor:** Moved `create_descriptor_allocators` and `init_imgui` logic into `vk_init_helpers.rs`.
-   **Refactor:** Updated `VkRender::new` to use the new helpers, reducing inline boilerplate.
-   **Clean up:** Removed duplicated code in `vk_init_helpers.rs` and fixed visibility issues.

## Architecture Overview
The engine uses a `VkRender` struct as the central hub for all Vulkan state.
-   **Render Graph:** Rendering logic is decoupled into a `RenderGraph` (in `render_graph.rs`) managing `RenderPass` traits.
-   **Initialization:** Core Vulkan setup is handled in `vk_init.rs` and the new `vk_init_helpers.rs`.
-   **Data Management:** `VkDataCache` manages assets (textures, meshes).

## Known Issues
-   `VkRender::new` is still too large.
-   `generate_environment` function contains duplicated code for Irradiance and Prefilter map generation.

## Agent Handover
Please continue the refactoring of `VkRender::new`. The next logical chunks to tackle are the "Core Structures Creation" block (instance, surface, device, swapchain) which takes up a lot of space at the beginning of `new`.
