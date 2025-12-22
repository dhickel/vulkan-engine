# Development Documentation

## Project Overview
This project is a Vulkan-based rendering engine written in Rust. It currently supports PBR rendering, glTF loading, skeletal animation, and basic input handling. The project is designed to be modular, with a separation between core rendering logic, data management, and input handling.

## Current Focal Point
**Goal:** Update dependencies and ensure the project runs.
**Next Steps:**
1.  Review and document the existing codebase (In Progress).
2.  Update `Cargo.toml` dependencies to recent versions.
3.  Fix compilation errors resulting from dependency updates.
4.  Verify the project runs and renders the demo scene.

## Architecture

### Modules
-   **`renderer`**: The core rendering crate.
    -   **`vulkan`**: Contains all Vulkan-specific logic.
        -   `vk_init`: Vulkan initialization (Instance, Device, Swapchain).
        -   `vk_render`: Main rendering loop, frame management, and drawing commands.
        -   `vk_types`: Core structs and enums wrappers.
        -   `vk_descriptor`: Descriptor set abstraction and management.
        -   `vk_pipeline`: Pipeline creation and caching.
        -   `vk_storage`: Buffer management and sub-allocation.
        -   `vk_util`: Utility functions for Vulkan commands.
    -   **`data`**: Asset management and data structures.
        -   `gpu_data`: Structs mirroring GPU data layouts (UBOs, PushConstants).
        -   `data_cache`: Caches for meshes, textures, pipelines, and descriptors.
-   **`input`**: Handles keyboard and mouse input.

### Logic Flow
1.  **Initialization**: `vk_init` sets up the Vulkan instance and device. `VkRender::new` initializes the renderer state, including swapchain, command pools, and descriptor allocators.
2.  **Asset Loading**: Assets (glTF) are loaded into `VkDataCache` (MeshCache, TextureCache).
3.  **Render Loop (`VkRender::render`)**:
    -   Acquire next swapchain image.
    -   Update scene data (camera, object transforms).
    -   Begin command buffer.
    -   Transition images/buffers.
    -   Draw Skybox.
    -   Draw Geometry (PBR pass).
    -   Draw UI (ImGui).
    -   Submit command buffer.
    -   Present image.

### Memory Management
-   Uses `vk-mem` (VMA) for memory allocation.
-   `VkSubAllocator` manages sub-allocations within larger buffers to reduce overhead.
-   `VkDynamicDescriptorAllocator` handles descriptor set allocation with pool rotation.

## Known Issues / Improvements
-   **Dependencies**: `glam`, `winit`, `ash`, `vk-mem` versions are outdated/mixed.
-   **Cleanup**: Some commented-out code and `TODO`s scattered throughout.
-   **Error Handling**: Some `unwrap()` calls should be converted to proper error propagation.
-   **Synchronization**: Fence handling in `VkFenceQueue` and `VkHostBuffer` needs review for robustness.

## Development Patterns
-   **Structs**: extensive use of builder patterns for Vulkan structs.
-   **Safety**: Unsafe blocks are localized to Vulkan API calls.
-   **Concurrency**: Asset loading is threaded (seen in `VkRender::new`), using channels/latches for synchronization.

## Recent Changes
-   Created `AGENTS.md` and `DEVELOPMENT.md`.
-   Started codebase review.
