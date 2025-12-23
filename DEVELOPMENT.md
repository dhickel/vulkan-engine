# Development Documentation

This document serves as the primary source of truth for the engine's architecture, current state, and development direction. All contributors (human and agent) must keep this file up-to-date.

## 1. Project Overview

This is a Vulkan-based 3D rendering engine written in Rust. It currently supports a PBR (Physically Based Rendering) pipeline, glTF asset loading, and a modular architecture. The design philosophy emphasizes explicit resource management, thread safety for asset loading, and type safety where possible using Rust's type system.

**Key Technologies:**
-   **Language:** Rust
-   **Graphics API:** Vulkan (via `ash` crate)
-   **Memory Management:** `vk-mem` (Vulkan Memory Allocator)
-   **Math:** `glam`
-   **Windowing:** `winit`
-   **Asset Loading:** `gltf`, `russimp`

---

## 2. Current Focal Point

**Goal:** Expand PBR & Optimization.
**Status:** IBL Integration complete. Environment maps (Irradiance/Prefilter) are now cached to disk (`assets/cache/env_maps`) to avoid runtime generation cost on subsequent runs.
Render logic moved to `RenderGraph` with passes (`GeometryPass`, `SkyboxPass`, `UiPass`, `CopyPass`).

**Immediate Next Steps:**
1.  **Further Cleanup:** Continue replacing `unwrap()` with proper error handling and fixing hardcoded paths.
2.  **ImGui Integration:** Improve UI layer abstraction.

---

## 2.1 Configuration
The engine now supports loading configuration from `config.toml` in the working directory.
It configures shader paths and default asset paths.
See `src/renderer/src/config.rs` for the structure.

## 3. Architecture Deep Dive

The engine is split into three main crates/modules:
1.  `renderer`: The core graphics engine.
2.  `input`: Input handling library.
3.  `engine` (root): The application entry point (`main.rs`) and game logic.

### 3.1. Renderer Module (`src/renderer`)

This is the heart of the engine. It is further divided into `vulkan` (API abstraction) and `data` (resources).

#### 3.1.1. Vulkan Core (`src/renderer/src/vulkan`)

*   **`vk_init.rs`**:
    *   **Purpose:** Boilerplate Vulkan initialization.
    *   **Key Functions:** `init_instance`, `create_swapchain`, `create_logical_device`.
    *   **Pattern:** Uses builder patterns for `vk::*CreateInfo` structs.
*   **`vk_render.rs` (`VkRender` struct)**:
    *   **Purpose:** The main driver. It owns the Device, Swapchain, and all global resources.
    *   **Flow:** `new()` initializes everything -> `render()` is called per frame.
    *   **Render Loop:**
        1.  `update_scene()`: Updates camera/object UBOs.
        2.  `presentation.get_next_frame()`: Rotates double-buffered resources.
        3.  `vkDeviceWaitSemaphores`: Waits for the previous frame.
        4.  `vkAcquireNextImageKHR`: Gets swapchain image.
        5.  `vkQueueSubmit`: Submits rendering commands.
        6.  `vkQueuePresentKHR`: Presents the image.
*   **`vk_types.rs`**:
    *   **Purpose:** Wrapper structs for Vulkan handles to provide clearer ownership semantics (e.g., `VkImageAlloc` vs raw `vk::Image`).
    *   **Key Structs:** `VkFrame` (per-frame data), `VkBuffer`, `VkImageAlloc`.
*   **`vk_descriptor.rs`**:
    *   **Purpose:** Abstraction over Descriptor Sets and Pools.
    *   **Mechanics:**
        *   `VkDescriptorAllocator`: Static pool.
        *   `VkDynamicDescriptorAllocator`: dynamic pool that grows by creating new pools as needed. Crucial for handling variable numbers of materials.
        *   `VkDescriptorWriter`: Helper to batch writes to descriptor sets.
*   **`vk_pipeline.rs`**:
    *   **Purpose:** Pipeline creation and caching.
    *   **Pattern:** `PipelineBuilder` struct simplifies the 100+ lines of code needed to create a `vk::Pipeline`.
*   **`vk_storage.rs`**:
    *   **Purpose:** Memory management strategies.
    *   **Key Component:** `VkSubAllocator`.
    *   **Strategy:** Allocates massive GPU buffers (e.g., 256MB) and sub-allocates small chunks (vertices, indices) from them. This reduces the number of `vkAllocateMemory` calls, which is a performance best practice.
    *   **Fragmentation:** Uses a `FreeChunk` list to track available space.
*   **`render_graph.rs`**:
    *   **Purpose:** High-level rendering abstraction.
    *   **Status:** Initial scaffolding. Goal is to manage render passes and dependencies.

#### 3.1.2. Data Management (`src/renderer/src/data`)

*   **`gpu_data.rs`**:
    *   **Purpose:** Defines structs that match GLSL/HLSL std140/std430 layouts.
    *   **Structs:** `Vertex`, `MaterialMeta`, `SceneDataUBO`, `VkModelPushConsts`.
*   **`data_cache.rs`**:
    *   **Purpose:** Asset Manager.
    *   **Components:** `TextureCache`, `MeshCache`, `VkShaderCache`.
    *   **Flow:**
        *   `add_mesh(MeshMeta)` -> returns `u32` ID.
        *   `allocate_id(id)` -> Uploads data to GPU (using `VkSubAllocator`) and returns `VkMeshBuffers`.
    *   **Threading:** Designed to allow loading assets on background threads while the main thread renders. Uses `VkHostBuffer` for async transfer queue operations.

### 3.2. Input Module (`src/input`)

*   **Purpose:** Decouples windowing events (winit) from game logic.
*   **Mechanism:** `InputManager` receives events, filters them, and broadcasts to registered listeners (`MousePosListener`, `KeyboardListener`).
*   **Usage:** The camera controller (`FPSController`) implements these listener traits to move the camera.

---

## 4. Key Logic Flows

### 4.1. Asset Upload Pipeline
1.  **Loading:** `gltf_util` or `assimp_util` reads file off disk into CPU structs (`MeshMeta`, `TextureMeta`).
2.  **Caching:** `TextureCache::add_texture` stores CPU data and returns an ID.
3.  **Allocation:** `TextureCache::allocate_textures` is called.
    *   It checks `VkHostBuffer` (staging buffer) availability.
    *   Copies CPU data -> Staging Buffer.
    *   Records command buffer: `Staging -> Device Local Image`.
    *   Submits to Transfer Queue (or Graphics queue if mips needed).
    *   Fences/Semaphores synchronize access.

### 4.2. Descriptor Management
*   **Global Sets:** Scene data (Camera, Lights) is in Set 0. Updated once per frame per frame-in-flight.
*   **Material Sets:** Textures are in Set 1 (or 2). Allocated via `VkDynamicDescriptorAllocator`.
*   **Object Data:** Push Constants are used for Model Matrix and buffer addresses (Bindless-style vertex pulling).

---

## 5. Known Issues & Technical Debt

*   **Dependencies:** While updated, some logic might still rely on older crate behaviors. `glam` and `winit` versions should be kept in sync across crates.
*   **Error Handling:** Many places use `unwrap()`. These should be converted to `Result` types for robustness.
*   **Synchronization:** The `VkFenceQueue` logic is simple. A more robust frame-graph or render-graph approach would handle barriers better.
*   **Hardcoded Paths:** Shader and asset paths are often hardcoded strings. Needs a configuration system.
*   **ImGui:** Integration is basic. Needs a proper UI layer abstraction.

---

## 6. Agent Instructions

When working on this project:
1.  **Context:** Read this file first.
2.  **Conventions:**
    *   Use `///` for struct/function documentation.
    *   Prefix Vulkan wrappers with `Vk`.
    *   Prefer `Result<T, String>` over panics.
3.  **State:** Update section "2. Current Focal Point" when you complete a task.
