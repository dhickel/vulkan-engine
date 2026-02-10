# Audit and Review Report

## 1. Architectural Review

### 1.1 Architecture Overview
The project is a Rust-based engine utilizing **Ash (Vulkan)** for its rendering backend. The architecture is decoupled, separating state management from rendering:
*   **GameLogic & InputManager**: Handle application state and user input.
*   **VkRender**: Manages the graphics pipeline and rendering commands.

### 1.2 Frame Lifecycle
The frame rendering process is triggered by `WindowEvent::RedrawRequested` and follows this sequence:
1.  **Scene Update**: `update_scene()` updates camera matrices and traverses the scene tree to populate a `DrawContext`.
2.  **Acquisition**: The engine waits for the current frame's fence and acquires the next swapchain image.
3.  **Rendering Passes**:
    *   **Skybox**: Rendered (order depends on depth settings) via `draw_skybox()`.
    *   **Geometry**: `draw_geometry()` binds pipelines and descriptors (SceneData, Joints, Materials) to render objects in `DrawContext`.
    *   **Blit**: The internal draw image is copied to the swapchain image using `vk_util::blit_copy_image_to_image()`.
    *   **UI**: `draw_imgui()` renders the ImGui overlay directly onto the swapchain image.
4.  **Submission & Present**: Command buffers are submitted with semaphores, followed by image presentation.

### 1.3 Key Components
*   **`VkRender`**: The core struct managing the Vulkan device, swapchain, and high-level rendering loop (`src/renderer/src/vulkan/vk_render.rs`).
*   **`VkDataCache`**: Centralized asset manager (`src/renderer/src/data/data_cache.rs`) handling `MeshCache`, `TextureCache`, and `EnvironmentCache`. It manages asynchronous loading and GPU allocation.
*   **`TextureCache`**: Handles texture metadata, format conversion (e.g., to RGBA8), and descriptor set management for PBR materials.
*   **`MeshCache`**: Manages vertex and index buffers using a sub-allocation strategy.

---

## 2. Code Review & Findings

### 2.1 Critical Synchronization Bug (Screen Artifacting/Flashing)
**Location:** `src/renderer/src/vulkan/vk_render.rs` (Lines ~1655)
**Issue:** The rendering loop submits commands to the graphics queue with a semaphore wait on `vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR`.
```rust
let wait_info = [vk_util::semaphore_submit_info(
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR, // <--- PROBLEM
    frame_sync.swap_semaphore,
)];
```
However, the *first* operation performed on the swapchain image (`present_image`) is a **Blit** operation (Transfer stage), not a Color Attachment Output.
```rust
// In render():
vk_util::blit_copy_image_to_image(..., present_image, ...);
```
**Impact:** The GPU is allowed to execute the Blit (`TRANSFER` stage) *before* the swapchain image is acquired (signaled at `COLOR_ATTACHMENT_OUTPUT`). This causes the engine to write to the swapchain image while it is potentially still being read by the presentation engine, leading to screen tearing, flashing, or garbage artifacts (like the skybox pattern appearing corrupt).

**Fix:** The wait stage must include `vk::PipelineStageFlags2::TRANSFER` or use `vk::PipelineStageFlags2::ALL_COMMANDS` to ensure the image is acquired before *any* operation starts.

### 2.2 Black Models & Texture Issues
**Location:** `src/renderer/src/data/data_cache.rs`
**Observation:** The codebase contains active debug code (`vk_debug::capture_and_save_image_view`) that dumps loaded textures to `debug_textures/`. The user reports black models.
**Potential Causes:**
1.  **Missing IBL/Environment Lighting:** The PBR shader relies on Irradiance and Prefilter maps. If the Skybox/Environment map fails to generate (e.g., due to the sync bug above or `upload_skybox` issues), the ambient lighting will be black.
2.  **Texture Format Conversion:** `TextureCache` attempts to convert unsupported formats to `R8G8B8A8_UNORM`. If `assimp` provides a format that is technically supported by Vulkan but not by the shader's sampler (e.g. BGR vs RGB), it might result in incorrect sampling (though usually not pure black).
3.  **Hardcoded Timeout:** `host_buffer.await_done(1000)` waits only 1 second for texture uploads. Large models might timeout, leaving textures uninitialized (black).

### 2.3 Skybox Pattern
**Location:** `src/renderer/src/vulkan/vk_util.rs` (`upload_skybox`)
**Issue:** The skybox upload calculates buffer offsets manually: `i * (tex_meta.bytes.len() / 6)`.
If the source bytes are not perfectly aligned or if `tex_meta.bytes` contains padding, this division might slice the data incorrectly, causing a "striped" or "patterned" look on the skybox faces.

---

## 3. Refactoring Targets

### 3.1 Fix Synchronization
*   **Target:** `src/renderer/src/vulkan/vk_render.rs`
*   **Action:** Update `wait_info` in `render()` to wait on `vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR`.

### 3.2 Robust Texture Uploads
*   **Target:** `src/renderer/src/data/data_cache.rs`
*   **Action:**
    *   Remove or increase the hardcoded `1000` ms timeout in `allocate_textures`.
    *   Add error logging if texture dimensions are invalid (0x0).
    *   Verify `EnvMaps` generation succeeds; if IBL is black, models will look black.

### 3.3 Safety & Cleanup
*   **Target:** Global
*   **Action:**
    *   Replace `unsafe` blocks that span large sections with smaller, granular blocks.
    *   Replace `unwrap()` in `init` functions (e.g., `VkRender::new`) with proper `Result` propagation to avoid crashes on startup.
    *   Remove the `debug_textures` dump code in production builds to save performance/disk IO.

### 3.4 Data Validation
*   **Target:** `src/renderer/src/data/assimp_util.rs` (Recommend investigation)
*   **Action:** Ensure that loaded textures are actually populated. The current `Unloaded` state relies on `TextureMeta` being correct.