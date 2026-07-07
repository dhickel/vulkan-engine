 # API-to-Backend Handoff

## 1. Purpose & Audience
This chapter is for contributors changing how scene-facing API state becomes Vulkan work, especially around ownership boundaries between `api::Renderer`, `SceneWorld`, and `VkRender`.

## 2. Where This Fits in Engine Flow
Current handoff path:
`Renderer::render_scene(...)` -> `Renderer::render_scene_internal(...)` -> `Scene::build_submission()` -> `VkRender::render_with_hooks(...)` -> `VkRenderCore::render_with_hooks(...)` -> rendergraph passes.

Caller-view handoff path:
app/root facade camera state -> `CameraView` -> `Renderer::render_scene_with_view(...)` -> `Scene::build_submission()` -> `VkRender::render_with_hooks(...)`.

## 3. Key Concepts
- `RenderSubmission` is the immutable frame snapshot boundary between scene data and backend execution.
- `CameraView` is the caller-provided camera/view DTO for app-owned runtime paths.
- `DebugUiFrameContext` is a facade-to-backend telemetry snapshot boundary for debug UI composition.
- Event envelopes are facade/runtime telemetry boundaries; they must not carry backend-owned Vulkan state.
- Scene-facing types stay Vulkan-opaque: they emit handles (`MeshHandle`, `EnvironmentHandle`) and transforms, not Vulkan objects.
- Backend code resolves handles to loaded cache data at draw time (`get_loaded_id`, `get_loaded_material_ptr`).
- Flattening scene graph data into `Vec<FrameDrawItem>` avoids cross-layer borrow/ownership coupling.
- Template contract reference: `docs/internal/00-index.md` (mandatory 10-section order).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs
scene.update_camera(camera_view, proj, camera_pos);
let submission = scene.build_submission();
runtime.core.debug_ui.update_frame_context(DebugUiFrameContext {
    frame_index,
    draw_item_count: submission.draw_items.len(),
    point_light_count: submission.point_lights.len(),
    // ...
});
runtime.render_with_hooks(frame_number, &submission, || { /* pre */ }, || { /* post */ });
```

Snippet Type: Real
```rust
// App-owned path shape
let view = CameraView::from_camera(&app_camera, aspect_ratio);
let outcome = renderer.render_scene_with_view(&mut scene, view)?;
```

Snippet Type: Real
```rust
// src/renderer/src/scene/scene_world.rs
pub(crate) fn build_submission(&mut self) -> RenderSubmission {
    let mut submission = RenderSubmission::new(self.camera, 400);
    submission.skybox_env_id = self.skybox_env_id;
    self.refresh_world_recursive(root_id, Mat4::IDENTITY, false);
    self.collect_draw_items_recursive(root_id, &mut submission);
    submission
}
```

Snippet Type: Real
```rust
// src/renderer/src/scene/render_submission.rs
pub struct RenderSubmission {
    pub camera: SceneDataUBO,
    pub draw_items: Vec<FrameDrawItem>,
    pub flags: SubmissionFlags,
    pub skybox_mesh_id: MeshHandle,
    pub skybox_env_id: EnvironmentHandle,
    pub point_lights: Vec<FramePointLight>,
}
```

Snippet Type: Real
```rust
// src/renderer/src/vulkan/vk_render.rs
fn resolve_submission_buckets(
    &self,
    submission: &RenderSubmission,
) -> [Vec<RenderObject>; VkPipelineType::COUNT] {
    for draw_item in submission.draw_items.iter().copied() {
        let mesh = match mesh_cache.get_loaded_id(draw_item.mesh_id) { /* ... */ };
        let material_ptr = match tex_cache.get_loaded_material_ptr(mesh.material_id) { /* ... */ };
        // build RenderObject for draw
    }
}
```

Submission ownership boundary table:

| Layer | Owns | Does not own |
|---|---|---|
| API/Scene (`api`, `scene`) | scene graph, camera, handles, transforms | Vulkan descriptors, command buffers, GPU pipeline objects |
| Caller view (`CameraView`) | app-owned camera/view/projection values for one render | renderer-owned input dispatch, app lifecycle events |
| Submission (`RenderSubmission`) | frame-local copy of draw intent | persistent cache storage, backend mutable state |
| Debug UI context (`DebugUiFrameContext`) | frame-local UI telemetry and runtime status | scene graph mutation, Vulkan handles |
| Backend (`vulkan`) | handle resolution, rendergraph execution, queue submit/present | scene graph mutation API |

Snippet Type: Pseudocode
```text
# Borrow-checker and ownership rationale:
scene (&mut) -> build_submission() -> submission (owned value)
drop mutable scene borrow for render step
backend takes &submission (read-only) while mutating Vulkan internals

Why this helps:
- avoids borrowing scene graph and backend mutable state at the same time
- keeps frame payload deterministic after handoff
```

## 5. Best Practices
- Keep scene/public API types Vulkan-opaque; only backend modules should touch Vulkan handles/objects.
- Resolve cache handles only in backend draw preparation code.
- Treat `RenderSubmission` as frame-local immutable data once built.
- Add fields to `RenderSubmission` only when they represent draw intent, not backend implementation details.

## 6. Gotchas & Failure Modes
- Leaking backend assumptions into scene API (for example exposing descriptor set concerns) couples layers and complicates refactors.
- Mutating cache/storage while a submission is being consumed can invalidate assumptions during draw bucket resolution.
- Adding raw pointers to scene-facing types bypasses handle validation and generation safety.
- Returning references into mutable scene internals across the handoff boundary can reintroduce borrow conflicts the flattening step avoids.

## 7. Debugging Playbook
- Step 1: inspect `submission.draw_items.len()`, flags, and environment handle immediately after `build_submission()`.
- Step 2: if geometry is missing, verify backend handle resolution (`mesh_cache.get_loaded_id`, `get_loaded_material_ptr`) is succeeding.
- Step 3: if environment switching is wrong, trace `submission.skybox_env_id` into `prepare_submission_environment(...)`.
- Step 4: if frame logic fails only with hooks, compare `render(...)` vs `render_with_hooks(...)` path.
- Step 5: when ownership refactors fail to compile, ensure scene mutation completes before backend consumption begins.

## 8. Cross-Module Links
- Facade frame orchestration: `src/renderer/src/api/renderer.rs`
- Scene flattening: `src/renderer/src/scene/scene_world.rs`
- Submission payload contract: `src/renderer/src/scene/render_submission.rs`
- Debug UI context + manager: `src/renderer/src/debug_ui/mod.rs`
- Event lifecycle internals: `docs/internal/10-event-system-and-lifecycle.md`
- Backend frame execution: `src/renderer/src/vulkan/vk_render.rs`
- High-level frame mental model: `docs/internal/01-rendering-pipeline-mental-model.md`

## 9. Standard References
- Rust ownership and borrowing: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
- Rust borrowing rules: https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html
- Vulkan queue operation model: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#fundamentals-queueoperation
- Vulkan Guide synchronization overview: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/02-synchronization-and-fencing.md`
- `docs/internal/03-asset-lifecycle-and-io.md`
- `src/renderer/src/vulkan/AGENTS.md`
