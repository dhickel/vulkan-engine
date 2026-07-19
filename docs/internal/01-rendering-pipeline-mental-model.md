# Rendering Pipeline Mental Model

## 1. Purpose & Audience
This chapter gives contributors a stable mental model for how scene-facing API calls become Vulkan commands and presented pixels.

## 2. Where This Fits in Engine Flow
Current end-to-end flow:
`Renderer::render_scene(...)` -> scene camera update -> `SceneWorld::build_submission()` -> `VkRenderCore::render_with_hooks(...)` -> rendergraph passes -> submit -> present.

## 3. Key Concepts
- `RenderSubmission` is a flat, frame-local snapshot of draw intent (handles + transforms + flags).
- Rendergraph pass order defines recording order and implicit pipeline of responsibilities.
- ImGui rendering is now manager-driven:
  - `VkRenderCore::draw_imgui(...)` creates the frame UI and delegates to `DebugUiManager`.
  - `DebugUiManager` composes built-in views, registered custom views, and console windows.
- Descriptor/pipeline ABI is a contract between shader layouts and draw code.
- Material resolution copies `CopiedMaterialDrawRecord` while cache guards are held; frame draw objects do not retain cache-owned raw pointers.
- Scene graph is CPU-side structure; it is not directly GPU render state. New scenes cull by default with authoritative known bounds, explicit tagged proxies, conservative-visible fallback, and post-order subtree unions.
- Template contract reference: `docs/internal/00-index.md` (mandatory 10-section order).

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs
scene.update_camera(camera_view, proj, camera_pos);
let submission = scene.build_submission();
runtime.render(frame_number, &submission);
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
// src/renderer/src/rendergraph/mod.rs
PrepareTargetsPass -> ShadowPass -> SkyboxPass -> GeometryPass -> PresentCopyPass -> ImguiPass -> DebugCapturePass -> TerminalPresentPass
```

Submission-to-present timeline:

| Stage | Input | Output |
|---|---|---|
| Facade render call | mutable scene | `RenderSubmission` |
| PrepareTargetsPass | frame draw/depth images | writable render targets |
| ShadowPass | directional light + opaque draw bounds | frame-local D32 map in shader-read layout |
| SkyboxPass | submission skybox flags/env handle | background color in draw target |
| GeometryPass | draw items + material/mesh handles | scene geometry in draw target |
| PresentCopyPass | draw target + present target | present image ready for UI/present |
| ImguiPass | present attachment + manager-composed UI draw data | final present image |

Descriptor/pipeline ABI notes (current convention):
- Set 0: scene descriptors (camera/environment UBOs, IBL maps, BRDF LUT, directional shadow map at binding 5)
- Set 1: skin/joint descriptors
- Set 2: material image descriptors
- Push constants carry model transform + GPU addresses/metadata used by geometry draws

Snippet Type: Pseudocode
```text
submission = flatten(scene graph)
for pass in ordered_passes:
  pass.execute(submission, frame)
submit(wait acquire_semaphore, signal render_semaphore)
present(wait render_semaphore)
# If recording fails after acquisition: reset/record drain -> submit -> retire present sync/image
```

## 5. Best Practices
- Keep pass responsibilities narrow and explicit (what each pass reads/writes).
- When changing pass order, update transitions and dependency assumptions in the same change.
- Preserve the submission boundary so scene code remains decoupled from Vulkan internals.
- Use tables/diagrams in docs for ownership and stage transitions.

## 6. Gotchas & Failure Modes
- Reordering passes without transition updates can produce undefined image layouts.
- Descriptor layout drift vs shader expectations can cause silent draw corruption.
- Treating scene nodes as direct GPU objects bypasses cache and handle validation rules.
- Pipeline bucketing changes in geometry draw path have known correctness risk; copying new material fields outside the cache guard reintroduces a lifetime hazard.
- Any early return after frame-fence reset must preserve the drain-transaction contract or the next frame-slot reuse can deadlock.

## 7. Debugging Playbook
- Step 1: confirm `RenderSubmission` contains expected draw items and flags.
- Step 2: validate pass order and that each expected pass executes.
- Step 3: inspect descriptor set layout compatibility with active shader pipeline.
- Step 4: confirm present transition path (`PresentCopyPass` + `ImguiPass`) for current frame flags.
- Step 5: if frame output is empty, verify mesh/material handles resolve to `Loaded` cache state.

## 8. Cross-Module Links
- Facade render entry: `src/renderer/src/api/renderer.rs`
- Scene flattening: `src/renderer/src/scene/scene_world.rs`
- Submission payload: `src/renderer/src/scene/render_submission.rs`
- Rendergraph orchestration: `src/renderer/src/rendergraph/mod.rs`
- Vulkan frame execution: `src/renderer/src/vulkan/vk_render.rs` (coordinator), `src/renderer/src/vulkan/vk_frame.rs` (frame lifecycle), `src/renderer/src/vulkan/vk_commands.rs` (command recording)
- Debug UI composition: `src/renderer/src/debug_ui/mod.rs`

## 9. Standard References
- Vulkan dynamic rendering: https://github.khronos.org/Vulkan-Site/guide/latest/dynamic_rendering.html
- Vulkan descriptor sets: https://github.khronos.org/Vulkan-Site/guide/latest/descriptorsets.html
- Vulkan push constants: https://github.khronos.org/Vulkan-Site/guide/latest/push_constants.html
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/internal/02-synchronization-and-fencing.md`
- `docs/internal/03-asset-lifecycle-and-io.md`
- `src/renderer/src/vulkan/AGENTS.md`
