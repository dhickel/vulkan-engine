# 03 - Frame Execution Flow

## Startup flow

High-level startup sequence:
1. Build window/event/input wiring.
2. Create `VkRender` (instance/device/swapchain/caches/pipelines).
3. Load startup scene metadata.
4. Allocate meshes/textures (startup worker).
5. Ensure default environment and descriptors.
6. Enter steady-state redraw loop.

Code example (in-tree/internal):
```rust
let (app, scene_world) = vk_render::VkRender::new(
    window_state,
    false,
    runtime_flags.compile_shaders,
    runtime_flags.debug_runtime_mode,
)?;
```

Best practice:
- Keep startup deterministic and front-load expensive first-time allocations.

Learn more:
- Startup construction: `src/renderer/src/vulkan/vk_render.rs` (`VkRenderCore::new`)
- Vulkan init overview: https://github.khronos.org/Vulkan-Site/guide/latest/initialization.html

## Engine-side per-frame flow (before Vulkan commands)

Code example (in-tree/internal):
```rust
state.input_manager.update();
state.app.core.window_state.controller.borrow_mut().update(dt_sec);

state.scene_world.update_camera(camera_view, proj, camera_pos);
let submission = state.scene_world.build_submission();

state.app.render(state.frame, &submission);
```

Best practice:
- Build one immutable submission per frame and avoid mutating scene caches during render command recording.

Learn more:
- Frame handler: `src/renderer/src/lib.rs` (`handle_redraw_requested`)

## Vulkan-side frame flow (`VkRenderCore::render`)

Order used now:
1. Service async transfers and environment requests.
2. Acquire frame slot + swapchain image.
3. Wait/reset current frame fence.
4. Reset per-frame transient resources.
5. Begin command buffer.
6. Execute rendergraph passes.
7. End command buffer.
8. Submit queue work.
9. Present.

Code example (structural, in-tree/internal):
```rust
self.service_transfers_and_prepare_environment(submission);
let Some(frame) = self.acquire_frame_slot() else { return; };

self.reset_and_begin_frame_cmd(frame.cmd_buffer);
unsafe { self.execute_rendergraph_for_frame(submission, rendergraph)?; }
self.end_frame_cmd(frame.cmd_buffer);

self.submit_frame(frame);
self.present_frame(frame);
```

Best practice:
- Do not reorder acquire/fence/reset/submit/present unless you also re-validate synchronization and image-layout assumptions.

Learn more:
- Renderer core frame path: `src/renderer/src/vulkan/vk_render.rs`
- Synchronization overview: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html

## Rendergraph pass order contract

Default pass order:
1. `PrepareTargetsPass`
2. `SkyboxPass`
3. `GeometryPass`
4. `PresentCopyPass`
5. `ImguiPass`

Code example (in-tree/internal):
```rust
RenderGraph::new(vec![
    Box::new(PrepareTargetsPass),
    Box::new(SkyboxPass),
    Box::new(GeometryPass),
    Box::new(PresentCopyPass),
    Box::new(ImguiPass),
])
```

Best practice:
- Keep pass order explicit and avoid implicit image-state assumptions when flags skip passes.

Learn more:
- Rendergraph implementation: `src/renderer/src/rendergraph/mod.rs`

## Geometry draw-order contract

Current order:
1. PBR opaque
2. Unlit opaque
3. PBR mask
4. Unlit mask
5. PBR blend (back-to-front)
6. Unlit blend (back-to-front)

Best practice:
- Keep alpha blend sorting camera-relative and stable; this is required for expected blending results.

Learn more:
- Draw bucketing/sort implementation: `src/renderer/src/vulkan/vk_render.rs` (`partition_geometry_draw_lists`, `sort_geometry_blended_lists`)
- Transparency ordering background: https://learnopengl.com/Advanced-OpenGL/Blending
