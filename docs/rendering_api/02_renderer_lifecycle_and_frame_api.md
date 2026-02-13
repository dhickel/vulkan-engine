# 02 - Renderer Lifecycle and Frame API

This chapter defines what the renderer facade guarantees today.

## Renderer Lifecycle Contract

Main public methods (`src/renderer/src/api/renderer.rs`):
- `Renderer::new(config, &Window) -> Result<Renderer, RendererError>`
- `update_input(&Window, &Event<()>) -> Result<(), RendererError>`
- `resize(width, height) -> Result<(), RendererError>`
- `render_scene(&Window, &mut Scene) -> Result<FrameRenderOutcome, RendererError>`

Explicit frame API:
- `begin_frame(&Window) -> Result<FrameContext, RendererError>`
- `render_scene_in_frame(&mut FrameContext, &mut Scene) -> Result<FrameRenderOutcome, RendererError>`
- `end_frame(FrameContext) -> Result<(), RendererError>`

## One-shot vs Explicit Frame API

Use one style per frame.

Allowed:
- `render_scene(...)`
- `begin_frame -> render_scene_in_frame -> end_frame`

Rejected with `RendererError::InvalidState`:
- Calling `render_scene` while an explicit frame is open.
- Calling `begin_frame` twice without `end_frame`.
- Calling `render_scene_in_frame` twice for the same frame.

`FrameRenderOutcome` values:
- `Rendered`: scene submission was rendered.
- `SkippedResizePending`: frame render was skipped because resize/swapchain state is pending.

## Per-Frame Internal Behavior (important for users)

`render_scene` and `begin_frame` both:
- Pump deferred asset tasks (`pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)`).
- Consume buffered input + update camera timing state.
- Prepare ImGui frame state.

Implication:
- If your app pauses rendering (menus, minimized window, loading screens), call
  `pump_asset_tasks` explicitly if you still want deferred loads to progress.

## Input Handling Contract

Input is a two-step flow:

1. `update_input(&Window, &Event<()>)` ingests raw winit events into renderer/input state.
2. `render_scene(...)` or `begin_frame(...)` calls internal frame prep, which applies that buffered input to the controller (`input_manager.update()` + controller `update(delta)`).

So movement is not applied inside `update_input` itself; it is applied during frame prep immediately before rendering.

Canonical event-loop ordering:
- For each incoming winit event: call `update_input(...)`.
- On redraw:
  - One-shot API: `render_scene(...)`
  - Explicit API: `begin_frame(...) -> render_scene_in_frame(...) -> end_frame(...)`

## Resize Contract

- `resize(0, 0)` is a no-op.
- Resize is rejected during open explicit frame (`InvalidState`).
- Swapchain rebuild failures are wrapped as frame resize errors.
- Resize-pending frames are surfaced as `FrameRenderOutcome::SkippedResizePending` instead of silent success.

## Startup Scene Contract

- `take_startup_scene()` returns `Some(Scene)` once, then `None`.
- If your game does full custom loading, discard startup scene and use `Scene::new()`.

## Environment Runtime Status

`environment_runtime_status()` returns:
- `requested: Option<EnvironmentHandle>`
- `active: EnvironmentHandle`
- `transitioning: bool`

Use it for UI/debug overlays while testing environment transitions.

## Validation Checklist

1. Ensure one frame API style per frame.
2. Ensure resize path is not called while explicit frame is open.
3. Ensure deferred loads are pumped when not actively rendering.

## Learn More

- Scene workflows: `03_scene_graph_and_fragment_workflows.md`
- Async loads: `04_assets_sync_deferred_and_handles.md`
- Example: `src/renderer/examples/api_test.rs`
