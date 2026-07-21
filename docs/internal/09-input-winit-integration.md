# Input winit Event Pump Integration

## 1. Purpose & Audience
This chapter is for contributors modifying renderer-input integration and the `input` crate bridge. It documents the exact event ingestion and frame-dispatch contracts.

## 2. Where This Fits in Engine Flow
Current runtime path:
`winit` event loop -> `Renderer::update_input(...)` -> `InputSystem` queue -> `Renderer::prepare_frame(...)` -> `InputSystem::dispatch_frame()` -> optional FPS plugin updates camera.

App-owned input path:
`winit` event loop -> `engine::input::route_platform_input_to_app(...)` -> renderer platform side effects via `Renderer::route_platform_input(...)` -> `engine::input::queue_routed_input_event(...)` for uncaptured events -> app frame boundary via `engine::frame::begin_app_frame(...)`.

## 3. Key Concepts
- Ingestion is per-event (`update_input`).
- Dispatch is per-frame (`dispatch_frame`).
- UI capture is handled during ingestion using ImGui `want_capture_*` flags plus explicit engine
  overlay visibility gating.
- `F1` toggles the left console panel and `F2` toggles the right debug panel independently.
- When either panel is visible, cursor grab is released and FPS camera-look updates are paused.
- Cursor confinement is an edge-triggered persistent request. Cursor leave/enter updates presence but
  does not tear down and recreate the Wayland constraint.
- Layer dispatch model uses priority groups with same-priority peer execution.
- Snapshot state is read-only to consumers after dispatch.
- Renderer event emission for input actions reads the refreshed snapshot after `dispatch_frame()`.
- App-owned input routing should use `route_platform_input_to_app` when the app wants one call that preserves renderer side effects and queues uncaptured input into its own `InputSystem`.
- `queue_routed_input_event` remains the lower-level app-owned queueing surface when callers already have a `RendererInputRouting` result.
- App-owned input event emission uses `InputActionEventEmitter`, which owns the observed action-value
  map for one app input stream and emits into the caller-owned `EventBus` through `begin_app_frame` or direct calls.
- Action profile parsing (`ActionMap::from_toml_str`) is strict `version = 1` with validated triggers.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs
pub fn route_platform_input(&mut self, window: &Window, event: &winit::event::Event<()>) -> Result<RendererInputRouting, RendererError> {
    self.runtime.core.imgui.handle_event(window, event);

    let io = self.runtime.core.imgui.context.io();
    let ui_visible = self.runtime.core.debug_ui.is_any_visible();
    let consume_keyboard = ui_visible || io.want_capture_keyboard;
    let consume_mouse = ui_visible || io.want_capture_mouse;

    match event {
        Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
            return Ok(if consume_mouse { RendererInputRouting::suppress(UiMouseCapture) } else { RendererInputRouting::queue() });
        }
        Event::WindowEvent { window_id, event } if *window_id == window.id() => {
            // F1/F2 toggle overlay panels.
            // keyboard/mouse are queued only when not UI-captured.
            // focus/modifiers are always tracked.
            // ...
        }
        _ => {}
    }

    Ok(())
}
```

Snippet Type: Real
```rust
fn prepare_frame(&mut self, window: &Window) -> Result<FramePrepareOutcome, RendererError> {
    let delta = ...;

    self.input_system.dispatch_frame();

    if let Some(plugin) = self.fps_plugin.as_mut() {
        let snapshot = self.input_system.snapshot();
        plugin.controller.update_from_snapshot(snapshot, delta.as_secs_f32(), &mut self.camera);
    }

    // imgui frame prep + resize handling
    // ...
}
```

Snippet Type: Pseudocode
```text
for event in os_events:
  route_platform_input_to_app(renderer, window, app_input, event)
  # lower-level equivalent: renderer.route_platform_input + queue_routed_input_event

once per frame:
  begin_app_frame(app_input, app_action_events, app_event_bus, frame_clock)
  gameplay systems read snapshot/action states
  controller updates app camera intent
  collision corrects app/player state
  app builds CameraView and calls renderer render-only/view API
  end_app_frame(app_event_bus, frame_index)
```

Priority band guidance:
- `900-1000`: engine capture/system overlays
- `500-899`: UI/input routing
- `100-499`: gameplay/action layers
- `0-99`: debug/fallback

## 5. Best Practices
- Keep renderer platform side effects in `Renderer::route_platform_input`.
- Track the last successful cursor-grab request separately from pointer presence. Apply only policy
  transitions, and defer transitions while the pointer is outside the window.
- Prefer `route_platform_input_to_app` for app-owned input paths; use `queue_routed_input_event` directly only when code already has a routing result.
- Preserve one dispatch boundary per frame.
- On the app-owned path, do not also call renderer frame APIs that dispatch renderer-owned input in
  the same app frame.
- Keep `InputSystem` hot path free of avoidable allocations.
- Keep layer priorities explicit and documented for runtime modules.

## 6. Gotchas & Failure Modes
- If `want_capture_keyboard/mouse` filtering is removed, UI and gameplay will both react.
- Dispatching more than once per frame causes incorrect transient state.
- Dispatching less than once per frame causes stale input and lag.
- Consuming in high-priority layers can mask lower gameplay layers unexpectedly.
- Profile reload failures should be surfaced to logs/UX; never silently drop invalid bindings.
- On winit 0.29 Wayland, `CursorLeft` is delivered after winit removes the pointer from the window's
  pointer list. Releasing confinement in that handler may not destroy the protocol object even
  though winit records `None`; requesting `Confined` on re-entry can then create a duplicate and
  trigger a fatal `zwp_pointer_constraints_v1` protocol error.
- `WindowEvent::Focused` is not part of the renderer cursor policy. Wayland activates/deactivates the
  persistent constraint with pointer focus; the renderer should not recreate it for focus events.

## 7. Debugging Playbook
- Step 1: on renderer-owned compatibility paths, verify `update_input` is called before event-specific branching; on app-owned paths, verify `route_platform_input_to_app` (or `route_platform_input` + `queue_routed_input_event`) runs before app-frame dispatch.
- Step 2: inspect `input_system.debug_snapshot()` each frame.
- Step 3: log layer priorities and enabled states.
- Step 4: validate ImGui capture flags against observed behavior and confirm cursor policy changes
  produce only one `set_cursor_grab` transition.
- Step 5: on Wayland, distinguish the compositor's current constraint activation from the renderer's
  persistent requested mode; do not infer protocol-object destruction from `CursorLeft`.
- Step 6: validate action map bindings when gameplay input appears dead.
- Step 7: subscribe through `Renderer::events_mut()` on legacy renderer-owned paths, or through the
  app-owned `EventBus` when validating `InputActionEventEmitter`.

## 8. Cross-Module Links
- Renderer integration: `src/renderer/src/api/renderer.rs`
- Input core: `src/input/src/lib.rs`
- Camera/plugin consumer: `src/renderer/src/data/camera.rs`
- Runtime loops: `src/renderer/examples/api_test.rs`, `apps/dungeon_dogfood/src/main.rs`
- Event lifecycle internals: `docs/internal/10-event-system-and-lifecycle.md`

## 9. Standard References
- winit event model: https://docs.rs/winit/latest/winit/event/index.html
- winit keyboard docs: https://docs.rs/winit/latest/winit/keyboard/index.html
- imgui IO capture semantics: https://docs.rs/imgui/latest/imgui/struct.Io.html

## 10. See Also
- `docs/api/06-input-polling-and-listeners.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `docs/api/12-events-and-lifecycle.md`
- `src/input/AGENTS.md`

## 11. Future Considerations
- Gamepad input support is a post-alpha target.
- Future gamepad triggers should plug into `ActionMap`/layer dispatch without changing consumption semantics.
- Multi-window editor/debug workflows are deferred post-alpha; current runtime remains single-window.
- Future windowing refactors should isolate per-window input+ImGui contexts before exposing a multi-window public API.
