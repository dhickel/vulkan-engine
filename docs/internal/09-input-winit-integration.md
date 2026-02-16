# Input winit Event Pump Integration

## 1. Purpose & Audience
This chapter is for contributors modifying renderer-input integration and the `input` crate bridge. It documents the exact event ingestion and frame-dispatch contracts.

## 2. Where This Fits in Engine Flow
Current runtime path:
`winit` event loop -> `Renderer::update_input(...)` -> `InputSystem` queue -> `Renderer::prepare_frame(...)` -> `InputSystem::dispatch_frame()` -> optional FPS plugin updates camera.

## 3. Key Concepts
- Ingestion is per-event (`update_input`).
- Dispatch is per-frame (`dispatch_frame`).
- UI capture is handled during ingestion using ImGui `want_capture_*` flags.
- Layer dispatch model uses priority groups with same-priority peer execution.
- Snapshot state is read-only to consumers after dispatch.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs
pub fn update_input(&mut self, window: &Window, event: &winit::event::Event<()>) -> Result<(), RendererError> {
    self.runtime.core.imgui.handle_event(window, event);

    let io = self.runtime.core.imgui.context.io();
    let consume_keyboard = io.want_capture_keyboard;
    let consume_mouse = io.want_capture_mouse;

    match event {
        Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
            if !consume_mouse {
                self.input_system.queue_mouse_motion(*delta);
            }
        }
        Event::WindowEvent { window_id, event } if *window_id == window.id() => {
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
  renderer.update_input(event)

once per frame:
  input.dispatch_frame()
  gameplay systems read snapshot/action states
```

## 5. Best Practices
- Keep all winit-to-input translation in `Renderer::update_input`.
- Preserve one dispatch boundary per frame.
- Keep `InputSystem` hot path free of avoidable allocations.
- Keep layer priorities explicit and documented for runtime modules.

## 6. Gotchas & Failure Modes
- If `want_capture_keyboard/mouse` filtering is removed, UI and gameplay will both react.
- Dispatching more than once per frame causes incorrect transient state.
- Dispatching less than once per frame causes stale input and lag.
- Consuming in high-priority layers can mask lower gameplay layers unexpectedly.

## 7. Debugging Playbook
- Step 1: verify `update_input` is called before event-specific branching in examples/apps.
- Step 2: inspect `input_system.debug_snapshot()` each frame.
- Step 3: log layer priorities and enabled states.
- Step 4: validate ImGui capture flags against observed behavior.
- Step 5: validate action map bindings when gameplay input appears dead.

## 8. Cross-Module Links
- Renderer integration: `src/renderer/src/api/renderer.rs`
- Input core: `src/input/src/lib.rs`
- Camera/plugin consumer: `src/renderer/src/data/camera.rs`
- Runtime loops: `src/renderer/examples/api_test.rs`, `apps/dungeon_dogfood/src/main.rs`

## 9. Standard References
- winit event model: https://docs.rs/winit/latest/winit/event/index.html
- winit keyboard docs: https://docs.rs/winit/latest/winit/keyboard/index.html
- imgui IO capture semantics: https://docs.rs/imgui/latest/imgui/struct.Io.html

## 10. See Also
- `docs/api/06-input-polling-and-listeners.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `src/input/AGENTS.md`
