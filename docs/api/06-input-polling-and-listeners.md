# Input Polling and Layered Dispatch

## 1. Purpose & Audience
This chapter is for students, hobbyists, and indie developers using the public renderer/input APIs. It explains how to ingest events, configure layers, and read per-frame input state safely.

## 2. Where This Fits in Engine Flow

> **renderer compatibility path** (demos, examples, smoke testing):
`winit` event loop -> `Renderer::update_input(...)` (queue raw input) -> `Renderer::begin_frame(...)` / `render_scene(...)` -> `InputSystem::dispatch_frame()` -> gameplay updates.

Direct crate flow:
`winit` events -> `InputSystem::queue_*` -> `InputSystem::dispatch_frame()` -> read `InputSnapshot`.

> **current app-owned path** (recommended for custom apps):
`winit` event loop -> `Renderer::route_platform_input(...)` -> `engine::input::queue_routed_input_event(...)` -> app-owned `InputSystem::dispatch_frame()` -> `InputActionEventEmitter::emit_from_snapshot(...)` -> app-owned `EventBus` stage drain. See [15-app-owned-loop.md](15-app-owned-loop.md).

## 3. Key Concepts
- Frame-buffered model: events are queued then dispatched once per frame.
- Layered dispatch: input handlers are grouped by priority.
- Consumption rule: all same-priority handlers run; if any consume, lower priorities do not run.
- Polling snapshot: gameplay can query held/just-pressed keys, mouse delta, scroll, and action values.
- Event bridge: the renderer emits `InputActionEvent` after `InputSystem::dispatch_frame()` from the refreshed snapshot.
- App-owned event bridge: `InputActionEventEmitter` emits `InputActionEvent` from a caller-provided
  `InputSnapshot` into a caller-owned `EventBus`, using its own observed action-value map per app
  input stream.
- App-owned lifecycle bridge: `engine::events::RuntimeEventDispatcher` can drain input and emit
  frame lifecycle events on the same caller-owned `EventBus`.
- Action mapping: bind semantic actions (`"move.forward"`) to chords (keys/buttons + modifiers).
- Input profiles: `ActionMap` load/save uses strict `version = 1` TOML with `trigger` + `modifiers`.
- Camera controls: `Renderer::install_default_fps_input()` installs the built-in WASD/mouse-look layer for the **renderer compatibility path**. Apps that own input should install an action layer in their own `InputSystem`, update a root `Camera` with `FPSController`, collision-correct app state, then pass a `CameraView` into the renderer. See [15-app-owned-loop.md](15-app-owned-loop.md).

## 4. Code Walkthrough
Snippet Type: Real (renderer compatibility path)
```rust
use renderer::{
    ActionMap, LayerDescriptor, LayerPriority, Renderer,
};
use winit::keyboard::KeyCode;

let mut renderer = Renderer::new(config, &window)?;

// Optional built-in classic FPS controls (renderer compatibility path).
renderer.install_default_fps_input();

// Custom action map layer (uses renderer-owned InputSystem).
let mut actions = ActionMap::new();
actions.bind_key("game.interact", KeyCode::KeyE);
actions.bind_key("game.jump", KeyCode::Space);

renderer.input_mut().add_layer(
    LayerDescriptor::new("game-actions", LayerPriority(20)),
    actions.into_layer(),
);
```

The default FPS input layer uses `W`, `A`, `S`, `D`, `Space`, `ShiftLeft`, and mouse motion. It updates the renderer-owned camera during frame preparation. Apps that only need a moving camera can install it and avoid the lower-level camera helper types. Apps that need app-owned camera/collision behavior should wire their own `InputSystem` and controller, as `dungeon_dogfood` does.

Snippet Type: Real (renderer compatibility path)
```rust
event_loop.run(move |event, elwt| {
    renderer.update_input(&window, &event)?;

    if let Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } = event {
        let mut frame = renderer.begin_frame(&window)?;

        // Read the latest frame snapshot from renderer-owned InputSystem.
        let snapshot = renderer.input().snapshot();
        let jump = snapshot.action_pressed(&"game.jump".into());
        let look_delta = snapshot.mouse_delta();

        renderer.render_scene_in_frame(&mut frame, &mut scene)?;
        renderer.end_frame(frame)?;
    }

    Ok::<(), renderer::RendererError>(())
});
```

Snippet Type: Pseudocode
```text
queue events all frame
-> dispatch once
-> run all priority=100 layers
-> if any consumed, stop at 99-
-> otherwise continue downward
```

Snippet Type: Real
```toml
version = 1

[[bindings]]
action = "move.forward"
trigger = { key = "KeyW" }
modifiers = { shift = false, ctrl = false, alt = false, super_key = false }
scale = 1.0
consume = false
```

## 5. Multi-binding action contribution

When multiple keys or buttons map to the same action, each binding receives a private `BindingInstanceId` at `ActionMapLayer` construction time. The input system tracks each binding's contribution independently:

- Pressing any binding activates the action at the binding's scale value.
- Releasing one binding leaves the action active as long as another binding instance still contributes.
- The aggregate action value is the maximum across all active binding instances.
- Transient edges (`just_pressed` / `just_released`) fire correctly: `just_released` only fires when the last active binding releases.

```rust
let mut map = ActionMap::new();
map.bind_key("move.forward", KeyCode::KeyW);
map.bind_key("move.forward", KeyCode::ArrowUp);
// Both keys contribute to "move.forward" independently.
```

Apps that use `InputContext::set_action_value` directly can also use `set_instance_value`
with a caller-assigned `BindingInstanceId` when they need multi-source aggregation.

## 6. Best Practices
- Call `update_input(...)` for every event.
- Apps that own their own `InputSystem` should call `route_platform_input(...)` for every event and
  queue only uncaptured events with `queue_routed_input_event(...)`.
- For app-owned camera paths, update the camera from the dispatched snapshot before building the
  `CameraView`; collision and gameplay correction should happen before the view DTO is submitted.
- Keep exactly one frame boundary (`begin_frame` or `render_scene`) per rendered frame.
- For app-owned input, keep exactly one `InputSystem::dispatch_frame()` call per app frame. Resize
  skipped render frames should not skip this app input dispatch; render skipping is independent.
- Use one caller-owned `EventBus` for app input and app lifecycle events. Do not mirror app lifecycle
  through `Renderer::events_mut()` on the no-dispatch render path.
- Use action names for gameplay logic; avoid hardcoding key codes in systems.
- Keep UI/input-capture layers at higher priority than gameplay layers.
- Let the renderer own cursor confinement transitions. On Wayland, do not add app-level
  `CursorLeft`/`CursorEntered` grab calls around `route_platform_input`; the renderer preserves one
  persistent request across pointer leave/re-entry.
- Prefer stable action IDs and load/save binding profiles for user rebinding.
- Keep layer registration in explicit bands:
  - `900-1000`: engine capture/system overlays
  - `500-899`: UI routing/capture (`priority_bands::EDITOR_UI_CAPTURE` is reserved for editor chrome)
  - `100-499`: gameplay
  - `0-99`: debug/fallback
- The renderer's `register_app_ui(...)` path treats native app chrome as active imgui capture:
  keyboard, mouse, cursor grab, and built-in FPS camera updates are suppressed while the app UI is
  registered. This is the editor shell's default so panel/menu interaction does not leak into game
  controls. If the app intercepts a show/hide hotkey before renderer routing, it calls
  `refresh_cursor_capture(window)` after callback registration changes so hidden UI restores cursor
  confinement and mouse-look immediately.
- Direct `InputSystem` users can install `editor_ui_capture_layer()` when they need the same
  high-priority editor-consumes-before-gameplay behavior outside the renderer facade.

## 6. Gotchas & Failure Modes
- If `dispatch_frame()` never runs (implicitly through renderer frame prep), snapshots will not advance.
- If two systems must both process input, keep them in the same priority group.
- If a high-priority layer consumes events unexpectedly, lower layers will appear "dead".
- `mouse_delta` and `just_*` states are transient and valid only for the current frame.
- Profile load is strict: unsupported keys, malformed triggers, or unknown fields are hard errors.
- The renderer facade does not auto-load an input profile path from `RendererConfig`; profile TOML is app-owned setup code.
- Modifier-release ordering is tracked per-binding. Releasing a modifier before releasing the bound key no longer leaves the action stuck active — the system tracks each binding's press/release independently via `BindingInstanceId` and recomputes the aggregate action value after every event.
- `set_action_value` and `set_instance_value` are the two public paths for direct action injection. Use `set_instance_value` with a caller-assigned `BindingInstanceId` when multiple sources contribute to the same action; use `set_action_value` for single-source actions (behaves as an anonymous instance).

## 7. Debugging Playbook
- Step 1: verify `renderer.update_input(...)` is called for every event.
- Step 2: inspect `renderer.input().debug_snapshot()` for queued event count and active layers.
- Step 3: confirm layer priorities and consumption behavior.
- Step 4: log action values from `snapshot.action_value(...)` to verify mappings.
- Step 5: subscribe through `Renderer::events_mut()` for legacy renderer-owned paths, or through the
  app-owned `EventBus` when using `InputActionEventEmitter`.
- Step 6: reproduce with `cargo run -p renderer --example api_test`.

## 8. Cross-Module Links
- Input crate source: `src/input/src/lib.rs`
- Renderer ingestion boundary: `src/renderer/src/api/renderer.rs`
- FPS controller action consumer: `src/renderer/src/data/camera.rs`

## 9. Standard References
- winit docs: https://docs.rs/winit/latest/winit/
- Rust trait objects: https://doc.rust-lang.org/book/ch17-02-trait-objects.html
- Rust enums/pattern matching: https://doc.rust-lang.org/book/ch06-00-enums.html

## 10. See Also
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/12-events-and-lifecycle.md`
- `docs/internal/09-input-winit-integration.md`
- `src/input/AGENTS.md`

## 11. Future Considerations
- Gamepad support is intentionally post-alpha.
- Plan for gamepad action triggers should reuse the same layered dispatch and consumption semantics.
