# Input Polling and Layered Dispatch

## 1. Purpose & Audience
This chapter is for students, hobbyists, and indie developers using the public renderer/input APIs. It explains how to ingest events, configure layers, and read per-frame input state safely.

## 2. Where This Fits in Engine Flow
Runtime flow:
`winit` event loop -> `Renderer::update_input(...)` (queue raw input) -> `Renderer::begin_frame(...)` / `render_scene(...)` -> `InputSystem::dispatch_frame()` -> gameplay updates.

Direct crate flow:
`winit` events -> `InputSystem::queue_*` -> `InputSystem::dispatch_frame()` -> read `InputSnapshot`.

## 3. Key Concepts
- Frame-buffered model: events are queued then dispatched once per frame.
- Layered dispatch: input handlers are grouped by priority.
- Consumption rule: all same-priority handlers run; if any consume, lower priorities do not run.
- Polling snapshot: gameplay can query held/just-pressed keys, mouse delta, scroll, and action values.
- Action mapping: bind semantic actions (`"move.forward"`) to chords (keys/buttons + modifiers).

## 4. Code Walkthrough
Snippet Type: Real
```rust
use renderer::{
    ActionMap, LayerDescriptor, LayerPriority, Renderer,
};
use winit::keyboard::KeyCode;

let mut renderer = Renderer::new(config, &window)?;

// Optional built-in classic FPS controls.
renderer.install_default_fps_input();

// Custom action map layer.
let mut actions = ActionMap::new();
actions.bind_key("game.interact", KeyCode::KeyE);
actions.bind_key("game.jump", KeyCode::Space);

renderer.input_mut().add_layer(
    LayerDescriptor::new("game-actions", LayerPriority(20)),
    actions.into_layer(),
);
```

Snippet Type: Real
```rust
event_loop.run(move |event, elwt| {
    renderer.update_input(&window, &event)?;

    if let Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } = event {
        let mut frame = renderer.begin_frame(&window)?;

        // Read the latest frame snapshot.
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

## 5. Best Practices
- Call `update_input(...)` for every event.
- Keep exactly one frame boundary (`begin_frame` or `render_scene`) per rendered frame.
- Use action names for gameplay logic; avoid hardcoding key codes in systems.
- Keep UI/input-capture layers at higher priority than gameplay layers.
- Prefer stable action IDs and load/save binding profiles for user rebinding.

## 6. Gotchas & Failure Modes
- If `dispatch_frame()` never runs (implicitly through renderer frame prep), snapshots will not advance.
- If two systems must both process input, keep them in the same priority group.
- If a high-priority layer consumes events unexpectedly, lower layers will appear "dead".
- `mouse_delta` and `just_*` states are transient and valid only for the current frame.

## 7. Debugging Playbook
- Step 1: verify `renderer.update_input(...)` is called for every event.
- Step 2: inspect `renderer.input().debug_snapshot()` for queued event count and active layers.
- Step 3: confirm layer priorities and consumption behavior.
- Step 4: log action values from `snapshot.action_value(...)` to verify mappings.
- Step 5: reproduce with `cargo run -p renderer --example api_test`.

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
- `docs/internal/09-input-winit-integration.md`
- `src/input/AGENTS.md`
