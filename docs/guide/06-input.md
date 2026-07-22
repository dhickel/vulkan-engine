# 06 — Input System

> Provenance: `G-06`

This chapter covers the app-owned input system from `src/input`. It explains layered dispatch, action maps, snapshots, the `dispatch_frame` boundary, `InputActionEventEmitter`, and how input is routed from platform events through the renderer's side effects to your app.

For the complete API reference (every type, every method signature), see [Input Polling & Listeners](../api/06-input-polling-and-listeners.md). For the internal winit integration and cursor confinement details, see [Input winit Integration](../internal/09-input-winit-integration.md).

## Architecture Overview

The input system is **frame-buffered**: raw platform events are queued during the event pump, and all dispatch happens at a single frame boundary (`dispatch_frame`). Between frames, the system exposes a pollable `InputSnapshot` with stable state and frame-scoped transient fields.

```
Platform Events (winit)
    │
    ├─ route_platform_input_to_app()
    │   ├─ Renderer side effects (ImGui, debug UI, cursor capture)
    │   └─ App-owned InputSystem.queue_event()
    │
    └─ RedrawRequested
        └─ begin_app_frame()
            └─ input.dispatch_frame()        ← THE frame boundary
                ├─ Reset transients
                ├─ Dispatch queued events through layers
                ├─ Refresh action values
                └─ on_frame_end callbacks
```

## Key Types

| Type | Purpose |
|------|---------|
| `InputSystem` | Frame-buffered input runtime with priority-ordered layers |
| `InputSnapshot` | Per-frame pollable state: key/button down/just-pressed/just-released, mouse delta, scroll delta, action values |
| `InputEvent` | Normalized input event enum (Key, MouseMotion, MouseButton, MouseWheel, ModifiersChanged, CursorFocus) |
| `ActionMap` | Collection of key/button → action bindings with modifiers and scale |
| `ActionMapLayer` | Wraps an `ActionMap` into the `InputLayer` trait for dispatch |
| `CaptureLayer` | Blocks keyboard and/or mouse events from reaching lower-priority layers |
| `LayerDescriptor` | Layer metadata: name, `LayerPriority`, enabled flag |
| `LayerHandle` / `LayerId` | Stable handle for layer lifecycle management |
| `InputActionEventEmitter` | Converts action snapshots into `EngineEvent::Input` events on the event bus |
| `InputChord` | Compound binding with key, mouse button, and modifier requirements |

## Layers, Priority Groups, and Consumption

> Provenance: `G-06-LAYERS` — Excerpt from live `src/input/src/lib.rs`

Layers are registered with a `LayerPriority`. The system dispatches events to layers **by descending priority** (higher numbers first). All layers at the same priority level always receive every event (**peer execution**). If any layer in a priority group returns `InputConsume::Consumed`, lower-priority groups do not see the event.

### Priority Bands

The crate defines suggested priority bands in `input::priority_bands`:

| Band | Range | Purpose |
|------|-------|---------|
| `ENGINE_CAPTURE_MIN`–`ENGINE_CAPTURE_MAX` | 900–1000 | Engine-internal capture (frame/debug hotkeys) |
| `UI_ROUTING_MIN`–`UI_ROUTING_MAX` | 500–899 | ImGui, editor UI, and platform shortcuts |
| `EDITOR_UI_CAPTURE` | 850 | Editor UI capture layer |
| `GAMEPLAY_MIN`–`GAMEPLAY_MAX` | 100–499 | Gameplay action bindings |
| `DEBUG_MIN`–`DEBUG_MAX` | 0–99 | Debug overlays, development tools |

Your app should use the `GAMEPLAY_MIN`–`GAMEPLAY_MAX` band (100–499) for action bindings. The engine's debug UI and ImGui layers use higher priorities so they can consume events before gameplay sees them.

### Layer Lifecycle

> Provenance: `G-06-LIFECYCLE` — Excerpt

```rust
use input::{ActionMap, InputSystem, LayerDescriptor, LayerHandle, LayerPriority};

let mut input = InputSystem::new();

// Add a named layer with a priority.
let handle: LayerHandle = input.add_layer(
    LayerDescriptor::new("my-gameplay", LayerPriority(200)),
    ActionMap::new().into_layer(),
);

// Disable temporarily (e.g., when a menu is open).
input.set_layer_enabled(handle, false);

// Re-enable.
input.set_layer_enabled(handle, true);

// Change priority at runtime (e.g., toggle between gameplay and UI mode).
input.set_layer_priority(handle, LayerPriority(600));

// Remove permanently.
input.remove_layer(handle);

// Inspect descriptor.
if let Some(desc) = input.layer_descriptor(handle) {
    println!("layer '{}' priority={:?}", desc.name, desc.priority);
}
```

## Action Maps

> Provenance: `G-06-ACTIONS` — Full Match (checkpoint excerpt; see Chapter 04 for full context)

```rust
let mut map = input::ActionMap::new();
map.bind_key("move.forward", KeyCode::KeyW);
map.bind_key("move.backward", KeyCode::KeyS);
map.bind_key("move.left", KeyCode::KeyA);
map.bind_key("move.right", KeyCode::KeyD);
map.bind_key("move.up", KeyCode::Space);
map.bind_key("move.down", KeyCode::ShiftLeft);

input.add_layer(
    input::LayerDescriptor::new("guide-fps", input::LayerPriority(10)),
    map.into_layer(),
);
```

`ActionMap` binds hardware inputs to logical action IDs. Supported triggers:

- `bind_key(action, KeyCode)` — single key press
- `bind_mouse_button(action, MouseButton)` — single mouse button
- `bind(ActionBinding)` — full binding with modifiers, scale, and consumption

Action values are floats clamped to `[0.0, 1.0]`. Press sets the value to the binding's `scale` (default 1.0); release sets it to 0.0. Modifier requirements (`BindingModifiers`) gate activation: if a binding requires Shift and Shift is not held, the binding does not fire.

### TOML Action Profiles

> Provenance: `G-06-TOML` — Excerpt

Actions can be loaded from versioned TOML profiles (`version = 1`):

```rust
// Load from file:
let map = ActionMap::load_toml_file("bindings/profile.toml")?;

// Or parse a string:
let map = ActionMap::from_toml_str(r#"
version = 1
[[bindings]]
action = "move.forward"
trigger = { key = "KeyW" }

[[bindings]]
action = "ui.click"
trigger = { mouse_button = "Left" }
modifiers = { shift = true }
consume = true
"#)?;

// Serialize back:
let toml = map.to_toml_string()?;
map.save_toml_file("bindings/exported.toml")?;
```

Supported key code names are listed in the `KEY_CODE_TABLE` constant in `src/input/src/lib.rs`. Mouse button names are `"Left"`, `"Right"`, `"Middle"`, `"Back"`, `"Forward"`, and `"Other(N)"`.

## The dispatch_frame Boundary

> Provenance: `G-06-DISPATCH` — Conceptual; behavior matches `InputSystem::dispatch_frame()`

`dispatch_frame()` is the single frame boundary. It runs exactly once per frame (called by `begin_app_frame`):

1. **Reset transients:** `mouse_delta`, `scroll_delta_lines`, `just_pressed`/`just_released` for keys, buttons, and actions are cleared.
2. **Rebuild layer layout:** If layers were added/removed/re-prioritized since the last frame, the dispatch groups are rebuilt (enabled layers sorted by descending priority, insertion order within priority).
3. **Dispatch queued events:** Each event is processed through priority groups. The raw snapshot is updated (tracking key/button state, mouse delta accumulation, scroll accumulation). Layer handlers receive the event and can update action values via `InputContext`.
4. **Refresh action snapshot:** Action values, `just_pressed`, and `just_released` are copied from the internal `ActionStateStore` into the pollable `InputSnapshot`.
5. **`on_frame_end` callbacks:** All enabled layers receive `on_frame_end(&snapshot, &mut ctx)` in insertion order, allowing post-frame work.
6. **Clear event queue.**

## Polling Snapshots

> Provenance: `G-06-SNAPSHOT` — Full Match (from checkpoint excerpt)

The snapshot is available after `dispatch_frame()` and exposes:

```rust
let snap = input.snapshot();

// Key state
if snap.key_down(KeyCode::Space) { /* ... */ }
if snap.key_just_pressed(KeyCode::KeyE) { /* ... */ }
if snap.key_just_released(KeyCode::Escape) { /* ... */ }

// Mouse state
let (dx, dy) = snap.mouse_delta();
let scroll = snap.scroll_delta_lines();
if snap.mouse_button_just_pressed(MouseButton::Left) { /* ... */ }
if snap.cursor_in_window() { /* ... */ }

// Action state (action values from action maps)
if snap.action_pressed(&ActionId::new("jump")) { /* ... */ }
let move_fwd = snap.action_value(&ActionId::new("move.forward"));
if snap.action_just_released(&ActionId::new("menu.toggle")) { /* ... */ }

// Iterate all active action values
for (action, value) in snap.action_values() {
    println!("{action}: {value}");
}
```

### Transient Fields (frame-scoped)

These fields are reset to zero at the start of each `dispatch_frame`:

| Field | Reset Value |
|-------|-------------|
| `mouse_delta` | `(0.0, 0.0)` |
| `scroll_delta_lines` | `0.0` |
| `keys_just_pressed` | empty |
| `keys_just_released` | empty |
| `buttons_just_pressed` | empty |
| `buttons_just_released` | empty |
| `action_just_pressed` | empty |
| `action_just_released` | empty |

Stable fields (`keys_down`, `buttons_down`, `action_values`, `cursor_in_window`) persist across frames until their state changes.

## Platform Input Routing

> Provenance: `G-06-ROUTING` — Full Match (from checkpoint)

```rust
match engine::input::route_platform_input_to_app(&mut renderer, &window, &mut input, &event) {
    Ok(_) => {}
    Err(e) => {
        eprintln!("input routing failed: {e}");
        elwt.exit();
        return;
    }
}
```

`route_platform_input_to_app` is the critical boundary. It:

1. Routes the raw `winit::Event` through the renderer's internal input system for platform side effects: ImGui keyboard/mouse capture, debug UI toggles (F1/F2), cursor confinement, capture hotkeys.
2. For events the renderer did **not** consume, queues normalized `InputEvent` values into your app-owned `InputSystem`.

This replaces `renderer.update_input()` (the compatibility helper). Your app never directly calls the renderer's input API.

### What the Renderer Consumes

The renderer may suppress input for these reasons (exposed as `RendererInputSuppression`):

| Suppression | When |
|-------------|------|
| `UiKeyboardCapture` | ImGui wants keyboard |
| `UiMouseCapture` | ImGui wants mouse (hovering a window) |
| `PlatformShortcut` | Renderer-internal hotkey (debug toggle, capture) |
| `OtherWindow` | Event targeted a different window |
| `UnsupportedEvent` | Event type the input system does not process |

When input is suppressed, it is **not** queued into your `InputSystem`.

## InputActionEventEmitter

> Provenance: `G-06-EMITTER` — Concept; behavior matches `InputActionEventEmitter::emit_from_snapshot()`

The `InputActionEventEmitter` bridges the input system to the event bus. After `dispatch_frame()`, `emit_from_snapshot()` reads the snapshot's action values and emits `EngineEvent::Input` events:

```rust
use engine::prelude::InputActionEventEmitter;

let mut action_events = InputActionEventEmitter::new();

// Called inside begin_app_frame:
action_events.emit_from_snapshot(&mut events, input.snapshot(), frame_id);
// This emits:
//   InputActionEvent { action, phase: Pressed, ... }  for actions that just became > 0
//   InputActionEvent { action, phase: Released, ... } for actions that just became 0
//   InputActionEvent { action, phase: Changed, ... }  for actions whose value changed
```

Subscribers on the event bus can listen for `EngineEvent::Input` to react to actions.

## FPSController Integration

> Provenance: `G-06-FPS` — Excerpt (checkpoint pattern)

```rust
let mut fps_controller = FPSController::new(0.002, 1.0);

// In the update loop, after dispatch_frame:
fps_controller.update_from_snapshot(
    input.snapshot(),
    simulated_seconds,
    &mut camera,
);
```

The `FPSController` reads the finalized snapshot:
- Mouse delta → camera yaw/pitch (sensitivity: first constructor arg, default 0.002)
- WASD action values → camera position (speed: second constructor arg, default 1.0)

It uses the default action names: `move.forward`, `move.backward`, `move.left`, `move.right`, `move.up`, `move.down`. Bind these actions in any layer to enable FPS movement.

## UI Capture Pattern

> Provenance: `G-06-CAPTURE` — Excerpt from `src/input/src/lib.rs`

Use `CaptureLayer` to block input from reaching gameplay when a UI is active:

```rust
use input::{CaptureLayer, LayerDescriptor, priority_bands};

// When opening a menu:
let capture_handle = input.add_layer(
    LayerDescriptor::new("menu-capture", priority_bands::EDITOR_UI_CAPTURE),
    CaptureLayer::new(true, true), // consume keyboard AND mouse
);

// When closing the menu:
input.remove_layer(capture_handle);
```

The convenience function `editor_ui_capture_layer()` returns a pre-configured `(LayerDescriptor, CaptureLayer)` at `EDITOR_UI_CAPTURE` priority:

```rust
let (descriptor, capture) = input::editor_ui_capture_layer();
let handle = input.add_layer(descriptor, capture);
```

## Cursor Handling

The renderer manages cursor grab mode via its platform side effects in `route_platform_input_to_app`. Cursor visibility and confinement are set through the renderer:

```rust
// The renderer tracks cursor-in-window state internally.
// Your snapshot reflects it:
if !input.snapshot().cursor_in_window() {
    // Cursor left the window — pause mouse-look, show system cursor.
}

// The renderer sets cursor grab internally for ImGui and debug capture.
// Apps should not call window.set_cursor_grab() directly — the renderer
// owns that state and may override it.
```

For full cursor confinement details, see [`docs/internal/09-input-winit-integration.md`](../internal/09-input-winit-integration.md).

## Debug Snapshot

> Provenance: `G-06-DEBUG` — Excerpt

`InputSystem` exposes a lightweight debug snapshot for diagnostics:

```rust
let debug = input.debug_snapshot();
println!(
    "queued={} layers={} active_layers={} consumed={}",
    debug.queued_events,
    debug.layer_count,
    debug.active_layer_count,
    debug.last_dispatch_consumed_events,
);
```

## Runnable Verification

Run the input crate's own test suite:

```sh
cargo test -p input
```

Expected: all tests pass (layer ordering, consumption, snapshot transients, action-map roundtrip, modifier state, profile parsing).

Build the checkpoint app (which exercises the complete app-owned input path):

```sh
cargo check --locked --manifest-path examples/guide_app/Cargo.toml
```

## App-Owned vs Renderer-Owned Input

| Path | `route_platform_input_to_app` | `update_input` |
|------|:---:|:---:|
| App-owned checkpoint (Ch 04) | ✓ | — |
| Renderer examples (`demo_pbr`, `api_test`, etc.) | — | ✓ (compatibility/diagnostic) |
| Custom apps | ✓ | — |

**Do not use `renderer.update_input()` in custom apps.** It couples your input to renderer-owned state. The app-owned path via `route_platform_input_to_app` is the supported pattern.

## Next

Continue to [07 — Events & Lifecycle](07-events-and-lifecycle.md) to understand how input actions flow into the event bus and how frame lifecycle events are emitted and drained.
