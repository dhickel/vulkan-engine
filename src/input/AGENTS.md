# Input Package Agent Guide (`src/input`)

This file is the detailed maintenance guide for the `input` crate.

## Package Role

`input` provides a lightweight, frame-based input broadcast system used by the renderer and camera controller.
It intentionally separates raw event ingestion from gameplay/camera consumers.

Primary file: `src/input/src/lib.rs`

## Core Types and Contracts

### `ListenerType`

- Enum values: `Window`, `Gui`, `GameInput`.
- Used with optional `ListenFilter` to restrict which listeners receive broadcasts.

### Listener Traits

- `KeyboardListener`
- `MousePosListener`
- `MouseBListener`
- `MouseScrollListener`

All listeners expose:
- `listener_type()` for class filtering.
- `listener_id()` for ID filtering.
- Event-specific `listener_for(...)` and/or `broadcast(...)` methods.

### `InputManager`

State model:
- Buffers incoming input deltas/events for one frame.
- On `update()`, broadcasts buffered state to all registered listeners.
- Clears all transient states after broadcast.

Important fields:
- `mouse_delta`, `scroll_state`
- `m_button_states`, `key_states`
- listener vectors (trait objects in `Rc<RefCell<...>>`)
- `listen_filter: Option<ListenFilter>`

## Runtime Integration (Current)

Renderer integration (`src/renderer/src/lib.rs`) is:
1. Create one `InputManager`.
2. Register FPS controller as key + mouse position listener.
3. Push winit events into manager via `add_keycode`, `update_mouse_pos`, `update_scroll_state`.
4. Call `input_manager.update()` once per frame before simulation/render.

This means input is edge/state buffered per frame and not processed immediately on event receipt.

## Important Behavior Details

- `add_keycode` appends `(KeyCode, pressed)` events to `key_states`.
- Keyboard listeners get all buffered key events for the frame and can filter by `listener_for`.
- Mouse position listener currently receives raw motion delta tuple from `DeviceEvent::MouseMotion`.
- Scroll listeners receive single `f32` state per frame.

## Current Gaps and Gotchas

1. Listener ID uniqueness is not enforced.
- Duplicate IDs are allowed and can cause duplicated processing for filters by ID.

2. No unregister API.
- Once registered, listeners cannot be removed without reconstructing manager state.

3. Modifier state is currently underused.
- `modifiers` exists and is passed to listeners, but renderer event loop does not populate it meaningfully.

4. Mouse button broadcasts are press-only in current manager logic.
- `broadcast_m_buttons` emits `pressed = true` for collected set members.
- Release semantics are not represented in `m_button_states`.

5. Window listener impl is surprising.
- `impl KeyboardListener for winit::window::Window` toggles blur on Escape.
- This is unusual coupling and is not central to current runtime flow.

6. `InputMap<K, V>` is generic utility but lightly used.
- Trait bounds are stricter than needed in most cases.
- If expanding it, audit bounds before widening usage.

## Safe Editing Guidance

When changing this crate:
- Preserve `update()` as the frame boundary unless intentionally redesigning event semantics.
- Keep listener trait APIs stable unless all downstream consumers are updated.
- If adding immediate-mode input, document coexistence with frame-buffered mode.
- If implementing unregister/replace listeners, define explicit ownership and lifecycle rules.

## Suggested First Improvements

1. Enforce listener ID uniqueness at registration.
2. Add unregister methods (`remove_*_listener(listener_id)`).
3. Add explicit modifier update API and wire from `WindowEvent::ModifiersChanged`.
4. Add button release tracking.
5. Add focused tests for filtering (`TypeFilter`, `IdFilter`) and per-frame reset behavior.

## Relevant Files

- `src/input/src/lib.rs`
- Consumer example: `src/renderer/src/lib.rs`
- Camera listener implementation: `src/renderer/src/data/camera.rs`
