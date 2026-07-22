# Input System Internals

> Source: [`src/input/src/lib.rs`](../../src/input/src/lib.rs) — no legacy docs consulted.

## Architecture

The input system is a single-file crate with three layers of abstraction:

1. **Event queue** — raw `InputEvent`s buffered per frame
2. **Layer dispatch** — priority-ordered `InputLayer` handlers with consumption
3. **Snapshot polling** — `InputSnapshot` built from dispatched events + action state

## Event Ingestion

### Winit Bridge

`InputSystem::queue_winit_window_event()` at [`lib.rs`](../../src/input/src/lib.rs) maps winit events:

| Winit Event | InputEvent |
|-------------|-----------|
| `KeyboardInput` → `PhysicalKey::Code(code)` | `Key { code, state, repeat, modifiers }` |
| `ModifiersChanged` | `ModifiersChanged { modifiers }` |
| `MouseInput` | `MouseButton { button, state, modifiers }` |
| `MouseWheel` → `LineDelta` | `MouseWheel { line_delta }` |
| `CursorEntered` | `CursorFocus { entered: true }` |
| `CursorLeft` | `CursorFocus { entered: false }` |

`CursorMoved` events are not automatically captured from winit — the renderer calls `queue_mouse_motion(delta)` separately.

### Manual Queuing

```rust
input.queue_event(InputEvent::Key { code, state, repeat, modifiers });
input.queue_mouse_motion((dx, dy));
input.queue_scroll_lines(line_delta);
```

Events are appended to a `Vec<InputEvent>` and dispatched on `dispatch_frame()`.

## Dispatch Algorithm

`dispatch_frame()` executes in three phases:

### Phase 1: Begin-Frame Reset

Clears transient snapshot fields (mouse delta, scroll delta, just-pressed/released sets). Clears action state transients.

### Phase 2: Layer Layout Rebuild

If the layer layout is dirty (layer added/removed/enabled/priority-changed):
1. Filter to enabled layers
2. Group by `LayerPriority` (layers with same priority form a peer group)
3. Sort groups by priority descending (higher priority dispatches first)
4. Store in `dispatch_groups: Vec<Vec<LayerHandle>>`

### Phase 3: Event Dispatch

For each queued event:
1. Apply event to raw snapshot (update key-down, mouse-delta, scroll, etc.)
2. Iterate priority groups from highest to lowest
3. Within each group, call `on_event()` on every layer
4. If **any layer** in a group returns `Consumed`, stop — lower priority groups don't see the event
5. Layers within the same priority group all see the event regardless of consumption

### Phase 4: Frame-End Callbacks

After all events are dispatched, `on_frame_end()` is called on every enabled layer in insertion order. This is where action-map layers typically resolve bindings.

### Phase 5: Action Snapshot Refresh

Action state is copied to the snapshot: `action_values`, `action_just_pressed`, `action_just_released`.

## Raw Snapshot vs Action Snapshot

The `InputSnapshot` has two families of data:

- **Raw input**: `key_down(key)`, `mouse_delta()`, `scroll_delta_lines()`, etc. — populated directly from events during dispatch
- **Action state**: `action_value(action)`, `action_pressed(action)`, `action_just_pressed(action)` — populated from `ActionStateStore` after all layers process

This split allows gameplay code to poll either raw input or configured actions depending on needs.

## Action Map Resolution

`ActionMapLayer::on_event()` checks each `ActionBinding` against the current event:

1. Match trigger type (Key or MouseButton) and code
2. Check modifier requirements (shift, ctrl, alt, super)
3. On match: set the action value to `binding.scale` (press) or 0.0 (release)
4. If `binding.consume` is true, the event is consumed

On `on_frame_end()`, the layer finalizes action state — this is where held-but-not-released actions maintain their value.

## Debug Counters

```rust
pub struct InputDebugSnapshot {
    pub queued_events: usize,              // events waiting for dispatch
    pub layer_count: usize,                // total registered layers
    pub active_layer_count: usize,         // enabled layers (sum across all groups)
    pub last_dispatch_consumed_events: usize, // events consumed this frame
}
```

Query via `input.debug_snapshot()`. Register a custom debug view on the `Renderer` to display these.

## TOML Profile Parser

`ActionMap::from_toml_str()` at [`lib.rs:1172-1225`](../../src/input/src/lib.rs#L1172) expects:

```toml
version = 1
[[bindings]]
action = "action_name"
trigger = { key = "KeyW" }           # OR { mouse_button = "Left" }
modifiers = { shift = false, ctrl = false, alt = false, super_key = false }
scale = 1.0
consume = false
context = "optional_context"
```

The parser validates:
- Exactly one trigger per binding (key XOR mouse_button)
- Non-empty action IDs
- Recognized key codes and mouse button names

## See Also

- [../api/06-input-polling-and-listeners.md](../api/06-input-polling-and-listeners.md) — public input API
- [src/input/src/lib.rs](../../src/input/src/lib.rs) — full implementation
