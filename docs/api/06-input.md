# Input System

> Source: [`src/input/src/lib.rs`](../src/input/src/lib.rs) — no legacy docs consulted.

## Overview

The input system provides layered event dispatch with priority-based consumption and frame-buffered polling. It's designed so engine-level handlers (debug UI, camera) can consume events before gameplay code sees them.

## Core Concepts

- **Layers** (`InputLayer` trait): handlers that receive events in priority order
- **Priority bands**: predefined ranges for engine, UI, gameplay, and debug code
- **Consumption**: higher-priority layers can consume events, blocking lower priorities
- **Snapshots** (`InputSnapshot`): per-frame pollable state for gameplay systems
- **Action maps** (`ActionMap`): configurable bindings from raw input to named actions

## InputSystem Lifecycle

```rust
let mut input = InputSystem::new();

// Feed winit events each frame
input.queue_winit_window_event(&event);

// Or feed manually
input.queue_mouse_motion((dx, dy));
input.queue_event(InputEvent::Key { code, state, repeat, modifiers });

// Dispatch all queued events through layers
input.dispatch_frame();

// Poll the snapshot
let snapshot = input.snapshot();
if snapshot.key_down(KeyCode::KeyW) { ... }
```

Defined at [`lib.rs`](../src/input/src/lib.rs).

### Integration with Renderer

The `Renderer` owns an `InputSystem` internally. Use `renderer.update_input()` to auto-feed winit events. Access the system directly via `renderer.input()`/`renderer.input_mut()`.

## Layers & Priority

### InputLayer Trait

```rust
pub trait InputLayer {
    fn on_event(&mut self, event: &InputEvent, ctx: &mut InputContext<'_>) -> InputConsume;
    fn on_frame_end(&mut self, snapshot: &InputSnapshot, ctx: &mut InputContext<'_>);
}

pub enum InputConsume { Ignored, Consumed }
```

Defined at [`lib.rs`](../src/input/src/lib.rs). Return `Consumed` to prevent lower-priority layers from seeing the event. `on_frame_end` is called after all events are dispatched — use for action-map resolution or state cleanup.

### Priority Bands

```rust
pub mod priority_bands {
    pub const ENGINE_CAPTURE_MIN: LayerPriority = LayerPriority(900);
    pub const ENGINE_CAPTURE_MAX: LayerPriority = LayerPriority(1000);
    pub const UI_ROUTING_MIN: LayerPriority = LayerPriority(500);
    pub const UI_ROUTING_MAX: LayerPriority = LayerPriority(899);
    pub const GAMEPLAY_MIN: LayerPriority = LayerPriority(100);
    pub const GAMEPLAY_MAX: LayerPriority = LayerPriority(499);
    pub const DEBUG_MIN: LayerPriority = LayerPriority(0);
    pub const DEBUG_MAX: LayerPriority = LayerPriority(99);
}
```

Defined at [`lib.rs`](../src/input/src/lib.rs). Higher numbers = higher priority. Within the same priority, multiple layers execute as peers (consumption by one doesn't block peers in the same band).

### Registering a Layer

```rust
let desc = LayerDescriptor::new("my_gameplay", priority_bands::GAMEPLAY_MIN);
let handle = input.add_layer(desc, MyLayer::new());

// Runtime control
input.set_layer_enabled(handle, false);
input.set_layer_priority(handle, priority_bands::UI_ROUTING_MIN);
input.remove_layer(handle);
```

## InputSnapshot — Polling

```rust
pub struct InputSnapshot {
    // Modifiers
    pub fn modifiers(&self) -> ModifiersState;

    // Mouse
    pub fn mouse_delta(&self) -> (f64, f64);
    pub fn scroll_delta_lines(&self) -> f32;
    pub fn cursor_in_window(&self) -> bool;

    // Keys (instantaneous state)
    pub fn key_down(&self, key: KeyCode) -> bool;
    pub fn key_just_pressed(&self, key: KeyCode) -> bool;
    pub fn key_just_released(&self, key: KeyCode) -> bool;

    // Mouse buttons
    pub fn mouse_button_down(&self, button: MouseButton) -> bool;
    pub fn mouse_button_just_pressed(&self, button: MouseButton) -> bool;
    pub fn mouse_button_just_released(&self, button: MouseButton) -> bool;

    // Actions (resolved from action maps)
    pub fn action_value(&self, action: &ActionId) -> f32;
    pub fn action_pressed(&self, action: &ActionId) -> bool;
    pub fn action_just_pressed(&self, action: &ActionId) -> bool;
    pub fn action_just_released(&self, action: &ActionId) -> bool;
}
```

Defined at [`lib.rs`](../src/input/src/lib.rs). "just_pressed" / "just_released" are edge-triggered (true only on the frame the transition occurred). "down" / "pressed" are level-triggered (true while held).

## Action Maps

Action maps decouple game logic from physical inputs. Bind keys/buttons to named actions with optional modifiers and scaling:

```rust
let mut map = ActionMap::new();

// Simple bindings
map.bind_key("move_forward", KeyCode::KeyW);
map.bind_mouse_button("fire", MouseButton::Left);

// With modifiers
map.bind(ActionBinding::key("sprint_forward", KeyCode::KeyW)
    .with_modifiers(BindingModifiers { shift: true, ..Default::default() }));

// From TOML text
let map = ActionMap::from_toml_str(&std::fs::read_to_string("input.toml")?)?;

// Convert to layer
let layer = map.into_layer();
input.add_layer(LayerDescriptor::new("bindings", priority_bands::GAMEPLAY_MIN), layer);
```

### TOML Profile Format

Action maps can be loaded from TOML files for data-driven input configuration:

```toml
version = 1

[[bindings]]
action = "move_forward"
trigger = { key = "KeyW" }
modifiers = { shift = false, ctrl = false, alt = false, super_key = false }
scale = 1.0
consume = false

[[bindings]]
action = "fire"
trigger = { mouse_button = "Left" }
modifiers = {}
scale = 1.0
consume = false
```

The profile parser lives on `ActionMap`. The renderer does not automatically
discover or load an input profile file; apps choose when to read TOML text,
call `ActionMap::from_toml_str`, and install the resulting layer with
`renderer.input_mut().add_layer(...)`. `ActionMap::save_toml_file` and
`ActionMap::load_toml_file` are available when the app wants direct file I/O.

#### Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | integer | Yes | Must be `1` |
| `bindings` | array | Yes | One or more binding entries |

**Per-binding fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `action` | string | Yes | — | Named action identifier (e.g. `"move_forward"`) |
| `trigger.key` | string | One of `key`/`mouse_button` | — | Key code (e.g. `"KeyW"`, `"Escape"`, `"Space"`) |
| `trigger.mouse_button` | string | One of `key`/`mouse_button` | — | Mouse button (`"Left"`, `"Right"`, `"Middle"`) |
| `modifiers.shift` | bool | No | `false` | Require Shift held |
| `modifiers.ctrl` | bool | No | `false` | Require Ctrl held |
| `modifiers.alt` | bool | No | `false` | Require Alt held |
| `modifiers.super_key` | bool | No | `false` | Require Super/Win held |
| `scale` | float | No | `1.0` | Action value scale (0.0–1.0) |
| `consume` | bool | No | `false` | If true, consume event when binding fires |
| `context` | string | No | — | Optional context tag (for context-sensitive bindings) |

**Rules:**
- Exactly one trigger field (`key` or `mouse_button`) must be present per binding — specifying both is an error.
- `action` must be non-empty.
- Key codes are Rust `KeyCode` variant names (e.g. `"KeyW"`, `"Digit1"`, `"F5"`, `"Escape"`).
- Mouse buttons: `"Left"`, `"Right"`, `"Middle"`, `"Back"`, `"Forward"`, `"Other(N)"`.

Defined at the parser in [`src/input/src/lib.rs:1172-1225`](../src/input/src/lib.rs:1172).

## InputContext — Setting Action Values

Inside `on_event` or `on_frame_end`, layers can set action values via the context:

```rust
fn on_event(&mut self, event: &InputEvent, ctx: &mut InputContext<'_>) -> InputConsume {
    ctx.set_action_value(&ActionId::new("move_forward"), 1.0);
    InputConsume::Ignored
}
```

Actions set this way appear in the snapshot's `action_value()`, `action_pressed()`, etc.

## Debug Snapshot

```rust
pub struct InputDebugSnapshot {
    pub queued_events: usize,
    pub layer_count: usize,
    pub active_layer_count: usize,
    pub last_dispatch_consumed_events: usize,
}
```

Query via `input.debug_snapshot()`. Useful for verifying the input pipeline is wired correctly.

## See Also

- [01-quickstart.md](01-quickstart.md) — basic render loop wiring
- [Internal: Input-winit integration](../internal/06-input-internals.md)
- [src/input/src/lib.rs](../src/input/src/lib.rs) — full implementation
