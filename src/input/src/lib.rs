//! Input system with layered dispatch, priority groups, and optional action mapping.
//!
//! Design goals:
//! - Frame-buffered ingest + dispatch boundary (`dispatch_frame`).
//! - Layered handling with same-priority peer execution.
//! - Event consumption that blocks only lower priorities.
//! - Polling snapshot for gameplay systems.
//! - Handle-based layer lifecycle (add/remove/enable/priority updates).

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

/// Stable typed action identifier. Re-exported from the canonical engine_events crate.
pub use engine_events::ActionId;

/// Private stable identifier for one binding instance within an action set.
///
/// Each `ActionBinding` registered with an `ActionMapLayer` receives a unique
/// `BindingInstanceId`. When multiple bindings map to the same action, the
/// input system tracks each binding's contribution independently so releasing
/// one binding does not deactivate the action while another binding still
/// contributes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BindingInstanceId(pub(crate) u64);

impl BindingInstanceId {
    /// Creates a caller-assigned binding instance id.
    ///
    /// Use a distinct non-zero id for every independent contribution to the
    /// same action. `ActionMapLayer` allocates ids automatically for mapped
    /// bindings.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

static NEXT_BINDING_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_binding_instance_id() -> BindingInstanceId {
    BindingInstanceId(NEXT_BINDING_INSTANCE_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LayerHandle(u64);

impl LayerHandle {
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Preferred name for a stable layer identifier in the rebuilt API.
pub type LayerId = LayerHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LayerPriority(pub i16);

#[derive(Clone, Debug)]
pub struct LayerDescriptor {
    pub name: String,
    pub priority: LayerPriority,
    pub enabled: bool,
}

impl LayerDescriptor {
    pub fn new(name: impl Into<String>, priority: LayerPriority) -> Self {
        Self {
            name: name.into(),
            priority,
            enabled: true,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Preferred name for a layer configuration descriptor in the rebuilt API.
pub type LayerSpec = LayerDescriptor;

/// Suggested priority bands for layer registration.
pub mod priority_bands {
    use super::LayerPriority;

    pub const ENGINE_CAPTURE_MIN: LayerPriority = LayerPriority(900);
    pub const ENGINE_CAPTURE_MAX: LayerPriority = LayerPriority(1000);
    pub const UI_ROUTING_MIN: LayerPriority = LayerPriority(500);
    pub const UI_ROUTING_MAX: LayerPriority = LayerPriority(899);
    pub const EDITOR_UI_CAPTURE: LayerPriority = LayerPriority(850);
    pub const GAMEPLAY_MIN: LayerPriority = LayerPriority(100);
    pub const GAMEPLAY_MAX: LayerPriority = LayerPriority(499);
    pub const DEBUG_MIN: LayerPriority = LayerPriority(0);
    pub const DEBUG_MAX: LayerPriority = LayerPriority(99);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputConsume {
    Ignored,
    Consumed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputDevice {
    Keyboard,
    Mouse,
}

#[derive(Clone, Copy, Debug)]
pub enum InputEvent {
    Key {
        code: KeyCode,
        state: ElementState,
        repeat: bool,
        modifiers: ModifiersState,
    },
    MouseMotion {
        delta: (f64, f64),
    },
    MouseButton {
        button: MouseButton,
        state: ElementState,
        modifiers: ModifiersState,
    },
    MouseWheel {
        line_delta: f32,
    },
    ModifiersChanged {
        modifiers: ModifiersState,
    },
    CursorFocus {
        entered: bool,
    },
}

impl InputEvent {
    pub fn device(self) -> Option<InputDevice> {
        match self {
            Self::Key { .. } => Some(InputDevice::Keyboard),
            Self::MouseMotion { .. } | Self::MouseButton { .. } | Self::MouseWheel { .. } => {
                Some(InputDevice::Mouse)
            }
            Self::ModifiersChanged { .. } | Self::CursorFocus { .. } => None,
        }
    }
}

#[derive(Clone, Default)]
struct ActionState {
    /// Aggregate value computed from all active binding contributions.
    value: f32,
    just_pressed: bool,
    just_released: bool,
}

/// Per-binding-instance contribution tracked for multi-binding actions.
#[derive(Clone, Copy, Debug, Default)]
struct BindingContribution {
    /// The value contributed by this specific binding instance.
    value: f32,
    /// Whether this binding instance is currently contributing (> 0).
    active: bool,
}

#[derive(Default)]
struct ActionStateStore {
    /// Aggregated per-action state.
    states: HashMap<ActionId, ActionState>,
    /// Per-binding-instance contributions, keyed by (action, instance).
    contributions: HashMap<(ActionId, BindingInstanceId), BindingContribution>,
}

impl ActionStateStore {
    /// Set the value for a specific binding instance, then recompute the
    /// aggregate action value from all active instances.
    fn set_instance_value(&mut self, action: &ActionId, instance: BindingInstanceId, value: f32) {
        let next_value = value.clamp(0.0, 1.0);
        let key = (action.clone(), instance);
        let contrib = self.contributions.entry(key).or_default();
        contrib.value = next_value;
        contrib.active = next_value > 0.0;

        self.recompute_action(action);
    }

    fn recompute_action(&mut self, action: &ActionId) {
        // Find the maximum contribution across all active instances for this action.
        let mut max_value = 0.0f32;
        let mut any_active = false;
        for ((key_action, _), contrib) in self.contributions.iter() {
            if key_action == action && contrib.active {
                max_value = max_value.max(contrib.value);
                any_active = true;
            }
        }

        let state = self.states.entry(action.clone()).or_default();
        let was_pressed = state.value > 0.0;
        let is_pressed = any_active;

        if is_pressed && !was_pressed {
            state.just_pressed = true;
        } else if !is_pressed && was_pressed {
            state.just_released = true;
        }

        state.value = if any_active { max_value } else { 0.0 };
    }

    /// Legacy path: sets an action value without an instance (for layers that
    /// don't use binding-instance tracking). Behaves as a single anonymous instance.
    fn set_action_value(&mut self, action: &ActionId, value: f32) {
        // Use instance 0 as the anonymous instance for backward compatibility.
        self.set_instance_value(action, BindingInstanceId(0), value);
    }

    fn clear_transients(&mut self) {
        for state in self.states.values_mut() {
            state.just_pressed = false;
            state.just_released = false;
        }
    }

    fn value(&self, action: &ActionId) -> f32 {
        self.states
            .get(action)
            .map(|state| state.value)
            .unwrap_or(0.0)
    }

    fn pressed(&self, action: &ActionId) -> bool {
        self.value(action) > 0.0
    }

    fn just_pressed(&self, action: &ActionId) -> bool {
        self.states
            .get(action)
            .map(|state| state.just_pressed)
            .unwrap_or(false)
    }

    fn just_released(&self, action: &ActionId) -> bool {
        self.states
            .get(action)
            .map(|state| state.just_released)
            .unwrap_or(false)
    }

    fn iter_values(&self) -> impl Iterator<Item = (&ActionId, f32)> {
        self.states
            .iter()
            .map(|(action, state)| (action, state.value))
    }
}

pub struct InputContext<'a> {
    action_state: &'a mut ActionStateStore,
}

impl<'a> InputContext<'a> {
    /// Set the value of one binding instance contributing to an action.
    ///
    /// When multiple bindings map to the same action, each binding should use
    /// its own `BindingInstanceId` so the system can track contributions
    /// independently. Releasing one binding does not deactivate the action
    /// while another binding instance still contributes.
    pub fn set_action_value(&mut self, action: &ActionId, value: f32) {
        self.action_state.set_action_value(action, value);
    }

    /// Set the value for a specific binding instance, then recompute the
    /// aggregate action value from all active instances.
    pub fn set_instance_value(
        &mut self,
        action: &ActionId,
        instance: BindingInstanceId,
        value: f32,
    ) {
        self.action_state
            .set_instance_value(action, instance, value);
    }
}

pub trait InputLayer {
    fn on_event(&mut self, _event: &InputEvent, _ctx: &mut InputContext<'_>) -> InputConsume {
        InputConsume::Ignored
    }

    fn on_frame_end(&mut self, _snapshot: &InputSnapshot, _ctx: &mut InputContext<'_>) {}
}

struct LayerEntry {
    desc: LayerDescriptor,
    layer: Box<dyn InputLayer>,
    insertion_order: u64,
}

#[derive(Clone, Debug)]
pub struct InputSnapshot {
    modifiers: ModifiersState,
    mouse_delta: (f64, f64),
    scroll_delta_lines: f32,
    keys_down: HashSet<KeyCode>,
    keys_just_pressed: HashSet<KeyCode>,
    keys_just_released: HashSet<KeyCode>,
    buttons_down: HashSet<MouseButton>,
    buttons_just_pressed: HashSet<MouseButton>,
    buttons_just_released: HashSet<MouseButton>,
    cursor_in_window: bool,
    action_values: Vec<(ActionId, f32)>,
    action_just_pressed: HashSet<ActionId>,
    action_just_released: HashSet<ActionId>,
}

/// Preferred name for frame-scoped pollable input state.
pub type FrameInputSnapshot = InputSnapshot;

impl Default for InputSnapshot {
    fn default() -> Self {
        Self {
            modifiers: ModifiersState::default(),
            mouse_delta: (0.0, 0.0),
            scroll_delta_lines: 0.0,
            keys_down: HashSet::new(),
            keys_just_pressed: HashSet::new(),
            keys_just_released: HashSet::new(),
            buttons_down: HashSet::new(),
            buttons_just_pressed: HashSet::new(),
            buttons_just_released: HashSet::new(),
            cursor_in_window: true,
            action_values: Vec::new(),
            action_just_pressed: HashSet::new(),
            action_just_released: HashSet::new(),
        }
    }
}

impl InputSnapshot {
    pub fn modifiers(&self) -> ModifiersState {
        self.modifiers
    }

    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    pub fn scroll_delta_lines(&self) -> f32 {
        self.scroll_delta_lines
    }

    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    pub fn key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys_just_pressed.contains(&key)
    }

    pub fn key_just_released(&self, key: KeyCode) -> bool {
        self.keys_just_released.contains(&key)
    }

    pub fn mouse_button_down(&self, button: MouseButton) -> bool {
        self.buttons_down.contains(&button)
    }

    pub fn mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        self.buttons_just_pressed.contains(&button)
    }

    pub fn mouse_button_just_released(&self, button: MouseButton) -> bool {
        self.buttons_just_released.contains(&button)
    }

    pub fn cursor_in_window(&self) -> bool {
        self.cursor_in_window
    }

    pub fn action_value(&self, action: &ActionId) -> f32 {
        self.action_values
            .iter()
            .find_map(|(id, value)| if id == action { Some(*value) } else { None })
            .unwrap_or(0.0)
    }

    pub fn action_pressed(&self, action: &ActionId) -> bool {
        self.action_value(action) > 0.0
    }

    pub fn action_just_pressed(&self, action: &ActionId) -> bool {
        self.action_just_pressed.contains(action)
    }

    pub fn action_just_released(&self, action: &ActionId) -> bool {
        self.action_just_released.contains(action)
    }

    pub fn action_values(&self) -> impl Iterator<Item = (&ActionId, f32)> {
        self.action_values
            .iter()
            .map(|(action, value)| (action, *value))
    }
}

#[derive(Clone, Debug, Default)]
pub struct InputDebugSnapshot {
    pub queued_events: usize,
    pub layer_count: usize,
    pub active_layer_count: usize,
    pub last_dispatch_consumed_events: usize,
}

/// Preferred name for frame debug counters.
pub type InputDebugFrame = InputDebugSnapshot;

pub struct InputSystem {
    next_layer_id: u64,
    next_insertion_order: u64,
    layers: HashMap<LayerHandle, LayerEntry>,
    queued_events: Vec<InputEvent>,
    ingest_modifiers: ModifiersState,
    snapshot: InputSnapshot,
    action_state: ActionStateStore,
    dispatch_groups: Vec<Vec<LayerHandle>>,
    frame_end_order: Vec<LayerHandle>,
    layer_layout_dirty: bool,
    debug: InputDebugSnapshot,
}

/// Preferred name for the input runtime in the rebuilt API.
pub type InputRuntime = InputSystem;

impl Default for InputSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl InputSystem {
    pub fn new() -> Self {
        Self {
            next_layer_id: 1,
            next_insertion_order: 1,
            layers: HashMap::new(),
            queued_events: Vec::new(),
            ingest_modifiers: ModifiersState::default(),
            snapshot: InputSnapshot::default(),
            action_state: ActionStateStore::default(),
            dispatch_groups: Vec::new(),
            frame_end_order: Vec::new(),
            layer_layout_dirty: true,
            debug: InputDebugSnapshot::default(),
        }
    }

    pub fn add_layer(
        &mut self,
        descriptor: LayerDescriptor,
        layer: impl InputLayer + 'static,
    ) -> LayerHandle {
        let handle = LayerHandle(self.next_layer_id);
        self.next_layer_id += 1;

        let entry = LayerEntry {
            desc: descriptor,
            layer: Box::new(layer),
            insertion_order: self.next_insertion_order,
        };

        self.next_insertion_order += 1;
        self.layers.insert(handle, entry);
        self.layer_layout_dirty = true;

        handle
    }

    pub fn remove_layer(&mut self, handle: LayerHandle) -> bool {
        let removed = self.layers.remove(&handle).is_some();
        if removed {
            self.layer_layout_dirty = true;
        }
        removed
    }

    pub fn set_layer_enabled(&mut self, handle: LayerHandle, enabled: bool) -> bool {
        if let Some(layer) = self.layers.get_mut(&handle) {
            layer.desc.enabled = enabled;
            self.layer_layout_dirty = true;
            return true;
        }

        false
    }

    pub fn set_layer_priority(&mut self, handle: LayerHandle, priority: LayerPriority) -> bool {
        if let Some(layer) = self.layers.get_mut(&handle) {
            layer.desc.priority = priority;
            self.layer_layout_dirty = true;
            return true;
        }

        false
    }

    pub fn layer_descriptor(&self, handle: LayerHandle) -> Option<&LayerDescriptor> {
        self.layers.get(&handle).map(|entry| &entry.desc)
    }

    pub fn queue_event(&mut self, event: InputEvent) {
        if let InputEvent::ModifiersChanged { modifiers } = event {
            self.ingest_modifiers = modifiers;
        }
        self.queued_events.push(event);
        self.debug.queued_events = self.queued_events.len();
    }

    pub fn queue_winit_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    self.queue_event(InputEvent::Key {
                        code,
                        state: key_event.state,
                        repeat: key_event.repeat,
                        modifiers: self.ingest_modifiers,
                    });
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.queue_event(InputEvent::ModifiersChanged {
                    modifiers: modifiers.state(),
                });
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.queue_event(InputEvent::MouseButton {
                    button: *button,
                    state: *state,
                    modifiers: self.ingest_modifiers,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let line_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        if y.abs() > x.abs() {
                            *y
                        } else {
                            *x
                        }
                    }
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
                };
                self.queue_event(InputEvent::MouseWheel { line_delta });
            }
            WindowEvent::CursorEntered { .. } => {
                self.queue_event(InputEvent::CursorFocus { entered: true });
            }
            WindowEvent::CursorLeft { .. } => {
                self.queue_event(InputEvent::CursorFocus { entered: false });
            }
            _ => {}
        }
    }

    pub fn queue_mouse_motion(&mut self, delta: (f64, f64)) {
        self.queue_event(InputEvent::MouseMotion { delta });
    }

    pub fn queue_scroll_lines(&mut self, line_delta: f32) {
        self.queue_event(InputEvent::MouseWheel { line_delta });
    }

    pub fn dispatch_frame(&mut self) {
        self.begin_frame_reset();
        self.rebuild_layer_layout_if_dirty();
        self.debug.layer_count = self.layers.len();
        self.debug.active_layer_count = self.dispatch_groups.iter().map(Vec::len).sum();

        let mut consumed_events = 0usize;

        for idx in 0..self.queued_events.len() {
            let event = self.queued_events[idx];
            self.apply_event_to_raw_snapshot(event);
            let mut stop_lower_priorities = false;
            for handles in &self.dispatch_groups {
                let mut group_consumed = false;
                for handle in handles {
                    let Some(layer_entry) = self.layers.get_mut(handle) else {
                        continue;
                    };

                    let mut ctx = InputContext {
                        action_state: &mut self.action_state,
                    };

                    if layer_entry.layer.on_event(&event, &mut ctx) == InputConsume::Consumed {
                        group_consumed = true;
                    }
                }

                if group_consumed {
                    stop_lower_priorities = true;
                    break;
                }
            }

            if stop_lower_priorities {
                consumed_events += 1;
            }
        }

        self.refresh_action_snapshot();
        for idx in 0..self.frame_end_order.len() {
            let handle = self.frame_end_order[idx];
            let Some(layer_entry) = self.layers.get_mut(&handle) else {
                continue;
            };
            let mut ctx = InputContext {
                action_state: &mut self.action_state,
            };
            layer_entry.layer.on_frame_end(&self.snapshot, &mut ctx);
        }

        self.debug.last_dispatch_consumed_events = consumed_events;
        self.queued_events.clear();
        self.debug.queued_events = 0;
    }

    pub fn snapshot(&self) -> &InputSnapshot {
        &self.snapshot
    }

    pub fn debug_snapshot(&self) -> &InputDebugSnapshot {
        &self.debug
    }

    pub fn action_value(&self, action: &ActionId) -> f32 {
        self.action_state.value(action)
    }

    pub fn action_pressed(&self, action: &ActionId) -> bool {
        self.action_state.pressed(action)
    }

    pub fn action_just_pressed(&self, action: &ActionId) -> bool {
        self.action_state.just_pressed(action)
    }

    pub fn action_just_released(&self, action: &ActionId) -> bool {
        self.action_state.just_released(action)
    }

    fn apply_event_to_raw_snapshot(&mut self, event: InputEvent) {
        match event {
            InputEvent::Key { code, state, .. } => match state {
                ElementState::Pressed => {
                    let inserted = self.snapshot.keys_down.insert(code);
                    if inserted {
                        self.snapshot.keys_just_pressed.insert(code);
                    }
                }
                ElementState::Released => {
                    self.snapshot.keys_down.remove(&code);
                    self.snapshot.keys_just_released.insert(code);
                }
            },
            InputEvent::MouseMotion { delta } => {
                self.snapshot.mouse_delta.0 += delta.0;
                self.snapshot.mouse_delta.1 += delta.1;
            }
            InputEvent::MouseButton { button, state, .. } => match state {
                ElementState::Pressed => {
                    let inserted = self.snapshot.buttons_down.insert(button);
                    if inserted {
                        self.snapshot.buttons_just_pressed.insert(button);
                    }
                }
                ElementState::Released => {
                    self.snapshot.buttons_down.remove(&button);
                    self.snapshot.buttons_just_released.insert(button);
                }
            },
            InputEvent::MouseWheel { line_delta } => {
                self.snapshot.scroll_delta_lines += line_delta;
            }
            InputEvent::ModifiersChanged { modifiers } => {
                self.snapshot.modifiers = modifiers;
            }
            InputEvent::CursorFocus { entered } => {
                self.snapshot.cursor_in_window = entered;
            }
        }
    }

    fn refresh_action_snapshot(&mut self) {
        self.snapshot.action_values.clear();
        self.snapshot.action_just_pressed.clear();
        self.snapshot.action_just_released.clear();

        for (action, value) in self.action_state.iter_values() {
            self.snapshot.action_values.push((action.clone(), value));
            if self.action_state.just_pressed(action) {
                self.snapshot.action_just_pressed.insert(action.clone());
            }
            if self.action_state.just_released(action) {
                self.snapshot.action_just_released.insert(action.clone());
            }
        }
    }

    fn begin_frame_reset(&mut self) {
        self.snapshot.mouse_delta = (0.0, 0.0);
        self.snapshot.scroll_delta_lines = 0.0;
        self.snapshot.keys_just_pressed.clear();
        self.snapshot.keys_just_released.clear();
        self.snapshot.buttons_just_pressed.clear();
        self.snapshot.buttons_just_released.clear();
        self.snapshot.action_just_pressed.clear();
        self.snapshot.action_just_released.clear();
        self.action_state.clear_transients();
    }

    fn rebuild_layer_layout_if_dirty(&mut self) {
        if !self.layer_layout_dirty {
            return;
        }

        let mut active_layers: Vec<(LayerPriority, u64, LayerHandle)> = self
            .layers
            .iter()
            .filter_map(|(handle, entry)| {
                if entry.desc.enabled {
                    Some((entry.desc.priority, entry.insertion_order, *handle))
                } else {
                    None
                }
            })
            .collect();
        active_layers.sort_by_key(|(priority, insertion, _)| (Reverse(*priority), *insertion));

        self.dispatch_groups.clear();
        let mut current_priority: Option<LayerPriority> = None;
        for (priority, _, handle) in active_layers {
            if current_priority != Some(priority) {
                self.dispatch_groups.push(Vec::new());
                current_priority = Some(priority);
            }
            if let Some(group) = self.dispatch_groups.last_mut() {
                group.push(handle);
            }
        }

        let mut frame_end: Vec<(u64, LayerHandle)> = self
            .layers
            .iter()
            .filter_map(|(handle, entry)| {
                if entry.desc.enabled {
                    Some((entry.insertion_order, *handle))
                } else {
                    None
                }
            })
            .collect();
        frame_end.sort_by_key(|(insertion, _)| *insertion);
        self.frame_end_order.clear();
        self.frame_end_order
            .extend(frame_end.into_iter().map(|(_, handle)| handle));

        self.layer_layout_dirty = false;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct BindingModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

impl BindingModifiers {
    fn matches(self, modifiers: ModifiersState) -> bool {
        if self.shift && !modifiers.shift_key() {
            return false;
        }
        if self.ctrl && !modifiers.control_key() {
            return false;
        }
        if self.alt && !modifiers.alt_key() {
            return false;
        }
        if self.super_key && !modifiers.super_key() {
            return false;
        }
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BindingTrigger {
    Key(KeyCode),
    MouseButton(MouseButton),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InputChord {
    pub key: Option<KeyCode>,
    pub mouse_button: Option<MouseButton>,
    pub require_shift: bool,
    pub require_ctrl: bool,
    pub require_alt: bool,
    pub require_super: bool,
}

impl InputChord {
    pub fn key(key: KeyCode) -> Self {
        Self {
            key: Some(key),
            mouse_button: None,
            require_shift: false,
            require_ctrl: false,
            require_alt: false,
            require_super: false,
        }
    }

    pub fn mouse(button: MouseButton) -> Self {
        Self {
            key: None,
            mouse_button: Some(button),
            require_shift: false,
            require_ctrl: false,
            require_alt: false,
            require_super: false,
        }
    }

    pub fn into_parts(self) -> Option<(BindingTrigger, BindingModifiers)> {
        let trigger = match (self.key, self.mouse_button) {
            (Some(key), None) => BindingTrigger::Key(key),
            (None, Some(button)) => BindingTrigger::MouseButton(button),
            _ => return None,
        };
        let modifiers = BindingModifiers {
            shift: self.require_shift,
            ctrl: self.require_ctrl,
            alt: self.require_alt,
            super_key: self.require_super,
        };
        Some((trigger, modifiers))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionBinding {
    pub action: ActionId,
    pub trigger: BindingTrigger,
    pub modifiers: BindingModifiers,
    pub scale: f32,
    pub consume: bool,
    pub context: Option<String>,
}

impl ActionBinding {
    pub fn key(action: impl Into<ActionId>, key: KeyCode) -> Self {
        Self {
            action: action.into(),
            trigger: BindingTrigger::Key(key),
            modifiers: BindingModifiers::default(),
            scale: 1.0,
            consume: false,
            context: None,
        }
    }

    pub fn mouse_button(action: impl Into<ActionId>, button: MouseButton) -> Self {
        Self {
            action: action.into(),
            trigger: BindingTrigger::MouseButton(button),
            modifiers: BindingModifiers::default(),
            scale: 1.0,
            consume: false,
            context: None,
        }
    }

    pub fn with_modifiers(mut self, modifiers: BindingModifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

}

fn default_binding_scale() -> f32 {
    1.0
}

#[derive(Clone, Debug, Default)]
pub struct ActionMap {
    bindings: Vec<ActionBinding>,
}

impl ActionMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bindings(&self) -> &[ActionBinding] {
        &self.bindings
    }

    pub fn bind_key(&mut self, action: impl Into<ActionId>, key: KeyCode) {
        self.bind(ActionBinding::key(action, key));
    }

    pub fn bind_mouse_button(&mut self, action: impl Into<ActionId>, button: MouseButton) {
        self.bind(ActionBinding::mouse_button(action, button));
    }

    pub fn bind(&mut self, binding: ActionBinding) {
        self.bindings.push(binding);
    }

    pub fn unbind_action(&mut self, action: &ActionId) {
        self.bindings.retain(|binding| &binding.action != action);
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn into_layer(self) -> ActionMapLayer {
        ActionMapLayer::new(self)
    }

    pub fn from_toml_str(content: &str) -> Result<Self, String> {
        let profile: InputProfileV1 =
            toml::from_str(content).map_err(|err| format!("toml parse error: {err}"))?;
        if profile.version != 1 {
            return Err(format!(
                "unsupported input profile version: {}",
                profile.version
            ));
        }

        let mut bindings = Vec::with_capacity(profile.bindings.len());
        for (index, binding) in profile.bindings.into_iter().enumerate() {
            if binding.action.trim().is_empty() {
                return Err(format!("binding[{index}] has empty action id"));
            }

            let trigger = match (binding.trigger.key, binding.trigger.mouse_button) {
                (Some(key_name), None) => {
                    BindingTrigger::Key(parse_key_code(&key_name).ok_or_else(|| {
                        format!("binding[{index}] has unsupported key code: {key_name}")
                    })?)
                }
                (None, Some(button_name)) => {
                    BindingTrigger::MouseButton(parse_mouse_button(&button_name).ok_or_else(
                        || format!("binding[{index}] has unsupported mouse button: {button_name}"),
                    )?)
                }
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "binding[{index}] must define exactly one trigger: key or mouse_button"
                    ));
                }
                (None, None) => {
                    return Err(format!(
                        "binding[{index}] must define a trigger (key or mouse_button)"
                    ));
                }
            };

            bindings.push(ActionBinding {
                action: ActionId::new(binding.action),
                trigger,
                modifiers: BindingModifiers {
                    shift: binding.modifiers.shift,
                    ctrl: binding.modifiers.ctrl,
                    alt: binding.modifiers.alt,
                    super_key: binding.modifiers.super_key,
                },
                scale: binding.scale,
                consume: binding.consume,
                context: binding.context,
            });
        }

        Ok(Self { bindings })
    }

    pub fn to_toml_string(&self) -> Result<String, String> {
        let bindings = self
            .bindings
            .iter()
            .map(|binding| {
                let trigger = match binding.trigger {
                    BindingTrigger::Key(key) => ProfileTriggerV1 {
                        key: Some(key_code_to_string(key)),
                        mouse_button: None,
                    },
                    BindingTrigger::MouseButton(button) => ProfileTriggerV1 {
                        key: None,
                        mouse_button: Some(mouse_button_to_string(button)),
                    },
                };

                ProfileBindingV1 {
                    action: binding.action.as_str().to_string(),
                    trigger,
                    modifiers: ProfileModifiersV1 {
                        shift: binding.modifiers.shift,
                        ctrl: binding.modifiers.ctrl,
                        alt: binding.modifiers.alt,
                        super_key: binding.modifiers.super_key,
                    },
                    scale: binding.scale,
                    consume: binding.consume,
                    context: binding.context.clone(),
                }
            })
            .collect();

        toml::to_string_pretty(&InputProfileV1 {
            version: 1,
            bindings,
        })
        .map_err(|err| err.to_string())
    }

    pub fn load_toml_file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let map = Self::from_toml_str(&content)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;
        Ok(map)
    }

    pub fn save_toml_file(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let content = self
            .to_toml_string()
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;
        std::fs::write(path, content)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputProfileV1 {
    version: u32,
    #[serde(default)]
    bindings: Vec<ProfileBindingV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileBindingV1 {
    action: String,
    trigger: ProfileTriggerV1,
    #[serde(default)]
    modifiers: ProfileModifiersV1,
    #[serde(default = "default_binding_scale")]
    scale: f32,
    #[serde(default)]
    consume: bool,
    #[serde(default)]
    context: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileTriggerV1 {
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    mouse_button: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileModifiersV1 {
    #[serde(default)]
    shift: bool,
    #[serde(default)]
    ctrl: bool,
    #[serde(default)]
    alt: bool,
    #[serde(default)]
    super_key: bool,
}

fn key_code_to_string(key: KeyCode) -> String {
    format!("{key:?}")
}

fn parse_key_code(value: &str) -> Option<KeyCode> {
    KEY_CODE_TABLE
        .iter()
        .find_map(|(name, key)| if *name == value { Some(*key) } else { None })
}

const KEY_CODE_TABLE: &[(&str, KeyCode)] = &[
    ("Backquote", KeyCode::Backquote),
    ("Backslash", KeyCode::Backslash),
    ("BracketLeft", KeyCode::BracketLeft),
    ("BracketRight", KeyCode::BracketRight),
    ("Comma", KeyCode::Comma),
    ("Digit0", KeyCode::Digit0),
    ("Digit1", KeyCode::Digit1),
    ("Digit2", KeyCode::Digit2),
    ("Digit3", KeyCode::Digit3),
    ("Digit4", KeyCode::Digit4),
    ("Digit5", KeyCode::Digit5),
    ("Digit6", KeyCode::Digit6),
    ("Digit7", KeyCode::Digit7),
    ("Digit8", KeyCode::Digit8),
    ("Digit9", KeyCode::Digit9),
    ("Equal", KeyCode::Equal),
    ("IntlBackslash", KeyCode::IntlBackslash),
    ("IntlRo", KeyCode::IntlRo),
    ("IntlYen", KeyCode::IntlYen),
    ("KeyA", KeyCode::KeyA),
    ("KeyB", KeyCode::KeyB),
    ("KeyC", KeyCode::KeyC),
    ("KeyD", KeyCode::KeyD),
    ("KeyE", KeyCode::KeyE),
    ("KeyF", KeyCode::KeyF),
    ("KeyG", KeyCode::KeyG),
    ("KeyH", KeyCode::KeyH),
    ("KeyI", KeyCode::KeyI),
    ("KeyJ", KeyCode::KeyJ),
    ("KeyK", KeyCode::KeyK),
    ("KeyL", KeyCode::KeyL),
    ("KeyM", KeyCode::KeyM),
    ("KeyN", KeyCode::KeyN),
    ("KeyO", KeyCode::KeyO),
    ("KeyP", KeyCode::KeyP),
    ("KeyQ", KeyCode::KeyQ),
    ("KeyR", KeyCode::KeyR),
    ("KeyS", KeyCode::KeyS),
    ("KeyT", KeyCode::KeyT),
    ("KeyU", KeyCode::KeyU),
    ("KeyV", KeyCode::KeyV),
    ("KeyW", KeyCode::KeyW),
    ("KeyX", KeyCode::KeyX),
    ("KeyY", KeyCode::KeyY),
    ("KeyZ", KeyCode::KeyZ),
    ("Minus", KeyCode::Minus),
    ("Period", KeyCode::Period),
    ("Quote", KeyCode::Quote),
    ("Semicolon", KeyCode::Semicolon),
    ("Slash", KeyCode::Slash),
    ("AltLeft", KeyCode::AltLeft),
    ("AltRight", KeyCode::AltRight),
    ("Backspace", KeyCode::Backspace),
    ("CapsLock", KeyCode::CapsLock),
    ("ContextMenu", KeyCode::ContextMenu),
    ("ControlLeft", KeyCode::ControlLeft),
    ("ControlRight", KeyCode::ControlRight),
    ("Enter", KeyCode::Enter),
    ("SuperLeft", KeyCode::SuperLeft),
    ("SuperRight", KeyCode::SuperRight),
    ("ShiftLeft", KeyCode::ShiftLeft),
    ("ShiftRight", KeyCode::ShiftRight),
    ("Space", KeyCode::Space),
    ("Tab", KeyCode::Tab),
    ("Convert", KeyCode::Convert),
    ("KanaMode", KeyCode::KanaMode),
    ("Lang1", KeyCode::Lang1),
    ("Lang2", KeyCode::Lang2),
    ("Lang3", KeyCode::Lang3),
    ("Lang4", KeyCode::Lang4),
    ("Lang5", KeyCode::Lang5),
    ("NonConvert", KeyCode::NonConvert),
    ("Delete", KeyCode::Delete),
    ("End", KeyCode::End),
    ("Help", KeyCode::Help),
    ("Home", KeyCode::Home),
    ("Insert", KeyCode::Insert),
    ("PageDown", KeyCode::PageDown),
    ("PageUp", KeyCode::PageUp),
    ("ArrowDown", KeyCode::ArrowDown),
    ("ArrowLeft", KeyCode::ArrowLeft),
    ("ArrowRight", KeyCode::ArrowRight),
    ("ArrowUp", KeyCode::ArrowUp),
    ("NumLock", KeyCode::NumLock),
    ("Numpad0", KeyCode::Numpad0),
    ("Numpad1", KeyCode::Numpad1),
    ("Numpad2", KeyCode::Numpad2),
    ("Numpad3", KeyCode::Numpad3),
    ("Numpad4", KeyCode::Numpad4),
    ("Numpad5", KeyCode::Numpad5),
    ("Numpad6", KeyCode::Numpad6),
    ("Numpad7", KeyCode::Numpad7),
    ("Numpad8", KeyCode::Numpad8),
    ("Numpad9", KeyCode::Numpad9),
    ("NumpadAdd", KeyCode::NumpadAdd),
    ("NumpadBackspace", KeyCode::NumpadBackspace),
    ("NumpadClear", KeyCode::NumpadClear),
    ("NumpadClearEntry", KeyCode::NumpadClearEntry),
    ("NumpadComma", KeyCode::NumpadComma),
    ("NumpadDecimal", KeyCode::NumpadDecimal),
    ("NumpadDivide", KeyCode::NumpadDivide),
    ("NumpadEnter", KeyCode::NumpadEnter),
    ("NumpadEqual", KeyCode::NumpadEqual),
    ("NumpadHash", KeyCode::NumpadHash),
    ("NumpadMemoryAdd", KeyCode::NumpadMemoryAdd),
    ("NumpadMemoryClear", KeyCode::NumpadMemoryClear),
    ("NumpadMemoryRecall", KeyCode::NumpadMemoryRecall),
    ("NumpadMemoryStore", KeyCode::NumpadMemoryStore),
    ("NumpadMemorySubtract", KeyCode::NumpadMemorySubtract),
    ("NumpadMultiply", KeyCode::NumpadMultiply),
    ("NumpadParenLeft", KeyCode::NumpadParenLeft),
    ("NumpadParenRight", KeyCode::NumpadParenRight),
    ("NumpadStar", KeyCode::NumpadStar),
    ("NumpadSubtract", KeyCode::NumpadSubtract),
    ("Escape", KeyCode::Escape),
    ("Fn", KeyCode::Fn),
    ("FnLock", KeyCode::FnLock),
    ("PrintScreen", KeyCode::PrintScreen),
    ("ScrollLock", KeyCode::ScrollLock),
    ("Pause", KeyCode::Pause),
    ("BrowserBack", KeyCode::BrowserBack),
    ("BrowserFavorites", KeyCode::BrowserFavorites),
    ("BrowserForward", KeyCode::BrowserForward),
    ("BrowserHome", KeyCode::BrowserHome),
    ("BrowserRefresh", KeyCode::BrowserRefresh),
    ("BrowserSearch", KeyCode::BrowserSearch),
    ("BrowserStop", KeyCode::BrowserStop),
    ("Eject", KeyCode::Eject),
    ("LaunchApp1", KeyCode::LaunchApp1),
    ("LaunchApp2", KeyCode::LaunchApp2),
    ("LaunchMail", KeyCode::LaunchMail),
    ("MediaPlayPause", KeyCode::MediaPlayPause),
    ("MediaSelect", KeyCode::MediaSelect),
    ("MediaStop", KeyCode::MediaStop),
    ("MediaTrackNext", KeyCode::MediaTrackNext),
    ("MediaTrackPrevious", KeyCode::MediaTrackPrevious),
    ("Power", KeyCode::Power),
    ("Sleep", KeyCode::Sleep),
    ("AudioVolumeDown", KeyCode::AudioVolumeDown),
    ("AudioVolumeMute", KeyCode::AudioVolumeMute),
    ("AudioVolumeUp", KeyCode::AudioVolumeUp),
    ("WakeUp", KeyCode::WakeUp),
    ("Meta", KeyCode::Meta),
    ("Hyper", KeyCode::Hyper),
    ("Turbo", KeyCode::Turbo),
    ("Abort", KeyCode::Abort),
    ("Resume", KeyCode::Resume),
    ("Suspend", KeyCode::Suspend),
    ("Again", KeyCode::Again),
    ("Copy", KeyCode::Copy),
    ("Cut", KeyCode::Cut),
    ("Find", KeyCode::Find),
    ("Open", KeyCode::Open),
    ("Paste", KeyCode::Paste),
    ("Props", KeyCode::Props),
    ("Select", KeyCode::Select),
    ("Undo", KeyCode::Undo),
    ("Hiragana", KeyCode::Hiragana),
    ("Katakana", KeyCode::Katakana),
    ("F1", KeyCode::F1),
    ("F2", KeyCode::F2),
    ("F3", KeyCode::F3),
    ("F4", KeyCode::F4),
    ("F5", KeyCode::F5),
    ("F6", KeyCode::F6),
    ("F7", KeyCode::F7),
    ("F8", KeyCode::F8),
    ("F9", KeyCode::F9),
    ("F10", KeyCode::F10),
    ("F11", KeyCode::F11),
    ("F12", KeyCode::F12),
    ("F13", KeyCode::F13),
    ("F14", KeyCode::F14),
    ("F15", KeyCode::F15),
    ("F16", KeyCode::F16),
    ("F17", KeyCode::F17),
    ("F18", KeyCode::F18),
    ("F19", KeyCode::F19),
    ("F20", KeyCode::F20),
    ("F21", KeyCode::F21),
    ("F22", KeyCode::F22),
    ("F23", KeyCode::F23),
    ("F24", KeyCode::F24),
    ("F25", KeyCode::F25),
    ("F26", KeyCode::F26),
    ("F27", KeyCode::F27),
    ("F28", KeyCode::F28),
    ("F29", KeyCode::F29),
    ("F30", KeyCode::F30),
    ("F31", KeyCode::F31),
    ("F32", KeyCode::F32),
    ("F33", KeyCode::F33),
    ("F34", KeyCode::F34),
    ("F35", KeyCode::F35),
];

fn mouse_button_to_string(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "Left".to_string(),
        MouseButton::Right => "Right".to_string(),
        MouseButton::Middle => "Middle".to_string(),
        MouseButton::Back => "Back".to_string(),
        MouseButton::Forward => "Forward".to_string(),
        MouseButton::Other(id) => format!("Other({id})"),
    }
}

fn parse_mouse_button(value: &str) -> Option<MouseButton> {
    match value {
        "Left" => Some(MouseButton::Left),
        "Right" => Some(MouseButton::Right),
        "Middle" => Some(MouseButton::Middle),
        "Back" => Some(MouseButton::Back),
        "Forward" => Some(MouseButton::Forward),
        _ => {
            let numeric = value
                .strip_prefix("Other(")?
                .strip_suffix(')')?
                .parse::<u16>()
                .ok()?;
            Some(MouseButton::Other(numeric))
        }
    }
}

/// Private tracked entry for one binding in the ActionMapLayer.
///
/// Each entry carries a full binding fingerprint (for reconciliation),
/// a unique stable instance ID, and the currently active trigger identity.
#[derive(Clone, Debug)]
struct TrackedBindingEntry {
    binding: ActionBinding,
    instance: BindingInstanceId,
    active_trigger: Option<BindingTrigger>,
}

pub struct ActionMapLayer {
    map: ActionMap,
    /// Tracked entries with stable IDs preserved across reconciliation.
    tracked: Vec<TrackedBindingEntry>,
}

impl ActionMapLayer {
    pub fn new(map: ActionMap) -> Self {
        let tracked: Vec<TrackedBindingEntry> = map
            .bindings
            .iter()
            .map(|binding| TrackedBindingEntry {
                binding: binding.clone(),
                instance: allocate_binding_instance_id(),
                active_trigger: None,
            })
            .collect();
        Self { map, tracked }
    }

    pub fn map(&self) -> &ActionMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut ActionMap {
        &mut self.map
    }

    /// Reconcile tracked entries with the current bindings.
    ///
    /// Greedy stable match by full binding equality plus occurrence ordinal:
    /// each new binding reuses the first unused prior entry with the same
    /// fingerprint.  Additions get new IDs.  Removed active entries have
    /// their contributions cleared.
    fn reconcile(&mut self, ctx: &mut InputContext<'_>) {
        let new_bindings = &self.map.bindings;
        let old_tracked = std::mem::take(&mut self.tracked);
        let mut used: Vec<bool> = vec![false; old_tracked.len()];
        let mut new_tracked: Vec<TrackedBindingEntry> = Vec::with_capacity(new_bindings.len());

        for new_binding in new_bindings {
            let reuse_idx = old_tracked
                .iter()
                .enumerate()
                .position(|(idx, old)| !used[idx] && old.binding == *new_binding);

            if let Some(idx) = reuse_idx {
                used[idx] = true;
                let mut entry = old_tracked[idx].clone();
                entry.binding = new_binding.clone();
                new_tracked.push(entry);
            } else {
                new_tracked.push(TrackedBindingEntry {
                    binding: new_binding.clone(),
                    instance: allocate_binding_instance_id(),
                    active_trigger: None,
                });
            }
        }

        // Clear contributions for removed bindings that were active.
        for (idx, old) in old_tracked.iter().enumerate() {
            if !used[idx] && old.active_trigger.is_some() {
                ctx.set_instance_value(&old.binding.action, old.instance, 0.0);
            }
        }

        self.tracked = new_tracked;
    }
}

impl InputLayer for ActionMapLayer {
    fn on_event(&mut self, event: &InputEvent, ctx: &mut InputContext<'_>) -> InputConsume {
        self.reconcile(ctx);

        let mut consumed = false;

        match event {
            InputEvent::Key {
                code,
                state,
                modifiers,
                ..
            } => {
                let trigger = BindingTrigger::Key(*code);
                match state {
                    ElementState::Pressed => {
                        for entry in &mut self.tracked {
                            if entry.binding.trigger == trigger
                                && entry.binding.modifiers.matches(*modifiers)
                            {
                                entry.active_trigger = Some(trigger);
                                ctx.set_instance_value(
                                    &entry.binding.action,
                                    entry.instance,
                                    entry.binding.scale,
                                );
                                if entry.binding.consume {
                                    consumed = true;
                                }
                            }
                        }
                    }
                    ElementState::Released => {
                        for entry in &mut self.tracked {
                            if entry.active_trigger == Some(trigger) {
                                entry.active_trigger = None;
                                ctx.set_instance_value(
                                    &entry.binding.action,
                                    entry.instance,
                                    0.0,
                                );
                                if entry.binding.consume {
                                    consumed = true;
                                }
                            }
                        }
                    }
                }
            }
            InputEvent::MouseButton {
                button,
                state,
                modifiers,
            } => {
                let trigger = BindingTrigger::MouseButton(*button);
                match state {
                    ElementState::Pressed => {
                        for entry in &mut self.tracked {
                            if entry.binding.trigger == trigger
                                && entry.binding.modifiers.matches(*modifiers)
                            {
                                entry.active_trigger = Some(trigger);
                                ctx.set_instance_value(
                                    &entry.binding.action,
                                    entry.instance,
                                    entry.binding.scale,
                                );
                                if entry.binding.consume {
                                    consumed = true;
                                }
                            }
                        }
                    }
                    ElementState::Released => {
                        for entry in &mut self.tracked {
                            if entry.active_trigger == Some(trigger) {
                                entry.active_trigger = None;
                                ctx.set_instance_value(
                                    &entry.binding.action,
                                    entry.instance,
                                    0.0,
                                );
                                if entry.binding.consume {
                                    consumed = true;
                                }
                            }
                        }
                    }
                }
            }
            InputEvent::CursorFocus { entered: false } => {
                // Focus loss clears all active triggers so no action gets stuck.
                for entry in &mut self.tracked {
                    if entry.active_trigger.is_some() {
                        entry.active_trigger = None;
                        ctx.set_instance_value(&entry.binding.action, entry.instance, 0.0);
                    }
                }
            }
            _ => {}
        }

        if consumed {
            InputConsume::Consumed
        } else {
            InputConsume::Ignored
        }
    }

    fn on_frame_end(&mut self, _snapshot: &InputSnapshot, ctx: &mut InputContext<'_>) {
        self.reconcile(ctx);
    }
}

pub struct CaptureLayer {
    pub consume_keyboard: bool,
    pub consume_mouse: bool,
}

impl CaptureLayer {
    pub fn new(consume_keyboard: bool, consume_mouse: bool) -> Self {
        Self {
            consume_keyboard,
            consume_mouse,
        }
    }
}

pub fn editor_ui_capture_layer() -> (LayerDescriptor, CaptureLayer) {
    (
        LayerDescriptor::new("editor-ui-capture", priority_bands::EDITOR_UI_CAPTURE),
        CaptureLayer::new(true, true),
    )
}

impl InputLayer for CaptureLayer {
    fn on_event(&mut self, event: &InputEvent, _ctx: &mut InputContext<'_>) -> InputConsume {
        let should_consume = match event {
            InputEvent::Key { .. } => self.consume_keyboard,
            InputEvent::MouseMotion { .. }
            | InputEvent::MouseButton { .. }
            | InputEvent::MouseWheel { .. } => self.consume_mouse,
            _ => false,
        };

        if should_consume {
            InputConsume::Consumed
        } else {
            InputConsume::Ignored
        }
    }
}

/// Converts winit `Modifiers` to a stable `ModifiersState`.
pub fn modifiers_state(modifiers: Modifiers) -> ModifiersState {
    modifiers.state()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use winit::event::DeviceId;

    struct TraceLayer {
        name: &'static str,
        consume: bool,
        trace: Rc<RefCell<Vec<&'static str>>>,
    }

    impl TraceLayer {
        fn new(name: &'static str, consume: bool, trace: Rc<RefCell<Vec<&'static str>>>) -> Self {
            Self {
                name,
                consume,
                trace,
            }
        }
    }

    impl InputLayer for TraceLayer {
        fn on_event(&mut self, _event: &InputEvent, _ctx: &mut InputContext<'_>) -> InputConsume {
            self.trace.borrow_mut().push(self.name);
            if self.consume {
                InputConsume::Consumed
            } else {
                InputConsume::Ignored
            }
        }
    }

    #[test]
    fn priority_group_consumption_runs_all_peers_blocks_lower_priorities() {
        let mut system = InputSystem::new();
        let trace = Rc::new(RefCell::new(Vec::new()));

        let a = system.add_layer(
            LayerDescriptor::new("a", LayerPriority(10)),
            TraceLayer::new("a", true, trace.clone()),
        );
        let b = system.add_layer(
            LayerDescriptor::new("b", LayerPriority(10)),
            TraceLayer::new("b", false, trace.clone()),
        );
        let c = system.add_layer(
            LayerDescriptor::new("c", LayerPriority(5)),
            TraceLayer::new("c", false, trace.clone()),
        );

        system.queue_event(InputEvent::CursorFocus { entered: true });
        system.dispatch_frame();

        assert_eq!(trace.borrow().as_slice(), &["a", "b"]);
        assert!(system.layer_descriptor(a).is_some());
        assert!(system.layer_descriptor(b).is_some());
        assert!(system.layer_descriptor(c).is_some());
        assert_eq!(system.debug_snapshot().last_dispatch_consumed_events, 1);
    }

    #[test]
    fn lower_priority_runs_when_higher_group_does_not_consume() {
        let mut system = InputSystem::new();
        let trace = Rc::new(RefCell::new(Vec::new()));

        system.add_layer(
            LayerDescriptor::new("high", LayerPriority(20)),
            TraceLayer::new("high", false, trace.clone()),
        );
        system.add_layer(
            LayerDescriptor::new("low", LayerPriority(10)),
            TraceLayer::new("low", false, trace.clone()),
        );

        system.queue_event(InputEvent::CursorFocus { entered: true });
        system.dispatch_frame();

        assert_eq!(trace.borrow().as_slice(), &["high", "low"]);
        assert_eq!(system.debug_snapshot().last_dispatch_consumed_events, 0);
    }

    #[test]
    fn layer_handle_lifecycle_supports_enable_priority_and_remove() {
        let mut system = InputSystem::new();
        let trace = Rc::new(RefCell::new(Vec::new()));

        let a = system.add_layer(
            LayerDescriptor::new("a", LayerPriority(1)),
            TraceLayer::new("a", false, trace.clone()),
        );
        let b = system.add_layer(
            LayerDescriptor::new("b", LayerPriority(2)),
            TraceLayer::new("b", false, trace.clone()),
        );

        assert!(system.set_layer_enabled(b, false));
        system.queue_event(InputEvent::CursorFocus { entered: true });
        system.dispatch_frame();
        assert_eq!(trace.borrow().as_slice(), &["a"]);

        trace.borrow_mut().clear();
        assert!(system.set_layer_enabled(b, true));
        assert!(system.set_layer_priority(a, LayerPriority(3)));
        system.queue_event(InputEvent::CursorFocus { entered: false });
        system.dispatch_frame();
        assert_eq!(trace.borrow().as_slice(), &["a", "b"]);

        assert!(system.remove_layer(b));
        assert!(!system.remove_layer(b));
    }

    #[test]
    fn snapshot_transients_reset_each_frame_boundary() {
        let mut system = InputSystem::new();
        system.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.queue_mouse_motion((3.0, -2.0));
        system.queue_scroll_lines(1.5);
        system.dispatch_frame();

        assert!(system.snapshot().key_down(KeyCode::Space));
        assert!(system.snapshot().key_just_pressed(KeyCode::Space));
        assert_eq!(system.snapshot().mouse_delta(), (3.0, -2.0));
        assert_eq!(system.snapshot().scroll_delta_lines(), 1.5);

        system.dispatch_frame();

        assert!(system.snapshot().key_down(KeyCode::Space));
        assert!(!system.snapshot().key_just_pressed(KeyCode::Space));
        assert_eq!(system.snapshot().mouse_delta(), (0.0, 0.0));
        assert_eq!(system.snapshot().scroll_delta_lines(), 0.0);
    }

    #[test]
    fn action_map_roundtrip() {
        let mut map = ActionMap::new();
        map.bind_key(ActionId::new("jump"), KeyCode::Space);

        let toml = map.to_toml_string().expect("toml serialize should work");
        let loaded = ActionMap::from_toml_str(&toml).expect("toml parse should work");

        assert_eq!(loaded.bindings().len(), 1);
        assert_eq!(loaded.bindings()[0].action, ActionId::new("jump"));
        assert_eq!(
            loaded.bindings()[0].trigger,
            BindingTrigger::Key(KeyCode::Space)
        );
    }

    #[test]
    fn input_snapshot_exposes_action_values_for_observers() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        system.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        let values = system.snapshot().action_values().collect::<Vec<_>>();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].0.as_str(), "jump");
        assert_eq!(values[0].1, 1.0);
        assert!(system.snapshot().action_just_pressed(values[0].0));
    }

    #[test]
    fn modifier_state_is_used_for_subsequent_window_event_queueing() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind(
            ActionBinding::mouse_button("ui.shift_click", MouseButton::Left).with_modifiers(
                BindingModifiers {
                    shift: true,
                    ..BindingModifiers::default()
                },
            ),
        );
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(10)),
            map.into_layer(),
        );

        system.queue_event(InputEvent::ModifiersChanged {
            modifiers: ModifiersState::SHIFT,
        });
        system.queue_winit_window_event(&WindowEvent::MouseInput {
            device_id: unsafe { DeviceId::dummy() },
            state: ElementState::Pressed,
            button: MouseButton::Left,
        });

        system.dispatch_frame();
        assert!(system
            .snapshot()
            .action_pressed(&ActionId::new("ui.shift_click")));
    }

    #[test]
    fn profile_parser_rejects_invalid_trigger_shapes() {
        let invalid = r#"
version = 1

[[bindings]]
action = "foo"
trigger = { key = "KeyW", mouse_button = "Left" }
"#;
        let err = ActionMap::from_toml_str(invalid).expect_err("must reject invalid trigger");
        assert!(err.contains("exactly one trigger"));
    }

    #[test]
    fn profile_parser_supports_extended_key_codes() {
        let mut map = ActionMap::new();
        map.bind(ActionBinding::key("debug.step", KeyCode::F35));

        let toml = map.to_toml_string().expect("toml serialize should work");
        let loaded = ActionMap::from_toml_str(&toml).expect("toml parse should work");
        assert_eq!(
            loaded.bindings()[0].trigger,
            BindingTrigger::Key(KeyCode::F35)
        );
    }

    #[test]
    fn editor_ui_capture_layer_uses_ui_priority_and_consumes_devices() {
        let (descriptor, mut layer) = editor_ui_capture_layer();
        assert_eq!(descriptor.name, "editor-ui-capture");
        assert_eq!(descriptor.priority, priority_bands::EDITOR_UI_CAPTURE);

        let mut action_state = ActionStateStore::default();
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };

        assert_eq!(
            layer.on_event(
                &InputEvent::Key {
                    code: KeyCode::KeyW,
                    state: ElementState::Pressed,
                    repeat: false,
                    modifiers: ModifiersState::empty(),
                },
                &mut ctx,
            ),
            InputConsume::Consumed
        );
        assert_eq!(
            layer.on_event(&InputEvent::MouseMotion { delta: (1.0, 2.0) }, &mut ctx),
            InputConsume::Consumed
        );
        assert_eq!(
            layer.on_event(&InputEvent::CursorFocus { entered: true }, &mut ctx,),
            InputConsume::Ignored
        );
    }

    // ── Multi-binding tests ──────────────────────────────────────────

    #[test]
    fn releasing_one_binding_keeps_action_active_when_another_contributes() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        // Two keys bound to the same action.
        map.bind_key("move.forward", KeyCode::KeyW);
        map.bind_key("move.forward", KeyCode::ArrowUp);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press both keys.
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.queue_event(InputEvent::Key {
            code: KeyCode::ArrowUp,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        assert!(system
            .snapshot()
            .action_pressed(&ActionId::new("move.forward")));
        assert_eq!(
            system
                .snapshot()
                .action_value(&ActionId::new("move.forward")),
            1.0
        );

        // Release only one key.
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        // Action should still be active because ArrowUp is still held.
        assert!(
            system
                .snapshot()
                .action_pressed(&ActionId::new("move.forward")),
            "action should remain active when one of two bindings is released"
        );
        assert!(
            !system
                .snapshot()
                .action_just_released(&ActionId::new("move.forward")),
            "action should not report just_released while another binding contributes"
        );

        // Release the second key.
        system.queue_event(InputEvent::Key {
            code: KeyCode::ArrowUp,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        // Now action should be released.
        assert!(!system
            .snapshot()
            .action_pressed(&ActionId::new("move.forward")));
        assert!(
            system
                .snapshot()
                .action_just_released(&ActionId::new("move.forward")),
            "action should report just_released when last binding releases"
        );
    }

    #[test]
    fn press_and_release_same_frame_preserves_transient_edges() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press and release in the same frame.
        system.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        assert!(system
            .snapshot()
            .action_just_pressed(&ActionId::new("jump")));
        assert!(system
            .snapshot()
            .action_just_released(&ActionId::new("jump")));

        // Next frame, transients clear.
        system.dispatch_frame();
        assert!(!system
            .snapshot()
            .action_just_pressed(&ActionId::new("jump")));
        assert!(!system
            .snapshot()
            .action_just_released(&ActionId::new("jump")));
    }

    #[test]
    fn separate_action_ids_do_not_interfere_with_each_other() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("move.forward", KeyCode::KeyW);
        map.bind_key("move.backward", KeyCode::KeyS);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        assert!(system
            .snapshot()
            .action_pressed(&ActionId::new("move.forward")));
        assert!(!system
            .snapshot()
            .action_pressed(&ActionId::new("move.backward")));

        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyS,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();

        assert!(system
            .snapshot()
            .action_pressed(&ActionId::new("move.forward")));
        assert!(system
            .snapshot()
            .action_pressed(&ActionId::new("move.backward")));
    }

    #[test]
    fn two_keys_one_action_use_distinct_instances() {
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        map.bind_key("jump", KeyCode::KeyJ);

        let layer = ActionMapLayer::new(map);
        assert_eq!(layer.tracked.len(), 2);
        assert_ne!(layer.tracked[0].instance, BindingInstanceId::new(0));
        assert_ne!(layer.tracked[1].instance, BindingInstanceId::new(0));
        assert_ne!(layer.tracked[0].instance, layer.tracked[1].instance);
    }

    // ── H-A6 modifier release ordering tests ─────────────────────────

    /// Shift+W: release Shift before W; action must clear when W releases.
    #[test]
    fn shift_w_modifier_first_release_clears_on_key_release() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind(
            ActionBinding::key("sprint", KeyCode::KeyW).with_modifiers(BindingModifiers {
                shift: true,
                ..BindingModifiers::default()
            }),
        );
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press Shift, then W
        system.queue_event(InputEvent::ModifiersChanged {
            modifiers: ModifiersState::SHIFT,
        });
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::SHIFT,
        });
        system.dispatch_frame();
        assert!(system.snapshot().action_pressed(&ActionId::new("sprint")));

        // Release Shift (modifier) first — action must stay active
        system.queue_event(InputEvent::ModifiersChanged {
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(
            system.snapshot().action_pressed(&ActionId::new("sprint")),
            "action must stay active when modifier releases before key"
        );

        // Release W — action must clear now
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(
            !system.snapshot().action_pressed(&ActionId::new("sprint")),
            "action must clear when the key itself releases, even without modifier"
        );
    }

    /// Shift+W: release W before Shift; action must clear immediately.
    #[test]
    fn shift_w_key_first_release_clears_action() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind(
            ActionBinding::key("sprint", KeyCode::KeyW).with_modifiers(BindingModifiers {
                shift: true,
                ..BindingModifiers::default()
            }),
        );
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press Shift, then W
        system.queue_event(InputEvent::ModifiersChanged {
            modifiers: ModifiersState::SHIFT,
        });
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::SHIFT,
        });
        system.dispatch_frame();
        assert!(system.snapshot().action_pressed(&ActionId::new("sprint")));

        // Release W first — action must clear
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::SHIFT,
        });
        system.dispatch_frame();
        assert!(!system.snapshot().action_pressed(&ActionId::new("sprint")));
    }

    /// Two modifiers required for activation; release matching uses trigger identity only.
    #[test]
    fn two_modifiers_with_key_release_uses_trigger_not_modifiers() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind(
            ActionBinding::key("action", KeyCode::KeyA).with_modifiers(BindingModifiers {
                shift: true,
                ctrl: true,
                ..BindingModifiers::default()
            }),
        );
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        let both = ModifiersState::SHIFT | ModifiersState::CONTROL;

        system.queue_event(InputEvent::ModifiersChanged { modifiers: both });
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyA,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: both,
        });
        system.dispatch_frame();
        assert!(system.snapshot().action_pressed(&ActionId::new("action")));

        // Release both modifiers, then release key — action must still clear
        system.queue_event(InputEvent::ModifiersChanged {
            modifiers: ModifiersState::empty(),
        });
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyA,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(!system.snapshot().action_pressed(&ActionId::new("action")));
    }

    /// Duplicate bindings (same fingerprint) tracked independently.
    #[test]
    fn duplicate_bindings_activate_independently() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        // Two identical bindings for the same action
        map.bind_key("action", KeyCode::KeyQ);
        map.bind_key("action", KeyCode::KeyQ);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press Q
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyQ,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(system.snapshot().action_pressed(&ActionId::new("action")));

        // Release Q — action must clear (both instances deactivated)
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyQ,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(!system.snapshot().action_pressed(&ActionId::new("action")));
    }

    // ── H-A7 mutable binding reconciliation tests ────────────────────

    /// Append a binding through map_mut(); action must trigger after reconciliation.
    #[test]
    fn append_binding_via_map_mut_activates_after_reconciliation() {
        let mut layer = ActionMapLayer::new(ActionMap::new());
        let mut action_state = ActionStateStore::default();

        let press_w = InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        };

        // No bindings yet — key should do nothing.
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(&press_w, &mut ctx);
        assert!(!action_state.pressed(&ActionId::new("move.forward")));

        // Append binding through map_mut — next event triggers action.
        layer.map_mut().bind_key("move.forward", KeyCode::KeyW);
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(&press_w, &mut ctx);
        assert!(
            action_state.pressed(&ActionId::new("move.forward")),
            "action must activate after binding appended via map_mut"
        );
    }

    /// Remove a binding through map_mut(); action contribution must clear.
    #[test]
    fn remove_binding_via_map_mut_clears_contribution() {
        let mut map = ActionMap::new();
        map.bind_key("action", KeyCode::KeyW);
        let mut layer = map.into_layer();
        let mut action_state = ActionStateStore::default();

        // Press key, verify action active.
        let press_w = InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        };
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(&press_w, &mut ctx);
        assert!(action_state.pressed(&ActionId::new("action")));

        // Remove binding via map_mut — call on_frame_end; action must clear.
        layer.map_mut().unbind_action(&ActionId::new("action"));
        let snapshot = InputSnapshot::default();
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_frame_end(&snapshot, &mut ctx);
        assert!(
            !action_state.pressed(&ActionId::new("action")),
            "removed binding must stop contributing after reconciliation"
        );
    }

    /// Reorder bindings; original contributions survive reorder.
    #[test]
    fn reorder_bindings_via_map_mut_preserves_active_state() {
        let mut map = ActionMap::new();
        map.bind_key("action.a", KeyCode::KeyA);
        map.bind_key("action.b", KeyCode::KeyB);
        let mut layer = map.into_layer();
        let mut action_state = ActionStateStore::default();

        // Press both keys.
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(
            &InputEvent::Key {
                code: KeyCode::KeyA,
                state: ElementState::Pressed,
                repeat: false,
                modifiers: ModifiersState::empty(),
            },
            &mut ctx,
        );
        layer.on_event(
            &InputEvent::Key {
                code: KeyCode::KeyB,
                state: ElementState::Pressed,
                repeat: false,
                modifiers: ModifiersState::empty(),
            },
            &mut ctx,
        );
        assert!(action_state.pressed(&ActionId::new("action.a")));
        assert!(action_state.pressed(&ActionId::new("action.b")));

        // Reorder: swap bindings via map_mut.
        layer.map_mut().bindings.swap(0, 1);
        // on_frame_end reconciliation — both actions must stay active.
        let snapshot = InputSnapshot::default();
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_frame_end(&snapshot, &mut ctx);
        assert!(
            action_state.pressed(&ActionId::new("action.a")),
            "action.a must stay active after reorder"
        );
        assert!(
            action_state.pressed(&ActionId::new("action.b")),
            "action.b must stay active after reorder"
        );
    }

    /// Remove one active binding while another contributes; only removed binding clears.
    #[test]
    fn remove_one_active_contributor_clears_only_that_contributor() {
        let mut map = ActionMap::new();
        map.bind_key("action", KeyCode::KeyW);
        map.bind_key("action", KeyCode::ArrowUp);
        let mut layer = map.into_layer();
        let mut action_state = ActionStateStore::default();

        // Press both keys.
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(
            &InputEvent::Key {
                code: KeyCode::KeyW,
                state: ElementState::Pressed,
                repeat: false,
                modifiers: ModifiersState::empty(),
            },
            &mut ctx,
        );
        layer.on_event(
            &InputEvent::Key {
                code: KeyCode::ArrowUp,
                state: ElementState::Pressed,
                repeat: false,
                modifiers: ModifiersState::empty(),
            },
            &mut ctx,
        );
        assert!(action_state.pressed(&ActionId::new("action")));

        // Remove only the KeyW binding via map_mut.
        layer
            .map_mut()
            .bindings
            .retain(|b| b.trigger != BindingTrigger::Key(KeyCode::KeyW));
        let snapshot = InputSnapshot::default();
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_frame_end(&snapshot, &mut ctx);
        assert!(
            action_state.pressed(&ActionId::new("action")),
            "action must remain active — ArrowUp still contributes"
        );
    }

    /// Mutation with no events reconciles on_frame_end.
    #[test]
    fn mutation_with_no_event_reconciles_on_frame_end() {
        let mut map = ActionMap::new();
        map.bind_key("action", KeyCode::KeyW);
        let mut layer = map.into_layer();
        let mut action_state = ActionStateStore::default();

        // Activate the action.
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_event(
            &InputEvent::Key {
                code: KeyCode::KeyW,
                state: ElementState::Pressed,
                repeat: false,
                modifiers: ModifiersState::empty(),
            },
            &mut ctx,
        );
        assert!(action_state.pressed(&ActionId::new("action")));

        // Mutate binding fingerprint (change scale) without queueing any event.
        layer.map_mut().bindings[0].scale = 0.0;
        let snapshot = InputSnapshot::default();
        let mut ctx = InputContext {
            action_state: &mut action_state,
        };
        layer.on_frame_end(&snapshot, &mut ctx);
        // The old contribution must clear because the binding changed.
        assert!(
            !action_state.pressed(&ActionId::new("action")),
            "binding change (scale=0) must be reflected even with no events"
        );
    }

    /// Repeated mutation cycles do not panic or leak state.
    #[test]
    fn repeated_mutation_cycles_do_not_panic() {
        let mut layer = ActionMapLayer::new(ActionMap::new());
        let mut action_state = ActionStateStore::default();
        let snapshot = InputSnapshot::default();

        let press_w = InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        };

        for _ in 0..10 {
            // Add binding and press key.
            layer.map_mut().bind_key("action", KeyCode::KeyW);
            let mut ctx = InputContext {
                action_state: &mut action_state,
            };
            layer.on_event(&press_w, &mut ctx);
            assert!(action_state.pressed(&ActionId::new("action")));

            // Remove binding and reconcile.
            layer.map_mut().unbind_action(&ActionId::new("action"));
            let mut ctx = InputContext {
                action_state: &mut action_state,
            };
            layer.on_frame_end(&snapshot, &mut ctx);
            assert!(!action_state.pressed(&ActionId::new("action")));
        }
    }

    /// Focus loss (CursorFocus{entered:false}) clears active instances.
    #[test]
    fn focus_loss_clears_active_binding_instances() {
        let mut system = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("action", KeyCode::KeyW);
        system.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        // Press key — action becomes active.
        system.queue_event(InputEvent::Key {
            code: KeyCode::KeyW,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        system.dispatch_frame();
        assert!(system.snapshot().action_pressed(&ActionId::new("action")));

        // Cursor leaves — action should clear.
        system.queue_event(InputEvent::CursorFocus { entered: false });
        system.dispatch_frame();
        assert!(
            !system.snapshot().action_pressed(&ActionId::new("action")),
            "focus loss must clear active binding instances"
        );
    }
}
