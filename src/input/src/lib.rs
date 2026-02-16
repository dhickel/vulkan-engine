//! Input system with layered dispatch, priority groups, and optional action mapping.
//!
//! Design goals:
//! - Frame-buffered ingest + dispatch boundary (`dispatch_frame`).
//! - Layered handling with same-priority peer execution.
//! - Event consumption that blocks only lower priorities.
//! - Polling snapshot for gameplay systems.
//! - Handle-based layer lifecycle (add/remove/enable/priority updates).

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

/// Stable typed action identifier.
#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct ActionId(String);

impl ActionId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ActionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ActionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for ActionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LayerHandle(u64);

impl LayerHandle {
    pub fn raw(self) -> u64 {
        self.0
    }
}

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
    value: f32,
    just_pressed: bool,
    just_released: bool,
}

#[derive(Default)]
struct ActionStateStore {
    states: HashMap<ActionId, ActionState>,
}

impl ActionStateStore {
    fn set_action_value(&mut self, action: &ActionId, value: f32) {
        let next_value = value.clamp(0.0, 1.0);
        let state = self.states.entry(action.clone()).or_default();
        let was_pressed = state.value > 0.0;
        let is_pressed = next_value > 0.0;

        if is_pressed && !was_pressed {
            state.just_pressed = true;
        } else if !is_pressed && was_pressed {
            state.just_released = true;
        }

        state.value = next_value;
    }

    fn clear_transients(&mut self) {
        for state in self.states.values_mut() {
            state.just_pressed = false;
            state.just_released = false;
        }
    }

    fn value(&self, action: &ActionId) -> f32 {
        self.states.get(action).map(|state| state.value).unwrap_or(0.0)
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
        self.states.iter().map(|(action, state)| (action, state.value))
    }
}

pub struct InputContext<'a> {
    action_state: &'a mut ActionStateStore,
}

impl<'a> InputContext<'a> {
    pub fn set_action_value(&mut self, action: &ActionId, value: f32) {
        self.action_state.set_action_value(action, value);
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
}

#[derive(Clone, Debug, Default)]
pub struct InputDebugSnapshot {
    pub queued_events: usize,
    pub layer_count: usize,
    pub active_layer_count: usize,
    pub last_dispatch_consumed_events: usize,
}

pub struct InputSystem {
    next_layer_id: u64,
    next_insertion_order: u64,
    layers: HashMap<LayerHandle, LayerEntry>,
    queued_events: Vec<InputEvent>,
    snapshot: InputSnapshot,
    action_state: ActionStateStore,
    debug: InputDebugSnapshot,
}

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
            snapshot: InputSnapshot::default(),
            action_state: ActionStateStore::default(),
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

        handle
    }

    pub fn remove_layer(&mut self, handle: LayerHandle) -> bool {
        self.layers.remove(&handle).is_some()
    }

    pub fn set_layer_enabled(&mut self, handle: LayerHandle, enabled: bool) -> bool {
        if let Some(layer) = self.layers.get_mut(&handle) {
            layer.desc.enabled = enabled;
            return true;
        }

        false
    }

    pub fn set_layer_priority(&mut self, handle: LayerHandle, priority: LayerPriority) -> bool {
        if let Some(layer) = self.layers.get_mut(&handle) {
            layer.desc.priority = priority;
            return true;
        }

        false
    }

    pub fn layer_descriptor(&self, handle: LayerHandle) -> Option<&LayerDescriptor> {
        self.layers.get(&handle).map(|entry| &entry.desc)
    }

    pub fn queue_event(&mut self, event: InputEvent) {
        self.queued_events.push(event);
        self.debug.queued_events = self.queued_events.len();
    }

    pub fn queue_winit_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    self.queue_event(InputEvent::Key {
                        code,
                        state: key_event.state,
                        repeat: key_event.repeat,
                        modifiers: self.snapshot.modifiers,
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
                    modifiers: self.snapshot.modifiers,
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
        self.debug.layer_count = self.layers.len();
        self.debug.active_layer_count = self
            .layers
            .values()
            .filter(|entry| entry.desc.enabled)
            .count();

        let mut consumed_events = 0usize;

        let mut priority_groups: BTreeMap<Reverse<LayerPriority>, Vec<LayerHandle>> =
            BTreeMap::new();

        for (handle, entry) in &self.layers {
            if !entry.desc.enabled {
                continue;
            }

            priority_groups
                .entry(Reverse(entry.desc.priority))
                .or_default()
                .push(*handle);
        }

        for handles in priority_groups.values_mut() {
            handles.sort_by_key(|handle| {
                self.layers
                    .get(handle)
                    .map(|entry| entry.insertion_order)
                    .unwrap_or(0)
            });
        }

        for idx in 0..self.queued_events.len() {
            let event = self.queued_events[idx];
            self.apply_event_to_raw_snapshot(event);
            let mut stop_lower_priorities = false;
            for handles in priority_groups.values() {
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

        let mut frame_end_handles: Vec<(u64, LayerHandle)> = self
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
        frame_end_handles.sort_by_key(|(insertion, _)| *insertion);

        for (_, handle) in frame_end_handles {
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

    fn matches(&self, event: &InputEvent) -> bool {
        match event {
            InputEvent::Key {
                code,
                state,
                modifiers,
                ..
            } => {
                if *state != ElementState::Pressed {
                    return false;
                }

                if self.key != Some(*code) {
                    return false;
                }

                self.matches_modifiers(*modifiers)
            }
            InputEvent::MouseButton {
                button,
                state,
                modifiers,
            } => {
                if *state != ElementState::Pressed {
                    return false;
                }

                if self.mouse_button != Some(*button) {
                    return false;
                }

                self.matches_modifiers(*modifiers)
            }
            _ => false,
        }
    }

    fn matches_release(&self, event: &InputEvent) -> bool {
        match event {
            InputEvent::Key {
                code,
                state,
                modifiers,
                ..
            } => {
                if *state != ElementState::Released {
                    return false;
                }

                if self.key != Some(*code) {
                    return false;
                }

                self.matches_modifiers(*modifiers)
            }
            InputEvent::MouseButton {
                button,
                state,
                modifiers,
            } => {
                if *state != ElementState::Released {
                    return false;
                }

                if self.mouse_button != Some(*button) {
                    return false;
                }

                self.matches_modifiers(*modifiers)
            }
            _ => false,
        }
    }

    fn matches_modifiers(&self, modifiers: ModifiersState) -> bool {
        if self.require_shift && !modifiers.shift_key() {
            return false;
        }

        if self.require_ctrl && !modifiers.control_key() {
            return false;
        }

        if self.require_alt && !modifiers.alt_key() {
            return false;
        }

        if self.require_super && !modifiers.super_key() {
            return false;
        }

        true
    }
}

#[derive(Clone, Debug)]
pub struct ActionBinding {
    pub action: ActionId,
    pub chord: InputChord,
    pub scale: f32,
    pub consume: bool,
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
        self.bind(ActionBinding {
            action: action.into(),
            chord: InputChord::key(key),
            scale: 1.0,
            consume: false,
        });
    }

    pub fn bind_mouse_button(&mut self, action: impl Into<ActionId>, button: MouseButton) {
        self.bind(ActionBinding {
            action: action.into(),
            chord: InputChord::mouse(button),
            scale: 1.0,
            consume: false,
        });
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
        let profile: ActionProfile =
            toml::from_str(content).map_err(|err| format!("toml parse error: {err}"))?;
        let mut bindings = Vec::with_capacity(profile.bindings.len());
        for binding in profile.bindings {
            let key = if let Some(key_name) = binding.key {
                Some(parse_key_code(&key_name).ok_or_else(|| {
                    format!("unsupported key code in profile: {key_name}")
                })?)
            } else {
                None
            };
            let mouse_button = if let Some(button_name) = binding.mouse_button {
                Some(parse_mouse_button(&button_name).ok_or_else(|| {
                    format!("unsupported mouse button in profile: {button_name}")
                })?)
            } else {
                None
            };

            bindings.push(ActionBinding {
                action: ActionId::new(binding.action),
                chord: InputChord {
                    key,
                    mouse_button,
                    require_shift: binding.require_shift,
                    require_ctrl: binding.require_ctrl,
                    require_alt: binding.require_alt,
                    require_super: binding.require_super,
                },
                scale: binding.scale,
                consume: binding.consume,
            });
        }

        Ok(Self {
            bindings,
        })
    }

    pub fn to_toml_string(&self) -> Result<String, String> {
        let bindings = self
            .bindings
            .iter()
            .map(|binding| ProfileBinding {
                action: binding.action.as_str().to_string(),
                key: binding.chord.key.map(key_code_to_string),
                mouse_button: binding.chord.mouse_button.map(mouse_button_to_string),
                require_shift: binding.chord.require_shift,
                require_ctrl: binding.chord.require_ctrl,
                require_alt: binding.chord.require_alt,
                require_super: binding.chord.require_super,
                scale: binding.scale,
                consume: binding.consume,
            })
            .collect();

        toml::to_string_pretty(&ActionProfile {
            version: 1,
            bindings,
        })
        .map_err(|err| err.to_string())
    }

    pub fn load_toml_file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let map = Self::from_toml_str(&content).map_err(|err| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{err}"))
        })?;
        Ok(map)
    }

    pub fn save_toml_file(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let content = self.to_toml_string().map_err(|err| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{err}"))
        })?;
        std::fs::write(path, content)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ActionProfile {
    version: u32,
    #[serde(default)]
    bindings: Vec<ProfileBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProfileBinding {
    action: String,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    mouse_button: Option<String>,
    #[serde(default)]
    require_shift: bool,
    #[serde(default)]
    require_ctrl: bool,
    #[serde(default)]
    require_alt: bool,
    #[serde(default)]
    require_super: bool,
    #[serde(default = "default_binding_scale")]
    scale: f32,
    #[serde(default)]
    consume: bool,
}

fn key_code_to_string(key: KeyCode) -> String {
    format!("{key:?}")
}

fn parse_key_code(value: &str) -> Option<KeyCode> {
    if let Some(rest) = value.strip_prefix("Key") {
        if rest.len() == 1 {
            return match rest.chars().next()? {
                'A' => Some(KeyCode::KeyA),
                'B' => Some(KeyCode::KeyB),
                'C' => Some(KeyCode::KeyC),
                'D' => Some(KeyCode::KeyD),
                'E' => Some(KeyCode::KeyE),
                'F' => Some(KeyCode::KeyF),
                'G' => Some(KeyCode::KeyG),
                'H' => Some(KeyCode::KeyH),
                'I' => Some(KeyCode::KeyI),
                'J' => Some(KeyCode::KeyJ),
                'K' => Some(KeyCode::KeyK),
                'L' => Some(KeyCode::KeyL),
                'M' => Some(KeyCode::KeyM),
                'N' => Some(KeyCode::KeyN),
                'O' => Some(KeyCode::KeyO),
                'P' => Some(KeyCode::KeyP),
                'Q' => Some(KeyCode::KeyQ),
                'R' => Some(KeyCode::KeyR),
                'S' => Some(KeyCode::KeyS),
                'T' => Some(KeyCode::KeyT),
                'U' => Some(KeyCode::KeyU),
                'V' => Some(KeyCode::KeyV),
                'W' => Some(KeyCode::KeyW),
                'X' => Some(KeyCode::KeyX),
                'Y' => Some(KeyCode::KeyY),
                'Z' => Some(KeyCode::KeyZ),
                _ => None,
            };
        }
    }

    if let Some(rest) = value.strip_prefix("Digit") {
        if rest.len() == 1 {
            return match rest.chars().next()? {
                '0' => Some(KeyCode::Digit0),
                '1' => Some(KeyCode::Digit1),
                '2' => Some(KeyCode::Digit2),
                '3' => Some(KeyCode::Digit3),
                '4' => Some(KeyCode::Digit4),
                '5' => Some(KeyCode::Digit5),
                '6' => Some(KeyCode::Digit6),
                '7' => Some(KeyCode::Digit7),
                '8' => Some(KeyCode::Digit8),
                '9' => Some(KeyCode::Digit9),
                _ => None,
            };
        }
    }

    match value {
        "Escape" => Some(KeyCode::Escape),
        "Space" => Some(KeyCode::Space),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft),
        "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft),
        "AltRight" => Some(KeyCode::AltRight),
        "SuperLeft" => Some(KeyCode::SuperLeft),
        "SuperRight" => Some(KeyCode::SuperRight),
        "ArrowUp" => Some(KeyCode::ArrowUp),
        "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft),
        "ArrowRight" => Some(KeyCode::ArrowRight),
        "Tab" => Some(KeyCode::Tab),
        "Enter" => Some(KeyCode::Enter),
        "Backspace" => Some(KeyCode::Backspace),
        _ => None,
    }
}

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

pub struct ActionMapLayer {
    map: ActionMap,
}

impl ActionMapLayer {
    pub fn new(map: ActionMap) -> Self {
        Self { map }
    }

    pub fn map(&self) -> &ActionMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut ActionMap {
        &mut self.map
    }
}

impl InputLayer for ActionMapLayer {
    fn on_event(&mut self, event: &InputEvent, ctx: &mut InputContext<'_>) -> InputConsume {
        let mut consumed = false;

        for binding in &self.map.bindings {
            if binding.chord.matches(event) {
                ctx.set_action_value(&binding.action, binding.scale);
                if binding.consume {
                    consumed = true;
                }
            }
            if binding.chord.matches_release(event) {
                ctx.set_action_value(&binding.action, 0.0);
                if binding.consume {
                    consumed = true;
                }
            }
        }

        if consumed {
            InputConsume::Consumed
        } else {
            InputConsume::Ignored
        }
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
    }
}
