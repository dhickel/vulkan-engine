#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::cell::RefCell;

use crate::ListenFilter::TypeFilter;
use glam::Vec2;
use std::cmp::PartialEq;
use std::collections::HashSet;

use std::rc::Rc;
use winit;
use winit::event::{Modifiers, MouseButton};
use winit::keyboard::KeyCode;
use winit::window::Window;

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum ListenerType {
    Window,
    Gui,
    GameInput,
}

#[derive(Default)]
pub struct InputMap<K, V>
where
    K: enum_map::Enum + enum_map::EnumArray<V>,
    V: Default,
{
    map: enum_map::EnumMap<K, V>,
}

#[derive(enum_map::Enum, Debug, Copy, Clone, Default)]
pub enum PressState {
    #[default]
    Unpressed,
    Pressed,
}

impl KeyboardListener for Window {
    fn listener_type(&self) -> ListenerType {
        ListenerType::Window
    }

    fn listener_id(&self) -> u32 {
        1
    }

    fn listener_for(&self, key: crate::KeyCode) -> bool {
        key == crate::KeyCode::Escape
    }

    fn broadcast(&mut self, key: crate::KeyCode, pressed: bool, modifiers: &HashSet<Modifiers>) {
        if key == crate::KeyCode::Escape {
            self.set_blur(pressed)
        }
    }
}

impl<K, V> InputMap<K, V>
where
    K: enum_map::Enum + enum_map::EnumArray<V>,
    V: enum_map::Enum + PartialOrd + Default,
{
    pub fn add_binding(&mut self, key: K, value: V) {
        self.map[key] = value;
    }

    pub fn contains(&self, key: K) -> bool {
        self.map[key] != V::default()
    }

    pub fn get(&self, key: K) -> &V {
        &self.map[key]
    }

    pub fn get_opt(&self, key: K) -> Option<&V> {
        let value = &self.map[key];
        if *value == V::default() {
            None
        } else {
            Some(value)
        }
    }
}

pub trait MouseBListener {
    fn listener_type(&self) -> ListenerType;
    fn listener_id(&self) -> u32;
    fn listener_for(&self, button: MouseButton) -> bool;
    fn broadcasts(
        &self,
        button: MouseButton,
        pressed: bool,
        modifiers: &HashSet<winit::event::Modifiers>,
    );
}

pub trait MousePosListener {
    fn listener_type(&self) -> ListenerType;
    fn listener_id(&self) -> u32;
    fn broadcast(&mut self, delta: (f64, f64), modifiers: &HashSet<winit::event::Modifiers>);
}

pub trait MouseScrollListener {
    fn listener_type(&self) -> ListenerType;
    fn listener_id(&self) -> u32;
    fn broadcast(&mut self, delta: f32);
}

pub trait KeyboardListener {
    fn listener_type(&self) -> ListenerType;
    fn listener_id(&self) -> u32;
    fn listener_for(&self, key: KeyCode) -> bool;
    fn broadcast(
        &mut self,
        key: KeyCode,
        pressed: bool,
        modifiers: &HashSet<winit::event::Modifiers>,
    );
}

pub enum ListenFilter {
    TypeFilter(ListenerType),
    IdFilter(u32),
}

#[derive(Default)]
pub struct InputManager {
    mouse_delta: (f64, f64),
    scroll_state: f32,
    m_button_states: HashSet<MouseButton>,
    key_states: Vec<(KeyCode, bool)>,
    modifiers: HashSet<winit::event::Modifiers>,
    m_pos_listeners: Vec<Rc<RefCell<dyn MousePosListener>>>,
    m_scroll_listeners: Vec<Rc<RefCell<dyn MouseScrollListener>>>,
    m_button_listeners: Vec<Rc<RefCell<dyn MouseBListener>>>,
    key_state_listener: Vec<Rc<RefCell<dyn KeyboardListener>>>,
    listen_filter: Option<ListenFilter>,
}

impl InputManager {
    pub fn register_m_pos_listener(&mut self, listener: Rc<RefCell<dyn MousePosListener>>) {
        self.m_pos_listeners.push(listener)
    }

    pub fn register_key_listener(&mut self, listener: Rc<RefCell<dyn KeyboardListener>>) {
        self.key_state_listener.push(listener)
    }

    pub fn update_mouse_pos(&mut self, delta: (f64, f64)) {
        self.mouse_delta = delta;
    }

    pub fn update_scroll_state(&mut self, delta: f32) {
        self.scroll_state = delta;
    }

    pub fn add_mouse_button(&mut self, button: MouseButton) {
        // let m_button = MouseButton::from_winit(button);
        self.m_button_states.insert(button);
    }

    pub fn add_keycode(&mut self, key: KeyCode, pressed: bool) {
        // let k_code = KeyCode::from_winit(key);
        self.key_states.push((key, pressed));
    }

    pub fn update(&mut self) {
        self.broadcast_m_pos();
        self.broadcast_m_scroll();
        self.broadcast_m_buttons();
        self.broadcast_key_states();

        self.scroll_state = 0.0;
        self.key_states.clear();
        self.modifiers.clear();
        self.m_button_states.clear();
        self.mouse_delta = (0_f64, 0_f64)
    }

    fn broadcast_m_pos(&mut self) {
        for listener in self.m_pos_listeners.iter_mut() {
            match self.listen_filter {
                None => listener
                    .borrow_mut()
                    .broadcast(self.mouse_delta, &self.modifiers),
                Some(TypeFilter(typ)) => {
                    if listener.borrow().listener_type() == typ {
                        listener
                            .borrow_mut()
                            .broadcast(self.mouse_delta, &self.modifiers);
                    }
                }
                Some(ListenFilter::IdFilter(id)) => {
                    if listener.borrow().listener_id() == id {
                        listener
                            .borrow_mut()
                            .broadcast(self.mouse_delta, &self.modifiers);
                    }
                }
            }
        }
    }

    fn broadcast_m_scroll(&mut self) {
        for listener in self.m_scroll_listeners.iter_mut() {
            match self.listen_filter {
                None => listener.borrow_mut().broadcast(self.scroll_state),
                Some(TypeFilter(typ)) => {
                    if listener.borrow().listener_type() == typ {
                        listener.borrow_mut().broadcast(self.scroll_state);
                    }
                }
                Some(ListenFilter::IdFilter(id)) => {
                    if listener.borrow().listener_id() == id {
                        listener.borrow_mut().broadcast(self.scroll_state);
                    }
                }
            }
        }
    }

    fn broadcast_m_buttons(&mut self) {
        for listener in self.m_button_listeners.iter_mut() {
            match self.listen_filter {
                None => {}
                Some(TypeFilter(typ)) => {
                    if listener.borrow().listener_type() != typ {
                        continue;
                    }
                }
                Some(ListenFilter::IdFilter(id)) => {
                    if listener.borrow().listener_id() != id {
                        continue;
                    }
                }
            }
            self.m_button_states
                .iter()
                .filter(|btn| listener.borrow().listener_for(**btn))
                .for_each(|btn| {
                    listener
                        .borrow_mut()
                        .broadcasts(*btn, true, &self.modifiers)
                })
        }
    }

    fn broadcast_key_states(&mut self) {
        for listener in self.key_state_listener.iter_mut() {
            match self.listen_filter {
                None => {}
                Some(TypeFilter(typ)) => {
                    if listener.borrow().listener_type() != typ {
                        continue;
                    }
                }
                Some(ListenFilter::IdFilter(id)) => {
                    if listener.borrow().listener_id() != id {
                        continue;
                    }
                }
            }

            for key in &self.key_states {
                if listener.borrow().listener_for(key.0) {
                    listener
                        .borrow_mut()
                        .broadcast(key.0, key.1, &self.modifiers)
                }
            }
        }
    }
}
