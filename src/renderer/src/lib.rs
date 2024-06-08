#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod renderer;
mod texture;
mod vulkan_init;

use ash::vk;
use bytemuck;
use glam::*;
use image::GenericImageView;
use input;
use std::collections::HashSet;
use std::env;
use std::fmt::Debug;
use std::process::exit;

use input::{InputManager, KeyboardListener, ListenerType, MousePosListener};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

pub struct Empty {}

impl KeyboardListener for Empty {
    fn listener_type(&self) -> ListenerType {
        ListenerType::GameInput
    }

    fn listener_id(&self) -> u32 {
        1
    }

    fn listener_for(&self, key: KeyCode) -> bool {
        true
    }

    fn broadcast(&mut self, key: KeyCode, pressed: bool, modifiers: &HashSet<Modifiers>) {
        println!("Key: {:?} : {:?}", key, pressed)
    }
}

pub struct GameLogic {
    input_manager: InputManager,
}

pub fn init_window(
    event_loop: &EventLoop<()>,
    title: String,
    size: (i32, i32),
) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(size.0, size.1))
        .build(event_loop)
        .expect("Fatal: Failed to create window")
}

pub fn run() {
    env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&*env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let mut input_manager = InputManager::new();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let window_id = window.id();
    let entry = vulkan_init::init_entry();
    let instance = vulkan_init::init_instance(&entry, &window, "test".to_string(), true).unwrap();

    let devices = vulkan_init::get_physical_devices(
        instance.0,
        vec![vk::QueueFlags::GRAPHICS],
        None,
    ).unwrap();



    event_loop
        .run(move |event, control_flow| {
            input_manager.update();
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window_id => match event {
                    WindowEvent::ActivationTokenDone { .. } => {}
                    WindowEvent::Resized(_) => {}
                    WindowEvent::Moved(_) => {}
                    WindowEvent::CloseRequested => exit(10),
                    WindowEvent::Destroyed => {}
                    WindowEvent::DroppedFile(_) => {}
                    WindowEvent::HoveredFile(_) => {}
                    WindowEvent::HoveredFileCancelled => {}
                    WindowEvent::Focused(_) => {}
                    WindowEvent::KeyboardInput { event, .. } => {
                        input_manager.update();
                        if let PhysicalKey::Code(key) = event.physical_key {
                            input_manager.add_keycode(key) // TODO need to take pressed state into account
                        }
                    }
                    WindowEvent::ModifiersChanged(modd, ..) => {}
                    WindowEvent::Ime(_) => {}
                    WindowEvent::CursorMoved { position, .. } => {
                        input_manager.update_mouse_pos(position.x as f32, position.y as f32)
                        //TODO Maybe track as f64?
                    }
                    WindowEvent::CursorEntered { .. } => {}
                    WindowEvent::CursorLeft { .. } => {}
                    WindowEvent::MouseWheel { delta, .. } => {
                        if let MouseScrollDelta::LineDelta(delta, ..) = delta {
                            input_manager.update_scroll_state(*delta)
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        input_manager.add_mouse_button(*button)
                    }
                    WindowEvent::TouchpadMagnify { .. } => {}
                    WindowEvent::SmartMagnify { .. } => {}
                    WindowEvent::TouchpadRotate { .. } => {}
                    WindowEvent::TouchpadPressure { .. } => {}
                    WindowEvent::AxisMotion { .. } => {}
                    WindowEvent::Touch(_) => {}
                    WindowEvent::ScaleFactorChanged { .. } => {} // Tutorial resizes here but api changed
                    WindowEvent::ThemeChanged(_) => {}
                    WindowEvent::Occluded(_) => {}
                    WindowEvent::RedrawRequested => {}
                },
                _ => {}
            }
        })
        .expect("TODO: panic message");
}
