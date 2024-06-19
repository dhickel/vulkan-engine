#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod renderer;
mod texture;
mod vk_init;
mod vk_util;
mod vk_render;



use crate::vk_init::{QueueType, VkFrameSync, VkPresent};
use ash::vk;
use ash::vk::{
    CommandBuffer, CommandBufferResetFlags, CommandBufferUsageFlags,
    ExtendsPhysicalDeviceFeatures2, PhysicalDeviceFeatures, SubmitInfo2,
};
use glam::*;
use glfw::{Action, ClientApiHint, Context, Key, WindowHint};
use image::GenericImageView;
use input;
use input::{InputManager, KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;

use raw_window_handle::{RawWindowHandle};
use std::time::{Duration, Instant, SystemTime};
use std::{env, ptr, time};
use crate::vk_render::VkRender;


const NANO: f64 = 1000000000.0;

pub struct GameLogic {
    input_manager: InputManager,
}

pub fn run2() {
    let mut glfw = glfw::init(glfw::log_errors).unwrap();

    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
    let (mut window, events) = glfw
        .create_window(
            1920,
            1080,
            "Hello this is window",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create GLFW window.");

    window.set_key_polling(false);
    window.set_raw_mouse_motion(true);
    window.make_current();

    let mut app = VkRender::new(window, (1920, 1080), true).unwrap();

    // window.set_key_callback(|_, key, _, action, _| println!("Input: {:?}", action));

    let logic_ups = 10000.0;
    let frame_ups = 10000.0;

    let time_u = NANO / logic_ups;
    let time_r = if frame_ups > 0.0 {
        NANO / frame_ups
    } else {
        0.0
    };
    let mut delta_update = 0.0;
    let mut delta_fps = 0.0;

    let init_time = SystemTime::now();
    let mut last_time = init_time;
    let mut frames = 0;

    let mut fps_timer = SystemTime::now();
    let mut frame = 0;
    let running = true;

    while !app.window.should_close() {
        let now = SystemTime::now();
        let elapsed = now.duration_since(last_time).unwrap().as_nanos() as f64;
        delta_update += elapsed / time_u;
        delta_fps += elapsed / time_r;


        while delta_update >= 1.0 {
            delta_update -= 1.0;
            glfw.poll_events();
            // update logic here
        }

        if delta_fps >= 1.0 {
            app.render(frame);
            delta_fps -= 1.0;
            frames += 1;
            frame += 1;
        }

        if now.duration_since(fps_timer).unwrap() > Duration::from_secs(1) {
            app.window.set_title(&format!("FPS: {}", frames));
            fps_timer = SystemTime::now();
            frames = 0;
        }

        last_time = now;
    }
}

