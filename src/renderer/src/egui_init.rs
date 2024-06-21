use std::sync::{Arc, Mutex};

use crate::vk_init::LogicalDevice;
use egui_ash_renderer::{DynamicRendering, Options};

pub struct EGUIInstance {
    pub context: egui::Context,
    pub state: egui_winit::State,
    pub renderer: egui_ash_renderer::Renderer,
}

impl EGUIInstance {
    pub fn new(
        window: &winit::window::Window,
        allocator: Arc<Mutex<vk_mem::Allocator>>,
        device: &LogicalDevice,
        dynamic_render: DynamicRendering,
        options: Options,
    ) -> Self {
        let context = egui::Context::default();
        let viewport_id = context.viewport_id();
        // Tutorial clone context so I am to, I assume it's stateless? It can also be accessed via state so idk.
        let state = egui_winit::State::new(context.clone(), viewport_id, &window, None, None);
        let renderer = egui_ash_renderer::Renderer::with_vk_mem_allocator(
            allocator,
            device.device.clone(),
            dynamic_render,
            options,
        )
        .unwrap();

        Self {
            context,
            state,
            renderer,
        }
    }


}
