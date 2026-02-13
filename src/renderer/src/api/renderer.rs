use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use ash::vk::Extent2D;
use glam::Mat4;
use input::InputManager;
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;
use winit::window::Window;

use crate::data::camera::{Camera, FPSController};
use crate::vulkan::vk_render;
use crate::vulkan::vk_types::VkWindowState;

use super::config::RendererConfig;
use super::errors::{
    map_frame_input_err, map_frame_render_err, map_frame_resize_err, map_init_err, HookError,
    RendererError, RendererInitError,
};
use super::scene::Scene;

pub struct FrameContext {
    frame_number: u32,
    rendered: bool,
}

pub struct Renderer {
    runtime: vk_render::VkRender,
    input_manager: InputManager,
    frame_number: u32,
    last_frame_time: Instant,
    open_frame: Option<u32>,
    startup_scene: Option<Scene>,
}

impl Renderer {
    /// Thread: Main
    /// May Stall: Yes
    pub fn new(config: RendererConfig, window: &Window) -> Result<Self, RendererError> {
        if config.headless {
            return Err(RendererError::Unsupported("headless mode not implemented"));
        }

        let (window_state, input_manager) = create_window_state(window, &config);
        let vk_debug_mode: vk_render::DebugRuntimeMode = config.shader_debug_mode.into();
        let (runtime, scene_world) = vk_render::VkRender::new(
            window_state,
            window,
            config.app_name.as_str(),
            config.validation_layer,
            config.compile_shaders,
            vk_debug_mode,
        )
        .map_err(|err| map_vk_init_err(err, config.compile_shaders))?;

        Ok(Self {
            runtime,
            input_manager,
            frame_number: 0,
            last_frame_time: Instant::now(),
            open_frame: None,
            startup_scene: Some(Scene::from_world(scene_world)),
        })
    }

    /// Thread: Main
    /// May Stall: No
    pub fn update_input(
        &mut self,
        window: &Window,
        event: &winit::event::Event<()>,
    ) -> Result<(), RendererError> {
        self.runtime.core.imgui.handle_event(window, event);

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                self.input_manager.update_mouse_pos(*delta);
            }
            Event::DeviceEvent {
                event:
                    DeviceEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(delta, ..),
                    },
                ..
            } => {
                self.input_manager.update_scroll_state(*delta);
            }
            Event::WindowEvent { window_id, event } if *window_id == window.id() => match event {
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    if let PhysicalKey::Code(key) = key_event.physical_key {
                        self.input_manager
                            .add_keycode(key, key_event.state.is_pressed());
                    }
                }
                WindowEvent::CursorEntered { .. } => self.handle_cursor_focus(window, true)?,
                WindowEvent::CursorLeft { .. } => self.handle_cursor_focus(window, false)?,
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(delta, ..),
                    ..
                } => {
                    self.input_manager.update_scroll_state(*delta);
                }
                _ => {}
            },
            _ => {}
        }

        Ok(())
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "cannot resize while an explicit frame is open",
            ));
        }

        if width == 0 || height == 0 {
            return Ok(());
        }

        self.runtime.core.resize_requested = true;
        let new_extent = Extent2D::default().width(width).height(height);
        catch_unwind(AssertUnwindSafe(|| {
            self.runtime.rebuild_swapchain(new_extent)
        }))
        .map_err(|panic| {
            map_frame_resize_err(format!(
                "swapchain rebuild panicked: {}",
                panic_payload_to_string(panic)
            ))
        })?;
        Ok(())
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene(
        &mut self,
        window: &Window,
        scene: &mut Scene,
    ) -> Result<(), RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "render_scene cannot run while an explicit frame is open",
            ));
        }

        self.prepare_frame(window)?;
        self.render_scene_internal(scene, self.frame_number)?;
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(())
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn begin_frame(&mut self, window: &Window) -> Result<FrameContext, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "begin_frame called while another frame is open",
            ));
        }

        self.prepare_frame(window)?;
        let frame_number = self.frame_number;
        self.open_frame = Some(frame_number);

        Ok(FrameContext {
            frame_number,
            rendered: false,
        })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_in_frame(
        &mut self,
        frame: &mut FrameContext,
        scene: &mut Scene,
    ) -> Result<(), RendererError> {
        if self.open_frame != Some(frame.frame_number) {
            return Err(RendererError::Frame(
                super::errors::RendererFrameError::FrameContext(
                    "frame context is not active".to_string(),
                ),
            ));
        }

        if frame.rendered {
            return Err(RendererError::InvalidState(
                "render_scene_in_frame was already called for this frame",
            ));
        }

        self.render_scene_internal(scene, frame.frame_number)?;
        frame.rendered = true;
        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn end_frame(&mut self, frame: FrameContext) -> Result<(), RendererError> {
        if self.open_frame != Some(frame.frame_number) {
            return Err(RendererError::Frame(
                super::errors::RendererFrameError::FrameContext(
                    "end_frame called with inactive frame context".to_string(),
                ),
            ));
        }

        self.open_frame = None;
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn take_startup_scene(&mut self) -> Option<Scene> {
        self.startup_scene.take()
    }

    /// Thread: Main
    /// May Stall: No
    pub fn register_render_hook(&mut self, _name: &str) -> Result<(), HookError> {
        Err(HookError::Unsupported(
            "custom render hooks are not implemented in this phase".to_string(),
        ))
    }

    /// Thread: Main
    /// May Stall: No
    pub fn resize_requested(&self) -> bool {
        self.runtime.resize_requested()
    }

    fn prepare_frame(&mut self, window: &Window) -> Result<(), RendererError> {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;

        self.input_manager.update();
        self.runtime
            .core
            .window_state
            .controller
            .borrow_mut()
            .update(delta.as_secs_f32());

        if self.runtime.resize_requested() {
            return Ok(());
        }

        self.runtime
            .core
            .imgui
            .context
            .io_mut()
            .update_delta_time(delta);
        self.runtime
            .core
            .imgui
            .platform
            .prepare_frame(self.runtime.core.imgui.context.io_mut(), window)
            .map_err(|err| map_frame_input_err(format!("imgui prepare_frame failed: {err}")))?;

        Ok(())
    }

    fn render_scene_internal(
        &mut self,
        scene: &mut Scene,
        frame_number: u32,
    ) -> Result<(), RendererError> {
        if self.runtime.resize_requested() {
            return Ok(());
        }

        let (camera_view, camera_pos) = {
            let controller = self.runtime.core.window_state.controller.borrow();
            (
                controller.get_camera().get_view_matrix(),
                controller.get_camera().get_position(),
            )
        };

        let fovy = 70_f32.to_radians();
        let aspect_ratio = self.runtime.core.window_state.get_aspect_ratio();
        let far = 0.1;
        let near = 10_000.0;
        let proj = Mat4::perspective_rh(fovy, aspect_ratio, far, near);

        scene.update_camera(camera_view, proj, camera_pos);
        let submission = scene.build_submission();

        catch_unwind(AssertUnwindSafe(|| {
            self.runtime.render(frame_number, &submission);
        }))
        .map_err(|panic| {
            map_frame_render_err(format!(
                "render panicked: {}",
                panic_payload_to_string(panic)
            ))
        })?;

        Ok(())
    }

    fn handle_cursor_focus(
        &mut self,
        window: &Window,
        entered_window: bool,
    ) -> Result<(), RendererError> {
        if entered_window {
            window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .map_err(|err| map_frame_input_err(format!("cursor grab failed: {err}")))?;
            window.set_cursor_visible(false);
        } else {
            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .map_err(|err| map_frame_input_err(format!("cursor release failed: {err}")))?;
            window.set_cursor_visible(true);
        }
        Ok(())
    }

    pub(crate) fn raw_core_mut(&mut self) -> &mut vk_render::VkRenderCore {
        &mut self.runtime.core
    }
}

fn create_window_state(window: &Window, config: &RendererConfig) -> (VkWindowState, InputManager) {
    let inner_size = window.inner_size();

    let width = if inner_size.width > 0 {
        inner_size.width
    } else {
        config.window_width.max(1)
    };
    let height = if inner_size.height > 0 {
        inner_size.height
    } else {
        config.window_height.max(1)
    };

    let monitor_size = window.current_monitor().map(|monitor| monitor.size());
    let max_width = monitor_size
        .map(|size| size.width)
        .unwrap_or(width)
        .max(width);
    let max_height = monitor_size
        .map(|size| size.height)
        .unwrap_or(height)
        .max(height);

    let curr_extent = Extent2D::default().width(width).height(height);
    let max_extent = Extent2D::default().width(max_width).height(max_height);

    let camera = Camera::default();
    let fps_controller = FPSController::new(1, camera, 0.002, 1.0);

    let window_state = VkWindowState::new(curr_extent, max_extent, fps_controller);
    let mut input_manager = InputManager::default();
    input_manager.register_key_listener(window_state.controller.clone());
    input_manager.register_m_pos_listener(window_state.controller.clone());

    (window_state, input_manager)
}

fn map_vk_init_err(err: String, compile_shaders: bool) -> RendererError {
    if compile_shaders && err.contains("Error compiling shaders") {
        return RendererInitError::ShaderCompile(err).into();
    }

    map_init_err(err)
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    let payload = payload.as_ref();
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    "unknown panic payload".to_string()
}
