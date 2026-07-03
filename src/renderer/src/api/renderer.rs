use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use ash::vk::Extent2D;
use engine_events::{
    ActionPhase, EngineEvent, EventBus, EventRecorder, EventStage, FrameId, InputActionEvent,
    LifecycleEvent,
};
use glam::Mat4;
use input::{
    ActionId, ActionMap, InputSnapshot, InputSystem, LayerDescriptor, LayerHandle, LayerPriority,
};
use log::{error, warn};
use winit::event::{DeviceEvent, ElementState, Event, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::data::camera::{Camera, FPSController, FpsActionBindings};
use crate::data::handles::EnvironmentHandle;
use crate::debug_ui::{
    AppUiCallback, DebugUiFrameContext, DebugViewCallback, DebugViewDescriptor, DebugViewId,
};
use crate::vulkan::vk_render;
use crate::vulkan::vk_types::VkWindowState;

use super::assets::{AssetLoadTracker, AssetManager};
use super::config::{
    AssetPolicyConfig, CaptureTarget, FrameCaptureRequest, FrameCaptureScheduler,
    FrameCaptureSequence, FrameCaptureStatus, RendererConfig,
};
use super::errors::{
    map_frame_input_err, map_frame_render_err, map_frame_resize_err, map_init_err, RendererError,
    RendererInitError,
};
use super::hooks::{invoke_render_hook, RenderHook, RenderHookStage};
use super::scene::Scene;

const DEFAULT_ASSET_PUMP_STEPS: usize = 32;

#[derive(Debug, Copy, Clone)]
pub struct EnvironmentRuntimeStatus {
    pub requested: Option<EnvironmentHandle>,
    pub active: EnvironmentHandle,
    pub transitioning: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FrameRenderOutcome {
    Rendered,
    SkippedResizePending,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FramePrepareOutcome {
    Ready,
    SkippedResizePending,
}

pub struct FrameContext {
    frame_number: u32,
    render_attempted: bool,
}

struct FpsInputPlugin {
    action_layer: LayerHandle,
    controller: FPSController,
}

pub struct Renderer {
    runtime: vk_render::VkRender,
    input_system: InputSystem,
    frame_number: u32,
    last_frame_time: Instant,
    last_frame_delta_seconds: f32,
    last_asset_pump_steps: usize,
    open_frame: Option<u32>,
    startup_scene: Option<Scene>,
    asset_loads: AssetLoadTracker,
    asset_policy: AssetPolicyConfig,
    event_bus: EventBus,
    observed_action_values: HashMap<ActionId, f32>,
    pre_render_hook: Option<RenderHook>,
    post_render_hook: Option<RenderHook>,
    frame_capture_scheduler: FrameCaptureScheduler,
    resize_skip_state_logged: bool,
    camera: Camera,
    fps_plugin: Option<FpsInputPlugin>,
    cursor_in_window: bool,
}

impl Renderer {
    /// Thread: Main
    /// May Stall: Yes
    pub fn new(config: RendererConfig, window: &Window) -> Result<Self, RendererError> {
        let app_name = config.app_name.clone();
        let window_state = create_window_state(window, &config);
        let vk_debug_mode: vk_render::DebugRuntimeMode = config.shader_debug_mode.into();
        let (runtime, scene_world) = vk_render::VkRender::new(
            window_state,
            window,
            config.app_name.as_str(),
            config.validation_layer,
            config.compile_shaders,
            vk_debug_mode,
            config.preload_startup_scene,
            config.visual_tuning,
        )
        .map_err(|err| map_vk_init_err(err, config.compile_shaders))?;

        let asset_policy = config.asset_policy.clone();

        Ok(Self {
            runtime,
            input_system: InputSystem::new(),
            frame_number: 0,
            last_frame_time: Instant::now(),
            last_frame_delta_seconds: 0.0,
            last_asset_pump_steps: 0,
            open_frame: None,
            startup_scene: Some(Scene::from_world(scene_world)),
            asset_loads: AssetLoadTracker::new(),
            asset_policy,
            event_bus: EventBus::new(),
            observed_action_values: HashMap::new(),
            pre_render_hook: None,
            post_render_hook: None,
            frame_capture_scheduler: FrameCaptureScheduler::new(app_name),
            resize_skip_state_logged: false,
            camera: Camera::default(),
            fps_plugin: None,
            cursor_in_window: true,
        })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn new_headless(mut config: RendererConfig) -> Result<Self, RendererError> {
        config.headless = true;
        let app_name = config.app_name.clone();
        let window_state = create_headless_window_state(&config);
        let vk_debug_mode: vk_render::DebugRuntimeMode = config.shader_debug_mode.into();
        let (runtime, scene_world) = vk_render::VkRender::new_headless(
            window_state,
            config.app_name.as_str(),
            config.validation_layer,
            config.compile_shaders,
            vk_debug_mode,
            config.preload_startup_scene,
            config.visual_tuning,
        )
        .map_err(|err| map_vk_init_err(err, config.compile_shaders))?;

        let asset_policy = config.asset_policy.clone();

        Ok(Self {
            runtime,
            input_system: InputSystem::new(),
            frame_number: 0,
            last_frame_time: Instant::now(),
            last_frame_delta_seconds: 0.0,
            last_asset_pump_steps: 0,
            open_frame: None,
            startup_scene: Some(Scene::from_world(scene_world)),
            asset_loads: AssetLoadTracker::new(),
            asset_policy,
            event_bus: EventBus::new(),
            observed_action_values: HashMap::new(),
            pre_render_hook: None,
            post_render_hook: None,
            frame_capture_scheduler: FrameCaptureScheduler::new(app_name),
            resize_skip_state_logged: false,
            camera: Camera::default(),
            fps_plugin: None,
            cursor_in_window: true,
        })
    }

    /// Installs classic FPS controls as an optional plugin layer.
    pub fn install_default_fps_input(&mut self) -> LayerHandle {
        if let Some(plugin) = self.fps_plugin.take() {
            self.input_system.remove_layer(plugin.action_layer);
        }

        let bindings = FpsActionBindings::default();
        let mut map = ActionMap::new();
        map.bind_key(bindings.forward.clone(), KeyCode::KeyW);
        map.bind_key(bindings.backward.clone(), KeyCode::KeyS);
        map.bind_key(bindings.left.clone(), KeyCode::KeyA);
        map.bind_key(bindings.right.clone(), KeyCode::KeyD);
        map.bind_key(bindings.up.clone(), KeyCode::Space);
        map.bind_key(bindings.down.clone(), KeyCode::ShiftLeft);

        let action_layer = self.input_system.add_layer(
            LayerDescriptor::new("fps-actions", LayerPriority(10)),
            map.into_layer(),
        );

        let controller = FPSController::new(0.002, 1.0).with_bindings(bindings);
        self.fps_plugin = Some(FpsInputPlugin {
            action_layer,
            controller,
        });

        action_layer
    }

    pub fn uninstall_default_fps_input(&mut self) {
        if let Some(plugin) = self.fps_plugin.take() {
            self.input_system.remove_layer(plugin.action_layer);
        }
    }

    pub fn input_mut(&mut self) -> &mut InputSystem {
        &mut self.input_system
    }

    pub fn input(&self) -> &InputSystem {
        &self.input_system
    }

    pub fn events(&self) -> &EventBus {
        &self.event_bus
    }

    pub fn events_mut(&mut self) -> &mut EventBus {
        &mut self.event_bus
    }

    pub fn set_event_recorder(&mut self, recorder: Option<EventRecorder>) {
        self.event_bus.set_recorder(recorder);
    }

    /// Thread: Main
    /// May Stall: No
    pub fn update_input(
        &mut self,
        window: &Window,
        event: &winit::event::Event<()>,
    ) -> Result<(), RendererError> {
        if let Some(imgui) = self.runtime.core.imgui.as_mut() {
            imgui.handle_event(window, event);
        }

        let io = self
            .runtime
            .core
            .imgui
            .as_ref()
            .map(|imgui| imgui.context.io());
        let ui_visible = self.runtime.core.debug_ui.is_any_visible();
        let app_ui_active = self.runtime.core.debug_ui.has_app_ui();
        let consume_keyboard =
            app_ui_active || ui_visible || io.is_some_and(|io| io.want_capture_keyboard);
        let consume_mouse =
            app_ui_active || ui_visible || io.is_some_and(|io| io.want_capture_mouse);

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if !consume_mouse {
                    self.input_system.queue_mouse_motion(*delta);
                }
            }
            Event::DeviceEvent {
                event:
                    DeviceEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(delta, ..),
                    },
                ..
            } => {
                if !consume_mouse {
                    self.input_system.queue_scroll_lines(*delta);
                }
            }
            Event::WindowEvent { window_id, event } if *window_id == window.id() => match event {
                WindowEvent::CursorEntered { .. } => {
                    self.handle_cursor_focus(window, true)?;
                    self.input_system.queue_winit_window_event(event);
                }
                WindowEvent::CursorLeft { .. } => {
                    self.handle_cursor_focus(window, false)?;
                    self.input_system.queue_winit_window_event(event);
                }
                WindowEvent::ModifiersChanged(_) => {
                    self.input_system.queue_winit_window_event(event);
                }
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    if !key_event.repeat
                        && key_event.state == ElementState::Pressed
                        && matches!(key_event.physical_key, PhysicalKey::Code(KeyCode::F1))
                    {
                        self.toggle_console_ui();
                        self.apply_cursor_policy(window)?;
                        return Ok(());
                    }

                    if !key_event.repeat
                        && key_event.state == ElementState::Pressed
                        && matches!(key_event.physical_key, PhysicalKey::Code(KeyCode::F2))
                    {
                        self.toggle_debug_overlay_ui();
                        self.apply_cursor_policy(window)?;
                        return Ok(());
                    }

                    if !key_event.repeat
                        && key_event.state == ElementState::Pressed
                        && matches!(key_event.physical_key, PhysicalKey::Code(KeyCode::F12))
                    {
                        if let Err(err) = self.queue_manual_frame_capture(CaptureTarget::Present) {
                            warn!("manual frame capture request rejected: {err}");
                        }
                        return Ok(());
                    }

                    if !consume_keyboard {
                        self.input_system.queue_winit_window_event(event);
                    }
                }
                WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                    if !consume_mouse {
                        self.input_system.queue_winit_window_event(event);
                    }
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
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "render_scene cannot run while an explicit frame is open",
            ));
        }

        self.pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)?;
        self.emit_lifecycle_event(
            EventStage::PreUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameStarted,
        );
        let prepare_outcome = self.prepare_frame(window)?;
        if prepare_outcome == FramePrepareOutcome::SkippedResizePending {
            self.emit_lifecycle_event(
                EventStage::PostUpdate,
                Some(FrameId(self.frame_number as u64)),
                LifecycleEvent::FrameEnded,
            );
            self.frame_number = self.frame_number.wrapping_add(1);
            return Ok(FrameRenderOutcome::SkippedResizePending);
        }

        let outcome = self.render_scene_internal(scene, self.frame_number)?;
        self.emit_lifecycle_event(
            EventStage::PostUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameEnded,
        );
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(outcome)
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_headless(
        &mut self,
        scene: &mut Scene,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "render_scene_headless cannot run while an explicit frame is open",
            ));
        }
        if !self.runtime.is_headless() {
            return Err(RendererError::InvalidState(
                "render_scene_headless requires a headless renderer",
            ));
        }

        self.pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)?;
        self.emit_lifecycle_event(
            EventStage::PreUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameStarted,
        );
        self.prepare_frame_headless();
        let outcome = self.render_scene_internal(scene, self.frame_number)?;
        self.emit_lifecycle_event(
            EventStage::PostUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameEnded,
        );
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(outcome)
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn begin_frame(&mut self, window: &Window) -> Result<FrameContext, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "begin_frame called while another frame is open",
            ));
        }

        self.pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)?;
        self.emit_lifecycle_event(
            EventStage::PreUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameStarted,
        );
        let _ = self.prepare_frame(window)?;
        let frame_number = self.frame_number;
        self.open_frame = Some(frame_number);

        Ok(FrameContext {
            frame_number,
            render_attempted: false,
        })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_in_frame(
        &mut self,
        frame: &mut FrameContext,
        scene: &mut Scene,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.open_frame != Some(frame.frame_number) {
            return Err(RendererError::Frame(
                super::errors::RendererFrameError::FrameContext(
                    "frame context is not active".to_string(),
                ),
            ));
        }

        if frame.render_attempted {
            return Err(RendererError::InvalidState(
                "render_scene_in_frame was already called for this frame",
            ));
        }

        let outcome = self.render_scene_internal(scene, frame.frame_number)?;
        frame.render_attempted = true;
        Ok(outcome)
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
        self.emit_lifecycle_event(
            EventStage::PostUpdate,
            Some(FrameId(frame.frame_number as u64)),
            LifecycleEvent::FrameEnded,
        );
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
    pub fn assets(&mut self) -> AssetManager<'_> {
        AssetManager::new(
            &mut self.runtime.core,
            &mut self.asset_loads,
            &self.asset_policy,
        )
    }

    /// Thread: Main
    /// May Stall: No
    pub fn pump_asset_tasks(&mut self, max_steps: usize) -> Result<usize, RendererError> {
        let pumped = self.asset_loads.pump(&mut self.runtime.core, max_steps);
        self.last_asset_pump_steps = pumped;
        Ok(pumped)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn set_pre_render_hook(&mut self, hook: Option<RenderHook>) {
        self.pre_render_hook = hook;
    }

    /// Thread: Main
    /// May Stall: No
    pub fn set_post_render_hook(&mut self, hook: Option<RenderHook>) {
        self.post_render_hook = hook;
    }

    /// Registers a debug view callback rendered by the in-engine debug UI manager.
    pub fn register_debug_view(
        &mut self,
        descriptor: DebugViewDescriptor,
        callback: DebugViewCallback,
    ) -> Result<DebugViewId, RendererError> {
        let id = descriptor.id.clone();
        if self
            .runtime
            .core
            .debug_ui
            .register_view(descriptor, callback)
        {
            return Ok(id);
        }

        Err(RendererError::InvalidState(
            "debug view id already registered",
        ))
    }

    /// Removes a previously registered debug view.
    pub fn unregister_debug_view(&mut self, id: &DebugViewId) -> bool {
        self.runtime.core.debug_ui.unregister_view(id)
    }

    /// Registers an always-rendered imgui callback for app-owned UI chrome.
    ///
    /// This is intended for first-screen native app surfaces such as an editor shell.
    /// Registering app UI also marks the renderer as app-UI-active for cursor/FPS input
    /// capture so classic camera controls do not receive editor interactions.
    pub fn register_app_ui(
        &mut self,
        id: impl Into<DebugViewId>,
        callback: AppUiCallback,
    ) -> Result<DebugViewId, RendererError> {
        let id = id.into();
        if self
            .runtime
            .core
            .debug_ui
            .register_app_ui(id.clone(), callback)
        {
            return Ok(id);
        }

        Err(RendererError::InvalidState("app ui id already registered"))
    }

    /// Removes a previously registered app UI callback.
    pub fn unregister_app_ui(&mut self, id: &DebugViewId) -> bool {
        self.runtime.core.debug_ui.unregister_app_ui(id)
    }

    /// Returns true when app-owned imgui chrome is registered.
    pub fn has_app_ui(&self) -> bool {
        self.runtime.core.debug_ui.has_app_ui()
    }

    /// Returns true when imgui wants keyboard input for an active widget.
    ///
    /// App shells should use this to suppress their own raw keyboard shortcuts while
    /// text or numeric controls are active. This is narrower than `has_app_ui` so
    /// global editor shortcuts can still work from normal chrome or viewport focus.
    pub fn imgui_wants_keyboard_capture(&self) -> bool {
        self.runtime
            .core
            .imgui
            .as_ref()
            .is_some_and(|imgui| imgui.context.io().want_capture_keyboard)
    }

    /// Enables or disables a debug view by id.
    pub fn set_debug_view_enabled(&mut self, id: &DebugViewId, enabled: bool) -> bool {
        self.runtime.core.debug_ui.set_view_enabled(id, enabled)
    }

    /// Toggles global debug UI visibility.
    pub fn toggle_debug_ui(&mut self) {
        self.runtime.core.debug_ui.toggle_visible();
    }

    /// Sets global debug UI visibility.
    pub fn set_debug_ui_visible(&mut self, visible: bool) {
        self.runtime.core.debug_ui.set_visible(visible);
    }

    /// Returns current global debug UI visibility.
    pub fn is_debug_ui_visible(&self) -> bool {
        self.runtime.core.debug_ui.is_visible()
    }

    /// Toggles in-engine console visibility.
    pub fn toggle_console_ui(&mut self) {
        self.runtime.core.debug_ui.toggle_console_visible();
    }

    /// Sets in-engine console visibility.
    pub fn set_console_ui_visible(&mut self, visible: bool) {
        self.runtime.core.debug_ui.set_console_visible(visible);
    }

    /// Returns current in-engine console visibility.
    pub fn is_console_ui_visible(&self) -> bool {
        self.runtime.core.debug_ui.is_console_visible()
    }

    /// Toggles debug overlay visibility.
    pub fn toggle_debug_overlay_ui(&mut self) {
        self.runtime.core.debug_ui.toggle_debug_visible();
    }

    /// Sets debug overlay visibility.
    pub fn set_debug_overlay_ui_visible(&mut self, visible: bool) {
        self.runtime.core.debug_ui.set_debug_visible(visible);
    }

    /// Returns current debug overlay visibility.
    pub fn is_debug_overlay_ui_visible(&self) -> bool {
        self.runtime.core.debug_ui.is_debug_visible()
    }

    /// Returns true if either debug overlay or console is visible.
    pub fn is_any_debug_ui_visible(&self) -> bool {
        self.runtime.core.debug_ui.is_any_visible()
    }

    /// Configures launch-time debug timing recording options.
    pub fn configure_debug_timing_recording(
        &mut self,
        duration_secs: Option<u64>,
        interval_ms: Option<u64>,
        output_path: Option<String>,
    ) -> Result<(), RendererError> {
        self.runtime
            .core
            .debug_ui
            .configure_timing_recording_options(duration_secs, interval_ms, output_path)
            .map_err(|err| map_frame_input_err(format!("debug timing configuration failed: {err}")))
    }

    /// Starts debug timing recording immediately using configured options.
    pub fn start_debug_timing_recording(&mut self) -> Result<String, RendererError> {
        self.runtime
            .core
            .debug_ui
            .start_timing_recording_now()
            .map_err(|err| map_frame_input_err(format!("debug timing start failed: {err}")))
    }

    /// Queues one frame capture for the next rendered frame.
    pub fn request_frame_capture(
        &mut self,
        request: FrameCaptureRequest,
    ) -> Result<(), RendererError> {
        self.frame_capture_scheduler
            .schedule_single_capture(self.frame_number.wrapping_add(1), request)?;
        Ok(())
    }

    /// Queues one frame capture for an exact renderer frame number.
    pub fn request_frame_capture_at(
        &mut self,
        frame_number: u32,
        request: FrameCaptureRequest,
    ) -> Result<(), RendererError> {
        self.frame_capture_scheduler
            .schedule_single_capture(frame_number, request)?;
        Ok(())
    }

    /// Configures a finite frame-capture sequence.
    pub fn configure_frame_capture_sequence(
        &mut self,
        sequence: FrameCaptureSequence,
    ) -> Result<(), RendererError> {
        self.frame_capture_scheduler.configure_sequence(sequence)?;
        Ok(())
    }

    /// Sets the manual-capture output directory. `None` restores the default location.
    pub fn configure_manual_frame_capture_dir(
        &mut self,
        output_dir: Option<std::path::PathBuf>,
    ) -> Result<(), RendererError> {
        self.frame_capture_scheduler
            .configure_manual_output_dir(output_dir)?;
        Ok(())
    }

    /// Queues a manual capture for the next rendered frame.
    pub fn queue_manual_frame_capture(
        &mut self,
        target: CaptureTarget,
    ) -> Result<(), RendererError> {
        self.frame_capture_scheduler
            .queue_manual_capture(self.frame_number, target)?;
        Ok(())
    }

    pub fn last_frame_capture_status(&self) -> Option<&FrameCaptureStatus> {
        self.frame_capture_scheduler.last_status()
    }

    /// Thread: Main
    /// May Stall: No
    pub fn resize_requested(&self) -> bool {
        self.runtime.resize_requested()
    }

    /// Thread: Any
    /// May Stall: No
    pub fn environment_runtime_status(&self) -> EnvironmentRuntimeStatus {
        let status = self.runtime.environment_runtime_status();
        EnvironmentRuntimeStatus {
            requested: status.requested,
            active: status.active,
            transitioning: status.transitioning,
        }
    }

    fn prepare_frame(&mut self, window: &Window) -> Result<FramePrepareOutcome, RendererError> {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        self.last_frame_delta_seconds = delta.as_secs_f32();

        self.input_system.dispatch_frame();
        emit_input_action_events_from_snapshot(
            &mut self.event_bus,
            &mut self.observed_action_values,
            self.frame_number as u64,
            self.input_system.snapshot(),
        );
        self.dispatch_events_for_stage(EventStage::Input);
        if !self.imgui_capture_active() {
            if let Some(plugin) = self.fps_plugin.as_mut() {
                let snapshot = self.input_system.snapshot();
                plugin.controller.update_from_snapshot(
                    snapshot,
                    delta.as_secs_f32(),
                    &mut self.camera,
                );
            }
        }

        self.apply_cursor_policy(window)?;

        if self.runtime.resize_requested() {
            self.enter_resize_skip_state();
            return Ok(FramePrepareOutcome::SkippedResizePending);
        }

        self.clear_resize_skip_state();
        if let Some(imgui) = self.runtime.core.imgui.as_mut() {
            imgui.context.io_mut().update_delta_time(delta);
            imgui
                .platform
                .prepare_frame(imgui.context.io_mut(), window)
                .map_err(|err| map_frame_input_err(format!("imgui prepare_frame failed: {err}")))?;
        }

        Ok(FramePrepareOutcome::Ready)
    }

    fn prepare_frame_headless(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        self.last_frame_delta_seconds = delta.as_secs_f32();

        self.input_system.dispatch_frame();
        emit_input_action_events_from_snapshot(
            &mut self.event_bus,
            &mut self.observed_action_values,
            self.frame_number as u64,
            self.input_system.snapshot(),
        );
        self.dispatch_events_for_stage(EventStage::Input);
        if let Some(plugin) = self.fps_plugin.as_mut() {
            let snapshot = self.input_system.snapshot();
            plugin
                .controller
                .update_from_snapshot(snapshot, delta.as_secs_f32(), &mut self.camera);
        }
        self.clear_resize_skip_state();
    }

    fn render_scene_internal(
        &mut self,
        scene: &mut Scene,
        frame_number: u32,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.runtime.resize_requested() {
            self.enter_resize_skip_state();
            return Ok(FrameRenderOutcome::SkippedResizePending);
        }

        self.clear_resize_skip_state();
        let camera_view = self.camera.get_view_matrix();
        let camera_pos = self.camera.get_position();

        let fovy = 70_f32.to_radians();
        let aspect_ratio = self.runtime.core.window_state.get_aspect_ratio();
        let far = 0.1;
        let near = 10_000.0;
        let proj = Mat4::perspective_rh(fovy, aspect_ratio, far, near);

        scene.update_camera(camera_view, proj, camera_pos);
        let submission = scene.build_submission();
        let viewport_size = self.viewport_size();
        let frame_index = frame_number as u64;
        let hooks_enabled = self.pre_render_hook.is_some() || self.post_render_hook.is_some();

        let due_captures = self.frame_capture_scheduler.due_captures(frame_number);

        let runtime = &mut self.runtime;
        let pre_hook = &mut self.pre_render_hook;
        let post_hook = &mut self.post_render_hook;

        catch_unwind(AssertUnwindSafe(|| {
            if hooks_enabled {
                runtime.render_with_hooks(
                    frame_number,
                    &submission,
                    due_captures,
                    || {
                        if let Err(err) = invoke_render_hook(
                            pre_hook,
                            RenderHookStage::PreRender,
                            frame_index,
                            viewport_size,
                            None, // TODO: plumb depth texture from rendergraph
                        ) {
                            error!("pre_render hook failed at frame {}: {}", frame_index, err);
                        }
                    },
                    || {
                        if let Err(err) = invoke_render_hook(
                            post_hook,
                            RenderHookStage::PostRender,
                            frame_index,
                            viewport_size,
                            None, // TODO: plumb depth texture from rendergraph
                        ) {
                            error!("post_render hook failed at frame {}: {}", frame_index, err);
                        }
                    },
                );
            } else {
                runtime.render_with_hooks(frame_number, &submission, due_captures, || {}, || {});
            }
        }))
        .map_err(|panic| {
            map_frame_render_err(format!(
                "render panicked: {}",
                panic_payload_to_string(panic)
            ))
        })?;

        self.record_frame_capture_statuses();

        let env_status = self.runtime.environment_runtime_status();
        let fps = if self.last_frame_delta_seconds > 0.0 {
            1.0 / self.last_frame_delta_seconds
        } else {
            0.0
        };
        self.runtime
            .core
            .debug_ui
            .update_frame_context(DebugUiFrameContext {
                frame_index,
                delta_seconds: self.last_frame_delta_seconds,
                fps,
                viewport_size,
                resize_pending: self.runtime.resize_requested(),
                environment_requested: env_status.requested,
                environment_active: env_status.active,
                environment_transitioning: env_status.transitioning,
                draw_item_count: submission.draw_items.len(),
                point_light_count: submission.point_lights.len(),
                draw_skybox: submission.flags.draw_skybox,
                draw_geometry: submission.flags.draw_geometry,
                draw_imgui: submission.flags.draw_imgui,
                asset_tasks_pumped_last: self.last_asset_pump_steps,
                input_debug: self.input_system.debug_snapshot().clone(),
                timings: self.runtime.frame_timing_snapshot(),
            });

        Ok(FrameRenderOutcome::Rendered)
    }

    fn viewport_size(&self) -> (u32, u32) {
        let extent = self.runtime.core.window_state.get_curr_extent();
        (extent.width, extent.height)
    }

    fn record_frame_capture_statuses(&mut self) {
        for status in self.runtime.take_frame_capture_statuses() {
            self.frame_capture_scheduler.record_status(status);
        }
    }

    fn apply_cursor_policy(&mut self, window: &Window) -> Result<(), RendererError> {
        if self.imgui_capture_active() || !self.cursor_in_window {
            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .map_err(|err| map_frame_input_err(format!("cursor release failed: {err}")))?;
            window.set_cursor_visible(true);
        } else {
            window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .map_err(|err| map_frame_input_err(format!("cursor grab failed: {err}")))?;
            window.set_cursor_visible(false);
        }
        Ok(())
    }

    fn imgui_capture_active(&self) -> bool {
        self.runtime.core.debug_ui.has_app_ui()
            || self.runtime.core.debug_ui.is_any_visible()
            || self.runtime.core.imgui.as_ref().is_some_and(|imgui| {
                let io = imgui.context.io();
                io.want_capture_keyboard || io.want_capture_mouse
            })
    }

    fn handle_cursor_focus(
        &mut self,
        window: &Window,
        entered_window: bool,
    ) -> Result<(), RendererError> {
        self.cursor_in_window = entered_window;
        self.apply_cursor_policy(window)
    }

    pub(crate) fn raw_core_mut(&mut self) -> &mut vk_render::VkRenderCore {
        &mut self.runtime.core
    }

    fn enter_resize_skip_state(&mut self) {
        if !self.resize_skip_state_logged {
            warn!("frame rendering skipped while resize_requested is pending");
            self.resize_skip_state_logged = true;
        }
    }

    fn clear_resize_skip_state(&mut self) {
        self.resize_skip_state_logged = false;
    }

    /// Get current camera position in world space
    ///
    /// Thread: Main
    /// May Stall: No
    pub fn camera_position(&self) -> glam::Vec3 {
        self.camera.get_position()
    }

    /// Set camera position in world space
    ///
    /// Thread: Main
    /// May Stall: No
    pub fn set_camera_position(&mut self, position: glam::Vec3) {
        self.camera.set_position(position);
    }

    fn emit_lifecycle_event(
        &mut self,
        stage: EventStage,
        frame: Option<FrameId>,
        event: LifecycleEvent,
    ) {
        self.event_bus
            .emit(stage, frame, EngineEvent::Lifecycle(event));
        self.dispatch_events_for_stage(stage);
    }

    fn dispatch_events_for_stage(&mut self, stage: EventStage) {
        let report = self.event_bus.drain_stage(stage);
        for failure in report.failures {
            warn!(
                "event listener {:?} failed for event {:?}: {}",
                failure.listener, failure.sequence, failure.message
            );
        }
    }
}

fn emit_input_action_events_from_snapshot(
    event_bus: &mut EventBus,
    observed_action_values: &mut HashMap<ActionId, f32>,
    frame_index: u64,
    snapshot: &InputSnapshot,
) {
    for (action, value) in snapshot.action_values() {
        let previous = observed_action_values.get(action).copied().unwrap_or(0.0);
        let phase = if snapshot.action_just_pressed(action) {
            Some(ActionPhase::Pressed)
        } else if snapshot.action_just_released(action) {
            Some(ActionPhase::Released)
        } else if (previous - value).abs() > f32::EPSILON {
            Some(ActionPhase::Changed)
        } else {
            None
        };

        if let Some(phase) = phase {
            event_bus.emit(
                EventStage::Input,
                Some(FrameId(frame_index)),
                EngineEvent::Input(
                    InputActionEvent::new(
                        engine_events::ActionId::new(action.as_str()),
                        phase,
                        value,
                    )
                    .with_source("input_snapshot"),
                ),
            );
        }

        observed_action_values.insert(action.clone(), value);
    }

    observed_action_values.retain(|action, _| snapshot.action_pressed(action));
}

fn create_window_state(window: &Window, config: &RendererConfig) -> VkWindowState {
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

    let (max_width, max_height) = window
        .available_monitors()
        .map(|monitor| monitor.size())
        .fold((width, height), |(acc_w, acc_h), monitor_size| {
            (
                acc_w.max(monitor_size.width),
                acc_h.max(monitor_size.height),
            )
        });

    let curr_extent = Extent2D::default().width(width).height(height);
    let max_extent = Extent2D::default().width(max_width).height(max_height);

    VkWindowState::new(curr_extent, max_extent)
}

fn create_headless_window_state(config: &RendererConfig) -> VkWindowState {
    let width = config.window_width.max(1);
    let height = config.window_height.max(1);
    let extent = Extent2D::default().width(width).height(height);
    VkWindowState::new(extent, extent)
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use engine_events::{ActionPhase, EngineEvent, EventBus, EventStage};
    use input::{ActionMap, InputEvent, InputSystem, LayerDescriptor, LayerPriority};
    use winit::event::ElementState;
    use winit::keyboard::{KeyCode, ModifiersState};

    use super::emit_input_action_events_from_snapshot;

    #[test]
    fn input_action_bridge_emits_after_snapshot_dispatch() {
        let mut input = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        input.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        input.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        input.dispatch_frame();

        let mut bus = EventBus::new();
        let mut observed = HashMap::new();
        emit_input_action_events_from_snapshot(&mut bus, &mut observed, 7, input.snapshot());

        let report = bus.drain_stage(EventStage::Input);
        assert_eq!(report.dispatched, 1);
        let recorded = bus.recorder();
        assert!(recorded.is_none());
    }

    #[test]
    fn input_action_bridge_records_press_and_release_order() {
        let mut input = InputSystem::new();
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        input.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );

        let mut bus = EventBus::new();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_listener = std::sync::Arc::clone(&seen);
        bus.subscribe(move |event| {
            if let EngineEvent::Input(action) = &event.event {
                seen_listener.lock().unwrap().push(action.phase);
            }
            Ok(())
        });
        let mut observed = HashMap::new();

        input.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        input.dispatch_frame();
        emit_input_action_events_from_snapshot(&mut bus, &mut observed, 1, input.snapshot());
        bus.drain_stage(EventStage::Input);

        input.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        input.dispatch_frame();
        emit_input_action_events_from_snapshot(&mut bus, &mut observed, 2, input.snapshot());
        bus.drain_stage(EventStage::Input);

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [ActionPhase::Pressed, ActionPhase::Released]
        );
    }
}
