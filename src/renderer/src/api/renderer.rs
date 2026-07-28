use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use ash::vk::Extent2D;
use engine_events::{
    ActionPhase, EngineEvent, EventBus, EventRecorder, EventStage, FrameId, InputActionEvent,
    LifecycleEvent,
};
use glam::{Mat4, Vec3};
use input::{
    ActionId, ActionMap, InputDebugSnapshot, InputSnapshot, InputSystem, LayerDescriptor,
    LayerHandle, LayerPriority,
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

use super::frame_extensions::FrameExtensions;
use super::assets::{AssetLoadTracker, AssetManager};
use super::config::{
    AssetPolicyConfig, CaptureTarget, FrameCaptureRequest, FrameCaptureScheduler,
    FrameCaptureSequence, FrameCaptureStatus, RendererConfig,
};
use super::errors::{
    map_frame_input_err, map_frame_render_err, map_init_err, HookReport, RendererError,
    RendererInitError,
};
use super::hooks::{invoke_render_hook, BoxedRenderHook, RenderHookStage};
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
    /// Frame was rendered and presented successfully.
    Rendered,
    /// Frame was skipped because no swapchain image became available within the bounded acquire
    /// budget. This is transient and does not request a swapchain rebuild.
    SkippedAcquireUnavailable,
    /// Frame was skipped because a resize is pending.
    SkippedResizePending,
    /// Frame reached GPU submission, but an out-of-date swapchain prevented presentation.
    SubmittedNotPresented,
    /// Frame was presented, but acquire or presentation reported a suboptimal swapchain.
    PresentedSuboptimal,
}

/// Fence-derived serials used by app-owned stores that retain data referenced by rendered frames.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RetirementSerials {
    pub latest_submitted: u64,
    pub latest_completed: u64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RendererInputSuppression {
    None,
    UiKeyboardCapture,
    UiMouseCapture,
    PlatformShortcut,
    OtherWindow,
    UnsupportedEvent,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RendererInputRouting {
    pub queue_input: bool,
    pub suppression: RendererInputSuppression,
}

impl RendererInputRouting {
    pub const fn queue() -> Self {
        Self {
            queue_input: true,
            suppression: RendererInputSuppression::None,
        }
    }

    pub const fn suppress(suppression: RendererInputSuppression) -> Self {
        Self {
            queue_input: false,
            suppression,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CameraView {
    pub view: Mat4,
    pub projection: Mat4,
    pub position: Vec3,
}

impl CameraView {
    pub fn new(view: Mat4, projection: Mat4, position: Vec3) -> Self {
        Self {
            view,
            projection,
            position,
        }
    }

    pub fn from_matrices(view: Mat4, projection: Mat4, position: Vec3) -> Self {
        Self::new(view, projection, position)
    }

    pub fn perspective(
        view: Mat4,
        position: Vec3,
        fovy_radians: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            view,
            projection: Mat4::perspective_rh(fovy_radians, aspect_ratio, near, far),
            position,
        }
    }

    pub fn from_camera(camera: &Camera, aspect_ratio: f32) -> Self {
        Self::perspective(
            camera.get_view_matrix(),
            camera.get_position(),
            70_f32.to_radians(),
            aspect_ratio,
            0.1,
            10_000.0,
        )
    }
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
    // Worker jobs can retain cache Arcs, so the pool must join before Vulkan backend teardown.
    // Rust drops struct fields in declaration order.
    asset_loads: AssetLoadTracker,
    runtime: vk_render::VkRender,
    input_system: InputSystem,
    frame_number: u32,
    last_frame_time: Instant,
    last_frame_delta_seconds: f32,
    last_asset_pump_steps: usize,
    open_frame: Option<u32>,
    startup_scene: Option<Scene>,
    asset_policy: AssetPolicyConfig,
    event_bus: EventBus,
    observed_action_values: HashMap<ActionId, f32>,
    pre_render_hook: Option<BoxedRenderHook>,
    post_render_hook: Option<BoxedRenderHook>,
    last_hook_report: HookReport,
    frame_capture_scheduler: FrameCaptureScheduler,
    resize_skip_state_logged: bool,
    camera: Camera,
    fps_plugin: Option<FpsInputPlugin>,
    cursor_in_window: bool,
    /// Last grab mode successfully requested from winit. This tracks the persistent request, not
    /// whether the compositor is currently activating the constraint for a focused pointer.
    cursor_grab_requested: bool,
    /// Per-frame extensions set by the app; consumed during submission and cleared.
    frame_extensions: FrameExtensions,
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
            config.startup_model_path.clone(),
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
            last_hook_report: HookReport::default(),
            frame_capture_scheduler: FrameCaptureScheduler::new(app_name),
            resize_skip_state_logged: false,
            camera: Camera::default(),
            fps_plugin: None,
            cursor_in_window: true,
            cursor_grab_requested: false,
            frame_extensions: FrameExtensions::new(),
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
            config.startup_model_path.clone(),
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
            last_hook_report: HookReport::default(),
            frame_capture_scheduler: FrameCaptureScheduler::new(app_name),
            resize_skip_state_logged: false,
            camera: Camera::default(),
            fps_plugin: None,
            cursor_in_window: true,
            cursor_grab_requested: false,
            frame_extensions: FrameExtensions::new(),
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

    /// Drain all pending events for a specific stage.
    ///
    /// Typically called after `begin_frame` (for PreUpdate events) and
    /// at frame boundaries (for PostUpdate events). Failures are logged.
    pub fn drain_events(&mut self, stage: EventStage) {
        self.dispatch_events_for_stage(stage);
    }

    /// Thread: Main
    /// May Stall: No
    pub fn update_input(
        &mut self,
        window: &Window,
        event: &winit::event::Event<()>,
    ) -> Result<(), RendererError> {
        let routing = self.route_platform_input(window, event)?;
        self.queue_renderer_owned_input(routing, event);
        Ok(())
    }

    /// Applies renderer-owned platform/UI/debug/capture side effects and
    /// returns whether uncaptured input should be queued by the caller.
    ///
    /// New app-owned input paths call this during the winit event pump, then
    /// queue only `routing.queue_input` events into their own [`InputSystem`].
    /// The app remains responsible for dispatching that input exactly once at
    /// its own frame boundary, even when the renderer later skips a resize
    /// pending frame.
    pub fn route_platform_input(
        &mut self,
        window: &Window,
        event: &winit::event::Event<()>,
    ) -> Result<RendererInputRouting, RendererError> {
        if let Some(imgui) = self.runtime.core.imgui.as_mut() {
            imgui.handle_event(window, event);
        }

        let ui_visible = self.runtime.core.debug_ui.is_any_visible();
        let app_ui_active = self.runtime.core.debug_ui.has_app_ui();
        let (consume_keyboard, consume_mouse) = ui_capture_policy(app_ui_active, ui_visible);

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                let _ = delta;
                if consume_mouse {
                    return Ok(RendererInputRouting::suppress(
                        RendererInputSuppression::UiMouseCapture,
                    ));
                }
                return Ok(RendererInputRouting::queue());
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseWheel { delta },
                ..
            } => {
                let _ = delta;
                if consume_mouse {
                    return Ok(RendererInputRouting::suppress(
                        RendererInputSuppression::UiMouseCapture,
                    ));
                }
                return Ok(RendererInputRouting::queue());
            }
            Event::WindowEvent { window_id, event } if *window_id == window.id() => match event {
                WindowEvent::CursorEntered { .. } => {
                    self.handle_cursor_focus(window, true)?;
                    return Ok(RendererInputRouting::queue());
                }
                WindowEvent::CursorLeft { .. } => {
                    self.handle_cursor_focus(window, false)?;
                    return Ok(RendererInputRouting::queue());
                }
                WindowEvent::ModifiersChanged(_) => {
                    return Ok(RendererInputRouting::queue());
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
                        return Ok(RendererInputRouting::suppress(
                            RendererInputSuppression::PlatformShortcut,
                        ));
                    }

                    if !key_event.repeat
                        && key_event.state == ElementState::Pressed
                        && matches!(key_event.physical_key, PhysicalKey::Code(KeyCode::F2))
                    {
                        self.toggle_debug_overlay_ui();
                        self.apply_cursor_policy(window)?;
                        return Ok(RendererInputRouting::suppress(
                            RendererInputSuppression::PlatformShortcut,
                        ));
                    }

                    if !key_event.repeat
                        && key_event.state == ElementState::Pressed
                        && matches!(key_event.physical_key, PhysicalKey::Code(KeyCode::F12))
                    {
                        if let Err(err) = self.queue_manual_frame_capture(CaptureTarget::Present) {
                            warn!("manual frame capture request rejected: {err}");
                        }
                        return Ok(RendererInputRouting::suppress(
                            RendererInputSuppression::PlatformShortcut,
                        ));
                    }

                    if consume_keyboard {
                        return Ok(RendererInputRouting::suppress(
                            RendererInputSuppression::UiKeyboardCapture,
                        ));
                    }
                    return Ok(RendererInputRouting::queue());
                }
                WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                    if consume_mouse {
                        return Ok(RendererInputRouting::suppress(
                            RendererInputSuppression::UiMouseCapture,
                        ));
                    }
                    return Ok(RendererInputRouting::queue());
                }
                _ => {}
            },
            Event::WindowEvent { .. } => {
                return Ok(RendererInputRouting::suppress(
                    RendererInputSuppression::OtherWindow,
                ));
            }
            _ => {}
        }

        Ok(RendererInputRouting::suppress(
            RendererInputSuppression::UnsupportedEvent,
        ))
    }

    fn queue_renderer_owned_input(
        &mut self,
        routing: RendererInputRouting,
        event: &winit::event::Event<()>,
    ) -> bool {
        if !routing.queue_input {
            return false;
        }

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                self.input_system.queue_mouse_motion(*delta);
                true
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseWheel { delta },
                ..
            } => {
                self.input_system
                    .queue_scroll_lines(scroll_delta_to_lines(delta));
                true
            }
            Event::WindowEvent {
                event:
                    window_event @ (WindowEvent::CursorEntered { .. }
                    | WindowEvent::CursorLeft { .. }
                    | WindowEvent::ModifiersChanged(_)
                    | WindowEvent::KeyboardInput { .. }
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::MouseWheel { .. }),
                ..
            } => {
                self.input_system.queue_winit_window_event(window_event);
                true
            }
            _ => false,
        }
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "cannot resize while an explicit frame is open".to_string(),
            ));
        }

        let new_extent = Extent2D::default().width(width).height(height);
        if !self.runtime.resize_requested()
            && self.runtime.core.swapchain_owner.installed_extent() == Some(new_extent)
        {
            return Ok(());
        }

        self.runtime.core.swapchain_owner.request_resize(new_extent);
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.runtime
            .rebuild_swapchain(new_extent)
            .map_err(renderer_error_from_backend)
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
                "render_scene cannot run while an explicit frame is open".to_string(),
            ));
        }
        self.execute_frame_lifecycle(scene, |slf| slf.prepare_frame(window))
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_headless(
        &mut self,
        scene: &mut Scene,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "render_scene_headless cannot run while an explicit frame is open".to_string(),
            ));
        }
        if !self.runtime.is_headless() {
            return Err(RendererError::InvalidState(
                "render_scene_headless requires a headless renderer".to_string(),
            ));
        }
        self.execute_frame_lifecycle(scene, |slf| {
            slf.prepare_frame_headless();
            Ok(FramePrepareOutcome::Ready)
        })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_with_view(
        &mut self,
        scene: &mut Scene,
        view: CameraView,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "render_scene_with_view cannot run while an explicit frame is open".to_string(),
            ));
        }

        self.pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)?;
        let frame_number = self.frame_number;
        let outcome = self.render_scene_internal_with_view(
            scene,
            frame_number,
            view,
            InputDebugSnapshot::default(),
        )?;
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(outcome)
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn render_scene_headless_with_view(
        &mut self,
        scene: &mut Scene,
        view: CameraView,
    ) -> Result<FrameRenderOutcome, RendererError> {
        if !self.runtime.is_headless() {
            return Err(RendererError::InvalidState(
                "render_scene_headless_with_view requires a headless renderer".to_string(),
            ));
        }

        self.render_scene_with_view(scene, view)
    }

    /// Shared frame lifecycle for one-shot render paths.
    ///
    /// Handles asset pumping, lifecycle event emission, frame preparation,
    /// resize-skip logic, and the render_scene_internal call. The `prepare`
    /// closure handles window- or headless-specific frame setup.
    fn execute_frame_lifecycle(
        &mut self,
        scene: &mut Scene,
        prepare: impl FnOnce(&mut Self) -> Result<FramePrepareOutcome, RendererError>,
    ) -> Result<FrameRenderOutcome, RendererError> {
        self.pump_asset_tasks(DEFAULT_ASSET_PUMP_STEPS)?;
        self.emit_lifecycle_event(
            EventStage::PreUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameStarted,
        );
        self.dispatch_events_for_stage(EventStage::PreUpdate);

        let prepare_outcome = prepare(self)?;
        if prepare_outcome == FramePrepareOutcome::SkippedResizePending {
            self.emit_lifecycle_event(
                EventStage::PostUpdate,
                Some(FrameId(self.frame_number as u64)),
                LifecycleEvent::FrameEnded,
            );
            self.dispatch_events_for_stage(EventStage::PostUpdate);
            self.frame_number = self.frame_number.wrapping_add(1);
            return Ok(FrameRenderOutcome::SkippedResizePending);
        }

        let outcome = self.render_scene_internal(scene, self.frame_number)?;
        self.emit_lifecycle_event(
            EventStage::PostUpdate,
            Some(FrameId(self.frame_number as u64)),
            LifecycleEvent::FrameEnded,
        );
        self.dispatch_events_for_stage(EventStage::PostUpdate);
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(outcome)
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn begin_frame(&mut self, window: &Window) -> Result<FrameContext, RendererError> {
        if self.open_frame.is_some() {
            return Err(RendererError::InvalidState(
                "begin_frame called while another frame is open".to_string(),
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
                "render_scene_in_frame was already called for this frame".to_string(),
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
        self.dispatch_events_for_stage(EventStage::PostUpdate);
        self.frame_number = self.frame_number.wrapping_add(1);
        Ok(())
    }

    /// Execute rendering within a managed frame lifecycle.
    ///
    /// This is the recommended API for most rendering loops. It calls
    /// `begin_frame`, invokes the closure, and automatically calls `end_frame`.
    ///
    /// If the closure returns an `Err`, the frame is still properly ended.
    pub fn with_frame(
        &mut self,
        window: &Window,
        scene: &mut Scene,
        f: impl FnOnce(&mut FrameContext, &mut Scene) -> Result<(), RendererError>,
    ) -> Result<FrameRenderOutcome, RendererError> {
        let mut frame = self.begin_frame(window)?;
        self.dispatch_events_for_stage(EventStage::PreUpdate);

        let frame_result = match f(&mut frame, scene) {
            Ok(()) => self.render_scene_in_frame(&mut frame, scene),
            Err(err) => Err(err),
        };
        let end_result = self.end_frame(frame);

        match frame_result {
            Ok(outcome) => {
                end_result?;
                Ok(outcome)
            }
            Err(err) => {
                if let Err(end_err) = end_result {
                    warn!("end_frame failed after with_frame error: {end_err}");
                }
                Err(err)
            }
        }
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

    /// Returns the latest successful submit and fence-observed completion serials.
    ///
    /// App-owned generation stores use these values to invalidate handles immediately while
    /// delaying payload destruction and slot reuse until the consuming frame completes.
    pub fn retirement_serials(&self) -> RetirementSerials {
        RetirementSerials {
            latest_submitted: self.runtime.core.latest_submitted_serial,
            latest_completed: self.runtime.core.latest_completed_serial,
        }
    }

    /// Set per-frame extensions consumed during the next render submission.
    ///
    /// The renderer takes ownership of the value and replaces it with a
    /// default empty set after the frame is submitted. This is safe to call
    /// at any time; extensions are applied on the next frame.
    ///
    /// # Design
    ///
    /// Frame extensions are immutable once submitted. Transform overrides
    /// propagate to subtrees but never mutate scene-graph local transforms.
    /// Debug lines are consumed by the debug-line render pass (only when the
    /// `debug-draw` feature is enabled).
    pub fn set_frame_extensions(&mut self, extensions: FrameExtensions) {
        self.frame_extensions = extensions;
    }

    /// Retire a detached BSP mount through the renderer's fence-aware queue.
    ///
    /// Preflight validates every owned handle, computes the common `retire_after`
    /// serial (`max(last_referenced, latest_submitted)`), extracts the arena
    /// closure from the surface cache, and enqueues the closure for deferred GPU
    /// destruction. On success, returns a [`BspRetirementAcknowledgement`].
    ///
    /// On preflight failure, returns a [`BspRetirementRejection`] that preserves
    /// the intact lease and mount state.
    ///
    /// Available only with the `bsp` feature.
    #[cfg(feature = "bsp")]
    pub fn retire_bsp_mount(
        &mut self,
        detached: crate::api::bsp::DetachedBspMount,
    ) -> Result<
        crate::api::bsp::BspRetirementAcknowledgement,
        crate::api::bsp::BspRetirementRejection,
    > {
        let core = &mut self.runtime.core;
        let crate::api::bsp::DetachedBspMount {
            state: mount_state,
            lease,
        } = detached;

        // ── Preflight: compute retire_after ────────────────────────
        let retire_after = crate::data::retirement::FrameSerial::new(
            core.latest_submitted_serial.max(0),
        );

        let mesh_count = lease.mesh_handles.len();
        let texture_count = lease.texture_handles.len();
        let material_count = lease.material_handles.len();

        // Validate every mesh handle.
        match core.data_cache.mesh_cache.lock() {
            Ok(mesh_cache) => {
                for handle in &lease.mesh_handles {
                    if mesh_cache.get_id(*handle).is_err() {
                        return Err(bsp_retire_rejection(
                            format!("stale or invalid mesh handle {:?}", handle),
                            lease,
                            mount_state,
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(bsp_retire_rejection(
                    "mesh_cache lock poisoned".to_string(),
                    lease,
                    mount_state,
                ));
            }
        }

        // Validate every texture handle.
        match core.data_cache.texture_cache.lock() {
            Ok(texture_cache) => {
                for handle in &lease.texture_handles {
                    if texture_cache.get_texture(*handle).is_err() {
                        return Err(bsp_retire_rejection(
                            format!("stale or invalid texture handle {:?}", handle),
                            lease,
                            mount_state,
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(bsp_retire_rejection(
                    "texture_cache lock poisoned".to_string(),
                    lease,
                    mount_state,
                ));
            }
        }

        // Extract the retirement closure from the surface cache.
        let closure = match core.data_cache.bsp_surface_cache.lock() {
            Ok(mut surface_cache) => {
                match surface_cache.extract_retirement_closure(lease.arena_id) {
                    Some(c) => c,
                    None => {
                        return Err(bsp_retire_rejection(
                            format!("arena {} has no active payloads", lease.arena_id),
                            lease,
                            mount_state,
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(bsp_retire_rejection(
                    "bsp_surface_cache lock poisoned".to_string(),
                    lease,
                    mount_state,
                ));
            }
        };

        let arena_id = closure.arena_id;
        let lightmap_atlas_count = if closure.lightmap_atlas.is_some() { 1 } else { 0 };

        // Enqueue the closure for deferred GPU destruction.
        core.bsp_retirement_queue.enqueue(
            crate::data::retirement::RetirementClass::BspArenaRetirement,
            retire_after,
            closure,
        );

        Ok(crate::api::bsp::BspRetirementAcknowledgement {
            arena_id,
            retire_after,
            mesh_count,
            texture_count,
            material_count,
            lightmap_atlas_count,
        })
    }

    /// Thread: Main
    /// May Stall: No
    pub fn pump_asset_tasks(&mut self, max_steps: usize) -> Result<usize, RendererError> {
        let _panic_guard = self
            .runtime
            .backend_operation_guard()
            .map_err(renderer_error_from_backend)?;
        let result = self.asset_loads.pump(&mut self.runtime.core, max_steps);
        let pumped = self
            .runtime
            .complete_backend_operation(result)
            .map_err(renderer_error_from_backend)?;
        self.last_asset_pump_steps = pumped;
        Ok(pumped)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn set_pre_render_hook(&mut self, hook: Option<BoxedRenderHook>) {
        self.pre_render_hook = hook;
    }

    /// Thread: Main
    /// May Stall: No
    pub fn set_post_render_hook(&mut self, hook: Option<BoxedRenderHook>) {
        self.post_render_hook = hook;
    }

    /// Returns structured hook diagnostics captured during the most recent render attempt.
    pub fn last_hook_report(&self) -> &HookReport {
        &self.last_hook_report
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
            "debug view id already registered".to_string(),
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

        Err(RendererError::InvalidState(
            "app ui id already registered".to_string(),
        ))
    }

    /// Removes a previously registered app UI callback.
    pub fn unregister_app_ui(&mut self, id: &DebugViewId) -> bool {
        self.runtime.core.debug_ui.unregister_app_ui(id)
    }

    /// Returns true when app-owned imgui chrome is registered.
    pub fn has_app_ui(&self) -> bool {
        self.runtime.core.debug_ui.has_app_ui()
    }

    /// Immediately reapplies cursor visibility/confinement after app UI registration changes.
    ///
    /// App-owned loops that intercept their UI hotkey before [`Self::route_platform_input`] should
    /// call this after registering or unregistering app UI. This keeps cursor policy renderer-owned
    /// while restoring mouse-look without waiting for another pointer enter/leave event.
    pub fn refresh_cursor_capture(&mut self, window: &Window) -> Result<(), RendererError> {
        self.apply_cursor_policy(window)
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
        let aspect_ratio = self.runtime.core.window_state.get_aspect_ratio();
        let view = CameraView::from_camera(&self.camera, aspect_ratio);
        let input_debug = self.input_system.debug_snapshot().clone();
        self.render_scene_internal_with_view(scene, frame_number, view, input_debug)
    }

    fn render_scene_internal_with_view(
        &mut self,
        scene: &mut Scene,
        frame_number: u32,
        view: CameraView,
        input_debug: InputDebugSnapshot,
    ) -> Result<FrameRenderOutcome, RendererError> {
        let frame_index = frame_number as u64;
        self.last_hook_report = HookReport::new(frame_index);

        if self.runtime.resize_requested() {
            self.enter_resize_skip_state();
            return Ok(FrameRenderOutcome::SkippedResizePending);
        }

        self.clear_resize_skip_state();

        // Phase 07: Thread evidence request to scene before building submission.
        #[cfg(feature = "bsp")]
        {
            if let Some(ref evidence_req) = self.runtime.core.bsp_evidence_request {
                scene.set_bsp_evidence_request(
                    evidence_req.1.corpus_identity.clone(),
                    evidence_req.1.request_identity.clone(),
                    evidence_req.1.visibility,
                    frame_number,
                );
            } else {
                scene.set_bsp_evidence_request(
                    String::new(), String::new(),
                    crate::api::bsp::BspEvidenceVisibility::NormalPvs,
                    frame_number,
                );
            }
        }

        // Extensions are consumed exactly once, before submission construction.
        // Invalid override batches fail before any backend frame work begins.
        let extensions = std::mem::take(&mut self.frame_extensions);
        scene.update_camera(view.view, view.projection, view.position);
        #[allow(unused_mut)]
        let mut submission = scene
            .build_submission_with_transform_overrides(&extensions.transform_overrides)
            .map_err(RendererError::InvalidState)?;

        #[cfg(feature = "debug-draw")]
        {
            submission.debug_lines = extensions.debug_lines;
        }
        debug_assert!(self.frame_extensions.is_empty());

        let viewport_size = self.viewport_size();
        let hooks_enabled = self.pre_render_hook.is_some() || self.post_render_hook.is_some();
        let hook_report = RefCell::new(HookReport::new(frame_index));

        let due_captures = self.frame_capture_scheduler.due_captures(frame_number);

        let runtime = &mut self.runtime;
        let pre_hook = &mut self.pre_render_hook;
        let post_hook = &mut self.post_render_hook;

        let backend_result = if hooks_enabled {
            runtime.render_with_hooks(
                frame_number,
                &submission,
                due_captures,
                || {
                    let (result, entry) = invoke_render_hook(
                        pre_hook,
                        RenderHookStage::PreRender,
                        frame_index,
                        viewport_size,
                        None, // TODO: plumb depth texture from rendergraph
                    );
                    if let Err(err) = result {
                        error!("pre_render hook failed at frame {}: {}", frame_index, err);
                        if let Some(entry) = entry {
                            warn!(
                                "pre_render hook failure entry: frame={} stage={:?} message={}",
                                entry.frame_index, entry.stage, entry.message
                            );
                            hook_report.borrow_mut().push_failure(entry);
                        }
                    }
                },
                || {
                    let (result, entry) = invoke_render_hook(
                        post_hook,
                        RenderHookStage::PostRender,
                        frame_index,
                        viewport_size,
                        None, // TODO: plumb depth texture from rendergraph
                    );
                    if let Err(err) = result {
                        error!("post_render hook failed at frame {}: {}", frame_index, err);
                        if let Some(entry) = entry {
                            warn!(
                                "post_render hook failure entry: frame={} stage={:?} message={}",
                                entry.frame_index, entry.stage, entry.message
                            );
                            hook_report.borrow_mut().push_failure(entry);
                        }
                    }
                },
            )
        } else {
            runtime.render_with_hooks(frame_number, &submission, due_captures, || {}, || {})
        };
        self.last_hook_report = hook_report.into_inner();
        let backend_outcome = backend_result.map_err(renderer_error_from_backend)?;

        // Phase 07: After rendering, check for sealed evidence report from the recording pipeline.
        #[cfg(feature = "bsp")]
        {
            let evidence_collected = submission.bsp_evidence_collector.borrow_mut().take();
            if let Some(collector) = evidence_collected {
                let report = collector.seal();
                let request_key = self.runtime.core.bsp_evidence_request.as_ref()
                    .map(|(key, _)| *key)
                    .unwrap_or(crate::api::bsp::BspEvidenceRequestKey(0));
                self.runtime.core.bsp_evidence_report = Some((request_key, crate::api::bsp::BspEvidenceStatus::Sealed(report)));
                self.runtime.core.bsp_evidence_frame_number = frame_number;
                // Consume the request since evidence was collected.
                self.runtime.core.bsp_evidence_request = None;
            } else if self.runtime.core.bsp_evidence_request.is_some() {
                // Request was pending but no evidence was collected (e.g., no BSP mount active).
                let (key, _) = self.runtime.core.bsp_evidence_request.take().unwrap();
                self.runtime.core.bsp_evidence_report = Some((key, crate::api::bsp::BspEvidenceStatus::RejectedNoMount));
            }
        }

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
                input_debug,
                timings: self.runtime.frame_timing_snapshot(),
            });

        Ok(frame_outcome_from_backend(backend_outcome))
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
        let should_grab = !self.imgui_capture_active();
        if let Some(request_grab) = cursor_grab_transition(
            self.cursor_in_window,
            self.cursor_grab_requested,
            should_grab,
        ) {
            let mode = if request_grab {
                winit::window::CursorGrabMode::Confined
            } else {
                winit::window::CursorGrabMode::None
            };
            match window.set_cursor_grab(mode) {
                Ok(()) => {
                    self.cursor_grab_requested = request_grab;
                }
                Err(err) => {
                    log::warn!(
                        "cursor {} failed: {err}",
                        if request_grab { "grab" } else { "release" }
                    );
                }
            }
        }

        window.set_cursor_visible(!should_grab);
        Ok(())
    }

    fn imgui_capture_active(&self) -> bool {
        let app_ui_active = self.runtime.core.debug_ui.has_app_ui();
        let ui_visible = self.runtime.core.debug_ui.is_any_visible();
        let (keyboard, mouse) = ui_capture_policy(app_ui_active, ui_visible);
        keyboard || mouse
    }

    fn handle_cursor_focus(
        &mut self,
        window: &Window,
        entered_window: bool,
    ) -> Result<(), RendererError> {
        self.cursor_in_window = entered_window;
        self.apply_cursor_policy(window)
    }

    #[cfg_attr(
        not(feature = "advanced-interop"),
        allow(dead_code, reason = "used by the opt-in advanced-interop facade")
    )]
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

    /// Set camera position and orientation from a right-handed look-at target.
    ///
    /// Thread: Main
    /// May Stall: No
    pub fn set_camera_look_at(
        &mut self,
        eye: Vec3,
        target: Vec3,
        up: Vec3,
    ) -> Result<(), RendererError> {
        self.camera
            .look_at(eye, target, up)
            .map_err(|err| RendererError::InvalidState(err.message().to_string()))
    }

    /// Prepare a BSP mount from extracted BSP data, uploading GPU resources.
    ///
    /// Available only with the `bsp` feature. Runs synchronously on the calling
    /// thread. Returns a mount ready for [`Scene::set_bsp_mount`].
    #[cfg(feature = "bsp")]
    pub fn prepare_bsp_mount(
        &mut self,
        extracted: &bsp::extract::ExtractedBsp,
    ) -> Result<crate::api::bsp::PreparedBspMount, RendererError> {
        self.assets()
            .prepare_bsp_mount(extracted)
            .map_err(|e| RendererError::InvalidState(format!("BSP upload failed: {e}")))
    }

    /// Request a single bounded post-command evidence report for BSP static-world draws.
    ///
    /// Available only with the `bsp` feature. Returns an opaque request key that must be
    /// passed to [`Self::take_bsp_frame_evidence`] to retrieve the sealed report.
    ///
    /// Only one pending request is allowed; a second request is rejected.
    /// The report is populated during the next frame that renders a BSP mount.
    #[cfg(feature = "bsp")]
    pub fn request_bsp_frame_evidence(
        &mut self,
        corpus_identity: String,
        request_identity: String,
        visibility: crate::api::bsp::BspEvidenceVisibility,
    ) -> Result<crate::api::bsp::BspEvidenceRequestKey, RendererError> {
        let core = &mut self.runtime.core;
        if core.bsp_evidence_request.is_some() {
            return Err(RendererError::InvalidState(
                "a BSP evidence request is already pending".to_string(),
            ));
        }
        let key = crate::api::bsp::BspEvidenceRequestKey(core.bsp_evidence_next_key);
        core.bsp_evidence_next_key = core.bsp_evidence_next_key.wrapping_add(1);
        let request = crate::api::bsp::BspEvidenceRequest {
            corpus_identity,
            request_identity,
            visibility,
            key,
        };
        core.bsp_evidence_request = Some((key, request));
        core.bsp_evidence_report = None;
        Ok(key)
    }

    /// Retrieve the sealed evidence report for a previously submitted request.
    ///
    /// Available only with the `bsp` feature. The request key must match the one returned
    /// by [`Self::request_bsp_frame_evidence`]. Each request can be taken only once;
    /// subsequent calls return [`BspEvidenceStatus::MissingReport`].
    #[cfg(feature = "bsp")]
    pub fn take_bsp_frame_evidence(
        &mut self,
        key: crate::api::bsp::BspEvidenceRequestKey,
    ) -> crate::api::bsp::BspEvidenceStatus {
        let core = &mut self.runtime.core;
        match core.bsp_evidence_report.take() {
            Some((report_key, status)) if report_key == key => status,
            Some((report_key, status)) => {
                // Wrong key — put it back
                core.bsp_evidence_report = Some((report_key, status));
                crate::api::bsp::BspEvidenceStatus::MissingReport
            }
            None => crate::api::bsp::BspEvidenceStatus::MissingReport,
        }
    }

    /// Emit a lifecycle event into the bus. Does NOT drain — the caller must
    /// explicitly drain at the correct boundary via `drain_events` or
    /// `dispatch_events_for_stage`.
    fn emit_lifecycle_event(
        &mut self,
        stage: EventStage,
        frame: Option<FrameId>,
        event: LifecycleEvent,
    ) {
        self.event_bus
            .emit(stage, frame, EngineEvent::Lifecycle(event));
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

#[cfg(feature = "bsp")]
fn bsp_retire_rejection(
    reason: String,
    lease: crate::api::bsp::BspResourceLease,
    state: crate::scene::bsp_visibility::BspMountState,
) -> crate::api::bsp::BspRetirementRejection {
    crate::api::bsp::BspRetirementRejection {
        reason,
        lease,
        state,
    }
}

fn build_submission_with_camera_view(
    scene: &mut Scene,
    view: CameraView,
) -> crate::scene::render_submission::RenderSubmission {
    scene.update_camera(view.view, view.projection, view.position);
    scene.build_submission()
}

/// Returns the next persistent grab request, if one can safely be sent to winit.
///
/// Wayland removes the pointer from winit's surface pointer list before delivering
/// `CursorLeft`. Releasing at that point can update winit's requested mode without destroying the
/// existing protocol object. Re-acquiring on `CursorEntered` then creates a second constraint for
/// the same surface, which is a fatal protocol error. Keep the persistent request across pointer
/// leave/enter and defer genuine policy changes until the pointer is in the window.
fn cursor_grab_transition(
    cursor_in_window: bool,
    grab_requested: bool,
    should_grab: bool,
) -> Option<bool> {
    (cursor_in_window && grab_requested != should_grab).then_some(should_grab)
}

fn ui_capture_policy(app_ui_active: bool, debug_ui_visible: bool) -> (bool, bool) {
    let ui_surface_active = app_ui_active || debug_ui_visible;
    (ui_surface_active, ui_surface_active)
}

fn scroll_delta_to_lines(delta: &MouseScrollDelta) -> f32 {
    match delta {
        MouseScrollDelta::LineDelta(x, y) => {
            if y.abs() > x.abs() {
                *y
            } else {
                *x
            }
        }
        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 120.0,
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
        let just_pressed = snapshot.action_just_pressed(action);
        let just_released = snapshot.action_just_released(action);

        if just_pressed {
            event_bus.emit(
                EventStage::Input,
                Some(FrameId(frame_index)),
                EngineEvent::Input(
                    InputActionEvent::new(action.clone(), ActionPhase::Pressed, value)
                        .with_source("input_snapshot"),
                ),
            );
        }

        if just_released {
            event_bus.emit(
                EventStage::Input,
                Some(FrameId(frame_index)),
                EngineEvent::Input(
                    InputActionEvent::new(action.clone(), ActionPhase::Released, value)
                        .with_source("input_snapshot"),
                ),
            );
        } else if !just_pressed && (previous - value).abs() > f32::EPSILON {
            event_bus.emit(
                EventStage::Input,
                Some(FrameId(frame_index)),
                EngineEvent::Input(
                    InputActionEvent::new(action.clone(), ActionPhase::Changed, value)
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

fn renderer_error_from_backend(err: vk_render::VkRenderError) -> RendererError {
    match err {
        vk_render::VkRenderError::DeviceLost(message) => {
            error!("{message}");
            RendererError::DeviceLost
        }
        vk_render::VkRenderError::Backend(message) => map_frame_render_err(message),
        vk_render::VkRenderError::RetryableResize(message) => {
            super::errors::RendererFrameError::Resize(message).into()
        }
        vk_render::VkRenderError::BackendPoisoned(reason) => RendererError::BackendPoisoned(reason),
    }
}

fn frame_outcome_from_backend(outcome: vk_render::VkFrameRenderOutcome) -> FrameRenderOutcome {
    match outcome {
        vk_render::VkFrameRenderOutcome::Rendered => FrameRenderOutcome::Rendered,
        vk_render::VkFrameRenderOutcome::SkippedAcquireUnavailable => {
            FrameRenderOutcome::SkippedAcquireUnavailable
        }
        vk_render::VkFrameRenderOutcome::SkippedResizePending => {
            FrameRenderOutcome::SkippedResizePending
        }
        vk_render::VkFrameRenderOutcome::SubmittedNotPresented => {
            FrameRenderOutcome::SubmittedNotPresented
        }
        vk_render::VkFrameRenderOutcome::PresentedSuboptimal => {
            FrameRenderOutcome::PresentedSuboptimal
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use engine_events::{ActionPhase, EngineEvent, EventBus, EventStage};
    use glam::{Mat4, Vec3};
    use input::{ActionMap, InputEvent, InputSystem, LayerDescriptor, LayerPriority};
    use winit::event::ElementState;
    use winit::keyboard::{KeyCode, ModifiersState};

    use crate::api::Scene;

    use super::{
        build_submission_with_camera_view, cursor_grab_transition,
        emit_input_action_events_from_snapshot, frame_outcome_from_backend,
        renderer_error_from_backend, ui_capture_policy, CameraView, FrameRenderOutcome,
    };

    #[test]
    fn cursor_grab_policy_is_edge_triggered_and_persists_across_pointer_leave() {
        let mut requested = false;

        let acquire = cursor_grab_transition(true, requested, true);
        assert_eq!(acquire, Some(true));
        requested = acquire.unwrap();

        assert_eq!(cursor_grab_transition(true, requested, true), None);
        assert_eq!(cursor_grab_transition(false, requested, false), None);
        assert_eq!(cursor_grab_transition(false, requested, true), None);
        assert_eq!(cursor_grab_transition(true, requested, true), None);
    }

    #[test]
    fn cursor_grab_policy_defers_real_changes_until_pointer_is_present() {
        assert_eq!(cursor_grab_transition(false, true, false), None);
        assert_eq!(cursor_grab_transition(true, true, false), Some(false));
        assert_eq!(cursor_grab_transition(false, false, true), None);
        assert_eq!(cursor_grab_transition(true, false, true), Some(true));
    }

    #[test]
    fn stale_imgui_capture_does_not_block_input_after_all_ui_is_hidden() {
        assert_eq!(ui_capture_policy(false, false), (false, false));
        assert_eq!(ui_capture_policy(true, false), (true, true));
        assert_eq!(ui_capture_policy(false, true), (true, true));
    }

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

    #[test]
    fn input_action_bridge_emits_same_frame_press_and_release() {
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
        input.queue_event(InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
        input.dispatch_frame();
        emit_input_action_events_from_snapshot(&mut bus, &mut observed, 1, input.snapshot());
        bus.drain_stage(EventStage::Input);

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            [ActionPhase::Pressed, ActionPhase::Released]
        );
    }

    #[test]
    fn camera_view_reaches_scene_submission() {
        let mut scene = Scene::new();
        let view = Mat4::look_at_rh(Vec3::new(2.0, 3.0, 4.0), Vec3::ZERO, Vec3::Y);
        let projection = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 250.0);
        let position = Vec3::new(2.0, 3.0, 4.0);
        let camera_view = CameraView::from_matrices(view, projection, position);

        let submission = build_submission_with_camera_view(&mut scene, camera_view);

        assert_eq!(submission.camera.view, view);
        assert_eq!(submission.camera.projection, projection);
        assert_eq!(submission.camera.cam_pos, position);
        assert_eq!(scene.camera_view_projection(), (view, projection));
    }

    #[test]
    fn backend_terminal_errors_map_to_public_variants() {
        let device_lost =
            renderer_error_from_backend(crate::vulkan::vk_render::VkRenderError::DeviceLost(
                "Vulkan device lost during fence wait".to_string(),
            ));
        assert!(matches!(device_lost, crate::api::RendererError::DeviceLost));

        let retryable_resize =
            renderer_error_from_backend(crate::vulkan::vk_render::VkRenderError::RetryableResize(
                "surface capabilities temporarily unavailable".to_string(),
            ));
        assert!(matches!(
            retryable_resize,
            crate::api::RendererError::Frame(
                crate::api::RendererFrameError::Resize(message)
            ) if message == "surface capabilities temporarily unavailable"
        ));

        let poisoned =
            renderer_error_from_backend(crate::vulkan::vk_render::VkRenderError::BackendPoisoned(
                "prior terminal queue failure".to_string(),
            ));
        assert!(matches!(
            poisoned,
            crate::api::RendererError::BackendPoisoned(reason)
                if reason == "prior terminal queue failure"
        ));
    }

    #[test]
    fn backend_frame_outcomes_map_without_collapsing() {
        use crate::vulkan::vk_render::VkFrameRenderOutcome as BackendOutcome;

        assert_eq!(
            frame_outcome_from_backend(BackendOutcome::SubmittedNotPresented),
            FrameRenderOutcome::SubmittedNotPresented
        );
        assert_eq!(
            frame_outcome_from_backend(BackendOutcome::PresentedSuboptimal),
            FrameRenderOutcome::PresentedSuboptimal
        );
        assert_eq!(
            frame_outcome_from_backend(BackendOutcome::SkippedAcquireUnavailable),
            FrameRenderOutcome::SkippedAcquireUnavailable
        );
        assert_eq!(
            frame_outcome_from_backend(BackendOutcome::SkippedResizePending),
            FrameRenderOutcome::SkippedResizePending
        );
    }
}
