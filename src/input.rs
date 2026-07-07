//! Raw input primitives and app-owned input helpers.

use std::collections::HashMap;

use engine_events::{ActionPhase, EngineEvent, EventBus, EventStage, FrameId, InputActionEvent};
use renderer::{Renderer, RendererError, RendererInputRouting};
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::window::Window;

pub use input::priority_bands;
pub use input::{
    editor_ui_capture_layer, ActionBinding, ActionId, ActionMap, ActionMapLayer, BindingModifiers,
    BindingTrigger, CaptureLayer, FrameInputSnapshot, InputChord, InputConsume, InputContext,
    InputDebugFrame, InputDebugSnapshot, InputDevice, InputEvent, InputLayer, InputRuntime,
    InputSnapshot, InputSystem, LayerDescriptor, LayerHandle, LayerId, LayerPriority, LayerSpec,
};

/// Emits action events from one app-owned input stream after `dispatch_frame`.
///
/// The emitter owns the observed action-value map for exactly one input stream.
/// Apps should call [`InputSystem::dispatch_frame`] once at their frame boundary,
/// then call [`InputActionEventEmitter::emit_from_snapshot`] with the resulting
/// snapshot. Resize-skipped render frames should not skip this app-frame input
/// dispatch; rendering may skip independently after app input has advanced.
#[derive(Debug, Default, Clone)]
pub struct InputActionEventEmitter {
    observed_action_values: HashMap<ActionId, f32>,
}

impl InputActionEventEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear_observed_actions(&mut self) {
        self.observed_action_values.clear();
    }

    pub fn emit_from_snapshot(
        &mut self,
        event_bus: &mut EventBus,
        snapshot: &InputSnapshot,
        frame_index: u64,
    ) -> usize {
        let mut emitted = 0usize;

        for (action, value) in snapshot.action_values() {
            let previous = self
                .observed_action_values
                .get(action)
                .copied()
                .unwrap_or(0.0);
            let just_pressed = snapshot.action_just_pressed(action);
            let just_released = snapshot.action_just_released(action);

            if just_pressed {
                emit_action_event(
                    event_bus,
                    frame_index,
                    action.clone(),
                    ActionPhase::Pressed,
                    value,
                );
                emitted += 1;
            }

            if just_released {
                emit_action_event(
                    event_bus,
                    frame_index,
                    action.clone(),
                    ActionPhase::Released,
                    value,
                );
                emitted += 1;
            } else if !just_pressed && (previous - value).abs() > f32::EPSILON {
                emit_action_event(
                    event_bus,
                    frame_index,
                    action.clone(),
                    ActionPhase::Changed,
                    value,
                );
                emitted += 1;
            }

            self.observed_action_values.insert(action.clone(), value);
        }

        self.observed_action_values
            .retain(|action, _| snapshot.action_pressed(action));
        emitted
    }
}

fn emit_action_event(
    event_bus: &mut EventBus,
    frame_index: u64,
    action: ActionId,
    phase: ActionPhase,
    value: f32,
) {
    event_bus.emit(
        EventStage::Input,
        Some(FrameId(frame_index)),
        EngineEvent::Input(
            InputActionEvent::new(action, phase, value).with_source("input_snapshot"),
        ),
    );
}

/// Routes a platform input event through the renderer and mirrors queued app
/// input into an app-owned [`InputSystem`].
///
/// This helper preserves renderer-owned platform side effects/capture routing
/// while deliberately avoiding [`Renderer::update_input`].
pub fn route_platform_input_to_app(
    renderer: &mut Renderer,
    window: &Window,
    input: &mut InputSystem,
    event: &Event<()>,
) -> Result<RendererInputRouting, RendererError> {
    let routing = renderer.route_platform_input(window, event)?;
    queue_routed_input_event(input, routing, event);
    Ok(routing)
}

/// Queues a renderer-routed winit event into an app-owned [`InputSystem`].
///
/// The renderer routing result owns platform side effects and capture/window
/// filtering. This helper only mirrors the legacy input queueing behavior for
/// events the renderer marked as uncaptured gameplay/app input.
pub fn queue_routed_input_event(
    input: &mut InputSystem,
    routing: RendererInputRouting,
    event: &Event<()>,
) -> bool {
    if !routing.queue_input {
        return false;
    }

    match event {
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            input.queue_mouse_motion(*delta);
            true
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseWheel { delta },
            ..
        } => {
            input.queue_scroll_lines(scroll_delta_to_lines(delta));
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
            input.queue_winit_window_event(window_event);
            true
        }
        _ => false,
    }
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
