use std::sync::{Arc, Mutex};

use engine::events::{EngineEvent, EventBus, EventStage};
use engine::input::{
    queue_routed_input_event, ActionId, ActionMap, InputActionEventEmitter, InputConsume,
    InputContext, InputEvent, InputLayer, InputSystem, LayerDescriptor, LayerPriority,
};
use engine_events::ActionPhase;
use renderer::RendererInputRouting;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, ElementState, Event, MouseScrollDelta};
use winit::keyboard::{KeyCode, ModifiersState};

fn add_jump_action(input: &mut InputSystem) {
    let mut map = ActionMap::new();
    map.bind_key("jump", KeyCode::Space);
    input.add_layer(
        LayerDescriptor::new("actions", LayerPriority(0)),
        map.into_layer(),
    );
}

fn queue_key(input: &mut InputSystem, code: KeyCode, state: ElementState) {
    input.queue_event(InputEvent::Key {
        code,
        state,
        repeat: false,
        modifiers: ModifiersState::empty(),
    });
}

fn collect_input_events(bus: &mut EventBus) -> Vec<(ActionId, ActionPhase, f32, Option<u64>)> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_listener = Arc::clone(&seen);
    bus.subscribe(move |event| {
        if let EngineEvent::Input(action) = &event.event {
            seen_listener.lock().unwrap().push((
                action.action.clone(),
                action.phase,
                action.value,
                event.frame.map(|frame| frame.0),
            ));
        }
        Ok(())
    });
    bus.drain_stage(EventStage::Input);
    let events = seen.lock().unwrap().clone();
    events
}

#[test]
fn action_emitter_emits_press_once() {
    let mut input = InputSystem::new();
    add_jump_action(&mut input);
    let mut bus = EventBus::new();
    let mut emitter = InputActionEventEmitter::new();

    queue_key(&mut input, KeyCode::Space, ElementState::Pressed);
    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 3), 1);

    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 4), 0);

    assert_eq!(
        collect_input_events(&mut bus),
        vec![(ActionId::new("jump"), ActionPhase::Pressed, 1.0, Some(3))]
    );
}

#[test]
fn action_emitter_emits_release_once() {
    let mut input = InputSystem::new();
    add_jump_action(&mut input);
    let mut bus = EventBus::new();
    let mut emitter = InputActionEventEmitter::new();

    queue_key(&mut input, KeyCode::Space, ElementState::Pressed);
    input.dispatch_frame();
    emitter.emit_from_snapshot(&mut bus, input.snapshot(), 1);
    let _ = bus.drain_stage(EventStage::Input);

    queue_key(&mut input, KeyCode::Space, ElementState::Released);
    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 2), 1);
    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 3), 0);

    assert_eq!(
        collect_input_events(&mut bus),
        vec![(ActionId::new("jump"), ActionPhase::Released, 0.0, Some(2))]
    );
}

struct AnalogActionLayer;

impl InputLayer for AnalogActionLayer {
    fn on_event(&mut self, event: &InputEvent, ctx: &mut InputContext<'_>) -> InputConsume {
        let InputEvent::Key {
            code,
            state: ElementState::Pressed,
            ..
        } = event
        else {
            return InputConsume::Ignored;
        };

        let value = match code {
            KeyCode::KeyA => 0.5,
            KeyCode::KeyB => 1.0,
            _ => return InputConsume::Ignored,
        };
        ctx.set_action_value(&ActionId::new("throttle"), value);
        InputConsume::Ignored
    }
}

#[test]
fn action_emitter_emits_changed_value_once() {
    let mut input = InputSystem::new();
    input.add_layer(
        LayerDescriptor::new("analog", LayerPriority(0)),
        AnalogActionLayer,
    );
    let mut bus = EventBus::new();
    let mut emitter = InputActionEventEmitter::new();

    queue_key(&mut input, KeyCode::KeyA, ElementState::Pressed);
    input.dispatch_frame();
    emitter.emit_from_snapshot(&mut bus, input.snapshot(), 1);
    let _ = bus.drain_stage(EventStage::Input);

    queue_key(&mut input, KeyCode::KeyB, ElementState::Pressed);
    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 2), 1);
    input.dispatch_frame();
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 3), 0);

    assert_eq!(
        collect_input_events(&mut bus),
        vec![(
            ActionId::new("throttle"),
            ActionPhase::Changed,
            1.0,
            Some(2)
        )]
    );
}

#[test]
fn action_emitter_preserves_same_frame_press_release_transients() {
    let mut input = InputSystem::new();
    add_jump_action(&mut input);
    let mut bus = EventBus::new();
    let mut emitter = InputActionEventEmitter::new();

    queue_key(&mut input, KeyCode::Space, ElementState::Pressed);
    queue_key(&mut input, KeyCode::Space, ElementState::Released);
    input.dispatch_frame();

    assert!(input.snapshot().action_just_pressed(&ActionId::new("jump")));
    assert!(input
        .snapshot()
        .action_just_released(&ActionId::new("jump")));
    assert_eq!(emitter.emit_from_snapshot(&mut bus, input.snapshot(), 9), 2);

    input.dispatch_frame();
    assert!(!input.snapshot().action_just_pressed(&ActionId::new("jump")));
    assert!(!input
        .snapshot()
        .action_just_released(&ActionId::new("jump")));
    assert_eq!(
        emitter.emit_from_snapshot(&mut bus, input.snapshot(), 10),
        0
    );

    assert_eq!(
        collect_input_events(&mut bus),
        vec![
            (ActionId::new("jump"), ActionPhase::Pressed, 0.0, Some(9)),
            (ActionId::new("jump"), ActionPhase::Released, 0.0, Some(9)),
        ]
    );
}

#[test]
fn queue_routed_input_event_respects_renderer_routing_decision() {
    let mut input = InputSystem::new();

    let event = Event::DeviceEvent {
        // SAFETY: Dummy device ids are only used to construct an inert unit-test event.
        device_id: unsafe { winit::event::DeviceId::dummy() },
        event: DeviceEvent::MouseMotion { delta: (5.0, -2.0) },
    };

    assert!(!queue_routed_input_event(
        &mut input,
        RendererInputRouting::suppress(renderer::RendererInputSuppression::UiMouseCapture),
        &event,
    ));
    input.dispatch_frame();
    assert_eq!(input.snapshot().mouse_delta(), (0.0, 0.0));

    assert!(queue_routed_input_event(
        &mut input,
        RendererInputRouting::queue(),
        &event,
    ));
    input.dispatch_frame();
    assert_eq!(input.snapshot().mouse_delta(), (5.0, -2.0));
}

#[test]
fn queue_routed_input_event_preserves_device_wheel_queueing() {
    let mut input = InputSystem::new();
    let event = Event::DeviceEvent {
        // SAFETY: Dummy device ids are only used to construct an inert unit-test event.
        device_id: unsafe { winit::event::DeviceId::dummy() },
        event: DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::LineDelta(2.0, 0.25),
        },
    };

    assert!(queue_routed_input_event(
        &mut input,
        RendererInputRouting::queue(),
        &event,
    ));
    input.dispatch_frame();
    assert_eq!(input.snapshot().scroll_delta_lines(), 2.0);
}

#[test]
fn queue_routed_input_event_preserves_device_pixel_wheel_queueing() {
    let mut input = InputSystem::new();
    let event = Event::DeviceEvent {
        // SAFETY: Dummy device ids are only used to construct an inert unit-test event.
        device_id: unsafe { winit::event::DeviceId::dummy() },
        event: DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::PixelDelta(PhysicalPosition::new(0.0, 240.0)),
        },
    };

    assert!(queue_routed_input_event(
        &mut input,
        RendererInputRouting::queue(),
        &event,
    ));
    input.dispatch_frame();
    assert_eq!(input.snapshot().scroll_delta_lines(), 2.0);
}
