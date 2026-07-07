use engine::prelude::*;
use engine::{camera, events, input, render};
use engine_events as raw_events;
use input as raw_input;
use renderer as raw_renderer;

#[test]
fn facade_and_raw_crate_imports_compile() {
    let mut facade_input = input::InputSystem::new();
    facade_input.dispatch_frame();
    let _: input::InputSnapshot = facade_input.snapshot().clone();
    let _: input::InputActionEventEmitter = input::InputActionEventEmitter::new();
    let _: input::LayerDescriptor =
        input::LayerDescriptor::new("gameplay", input::priority_bands::GAMEPLAY_MIN);
    let _: input::ActionMap = input::ActionMap::new();

    let mut facade_events = events::EventBus::new();
    facade_events.emit(
        events::EventStage::Startup,
        Some(events::FrameId(0)),
        events::EngineEvent::Lifecycle(events::LifecycleEvent::FrameStarted),
    );
    let _ = facade_events.drain_stage(events::EventStage::Startup);

    let _: camera::Camera = camera::Camera::default();
    let _: Option<camera::FPSController> = None;
    let _: Option<camera::OrbitController> = None;
    let _: Option<camera::Ray> = None;

    let _: render::RendererConfig = render::RendererConfig::default();
    let _: Option<render::Renderer> = None;
    let _: Option<render::Scene> = None;
    let _: Option<render::CameraView> = None;
    let _: Option<render::FrameRenderOutcome> = None;
    let _: Option<render::MeshHandle> = None;
    let _: Option<render::MaterialHandle> = None;
    let _: Option<render::TextureHandle> = None;
    let _: render::CaptureTarget = render::CaptureTarget::Present;

    let mut raw_input_system = raw_input::InputSystem::new();
    raw_input_system.dispatch_frame();
    let mut raw_event_bus = raw_events::EventBus::new();
    raw_event_bus.emit(
        raw_events::EventStage::Startup,
        None,
        raw_events::EngineEvent::Lifecycle(raw_events::LifecycleEvent::FrameStarted),
    );
    let _: raw_renderer::RendererConfig = raw_renderer::RendererConfig::default();
    let _: Option<raw_renderer::CameraView> = None;
    let _: raw_renderer::RendererInputRouting = raw_renderer::RendererInputRouting::queue();

    let mut frame_clock = FrameClock::new();
    let frame = frame_clock.tick();
    assert_eq!(frame.index, 0);
}

#[test]
fn prelude_imports_common_facade_types() {
    let _clock = FrameClock::new();
    let _input = InputSystem::new();
    let _action_events = InputActionEventEmitter::new();
    let _event_bus = EventBus::new();
    let _config = RendererConfig::default();
    let _camera = Camera::default();

    fn accepts_prelude_types(
        _action: ActionId,
        _event_stage: EventStage,
        _frame: FrameId,
        _input_event: Option<InputEvent>,
        _snapshot: Option<InputSnapshot>,
        _renderer: Option<Renderer>,
        _scene: Option<Scene>,
        _view: Option<CameraView>,
        _context: Option<FrameContext>,
        _outcome: Option<FrameRenderOutcome>,
        _node: Option<SceneNodeId>,
        _controller: Option<FPSController>,
        _orbit: Option<OrbitCamera>,
    ) {
    }

    accepts_prelude_types(
        ActionId::new("jump"),
        EventStage::Input,
        FrameId(0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
}
