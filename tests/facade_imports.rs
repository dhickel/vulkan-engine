use std::time::Duration;

use engine::prelude::*;
use engine::{camera, command, events, input, object, render, time};
use engine_events as raw_events;
use input as raw_input;
use renderer as raw_renderer;
use winit::event::Event;
use winit::window::Window;

#[test]
fn facade_and_raw_crate_imports_compile() {
    // ── Time facade ──
    let _time_config = time::TimeConfig::default();
    let mut time_instance = time::Time::new(time::TimeConfig::default()).unwrap();
    let _update: time::TimeUpdate = time_instance.advance(Duration::from_millis(16));
    let _: Result<f32, time::TimeError> = time_instance.set_time_scale(1.0);
    let _: f32 = time_instance.time_scale();

    // ── Axis types from the prelude ──
    let _contrib = AxisContributor::new(ActionId::new("move.right"), 1.0);
    let _compound = CompoundAxis::new(vec![_contrib.clone()]);
    let _axis2d = Axis2D::new(
        CompoundAxis::new(vec![AxisContributor::new(ActionId::new("x"), 1.0)]),
        CompoundAxis::new(vec![AxisContributor::new(ActionId::new("y"), 1.0)]),
        0.1,
    );

    let mut facade_input = input::InputSystem::new();
    facade_input.dispatch_frame();
    let _: input::InputSnapshot = facade_input.snapshot().clone();
    let _: input::InputActionEventEmitter = input::InputActionEventEmitter::new();
    let _: fn(
        &mut render::Renderer,
        &Window,
        &mut input::InputSystem,
        &Event<()>,
    ) -> Result<render::RendererInputRouting, render::RendererError> =
        input::route_platform_input_to_app;
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
    let _: fn(&camera::Camera, u32, u32) -> render::CameraView = render::camera_view_for_size;
    let _: Option<render::FrameRenderOutcome> = None;
    let _: Option<render::MeshHandle> = None;
    let _: Option<render::MaterialHandle> = None;
    let _: Option<render::TextureHandle> = None;
    let _: render::CaptureTarget = render::CaptureTarget::Present;
    let _: Option<render::ObjectLifecycleOutcome> = None;
    let _: Option<render::ObjectMutationOutcome> = None;
    let _: Option<render::ObjectSummary> = None;

    // ── Object facade module ──
    let _: object::ObjectKind = object::ObjectKind::Node;
    let _: Option<object::ObjectId> = None;
    let _: Option<object::SceneObjectId> = None;
    let _: Option<object::ObjectSummary> = None;
    let _: Option<object::ObjectCapabilities> = None;
    let _: Option<object::ComponentEnvelope> = None;
    let _: Option<object::ComponentRegistry> = None;
    let _: Option<object::ObjectQueryFilter> = None;
    let _: Option<object::RayHit> = None;
    let _: Option<object::Selection> = None;
    let _: Option<object::SelectionChange> = None;
    let _: fn(&object::ObjectId) -> object::ObjectKind = object::object_kind;
    let _: fn(object::ObjectKind) -> &'static str = object::object_kind_label;

    // ── Command facade module ──
    let _: command::CommandHistory = command::CommandHistory::new(32);
    let _: Option<Box<dyn command::Command>> = None;
    let _: Option<command::CommandResult> = None;
    let _: Option<command::DuplicateObjectsCommand> = None;
    let _: Option<command::SetObjectTransformCommand> = None;

    // ── Camera facade: EditorCamera ──
    let _: Option<camera::EditorCamera> = None;
    let _: Option<camera::EditorProjection> = None;

    // ── Events facade: persistent lifecycle and opt-in legacy adapter ──
    let _: events::LegacySceneEventAdapter = events::LegacySceneEventAdapter;
    let _: Option<events::SceneObjectLifecycleEvent> = None;
    let _: Option<events::SceneObjectLifecycleSnapshot> = None;
    let _: Option<events::SceneObjectLifecycleAction> = None;

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

    let _: engine::frame::FixedStepClock =
        engine::frame::FixedStepClock::new(engine::frame::FixedStepConfig {
            step: Duration::from_millis(16),
            max_steps_per_frame: 4,
        });
    let _: Option<engine::frame::FixedStepUpdate> = None;
    let _: Option<engine::frame::AppFrameBeginReport> = None;
    let _: Option<engine::frame::AppFrameEndReport> = None;
    let _: fn(
        &mut input::InputSystem,
        &mut input::InputActionEventEmitter,
        &mut events::EventBus,
        &mut engine::frame::FrameClock,
    ) -> engine::frame::AppFrameBeginReport = engine::frame::begin_app_frame;
    let _: fn(
        &mut input::InputSystem,
        &mut input::InputActionEventEmitter,
        &mut events::EventBus,
        &mut time::Time,
    ) -> engine::frame::AppFrameBeginReport = engine::frame::begin_app_frame_with_time;
    let _: fn(&mut events::EventBus, u64) -> engine::frame::AppFrameEndReport =
        engine::frame::end_app_frame;
}

#[test]
fn prelude_imports_common_facade_types() {
    let _clock = FrameClock::new();
    let _fixed_clock = FixedStepClock::new(FixedStepConfig {
        step: Duration::from_millis(16),
        max_steps_per_frame: 4,
    });
    let _: Option<FixedStepUpdate> = None;
    let _: Option<AppFrameBeginReport> = None;
    let _: Option<AppFrameEndReport> = None;
    let _input = InputSystem::new();
    let _action_events = InputActionEventEmitter::new();
    let _event_bus = EventBus::new();
    let _config = RendererConfig::default();
    let _camera = Camera::default();
    let _: fn(&Camera, u32, u32) -> CameraView = camera_view_for_size;
    let _: fn(
        &mut Renderer,
        &Window,
        &mut InputSystem,
        &Event<()>,
    ) -> Result<RendererInputRouting, render::RendererError> = route_platform_input_to_app;
    let _: fn(
        &mut InputSystem,
        &mut InputActionEventEmitter,
        &mut EventBus,
        &mut FrameClock,
    ) -> AppFrameBeginReport = begin_app_frame;
    let _: fn(&mut EventBus, u64) -> AppFrameEndReport = end_app_frame;
    let _: Option<EditorCamera> = None;
    let _: Option<EditorProjection> = None;
    let _: fn(&ObjectId) -> ObjectKind = object_kind;
    let _: fn(ObjectKind) -> &'static str = object_kind_label;

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
        _editor_camera: Option<EditorCamera>,
        _editor_projection: Option<EditorProjection>,
        _object_id: Option<ObjectId>,
        _object_kind: Option<ObjectKind>,
        _persistent_object_id: Option<SceneObjectId>,
        _selection: Option<Selection>,
        _selection_change: Option<SelectionChange>,
        _query_filter: Option<ObjectQueryFilter>,
        _ray_hit: Option<RayHit>,
        _command: Option<Box<dyn Command>>,
        _command_history: Option<CommandHistory>,
        _command_result: Option<CommandResult>,
        _axis_contrib: Option<AxisContributor>,
        _compound_axis: Option<CompoundAxis>,
        _axis2d: Option<Axis2D>,
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
        None,
        None,
        None,
        None,
        None,
    );
}
