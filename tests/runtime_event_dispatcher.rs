use std::sync::{Arc, Mutex};

use engine::events::{
    runtime_event_bus, EngineEvent, EventBus, EventStage, FrameId, LifecycleEvent, ListenerError,
    RuntimeEventDispatcher,
};

fn recorded_lifecycle(bus: &EventBus) -> Vec<(u64, EventStage, Option<u64>, LifecycleEvent)> {
    bus.recorder()
        .map(|recorder| {
            recorder
                .entries()
                .filter_map(|envelope| match &envelope.event {
                    EngineEvent::Lifecycle(event) => Some((
                        envelope.sequence.0,
                        envelope.stage,
                        envelope.frame.map(|frame| frame.0),
                        event.clone(),
                    )),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn runtime_dispatcher_preserves_one_bus_monotonic_sequence() {
    let mut bus = runtime_event_bus();

    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Startup,
        None,
        LifecycleEvent::AppStarting {
            app_name: "test".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::PreUpdate,
        Some(FrameId(0)),
        LifecycleEvent::FrameStarted,
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::PostUpdate,
        Some(FrameId(0)),
        LifecycleEvent::FrameEnded,
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Render,
        Some(FrameId(0)),
        LifecycleEvent::FrameEnded,
    );

    let recorded = recorded_lifecycle(&bus);
    let sequences = recorded
        .iter()
        .map(|(sequence, _, _, _)| *sequence)
        .collect::<Vec<_>>();
    assert_eq!(sequences, vec![0, 1, 2, 3]);
    assert_eq!(
        recorded
            .iter()
            .map(|(_, stage, _, _)| *stage)
            .collect::<Vec<_>>(),
        vec![
            EventStage::Startup,
            EventStage::PreUpdate,
            EventStage::PostUpdate,
            EventStage::Render
        ]
    );
}

#[test]
fn runtime_frame_helpers_emit_one_start_and_end_for_frame() {
    let mut bus = runtime_event_bus();

    RuntimeEventDispatcher::frame_started(&mut bus, 7);
    RuntimeEventDispatcher::drain_input(&mut bus);
    RuntimeEventDispatcher::frame_ended(&mut bus, 7);

    let lifecycle = recorded_lifecycle(&bus);
    assert_eq!(
        lifecycle,
        vec![
            (
                0,
                EventStage::PreUpdate,
                Some(7),
                LifecycleEvent::FrameStarted
            ),
            (
                1,
                EventStage::PostUpdate,
                Some(7),
                LifecycleEvent::FrameEnded
            )
        ]
    );
    assert_eq!(bus.pending_len(), 0);
}

#[test]
fn runtime_dispatcher_collects_listener_failure_and_continues() {
    let mut bus = runtime_event_bus();
    let observed = Arc::new(Mutex::new(Vec::new()));

    bus.subscribe(|_| Err(ListenerError::new("first listener failed")));

    let observed_for_listener = Arc::clone(&observed);
    bus.subscribe(move |envelope| {
        observed_for_listener
            .lock()
            .expect("observed lifecycle mutex poisoned")
            .push(envelope.sequence.0);
        Ok(())
    });

    let report = RuntimeEventDispatcher::frame_started(&mut bus, 3);

    assert_eq!(report.dispatched, 1);
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].message, "first listener failed");
    assert_eq!(
        *observed
            .lock()
            .expect("observed lifecycle mutex poisoned after dispatch"),
        vec![0]
    );
}

// ── Lifecycle scenario tests ─────────────────────────────────────────

#[test]
fn lifecycle_scenario_startup_shutdown_without_vulkan() {
    let mut bus = runtime_event_bus();
    let app_name = "scenario-test";

    // Simulate the startup sequence.
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Startup,
        None,
        LifecycleEvent::AppStarting {
            app_name: app_name.to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::ProjectLoad,
        None,
        LifecycleEvent::ProjectLoading {
            path: "test.project.toml".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::ProjectLoad,
        None,
        LifecycleEvent::ProjectLoaded {
            project: engine_events::ProjectId::new("project.test"),
            path: "test.project.toml".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::SceneLoad,
        None,
        LifecycleEvent::SceneLoaded {
            scene: engine_events::SceneId::new("scene.start"),
            path: "scenes/start.engine.scene.json".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Startup,
        None,
        LifecycleEvent::AppStarted {
            app_name: app_name.to_string(),
        },
    );

    // Simulate a few frames.
    for i in 0..3u64 {
        RuntimeEventDispatcher::frame_started(&mut bus, i);
        RuntimeEventDispatcher::drain_input(&mut bus);
        RuntimeEventDispatcher::frame_ended(&mut bus, i);
    }

    // Simulate shutdown.
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Shutdown,
        None,
        LifecycleEvent::ShutdownRequested {
            reason: "test complete".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Shutdown,
        None,
        LifecycleEvent::ShutdownCompleted,
    );

    let lifecycle = recorded_lifecycle(&bus);
    let labels: Vec<&str> = lifecycle
        .iter()
        .map(|(_, _, _, event)| match event {
            LifecycleEvent::AppStarting { .. } => "app_starting",
            LifecycleEvent::AppStarted { .. } => "app_started",
            LifecycleEvent::ProjectLoading { .. } => "project_loading",
            LifecycleEvent::ProjectLoaded { .. } => "project_loaded",
            LifecycleEvent::SceneLoading { .. } => "scene_loading",
            LifecycleEvent::SceneLoaded { .. } => "scene_loaded",
            LifecycleEvent::FrameStarted => "frame_started",
            LifecycleEvent::FrameEnded => "frame_ended",
            LifecycleEvent::ShutdownRequested { .. } => "shutdown_requested",
            LifecycleEvent::ShutdownCompleted => "shutdown_completed",
            _ => "other",
        })
        .collect();

    assert_eq!(
        labels,
        vec![
            "app_starting",
            "project_loading",
            "project_loaded",
            "scene_loaded",
            "app_started",
            "frame_started",
            "frame_ended",
            "frame_started",
            "frame_ended",
            "frame_started",
            "frame_ended",
            "shutdown_requested",
            "shutdown_completed",
        ]
    );
}

#[test]
fn lifecycle_scenario_paired_frame_events_never_dangling() {
    let mut bus = runtime_event_bus();

    // Even on error, we should get paired frame events.
    for i in 0..5u64 {
        RuntimeEventDispatcher::frame_started(&mut bus, i);
        RuntimeEventDispatcher::drain_input(&mut bus);
        RuntimeEventDispatcher::frame_ended(&mut bus, i);
    }

    let mut frame_starts = 0u64;
    let mut frame_ends = 0u64;
    for (_, _, _, event) in recorded_lifecycle(&bus) {
        if matches!(event, LifecycleEvent::FrameStarted) {
            frame_starts += 1;
        }
        if matches!(event, LifecycleEvent::FrameEnded) {
            frame_ends += 1;
        }
    }

    assert_eq!(frame_starts, 5);
    assert_eq!(frame_ends, 5);
    assert_eq!(bus.pending_len(), 0);
}

#[test]
fn lifecycle_scenario_terminal_error_records_shutdown() {
    let mut bus = runtime_event_bus();

    // Partial startup then immediate shutdown due to an error.
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Startup,
        None,
        LifecycleEvent::AppStarting {
            app_name: "failing".to_string(),
        },
    );

    // Simulate a pre-frame error: emit shutdown without finishing the frame.
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Shutdown,
        None,
        LifecycleEvent::ShutdownRequested {
            reason: "device lost during frame setup".to_string(),
        },
    );
    RuntimeEventDispatcher::emit_lifecycle_and_drain(
        &mut bus,
        EventStage::Shutdown,
        None,
        LifecycleEvent::ShutdownCompleted,
    );

    let lifecycle = recorded_lifecycle(&bus);
    assert_eq!(lifecycle.len(), 3);
    assert!(matches!(lifecycle[0].3, LifecycleEvent::AppStarting { .. }));
    assert!(matches!(
        lifecycle[1].3,
        LifecycleEvent::ShutdownRequested { .. }
    ));
    assert!(matches!(lifecycle[2].3, LifecycleEvent::ShutdownCompleted));
}
