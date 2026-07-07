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
