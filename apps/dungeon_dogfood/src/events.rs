use renderer::{ActionPhase, EngineEvent, EventEnvelope, EventRecorder, LifecycleEvent, Renderer};

const EVENT_RECORDER_CAPACITY: usize = 128;
const FRAME_LOG_INTERVAL: u64 = 120;

pub fn install_dogfood_event_logger(renderer: &mut Renderer) {
    renderer.set_event_recorder(Some(EventRecorder::bounded(EVENT_RECORDER_CAPACITY)));
    renderer.events_mut().subscribe(|event| {
        log_dogfood_event(event);
        Ok(())
    });
}

fn log_dogfood_event(event: &EventEnvelope) {
    match &event.event {
        EngineEvent::Lifecycle(LifecycleEvent::FrameStarted) => {
            if let Some(frame) = event.frame {
                if frame.0 % FRAME_LOG_INTERVAL == 0 {
                    log::debug!(
                        "dogfood event {:?} frame={} sequence={}",
                        event.stage,
                        frame.0,
                        event.sequence.0
                    );
                }
            }
        }
        EngineEvent::Input(action)
            if matches!(action.phase, ActionPhase::Pressed | ActionPhase::Released) =>
        {
            log::debug!(
                "dogfood input action={} phase={:?} value={} source={}",
                action.action,
                action.phase,
                action.value,
                action.source.as_deref().unwrap_or("unknown")
            );
        }
        EngineEvent::Audio(audio) => {
            log::info!("dogfood audio event {:?}", audio);
        }
        _ => {}
    }
}
