use crate::api::renderer::Renderer;
use engine_events::{ActionPhase, EngineEvent, EventEnvelope, EventRecorder, LifecycleEvent};

const EVENT_RECORDER_CAPACITY: usize = 128;
const FRAME_LOG_INTERVAL: u64 = 120;

/// Install a standard app event logger with an app-specific prefix.
///
/// This subscribes a listener that logs lifecycle and input events at
/// a throttled rate. The `app_name` parameter is used as a log prefix
/// to distinguish events from different applications.
pub fn install_app_event_logger(renderer: &mut Renderer, app_name: &str) {
    let prefix = app_name.to_string();
    renderer.set_event_recorder(Some(EventRecorder::bounded(EVENT_RECORDER_CAPACITY)));
    renderer.events_mut().subscribe(move |event| {
        log_app_event(event, &prefix);
        Ok(())
    });
}

fn log_app_event(event: &EventEnvelope, prefix: &str) {
    match &event.event {
        EngineEvent::Lifecycle(LifecycleEvent::FrameStarted) => {
            if let Some(frame) = event.frame {
                if frame.0 % FRAME_LOG_INTERVAL == 0 {
                    log::debug!(
                        "{prefix} event {:?} frame={} sequence={}",
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
                "{prefix} input action={} phase={:?} value={} source={}",
                action.action,
                action.phase,
                action.value,
                action.source.as_deref().unwrap_or("unknown")
            );
        }
        EngineEvent::Audio(audio) => {
            log::info!("{prefix} audio event {:?}", audio);
        }
        _ => {}
    }
}
