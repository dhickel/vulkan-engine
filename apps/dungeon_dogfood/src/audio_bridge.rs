use std::path::Path;

use audio::{AudioClip, AudioEngine, PlaybackOptions};
use engine::events::{AudioClipId, AudioEvent, EngineEvent, EventBus, EventStage};

use crate::content::{resolve_content_path, AudioClipSpec, ContentPack};

pub const AUDIO_SMOKE_ENV: &str = "DUNGEON_DOGFOOD_AUDIO_SMOKE";
pub const AUDIO_SMOKE_FLAG: &str = "--audio-smoke";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSmokeStatus {
    Skipped { reason: String },
    Passed,
    Failed { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioBridgeReport {
    pub clip_id: Option<String>,
    pub clip_path: Option<String>,
    pub device_smoke_status: DeviceSmokeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioRuntimeOutcome {
    Started,
    Stopped,
    Failed(String),
}

pub fn audio_smoke_requested_from<I, S>(args: I, env_value: Option<&str>) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    args.into_iter().any(|arg| arg.as_ref() == AUDIO_SMOKE_FLAG)
        || env_value
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
}

pub fn audio_smoke_requested() -> bool {
    audio_smoke_requested_from(
        std::env::args().skip(1),
        std::env::var(AUDIO_SMOKE_ENV).ok().as_deref(),
    )
}

pub fn run_startup_audio_probe(
    events: &mut EventBus,
    content_pack: &ContentPack,
    device_smoke_enabled: bool,
) -> AudioBridgeReport {
    let Some(clip_spec) = content_pack.startup_audio_clip() else {
        let report = AudioBridgeReport {
            clip_id: None,
            clip_path: None,
            device_smoke_status: DeviceSmokeStatus::Skipped {
                reason: "no audio clips configured".to_string(),
            },
        };
        log_audio_report(&report);
        return report;
    };

    let report = run_clip_probe(events, clip_spec, device_smoke_enabled);
    log_audio_report(&report);
    report
}

fn run_clip_probe(
    events: &mut EventBus,
    clip_spec: &AudioClipSpec,
    device_smoke_enabled: bool,
) -> AudioBridgeReport {
    let resolved_path = resolve_content_path(&clip_spec.path);
    let clip_path = resolved_path.display().to_string();
    let mut report = AudioBridgeReport {
        clip_id: Some(clip_spec.id.clone()),
        clip_path: Some(clip_path.clone()),
        device_smoke_status: DeviceSmokeStatus::Skipped {
            reason: "device smoke not requested".to_string(),
        },
    };

    if !device_smoke_enabled {
        log::info!(
            "Dogfood audio clip '{}' ({:?}/{:?}) is configured at '{}'; skipping probe/playback until {} or {}=1 is set",
            clip_spec.id,
            clip_spec.format,
            clip_spec.usage,
            clip_path,
            AUDIO_SMOKE_FLAG,
            AUDIO_SMOKE_ENV
        );
        return report;
    }

    let clip = match load_and_probe_clip(&clip_spec.id, &resolved_path) {
        Ok(clip) => clip,
        Err(message) => {
            emit_audio_event(
                events,
                &clip_spec.id,
                AudioRuntimeOutcome::Failed(message.clone()),
            );
            dispatch_audio_events(events);
            report.device_smoke_status = DeviceSmokeStatus::Failed { message };
            return report;
        }
    };

    log::info!(
        "Dogfood audio clip '{}' ({:?}/{:?}) probed from '{}' without opening an output device",
        clip_spec.id,
        clip_spec.format,
        clip_spec.usage,
        clip_path
    );

    match AudioEngine::new().and_then(|engine| {
        let handle = engine.play_with_options(
            &clip,
            PlaybackOptions::new(clip_spec.default_gain.unwrap_or(1.0)),
        )?;
        emit_audio_event(events, &clip_spec.id, AudioRuntimeOutcome::Started);
        handle.stop();
        emit_audio_event(events, &clip_spec.id, AudioRuntimeOutcome::Stopped);
        Ok(())
    }) {
        Ok(()) => {
            dispatch_audio_events(events);
            report.device_smoke_status = DeviceSmokeStatus::Passed;
        }
        Err(err) => {
            let message = err.to_string();
            emit_audio_event(
                events,
                &clip_spec.id,
                AudioRuntimeOutcome::Failed(message.clone()),
            );
            dispatch_audio_events(events);
            report.device_smoke_status = DeviceSmokeStatus::Failed { message };
        }
    }

    report
}

fn load_and_probe_clip(id: &str, path: &Path) -> Result<AudioClip, String> {
    let clip = AudioClip::load(id, path).map_err(|err| err.to_string())?;
    clip.probe().map_err(|err| err.to_string())?;
    Ok(clip)
}

pub fn emit_audio_event(events: &mut EventBus, clip_id: &str, outcome: AudioRuntimeOutcome) {
    events.emit(
        EventStage::Startup,
        None,
        audio_event_for_outcome(clip_id, outcome),
    );
}

pub fn audio_event_for_outcome(clip_id: &str, outcome: AudioRuntimeOutcome) -> EngineEvent {
    let clip = AudioClipId::new(clip_id.to_string());
    EngineEvent::Audio(match outcome {
        AudioRuntimeOutcome::Started => AudioEvent::ClipStarted { clip },
        AudioRuntimeOutcome::Stopped => AudioEvent::ClipStopped { clip },
        AudioRuntimeOutcome::Failed(message) => AudioEvent::ClipFailed { clip, message },
    })
}

fn dispatch_audio_events(events: &mut EventBus) {
    let report = events.drain_stage(EventStage::Startup);
    for failure in report.failures {
        log::warn!(
            "event listener {:?} failed for audio event {:?}: {}",
            failure.listener,
            failure.sequence,
            failure.message
        );
    }
}

fn log_audio_report(report: &AudioBridgeReport) {
    match &report.device_smoke_status {
        DeviceSmokeStatus::Skipped { reason } => {
            log::info!("Dogfood audio device smoke skipped: {reason}");
        }
        DeviceSmokeStatus::Passed => {
            log::info!("Dogfood audio device smoke passed");
        }
        DeviceSmokeStatus::Failed { message } => {
            log::warn!("Dogfood audio device smoke failed: {message}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use engine::events::{AudioEvent, EngineEvent, EventBus};

    use super::*;

    #[test]
    fn smoke_gate_accepts_flag_or_env() {
        assert!(audio_smoke_requested_from([AUDIO_SMOKE_FLAG], None));
        assert!(audio_smoke_requested_from(
            ["--level", "generated"],
            Some("true")
        ));
        assert!(!audio_smoke_requested_from(
            ["--level", "generated"],
            Some("0")
        ));
        assert!(!audio_smoke_requested_from(["--level", "generated"], None));
    }

    #[test]
    fn runtime_outcomes_map_to_audio_events_with_durable_clip_id() {
        assert!(matches!(
            audio_event_for_outcome("dogfood.audio.startup_ping", AudioRuntimeOutcome::Started),
            EngineEvent::Audio(AudioEvent::ClipStarted { clip }) if clip.as_str() == "dogfood.audio.startup_ping"
        ));
        assert!(matches!(
            audio_event_for_outcome("dogfood.audio.startup_ping", AudioRuntimeOutcome::Stopped),
            EngineEvent::Audio(AudioEvent::ClipStopped { clip }) if clip.as_str() == "dogfood.audio.startup_ping"
        ));
        assert!(matches!(
            audio_event_for_outcome(
                "dogfood.audio.startup_ping",
                AudioRuntimeOutcome::Failed("decode failed".to_string())
            ),
            EngineEvent::Audio(AudioEvent::ClipFailed { clip, message })
                if clip.as_str() == "dogfood.audio.startup_ping" && message == "decode failed"
        ));
    }

    #[test]
    fn emitted_audio_event_reaches_subscribers_without_device() {
        let mut events = EventBus::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);
        events.subscribe(move |event| {
            if let EngineEvent::Audio(audio_event) = &event.event {
                seen_listener.lock().unwrap().push(audio_event.clone());
            }
            Ok(())
        });

        emit_audio_event(
            &mut events,
            "dogfood.audio.startup_ping",
            AudioRuntimeOutcome::Failed("probe failed".to_string()),
        );
        dispatch_audio_events(&mut events);

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(matches!(
            &seen[0],
            AudioEvent::ClipFailed { clip, message }
                if clip.as_str() == "dogfood.audio.startup_ping" && message == "probe failed"
        ));
    }
}
