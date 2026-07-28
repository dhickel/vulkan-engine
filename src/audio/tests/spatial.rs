#![cfg(feature = "spatial-audio")]

//! Integration tests for the spatial-audio feature.
//!
//! Validates the full pipeline: source adapter, panning math, attenuation,
//! device-backed spatial playback, mono-only validation, component DTOs,
//! and legacy non-spatial playback compatibility.

use audio::components::{AudioListener, AudioSource, AUDIO_LISTENER_COMPONENT_KEY, AUDIO_SPATIAL_SOURCE_COMPONENT_KEY};
use audio::spatial::{
    self, AtomicSpatialGains, Attenuation, ListenerPose, SourcePose, SpatialGain, SpatialSource, Vec3,
};
use audio::{AudioClip, AudioEngine, AudioError, PlaybackOptions};
use rodio::buffer::SamplesBuffer;
use rodio::Source;
use std::sync::Arc;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a tiny mono WAV in memory (the same helper as the unit tests).
fn tiny_wav_bytes_mono() -> Vec<u8> {
    let channels = 1u16;
    let sample_rate = 8_000u32;
    let bits_per_sample = 16u16;
    let samples: [i16; 32] = [
        0, 2048, 4096, 2048, 0, -2048, -4096, -2048, 0, 2048, 4096, 2048, 0, -2048, -4096,
        -2048, 0, 2048, 4096, 2048, 0, -2048, -4096, -2048, 0, 2048, 4096, 2048, 0, -2048,
        -4096, -2048,
    ];
    let data_len = samples.len() as u32 * std::mem::size_of::<i16>() as u32;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Generate a stereo WAV in memory for testing multi-channel rejection.
fn tiny_wav_bytes_stereo() -> Vec<u8> {
    let channels = 2u16;
    let sample_rate = 8_000u32;
    let bits_per_sample = 16u16;
    // Interleaved stereo: [L, R, L, R, ...]
    let samples: [i16; 64] = [
        0, 0, 2048, 1024, 4096, 2048, 2048, 1024,
        0, 0, -2048, -1024, -4096, -2048, -2048, -1024,
        0, 0, 2048, 1024, 4096, 2048, 2048, 1024,
        0, 0, -2048, -1024, -4096, -2048, -2048, -1024,
        0, 0, 2048, 1024, 4096, 2048, 2048, 1024,
        0, 0, -2048, -1024, -4096, -2048, -2048, -1024,
        0, 0, 2048, 1024, 4096, 2048, 2048, 1024,
        0, 0, -2048, -1024, -4096, -2048, -2048, -1024,
    ];
    let data_len = samples.len() as u32 * std::mem::size_of::<i16>() as u32;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;

    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn collect_samples<I: Source<Item = f32>>(source: I, limit: usize) -> Vec<f32> {
    source.take(limit).collect()
}

// ── Spatial source sample-level proof ──────────────────────────────────────

#[test]
fn spatial_source_produces_different_left_and_right_samples_from_mono_input() {
    // This is the admission gate: scalar Sink::set_volume alone cannot produce
    // different L/R output from a mono source. Our SpatialSource can.
    let mono_buf = SamplesBuffer::new(1, 8000, vec![1.0_f32; 10]);
    let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::new(0.8, 0.2)));
    let spatial = SpatialSource::new(mono_buf, gains);

    let samples: Vec<f32> = collect_samples(spatial, 10);
    for pair in samples.chunks(2) {
        let left = pair[0];
        let right = pair[1];
        assert!(
            (left - 0.8).abs() < 1e-5,
            "expected left=0.8, got {left}"
        );
        assert!(
            (right - 0.2).abs() < 1e-5,
            "expected right=0.2, got {right}"
        );
        assert!(
            (left - right).abs() > 0.1,
            "L and R must differ (asymmetric gains), got L={left} R={right}"
        );
    }
}

#[test]
fn spatial_source_center_gains_produce_equal_channels() {
    let mono_buf = SamplesBuffer::new(1, 8000, vec![0.5_f32; 8]);
    let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::CENTER));
    let spatial = SpatialSource::new(mono_buf, gains);

    let samples: Vec<f32> = collect_samples(spatial, 8);
    let center = std::f32::consts::FRAC_1_SQRT_2 * 0.5;
    for pair in samples.chunks(2) {
        assert!((pair[0] - center).abs() < 1e-5);
        assert!((pair[1] - center).abs() < 1e-5);
        assert!((pair[0] - pair[1]).abs() < 1e-5);
    }
}

// ── Full spatialization pipeline tests ──────────────────────────────────────

#[test]
fn spatialize_full_pipeline_right_half_distance() {
    let listener = ListenerPose::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(Vec3::new(2.0, 0.0, 0.0)); // 2 units right
    let att = Attenuation {
        min_distance: 1.0,
        max_distance: 100.0,
    };
    let g = spatial::spatialize(&listener, &source, &att);
    // Pan: hard right → left=0, right=1
    // Attenuation: (1/2)² = 0.25
    // Final: left=0, right=0.25
    assert!((g.left - 0.0).abs() < 1e-5, "left: {}", g.left);
    assert!((g.right - 0.25).abs() < 1e-5, "right: {}", g.right);
}

#[test]
fn spatialize_degenerate_listener_right_produces_center_pan() {
    let listener = ListenerPose::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);
    let source = SourcePose::new(Vec3::new(1.0, 0.0, 0.0));
    let att = Attenuation::default();
    let g = spatial::spatialize(&listener, &source, &att);
    // Pan: center (zero right → pan=0)
    // Attenuation: (1/1)² = 1.0
    let c = std::f32::consts::FRAC_1_SQRT_2;
    assert!((g.left - c).abs() < 1e-4, "left: {}", g.left);
    assert!((g.right - c).abs() < 1e-4, "right: {}", g.right);
}

#[test]
fn spatialize_coincident_positions_produces_center_with_full_gain() {
    let pos = Vec3::new(5.0, 10.0, -3.0);
    let listener = ListenerPose::new(pos, Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(pos);
    let att = Attenuation::default();
    let g = spatial::spatialize(&listener, &source, &att);
    let c = std::f32::consts::FRAC_1_SQRT_2;
    assert!((g.left - c).abs() < 1e-4);
    assert!((g.right - c).abs() < 1e-4);
}

#[test]
fn spatialize_beyond_max_distance_silent() {
    let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(Vec3::new(500.0, 0.0, 0.0));
    let att = Attenuation {
        min_distance: 1.0,
        max_distance: 50.0,
    };
    let g = spatial::spatialize(&listener, &source, &att);
    assert_eq!(g.left, 0.0);
    assert_eq!(g.right, 0.0);
}

// ── blend_gains / spatial_blend tests ───────────────────────────────────────

#[test]
fn spatial_blend_zero_is_center() {
    let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(Vec3::new(1.0, 0.0, 0.0)); // hard right
    let att = Attenuation {
        min_distance: 1.0,
        max_distance: 100.0,
    };
    let spatial_gains = spatial::spatialize(&listener, &source, &att);
    // spatial: hard right (left=0, right=1)
    assert!((spatial_gains.right - 1.0).abs() < 1e-5);
    let blended = spatial::blend_gains(SpatialGain::CENTER, spatial_gains, 0.0);
    // blend=0 → CENTER
    let c = std::f32::consts::FRAC_1_SQRT_2;
    assert!((blended.left - c).abs() < 1e-4);
    assert!((blended.right - c).abs() < 1e-4);
}

#[test]
fn spatial_blend_one_is_fully_spatial() {
    let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(Vec3::new(1.0, 0.0, 0.0));
    let att = Attenuation {
        min_distance: 1.0,
        max_distance: 100.0,
    };
    let spatial_gains = spatial::spatialize(&listener, &source, &att);
    let blended = spatial::blend_gains(SpatialGain::CENTER, spatial_gains, 1.0);
    assert!((blended.left - spatial_gains.left).abs() < 1e-5);
    assert!((blended.right - spatial_gains.right).abs() < 1e-5);
}

// ── Device-backed spatial playback tests ────────────────────────────────────

#[test]
fn play_spatial_mono_rejects_multichannel_clip() {
    let clip = AudioClip::from_bytes("test.stereo", tiny_wav_bytes_stereo()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let err = engine.play_spatial_mono(&clip).unwrap_err();
    assert!(
        matches!(&err, AudioError::Playback { clip_id, message } if clip_id.as_str() == "test.stereo" && message.contains("channels")),
        "expected channel rejection, got: {err:?}"
    );
}

#[test]
fn play_spatial_mono_accepts_mono_clip_and_produces_handle() {
    let clip = AudioClip::from_bytes("test.mono", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let handle = engine.play_spatial_mono(&clip).unwrap();
    // Immediately stop to avoid any audible output.
    // (stop() clears the pending queue; a currently-playing source
    // may report non-empty briefly.)
    handle.stop();
}

#[test]
fn spatial_handle_update_spatial_changes_gains() {
    let clip = AudioClip::from_bytes("test.spatial.update", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let handle = engine.play_spatial_mono(&clip).unwrap();

    // Initial gains: CENTER
    let g = handle.spatial_gains();
    let c = std::f32::consts::FRAC_1_SQRT_2;
    assert!((g.left - c).abs() < 1e-4);
    assert!((g.right - c).abs() < 1e-4);

    // Update to spatialized position (hard right at min distance)
    let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    let source = SourcePose::new(Vec3::new(1.0, 0.0, 0.0));
    let att = Attenuation {
        min_distance: 1.0,
        max_distance: 100.0,
    };
    handle.update_spatial(&listener, &source, &att, 1.0);

    let g = handle.spatial_gains();
    assert!((g.left - 0.0).abs() < 1e-5, "left: {}", g.left);
    assert!((g.right - 1.0).abs() < 1e-5, "right: {}", g.right);

    handle.stop();
}

#[test]
fn spatial_handle_set_spatial_gains_directly() {
    let clip = AudioClip::from_bytes("test.spatial.direct", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let handle = engine.play_spatial_mono(&clip).unwrap();
    handle.set_spatial_gains(SpatialGain::new(0.3, 0.7));
    let g = handle.spatial_gains();
    assert!((g.left - 0.3).abs() < 1e-5);
    assert!((g.right - 0.7).abs() < 1e-5);
    handle.stop();
}

#[test]
fn spatial_handle_master_volume_is_independent() {
    let clip = AudioClip::from_bytes("test.spatial.vol", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let handle = engine.play_spatial_mono(&clip).unwrap();
    // Master volume is separate from spatial gains
    assert_eq!(handle.volume(), 1.0);
    handle.set_volume(0.5);
    assert_eq!(handle.volume(), 0.5);
    // Spatial gains should be unchanged
    let g = handle.spatial_gains();
    let c = std::f32::consts::FRAC_1_SQRT_2;
    assert!((g.left - c).abs() < 1e-4);
    assert!((g.right - c).abs() < 1e-4);
    handle.stop();
}

#[test]
fn spatial_handle_lifecycle_pause_play_stop() {
    let clip = AudioClip::from_bytes("test.spatial.lifecycle", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping spatial playback test");
            return;
        }
    };

    let handle = engine.play_spatial_mono(&clip).unwrap();
    handle.pause();
    handle.play();
    handle.stop();
    // stop() clears the pending queue; a currently-playing source may
    // still report non-empty briefly. The lifecycle methods themselves
    // must not panic.
}

// ── Legacy playback compatibility ───────────────────────────────────────────

#[test]
fn non_spatial_playback_still_works_with_spatial_audio_feature_enabled() {
    let clip = AudioClip::from_bytes("test.legacy", tiny_wav_bytes_mono()).unwrap();
    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("no audio device available; skipping test");
            return;
        }
    };

    // Legacy play should still work
    let handle = engine.play(&clip).unwrap();
    handle.set_volume(0.5);
    assert_eq!(handle.volume(), 0.5);
    handle.stop();

    // Legacy play_with_options should still work
    let handle = engine
        .play_with_options(&clip, PlaybackOptions::new(0.3))
        .unwrap();
    assert_eq!(handle.volume(), 0.3);
    handle.stop();
}

// ── Component DTO tests ─────────────────────────────────────────────────────

#[test]
fn audio_listener_component_marker() {
    let a = AudioListener;
    let b = AudioListener::default();
    assert_eq!(a, b);
}

#[test]
fn audio_listener_component_key_is_stable() {
    assert_eq!(AUDIO_LISTENER_COMPONENT_KEY, "audio.listener");
}

#[test]
fn audio_spatial_source_component_key_is_stable() {
    assert_eq!(AUDIO_SPATIAL_SOURCE_COMPONENT_KEY, "audio.spatial_source");
}

#[test]
fn audio_source_component_round_trip_fields() {
    let clip_id = audio::AudioClipId::new("dogfood.audio.step").unwrap();
    let src = AudioSource::new(clip_id.clone(), 0.8, 1.5, true, 0.6);

    assert_eq!(src.clip_id, clip_id);
    assert_eq!(src.gain, 0.8);
    assert_eq!(src.pitch, 1.5);
    assert!(src.looping);
    assert_eq!(src.spatial_blend, 0.6);
}

// ── Atomic gain thread-safety proof ─────────────────────────────────────────

#[test]
fn atomic_gains_can_be_shared_across_threads() {
    let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::CENTER));
    let g2 = Arc::clone(&gains);

    // Update from "main thread"
    gains.set(SpatialGain::new(0.2, 0.8));

    // Read from "audio thread" (simulated)
    let handle = std::thread::spawn(move || {
        let g = g2.get();
        assert!((g.left - 0.2).abs() < 1e-5);
        assert!((g.right - 0.8).abs() < 1e-5);
    });
    handle.join().unwrap();
}
