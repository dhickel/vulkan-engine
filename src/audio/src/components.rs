//! Audio components for spatial audio scene attachment.
//!
//! These are version-1 component DTOs with exact built-in keys:
//! - `audio.listener` — [`AudioListener`]
//! - `audio.spatial_source` — [`AudioSource`]
//!
//! Always available (no feature gate).

use crate::AudioClipId;

/// Component key for the audio listener: `audio.listener`.
pub const AUDIO_LISTENER_COMPONENT_KEY: &str = "audio.listener";

/// Component key for a spatial audio source: `audio.spatial_source`.
pub const AUDIO_SPATIAL_SOURCE_COMPONENT_KEY: &str = "audio.spatial_source";

/// Marks a scene node (typically a Camera) as the audio listener.
///
/// Exactly one active listener is allowed per app bridge. Duplicate
/// activation is a typed error.
///
/// This is a marker component with no configurable fields. The listener
/// pose is derived from the owning node's world transform at runtime.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioListener;

/// Attaches a spatial audio clip to a scene node.
///
/// The owning node's world transform provides the source position at
/// runtime. Playback is managed by the app bridge, not by component
/// presence alone.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSource {
    /// Durable identity of the audio clip to play.
    pub clip_id: AudioClipId,
    /// Linear gain multiplier clamped to `[0, 1]`.
    pub gain: f32,
    /// Pitch multiplier (1.0 = normal). Reserved for future use.
    pub pitch: f32,
    /// Whether the clip restarts after finishing.
    pub looping: bool,
    /// Blend between fully 2D (0.0 = center pan) and fully spatial (1.0).
    pub spatial_blend: f32,
}

impl AudioSource {
    /// Create a new audio source component with validated fields.
    pub fn new(clip_id: AudioClipId, gain: f32, pitch: f32, looping: bool, spatial_blend: f32) -> Self {
        Self {
            clip_id,
            gain: clamp_finite(gain),
            pitch: clamp_pitch(pitch),
            looping,
            spatial_blend: clamp_finite_spatial_blend(spatial_blend),
        }
    }
}

fn clamp_finite(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn clamp_pitch(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.25, 4.0)
    } else {
        1.0
    }
}

fn clamp_finite_spatial_blend(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_listener_is_marker() {
        let a = AudioListener;
        let b = AudioListener::default();
        assert_eq!(a, b);
    }

    #[test]
    fn audio_source_clamps_fields() {
        let clip = AudioClipId::new("test.clip").unwrap();
        let src = AudioSource::new(clip.clone(), -0.5, 10.0, true, 1.5);
        assert_eq!(src.clip_id, clip);
        assert_eq!(src.gain, 0.0);
        assert_eq!(src.pitch, 4.0);
        assert!(src.looping);
        assert_eq!(src.spatial_blend, 1.0);
    }

    #[test]
    fn audio_source_nan_fallback() {
        let clip = AudioClipId::new("test.nan").unwrap();
        let src = AudioSource::new(clip, f32::NAN, f32::NAN, false, f32::NAN);
        assert_eq!(src.gain, 1.0);
        assert_eq!(src.pitch, 1.0);
        assert_eq!(src.spatial_blend, 0.0);
    }
}
