//! Renderer-independent alpha audio facade.
//!
//! Clips have durable authored IDs and can be loaded, constructed, and probed
//! without opening an output device. Device-backed playback is explicit through
//! `AudioEngine`.
//!
//! ## Spatial audio
//!
//! Stereo panning + distance attenuation, [`AudioSource`]
//! and [`AudioListener`] components, and [`AudioEngine::play_spatial_mono`]
//! are always available.

pub mod components;
pub mod spatial;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use std::error::Error;
use std::fmt;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use std::sync::Arc as StdArc;

/// Stable authored identity for an audio clip.
///
/// Wraps the canonical `engine_events::AudioClipId` with validation that
/// ensures the ID is non-empty and contains only allowed characters.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AudioClipId(engine_events::AudioClipId);

impl AudioClipId {
    /// Create a durable clip ID such as `dogfood.audio.pickup`.
    pub fn new(id: impl Into<String>) -> Result<Self, AudioError> {
        let id = id.into();
        if is_valid_clip_id(&id) {
            Ok(Self(engine_events::AudioClipId::new(id)))
        } else {
            Err(AudioError::InvalidClipId { id })
        }
    }

    /// Borrow the durable ID as a string.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for AudioClipId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<AudioClipId> for engine_events::AudioClipId {
    fn from(id: AudioClipId) -> Self {
        id.0
    }
}

/// Audio errors with stable variants for validation and integration tests.
#[derive(Debug)]
pub enum AudioError {
    /// The durable clip ID is empty or contains unsupported characters.
    InvalidClipId { id: String },
    /// Clip bytes could not be read from disk.
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Clip bytes could not be decoded by the supported decoder stack.
    Decode {
        clip_id: AudioClipId,
        message: String,
    },
    /// The host output device or audio stream could not be opened.
    Device { message: String },
    /// A playback sink could not be created or used.
    Playback {
        clip_id: AudioClipId,
        message: String,
    },
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidClipId { id } => write!(f, "invalid audio clip id: {id}"),
            Self::Read { path, source } => {
                write!(f, "failed to read audio file {}: {source}", path.display())
            }
            Self::Decode { clip_id, message } => {
                write!(f, "failed to decode audio clip {clip_id}: {message}")
            }
            Self::Device { message } => write!(f, "failed to open audio device: {message}"),
            Self::Playback { clip_id, message } => {
                write!(f, "failed to play audio clip {clip_id}: {message}")
            }
        }
    }
}

impl Error for AudioError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Device-independent metadata probed from decoded clip bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipProbe {
    pub clip_id: AudioClipId,
    pub byte_len: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration: Option<Duration>,
}

/// Audio clip bytes plus durable authored identity.
#[derive(Clone, Debug)]
pub struct AudioClip {
    id: AudioClipId,
    data: Arc<[u8]>,
    source_path: Option<PathBuf>,
}

impl AudioClip {
    /// Load an audio clip from a file without opening an output device.
    pub fn load(id: impl Into<String>, path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let id = AudioClipId::new(id)?;
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(|source| AudioError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            id,
            data: data.into(),
            source_path: Some(path.to_path_buf()),
        })
    }

    /// Create a clip from encoded audio bytes without opening an output device.
    pub fn from_bytes(id: impl Into<String>, data: impl Into<Vec<u8>>) -> Result<Self, AudioError> {
        Ok(Self {
            id: AudioClipId::new(id)?,
            data: data.into().into(),
            source_path: None,
        })
    }

    /// Durable authored identity for this clip.
    pub fn id(&self) -> &AudioClipId {
        &self.id
    }

    /// Encoded audio byte length.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Whether the encoded clip byte buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Optional path this clip was loaded from.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Probe decode metadata without opening an output device.
    pub fn probe(&self) -> Result<ClipProbe, AudioError> {
        let decoder = self.decoder()?;
        Ok(ClipProbe {
            clip_id: self.id.clone(),
            byte_len: self.byte_len(),
            channels: decoder.channels(),
            sample_rate: decoder.sample_rate(),
            duration: decoder.total_duration(),
        })
    }

    fn decoder(&self) -> Result<Decoder<BufReader<Cursor<Vec<u8>>>>, AudioError> {
        let cursor = Cursor::new(self.data.to_vec());
        Decoder::new(BufReader::new(cursor)).map_err(|err| AudioError::Decode {
            clip_id: self.id.clone(),
            message: err.to_string(),
        })
    }
}

/// Playback settings applied when a clip is attached to a sink.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaybackOptions {
    volume: f32,
}

impl PlaybackOptions {
    /// Create playback options with volume clamped to the supported range.
    pub fn new(volume: f32) -> Self {
        Self {
            volume: clamp_volume(volume),
        }
    }

    /// Effective playback volume in the `0.0..=1.0` range.
    pub fn volume(&self) -> f32 {
        self.volume
    }
}

impl Default for PlaybackOptions {
    fn default() -> Self {
        Self { volume: 1.0 }
    }
}

/// Explicit device-backed audio engine.
pub struct AudioEngine {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    master_volume: f32,
}

impl AudioEngine {
    /// Open the host default output stream.
    pub fn new() -> Result<Self, AudioError> {
        let (stream, stream_handle) =
            OutputStream::try_default().map_err(|err| AudioError::Device {
                message: err.to_string(),
            })?;
        Ok(Self {
            _stream: stream,
            stream_handle,
            master_volume: 1.0,
        })
    }

    /// Play an audio clip with default playback options.
    pub fn play(&self, clip: &AudioClip) -> Result<PlaybackHandle, AudioError> {
        self.play_with_options(clip, PlaybackOptions::default())
    }

    /// Play an audio clip with explicit playback options.
    pub fn play_with_options(
        &self,
        clip: &AudioClip,
        options: PlaybackOptions,
    ) -> Result<PlaybackHandle, AudioError> {
        let source = clip.decoder()?;
        let sink = Sink::try_new(&self.stream_handle).map_err(|err| AudioError::Playback {
            clip_id: clip.id().clone(),
            message: err.to_string(),
        })?;
        sink.set_volume(options.volume());
        sink.append(source);
        Ok(PlaybackHandle { sink })
    }

    /// Current master volume, clamped to `0.0..=1.0`.
    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }

    /// Set the master volume, clamped to `0.0..=1.0`.
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = clamp_volume(volume);
    }

    /// Play a mono audio clip with spatial stereo panning.
    ///
    /// The clip must be mono (1 channel); multi-channel clips return an error
    /// without side effects. The returned [`SpatialPlaybackHandle`] provides
    /// independent left/right gain control via shared atomic state.
    pub fn play_spatial_mono(
        &self,
        clip: &AudioClip,
    ) -> Result<SpatialPlaybackHandle, AudioError> {
        let source = clip.decoder()?;
        if source.channels() != 1 {
            return Err(AudioError::Playback {
                clip_id: clip.id().clone(),
                message: format!(
                    "spatial audio requires mono source, got {} channels",
                    source.channels()
                ),
            });
        }
        let gains = StdArc::new(spatial::AtomicSpatialGains::new(
            spatial::SpatialGain::CENTER,
        ));
        let spatial_source =
            spatial::SpatialSource::new(source, StdArc::clone(&gains));
        let sink = Sink::try_new(&self.stream_handle).map_err(|err| AudioError::Playback {
            clip_id: clip.id().clone(),
            message: err.to_string(),
        })?;
        sink.append(spatial_source);
        Ok(SpatialPlaybackHandle { sink, gains })
    }
}

/// Handle to active device-backed playback.
pub struct PlaybackHandle {
    sink: Sink,
}

impl PlaybackHandle {
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(clamp_volume(volume));
    }

    pub fn volume(&self) -> f32 {
        self.sink.volume()
    }

    pub fn pause(&self) {
        self.sink.pause();
    }

    pub fn play(&self) {
        self.sink.play();
    }

    pub fn stop(&self) {
        self.sink.stop();
    }

    pub fn is_playing(&self) -> bool {
        !self.sink.empty()
    }
}

/// Handle to active device-backed spatial playback.
///
/// Wraps a rodio [`Sink`] together with shared atomic stereo gains so the
/// caller can update the listener/source position and have the changes take
/// effect on the audio thread without locks.
pub struct SpatialPlaybackHandle {
    sink: Sink,
    gains: StdArc<spatial::AtomicSpatialGains>,
}

impl fmt::Debug for SpatialPlaybackHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialPlaybackHandle")
            .field("volume", &self.sink.volume())
            .field("empty", &self.sink.empty())
            .field("gains", &self.gains.get())
            .finish()
    }
}

impl SpatialPlaybackHandle {
    /// Set the master (sink) scalar volume, clamped to `[0, 1]`.
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(clamp_volume(volume));
    }

    /// Current master (sink) scalar volume.
    pub fn volume(&self) -> f32 {
        self.sink.volume()
    }

    /// Overwrite the spatial stereo gains directly.
    pub fn set_spatial_gains(&self, gains: spatial::SpatialGain) {
        self.gains.set(gains);
    }

    /// Compute and apply spatial gains from listener, source, and attenuation.
    pub fn update_spatial(
        &self,
        listener: &spatial::ListenerPose,
        source: &spatial::SourcePose,
        attenuation: &spatial::Attenuation,
        spatial_blend: f32,
    ) {
        let spatial_gains = spatial::spatialize(listener, source, attenuation);
        let blended = spatial::blend_gains(
            spatial::SpatialGain::CENTER,
            spatial_gains,
            spatial_blend,
        );
        self.gains.set(blended);
    }

    /// Read the current spatial gain pair.
    pub fn spatial_gains(&self) -> spatial::SpatialGain {
        self.gains.get()
    }

    pub fn pause(&self) {
        self.sink.pause();
    }

    pub fn play(&self) {
        self.sink.play();
    }

    pub fn stop(&self) {
        self.sink.stop();
    }

    pub fn is_playing(&self) -> bool {
        !self.sink.empty()
    }
}

fn clamp_volume(volume: f32) -> f32 {
    if volume.is_nan() {
        1.0
    } else {
        volume.clamp(0.0, 1.0)
    }
}

fn is_valid_clip_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn clip_from_bytes_keeps_durable_identity() {
        let bytes = tiny_wav_bytes();
        let clip = AudioClip::from_bytes("test.audio.beep", bytes.clone()).unwrap();

        assert_eq!(clip.id().as_str(), "test.audio.beep");
        assert_eq!(clip.byte_len(), bytes.len());
        assert!(clip.source_path().is_none());
    }

    #[test]
    fn valid_wav_probe_is_device_independent() {
        let clip = AudioClip::from_bytes("test.audio.probe", tiny_wav_bytes()).unwrap();
        let probe = clip.probe().unwrap();

        assert_eq!(probe.clip_id.as_str(), "test.audio.probe");
        assert_eq!(probe.channels, 1);
        assert_eq!(probe.sample_rate, 8_000);
        assert!(probe.byte_len > 44);
        assert_eq!(probe.duration, Some(Duration::from_millis(4)));
    }

    #[test]
    fn invalid_bytes_report_decode_error_without_device() {
        let clip = AudioClip::from_bytes("test.audio.invalid", vec![0, 1, 2, 3]).unwrap();

        assert!(matches!(
            clip.probe(),
            Err(AudioError::Decode { clip_id, .. }) if clip_id.as_str() == "test.audio.invalid"
        ));
    }

    #[test]
    fn missing_file_reports_read_error() {
        let path = std::env::temp_dir().join(format!(
            "engine-audio-missing-{}-{}.wav",
            std::process::id(),
            unique_suffix()
        ));

        let err = AudioClip::load("test.audio.missing", &path).unwrap_err();

        assert!(matches!(err, AudioError::Read { .. }));
    }

    #[test]
    fn load_from_file_preserves_source_path_and_probes() {
        let path = std::env::temp_dir().join(format!(
            "engine-audio-valid-{}-{}.wav",
            std::process::id(),
            unique_suffix()
        ));
        fs::write(&path, tiny_wav_bytes()).unwrap();

        let clip = AudioClip::load("test.audio.file", &path).unwrap();

        assert_eq!(clip.source_path(), Some(path.as_path()));
        assert_eq!(clip.probe().unwrap().sample_rate, 8_000);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_clip_id_is_typed() {
        let err = AudioClip::from_bytes("bad clip id", tiny_wav_bytes()).unwrap_err();

        assert!(matches!(err, AudioError::InvalidClipId { .. }));
    }

    #[test]
    fn playback_options_clamp_volume_without_device() {
        assert_eq!(PlaybackOptions::new(-1.0).volume(), 0.0);
        assert_eq!(PlaybackOptions::new(0.25).volume(), 0.25);
        assert_eq!(PlaybackOptions::new(3.0).volume(), 1.0);
        assert_eq!(PlaybackOptions::new(f32::NAN).volume(), 1.0);
    }

    #[test]
    fn master_volume_setter_getter_round_trip() {
        let mut engine = AudioEngine::new().unwrap();
        assert_eq!(engine.master_volume(), 1.0);
        engine.set_master_volume(0.5);
        assert_eq!(engine.master_volume(), 0.5);
        engine.set_master_volume(-1.0);
        assert_eq!(engine.master_volume(), 0.0);
        engine.set_master_volume(3.0);
        assert_eq!(engine.master_volume(), 1.0);
        engine.set_master_volume(f32::NAN);
        assert_eq!(engine.master_volume(), 1.0);
    }

    #[test]
    #[ignore = "manual smoke: set ENGINE_AUDIO_DEVICE_SMOKE=1 to open the default output device"]
    fn device_playback_smoke_is_manual_and_gated() {
        if std::env::var("ENGINE_AUDIO_DEVICE_SMOKE").as_deref() != Ok("1") {
            return;
        }

        let engine = AudioEngine::new().unwrap();
        let clip = AudioClip::from_bytes("test.audio.device_smoke", tiny_wav_bytes()).unwrap();
        let handle = engine
            .play_with_options(&clip, PlaybackOptions::new(0.05))
            .unwrap();
        handle.stop();
    }

    fn tiny_wav_bytes() -> Vec<u8> {
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
        bytes.extend_from_slice(&1u16.to_le_bytes());
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

    fn unique_suffix() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
