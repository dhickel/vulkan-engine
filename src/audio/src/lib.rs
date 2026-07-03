//! Audio subsystem — spatial audio playback via rodio.
//!
//! Provides `AudioEngine` for loading and playing audio clips,
//! with volume control and spatial positioning support.

use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// An audio clip loaded from a file.
pub struct AudioClip {
    data: Vec<u8>,
}

impl AudioClip {
    /// Load an audio clip from a file (WAV, MP3, FLAC, OGG).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let data =
            std::fs::read(path.as_ref()).map_err(|e| format!("failed to read audio file: {e}"))?;
        Ok(Self { data })
    }

    /// Create a clip from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }
}

/// The audio engine — owns the output stream and manages playback.
pub struct AudioEngine {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
}

impl AudioEngine {
    /// Initialize the audio engine.
    pub fn new() -> Result<Self, String> {
        let (stream, stream_handle) =
            OutputStream::try_default().map_err(|e| format!("failed to open audio stream: {e}"))?;
        Ok(Self {
            _stream: stream,
            stream_handle,
        })
    }

    /// Play an audio clip. Returns a handle that can be used to control playback.
    pub fn play(&self, clip: &AudioClip) -> Result<PlaybackHandle, String> {
        let cursor = std::io::Cursor::new(clip.data.clone());
        let source = rodio::Decoder::new(BufReader::new(cursor))
            .map_err(|e| format!("failed to decode audio: {e}"))?;
        let sink = Sink::try_new(&self.stream_handle)
            .map_err(|e| format!("failed to create audio sink: {e}"))?;
        sink.append(source);
        Ok(PlaybackHandle { sink })
    }

    /// Get the master volume (0.0 - 1.0). Not directly supported by rodio sinks;
    /// returns a fixed 1.0 for now.
    pub fn master_volume(&self) -> f32 {
        1.0
    }
}

/// Handle to an active audio playback. Drop to stop.
pub struct PlaybackHandle {
    sink: Sink,
}

impl PlaybackHandle {
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 1.0));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_initialization() {
        let engine = AudioEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn clip_from_bytes() {
        let clip = AudioClip::from_bytes(vec![0u8; 1024]);
        assert_eq!(clip.data.len(), 1024);
    }
}
