# Audio Crate Agent Guide (`src/audio`)

Use this guide for audio asset identity, device-backed playback, and alpha audio integration.

## Crate Role

`audio` provides a renderer-independent alpha audio facade:

- Durable `AudioClipId` for authored clip identity
- `AudioClip` loading from file paths (WAV/OGG/FLAC via `rodio`)
- `AudioEngine` for device-backed playback with volume control
- `AudioError` with stable variants for validation and integration tests

## Public API

- `AudioClipId` — stable authored identity for an audio clip
- `AudioClip` — decoded in-memory audio with metadata
- `AudioEngine` — opens host output device, creates playback sinks
- `AudioError` — read/decode/device/playback error variants

## Architecture

- Clips have durable authored IDs and can be created without an output device
- Device-backed playback is explicit through `AudioEngine::play`
- Volume control is a single master f32 (fixed 1.0 for alpha)
- Uses `rodio` for decoding and device output
- Event bridging is external: the audio crate itself has no dependency on `engine_events`; all event bridging lives in `apps/dungeon_dogfood/src/audio_bridge.rs`

## Current Alpha Status

- Core clip loading and playback work
- Device behavior and error handling are thin
- No spatial audio, 3D positioning, or channel groups
- No editor attachment or asset pipeline integration
- Additional audio features deferred to future sprint track

## Working Rules

- Do not expose rodio types in the public API
- Keep clip IDs stable and comparable
- Validate clip IDs against the alphanumeric+dots+dashes pattern
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p audio`
- `cargo test -p audio`
