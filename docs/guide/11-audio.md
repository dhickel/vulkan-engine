# 11 — Audio

> Provenance: `G-11`

> **Alpha Status**: The `audio` crate is alpha-stage. Core clip loading and device-backed playback work, but spatial audio, 3D positioning, channel groups, mixing, streaming, and editor integration are not yet implemented. This chapter documents what exists today without production-readiness promises.

This chapter covers the renderer-independent audio crate: loading audio clips, probing metadata without a device, opening the output device for playback, controlling volume, and bridging audio outcomes into the event bus.

For the full API reference, see [`src/audio/src/lib.rs`](../../src/audio/src/lib.rs). The crate uses `rodio` internally for decoding and device output.

## Architecture

The `audio` crate is **renderer-independent**. It depends only on `engine_events` (for `AudioClipId`) and `rodio` (wrapped internally). It does not depend on the renderer, Vulkan, windowing, or any other engine crate.

```
┌──────────────────────────────────────┐
│          AudioClip                   │
│  (device-independent, load/probe)    │
│                                      │
│  AudioClip::load(path) ──► bytes     │
│  AudioClip::from_bytes(data)         │
│  clip.probe() ──► ClipProbe         │
│  clip.id() ──► AudioClipId          │
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│          AudioEngine                 │
│  (device-backed playback)            │
│                                      │
│  AudioEngine::new() ──► device       │
│  engine.play(&clip) ──► PlaybackHandle│
│  engine.set_master_volume(v)        │
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Event Bridging (external)           │
│  (app-owned, in dogfood audio_bridge)│
│                                      │
│  EngineEvent::Audio(ClipStarted)     │
│  EngineEvent::Audio(ClipFinished)    │
└──────────────────────────────────────┘
```

Rodio types are never exposed in the public API. The event bridge lives in the app layer (`apps/dungeon_dogfood/src/audio_bridge.rs`), not in the audio crate — the crate depends on `engine_events` only for the canonical `AudioClipId` identity type and conversion into event payload IDs.

## Key Types

| Type | Purpose |
|------|---------|
| `AudioClipId` | Durable string ID validated for alphanumeric + `.`, `_`, `-` |
| `AudioClip` | Decoded in-memory audio with metadata, loadable without a device |
| `ClipProbe` | Device-independent metadata: channels, sample rate, duration |
| `AudioEngine` | Opens host output device, creates playback sinks |
| `PlaybackHandle` | Controls an active playback: play, pause, stop, volume |
| `PlaybackOptions` | Per-clip playback settings (volume, clamped 0.0–1.0) |
| `AudioError` | Typed error: `InvalidClipId`, `Read`, `Decode`, `Device`, `Playback` |

## Loading Audio Clips (Device-Independent)

> Provenance: `G-11-LOAD` — Excerpt

Clips can be loaded and probed without opening an audio output device. This makes them suitable for headless validation, asset scanning, and server-side workflows.

### From a File

```rust
use audio::AudioClip;

let clip = AudioClip::load(
    "dogfood.audio.pickup",          // durable clip ID
    "apps/dungeon_dogfood/assets/audio/pickup.wav",
)?;
```

Supported formats: WAV, OGG, FLAC, and MP3 (via `rodio`'s decoder stack).

### From Bytes

```rust
let clip = AudioClip::from_bytes("my_app.audio.beep", my_audio_bytes)?;
```

### Clip Metadata

```rust
println!("clip id: {}", clip.id().as_str());
println!("byte length: {}", clip.byte_len());
println!("is empty: {}", clip.is_empty());

if let Some(path) = clip.source_path() {
    println!("loaded from: {}", path.display());
}
```

## Probing Clips (No Device Required)

> Provenance: `G-11-PROBE` — Excerpt

```rust
let probe = clip.probe()?;

println!("channels: {}", probe.channels);
println!("sample rate: {}", probe.sample_rate);
println!("byte length: {}", probe.byte_len);
if let Some(duration) = probe.duration {
    println!("duration: {:.2}s", duration.as_secs_f32());
}
```

`probe()` decodes the audio data enough to extract metadata but does not open an output device. This is the preferred way to validate audio assets in headless or build-pipeline contexts.

### Decode Errors

Invalid or corrupt audio data returns an `AudioError::Decode` without touching the device:

```rust
match clip.probe() {
    Err(AudioError::Decode { clip_id, message }) => {
        eprintln!("clip {clip_id} failed to decode: {message}");
    }
    // ...
}
```

## Device-Backed Playback

> Provenance: `G-11-PLAYBACK` — Excerpt

Playback requires an `AudioEngine`, which opens the host's default output device:

```rust
use audio::AudioEngine;

let engine = AudioEngine::new()?;
```

If the default output device is unavailable (no speakers, no audio driver, device in use), `AudioEngine::new()` returns `AudioError::Device { message }`.

### Playing a Clip

```rust
use audio::PlaybackOptions;

// Play with default options (volume 1.0)
let handle = engine.play(&clip)?;

// Play with explicit volume
let handle = engine.play_with_options(
    &clip,
    PlaybackOptions::new(0.5), // 50% volume
)?;
```

### Controlling Playback

```rust
handle.pause();
handle.play();   // resume after pause
handle.stop();   // stop permanently

if handle.is_playing() {
    // ...
}

// Adjust per-sink volume
handle.set_volume(0.8);
let current_vol = handle.volume();
```

### Master Volume

```rust
engine.set_master_volume(0.75);
assert_eq!(engine.master_volume(), 0.75);

// Volume is clamped to [0.0, 1.0]:
engine.set_master_volume(2.0);   // → 1.0
engine.set_master_volume(-0.5);  // → 0.0
engine.set_master_volume(f32::NAN); // → 1.0
```

## Audio Clip ID Validation

`AudioClipId` enforces a naming convention:

- Non-empty
- Only ASCII alphanumeric, `.`, `_`, and `-`

Invalid IDs return `AudioError::InvalidClipId`:

```rust
// Valid:
AudioClipId::new("dogfood.audio.pickup")?;

// Invalid (space):
AudioClipId::new("bad clip id")?; // → AudioError::InvalidClipId
```

## Event Bridging (External)

The audio crate does **not** emit `EngineEvent::Audio` events. Event bridging is external — your app is responsible for emitting audio lifecycle events at safe boundaries.

The dogfood app demonstrates the pattern in `apps/dungeon_dogfood/src/audio_bridge.rs`:

```rust
use engine_events::{EngineEvent, AudioEvent, EventBus, EventStage};

fn on_clip_started(events: &mut EventBus, clip_id: AudioClipId) {
    events.emit(
        EventStage::PostUpdate,
        None,
        EngineEvent::Audio(AudioEvent::ClipStarted { clip: clip_id.into() }),
    );
}
```

The `AudioClipId` in the audio crate wraps `engine_events::AudioClipId`. Convert between them with `.into()`:

```rust
let crate_id: audio::AudioClipId = AudioClipId::new("my.clip")?;
let event_id: engine_events::AudioClipId = crate_id.into();
```

## What's Not Yet Implemented

- **Spatial audio / 3D positioning**: No HRTF, distance attenuation, or directionality
- **Channel groups / submixing**: Single master volume only
- **Streaming**: All clips are fully decoded into memory before playback
- **Mixing bus**: No per-category or per-instance audio mixing
- **Editor integration**: No editor asset browser preview or timeline attachment
- **Root runtime ownership**: The root `engine` runtime does not own an `AudioEngine`; apps must create their own

These are deferred to future sprint tracks.

## Error Handling

| Variant | When |
|---------|------|
| `InvalidClipId { id }` | Clip ID contains unsupported characters or is empty |
| `Read { path, source }` | File read failed (IO error) |
| `Decode { clip_id, message }` | Audio bytes could not be decoded |
| `Device { message }` | Host output device could not be opened |
| `Playback { clip_id, message }` | Sink creation or playback failed |

## Runnable Verification

Run the audio crate test suite:

```sh
cargo test -p audio
```

Expected: all non-ignored tests pass (clip identity, device-independent probe, invalid bytes decode error, missing file read error, load-from-file path preservation, invalid clip ID, volume clamping, master volume round-trip).

Build the audio crate standalone:

```sh
cargo check -p audio
```

### Device Smoke Test (Manual)

The audio crate includes a `#[ignore]`-gated device playback smoketest. To run it on a system with an audio output device:

```sh
ENGINE_AUDIO_DEVICE_SMOKE=1 cargo test -p audio -- device_playback --nocapture
```

This test is **device-dependent** and is skipped by default (ignored). It validates that the default output device can be opened and a short synthesized WAV plays at low volume.

## Working with Dogfood Assets

The dogfood app (`apps/dungeon_dogfood/`) provides a complete example of audio integration, including audio assets under `apps/dungeon_dogfood/assets/audio/`. Examine its audio bridge and main loop for the recommended integration pattern.

## Next

Continue to [12 — Debug & Diagnostics](12-debug-and-diagnostics.md) to learn about logging, timing capture, headless frame capture, validation layers, and diagnostic tooling.
