# Audio Foundation

## 1. Purpose & Audience

This chapter documents the alpha audio contract for packaged clips, scene/app references, device-independent loading/probing, optional playback, and event reporting.

Use this when you need to declare audio in package data, validate scene references, or add an app-owned opt-in playback proof. This is not a production mixer, spatial audio system, streaming system, editor authoring UI, or platform support matrix.

## 2. Package Metadata

Package manifests may declare audio assets with durable IDs:

```toml
[[assets]]
id = "core.audio.pickup"
kind = "audio"
path = "audio/pickup.ogg"

[assets.metadata.audio]
format = "ogg"
usage = "effect"
volume = 0.75
default_gain = 1.0
```

Validated audio formats are `wav`, `ogg`, `flac`, and `mp3`. Valid usage values are `effect`, `music`, `ambient`, `voice`, and `ui`. `volume` and `default_gain` must be positive finite numbers when present.

Audio IDs use authored durable identity. Valid audio IDs are non-empty ASCII alphanumeric strings with `.`, `_`, or `-`. Runtime-handle-shaped strings containing both `slot` and `generation` are rejected in package and scene validation.

## 3. Scene References

Scene files may include a top-level `audio` array:

```json
{
  "audio": [
    {
      "id": "scene.audio.pickup",
      "clip": { "id": "core.audio.pickup", "path_hint": "audio/pickup.ogg" },
      "trigger": "startup",
      "usage": "effect",
      "volume": 0.5,
      "default_gain": 1.0
    }
  ]
}
```

`clip.id` is required and is checked against known package asset IDs when validation has project context. `path_hint` is diagnostic/fallback data and is not the durable identity.

Scene validation rejects duplicate audio reference IDs, missing clip IDs, unknown clip IDs when a project registry is supplied, invalid usage/gain values, and serialized runtime handles.

## 4. Loading And Optional Playback

The `audio` crate is renderer-independent. `AudioClip::load` and `AudioClip::from_bytes` keep durable `AudioClipId` identity and can probe decode metadata without opening an output device:

```rust
use audio::AudioClip;

let clip = AudioClip::load("core.audio.pickup", "assets/audio/pickup.ogg")?;
let probe = clip.probe()?;
```

Device-backed playback is explicit:

```rust
use audio::{AudioEngine, PlaybackOptions};

let engine = AudioEngine::new()?;
let handle = engine.play_with_options(&clip, PlaybackOptions::new(0.75))?;
handle.stop();
```

Opening `AudioEngine` depends on the host default output device and may fail on headless machines. Core tests do not require a device.

## 5. Dogfood Proof

`apps/dungeon_dogfood` declares `dogfood.audio.startup_ping` in its content pack and ships a tiny internal/generated WAV fixture for the Sprint 07 proof.

Normal dogfood startup parses and validates the configured audio metadata, but it does not load, probe, play, or open an audio device. The app-owned opt-in smoke path is:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

The same path can be enabled with `DUNGEON_DOGFOOD_AUDIO_SMOKE=1`.

## 6. Audio Events

The event vocabulary lives in `engine_events` and is re-exported by `renderer`:

- `AudioEvent::ClipStarted`
- `AudioEvent::ClipStopped`
- `AudioEvent::ClipFinished`
- `AudioEvent::ClipFailed`

`engine_events` does not depend on `audio`. App code owns the bridge from audio runtime outcomes to `EngineEvent::Audio(...)`. `apps/dungeon_dogfood/src/audio_bridge.rs` is the current example.

## 7. Current Limits

- No editor audio placement UI.
- No spatialization, mixer buses, DSP, streaming, occlusion, or device matrix.
- No guarantee that a default audio output device exists.
- No automatic root-runtime audio playback from project manifests.
- `cargo test -p dungeon_dogfood` is currently blocked by a renderer test-profile `russimp_sys` binding issue before dogfood tests run; `cargo check -p dungeon_dogfood` remains the validated dogfood compile gate for this sprint.

## 8. See Also

- [Packaging CLI](10-packaging-cli.md)
- [Events and Lifecycle](12-events-and-lifecycle.md)
- [Runtime Project Launcher](11-runtime-project-launcher.md)
- [Internal Audio Foundation](../internal/12-audio-foundation.md)
