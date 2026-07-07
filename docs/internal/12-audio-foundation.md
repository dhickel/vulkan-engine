# Audio Foundation Internals

## 1. Purpose & Audience

This page is for contributors changing the alpha audio crate, package/scene audio validation, or app-owned audio event bridges.

The current implementation separates authored data validation, renderer-independent clip loading/probing, and optional device-backed playback. Do not collapse those boundaries for convenience.

## 2. Boundary Map

```text
package manifests / scene files
  -> renderer validators
      -> durable audio metadata diagnostics

audio crate
  -> clip identity, encoded bytes, decode probe, explicit device playback
  -> no renderer, Vulkan, windowing, editor, or dogfood dependency

app crates
  -> audio crate
  -> renderer facade event bus
  -> app-owned mapping into EngineEvent::Audio
```

`engine_events` owns only typed event vocabulary. It must not depend on `audio`, `renderer`, dogfood, editor, physics, or scripting crates.

## 3. Implemented Surfaces

- `src/audio/src/lib.rs` defines `AudioClipId`, `AudioClip`, `ClipProbe`, `PlaybackOptions`, `AudioEngine`, `PlaybackHandle`, and typed `AudioError` values.
- `src/renderer/src/data/asset_registry.rs` validates package `kind = "audio"` and `[assets.metadata.audio]`.
- `src/renderer/src/api/scene.rs` validates top-level scene `audio[]` references.
- `tools/engine_pack` recognizes audio file extensions and surfaces audio diagnostics through existing CLI validation flows.
- `apps/dungeon_dogfood/src/audio_bridge.rs` demonstrates app-owned mapping from audio outcomes to `AudioEvent`.

## 4. Device Boundary

Clip loading and probing must remain device-independent. `AudioEngine::new` is the explicit boundary that may open the host default output stream. Unit tests should cover ID validation, byte loading, decode/probe behavior, and playback options without opening a device.

Manual device smoke is allowed only behind an explicit gate:

- `ENGINE_AUDIO_DEVICE_SMOKE=1` for the ignored audio crate smoke test.
- `--audio-smoke` or `DUNGEON_DOGFOOD_AUDIO_SMOKE=1` for the dogfood proof.

## 5. Validation Rules

Package and scene audio IDs use the same character set as `AudioClipId`: non-empty ASCII alphanumeric plus `.`, `_`, or `-`. Renderer validators also reject strings containing both `slot` and `generation` as runtime-handle-shaped identities.

Supported audio formats: `wav`, `ogg`, `flac`, `mp3`.

Supported usage values: `effect`, `music`, `ambient`, `voice`, `ui`.

Gain fields must be positive finite numbers.

## 6. Event Bridge

Use existing `AudioEvent` variants unless a future plan proves the vocabulary is insufficient. Keep mapping at the consumer boundary:

```rust
EngineEvent::Audio(AudioEvent::ClipStarted { clip })
```

> **Compatibility note:** The dogfood bridge emits through `renderer.events_mut()` because dogfood already owns the renderer facade. This uses the **renderer compatibility path** `EventBus`. For app-owned paths, prefer emitting into a caller-owned `EventBus` via `engine::events::runtime_event_bus()`. Do not make `engine_events` aware of `rodio`, `AudioEngine`, `PlaybackHandle`, or dogfood content types.

## 7. Validation Commands

Core audio sprint checks:

```sh
cargo fmt --check
cargo check
cargo test -p audio
cargo check -p audio
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

`cargo test -p dungeon_dogfood` is expected to run when the renderer test-profile binding issue is resolved. During Sprint 07 it was blocked before dogfood tests executed by `russimp_sys` `aiScene`/`aiNode` field mismatches in `src/renderer/src/data/assimp_util.rs`.

Use true headless draw-target capture only if an audio change affects visible renderer or editor behavior. Audio metadata, event bridge, and docs changes do not need image evidence.

## 8. Deferred Work

- Editor placement/authoring UI for audio.
- Runtime project-manifest audio playback.
- Mixer buses, spatialization, streaming, DSP, occlusion, fades, and device matrix.
- Broader dogfood gameplay audio integration.
- Asset import/reimport pipeline and thumbnails for audio assets.

## 9. Cross-Module Links

- Public docs: `docs/api/13-audio-foundation.md`
- Audio crate: `src/audio/src/lib.rs`
- Event crate: `src/events/src/lib.rs`
- Package validation: `src/renderer/src/data/asset_registry.rs`
- Scene validation: `src/renderer/src/api/scene.rs`
- Dogfood bridge: `apps/dungeon_dogfood/src/audio_bridge.rs`
