# Current State Analysis

## Verified Facts

- Workspace includes `src/audio`, `src/events`, `src/renderer`, `tools/engine_pack`, `apps/dungeon_dogfood`, and `apps/editor`.
- `src/audio/src/lib.rs` currently wraps `rodio` with `AudioClip::load`, `AudioClip::from_bytes`, `AudioEngine::new`, `AudioEngine::play`, and `PlaybackHandle` controls for volume/pause/play/stop/is_playing.
- `AudioEngine::new` opens the default output stream immediately, and the existing `engine_initialization` test asserts that this succeeds. That is device-dependent and unsuitable as a required CI-like test.
- `src/events/src/lib.rs` already defines durable `AudioClipId` and `AudioEvent::{ClipStarted, ClipStopped, ClipFinished, ClipFailed}` while remaining independent from audio/renderer/editor/dogfood.
- Package validation in `src/renderer/src/data/asset_registry.rs` owns durable asset records, package/project validation options, source-file checks, runtime-handle diagnostics, and collision metadata validation.
- `AssetKind` currently includes model/texture/material/environment/prefab/wall_chunk/scene_fragment. Audio support likely needs a new package asset kind such as `audio` or `audio_clip`.
- Scene validation in `src/renderer/src/api/scene.rs` owns versioned scene JSON validation and currently has typed collision components on serialized nodes. Audio references can follow the same durable-ID validation style but must not claim editor placement UI unless implemented.
- `tools/engine_pack/src/main.rs` routes package/project/scene validation through renderer validation APIs. Sprint 07 should preserve that single validation source.
- `apps/dungeon_dogfood` currently depends on renderer, input, glam, log, env_logger, thiserror, winit, serde, and toml. It does not currently depend on `audio`.

## Architecture Fit

- The audio crate should be the runtime playback facade and device boundary, not the package/scene schema owner.
- The renderer validation modules currently own package and scene schemas because prior sprints put durable project/package/scene contracts there. Extending that validation is consistent for Sprint 07 even though audio itself should remain renderer-independent.
- `engine_events` should remain vocabulary-only. Audio or app code can translate playback lifecycle into existing `AudioEvent` values without inverting dependencies.
- Dogfood should be a consumer proof only. It should not become the owner of audio API design.

## Gaps To Close

- Device-independent audio decode/probe/metadata tests are missing.
- Supported audio formats and metadata validation are not represented in package manifests.
- Scene/app references to audio clips are not represented or validated.
- CLI validation does not report audio-specific metadata failures.
- No sample or dogfood path shows a packaged audio clip reference and opt-in playback attempt.
- Docs do not yet explain the alpha audio contract or device-dependent limitations.

## Validation Blind Spots

- Opening a default audio device in unit tests can fail on headless hosts and should not be treated as core correctness.
- A decode-only test can prove bytes/format handling but not actual speaker playback.
- Rodio sinks may run background threads; validators should check handles are dropped/stopped and no obvious hang/leak is introduced.
- CLI/package validation can pass while dogfood references stale content paths unless the sample path is checked through project/package validation.
- Docs can easily overclaim support for spatial audio because the current crate comment mentions spatial positioning even though no such API is implemented.

## Protected State

- Preserve unrelated `.idea/engine.iml`.
- Preserve unrelated `.reasonix/`.
