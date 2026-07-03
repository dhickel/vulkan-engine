# Sprint 07 Changelog: Audio Foundation

Date: 2026-07-03

Branch: `sprint/alpha-07-audio-foundation`

## Completed

- Reworked `src/audio` into a renderer-independent alpha audio facade with durable `AudioClipId`, `AudioClip`, `ClipProbe`, `PlaybackOptions`, explicit `AudioEngine`, typed errors, device-independent tests, and an ignored/manual device smoke gate.
- Added package audio support with `AssetKind::Audio`, `[assets.metadata.audio]` validation, supported format/usage/gain checks, durable ID checks, and runtime-handle rejection.
- Added scene-level audio reference validation with durable audio IDs, required `clip.id`, project-known asset lookup, duplicate detection, usage/gain validation, and runtime-handle rejection.
- Updated `engine_pack` to scan `.wav`, `.ogg`, `.flac`, and `.mp3` as audio assets and added CLI validation coverage for audio metadata and scene references.
- Added dogfood packaged audio metadata for `dogfood.audio.startup_ping`, a tiny internal/generated WAV fixture, and an app-owned `audio_bridge` that maps gated audio outcomes into `EngineEvent::Audio`.
- Documented the public and internal alpha audio contract, device boundary, event bridge, dogfood smoke command, and current limitations.
- Closed the sprint with phase validation reports, final quality review, validation summary, pushed phase/evidence commits, and phase/final report emails.

## Validation

- `cargo fmt --check`
- `cargo check`
- `cargo test -p audio`
- `cargo check -p audio`
- `cargo test -p engine_events`
- `cargo test -p renderer`
- `cargo test -p engine_pack`
- `cargo check -p renderer --examples`
- `cargo check -p editor`
- `cargo check -p dungeon_dogfood`
- `cargo test -p dungeon_dogfood` attempted and blocked before dogfood tests ran by an existing renderer test-profile `russimp_sys` binding issue.
- stale-reference sweep over public/internal docs and the Sprint 07 plan directory
- Phase 01 through Phase 04 validation reports
- Final quality review

Core checks passed. `cargo test -p dungeon_dogfood` remains an accepted residual because the failure occurs in renderer test-profile compilation before dogfood tests execute and is outside the Sprint 07 audio scope.

## Deferred

- Device smoke on this host; playback remains opt-in and host-output dependent.
- Root-runtime/project-manifest audio playback.
- Editor audio placement/authoring UI.
- Production mixer, spatialization, streaming, DSP, occlusion, fades, and platform/device matrix.
- Broad dogfood gameplay audio integration beyond the opt-in startup proof.
- Resolving the existing renderer test-profile `russimp_sys` binding issue that blocks `cargo test -p dungeon_dogfood`.
