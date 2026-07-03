# Sprint 07 Phase 01 Audio Crate Alpha Contract

## Summary

Phase 01 is ready for validator review. The `audio` crate now exposes a small renderer-independent alpha facade with durable `AudioClipId` values, byte/path clip construction, device-independent decode probing, typed `AudioError` variants, explicit device-backed `AudioEngine` playback, and clamped `PlaybackOptions`.

Normal audio tests no longer open the default audio device. Device playback remains available through `AudioEngine::new()` and an ignored/manual smoke test gated by `ENGINE_AUDIO_DEVICE_SMOKE=1`.

## Changed Files

- `src/audio/src/lib.rs`: replaced the previous string-error/device-test wrapper with durable clip identity, typed errors, decode-only probing, playback options, explicit device playback, and device-independent tests.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-01-email.md`: phase report draft.

## Validation

- `cargo fmt --check`: passed.
- `cargo test -p audio`: passed, 7 passed, 1 ignored. The ignored test is the manual device playback smoke.
- `cargo check -p audio`: passed.
- `cargo check`: passed with existing renderer dead-code warnings; no audio errors.
- `cargo tree -p audio`: passed. Direct dependencies remain `rodio v0.19.0` and `log v0.4.28`; no renderer/window/editor/dogfood dependency.
- `cargo test -p audio device_playback_smoke_is_manual_and_gated -- --ignored`: passed with `ENGINE_AUDIO_DEVICE_SMOKE` unset, validating the gate without opening a device.
- `rg -n "renderer|ash|vulkan|winit|imgui|dungeon_dogfood|editor" src/audio`: no matches.
- `rg -n "spatial audio|spatial positioning|Serialize|Deserialize" src/audio`: no matches.

## Device Smoke Status

`device_smoke_status`: skipped by design. The manual smoke was not run with `ENGINE_AUDIO_DEVICE_SMOKE=1`, so no default audio device was opened during implementation validation.

## Notes

- The decode/probe tests generate tiny valid WAV bytes in memory instead of adding binary fixtures.
- Playback sink creation remains typed as `AudioError::Playback`; device stream open failures are typed as `AudioError::Device`; read/decode/invalid-ID failures have stable variants for tests and validators.
- No renderer, engine_pack, events, editor, dogfood, or public docs were edited in this phase.
