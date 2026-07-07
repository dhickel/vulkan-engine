# Implementation Notes

## Direct Targets

- Phase 01 owns `src/audio`.
- Phase 02 owns package/scene validation and `engine_pack` CLI validation.
- Phase 03 owns event translation and sample/dogfood proof.
- Phase 04 owns docs, final evidence, and closeout artifacts.

## Suggested Audio API Properties

- `AudioClip` should expose durable ID where appropriate, bytes length, and/or decoded metadata without requiring playback.
- `AudioEngine::new` may remain device-backed, but normal tests should avoid it or use a clearly named `try_default_device` path.
- Playback errors should be typed enough for docs/tests to distinguish read/decode/device/sink failures.
- `PlaybackHandle` should provide stop/pause/play/volume only as runtime control.

## Suggested Validation Additions

- Add `AssetKind::Audio` or equivalent.
- Validate audio metadata under existing package metadata maps before accepting package records.
- Check declared format values against a small supported set such as `wav`, `ogg`, `mp3`, `flac`, matching what rodio can decode in this repo context.
- Add scene-level or node-level audio references only if validation can reject blank/unknown/runtime-handle-shaped clip IDs.
- Keep old files without audio data valid.

## Reporting Expectations

- Each phase should provide a concise phase email/report draft under `reports/phase-XX-email.md`.
- The main thread owns actually sending email and pushing commits.
- If dogfood proof is deferred, produce `reports/dogfood-audio-proof-debt.md`.
- Do not use `/tmp` as authoritative final evidence; use this sprint directory or `.internal-dev/debug_reports/`.
