# Sprint 07 Phase 02 Package/Scene Audio Metadata Report

## Summary

Implemented durable package and scene audio metadata validation without touching runtime audio playback, output devices, `rodio`, editor UI, dogfood, events, or `src/audio`.

## Accepted Schema

Package manifests now accept:

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

- `kind = "audio"` is a first-class asset kind.
- Audio asset IDs use the same authored ID character set as `AudioClipId`: non-empty ASCII alphanumeric plus `.`, `_`, or `-`, with runtime-handle-shaped strings rejected.
- `metadata.audio` is optional.
- Optional `metadata.audio.format` accepts `wav`, `ogg`, `flac`, or `mp3`.
- Optional `metadata.audio.usage` accepts `effect`, `music`, `ambient`, `voice`, or `ui`.
- Optional `metadata.audio.volume` and `metadata.audio.default_gain` must be positive finite numbers.
- Optional `metadata.audio.id` and `metadata.audio.clip_id` must use the same durable authored ID rule if present.
- Source-file checks continue to use the existing package asset `path`; no audio decode or device access is attempted.

Scene files now accept a top-level reference array:

```json
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
```

- `audio` is optional and defaults to empty for backward compatibility.
- `id` must be a durable scene audio reference ID.
- `clip.id` is required and must be a durable package asset ID.
- `clip.path_hint` is optional.
- `usage`, `volume`, and `default_gain` use the same validation rules as package metadata.
- If project context supplies known package asset IDs, `clip.id` is checked against them.

## Diagnostic Codes

Added:

- `asset.audio_invalid_id`
- `asset.audio_invalid_schema`
- `asset.audio_unsupported_format`
- `asset.audio_invalid_usage`
- `asset.audio_invalid_gain`
- `scene.audio_invalid_id`
- `scene.duplicate_audio_id`
- `scene.audio_missing_clip_id`
- `scene.unknown_audio_clip_id`
- `scene.audio_invalid_usage`
- `scene.audio_invalid_gain`

Existing runtime-handle diagnostics now recurse through audio metadata/references:

- `asset.runtime_handle_identity`
- `scene.runtime_handle_identity`

## Changed Files

- `src/renderer/src/data/asset_registry.rs`: added `AssetKind::Audio`, package audio metadata validation, durable ID/runtime-handle checks, and renderer tests.
- `src/renderer/src/api/scene.rs`: added top-level serialized scene audio references, validation for durable clip references, known-asset checks, runtime-handle recursion, and renderer tests.
- `tools/engine_pack/src/main.rs`: added audio extension scanning and `add-asset --kind audio` support.
- `tools/engine_pack/tests/cli_validation.rs`: added CLI validation coverage for audio package failures, project-backed unknown scene audio references, and audio scan output.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-07-audio-foundation/reports/phase-02-email.md`: this report.

## Validation

- `cargo fmt --check`: pass.
- `cargo test -p renderer`: pass, 160 unit tests, 17 integration tests, 5 ignored doctests. Existing renderer warning noise remains.
- `cargo test -p engine_pack`: pass, 17 CLI tests. Existing renderer warning noise remains.
- `cargo check -p engine_pack`: pass. Existing renderer warning noise remains.
- `cargo check`: pass. Existing renderer warning noise remains.

## Blockers And Risks

- No blockers.
- Runtime loading/playback is intentionally not implemented in this phase.
- Scene audio persistence is limited to the validated top-level schema and does not imply editor placement UI.
