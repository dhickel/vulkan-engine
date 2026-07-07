# Sprint 07 Phase 03 Report Draft

## Changed Files

- `apps/dungeon_dogfood/Cargo.toml`: added the `audio` crate dependency for the app-boundary proof.
- `apps/dungeon_dogfood/assets/content_pack.toml`: added a packaged `dogfood.audio.startup_ping` audio clip reference.
- `apps/dungeon_dogfood/assets/content_manifest.md`: recorded the internal/generated audio fixture provenance.
- `apps/dungeon_dogfood/assets/audio/startup_ping.wav`: tiny internal generated WAV fixture for probe/smoke proof.
- `apps/dungeon_dogfood/src/content.rs`: added optional strict `audio_clips` metadata parsing and validation.
- `apps/dungeon_dogfood/src/audio_bridge.rs`: added packaged clip reference reporting, opt-in device smoke/probe, audio outcome to `AudioEvent` mapping, and tests.
- `apps/dungeon_dogfood/src/events.rs`: logs dogfood audio events.
- `apps/dungeon_dogfood/src/main.rs`: wires passive startup probe and `--audio-smoke` / `DUNGEON_DOGFOOD_AUDIO_SMOKE=1` opt-in.

## Validation Commands

```bash
cargo fmt --check
```

Pass.

```bash
cargo test -p audio
```

Pass: 7 passed, 1 ignored. The ignored test is the existing manual audio-device smoke gate.

```bash
cargo test -p engine_events
```

Pass: 7 passed.

```bash
cargo check -p dungeon_dogfood
```

Pass with existing renderer/dogfood dead-code warning noise.

```bash
cargo test -p dungeon_dogfood
```

Blocked before dogfood tests ran by an existing renderer test-profile compile failure in `src/renderer/src/data/assimp_util.rs`: `russimp_sys` `aiScene`/`aiNode` fields such as `mRootNode`, `mNumMaterials`, and `mNumMeshes` are unavailable in the current generated bindings. This is outside the Phase 03 editable target set.

```bash
cargo check
```

Pass with existing renderer dead-code warning noise.

## Device Smoke Status

`device_smoke_status`: skipped by default.

Normal dogfood startup parses and reports the packaged clip reference but does not load, probe, or open the default audio device. Device-backed probe/playback is attempted only with:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

or `DUNGEON_DOGFOOD_AUDIO_SMOKE=1`.

The worker did not run the opt-in device smoke. No default audio output device was opened by the validation commands above.
