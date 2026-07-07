# Preflight Validation Drift

Date: 2026-07-07

## Commands Run

```sh
cargo check -p renderer --examples
cargo check -p dungeon_dogfood
git status --short
rg -n "set_camera_look_at|pub fn look_at|CameraLookAtError|fn set_camera" src/renderer/src apps docs .internal-dev/specifications
cargo check -p audio
cargo clean -p audio
cargo test -p renderer camera -- --nocapture
git diff --stat
```

## Results

`cargo check -p renderer --examples` failed before implementation work:

- `src/renderer/examples/capture_tests/common.rs` calls `renderer.set_camera_look_at(eye, target, Vec3::Y)`.
- `src/renderer/src/data/camera.rs` has internal `Camera::look_at(...)` support and tests.
- `src/renderer/src/api/renderer.rs` exposes `set_camera_position(...)` but no public `set_camera_look_at(...)`.
- `.internal-dev/specifications/api.md` still documents `Renderer::set_camera_look_at(...)` as active intended truth.

`cargo check -p dungeon_dogfood` failed before implementation work:

- `apps/dungeon_dogfood/src/audio_bridge.rs` imports `audio::{AudioClip, AudioEngine, PlaybackOptions}`.
- The compiler reports `can't find crate for audio`, followed by type inference errors caused by the missing crate.
- `apps/dungeon_dogfood/Cargo.toml` does include `audio = { path = "../../src/audio" }`.
- `src/audio/Cargo.toml` declares package name `audio`.
- `cargo check -p audio` passes independently.
- A rerun of `cargo check -p dungeon_dogfood` after `cargo check -p audio` still fails with `can't find crate for audio`.

Phase 00 follow-up:

- Restored `Renderer::set_camera_look_at(eye, target, up)` in `src/renderer/src/api/renderer.rs` as a narrow compatibility facade over `Camera::look_at`.
- Invalid look-at inputs now surface as `RendererError::InvalidState` with the existing `CameraLookAtError` message. `Camera::look_at` validates all inputs before mutating camera state, so invalid inputs preserve the prior camera position and orientation.
- `cargo metadata --format-version 1 --no-deps` showed `dungeon_dogfood` has a normal path dependency on `audio`; `cargo check -p dungeon_dogfood -vv` showed rustc receiving `--extern audio=/home/hickelpickle/Code/Rust/engine/target/debug/deps/libaudio-032a7939a230f381.rmeta` while still reporting `can't find crate for audio`.
- The selected audio artifact was stale/corrupt target metadata, not missing source wiring. `cargo clean -p audio` removed package build artifacts and a subsequent `cargo check -p dungeon_dogfood` passed.

Post-repair validation:

- `cargo check -p audio` passed.
- `cargo test -p renderer camera -- --nocapture` passed.
- `cargo check -p renderer --examples` passed.
- `cargo check -p dungeon_dogfood` passed.
- `git diff --stat` showed only `src/renderer/src/api/renderer.rs` in the tracked source diff; this ignored `.internal-dev` research note was updated directly.

`git status --short` returned no tracked output before this plan artifact work.

## Planning Impact

The advanced plan must treat these as pre-existing validation drift:

- The renderer example compile gate is clean after restoring `Renderer::set_camera_look_at`.
- The dungeon dogfood compile gate is clean after quarantining the stale target-state cause with `cargo clean -p audio`; no dogfood source or manifest change was required.
- Future validation can treat these gates as clean from this point, while noting that the original audio failure was pre-existing build-artifact drift rather than runtime abstraction refactor fallout.
