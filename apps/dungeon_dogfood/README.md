# Dungeon Dogfood

`dungeon_dogfood` is the alpha demo game for the renderer facade. The app stays intentionally small so it is easy to read as an example of engine usage rather than an abstraction exercise.

## Application Path

`dungeon_dogfood` is currently a custom Rust app crate, not a project-manifest migration target. It owns its control flow, scene seeding, generator selection, and renderer facade usage directly:

```bash
cargo run -p dungeon_dogfood
```

Use the root `engine` launcher for data-driven project manifests such as the editor sample project. Use app crates under `apps/<name>` when custom Rust behavior is required. Dynamic Rust hot reload, scripting, broad physics/collision gameplay migration, production audio gameplay integration, generated app templates, and migrating dogfood to project manifests are deferred roadmap work. The current app does include alpha event logging, bespoke collision, and an opt-in audio smoke proof.

## Audio Smoke

The content pack declares `dogfood.audio.startup_ping` as an internal/generated WAV fixture. Normal startup validates the metadata and fixture path but does not load, probe, play, or open an audio output device.

Opt-in audio smoke:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

The same gate can be enabled with `DUNGEON_DOGFOOD_AUDIO_SMOKE=1`. Device availability is host-dependent.

## Visual Lock Baseline

Phase 07 keeps the current presentation in one app-owned code location: [`scene_seed.rs`](/home/hickelpickle/Code/Rust/engine/apps/dungeon_dogfood/src/scene_seed.rs).

- exposure: `2.8`
- gamma: `2.2`
- IBL ambient scale: `0.45`
- point light intensity baseline: `30.0`
- point light range baseline: `6.0`

The app passes the exposure, gamma, and IBL values through `RendererConfig::visual_tuning`, so the demo owns its look without changing renderer-wide defaults.

## Procedural Dungeon Default

The default selector now boots a generated sprawling multi-level dungeon instead of the old authored intro map.

- default selector: `generated_sprawl`
- generator env knobs: `DUNGEON_DOGFOOD_GENERATOR_SEED`, `DUNGEON_DOGFOOD_GENERATOR_WIDTH`, `DUNGEON_DOGFOOD_GENERATOR_HEIGHT`, `DUNGEON_DOGFOOD_GENERATOR_LAYERS`
- authored validation maps remain available through `--level level_01`, `--level level_02_ramps`, and `--level level_03_lighting`

## Known Visual Compromise

The torch prop remains an intentional fallback marker prop and not the PBR quality target for this demo. It is acceptable for alpha because it makes light positions readable, but wall, floor, crate, and environment assets are still the reference when judging overall visual quality.

## Validation

Use full content when checking the locked visual baseline:

```bash
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
cargo run -p dungeon_dogfood -- --level generated_sprawl
```

Repeat the same command with `level_02_ramps` and `level_03_lighting`.
