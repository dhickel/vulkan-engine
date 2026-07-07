# Dungeon Dogfood

`dungeon_dogfood` is the alpha vertical slice demo for the engine renderer facade. The app stays intentionally small so it is easy to read as an example of engine usage rather than an abstraction exercise.

## Status (Sprint 11)

Dogfood now has canonical project/package/scene contracts validated through `engine_pack`, while keeping custom Rust gameplay in the app crate. The app-level `content_pack.toml` remains as transitional config for concepts not yet expressible in the engine data contracts (PBR material definitions, light presets, prop placement policies, audio format/usage metadata).

## Application Path

`dungeon_dogfood` is a custom Rust app crate. It owns its control flow, scene seeding, generator selection, collision, and renderer facade usage directly:

```bash
cargo run -p dungeon_dogfood -- --level generated_sprawl
```

Dogfood is also the real-app proof for the root `engine` app-owned loop helpers: it uses `engine::input::route_platform_input_to_app`, `engine::frame::begin_app_frame`, `engine::frame::end_app_frame`, and `engine::render::camera_view_for_size` while keeping gameplay, collision, camera, input dispatch, and events app-owned.

Use the root `engine` launcher for data-driven project manifests. Use app crates under `apps/<name>` when custom Rust behavior is required.

## Package and Project Validation

Dogfood content identity is now expressed through engine contract files. Validate them with `engine_pack`:

```bash
# Validate package manifest
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon

# Validate project manifest (also validates startup scene)
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml

# Validate startup scene against project asset IDs
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

**Contract files:**
- `engine.project.toml` -- project identity, package references, settings (`project.dogfood_dungeon`)
- `assets/dogfood_dungeon.package.toml` -- 10 registered assets (models, textures, environment, audio)
- `scenes/start.engine.scene.json` -- minimal scene root node for validation anchoring

The dungeon geometry, materials, lights, and props are built procedurally at runtime. The startup scene provides the root node anchor; app code owns the scene seeding through `scene_seed.rs`.

## Level Selection

```bash
# Procedural dungeon (default)
cargo run -p dungeon_dogfood -- --level generated_sprawl

# Authored validation maps
cargo run -p dungeon_dogfood -- --level level_01
cargo run -p dungeon_dogfood -- --level level_02_ramps
cargo run -p dungeon_dogfood -- --level level_03_lighting

# Via environment variable
DUNGEON_DOGFOOD_LEVEL=level_03_lighting cargo run -p dungeon_dogfood
```

Generator environment knobs: `DUNGEON_DOGFOOD_GENERATOR_SEED`, `DUNGEON_DOGFOOD_GENERATOR_WIDTH`, `DUNGEON_DOGFOOD_GENERATOR_HEIGHT`, `DUNGEON_DOGFOOD_GENERATOR_LAYERS`.

## Full-Content Visual Baseline

Enable props and custom environment for the reference visual baseline:

```bash
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level generated_sprawl
```

Repeat with `level_02_ramps` and `level_03_lighting` for coverage.

## Headless Draw Capture (Sprint 11)

Dogfood supports a true engine-owned headless draw capture path:

```bash
# Full-content headless draw capture baseline
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames=3 \
  --capture_frame_start=5 \
  --capture_frame_interval=5 \
  --capture_dir .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-baseline
```

Headless capture options:
- `--headless` -- use headless renderer (no window)
- `--capture_target <present|draw>` -- capture target (default: present)
- `--capture_frames <n>` -- number of frames to capture
- `--capture_frame_start <n>` -- frame to start capturing (default: 0)
- `--capture_frame_interval <n>` -- frames between captures (default: 1)
- `--capture_dir <dir>` -- output directory for captures

Known/quarantined capture flag drift: dogfood's local parser currently accepts the space-form capture flags shown above (for example `--capture_frames 3`), while the example command still shows equals-form for sequence/count flags. This is documented debt and was not changed by the app-owned loop helper adoption.

## Audio Smoke

The content pack declares `dogfood.audio.startup_ping` as an internal/generated WAV fixture. Normal startup validates the metadata and fixture path but does not load, probe, play, or open an audio output device.

Opt-in audio smoke:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --audio-smoke
```

The same gate can be enabled with `DUNGEON_DOGFOOD_AUDIO_SMOKE=1`. Device availability is host-dependent.

## Visual Lock Baseline

Visual tuning is owned in one app code location: `src/scene_seed.rs`.

- exposure: `2.8`
- gamma: `2.2`
- IBL ambient scale: `0.45`
- point light intensity baseline: `30.0`
- point light range baseline: `6.0`

The app passes these through `RendererConfig::visual_tuning`, so the demo owns its look without changing renderer-wide defaults.

## Known Limitations

- PBR material definitions (named texture sets with base_path discovery) remain in `content_pack.toml`; the engine PackageManifest `kind = "material"` is a different concept.
- Light presets (warm/cool/accent) with color/intensity/range remain in `content_pack.toml`; scene JSON supports inline lights but not named reusable presets.
- Prop placement policies (scale, yaw, y_offset) and audio format/usage metadata are app-specific content_pack fields with no engine contract equivalent.
- The torch prop is an intentional fallback marker, not the PBR quality target.
- `game_state.rs` is a placeholder struct not wired into the active game loop.
- Runtime validation currently blocked by a build environment incompatibility (libclang 22 vs clang-sys 1.8.1 in russimp-sys). Code changes are implemented and reviewed for correctness.
