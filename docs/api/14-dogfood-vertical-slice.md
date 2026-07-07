# Dogfood Vertical Slice (Alpha)

`dungeon_dogfood` is the alpha vertical slice demo for the engine facade. It demonstrates the end-to-end path from engine data contracts through custom Rust gameplay, app-owned runtime primitives, renderer view submission, and headless visual capture.

## What Dogfood Is

- A custom Rust app crate under `apps/dungeon_dogfood`
- A procedural dungeon explorer with input, camera, PBR materials, point lights, model props, and environment
- A validation target for engine project/package/scene contracts
- The integration proof for the app-owned input/event/camera path in the root `engine` facade
- A headless draw capture source for engine-owned visual evidence

## What Dogfood Is Not

- A complete game (no enemies, UI, inventory, save/load)
- An editor or authoring tool
- A scripting or physics migration target
- A release-candidate demo

## Quick Start

```bash
# Windowed, default procedural dungeon
cargo run -p dungeon_dogfood -- --level generated_sprawl

# Specific authored level
cargo run -p dungeon_dogfood -- --level level_02_ramps
```

## Content Architecture

Dogfood content is split between engine data contracts and app-specific config:

### Engine Data Contracts (canonical)

**Project** (`apps/dungeon_dogfood/engine.project.toml`):
```toml
format_version = 1
project_id = "project.dogfood_dungeon"
name = "Dungeon Dogfood"
startup_scene = "scenes/start.engine.scene.json"

[[packages]]
package_id = "dogfood_dungeon"
manifest = "assets/dogfood_dungeon.package.toml"
enabled = true
```

**Package** (`apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml`):
Registers 10 assets including models (torch_sconce, crate_a), textures (wall/floor PBR sets), environment (dungeon_neutral.exr), and audio (startup_ping.wav).

**Scene** (`apps/dungeon_dogfood/scenes/start.engine.scene.json`):
Minimal scene with a single root node. Dungeon geometry, lights, props, and environment are built procedurally at runtime by the app crate.

Validate all three:
```bash
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

### App-Specific Config (transitional)

`assets/content_pack.toml` defines PBR material texture sets (with base_path discovery), light presets (warm/cool/accent with color/intensity/range), prop placement policies, and audio format/usage metadata. These concepts do not yet have equivalents in the engine PackageManifest or Scene JSON formats. The content pack is kept as transitional debt; new data-driven features should use the engine contracts where supported.

## Runtime Architecture

```
main.rs
  ├── Load content_pack.toml (content.rs)
  ├── Select level (generator.rs / layout.rs)
  ├── Create Renderer (windowed or headless)
  ├── Create app-owned EventBus, InputSystem, FrameClock, Camera, and FPSController
  ├── Build Scene via LevelScene::from_level() (scene_seed.rs)
  │     ├── PBR materials from content pack
  │     ├── Chunked dungeon mesh (geometry.rs)
  │     ├── Point lights with 7/2/1 preset mapping
  │     ├── Model props (optional)
  │     └── Custom environment (optional)
  └── Run event loop (windowed) or headless capture loop
        ├── Renderer platform routing → app-owned input dispatch
        ├── Input action events + frame lifecycle on the app EventBus
        ├── FPS camera intent → player collision (player.rs, collision.rs)
        ├── Build CameraView from the corrected app-owned camera
        ├── Render scene frame through render_scene_with_view / render_scene_headless_with_view
        └── Log captured frames (headless)
```

The active dogfood path does not use renderer-owned gameplay camera state, renderer-owned input dispatch, or `Renderer::events_mut()` for app lifecycle/audio telemetry. The renderer still owns Vulkan frame submission, assets, capture output, debug UI platform side effects, and resize handling.

## Controls (Windowed)

- **WASD** or **Arrow Keys**: Move
- **Mouse**: Look around
- **Esc**: Exit

## Headless Draw Capture

Dogfood supports an engine-owned headless capture path (Sprint 11):

```bash
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

Capture options:
- `--headless` -- use headless renderer (no window)
- `--capture_target <present|draw>` -- capture target
- `--capture_frames <n>` -- number of frames to capture
- `--capture_frame_start <n>` -- frame to start capturing
- `--capture_frame_interval <n>` -- frames between captures
- `--capture_dir <dir>` -- output directory

Current issue #35-#37 validation proof:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --headless \
  --capture_target draw \
  --capture_frames 1 \
  --capture_dir .internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood
```

The headless proof writes a draw-target PNG plus JSON sidecar for the app-owned camera/view path.

## Visual Baseline

Visual tuning is locked in `src/scene_seed.rs`:

| Parameter | Value |
|-----------|-------|
| Exposure | 2.8 |
| Gamma | 2.2 |
| IBL ambient scale | 0.45 |
| Point light intensity | 30.0 |
| Point light range | 6.0 |

Light presets (from content_pack.toml):
- **Warm**: color (1.0, 0.6, 0.3), intensity 30.0, range 6.0
- **Cool**: color (0.55, 0.68, 1.0), intensity 24.0, range 6.5
- **Accent**: color (0.92, 0.25, 1.0), intensity 38.0, range 5.0

## Levels

| Level | Type | Description |
|-------|------|-------------|
| `generated_sprawl` | Procedural | Multi-layer dungeon with connectors, default 48x48x3 |
| `level_01` | Authored | Intro map with markers |
| `level_02_ramps` | Authored | All 4 ramp directions + multi-layer |
| `level_03_lighting` | Authored | Dense light marker placement |

## Known Limitations (Sprint 11)

1. PBR material definitions (texture set naming, base_path discovery) live in content_pack.toml, not in the engine PackageManifest
2. Light presets are content_pack-only; scene JSON supports inline lights but not named presets
3. Prop placement policies and audio format/usage metadata are content_pack fields
4. The torch prop uses unlit fallback shading, not full PBR
5. No runtime physics integration; collision is bespoke AABB-based
6. Build currently blocked on systems with libclang >= 22 due to russimp-sys / clang-sys 1.8.1 incompatibility
