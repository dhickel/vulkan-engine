# 14 — Case Study: Dungeon Dogfood

> Provenance: `G-14` — excerpts from `apps/dungeon_dogfood/src/`; canonical source at [`apps/dungeon_dogfood/`](../../apps/dungeon_dogfood/)

This chapter walks through the `dungeon_dogfood` application — the alpha vertical-slice proof that uses the full app-owned loop from [Chapter 04](04-app-owned-loop.md). It is a procedural dungeon explorer demonstrating input dispatch, event lifecycle, fixed-step physics collision, mesh-collider recipes, audio telemetry bridging, renderer view submission, and headless draw capture.

This is **not** a full source tour. It covers selected architectural excerpts. For complete source, see the files listed at the end of this chapter.

> **Label**: This app is **not standalone**. It is a workspace member under `apps/dungeon_dogfood/` and requires the root workspace to build. Use `cargo run -p dungeon_dogfood`, never `cargo run --manifest-path`.

## Quick Start

```sh
# Windowed, default procedural dungeon
cargo run -p dungeon_dogfood -- --level generated_sprawl

# Specific authored level
cargo run -p dungeon_dogfood -- --level level_02_ramps

# Headless draw-capture proof (engine-owned visual evidence)
RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/dogfood-baseline
```

> Provenance: `G-14-QS` — commands match `AGENTS.md` runtime validation section and dogfood headless capture flags.

## Content Architecture

Dogfood content is split between engine data contracts and app-specific config.

### Engine Data Contracts

Three canonical files define the project structure:

**Project** — [`apps/dungeon_dogfood/engine.project.toml`](../../apps/dungeon_dogfood/engine.project.toml):
- Declares `project.dogfood_dungeon` with a startup scene and package manifest reference.
- Validated by `engine_pack validate-project`.

**Package** — [`apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml`](../../apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml):
- Registers 10 assets: models (torch_sconce, crate_a), PBR texture sets (wall/floor), environment (dungeon_neutral.exr), audio (startup_ping.wav).
- Validated by `engine_pack validate-package`.

**Scene** — [`apps/dungeon_dogfood/scenes/start.engine.scene.json`](../../apps/dungeon_dogfood/scenes/start.engine.scene.json):
- Minimal root node only. Dungeon geometry, lights, props, and environment are built procedurally at runtime.

Validate all three:

```sh
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

> Provenance: `G-14-VALIDATE` — commands from `docs/api/14-dogfood-vertical-slice.md`.

### App-Specific Config (Transitional)

[`apps/dungeon_dogfood/assets/content_pack.toml`](../../apps/dungeon_dogfood/assets/content_pack.toml) defines:

- **Runtime PBR material discovery** with base_path scanning for base-color, normal, and ARM maps
- **Light presets**: warm (`(1.0, 0.6, 0.3)`, intensity 30.0, range 6.0), cool (`(0.55, 0.68, 1.0)`, intensity 24.0, range 6.5), accent (`(0.92, 0.25, 1.0)`, intensity 38.0, range 5.0)
- **Prop placement policies** and **app audio clip selection metadata**

The engine package manifest registers durable assets, but these runtime material-discovery, preset-selection, placement, and startup-audio policies are still app-specific transitional config.

## Module Map

| File | Responsibility |
|------|---------------|
| [`main.rs`](../../apps/dungeon_dogfood/src/main.rs) | Entrypoint, CLI parsing, event loop, headless/windowed dispatch |
| [`player.rs`](../../apps/dungeon_dogfood/src/player.rs) | `PlayerState`, camera intent ingestion, noclip, velocity guards |
| [`collision.rs`](../../apps/dungeon_dogfood/src/collision.rs) | `CollisionWorld` from level data, AABB wall/ramp/floor collision |
| [`mesh_collider_bridge.rs`](../../apps/dungeon_dogfood/src/mesh_collider_bridge.rs) | App-owned mesh→physics recipe bridge, fence-aware retirement |
| [`scene_seed.rs`](../../apps/dungeon_dogfood/src/scene_seed.rs) | `LevelScene::from_level()` — materials, chunked mesh upload, lights, props |
| [`geometry.rs`](../../apps/dungeon_dogfood/src/geometry.rs) | Volumetric dungeon mesh construction (floor slabs, wall boxes, ramp wedges) |
| [`layout.rs`](../../apps/dungeon_dogfood/src/layout.rs) | Level file loading, `ParsedLevel` representation |
| [`generator/`](../../apps/dungeon_dogfood/src/generator/) | Procedural dungeon generation (room carving, connector corridors) |
| [`content.rs`](../../apps/dungeon_dogfood/src/content.rs) | Content pack loading and resolution |
| [`audio_bridge.rs`](../../apps/dungeon_dogfood/src/audio_bridge.rs) | Startup audio probe, optional device smoke test |
| [`events.rs`](../../apps/dungeon_dogfood/src/events.rs) | Dogfood-specific event logger |

## App-Owned Input, Events, Frame, and Camera Flow

Dogfood uses the exact same app-owned loop pattern described in [Chapter 04](04-app-owned-loop.md). Here is the per-frame sequence extracted from `main.rs`:

> Provenance: `G-14-LOOP` — Excerpt from [`apps/dungeon_dogfood/src/main.rs`](../../apps/dungeon_dogfood/src/main.rs) (`render_frame` function)

```
route_platform_input_to_app()     ← every winit event
    │
    └─ RedrawRequested
        │
        ├─ begin_app_frame()      ← tick clock, dispatch input, emit FrameStarted
        ├─ fixed_clock.update()   ← accumulate delta, produce steps
        ├─ fps_controller.update_from_snapshot()
        ├─ player.ingest_camera_intent()   ← velocity guard, noclip check
        ├─ collision::resolve_player_step() × N  ← per-step AABB resolution
        ├─ bridge.world.step(FIXED_DT)      ← per-step physics tick
        ├─ interpolated_player_position()   ← render one step behind
        ├─ camera_view_for_size()
        ├─ renderer.pump_asset_tasks(32)
        ├─ render_scene_with_view() / render_scene_headless_with_view()
        ├─ end_app_frame()        ← emit FrameEnded
        ├─ bridge.reap_retired()  ← fence-aware recipe retirement
        ├─ bridge.writeback_dynamic_transforms()  ← physics→scene pose sync
        └─ window.request_redraw()
```

Key differences from the guide checkpoint:

- **Interpolation**: Dogfood renders one simulation step behind the authoritative player state and interpolates using the fixed-clock alpha remainder. This smooths camera motion on displays that do not match the 60 Hz simulation rate.
- **Physics stepping**: `bridge.world.step(FIXED_DT)` runs per-simulation-step, in sync with collision resolution.
- **Headless path**: Uses `render_scene_headless_with_view` instead of `render_scene_with_view`.

## Scene and Content Setup

> Provenance: `G-14-SCENE` — Excerpt from [`apps/dungeon_dogfood/src/scene_seed.rs`](../../apps/dungeon_dogfood/src/scene_seed.rs)

`LevelScene::from_level()` builds the entire scene procedurally:

1. **Preflight**: Validates the point-light budget (≤ 16 markers).
2. **Materials**: Builds `PbrMaterialDesc` for wall and floor from content-pack texture sets, with safe fallback base colors.
3. **Chunked mesh upload**: Calls `build_level_chunks()` which constructs volumetric primitives (floor slabs, wall boxes, ramp wedges, ceiling closures) and uploads each chunk as a `ProceduralMeshData` via `assets.upload_procedural_mesh()`.
4. **Bounds**: Each chunk queries `assets.mesh_scene_bounds()` for authoritative bounds; falls back to `ConservativeVisible` on failure.
5. **Nodes**: Each chunk gets a child node under a `level_root`, positioned at its world origin via `Mat4::from_translation`.
6. **Lights**: Maps light markers to presets (7 warm, 2 cool, 1 accent distribution), creates `PointLight` objects, and resolves the optional directional light.
7. **Props**: Optionally loads model props (torch_sconce, crate_a) at marker positions.
8. **Collider policies**: Assigns `ColliderPolicy::StaticTrimesh` to every chunk mesh and `ConvexHull` to the dynamic proof mesh.

Visual tuning is locked for renderer exposure and the warm-light baseline:

| Parameter | Value |
|-----------|-------|
| Exposure | 2.8 |
| Gamma | 2.2 |
| IBL ambient scale | 0.45 |
| Warm preset baseline intensity | 30.0 |
| Warm preset baseline range | 6.0 |

Point lights are created from the content-pack preset sequence, with runtime intensity scaled by `0.95` and range scaled by `1.55`.

> Provenance: `G-14-VISUAL` — constants and point-light creation from `scene_seed.rs`.

## Fixed-Step Collision

> Provenance: `G-14-COLLISION` — Excerpt from [`apps/dungeon_dogfood/src/collision.rs`](../../apps/dungeon_dogfood/src/collision.rs)

Dogfood uses bespoke AABB-based collision — not the `physics` crate's collision detection:

- `CollisionWorld::from_level()` builds `WallCollider` AABBs, ramp records, and floor-tile height records from the parsed level.
- `resolve_player_step()` runs per simulation tick. It applies horizontal camera intent, resolves wall penetration with player radius `0.3`, then solves ground height from floor and ramp tiles for eye height `1.6`.
- A maximum of 4 wall-penetration iteration passes handles corner cases (player wedged between intersecting walls).
- Vertical correction snaps upward to the solved ground target and eases downward by at most `0.15` world units per frame.

## Mesh-Collider Bridge

> Provenance: `G-14-BRIDGE` — Excerpt from [`apps/dungeon_dogfood/src/mesh_collider_bridge.rs`](../../apps/dungeon_dogfood/src/mesh_collider_bridge.rs)

The `MeshColliderBridge` converts renderer-neutral `MeshGeometryDto` into physics recipes. This is an app-owned bridge, not a renderer feature:

- **Recipe registration**: `register_policy(&dto, policy)` creates a static trimesh or convex-hull recipe from the mesh geometry DTO.
- **Instantiation**: `instantiate_collider(recipe, body_kind, ...)` creates a `PhysicsBody` + `PhysicsCollider` pair and registers them in the app-owned `PhysicsWorld`.
- **Fence-aware retirement**: `reap_retired(serial)` checks the renderer's `RetirementClass::ColliderRecipe` queue against a completed frame serial. Recipes are only unloaded after the GPU has finished with the mesh.
- **Transform writeback**: `writeback_dynamic_transforms(scene)` reads dynamic/kinematic physics body transforms and writes them back to scene nodes.

**Status**: The mesh-collider bridge is active but transitional. It demonstrates the integration pattern (app-owned physics ↔ renderer-owned meshes) but the recipe lifecycle, retirement, and error-recovery paths are not yet stabilized for general use. Treat it as a reference implementation, not a supported public API.

## Audio Telemetry

> Provenance: `G-14-AUDIO` — Excerpt from [`apps/dungeon_dogfood/src/audio_bridge.rs`](../../apps/dungeon_dogfood/src/audio_bridge.rs)

Dogfood includes an audio smoke probe hook at startup:

- `run_startup_audio_probe()` resolves the first configured startup clip and reports whether device smoke was requested.
- The smoke test is opt-in via `DUNGEON_DOGFOOD_AUDIO_SMOKE=1` or `--audio-smoke`.
- When smoke is requested, the app loads and probes the clip without opening an output device, then creates an `AudioEngine` and plays/stops it once.
- Started, stopped, and failed smoke outcomes emit `AudioEvent` records onto the app-owned event bus.
- Failures are logged but are not fatal — the app continues without audio.

**Status**: Alpha. Audio device creation may fail on systems without a working audio backend (PulseAudio/PipeWire on Linux). This is expected — audio is not required for the dungeon demo to function.

## Renderer View Submission

Dogfood uses the app-owned camera path identically to the guide checkpoint:

```rust
let view = engine::render::camera_view_for_size(camera, viewport_width, viewport_height);
```

The renderer receives only a `CameraView` DTO — it does not own or mutate the app's camera. For windowed mode, `render_scene_with_view` is used. For headless mode, `render_scene_headless_with_view` renders to the offscreen draw target without swapchain presentation.

## Headless Validation

> Provenance: `G-14-HEADLESS` — Excerpt from `run_headless` in [`apps/dungeon_dogfood/src/main.rs`](../../apps/dungeon_dogfood/src/main.rs)

The headless path:

1. Creates `Renderer::new_headless(config)` — no window, no swapchain.
2. Builds the scene identically to the windowed path.
3. Registers collider recipes and validates them (when `--validate-colliders` is passed).
4. Configures a `FrameCaptureSequence` with `--capture_frames`, `--capture_frame_start`, and `--capture_frame_interval`.
5. Runs a loop: `render_frame(..., headless: true)` for each frame up to the last capture frame.
6. Verifies the expected number of captures were written.

Example command:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/dogfood-baseline
```

> **Note**: Dogfood CLI uses **space-form** capture flags (`--capture_frames 3`), not equals-form (`--capture_frames=3`). The parser accepts only space-separated values.

Environment flags:

| Variable | Effect |
|----------|--------|
| `DUNGEON_DOGFOOD_FAST_STARTUP` | When `1`, loads only base-color maps for wall/floor materials and skips normal/ARM map discovery |
| `DUNGEON_DOGFOOD_LOAD_PROPS` | When `1`, loads model props at marker positions |
| `DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV` | When `1`, loads the custom dungeon environment map |
| `DUNGEON_DOGFOOD_AUDIO_SMOKE` | When `1`, runs device-backed audio smoke test |
| `DUNGEON_DOGFOOD_VALIDATION` | When `1`, enables Vulkan validation layers |
| `DUNGEON_DOGFOOD_LEVEL` | Override level selection |
| `DUNGEON_DOGFOOD_GENERATOR_SEED` | Override procedural generator seed |
| `DUNGEON_DOGFOOD_GENERATOR_WIDTH` | Override generator grid width |
| `DUNGEON_DOGFOOD_GENERATOR_HEIGHT` | Override generator grid height |
| `DUNGEON_DOGFOOD_GENERATOR_LAYERS` | Override generator layer count |

> Provenance: `G-14-ENV` — constants from `main.rs` and `scene_seed.rs`.

## Windowed Controls

| Input | Action |
|-------|--------|
| **W A S D** | Horizontal movement |
| **Space** / **Left Shift** | Move up / down |
| **Mouse** | Look around (FPS camera) |
| **F** | Toggle noclip fly mode |
| **C** | Capture draw target to `captures/dungeon-dogfood-<timestamp>-pid<PID>/` |
| **Esc** | Exit via close request |

## Known Limitations

1. Runtime PBR material map discovery lives in `content_pack.toml`; the engine PackageManifest only registers durable texture assets.
2. Light presets are content-pack-only; scene JSON supports inline lights but not named presets.
3. Prop placement policies and app startup-audio selection are content-pack fields.
4. The torch prop uses unlit fallback shading, not full PBR.
5. Collision is bespoke AABB-based, not `physics`-crate-driven (the `physics` crate handles only the mesh-collider recipe/integration proof).
6. Build currently blocked on systems with libclang ≥ 22 due to `russimp-sys` / `clang-sys` 1.8.1 incompatibility.

## Next

[Chapter 15 — Case Study: Voxel Demo](15-voxel-demo-walkthrough.md) covers the configurable procedural-cave application with v2 presets, MC33 meshing, and windowed regeneration.
