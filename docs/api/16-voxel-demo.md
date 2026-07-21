# Voxel Cave Demo

`voxel_demo` is a permanent workspace application that generates procedural 3D cave systems and renders them interactively with PBR materials, point lights, and IBL environment. It demonstrates the full engine pipeline: voxel generation → isosurface extraction → procedural mesh upload → renderer scene construction → app-owned interactive or headless capture loop.

## What It Is

- A standalone app crate under `apps/voxel_demo`
- A topology-first procedural cave generator with deterministic RNG and resolution-gated validation
- An MC33 (Marching Cubes 33) isosurface mesher with Lewiner tables
- A PBR stone material scene with 9 fixed point lights (within 16-light cap)
- An app-owned interactive windowed loop (WASD + mouse, noclip, manual frame capture)
- A headless capture mode that renders at 5 landmark viewpoints with enriched JSON sidecars

## What It Is Not

- A terrain or landscape generator
- A game (no enemies, inventory, persistence, physics)
- A level editor or authoring tool
- A dungeon_dogfood replacement or migration target

## Quick Start

```bash
# Windowed, default 96³ resolution, seed 0
cargo run -p voxel_demo

# Higher resolution (128³) or different seed
cargo run -p voxel_demo -- --seed 77 --resolution 128

# With environment map
cargo run -p voxel_demo -- --env apps/dungeon_dogfood/assets/sky_maps/indoor_4k.exr
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--seed <N>` | `0` | RNG seed for deterministic generation |
| `--resolution <N>` | `96` | Cubic lattice resolution; must be 64, 96, or 128 |
| `--shell-thickness <N>` | `2` | Solid shell thickness in voxel units |
| `--light-budget <N>` | `9` | Reserved — always uses 9 fixed point lights |
| `--headless` | off | Run without a window; capture at 5 landmark viewpoints |
| `--capture-dir <PATH>` | auto-generated | Output directory for headless frame captures |
| `--env <PATH>` | indoor_4k.exr fallback | Environment map for IBL skybox |
| `--help` | — | Print usage and exit |

## Controls (Windowed Mode)

| Input | Action |
|-------|--------|
| **W A S D** | Move forward / left / backward / right |
| **Mouse** | Look around (FPS-style) |
| **Space** | Move up |
| **Left Shift** | Move down |
| **F** | Log noclip toggle (movement is noclip-only — no collision) |
| **C** | Capture current frame (PNG saved to `captures/`) |

## Headless Mode

```bash
# Capture at 5 landmark viewpoints (spawn, junction, grand_cavern, shaft, destination)
cargo run -p voxel_demo -- --headless --seed 42 --capture-dir ./captures/voxel-test
```

Headless mode:
1. Generates cave, meshes it, and seeds the scene (no window)
2. Warms up 10 render frames
3. Positions camera at each site viewpoint
4. Requests a draw-target frame capture, polls until it completes
5. Writes PNG + renderer sidecar JSON + enriched sidecar JSON (seed, resolution, site metadata)

## Architecture Overview

```
CLI (--seed, --resolution, --headless, ...)
 │
 ├──▶ NormalizedConfig           PresentationConfig
 │       (generation-affecting)    (runtime-only)
 │
 ▼
validate_normalized()  ◀── range gates, allowed resolutions, light budget
 │
 ▼
TopologyFirst generator ▶ VoxelWorld (DenseLattice<i8> density + DenseLattice<u8> material)
 │                      └── RNG (PCG32 V1, phase-tagged streams)
 │                      └── 3D Perlin noise with FBM for cavern carving
 │
 ▼
Mc33 mesher ▶ MeshResult { vertices, normals, tangents, uvs, indices }
 │             └── Marching Cubes 33 with Lewiner et al. tables
 │             └── Central-difference gradient → dominant-axis UV projection
 │
 ▼
ProceduralMeshData upload ▶ MeshHandle + MaterialHandle
 │
 ▼
Renderer scene (PBR stone, point lights, IBL environment)
 │
 ├──▶ Windowed: winit loop + FPSController + app-owned InputSystem + render_scene_with_view
 └──▶ Headless: set_camera_look_at + request_frame_capture + render_scene_headless
```

### Generator: Topology-First

The `TopologyFirst` generator places semantic sites (spawn, junction, grand_cavern, shaft, destination) then carves connecting routes between them using noise-warped ellipsoids. A solid shell of `shell_thickness` voxels is enforced on all six lattice faces. The algorithm produces deterministic output for a given seed and resolution. Generator parity is verified against golden binary dumps for seeds 1–12.

### Mesher: MC33 (Marching Cubes 33)

The `Mc33` mesher walks the density field cell by cell, classifies each cell corner against the isosurface at density=0, and selects the appropriate triangle configuration from the Lewiner et al. base lookup tables. This is a **base-table MC33 implementation** — it uses the 256-entry edge and triangle tables from the Lewiner MC33 paper which embed 33 distinct case configurations, but it does **not** implement a runtime asymptotic-decider for sub-case selection. Per-vertex normals use central-difference gradients; tangent frames use dominant-axis UV projection (a cosmetic limitation — may show stretching or banding on some surfaces). Output passes structural validation on all six required test fields with zero open boundary edges and zero non-manifold edges.

### Material

A single PBR material approximates cave limestone:
- Base color: warm gray `(0.52, 0.47, 0.42)`
- Metallic: 0.0, Roughness: 0.75
- IBL ambient scale: 0.35 (dim caves)
- Exposure: 4.0, Gamma: 2.2

### Lights

Up to 9 point lights are placed at in-air positions around the cave:
- **5 site lights**: one at each landmark site with site-themed color (warm orange, amber, cool blue, pale green) and intensity (18–40)
- **4 edge lights**: at midpoints between connected sites with smaller range and intensity
- All light positions are validated against the density field before placement

### Cameras and Viewpoints

- **Windowed**: Camera starts at the first site viewpoint; FPS controller handles movement
- **Noclip**: Always enabled by default; camera can pass through solid rock
- **Headless**: 5 fixed viewpoints derived from site positions, each offset from the target for a wide-angle view

## Source Layout

```
apps/voxel_demo/
  Cargo.toml
  src/
    main.rs           — CLI, windowed/headless loops, scene seeding
    config.rs         — NormalizedConfig / PresentationConfig types
    validate.rs       — Config validation gates
    cave_gen/
      mod.rs          — Module root
      generators/
        mod.rs        — Generator trait, carving helpers, shell enforcement
        topology_first.rs  — Topology-first cave generator
      lattice.rs      — DenseLattice<i8>/<u8>, VoxelWorld
      metrics.rs      — Site/RouteEdge types, camera pose derivation
      noise.rs        — 3D Perlin noise with FBM
      rng.rs          — PCG32 V1 RNG with phase-tagged streams
    meshers/
      mod.rs          — MeshResult, FieldMesher trait, mesh validation
      mc33.rs         — Marching Cubes 33 extractor
  test_data/
    goldens/          — Golden density/material/sites/edges for seeds 1–12
```

## Dependencies

- `engine` (root facade): input routing, camera view helpers
- `renderer` (renderer crate): procedural mesh upload, PBR materials, point lights, IBL, headless capture
- `glam`: linear algebra (Vec3/Vec4)
- `winit`: window creation and event loop
- `serde`/`serde_json`: configuration, golden site/edge serialization, enriched sidecars
- `sha2`: configuration hashing (deterministic identity)
- `log`/`env_logger`: runtime logging

## Testing

```bash
cargo test -p voxel_demo                     # All voxel demo unit tests
cargo test -p voxel_demo -- --nocapture      # With output
cargo test -p voxel_demo config              # Config validation only
cargo test -p voxel_demo parity              # Generator parity (requires golden files)
```

Generator parity tests compare density, material, site positions, and edge metadata against golden files in `test_data/goldens/` for seeds 1–12. These tests are skipped if the golden directory is absent.

## See Also

- [Dogfood Vertical Slice](14-dogfood-vertical-slice.md) — the app-owned loop proof
- [App-Owned Loop](15-app-owned-loop.md) — full input/frame/render helpers guide
- [Quickstart](01-quickstart.md) — first-frame example
- [Scene Construction](03-scene-graph-and-fragment-workflows.md) — scene graph and mesh adding
- [Configuration](07-config.md) — RendererConfig reference
