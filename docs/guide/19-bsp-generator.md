# BSP Dungeon Generator — Usage Guide

> App-builder guide for generating procedural Quake-format dungeon maps with the `bsp_generator` crate.

## Audience

Rust developers building applications that need procedural dungeon geometry as compiled BSP map content. You should be comfortable with the [BSP Map Support](18-bsp-beta.md) chapter and the [Asset Pipeline](09-asset-pipeline.md) before reading this chapter.

## Overview

The BSP dungeon generator is a **pure-Rust offline pipeline** that produces Quake 1 `.map` files from a `(seed, config)` pair. It has zero renderer, Vulkan, physics, audio, or windowing dependencies — it depends only on the `bsp` format crate and `sha2` for deterministic seeding.

Every run with identical `(seed, config)` produces **byte-identical** `.map` output. The output is designed for compilation through the pinned `ericw-tools 2.0.0-alpha3` toolchain into BSP2 format.

```
DungeonConfig + u64 seed  →  bsp_generator::generate()  →  .map text
                                                              ↓
                                          engine_pack compile-bsp  →  .bsp + .lit
```

## Quick Start

```rust
use bsp_generator::{generate, DungeonConfig};

// Generate a nominal M1 dungeon (12 rooms, 1 loop, 1024×1024, Z 192)
let (map_text, meta) = generate(42, DungeonConfig::nominal_m1())
    .expect("generation failed");

std::fs::write("my_dungeon.map", &map_text)?;
println!("Generated {} rooms, {} corridors, {} faces (est)",
    meta.room_count, meta.corridor_count, meta.face_count_estimate);
```

## Map Classes

The generator targets two output tiers, defined by the frozen generation contract.

| class | room count | loop count | XY max           | Z max | typical use                          |
|-------|-----------|------------|------------------|-------|--------------------------------------|
| M1    | 8–16      | 0–2        | ≤ 1536 × 1536    | ≤ 256 | test maps, deathmatch arenas         |
| M2    | 17–40     | 1–6        | ≤ 3072 × 3072    | ≤ 384 | representative single-player dungeons |

## Configuration

### Pre-built Nominal Configurations

```rust
// M1 nominal: 12 rooms, 1 loop, 1024², Z 192
let cfg = DungeonConfig::nominal_m1();

// M2 nominal: 28 rooms, 3 loops, 2048², Z 256
let cfg = DungeonConfig::nominal_m2();
```

### Custom Configuration

```rust
use bsp_generator::{DungeonConfig, MapClass};

let cfg = DungeonConfig {
    class: MapClass::M2,
    room_count: 20,
    loop_count: 2,
    xy_bounds: (1536, 1536),   // must be multiples of 16
    z_span: 256,               // must be multiples of 16
    placement_candidates: 32,  // ≤ class maximum
    max_placement_attempts: 96,
    max_astar_expansions: 524_288,
};
let validated = cfg.validate()?;  // explicit validation
```

All `xy_bounds` and `z_span` values must be multiples of the **construction quantum** (16 Quake units). The `validate()` method checks all fields against frozen per-class bounds and returns a `ValidatedConfig` or `GeneratorError::InvalidConfig`.

### Configuration Field Reference

| field | M1 default | M2 default | constraint |
|-------|-----------|-----------|------------|
| `class` | `M1` | `M2` | M1 or M2 |
| `room_count` | 12 | 28 | M1: 8–16, M2: 17–40 |
| `loop_count` | 1 | 3 | M1: 0–2, M2: 1–6 |
| `xy_bounds` | `(1024,1024)` | `(2048,2048)` | ≤ class max, multiple of 16 |
| `z_span` | `192` | `256` | ≤ class max, multiple of 16 |
| `placement_candidates` | 16 | 32 | 1..=class max |
| `max_placement_attempts` | 64 | 96 | 1..=class max |
| `max_astar_expansions` | 131072 | 524288 | 1..=class max |

## Seeds

The generator is **deterministically seeded**: identical `(seed, config)` pairs produce byte-identical `.map` output.

```rust
// Any u64 value is a valid seed
let (map1, _) = generate(0, cfg.clone())?;
let (map2, _) = generate(0, cfg.clone())?;
assert_eq!(map1, map2);  // byte-identical

// Different seeds produce different maps
let (map3, _) = generate(1, cfg.clone())?;
assert_ne!(map1, map3);
```

### Support Corpus Seeds

The frozen support corpus guarantees that these seeds produce valid output with nominal configurations:

| seed              | class | behavior                         |
|-------------------|-------|----------------------------------|
| `0`               | M1    | deterministic; no errors         |
| `1`               | M1    | deterministic; no errors         |
| `2`               | M1    | deterministic; no errors         |
| `3`               | M1    | deterministic; no errors         |
| `17`              | M2    | deterministic; no errors         |
| `255`             | M2    | deterministic; no errors         |
| `0x5555555555555555` | M2 | deterministic; no errors       |
| `u64::MAX`        | M2    | deterministic; no errors         |

Boundary configurations (minimum/maximum room and loop counts) use seeds `42`–`45`.

## Errors

All failures return `GeneratorError` — never panics, never silently falls back.

| variant | cause |
|---------|-------|
| `InvalidConfig` | configuration field out of frozen bounds |
| `PlacementExhausted` | could not place all rooms within attempt budget |
| `RouteExhausted` | could not find corridor route within A* budget |
| `InvariantViolation` | internal generator bug |
| `SerializationFailed` | `.map` formatting overflow |
| `ArithmeticOverflow` | bounds computation overflow |

### Error Recovery

- `PlacementExhausted` and `RouteExhausted` are **seed-dependent**: retrying with a different seed often succeeds.
- Increase `placement_candidates` or `max_astar_expansions` within class limits for dense configurations.
- `InvalidConfig` always indicates a programming error — fix the configuration.

## Theme

The beta generator uses the **CC0 Stone Beta** theme — a project-authored, license-clean procedural texture set. Theme assets live in `src/bsp_generator/themes/cc0_stone_beta/` and include:

- `cc0_stone_beta.wad` — WAD2 archive with four distinct 1024×1024 visible stone roles and a 64×64 compiler-only `skip` miptex
- `palette.lmp` — 768-byte project-authored Quake palette
- `textures/*.png` — matching 1024×1024 albedo, normal, and gloss companions for each visible surface role; none for `skip`
- `theme.toml` — texture role bindings (floor, wall, ceiling, accent)
- `LICENSE` — CC0 public domain dedication

The theme is **self-contained**: the generator references `cc0_stone_beta.wad` by basename in emitted `.map` output. No external texture packs are needed for compilation.

### Rebuilding Theme Assets

```bash
cd src/bsp_generator/themes/cc0_stone_beta
python3 build.py
```

`build.py` is deterministic — it produces byte-identical output on every run. It requires Pillow (`pip install Pillow`) and generates only project-authored CC0 procedural pixels.

## Compilation Pipeline

Generated `.map` files must be compiled through `ericw-tools 2.0.0-alpha3`:

```bash
# Compile with the pinned BSP2 profile
engine_pack compile-bsp my_dungeon.map \
    --profile tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml \
    --wad src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad \
    --palette src/bsp_generator/themes/cc0_stone_beta/palette.lmp
```

This produces `my_dungeon.bsp` (BSP2 format) and `my_dungeon.lit` (colored light data). The profile uses:
- `-bsp2` (BSP2 format, mandatory for generated output)
- `light -threads 1 -lit` (deterministic, external `.lit` output)

`engine_pack` fails closed if `qbsp`, `vis`, or `light` reports a warning, including missing textures or skipped map filling. A successful command therefore means the published BSP was compiled warning-free; inspect the reported `CompilerWarning` instead of using a partial output.

### Deterministic Builds

Two independent compilations of the same `.map` with the same WAD and palette produce byte-identical `.bsp` and `.lit` files. Combined with seed determinism in the generator, this guarantees reproducible builds from `(seed, config)` to compiled output.

## Output Guarantees

Generated maps guarantee:

1. **Sealed geometry**: no leaks and no skipped compiler fill
2. **Walkable topology**: all rooms reachable from `info_player_start`
3. **Real open portals**: corridor routes reach room walls and the additive brush set omits each 64×80 arch aperture
4. **Clear junctions**: L/T/X centers retain the full 64-unit route width
5. **Non-solid point entities**: the spawn and room lights are above the floor slab in clear room volume
6. **No overlapping rooms**: rooms are separated by at least 16-unit wall thickness and have a minimum 112-unit outer span
7. **Open arches only**: no doors, buttons, or moving geometry
8. **Single-layer**: flat floors, common room ceiling height
9. **BSP2 only**: no BSP29 output
10. **Deterministic**: byte-identical `.map` for identical inputs
11. **Warning-free compilation**: any `qbsp`, `vis`, or `light` warning is an error

## Support Corpus

All 12 frozen configurations (8 nominal seeds + 4 boundary configs) compile warning-free through the BSP2 profile, reload in strict mode, remain sealed, and return non-solid BSP contents at room, spawn/light, corridor, portal-throat, and junction witnesses. Face and entity ceilings pass; static-batch ceiling enforcement is tracked separately in GitHub issue #57. The corpus is executed by:

```bash
cargo test -p bsp_generator --test corpus_execution
```

## Enhanced v2 Two-Layer Dungeons

The `bsp_generator` crate also provides an **Enhanced v2** profile that produces
M2-only, two-layer dungeons with stairs, theme palette assignment, and
per-room corridor/ceiling/pillar variance.

### Key Features

- **Two layers**: lower floor at Z=0, upper floor at Z=192, 176-unit room height
- **Two stair types**: room-scale grand stairs span a host room's usable width; wall-edge narrow stairs use a 64-unit strip. Both use twelve 16-unit treads and risers.
- **Corridor variance**: per-route widths of 64, 80, or 96 Quake units
- **Ceiling variance**: per-room ceiling heights of 128, 144, or 176 Quake units
- **Pillars**: up to 8 freestanding axis-aligned pillars per room
- **Theme roles**: rooms get typed roles (Entry, Hub, DeadEnd, Side) with
  palette assignment via Uniform or ByZone strategies
- **CC0 Dungeon v2 theme**: separate theme WAD from Legacy v1

### CLI Usage

```bash
# Generate, compile, and publish an enhanced dungeon through engine_pack.
# The published closure includes the BSP, .lit, palette, WAD, metadata, and
# referenced normal/gloss companion files under textures/. Albedo remains
# WAD-backed per the frozen contract.
engine_pack enhanced-dungeon --seed 42 --out ./dungeon_package \
    --tool-path ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin
```

**Published package layout:**

```text
dungeon_package/
├── enhanced_dungeon.map      ← generated .map source
├── enhanced_dungeon.bsp      ← compiled BSP2
├── enhanced_dungeon.lit      ← colored light data
├── palette.lmp               ← CC0 Dungeon v2 palette
├── cc0_dungeon_v2.wad        ← theme WAD (albedo source)
├── metadata.json             ← generation evidence
└── textures/                 ← PBR companions
    ├── bs_floor_norm.png
    ├── bs_floor_gloss.png
    ├── bs_wall_norm.png
    ├── bs_wall_gloss.png
    ├── bs_ceil_norm.png
    ├── bs_ceil_gloss.png
    ├── conn_floor_norm.png
    ├── conn_floor_gloss.png
    ├── conn_wall_norm.png
    ├── conn_wall_gloss.png
    ├── conn_ceil_norm.png
    └── conn_ceil_gloss.png
```

### Quick Start

```rust
use bsp_generator::enhanced::pipeline::generate_enhanced;
use bsp_generator::enhanced::config::EnhancedConfig;

// Nominal M2 two-layer dungeon
let cfg = EnhancedConfig::nominal();
let (map_text, meta) = generate_enhanced(42, cfg)?;

// Minimal M2 (17 rooms, compact 1024² bounds)
let cfg = EnhancedConfig::minimal();
let (map_text, meta) = generate_enhanced(17, cfg)?;

// Maximal M2 (40 rooms, 6 loops, 3 stairs, full 3072² bounds)
let cfg = EnhancedConfig::maximal();
let (map_text, meta) = generate_enhanced(200, cfg)?;
```

### EnhancedConfig Fields

| field | default | range | description |
|-------|---------|-------|-------------|
| `room_count` | 28 | 17–40 | total rooms across both layers |
| `loop_count` | 3 | 1–6 | extra loop edges beyond spanning tree |
| `vertical_edges` | 1 | 1–3 | number of stair connections |
| `tread_depth` | 16 | exactly 16 | fixed stair tread depth; other values are rejected |
| `xy_extent` | 2048 | ≤ 3072, multiple of 16 | XY bounds per axis |
| `max_pillars_per_room` | 2 | 0–8 | maximum pillars per room |

### Cross-Layer Guarantees

- Rooms on different layers never overlap in XY projection
- Each layer is independently connected (per-layer spanning tree)
- Stair transitions seal both layers together — the whole dungeon is connected
- Room-scale grand stairs use a 192-unit run across the host room's full usable width
- Wall-edge narrow stairs use a 192×64-unit strip hugging a room wall
- 12 treads × 16-unit riser = 192-unit climb (exactly the inter-layer offset)
- Lower routes join the stair entrance through a split wall aperture; the upper path uses a split ceiling exit and upper-room wall aperture
- Socket portals are 64 units wide with 32-unit corner margins
- All rooms have 176-unit height (≥ 80-unit required headroom for portals)

### Theme

The CC0 Dungeon v2 theme at `src/bsp_generator/themes/cc0_dungeon_v2/` provides
a separate set of CC0 textures from Legacy v1's CC0 Stone Beta. Both themes are
project-authored and deterministically built. Enhanced publication requires and
publishes both normal and gloss companions for every referenced eligible identity;
a missing, malformed, or dimension-mismatched companion rejects the package before
atomic publication. Albedo remains WAD-backed per the frozen PBR companion naming
contract (§9 of the dungeon generation specification). This Enhanced-only closure
rule does not change generic `engine_pack compile-bsp` optional-companion behavior.

## Limitations (Beta) and Known Issues

### Legacy v1 Design Limitations

These limitations apply to the Legacy v1 generator (`bsp_generator::generate`)
only. The Enhanced v2 profile supports two-layer dungeons with stairs, multiple
room palettes, and corridor/ceiling/pillar variance.

- Single-layer Cartesian only (no ramps, stairs, multi-floor)
- Axis-aligned rectangular rooms only
- Axis-aligned straight corridors only
- Open arches only (no doors, buttons, platforms)
- No liquid volumes, monster placement, puzzle logic
- Single theme (CC0 Stone Beta)
- No runtime regeneration — the `.map` is generated offline

### Runtime Blockers (Open GitHub Issues)

The generator itself passes all tests, but the end-to-end pipeline from generated `.map` → compiled `.bsp` → runtime mount is blocked:

| issue | effect |
|-------|--------|
| [#57](https://github.com/dhickel/vulkan-engine/issues/57) | Static batch ceiling not enforced across frozen corpus |
| [#58](https://github.com/dhickel/vulkan-engine/issues/58) | Strict extraction fails on generated faces (missing lightmap) |
| [#61](https://github.com/dhickel/vulkan-engine/issues/61) | GPU upload rollback crashes (SIGSEGV) |
| [#62](https://github.com/dhickel/vulkan-engine/issues/62) | Planned mesh bounds lost after GPU transfer |
| [#63](https://github.com/dhickel/vulkan-engine/issues/63) | First material slot rejected as null |

Development-mode authorization (`--development`) can reach upload preflight (6 batches for nominal M1 seed 0) but cannot complete GPU mount until #61–#63 are resolved.

## Integration with Applications

### bsp_beta

`bsp_beta` (at `apps/bsp_beta/`) includes generated map load tests:

```bash
cargo test -p bsp_beta --test generated_map_load
```

These tests generate a dungeon through the full pipeline (generator → compiler → loader), proving end-to-end integration.

### Custom Applications

```rust
use bsp_generator::generate;
use std::process::Command;

// 1. Generate .map
let (map_text, _meta) = generate(42, DungeonConfig::nominal_m1())?;
std::fs::write("/tmp/gen.map", &map_text)?;

// 2. Compile via engine_pack (or call qbsp/vis/light directly)
let status = Command::new("engine_pack")
    .args(["compile-bsp", "/tmp/gen.map",
           "--profile", "tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml",
           "--wad", "src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad"])
    .status()?;

// 3. Load through bsp_runtime coordinator (see BSP Beta guide)
```

## See Also

- [BSP Generator API Reference](../api/19-bsp-generator.md) — full type and function documentation
- [BSP Generator Internals](../internal/19-bsp-generator.md) — architecture and algorithms
- [BSP Map Support Guide](18-bsp-beta.md) — loading and rendering compiled BSP maps
- [Dungeon Generation Specification](../../.internal-dev/specifications/bsp-dungeon-generation.md) — frozen contract values
