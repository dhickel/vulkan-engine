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

- `cc0_stone_beta.wad` — WAD2 archive with four visible stone roles and a compiler-only `skip` miptex
- `palette.lmp` — 768-byte Quake palette
- `textures/*.png` — PBR companions (normal + gloss for each visible surface role; none for `skip`)
- `theme.toml` — texture role bindings (floor, wall, ceiling, accent)
- `LICENSE` — CC0 public domain dedication

The theme is **self-contained**: the generator references `cc0_stone_beta.wad` by basename in emitted `.map` output. No external texture packs are needed for compilation.

### Rebuilding Theme Assets

```bash
cd src/bsp_generator/themes/cc0_stone_beta
python3 build.py
```

`build.py` is deterministic — it produces byte-identical output on every run.

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

## Limitations (Beta)

- Single-layer Cartesian only (no ramps, stairs, multi-floor)
- Axis-aligned rectangular rooms only
- Axis-aligned straight corridors only
- Open arches only (no doors, buttons, platforms)
- No liquid volumes, monster placement, puzzle logic
- Single theme (CC0 Stone Beta)
- No runtime regeneration — the `.map` is generated offline

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
