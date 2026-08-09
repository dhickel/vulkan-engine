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
- **Safe stair-facing spawn**: the player starts at the center of a 64×64 lower stair landing, with a 16-unit hull radius plus 16-unit safety margin from the landing sides and stair solids
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
- The inter-layer slab aperture covers the full 192-unit stair run; the supported upper landing retains 80 units of headroom and a full 64-unit crest throat
- `info_player_start` is centered on the canonical lower landing and emits a cardinal `angle` facing the stair opening
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

## Enhanced v3 Two-Layer Dungeons with Cardinal + 45° Geometry

The `bsp_generator` crate also provides an **Enhanced v3** profile that produces
M2-only, two-layer dungeons with cardinal (axis-aligned) and exact 45° diagonal
chamfered-octagonal rooms, selectable cardinal portal surrounds, grounded
assemblies, and Sparse/Moderate/Rich density presets. Pointed arches remain the
byte-compatible default; rectangular and segmented surrounds are explorer
overrides. The profile reuses the CC0 Dungeon v2 theme.

### Key Features

- **Cardinal + 45° geometry**: rooms use axis-aligned and 45° diagonal footprints
  with chamfered/octagonal shapes
- **Selectable cardinal portal surrounds**: rectangular, pointed (default), or
  segmented full-depth shell omissions with 64×80 swept clearance at the throat;
  segmented crowns use a corridor-side backing cap so the decorative recess is sealed
- **Grounded assemblies**: acyclic support graph ensuring every feature brush
  is transitively supported by floor surfaces
- **Three density presets**: Sparse (≥12 rooms), Moderate (≥20 rooms, 2 loops),
  Rich (≥28 rooms, 4 loops)
- **6 grammar families**: PortalChamber, ButtressedHall, ColumnGrove,
  FracturedVault, TerracedShrine, MonolithicChamber — real integrated feature
  generators materializing grounded family-distinct brushes
- **Two-layer M2 arrangement**: lower floor Z=0, upper floor Z=192, 176-unit
  room height — identical vertical contract to Enhanced v2
- **Minimum-identity enforcement**: presets require specific minimum grammar
  family and feature counts; under-resourced configurations produce typed
  `MinimumIdentityFailure` errors
- **CC0 Dungeon v2 theme reused**: no new theme assets required

### CLI Usage

```bash
# Generate an m3 dungeon (defaults to Sparse, x2048)
dungeon_gen --class m3 --seed 42

# Explicit preset and extent
dungeon_gen --class m3 --seed 42 --preset moderate --extent 2048
dungeon_gen --class m3 --seed 99 --preset rich --extent 3072

# Explorer overrides, including the sealed segmented surround
dungeon_gen --class m3 --seed 42 --preset moderate --rooms 20 \
  --corridors 25 --loops 3 --arch-type segmented --minlight 32 --light-count 4

# Publish through engine_pack with a preset
engine_pack enhanced-dungeon-v3 --seed 42 --preset moderate --out /tmp/pkg

# Explore with the in-game GUI (architectural is default)
./tools/dungeon_explore.sh --seed 42
# Or select a specific mode
./tools/dungeon_explore.sh --m3 --preset rich --rooms 20
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
use bsp_generator::enhanced_v3::{generate_v3, V3Config, V3Preset};

// Nominal Sparse m3 dungeon (12+ rooms, 2048²)
let config = V3Config::nominal_sparse();
let map_text = generate_v3(&config)?;

// Custom seed and preset
let config = V3Config::new(42, V3Preset::Moderate, 2048)?;
let map_text = generate_v3(&config)?;

// Rich preset (28+ rooms, 4 loops, 3072²)
let config = V3Config::nominal_rich();
let map_text = generate_v3(&config)?;
```

### V3Config Fields

`V3Config::new()` retains the byte-compatible preset defaults. The remaining
public fields are explorer overrides and must pass `validate()` after mutation.
`run_pipeline()` and package entry points perform that validation again.

| field | compatibility default | accepted explorer values |
|-------|-----------------------|--------------------------|
| `seed` | caller supplied | any `u64` |
| `preset` | caller supplied | `Sparse`, `Moderate`, `Rich` |
| `xy_extent` | 2048 (Sparse/Moderate), 3072 (Rich) | 1024–3072, multiple of 16 |
| `rooms` | preset count | 3–40 |
| `corridors` | one segment per route | exact constructible segment count |
| `loops` | preset target | 0–6 |
| `vertical_edges` | 1 | 0–3 |
| `chamfer` | `true` | boolean |
| `arch_type` | `Pointed` | `None`, `Pointed`, `Segmented` |
| `stairs` | `true` | boolean |
| `room_span_min` / `room_span_max` | 112 / 448 | quantum-aligned valid span |
| grammar families / mode | all / `Mixed` | family allowlist; `Single` or `Mixed` |
| feature flags / density | all / 0.5 | category flags; finite 0.0–1.0 |
| `minlight` | 16 | 0–255 |
| `light_count` | room count | 0–rooms |

### Preset Comparison

| preset | exact rooms | same-layer routes | target loops | min families | min assemblies | min features | face budget |
|--------|-------------|-------------------|--------------|--------------|----------------|--------------|-------------|
| Sparse | 12 | 10 | 0 | 1 | 1 | 2 | 3,000 |
| Moderate | 20 | 20 | 2 | 3 | 3 | 6 | 5,000 |
| Rich | 28 | 30 | 4 | 6 | 6 | 12 | 8,000 |

The measured default-extent seed matrix (0, 42, 99, 255) emits 1,856–1,883
Sparse, 3,275–3,310 Moderate, and 4,725–4,782 Rich source faces. The preset
and M2 ceilings remain 3,000/5,000/8,000 and 10,000 respectively.

### Chamfer / Octagon Policy

When chamfers are enabled (`chamfer: true`, the default), rooms preferentially
receive multi-corner chamfer patterns rather than pure rectangular footprints.
Full octagonal rooms occur in ordinary output — not as rare specials. The
chamfer cut depth scales with the shorter room axis:

| shorter axis | chamfer depth |
|-------------|---------------|
| 112–191     | 32 units      |
| 192–255     | 48 units      |
| ≥ 256       | 64 units      |

The `--no-chamfer` flag or `chamfer: false` in config produces pure axis-aligned
rectangular rooms only.

### Visible Pointed Arch Crowns

Every Pointed portal (`arch_type: Pointed`, the default) retains a 64×80
clear core at the throat for gameplay passage. Above the core, a stepped
crown rises at least 48 units high in 16-unit bands, using `bs_accent` trim
texture. The crown is a real visible geometric feature carved into the wall
brushwork — not a decal, not an entity, not a flat texture trick. The stepped
bands create a recognizably pointed silhouette when viewed from either side of
the portal, and the `bs_accent` trim visually separates the crown from the
surrounding `bs_wall` surface.

### Room-Scaled Features

Feature brushes scale with the host room's shorter axis to maintain visual
proportion and gameplay readability:

- **Pillars**: 16×16 in rooms with shorter axis < 192; 32×32 in rooms with
  shorter axis ≥ 192.
- **Buttresses**: use two-quantum thickness (32 units) regardless of room
  size, projecting from walls at the full room height.
- **Blade walls**: span the full clear room height (floor to ceiling) rather
  than a partial divider, creating tall narrow passage-splitting geometry.

### Deferred Capabilities

The following capabilities are deferred and not available in Enhanced v3 production:

- Diagonal portals (shaped apertures on 45° walls)
- Concave rooms (T/L/alcove shapes)
- Trim theme role
- Lattice-slope walls (15°/30°)
- Third navigation layer

## EnhancedV3 Richness V1 — Gameplay Content Generation

The EnhancedV3 Richness V1 profile (`m3-richness-v1`) extends the baseline
EnhancedV3 output with gameplay-bearing content: 30 archetypes, 15 props,
12 lighting recipes, 3 themes, cave cells, and multi-storey vertical openings
with ladder/drop controller semantics. The profile is an additive extension —
baseline V3 map geometry, presets, tags, RNG domains, CLI, GUI, and package
manifests remain frozen and unchanged.

### Profile Identity

| concept | value |
|---------|-------|
| profile variant | `GenerationProfile::EnhancedV3RichnessV1` |
| production tag | `"m3-richness-v1"` |
| dispatch | `GenerationProfile::from_tag("m3-richness-v1")` → `Some(EnhancedV3RichnessV1)` |
| CLI | `dungeon_gen --class m3-richness-v1` |
| packaging | `engine_pack enhanced-dungeon-v3-richness-v1` |
| explorer | `dungeon_explore.sh --richness` or `--m3-richness-v1` |

### Key Features

- **30 shared semantic archetypes**: gameplay-bearing prefabs (chambers, halls,
  shrines, groves, vaults) with distinct geometry per family
- **15 shared semantic props**: decorative non-collidable entities (braziers,
  pillars, rubble, banners, chains, statuary) placed by the solver
- **12 shared semantic lighting recipes**: environment-type lighting with
  authored falloff, intensity, and placement rules
- **3 CC0 themes**: Ancient (stone), Egyptian (sandstone/hieroglyph), Brutalist
  (concrete/metal). Theme affects visual presentation only — room IDs, cave
  decisions, topology, critical path, spawn safety, pacing beats, and landmarks
  are invariant under theme selection
- **4 cave cell types**: spawn cave, dead-end gallery, junction grotto,
  transition tunnel — independent navigable cells with carved subtractive
  void geometry and unadorned stone trim
- **4 vertical opening types**: ladder shaft, drop hole, open stairwell,
  spiral-stair opening — with compiler-preserved explicit brush-AABB descriptors
  for app-owned climb/drop controller semantics
- **Same-XY composite reservations**: controlled multi-storey occupancy where
  composite reservations own both lower and upper layers; ordinary footprints
  remain projection-exclusive
- **Theme invariance**: a v3 blueprint regenerated with a different theme
  produces identical `.map` geometry bytes (compliance gate enforced)

### Content Architecture

The Richness V1 request is a separate immutable contract domain
`dungeon-gen/v3-richness/v1` that is structurally isolated from baseline V3.
It never mutates `V3Config`, `V3Preset`, or baseline `V3Error`, and never
routes v1, v2, or baseline-v3 RNG, serialization, geometry, or metadata
through richness code.

```text
RichnessDocumentV1 → resolve → ResolvedRichnessRequestV1
    → pacing blueprint → zone blueprint → placement + topology
    → variation plan → complexity plan → composition (archetypes,
      props, lights, caves, vertical openings, presentation)
    → cross-stage invariant validation → canonical emission
    → (.map text, metadata, request export, asset roles)
```

### Controls and Presets

The authored `RichnessDocumentV1` carries required fields (seed, extent,
preset, theme) plus inheritable controls where every field preserves whether
a value was explicitly supplied or inherited from the preset.

| field | type | range | description |
|-------|------|-------|-------------|
| `seed` | `u64` | any | master seed for deterministic generation |
| `extent` | `u32` | 1024–3072, quantum-aligned | XY bounds per axis |
| `preset` | `RichnessPreset` | `Sparse`, `Moderate`, `Rich` | density preset |
| `theme` | `RichnessTheme` | `Ancient`, `Egyptian`, `Brutalist` | visual theme |
| `critical_path_landmarks` | `InheritedOr<u32>` | inherited or 1–5 | landmark nodes on critical path |
| `zone_count` | `InheritedOr<u32>` | inherited or 1–6 | distinct zone count |
| `cave_mode` | `InheritedOr<RichnessCaveMode>` | inherited or `Required`/`Preferred`/`Omitted` | cave eligibility |
| `vertical_openings` | `InheritedOr<u32>` | inherited or 0–12 | vertical feature count |
| `budget_ceiling` | `InheritedOr<u32>` | inherited or 1000–8000 | source face ceiling |

**Preset defaults:**

| preset | landmarks | min zones | cave mode | vertical openings | face budget |
|--------|-----------|-----------|-----------|-------------------|-------------|
| Sparse | 1 | 1 | Preferred | 0 | 3,000 |
| Moderate | 2 | 1 | Preferred | 2 | 5,000 |
| Rich | 3 | 1 | Preferred | 4 | 8,000 |

### Revision Envelope

Every request carries seven closed revision tags that identify the exact
sprint-qualified algorithm, content, preset, theme, asset, and convention
revisions. Unknown revision tags fail closed with typed
`RichnessError::UnknownRequestSchemaRevision` (or the specific revision
variant). Current frozen revisions:

| revision | tag |
|----------|-----|
| request schema | `enhanced-v3-richness-request/v1` |
| algorithm | `enhanced-v3-richness-algorithm/v1` |
| content | `enhanced-v3-richness-content/v1` |
| presets | `enhanced-v3-richness-presets/v1` |
| themes | `enhanced-v3-richness-themes/v1` |
| assets | `enhanced-v3-richness-assets/v1` |
| conventions | `enhanced-v3-richness-conventions/v1` |

### CLI Usage

```bash
# Generate a Richness dungeon through the CLI
./tools/dungeon_gen --class m3-richness-v1 --seed 42

# With explicit preset, theme, and extent
./tools/dungeon_gen --class m3-richness-v1 --seed 42 \
  --preset moderate --theme egyptian \
  --extent 2048

# With all inherited controls overridden
./tools/dungeon_gen --class m3-richness-v1 --seed 99 \
  --preset rich --theme brutalist \
  --extent 3072 --landmarks 3 \
  --zones 3 --cave-mode required \
  --vertical-openings 4

# Publish through engine_pack
engine_pack enhanced-dungeon-v3-richness-v1 \
  --seed 42 --preset moderate --theme ancient --out /tmp/pkg

# Launch the cache-backed in-game Richness explorer GUI
./tools/dungeon_explore.sh --richness --seed 42
./tools/dungeon_explore.sh --richness --preset rich --theme egyptian
```

The launcher verifies or builds the cached BSP/LIT pair, then starts
`bsp_beta --m3-richness-v1` with that complete prebuilt closure. The initial
world reuses the cache; later Generate actions still use the production
Richness worker and atomic mount transaction.

**Published package layout:**

```text
richness_package/
├── richness_dungeon.map          ← generated .map source
├── richness_dungeon.bsp          ← compiled BSP2
├── richness_dungeon.lit          ← colored light data
├── palette.lmp                   ← theme palette
├── <theme>.wad                   ← theme WAD (e.g., cc0_richness_ancient.wad)
├── metadata.json                 ← generation evidence + request provenance
├── request.txt                   ← canonical request export bytes
└── textures/                     ← PBR companions for referenced miptex identities
    ├── <identity>_norm.png
    └── <identity>_gloss.png
```

### Quick Start

```rust
use bsp_generator::{
    generate_richness_v1,
    RichnessDocumentV1, RichnessPreset, RichnessTheme,
};

// All-inherited Sparse dungeon (1 landmark, Ancient theme)
let doc = RichnessDocumentV1::new(
    42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient
)?;
let output = generate_richness_v1(doc)?;
println!("Generated {} faces, {} lights",
    output.actual.faces, output.actual.lights);

// Explicit Rich preset with cave requirement
let doc = RichnessDocumentV1::with_all_explicit(
    99, 3072, RichnessPreset::Rich, RichnessTheme::Egyptian,
    // revision envelope (all V1)
    Default::default(), // RichnessRequestSchemaRevision::V1
    Default::default(), // RichnessAlgorithmRevision::V1
    Default::default(), // RichnessContentRevision::V1
    Default::default(), // RichnessPresetRevision::V1
    Default::default(), // RichnessThemeRevision::V1
    Default::default(), // RichnessAssetRevision::V1
    Default::default(), // RichnessConventionRevision::V1
    // controls
    InheritedOr::Explicit(3),                    // landmarks
    InheritedOr::Explicit(2),                    // zones
    InheritedOr::Explicit(RichnessCaveMode::Required), // cave mode
    InheritedOr::Explicit(4),                    // vertical openings
    InheritedOr::Inherited,                      // budget (use preset default)
)?;
let output = generate_richness_v1(doc)?;
```

### Determinism and Theme Invariance

- Identical `(seed, extent, preset, theme)` → byte-identical `.map` output
- Different themes with identical (seed, extent, preset) produce identical
  geometry bytes — theme affects only visual presentation through WAD/miptex
  identity selection, not map geometry
- The RNG domain `"dungeon-gen/v3-richness/v1"` uses SHA-256 framing with
  length-framed UTF-8 stage tags, fully independent from baseline V3's
  `"dungeon-gen/v3"` domain
- Canonical request export bytes include the explicit-versus-inherited
  provenance markers, making the request identity independently reproducible

### Budgets and Ceilings

| metric | ceiling |
|--------|--------|
| compiled faces | < 15,000 (extends M2 baseline of 10,000) |
| entities | < 500 (extends M2 baseline of 300) |
| lights | < 100 |
| static batches | < 800 |
| per-archetype source brushes | < 64 |
| per-prop source brushes | < 16 |
| per-theme WAD miptex identities | < 64 |
| per-theme companion PNGs | < 128 |
| total package size | < 128 MiB |

### Stable Errors

All Richness failures return `RichnessError` — never panics, never partial
output. Key error codes:

| error code | condition |
|-----------|-----------|
| `UnknownRequestSchemaRevision` | unknown revision tag (any of 7 revisions) |
| `UnsupportedRichnessGate` | gate other than `"richness-v1"` |
| `UnknownPreset` / `UnknownTheme` | unrecognized preset or theme tag |
| `ValueOutOfRange` | field outside allowed range |
| `NotQuantumAligned` | field not multiple of 16 |
| `LandmarkCountInfeasible` | landmarks exceed extent capacity |
| `ZoneCountInfeasible` | zone count outside preset range |
| `CaveInfeasible` | cave mode infeasible for seed/layout |
| `BudgetInfeasible` / `BudgetOverrun` | budget conflict or exceeded |
| `PlacementExhausted` / `TopologyExhausted` | search exhaustion |
| `ThemeInvarianceViolated` | theme selection altered blueprint geometry |

### UI-Only Authoring Preferences

The explorer also retains non-canonical authoring preferences for pacing
(`relaxed`, `normal`, `intense`), variation intensity (`subtle`, `moderate`,
`wild`), prop density, and light density. They are saved separately from the
canonical request and affect only the explorer authoring view. They are not
inputs to `RichnessDocumentV1`, package generation, request identity, or
manifest hashes. Production generation is controlled by the canonical seed,
preset, theme, extent, landmarks, zones, cave mode, vertical openings, and
budget ceiling.

### In-Game Explorer GUI

The windowed `bsp_beta --m3-richness-v1` path provides a mode-exclusive
Richness overlay opened with F3 (keyboard) or F4 (mouse), leaving the baseline
M3 F1/F2 bindings unchanged. The overlay exposes every canonical Richness
control plus the separate UI-only authoring preferences described above.
Generate snapshots the validated draft; Apply & Close commits only after the matching latest
request succeeds. The pipeline reuses the existing background worker, package
publication, coordinator validation, and detached-mount retirement paths.

### Archetypes and Props

The 30 archetypes are drawn from 6 grammar descriptor families
(PortalChamber, ButtressedHall, ColumnGrove, FracturedVault, TerracedShrine,
MonolithicChamber). Each archetype is a grounded assembly brushwork variant
that fits within a reserved volume — the solver places them into committed
room envelopes during the composition stage. The 15 props are decorative
non-collidable entities placed by the same solver within navigable
clearance margins.

### Cave Cells and Vertical Openings

Cave cells are carved subtractive voids with unadorned stone trim and
minimum ambient `_minlight 32`. Eligibility is seed-and-layout-dependent
(never theme-gated). Vertical openings use compiler-preserved explicit
brush-AABB descriptors that the app-owned `BspPlayerMovementController`
interprets for climb/drop state without surface-flag dependencies.

## Limitations (Beta) and Known Issues

### Legacy v1 Design Limitations

These limitations apply to the Legacy v1 generator (`bsp_generator::generate`)
only. The Enhanced v2 profile supports two-layer dungeons with stairs, multiple
room palettes, and corridor/ceiling/pillar variance. The Enhanced v3 and
Richness V1 profiles add cardinal + 45° chamfered geometry, selectable cardinal
portal surrounds, grounded assemblies, archetypes, props, lighting recipes,
cave cells, and vertical openings.

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
