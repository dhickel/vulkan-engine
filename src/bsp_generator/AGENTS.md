# bsp_generator Crate Agent Guide

> Development guide for agents working in `src/bsp_generator/`.

## Scope

Pure-Rust offline dungeon generator producing Quake 1 `.map` text from `(u64 seed, DungeonConfig)` pairs. This crate depends **only** on `bsp` (format types) and `sha2` — it must never gain a dependency on `renderer`, `vulkan`, `winit`, `physics`, `audio`, `scripting`, `bsp_runtime`, or `engine_pack`.

## Frozen Contract

All construction parameters, bounds, and serialization rules are frozen in `.internal-dev/specifications/bsp-dungeon-generation.md`. **Do not change** any of the following without explicit re-review:

- Construction quantum (16), wall thickness (16), minimum room outer span (112), corridor/portal width (64), corridor/portal height (80)
- M1/M2 room count, loop count, XY max, Z max ranges
- Stage tags (`"room-placement"`, `"corridor-routing"`, `"entity-placement"`, `"light-placement"`)
- Domain separator `"dungeon-gen/v1"`
- Canonical serialization rules (entity order, key order, face order, axis format, line endings)
- Output guarantees (sealed, walkable, no overlap, BSP2 only, open arches only, single-layer)

## Source Layout

### Legacy v1

```
src/
  lib.rs          — crate root, generate(), GenerationMetadata, re-exports
  config.rs       — DungeonConfig, ValidatedConfig, MapClass, frozen constants
  error.rs        — GeneratorError enum
  seed.rs         — Seed, StageSeed, StageRng (SHA-256-chained)
  intent.rs       — IR types: RoomIntent, LayoutIntent, Corridor, Junction,
                      RoutedIntent, Brush, BrushFace, EntityIntent, EmissionIntent
  geometry.rs     — snap_to_quantum, rooms_overlap, quantum-snapped helpers
  placement.rs    — place_rooms() — bounded grid-based placement
  topology.rs     — build_topology() — Kruskal MST + loop augmentation
  routing.rs      — route_edge(), route_all_edges() — orthogonal A* routing
  junction.rs     — make_brush(), outer-quadrant L/T/X helpers,
                      solid wall pieces around omitted room portals
  emission.rs     — build_emission() — split room shells, corridor-union shell, entities
  serialize.rs    — serialize() — canonical .map text output

tests/
  config_validation.rs  — validation error paths
  determinism.rs        — byte-identical output guarantee
  placement.rs          — room placement correctness
  topology.rs           — MST + loops, reachability
  geometry_validation.rs — overlap, snapping, bounds
  routing.rs            — corridor routing correctness
  junction.rs           — junction brush geometry
  emission.rs           — brush/entity counts
  serialize.rs          — canonical format validation
  canonical.rs          — full-pipeline determinism
  invariants.rs         — output guarantees
  collision.rs          — clearance invariants
  theme_evidence.rs     — theme determinism
  corpus_execution.rs   — full support corpus compilation

themes/cc0_stone_beta/
  theme.toml            — texture role bindings
  palette.lmp           — 768-byte Quake palette
  cc0_stone_beta.wad    — WAD2 texture archive
  build.py              — deterministic asset generator
  LICENSE               — CC0 dedication
  textures/             — PBR companion PNGs
```

### Enhanced v2

```
src/enhanced/
  mod.rs          — module declarations
  profile.rs      — GenerationProfile (LegacyV1 | EnhancedV2), GenerationRequest
  config.rs       — EnhancedConfig, frozen vertical contract constants
  error.rs        — EnhancedError typed variants
  seed.rs         — EnhancedSeed, EnhancedStageSeed, EnhancedStageRng
                     (domain "dungeon-gen/v2", 6 frozen stage tags)
  intent.rs       — typed newtype IDs (LayerId, RoomId, SocketId, RouteId,
                     TransitionId, ReservationId, ZoneId, PaletteId),
                     IdAllocator, RouteIntent, TransitionIntent
  occupancy.rs    — OccupancyGrid with owner-bearing projected XY cells,
                     GridCheckpoint snapshots
  placement.rs    — place_rooms() — RNG-driven two-layer placement with
                     balanced membership, transactional journal rollback,
                     socket derivation from committed room walls
  topology.rs     — build_topology() — per-layer MST + loop edges,
                     canonical socket-pair backtracking, stair reservations
  routing.rs      — route_sockets() — width-aware A* corridor routing
  reservation.rs  — Transaction with mark/rollback/commit, socket claims,
                     loop budget tracking
  transition.rs   — reserve_transitions() — typed grand/wall-edge stair reservations
  theme.rs        — ThemePackage (frozen CC0 Dungeon v2), PaletteDefinition,
                     RoomRole derivation, AssignmentStrategy, zone partitioning
  features.rs     — apply_features() — corridor width (64/80/96),
                     ceiling height (128/144/176), pillars, spawn/light origins
  emission.rs     — emit_map() — explicit room shells, wall aperture masking,
                     corridor union, stairwell sealing, canonical .map text
  metadata.rs     — re-exports EnhancedMetadata from pipeline
  pipeline.rs     — generate_enhanced() — full pipeline entry point

themes/cc0_dungeon_v2/
  build.py              — deterministic asset generator
  palette.lmp           — 768-byte Quake palette
  cc0_dungeon_v2.wad    — WAD2 texture archive
```

### Enhanced v3

```
src/enhanced_v3/
  mod.rs          — module declarations, generate_v3(), re-exports
  config.rs       — V3Config, V3Preset, NormalClass, frozen constants
  error.rs        — V3Error (45 typed variants across 9 categories)
  rng.rs          — V3Seed, V3StageSeed (domain "dungeon-gen/v3", 4 tags)
  ids.rs          — typed newtype IDs (RoomId, PortalId, etc.),
                    CommittedTopology, V3IdAllocator
  geometry.rs     — Point3, Rational, Vector3, CanonicalPlane, ConvexBrush
  footprint.rs    — build_footprints(), Footprint, FootprintLayout
  reservation.rs  — build_reservations(), Reservation, ReservationSet
  topology.rs     — build_topology(), compute_reservations()
  intent.rs       — plan_composition(), CompositionPlan, GrammarDescriptor
  assembly.rs     — Assembly, AssemblyBrush, Interface, Support, ProtectedVolume
  composition.rs  — CompositionPlan, GrammarDescriptor
  emission.rs     — emit_map_text(), texture_for_role()
  metadata.rs     — EnhancedV3Metadata (22 read-only accessors)
  pipeline.rs     — run_pipeline(), V3PipelineOutput

tests/
  enhanced_v3_contract_baseline.rs   — 25 frozen contract identity tests
  enhanced_v3_public_api.rs          — public API surface tests
  enhanced_v3_generation.rs          — can-generate tests for all presets
  enhanced_v3_semantic_core.rs       — geometry, footprint, topology unit tests
  enhanced_v3_geometry.rs            — ConvexBrush, CanonicalPlane, half-space tests
  enhanced_v3_compatibility.rs       — v1/v2 baseline preservation
  enhanced_v3_budget.rs              — M2 budget evidence (faces, entities, batches)
  enhanced_v3_compiled_space.rs      — compiled spatial witnesses
  enhanced_v3_compiler.rs            — compiler integration tests
  enhanced_v3_compiler_smoke.rs      — quick compiler smoke
  enhanced_v3_integrated.rs          — integrated pipeline tests
  enhanced_v3_proof_model.rs         — proof model contract tests
  enhanced_v3_qualification.rs       — qualification suite
  enhanced_v3_production_acceptance.rs — exact preset identity and 12-entry source acceptance
  support/enhanced_v3_compiler.rs    — compiler test support
```

### Enhanced v3 Profile

The Enhanced v3 profile is an additive, structurally disjoint generation path
in `src/enhanced_v3/`. It produces M2-only two-layer dungeons with cardinal +
45° chamfered-octagonal rooms, pointed-arch portal apertures, grounded
assemblies, and Sparse/Moderate/Rich density presets.

#### Quick Start

```rust
use bsp_generator::enhanced_v3::{generate_v3, V3Config, V3Preset};

// Generate a Sparse m3 dungeon (seed 42, 2048² extent)
let config = V3Config::new(42, V3Preset::Sparse, 2048)?;
let map_text = generate_v3(&config)?;

// Or use nominal constructors
let config = V3Config::nominal_moderate();
let map_text = generate_v3(&config)?;
```

#### V3Config API

`V3Config` validates at construction — no separate `validate()` step:

```rust
use bsp_generator::enhanced_v3::{V3Config, V3Preset};

// Full validation at construction
let config = V3Config::new(42, V3Preset::Rich, 3072)?;

// Convenience constructors:
let config = V3Config::nominal_sparse();    // seed 0, Sparse, 2048²
let config = V3Config::nominal_moderate();  // seed 0, Moderate, 2048²
let config = V3Config::nominal_rich();      // seed 0, Rich, 3072²
```

#### Vertical Contract (frozen)

Identical to Enhanced v2:

| parameter | value |
|-----------|-------|
| lower floor Z | 0 |
| upper floor Z | 192 |
| room height (both layers) | 176 |
| total Z span | 368 (≤ 384 M2 max) |
| layer count | 2 (frozen) |

#### RNG Domains

Enhanced v3 uses domain separator `"dungeon-gen/v3"` — **independent** from
Legacy v1's `"dungeon-gen/v1"` and Enhanced v2's `"dungeon-gen/v2"`. Stage tags:

| tag | stage |
|-----|-------|
| `v3-placement` | two-layer room placement with 45° footprint support |
| `v3-topology` | topology and transition selection |
| `v3-features` | chamfered footprints, pointed arches, grounded assemblies |
| `v3-detail` | preset-driven feature density, pillar placement |

#### Presets

| preset | exact rooms | same-layer routes | min families | min assemblies | min features | face budget |
|--------|-------------|-------------------|--------------|----------------|--------------|-------------|
| Sparse | 12 | 10 | 1 | 1 | 2 | 3,000 |
| Moderate | 20 | 20 | 3 | 3 | 6 | 5,000 |
| Rich | 28 | 30 | 6 | 6 | 12 | 8,000 |

#### Key Differences from Enhanced v2

| aspect | Enhanced v2 | Enhanced v3 |
|--------|-------------|-------------|
| domain | `"dungeon-gen/v2"` | `"dungeon-gen/v3"` |
| geometry | axis-aligned only | cardinal + 45° diagonal |
| room shape | rectangular only | chamfered/octagonal + rectangular |
| portals | rectangular aperture | pointed-arch aperture |
| assemblies | none | grounded support graph |
| grammar families | 2 strategies only | 6 descriptor families |
| presets | none (single config) | Sparse/Moderate/Rich |
| theme | cc0_dungeon_v2 | cc0_dungeon_v2 (reused) |
| minimum-identity | none | typed MinimumIdentityFailure |

#### Geometry Contract

- **Approved normals**: cardinal (axis-aligned) and exact 45° diagonal in XY plane
- **Wall thickness (cardinal)**: 16 Quake units
- **Wall thickness (45° diagonal)**: ≥ 16 Quake units perpendicular (32/√2 ≈ 22.63)
- **Integer arithmetic only**: i128 Rational geometry, no floating-point
- **Construction quantum**: 16 Quake units
- **Unapproved normals** (15°, 30°, arbitrary-angle): typed `UnapprovedNormal` error

#### Theme

Enhanced v3 reuses the CC0 Dungeon v2 theme at
`src/bsp_generator/themes/cc0_dungeon_v2/` — no new theme is created.
Texture roles map to WAD identities:

| BrushRole | WAD texture |
|-----------|------------|
| WallShell | `bs_wall` |
| FloorSlab | `bs_floor` |
| CeilingSlab | `bs_ceil` |
| Column | `bs_accent` |
| Feature | `bs_accent` |

#### Testing

```bash
cargo test -p bsp_generator --test enhanced_v3_contract_baseline  # 25 contract tests
cargo test -p bsp_generator --test enhanced_v3_budget             # M2 budget evidence
cargo test -p bsp_generator --test enhanced_v3_public_api         # API surface
cargo test -p bsp_generator --test enhanced_v3_semantic_core      # geometry + topology
```

## Key Design Rules

1. **No renderer/Vulkan/windowing deps**: This crate is an offline pipeline. If you need BSP loading or rendering, use `bsp_runtime` + `renderer` downstream.
2. **Pure functions**: Every stage is a function from immutable input to `Result<IR, Error>`. No global state, no lazy initialization, no hidden mutation.
3. **Typed IRs**: Each pipeline stage produces a named struct (e.g., `LayoutIntent`, `RoutedIntent`, `EmissionIntent`). These are pure data records — no generation logic lives in them.
4. **Determinism by construction**: The only source of randomness is `StageRng`. No `HashMap`/`HashSet` iterators affect output order. All serialization uses pre-sorted keys and faces.
5. **Exhaustion, not panic**: Every bounded loop has a hard limit and returns a typed error on exhaustion. Never use `unwrap()`, `expect()`, or `panic!()` in production code paths.
6. **Frozen values**: Constants from the specification are declared as `const` items. Changing a `const` is a contract violation.
7. **No inline texture generation**: The CC0 Stone Beta theme assets are pre-built by `build.py`. The generator references WAD texture names — it does not produce textures.
8. **Hybrid shell construction**: Quake brushes are additive. Emit each room's floor, ceiling, and four walls explicitly; split wall masks around omitted routed apertures. Use a corridor-only open-cell union for floors, ceilings, boundary walls, and 64×64 endpoint chambers so turns remain clear without creating scene-spanning room slabs.
9. **Point entities occupy clear volume**: Legacy spawn Z includes floor-slab thickness plus the 24-unit eye offset. Enhanced spawn is centered on a 64×64 lower stair landing, leaving the 16-unit Quake hull radius plus a 16-unit safety margin and facing the opening. Lights use the midpoint between floor-slab top and ceiling-slab bottom.

## Testing

### Run All Tests

```bash
cargo test -p bsp_generator
```

### Run Specific Test Categories

```bash
cargo test -p bsp_generator --test corpus_execution   # support corpus (needs ericw-tools)
cargo test -p bsp_generator --test determinism         # output determinism
cargo test -p bsp_generator --test config_validation   # config error paths
cargo test -p bsp_generator --test invariants          # output guarantees
```

### Corpus Execution Test

The `corpus_execution` test requires `ericw-tools 2.0.0-alpha3` installed at `~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/`. It generates all 12 support corpus configurations, compiles them through the BSP2 profile, and validates:
- Successful warning-free compilation (no qbsp/vis/light errors, warnings, missing textures, or skipped fill)
- BSP2 magic in output headers and strict parser reload
- Sealed maps (no pointfile leaks)
- Non-solid `point_contents` at room centers, point-entity origins, corridor centers, portal throats, and junction centers
- Deterministic byte-identical output across duplicate runs
- Face/entity ceilings and M2 tier separation (static-batch ceiling enforcement remains tracked by GitHub #57)

## Enhanced v2 Pipeline

The Enhanced v2 generator is an additive profile that produces M2-only, two-layer
dungeons with stairs, theme palette assignment, and corridor/ceiling/pillar
variance. It lives in `src/enhanced/` and is structurally disjoint from Legacy v1.

### Quick Start

```rust
use bsp_generator::enhanced::pipeline::generate_enhanced;
use bsp_generator::enhanced::config::EnhancedConfig;

// Generate a nominal M2 two-layer dungeon (28 rooms, 3 loops, 1 stair)
let cfg = EnhancedConfig::nominal();
let (map_text, meta) = generate_enhanced(42, cfg)?;

assert_eq!(meta.room_count, 28);
assert!(meta.transition_count > 0);
assert!(meta.light_count > 0);
```

### EnhancedConfig API

`EnhancedConfig` validates at construction — no separate `validate()` step:

```rust
use bsp_generator::enhanced::config::EnhancedConfig;

// Full validation at construction
let cfg = EnhancedConfig::new(28, 3, 1, 16, 2048)?;

// With explicit placement and feature parameters
let cfg = EnhancedConfig::with_full_params(
    28,   // room_count   (17–40, M2 only)
    3,    // loop_count   (1–6)
    1,    // vertical_edges (1–3)
    16,   // tread_depth  (fixed; other values are rejected)
    2048, // xy_extent    (≤ 3072, multiple of 16)
    32,   // placement_candidates
    96,   // max_placement_attempts
    2,    // max_pillars_per_room (0–8)
)?;
```

**Convenience constructors:**
- `EnhancedConfig::nominal()` — 28 rooms, 3 loops, 1 stair, 2048²
- `EnhancedConfig::minimal()` — 17 rooms, 1 loop, 1 stair, 1024²
- `EnhancedConfig::maximal()` — 40 rooms, 6 loops, 3 stairs, 3072²

### Vertical Contract (frozen)

| parameter | value |
|-----------|-------|
| lower floor Z | 0 |
| upper floor Z | 192 |
| room height (both layers) | 176 |
| riser | 16 |
| tread (both stair types) | 16 |
| total Z span | 368 (≤ 384 M2 max) |
| layer count | 2 (frozen) |

### RNG Domains

Enhanced v2 uses domain separator `"dungeon-gen/v2"` — **independent** from
Legacy v1's `"dungeon-gen/v1"`. Stage tags are:

| tag | stage |
|-----|-------|
| `layer-placement` | two-layer room placement |
| `vertical-topology` | topology and transition selection |
| `vertical-routing` | reserved for future vertical routing |
| `theme-assignment` | palette assignment |
| `feature-placement` | pillars, ceiling variance |
| `corridor-variance` | per-route corridor width |

### Theme Package

The Enhanced v2 theme is `themes/cc0_dungeon_v2/cc0_dungeon_v2.wad` — a
distinct CC0 theme from Legacy v1's `cc0_stone_beta.wad`. Both are project-authored,
CC0-licensed, and deterministically built via `build.py`.

### Key Differences from Legacy v1

| aspect | Legacy v1 | Enhanced v2 |
|--------|-----------|-------------|
| domain | `"dungeon-gen/v1"` | `"dungeon-gen/v2"` |
| map classes | M1 + M2 | M2 only |
| layers | 1 (flat) | 2 (lower + upper) |
| vertical connections | none | room-scale grand or wall-edge narrow stairs (12-tread sealed shells; full-run slab apertures and 64-unit crest throats) |
| config type | `DungeonConfig` + `validate()` | `EnhancedConfig` (validates at construction) |
| entry point | `generate(seed, config)` | `generate_enhanced(seed, config)` |
| room placement | same-layer, no membership balancing | two-layer, balanced membership |
| topology | Kruskal MST + loops (flat) | per-layer MST + loops + stair reservations |
| theme | CC0 Stone Beta (uniform) | CC0 Dungeon v2 (Uniform + ByZone strategies) |
| corridor width | fixed 64 | per-route variance (64/80/96) |
| ceiling height | fixed (z_span) | per-room variance (128/144/176) |
| pillars | none | per-room freestanding pillars |
| room roles | none | Entry / Hub / DeadEnd / Side |
| error type | `GeneratorError` | `EnhancedError` |
| metadata | `GenerationMetadata` | `EnhancedMetadata` |

### Corpus Evidence

Enhanced v2 tests live in the enhanced module (inline `#[cfg(test)] mod tests`)
and the `pipeline.rs` integration tests. Run with:

```bash
cargo test -p bsp_generator
```

Key tests include determinism (`generate_deterministic`), nominal/minimal/maximal
config generation, metadata population, and configuration validation errors.

## Adding a New Configuration Preset

1. Add a constructor to `DungeonConfig` in `config.rs`:
   ```rust
   impl DungeonConfig {
       pub fn my_preset() -> Self { ... }
   }
   ```
2. Add a validation test in `tests/config_validation.rs`.
3. If this preset is part of the support corpus, add it to `tests/corpus_execution.rs`.
4. Update the frozen specification if the preset represents a new contract value.

## Adding a New Theme

1. Create `themes/<name>/` with `theme.toml`, `build.py`, `palette.lmp`, and texture assets.
2. Add `LICENSE` if not CC0 (CC0 themes are preferred).
3. Update texture role constants in `emission.rs` to accept a theme parameter (currently hard-coded to CC0 Stone Beta).
4. Add theme evidence tests in `tests/theme_evidence.rs`.
5. Update the frozen specification.

## Common Pitfalls

- **Seed exhaustion**: `PlacementExhausted` and `RouteExhausted` are seed-dependent. A config that works with seed `0` may fail with seed `1` and vice versa. The support corpus guarantees specific seeds work — do not "fix" seeds that fail by changing bounds.
- **Quantum snapping**: All positions and dimensions must be multiples of 16. Use `snap_to_quantum()` for any computed coordinate. Debug builds assert quantum alignment via `debug_assert!`.
- **Additive brush misconception**: Overlap is union, not subtraction. A portal represented by a solid "opening brush" is still solid after `qbsp`; omit the aperture from the wall boundary.
- **Junction center obstruction**: L/T/X closure posts belong only in outer quadrants. Test the entire central 64×64 square, not only the exact center point.
- **Compiler warnings**: Generated output is accepted only when all three compiler stages are warning-free. The theme WAD includes compiler-only `skip`; never reintroduce `generator_brick` or an unbacked miptex.
- **Canonical face order**: bottom, top, north, south, west, east. This order is frozen. Changing it breaks byte-compatibility.
- **Stair aperture sizing**: Enhanced inter-layer slab apertures cover the complete 192-unit tread run. Keep the upper landing supported and align its first approach to a full 64-unit crest throat; a high-tread-only hole recreates the mid-flight overhang.
- **Entity order**: worldspawn first, then creation-index order. Do not sort entities by classname or origin.
- **Key order**: alphabetical by key string (ASCII byte order). Use `sort_unstable()` before emission.
- **`HashMap`/`HashSet`**: Never iterate over these in serialization paths. Use `BTreeMap` or pre-sort into `Vec`.

## Validation Commands

```bash
cargo check -p bsp_generator
cargo test -p bsp_generator
cargo clippy -p bsp_generator
cargo fmt --check -p bsp_generator
```

## Downstream Impact

Changes to this crate affect:
- `apps/bsp_beta/tests/generated_map_load.rs` — generated BSP load integration
- `tools/engine_pack/tests/bsp_generator_pipeline.rs` — compilation pipeline
- `.internal-dev/specifications/bsp-dungeon-generation.md` — frozen contract
- `.internal-dev/plans/bsp-dungeon-contract-evidence/evidence-matrix.md` — evidence cells
- `docs/guide/19-bsp-generator.md` — usage guide
- `docs/api/19-bsp-generator.md` — API reference
- `docs/internal/19-bsp-generator.md` — internal docs

When changing generator behavior, run the full test suite (including `--test corpus_execution`) and update affected docs and evidence matrix rows.

## See Also

- [Dungeon Generation Specification](../../.internal-dev/specifications/bsp-dungeon-generation.md)
- [Evidence Matrix](../../.internal-dev/plans/bsp-dungeon-contract-evidence/evidence-matrix.md)
- [Generator Usage Guide](../../docs/guide/19-bsp-generator.md)
- [Generator API Reference](../../docs/api/19-bsp-generator.md)
- [Generator Internals](../../docs/internal/19-bsp-generator.md)
