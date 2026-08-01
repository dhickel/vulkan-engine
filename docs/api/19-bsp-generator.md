# BSP Dungeon Generator — API Reference

> Public API surface for the `bsp_generator` crate. Pure-Rust offline pipeline producing Quake 1 `.map` text from a seed and configuration.

## Audience

Rust developers integrating the dungeon generator into build pipelines, tools, or applications. Assumes familiarity with the [BSP Beta API](17-bsp-beta.md) for the loading/render side.

## Crate

```toml
[dependencies]
bsp_generator = { path = "src/bsp_generator" }
```

The crate depends only on `bsp` (format types) and `sha2` (deterministic seeding). It has **no** renderer, Vulkan, windowing, physics, audio, or scripting dependencies. An optional `serde` feature enables serialization of all public types.

## Entry Point

### `generate()`

```rust
pub fn generate(
    seed: u64,
    config: DungeonConfig,
) -> Result<(String, GenerationMetadata), GeneratorError>
```

The sole public entry point. Runs the full immutable-intent pipeline:

```
config validation → seed → place rooms → build topology →
route edges → build emission → serialize → (map_text, metadata)
```

**Parameters:**
- `seed`: Master `u64` seed. All random streams derive deterministically from this value.
- `config`: Raw dungeon configuration (validated internally).

**Returns:**
- `Ok((String, GenerationMetadata))`: Canonical `.map` text and metadata.
- `Err(GeneratorError)`: Validation failure, placement exhaustion, routing exhaustion, or internal error.

**Determinism guarantee:** Two calls with identical `(seed, config)` produce byte-identical `.map` output.

**Example:**
```rust
use bsp_generator::{generate, DungeonConfig};

let (map_text, meta) = generate(42, DungeonConfig::nominal_m1())?;
assert!(!map_text.is_empty());
assert_eq!(meta.room_count, 12);
```

## Configuration Types

### `DungeonConfig`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DungeonConfig {
    pub class: MapClass,
    pub room_count: u32,
    pub loop_count: u32,
    pub xy_bounds: (u32, u32),
    pub z_span: u32,
    pub placement_candidates: u32,
    pub max_placement_attempts: u32,
    pub max_astar_expansions: u32,
}
```

Raw user-supplied configuration. Call `validate()` to obtain a `ValidatedConfig`.

**Convenience constructors:**
- `DungeonConfig::nominal_m1()` — 12 rooms, 1 loop, 1024², Z 192
- `DungeonConfig::nominal_m2()` — 28 rooms, 3 loops, 2048², Z 256

### `DungeonConfig::validate()`

```rust
pub fn validate(&self) -> Result<ValidatedConfig, GeneratorError>
```

Validates all fields against the frozen per-class bounds. Checks:
- `room_count` and `loop_count` within class range
- `xy_bounds` non-zero, quantum-snapped (multiple of 16), ≤ class max
- `z_span` non-zero, quantum-snapped, ≤ class max
- `placement_candidates`, `max_placement_attempts`, `max_astar_expansions` non-zero, ≤ class max
- No arithmetic overflow in area/volume computations

### `ValidatedConfig`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedConfig {
    pub class: MapClass,
    pub room_count: u32,
    pub loop_count: u32,
    pub xy_bounds: (u32, u32),
    pub z_span: u32,
    pub placement_candidates: u32,
    pub max_placement_attempts: u32,
    pub max_astar_expansions: u32,
}
```

A fully validated configuration. Obtainable only via `DungeonConfig::validate()`.

### `MapClass`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapClass {
    M1,   // 8–16 rooms, 0–2 loops, ≤1536²
    M2,   // 17–40 rooms, 1–6 loops, ≤3072²
}
```

**Methods:**
- `max_placement_candidates() -> u32` — M1: 16, M2: 32
- `max_placement_attempts() -> u32` — M1: 64, M2: 96
- `max_astar_expansions() -> u32` — M1: 131,072, M2: 524,288

### Constants

```rust
pub const CONSTRUCTION_QUANTUM: u32 = 16;   // All geometry snaps to this

// M1 frozen bounds
pub const M1_ROOM_COUNT_MIN: u32 = 8;
pub const M1_ROOM_COUNT_MAX: u32 = 16;
pub const M1_LOOP_COUNT_MIN: u32 = 0;
pub const M1_LOOP_COUNT_MAX: u32 = 2;
pub const M1_XY_MAX: u32 = 1536;
pub const M1_Z_MAX: u32 = 256;

// M2 frozen bounds
pub const M2_ROOM_COUNT_MIN: u32 = 17;
pub const M2_ROOM_COUNT_MAX: u32 = 40;
pub const M2_LOOP_COUNT_MIN: u32 = 1;
pub const M2_LOOP_COUNT_MAX: u32 = 6;
pub const M2_XY_MAX: u32 = 3072;
pub const M2_Z_MAX: u32 = 384;
```

## Seed Types

### `Seed`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seed(u64);
```

Master seed for a generation run. All random streams derive deterministically from this single `u64`.

**Methods:**
- `Seed::new(value: u64) -> Self` — create from raw u64
- `Seed::raw(self) -> u64` — return raw value
- `Seed::stage_seed(&self, tag: &str) -> StageSeed` — derive per-stage sub-seed

**Implements:** `From<u64>`

### `StageSeed`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageSeed {
    pub digest: [u8; 32],
}
```

A 32-byte SHA-256 sub-seed derived from a `Seed` and semantic stage tag. Derivation formula:

```
SHA-256("dungeon-gen/v1" || seed_le_bytes || tag)
```

**Frozen tags:** `"room-placement"`, `"corridor-routing"`, `"entity-placement"`, `"light-placement"`.

**Methods:**
- `u64_at(&self, index: usize) -> u64` — read u64 at index 0..=3 (little-endian)
- `rng(&self) -> StageRng` — create a deterministic RNG from this sub-seed

### `StageRng`

```rust
#[derive(Clone)]
pub struct StageRng { /* private fields */ }
```

Deterministic random-number generator backed by SHA-256-chained 32-byte state. Produces an infinite stream of `u64` values.

**Methods:**
- `next_u64(&mut self) -> u64` — next u64 from the stream
- `range_u32(&mut self, range: u32) -> u32` — u32 in `[0, range)` (rejection sampling)
- `range_inclusive(&mut self, lo: u32, hi: u32) -> u32` — u32 in `[lo, hi)`

## Generation Metadata

### `GenerationMetadata`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationMetadata {
    pub room_count: u32,
    pub corridor_count: u32,
    pub entity_count: u32,
    pub face_count_estimate: u32,
    pub bounds: (i32, i32, i32, i32, i32, i32),
    pub seed: u64,
    pub config_hash: u64,
}
```

Descriptive metadata returned alongside generated `.map` text.

| field | description |
|-------|-------------|
| `room_count` | number of placed rooms |
| `corridor_count` | number of corridor segments |
| `entity_count` | non-worldspawn entities (spawn + lights) |
| `face_count_estimate` | `brushes × 6` (every brush is a rectangular prism) |
| `bounds` | `(min_x, min_y, min_z, max_x, max_y, max_z)` AABB in Quake units |
| `seed` | master seed used for this run |
| `config_hash` | deterministic SHA-256 hash of the validated config |

## Error Type

### `GeneratorError`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratorError {
    InvalidConfig(String),
    PlacementExhausted { attempts: u32 },
    RouteExhausted { expansions: u32 },
    InvariantViolation(String),
    SerializationFailed(String),
    ArithmeticOverflow,
}
```

All errors the generator can produce. Implements `std::error::Error`, `Display`, `Debug`.

| variant | cause | likely fix |
|---------|-------|------------|
| `InvalidConfig(msg)` | field out of frozen bounds | fix configuration |
| `PlacementExhausted { attempts }` | room placement budget exhausted | try different seed, increase candidates |
| `RouteExhausted { expansions }` | corridor A* budget exhausted | try different seed, increase expansions |
| `InvariantViolation(msg)` | internal generator bug | report bug |
| `SerializationFailed(msg)` | map formatting overflow | report bug |
| `ArithmeticOverflow` | bounds computation overflow | report bug |

## Intent Pipeline Types (Internal)

These types are public for transparency and testing but are primarily consumed by the generate pipeline internally.

### `RoomIntent`

```rust
pub struct RoomIntent {
    pub position: (i32, i32, i32),
    pub dimensions: (u32, u32, u32),
}
```

A placed room whose position and dimensions are multiples of the 16-unit construction quantum.

### `LayoutIntent`

```rust
pub struct LayoutIntent {
    pub rooms: Vec<RoomIntent>,
    pub edges: Vec<(usize, usize)>,
    pub loop_count: u32,
}
```

Placed rooms with their connectivity graph (MST + loops).

### `Corridor`

```rust
pub struct Corridor {
    pub start: (i32, i32, i32),
    pub end: (i32, i32, i32),
    pub width: u32,
    pub height: u32,
}
```

An axis-aligned straight corridor segment.

### `Junction`

```rust
pub struct Junction {
    pub position: (i32, i32, i32),
}
```

A junction point where corridors meet.

### `RoutedIntent`

```rust
pub struct RoutedIntent {
    pub corridors: Vec<Corridor>,
    pub junctions: Vec<Junction>,
}
```

The routed corridor network with junction nodes.

### `Brush`, `BrushFace`

```rust
pub struct BrushFace {
    pub plane_points: [(i32, i32, i32); 3],
    pub texture: String,
    pub u_axis: [i32; 4],
    pub v_axis: [i32; 4],
}

pub struct Brush {
    pub faces: Vec<BrushFace>,
}
```

Convex brush geometry: an ordered set of faces, each defined by three non-collinear plane points and texture mapping.

### `EntityIntent`

```rust
pub struct EntityIntent {
    pub classname: String,
    pub origin: (i32, i32, i32),
    pub properties: Vec<(String, String)>,
    pub brushes: Vec<Brush>,
}
```

A Quake entity with classname, origin, key-value properties, and optional brush geometry.

### `EmissionIntent`

```rust
pub struct EmissionIntent {
    pub brushes: Vec<Brush>,
    pub entities: Vec<EntityIntent>,
    pub wad: String,
}
```

The final emission-ready representation: worldspawn brushes, non-worldspawn entities, and WAD reference.

## Stage Functions (Advanced)

These functions compose the generation pipeline. They are public for testing but are typically not called directly.

### `place_rooms()`

```rust
pub fn place_rooms(
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<Vec<RoomIntent>, GeneratorError>
```

Bounded room placement with grid-based spatial reservation. For each of `config.room_count` rooms, generates up to `config.placement_candidates` random candidates per attempt, clamps horizontal spans to the 112–160-unit range required for an 80-unit clear interior, and accepts the first candidate that does not overlap previous rooms. Exhausts after `config.max_placement_attempts` attempts per room.

### `build_topology()`

```rust
pub fn build_topology(
    rooms: Vec<RoomIntent>,
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<LayoutIntent, GeneratorError>
```

Builds a minimum spanning tree over room centers (Kruskal's algorithm), then adds exactly `config.loop_count` extra edges from the remaining non-MST edges.

### `route_all_edges()`

```rust
pub fn route_all_edges(
    rooms: &[RoomIntent],
    edges: &[(usize, usize)],
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<RoutedIntent, GeneratorError>
```

Routes all topology edges into axis-aligned corridor segments. Tries direct and L-shaped orthogonal paths first, then falls back to bounded A* on the quantum grid. Returned routes include the normal approach legs from the offset routing grid to the actual room-wall portals.

### `build_emission()`

```rust
pub fn build_emission(
    layout: &LayoutIntent,
    routed: &RoutedIntent,
) -> EmissionIntent
```

Builds the final emission intent with explicit role-bound room shells and a corridor-only 16-unit grid union. Each room gets floor and ceiling slabs plus four full-height wall masks split around omitted portal apertures; corridor floors, ceilings, boundary walls, and 64×64 endpoint chambers remain connected without extending the low corridor ceiling into room interiors. The spawn and room lights are placed inside clear volume above the floor slab. Every resulting brush is a rectangular prism with 6 faces in canonical order.

### `serialize()`

```rust
pub fn serialize(emission: &EmissionIntent) -> String
```

Serializes an `EmissionIntent` to canonical `.map` text following the frozen serialization contract (worldspawn-first, alphabetical keys, creation-index brushes, canonical face order, LF line endings).

## Theme Constants

```rust
pub const CC0_STONE_BETA_THEME_DIR: &str = "themes/cc0_stone_beta";
```

Relative path from crate root to the CC0 Stone Beta theme directory.

## Routing Constants

```rust
pub const CORRIDOR_WIDTH: u32 = 64;    // Minimum clear interior width (4 quanta)
pub const CORRIDOR_HEIGHT: u32 = 80;   // Minimum clear interior height (5 quanta)
```

## Enhanced v2 API

The Enhanced v2 profile lives in `bsp_generator::enhanced`. It produces M2-only,
two-layer dungeons with stairs, theme variance, and corridor/ceiling/pillar
features.

### `generate_enhanced()`

```rust
pub fn generate_enhanced(
    seed: u64,
    config: EnhancedConfig,
) -> Result<(String, EnhancedMetadata), EnhancedError>
```

The Enhanced v2 entry point. Runs the full pipeline:

```text
EnhancedConfig → placement → topology → theme assignment →
feature variance → emission → (map_text, metadata)
```

**Determinism guarantee:** Two calls with identical `(seed, config)` produce
byte-identical `.map` output. The Enhanced v2 RNG domain (`"dungeon-gen/v2"`) is
independent from Legacy v1.

### `EnhancedConfig`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancedConfig { /* private fields */ }
```

Validated M2-only two-layer configuration. Validates all fields at construction —
no separate `validate()` step.

**Constructors:**

| constructor | description |
|-------------|-------------|
| `EnhancedConfig::new(room_count, loop_count, vertical_edges, tread_depth, xy_extent)` | standard constructor |
| `EnhancedConfig::with_placement_params(..., candidates, attempts)` | with placement tuning |
| `EnhancedConfig::with_full_params(..., candidates, attempts, max_pillars)` | full parameter set |
| `EnhancedConfig::nominal()` | 28 rooms, 3 loops, 1 stair, 2048² |
| `EnhancedConfig::minimal()` | 17 rooms, 1 loop, 1 stair, 1024² |
| `EnhancedConfig::maximal()` | 40 rooms, 6 loops, 3 stairs, 3072² |

**Accessors:**

| accessor | return type | description |
|----------|-------------|-------------|
| `room_count()` | `u32` | total rooms (17–40) |
| `loop_count()` | `u32` | extra loop edges (1–6) |
| `vertical_edges()` | `u32` | stair connections (1–3) |
| `tread_depth()` | `i32` | fixed stair tread depth (always 16) |
| `xy_extent()` | `u32` | XY bounds per axis |
| `placement_candidates()` | `u32` | candidates per room attempt |
| `max_placement_attempts()` | `u32` | max attempts per room |
| `max_pillars_per_room()` | `u32` | max pillars per room (0–8) |
| `layer_count()` | `u32` | always returns 2 |
| `lower_floor_z()` | `i32` | always returns 0 |
| `upper_floor_z()` | `i32` | always returns 192 |
| `room_height()` | `i32` | always returns 176 |
| `riser()` | `i32` | always returns 16 |

### `EnhancedMetadata`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancedMetadata {
    pub room_count: u32,
    pub route_count: u32,
    pub transition_count: u32,
    pub lower_floor_z: i32,
    pub upper_floor_z: i32,
    pub spawn_origin: (i32, i32, i32),
    pub light_count: u32,
    pub pillar_count: u32,
    pub seed: u64,
}
```

### `EnhancedError`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnhancedError {
    ConfigOutOfRange { field: &'static str, value: u64, min: u64, max: u64 },
    WrongProfile { expected: &'static str },
    ArithmeticOverflow { operation: &'static str },
    DuplicateId { kind: &'static str, id: u32 },
    IdOutOfOrder { kind: &'static str, id: u32, previous: u32 },
    ContractViolation { detail: String },
    PlacementExhausted { rooms_placed: u32, total_attempts: u32 },
    RoomTooLarge { room_index: u32, width: u32, height: u32, xy_extent: u32 },
    TopologyExhausted { detail: String },
    RouteExhausted { expansions: u32 },
    TransitionReservationFailed { detail: String },
    TopologyValidationFailed { detail: String },
}
```

### `StairType`

```rust
pub enum StairType {
    RoomScaleGrand,
    WallEdgeNarrow,
}
```

`RoomScaleGrand` uses a 192-unit run across the host room's full usable width.
`WallEdgeNarrow` uses a 192×64-unit strip along a room wall. Both types emit
exactly twelve 16-unit treads and risers. Each transition carries exact tread
boxes, lower and upper approach segments, and lower-wall, ceiling, and
upper-wall aperture descriptors. The slab aperture covers the complete
192-unit run, while the supported upper landing preserves 80-unit headroom and
a full 64-unit crest throat. Enhanced emission centers `info_player_start`
on the canonical 64×64 lower stair landing, reserves 16 units for the Quake
player hull plus a 16-unit safety margin on every landing side, and emits a
cardinal `angle` facing the stair opening.

### `GenerationProfile`

```rust
pub enum GenerationProfile {
    LegacyV1,
    EnhancedV2,
}
```

Dispatch discriminator for the `GenerationRequest` enum. Each variant carries
its own config type (`DungeonConfig` for LegacyV1, `EnhancedConfig` for
EnhancedV2).

### Enhanced RNG Types

```rust
// Master seed
pub struct EnhancedSeed(u64);

// Stage sub-seed (32-byte SHA-256 digest)
pub struct EnhancedStageSeed { pub digest: [u8; 32] }

// SHA-256-chained deterministic RNG
pub struct EnhancedStageRng { /* private */ }
```

Domain separator: `"dungeon-gen/v2"`. Stage tags: `layer-placement`,
`vertical-topology`, `vertical-routing`, `theme-assignment`,
`feature-placement`, `corridor-variance`.

### Enhanced Constants

```rust
pub const ENHANCED_LOWER_FLOOR_Z: i32 = 0;
pub const ENHANCED_UPPER_FLOOR_Z: i32 = 192;
pub const ENHANCED_ROOM_HEIGHT: i32 = 176;
pub const ENHANCED_RISER: i32 = 16;
pub const ENHANCED_TREAD: i32 = 16;
pub const ENHANCED_TREAD_DEFAULT: i32 = ENHANCED_TREAD; // compatibility alias
pub const ENHANCED_LAYER_COUNT: u32 = 2;
pub const ENHANCED_MIN_ROOM_SPAN: i32 = 112;
pub const ENHANCED_MAX_ROOM_SPAN: i32 = 256;
pub const SOCKET_APERTURE: i32 = 64;
pub const SOCKET_CORNER_MARGIN: i32 = 32;
```

## Features

| feature | description |
|---------|-------------|
| `default` | no optional deps |
| `serde` | `serde::Serialize` + `serde::Deserialize` for all public types |

## See Also

- [BSP Generator Usage Guide](../guide/19-bsp-generator.md) — how to generate and compile dungeons
- [BSP Generator Internals](../internal/19-bsp-generator.md) — architecture, algorithms, and pipeline
- [BSP Beta API](17-bsp-beta.md) — BSP loading and rendering API

## Enhanced v3 API

The Enhanced v3 profile lives in `bsp_generator::enhanced_v3`. It produces M2-only,
two-layer dungeons with cardinal + 45° geometry, pointed-default plus
rectangular/segmented cardinal portal surrounds, grounded assemblies, and
Sparse/Moderate/Rich density presets.

### `generate_v3()`

```rust
pub fn generate_v3(config: &V3Config) -> Result<String, V3Error>
```

The Enhanced v3 entry point. Runs the full pipeline:

```text
V3Config → footprints → topology → reservations → assembly → .map text
```

**Determinism guarantee:** Two calls with identical `V3Config` produce byte-identical
`.map` output. The Enhanced v3 RNG domain (`"dungeon-gen/v3"`) is independent from
Legacy v1 and Enhanced v2.

### `run_pipeline()`

```rust
pub fn run_pipeline(config: &V3Config) -> Result<V3PipelineOutput, V3Error>
```

Returns the full pipeline output including metadata:

```rust
pub struct V3PipelineOutput {
    pub map_text: String,
    pub metadata: EnhancedV3Metadata,
}
```

### `V3Config`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3Config {
    pub seed: u64,
    pub preset: V3Preset,
    pub xy_extent: u32,
    pub rooms: Option<u32>,
    pub corridors: Option<u32>,
    pub loops: Option<u32>,
    pub vertical_edges: Option<u32>,
    pub chamfer: bool,
    pub arch_type: ArchType,
    pub stairs: bool,
    pub room_span_min: Option<u32>,
    pub room_span_max: Option<u32>,
    pub grammar_families: Vec<String>,
    pub grammar_mode: GrammarMode,
    pub features: FeatureFlags,
    pub feature_density: f32,
    pub minlight: u32,
    pub light_count: Option<u32>,
}
```

M2-only two-layer configuration. `new()` creates and validates the byte-compatible
production defaults. Explorer fields are public overrides; call `validate()` after
mutating them. `run_pipeline()` and package entry points revalidate before generation.

**Constructors:**

| constructor | description |
|-------------|-------------|
| `V3Config::new(seed, preset, xy_extent)` | standard validated constructor |
| `V3Config::nominal_sparse()` | seed 0, Sparse, 2048² |
| `V3Config::nominal_moderate()` | seed 0, Moderate, 2048² |
| `V3Config::nominal_rich()` | seed 0, Rich, 3072² |

**Validation rules:**
- `xy_extent` must be 1024–3072 and a multiple of 16
- Mutated explorer fields must be revalidated by `validate()`; `run_pipeline()` and package entry points do this before generation
- `ArchType::None`, `Pointed`, and `Segmented` are accepted cardinal surrounds; segmented generation adds a corridor-side crown cap while retaining the complete 64×80 throat
- Non-quantum-aligned values produce `V3Error::ConfigNotQuantumAligned`
- Out-of-range values produce `V3Error::ConfigOutOfRange`

### `V3Preset`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum V3Preset {
    Sparse,
    Moderate,
    Rich,
}
```

**Methods:**

| method | Sparse | Moderate | Rich |
|--------|--------|----------|------|
| `tag()` | `"sparse"` | `"moderate"` | `"rich"` |
| `min_rooms()` | 12 | 20 | 28 |
| `target_loops()` | 0 | 2 | 4 |
| `minimum_families()` | 1 | 3 | 6 |
| `minimum_assemblies()` | 1 | 3 | 6 |
| `minimum_feature_brushes()` | 2 | 6 | 12 |
| `face_budget()` | 3,000 | 5,000 | 8,000 |

For every supported production seed/configuration, the current pipeline emits
exactly 12/20/28 rooms and 10/20/30 same-layer routes for
Sparse/Moderate/Rich. The methods retain `min_` names because they define the
validated preset contract; production acceptance treats those counts as exact.

**Parse from tag:**
```rust
V3Preset::from_tag("sparse")    // => Some(V3Preset::Sparse)
V3Preset::from_tag("dense")     // => None
```

### `ArchType`

`ArchType::from_tag()` accepts `"none"`, `"pointed"`, and `"segmented"`.
`cycle()` advances `Pointed → Segmented → None → Pointed`. `Pointed` remains
the byte-compatible default; the other variants affect only explicit explorer
configurations.

### `NormalClass`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormalClass {
    Cardinal,     // axis-aligned: ±X, ±Y, ±Z
    Diagonal45,   // exact 45° in XY: (±1, ±1, 0)
    Unapproved,   // all other normals produce typed errors
}
```

Classify an integer normal vector with `classify_normal(nx, ny, nz)`.

### `V3Error`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3Error {
    // Configuration errors
    ConfigOutOfRange { field, value, min, max },
    ConfigNotQuantumAligned { field, value, quantum },
    UnknownPreset { tag },
    // RNG errors
    ZeroBound,
    RejectionStreamExhausted,
    // Geometry errors
    UnapprovedNormal { nx, ny, nz },
    CoincidentPoints { p0, p1, p2 },
    CollinearPoints { p0, p1, p2 },
    ZeroVolume,
    EmptyIntersection,
    Unbounded,
    DegenerateIntersection,
    DuplicatePlane { existing, duplicate },
    InactivePlane { plane },
    FaceTooSmall { face, area },
    EdgeTooShort { edge, length },
    InsufficientThickness { direction, thickness },
    ArithmeticOverflow { operation },
    ZeroDenominator,
    MalformedRole { detail },
    NotGridAligned { coord, quantum },
    // Topology errors
    TopologyInvariant { detail },
    RoomOutOfBounds { room_id, extent },
    // Composition errors
    MinimumIdentityFailure { preset, required, actual },
    SupportGraphCycle { members },
    CompositionInvariant { detail },
    // Assembly errors
    PositiveVolumeOverlap { brush_a, brush_b },
    UndeclaredContact { brush_a, brush_b, plane },
    MissingInterface { interface_id, brush_a, brush_b },
    UnsupportedBrush { id },
    ProtectedVolumeIntrusion { brush_id, protected_id },
    ApertureInvalid { aperture_id, detail },
    DuplicateBrushId { id },
    UnknownBrush { id },
    AssemblyValidation { detail },
    // Emission errors
    UnvalidatedAssembly,
    EmissionInvariant { detail },
    // Reservation errors
    ReservationConflict { resource, existing },
    InvalidReservation { detail },
}
```

45 typed variants across 9 categories. Implements `std::error::Error`, `Display`, `Debug`.

### `EnhancedV3Metadata`

```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct EnhancedV3Metadata { /* private fields */ }
```

Read-only accessors:

| accessor | return type | description |
|----------|-------------|-------------|
| `seed()` | `u64` | master seed |
| `preset()` | `&str` | density preset name |
| `xy_extent()` | `u32` | requested XY extent |
| `schema_version()` | `&str` | always `"v3"` |
| `generator()` | `&str` | `"bsp_generator/enhanced_v3"` |
| `room_count()` | `u32` | total rooms |
| `lower_room_count()` | `u32` | rooms on lower layer |
| `upper_room_count()` | `u32` | rooms on upper layer |
| `portal_count()` | `u32` | committed portals |
| `transition_count()` | `u32` | stair transitions |
| `route_count()` | `u32` | committed routes |
| `grammar_families()` | `&[String]` | families in output |
| `identity_satisfied()` | `bool` | minimum-identity met |
| `estimated_faces()` | `u32` | conservative face estimate |
| `actual_faces()` | `u32` | emitted face count |
| `estimated_entities()` | `u32` | entity estimate |
| `actual_entities()` | `u32` | emitted entity count |
| `actual_brushes()` | `u32` | emitted brush count |
| `spawn_origin()` | `(i32,i32,i32)` | info_player_start origin |
| `light_count()` | `u32` | light entities |
| `bounds()` | `(i32,i32,i32,i32,i32,i32)` | AABB in Quake units |
| `has_upper_layer()` | `bool` | upper layer present |

### Profile Dispatch

`GenerationProfile` provides dispatch tags for all three pipeline variants:

```rust
pub enum GenerationProfile {
    LegacyV1,      // tag: "legacy-v1" → from_tag("m1") → Some
    EnhancedV2,    // tag: "enhanced-v2" → from_tag("m2") → Some
    EnhancedV3,    // tag: "m3"
}
```

Production dispatch uses `from_tag("m3")` → `Some(EnhancedV3)`.
The proof-only tag `"enhanced-v3"` returns `None`.
The `dungeon_gen` CLI uses `--class m3` and exposes every explorer field through
`--preset`, `--extent`, `--rooms`, `--corridors`, `--loops`, `--vertical-edges`,
`--chamfer`/`--no-chamfer`, `--arch-type`, `--stairs`/`--no-stairs`, room-span,
grammar, feature-density, minlight, and light-count options.

### Enhanced v3 Constants

```rust
// Construction
pub const CONSTRUCTION_QUANTUM: i32 = 16;
pub const ROUTE_WIDTH: i32 = 64;
pub const HEADROOM: i32 = 80;
pub const WALL_THICKNESS: i32 = 16;

// Two-layer M2 arrangement
pub const LOWER_FLOOR_Z: i32 = 0;
pub const UPPER_FLOOR_Z: i32 = 192;
pub const ROOM_HEIGHT: i32 = 176;
pub const TOTAL_Z_SPAN: i32 = 368;
pub const LAYER_COUNT: u32 = 2;

// XY bounds
pub const XY_MAX: u32 = 3072;
pub const XY_MIN: u32 = 1024;

// Budget ceilings
pub const FACE_BUDGET: u32 = 10000;
pub const ENTITY_BUDGET: u32 = 300;
pub const MAX_FACES_PER_FEATURE: u32 = 200;
pub const MAX_ENTITIES_PER_ROOM: u32 = 5;
```

### Texture Roles

```rust
pub fn texture_for_role(role: BrushRole) -> &'static str
```

| BrushRole | WAD texture |
|-----------|------------|
| `WallShell` | `"bs_wall"` |
| `FloorSlab` | `"bs_floor"` |
| `CeilingSlab` | `"bs_ceil"` |
| `Column` | `"bs_accent"` |
| `Feature` | `"bs_accent"` |

## See Also

- [BSP Generator Usage Guide](../guide/19-bsp-generator.md) — how to generate and compile dungeons
- [BSP Generator Internals](../internal/19-bsp-generator.md) — architecture, algorithms, and pipeline
- [BSP Beta API](17-bsp-beta.md) — BSP loading and rendering API
