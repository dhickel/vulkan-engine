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

Builds the final emission intent by rasterizing the complete room/corridor open-space union on the 16-unit grid, merging floor and ceiling cells, and emitting walls only on the union boundary. Room portals and 64-unit-clear L/T/X centers are openings by omission rather than overlapping additive brushes. The spawn and room lights are placed inside clear volume above the floor slab. Every resulting brush is a rectangular prism with 6 faces in canonical order.

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

## Features

| feature | description |
|---------|-------------|
| `default` | no optional deps |
| `serde` | `serde::Serialize` + `serde::Deserialize` for all public types |

## See Also

- [BSP Generator Usage Guide](../guide/19-bsp-generator.md) — how to generate and compile dungeons
- [BSP Generator Internals](../internal/19-bsp-generator.md) — architecture, algorithms, and pipeline
- [BSP Beta API](17-bsp-beta.md) — BSP loading and rendering API
