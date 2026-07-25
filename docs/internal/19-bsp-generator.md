# BSP Dungeon Generator — Internal Reference

> Architecture, intent pipeline, algorithms, and serialization contract for the `bsp_generator` crate. For engine maintainers working inside `src/bsp_generator/`.

## 1. Purpose & Audience

This chapter is for contributors modifying or debugging the dungeon generator. Assumes Rust proficiency and familiarity with the frozen generation contract in [`.internal-dev/specifications/bsp-dungeon-generation.md`](../../.internal-dev/specifications/bsp-dungeon-generation.md). The generator is a pure function from `(u64 seed, DungeonConfig)` to canonical `.map` bytes with zero runtime dependencies beyond `bsp` and `sha2`.

## 2. Crate Layering

```
bsp_generator (depends on bsp, sha2)
  → produces .map text (String)
  → zero renderer/Vulkan/windowing dependencies
  → optional serde feature for type serialization

↓ consumed by

engine_pack compile-bsp (depends on bsp, engine_pack)
  → produces .bsp + .lit via ericw-tools external invocation

bsp_beta / dungeon_dogfood (depends on renderer, bsp_runtime)
  → loads and renders compiled BSP output
```

The generator **must never** gain a dependency on `renderer`, `vulkan`, `winit`, `physics`, `audio`, `scripting`, `bsp_runtime`, or `engine_pack`. It is an offline pipeline.

## 3. Intent Pipeline

Every stage produces an immutable intermediate representation (IR). No stage mutates the output of a previous stage.

```
DungeonConfig
    │  .validate()
    ▼
ValidatedConfig  ────────────────────────────────┐
    │                                             │
    │  Seed::new(seed)                            │
    ▼                                             │
Seed ──► StageSeed("room-placement") ──► StageRng │
    │                                             │
    │  place_rooms()                              │
    ▼                                             │
Vec<RoomIntent>                                   │
    │                                             │
    │  StageSeed("corridor-routing") ──► StageRng │
    │  build_topology()                           │
    ▼                                             │
LayoutIntent { rooms, edges, loop_count }         │
    │                                             │
    │  route_all_edges()  (same RNG stream)       │
    ▼                                             │
RoutedIntent { corridors, junctions }             │
    │                                             │
    │  build_emission()                           │
    ▼                                             │
EmissionIntent { brushes, entities, wad }         │
    │                                             │
    │  serialize()                                │
    ▼                                             │
String  (.map bytes)                              │
    │
    │  compute_bounds(), compute_config_hash()
    ▼
(String, GenerationMetadata)
```

### Stage Ownership

| stage | input | output | module | error paths |
|-------|-------|--------|--------|-------------|
| validate | `DungeonConfig` | `ValidatedConfig` | `config.rs` | `InvalidConfig` |
| seed | `u64` | `Seed` | `seed.rs` | none (infallible) |
| placement | `ValidatedConfig`, `StageRng` | `Vec<RoomIntent>` | `placement.rs` | `PlacementExhausted` |
| topology | `Vec<RoomIntent>`, `ValidatedConfig`, `StageRng` | `LayoutIntent` | `topology.rs` | `InvariantViolation` |
| routing | `&[RoomIntent]`, `&[(usize,usize)]`, `ValidatedConfig`, `StageRng` | `RoutedIntent` | `routing.rs` | `RouteExhausted` |
| junction | corridor/room intersections | outer wall posts and portal wall pieces | `junction.rs` | none (pure geometry) |
| emission | `&LayoutIntent`, `&RoutedIntent` | `EmissionIntent` | `emission.rs` | none (infallible) |
| serialize | `&EmissionIntent` | `String` | `serialize.rs` | `SerializationFailed` |

## 4. Room Placement (`placement.rs`)

### Algorithm

1. For each room index 0..`room_count`:
   a. For attempt 0..`max_placement_attempts`:
      - Generate `placement_candidates` random candidate positions within XY bounds, snapped to 16-unit quantum
      - Consume the frozen 2–10-quanta random dimension draw, then clamp each horizontal span to 7–10 quanta (112–160 Quake units) so a room has an 80-unit clear interior
      - Test each candidate against all previously placed rooms using `rooms_overlap()` with `WALL_THICKNESS = 16`
      - Accept first non-overlapping candidate
   b. If no candidate succeeds after all attempts: return `PlacementExhausted`
2. Return `Vec<RoomIntent>` in placement order

### Spatial Reservation

Rooms are axis-aligned rectangles. The `rooms_overlap()` check expands each room by `WALL_THICKNESS/2` in all four horizontal directions and compares bounding boxes. Z overlap does not apply (single-layer contract — all rooms share the same floor and ceiling planes).

### Bounds

- `WALL_THICKNESS`: 16 units (1 quantum)
- Room horizontal dimensions: 7–10 quanta per axis (112–160 units)
- XY positions: `[WALL_THICKNESS, xy_bounds - room_width - WALL_THICKNESS]`
- Z: `floor = 0`, `ceiling = z_span`

## 5. Topology Construction (`topology.rs`)

### Algorithm (Kruskal's MST + loop augmentation)

1. Compute room centers: `position + dimensions / 2`
2. Build complete graph with Euclidean distance edge weights
3. Run Kruskal's algorithm to extract the minimum spanning tree (MST)
4. From the remaining non-MST edges, select exactly `loop_count` edges to add
   - Selection uses the `corridor-routing` RNG stream
   - Edges are drawn without replacement from a shuffled pool
5. Return `LayoutIntent { rooms, edges, loop_count }`

### Guarantees

- All rooms in a single connected component (MST property)
- Exact `loop_count` extra edges (cycle creation)
- Deterministic for identical `(rooms, seed, config)`

## 6. Corridor Routing (`routing.rs`)

### Frozen Dimensions

```rust
pub const CORRIDOR_WIDTH: u32 = 64;   // 4 quanta — clear interior
pub const CORRIDOR_HEIGHT: u32 = 80;  // 5 quanta — clear interior
```

### Algorithm

For each edge `(room_a_idx, room_b_idx)`:

1. Compute 64-unit portal centers on the two facing room walls and reserve normal approach lanes through the endpoint occupancy margins.
2. **Direct path attempt**: If the two offset centerline points are axis-aligned, emit a straight route.
3. **L-shaped path attempt**: Try both L-path orders (horizontal-then-vertical, vertical-then-horizontal). Check that the corner is not inside any non-endpoint room's expanded bounding box.
4. **A* fallback**: If direct and L-shaped paths fail, run bounded A* on the quantum grid. The cost function is Manhattan distance. Search terminates after `max_astar_expansions` node expansions.
5. Prepend and append the reserved approach legs so the routed intent reaches the actual room-wall portals, then simplify the complete orthogonal path into corridor segments.

### Quantum Grid

The A* grid resolution is `CONSTRUCTION_QUANTUM = 16` Quake units. The grid is axis-aligned. Obstacles are rooms expanded by `WALL_THICKNESS + CORRIDOR_HALF_CELLS` to prevent corridors from grazing room walls.

### Exhaustion

Returns `RouteExhausted { expansions }` if any edge cannot be routed within the expansion budget. This is seed-dependent — retrying with a different seed or increasing `max_astar_expansions` may succeed.

## 7. Junction Geometry (`junction.rs`)

Quake world brushes are additive: overlapping a corridor with a room-wall brush cannot subtract a doorway. Portal helpers therefore return only the solid wall columns and lintel around an omitted aperture. Junction helpers place closure posts only in outer wall-thickness quadrants; no helper brush may enter the central 64×64 clear square. Production emission applies the same rule through the floor-plan union described in §8.

### Junction Types

| type | description | helper brush count |
|------|-------------|--------------------|
| **Room portal** | corridor meets room wall | up to 3 wall pieces around the opening |
| **L-junction** | two perpendicular corridors meet at corner | 1 outer-corner post |
| **T-junction** | one corridor terminates into through corridor | 2 outer-corner posts |
| **X-junction** | two corridors cross at right angles | 4 outer-corner posts |

### `make_brush()`

```rust
pub fn make_brush(
    min: (i32, i32, i32),
    max: (i32, i32, i32),
    texture: &str,
) -> Brush
```

Produces an axis-aligned rectangular prism with exactly 6 faces in canonical order: **bottom, top, north, south, west, east**. Each face is defined by three non-collinear integer plane points and `[1 0 0 0]` texture axes.

## 8. Emission (`emission.rs`)

### Texture Roles

The CC0 Stone Beta theme provides four visual texture roles plus one compiler-only surface:

| role | WAD name | used on |
|------|----------|---------|
| floor | `stone_floor` | room and corridor floor slabs |
| wall | `stone_wall` | boundary walls, outer junction posts, portal lintels |
| ceiling | `stone_ceiling` | room and corridor ceiling slabs |
| accent | `stone_accent` | reserved for future visible use |
| compiler skip | `skip` | invisible compiler-helper faces only; no PBR companion |

Texture role bindings are hard-coded constants in `emission.rs` (not yet loaded from `theme.toml` at runtime — the beta uses a single theme). Every emitted visible face uses a theme-backed texture; `generator_brick` is not a valid role.

### Explicit Room Shell and Corridor Union

Emission keeps room role surfaces explicit while using a corridor-only 16-unit grid union for turns and intersections:

1. Every room emits one `stone_floor` slab at Z `0..16` and one `stone_ceiling` slab at `(ceiling-16)..ceiling`.
2. Each north/south/east/west room wall starts as a full-height 16-unit `stone_wall` mask. Portal rectangles and any routed 64×64 endpoint footprint intersecting that wall are removed, then the remaining cells are deterministically merged into non-overlapping wall boxes. No brush occupies an aperture.
3. Corridor centerline rectangles and full endpoint squares mark the corridor-open union. Floor and ceiling cells overlapping a room clear interior are omitted so the low corridor ceiling does not protrude into a tall room.
4. Corridor floors occupy Z `0..16`; corridor ceilings occupy Z `96..112`, leaving the frozen 80-unit clear headroom. Boundary wall cells surround only the corridor union and preserve open L/T/X centers.
5. Compiler tests validate the resulting BSP, not just source intent: room, point-entity, corridor, portal, and full 64×64 junction witnesses must all be non-solid.

Rooms have a minimum outer span of 112 units (7 quanta), which leaves an 80-unit clear interior after two 16-unit wall cells: enough for a 64-unit portal plus one quantum of total lateral margin. All emitted primitives remain rectangular six-face brushes in canonical face order.

### Entities

- `worldspawn`: all structural brushes + `"wad" "cc0_stone_beta.wad"`
- `info_player_start`: one in the first room, centered horizontally at Z = floor + 16-unit slab + 24-unit eye offset
- `light`: one per room, centered within the clear volume (midway between the floor-slab top and ceiling-slab bottom), intensity 300

Point entities are never placed in floor slabs. Compiled-corpus tests query their origins and require non-solid BSP contents.

## 9. Canonical Serialization (`serialize.rs`)

The serialization contract is frozen by `DECISION-20260724-08`:

| rule | value |
|------|-------|
| entity order | `worldspawn` first, remaining in creation-index order |
| key order | alphabetical by key string (ASCII byte order) |
| brush order | by creation index within entity |
| face order per brush | bottom, top, north, south, west, east |
| plane point format | three parenthesized integer triples `( x y z ) ( x y z ) ( x y z )` |
| integer formatting | decimal; no scientific notation; no leading zeros except `0` |
| texture name | double-quoted, follows plane points on same line |
| texture axes | `[ 1 0 0 0 ]` bracket format with single spaces |
| line endings | `\n` (LF) |
| terminal newline | exactly one trailing `\n` |

### Deterministic Byte Contract

Identical `(seed, config)` → byte-identical `.map` text. The serializer does not depend on hashmap iteration order, system time, or floating-point formatting. All keys and values are pre-sorted before emission.

## 10. Deterministic RNG (`seed.rs`)

### Derivation Formula

```
StageSeed = SHA-256("dungeon-gen/v1" || seed_le_bytes || tag)
```

Where:
- `"dungeon-gen/v1"` is the domain separator (UTF-8)
- `seed_le_bytes` is the 8-byte little-endian master seed
- `tag` is the UTF-8 stage identifier

### Stage Identifiers

| tag | stage | consumer |
|-----|-------|----------|
| `room-placement` | room candidate pos/dims | `place_rooms()` |
| `corridor-routing` | topology edges + A* path choices | `build_topology()`, `route_all_edges()` |
| `entity-placement` | spawn + light positions | (reserved) |
| `light-placement` | light entity parameters | (reserved) |

Note: `corridor-routing` is shared between `build_topology()` and `route_all_edges()` — both consume from the same RNG stream so the pipeline is deterministic regardless of call order.

### StageRng

The `StageRng` is a SHA-256-chained counter stream:

1. Initial state = `StageSeed.digest` (32 bytes)
2. Initial buffer = four `u64` values from the digest (little-endian)
3. When buffer is exhausted (4 values consumed), chain forward: `state = SHA-256(state)`, refill buffer

This produces an infinite deterministic stream. The RNG is cloneable (snapshots current state) for sub-computation isolation.

## 11. Theme Structure

```
src/bsp_generator/themes/cc0_stone_beta/
├── theme.toml          ← texture role bindings
├── palette.lmp         ← 768-byte Quake palette (deterministic procedural)
├── cc0_stone_beta.wad  ← WAD2 archive (deterministic procedural)
├── build.py            ← deterministic asset generator
├── LICENSE             ← CC0 public domain dedication
└── textures/
    ├── stone_floor_basecolor.png
    ├── stone_floor_norm.png
    ├── stone_floor_gloss.png
    ├── stone_wall_basecolor.png
    ├── stone_wall_norm.png
    ├── stone_wall_gloss.png
    ├── stone_ceiling_basecolor.png
    ├── stone_ceiling_norm.png
    ├── stone_ceiling_gloss.png
    ├── stone_accent_basecolor.png
    ├── stone_accent_norm.png
    └── stone_accent_gloss.png
```

`build.py` is deterministic — it produces byte-identical WAD, palette, and PNGs on every invocation. The WAD contains five 64×64 miptex entries: `STONE_FLR`, `STONE_WALL`, `STONE_CEIL`, `STONE_ACNT`, and compiler-only `SKIP`. The four visual roles have normal/gloss companions; `SKIP` deliberately does not. The generator's texture constants map to the visual WAD names.

## 12. Validation and Testing

### Test Categories

| test file | what it covers |
|-----------|---------------|
| `tests/config_validation.rs` | `DungeonConfig::validate()` for every error path |
| `tests/determinism.rs` | byte-identical output for identical inputs |
| `tests/placement.rs` | room placement with various seeds and configs |
| `tests/topology.rs` | MST + loop construction, reachability, edge validity |
| `tests/geometry_validation.rs` | overlap detection, quantum snapping, bounds |
| `tests/routing.rs` | corridor routing, A* correctness |
| `tests/junction.rs` | junction geometry, brush face counts |
| `tests/emission.rs` | brush counts, entity placement, face estimates |
| `tests/serialize.rs` | canonical format, key ordering, face ordering |
| `tests/canonical.rs` | full pipeline determinism |
| `tests/invariants.rs` | output guarantees (sealed, walkable, no overlap) |
| `tests/collision.rs` | corridor-to-room clearance |
| `tests/theme_evidence.rs` | theme asset determinism, WAD structure |
| `tests/corpus_execution.rs` | all 12 support corpus configs compile warning-free, reload strictly, remain sealed, and return non-solid contents at room/entity/corridor/portal/junction witnesses |

### Key Invariants Tested

- Determinism: `generate(seed, config)` called twice produces identical strings
- Config validation: every out-of-range field rejected with `InvalidConfig`
- Placement: rooms never overlap, always within XY bounds
- Topology: all rooms reachable, exact edge count = `room_count - 1 + loop_count`
- Routing: all corridors have width ≥ 64, height ≥ 80
- Emission: all brushes have exactly 6 faces, face order canonical, portals/junction centers remain outside every solid brush, point entities sit above the floor slab
- Serialization: worldspawn first, alphabetical keys, no trailing whitespace
- Compiler pipeline: any `qbsp`, `vis`, or `light` warning (including skipped fill) is a hard failure
- Corpus: all 12 nominal + boundary configs generate successfully, compile successfully, and M2 exceeds M1 on at least one metric

## 13. Frozen Contract Values

All values below are frozen in `bsp-dungeon-generation.md` and must not change without explicit re-review:

| value | where used | frozen value |
|-------|-----------|-------------|
| construction quantum | `CONSTRUCTION_QUANTUM` | 16 |
| wall thickness | `WALL_THICKNESS` (placement) | 16 |
| corridor width | `CORRIDOR_WIDTH` (routing) | 64 |
| corridor height | `CORRIDOR_HEIGHT` (routing) | 80 |
| slab thickness | `SLAB` (emission) | 16 |
| M1 room range | config validation | 8–16 |
| M2 room range | config validation | 17–40 |
| M1 loop range | config validation | 0–2 |
| M2 loop range | config validation | 1–6 |
| M1 XY max | config validation | 1536 |
| M2 XY max | config validation | 3072 |
| M1 Z max | config validation | 256 |
| M2 Z max | config validation | 384 |
| domain separator | seed derivation | `"dungeon-gen/v1"` |
| stage tags | seed derivation | `room-placement`, `corridor-routing`, `entity-placement`, `light-placement` |

## 14. See Also

- [BSP Generator Usage Guide](../guide/19-bsp-generator.md) — how to generate and compile
- [BSP Generator API Reference](../api/19-bsp-generator.md) — public type and function docs
- [Dungeon Generation Specification](../../.internal-dev/specifications/bsp-dungeon-generation.md) — frozen contract
- [Evidence Matrix](../../.internal-dev/plans/bsp-dungeon-contract-evidence/evidence-matrix.md) — evidence campaign
- [BSP Runtime and Lifetime](18-bsp-runtime-and-lifetime.md) — downstream BSP pipeline
