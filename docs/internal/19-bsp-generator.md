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

`build.py` is deterministic — it produces byte-identical WAD, palette, and PNGs on every invocation; Pillow is its only build-time Python dependency. The WAD contains four 1024×1024 visual miptex entries: `stone_floor`, `stone_wall`, `stone_ceiling`, and `stone_accent`, plus the compact compiler-only 64×64 `skip` entry. The four visual roles have matching 1024×1024 base-color, normal, and gloss companions with procedurally authored stone relief; `skip` deliberately has no companions. The generator's texture constants map to the lowercase WAD identities.

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

## 14. Enhanced v2 Pipeline

The Enhanced v2 profile (`src/bsp_generator/src/enhanced/`) is a structurally
disjoint pipeline from Legacy v1. It produces M2-only, two-layer dungeons with
stairs, theme palette assignment, and corridor/ceiling/pillar variance.

### 14.1 Architecture

```text
EnhancedConfig  (validates at construction, no separate validate())
    │
    │  EnhancedSeed::new(seed)
    ▼
EnhancedSeed ──► stage_seed("layer-placement")   ──► EnhancedStageRng
    │                                               place_rooms()
    ├── stage_seed("vertical-topology") ──► EnhancedStageRng
    │                                          build_topology()
    ├── stage_seed("theme-assignment")  ──► EnhancedStageRng
    │                                          assign_uniform() / assign_by_zone()
    ├── stage_seed("feature-placement") ──► EnhancedStageRng
    │   stage_seed("corridor-variance") ──► EnhancedStageRng
    │                                          apply_features()
    ▼
emit_map() ──► String  (.map bytes)
```

### 14.2 Module Map

| module | responsibility | key types |
|--------|---------------|-----------|
| `profile.rs` | profile dispatch and fixed stair contracts | `GenerationProfile`, `GenerationRequest`, `StairType` |
| `config.rs` | configuration and vertical contract | `EnhancedConfig`, frozen constants |
| `seed.rs` | deterministic RNG (domain `"dungeon-gen/v2"`) | `EnhancedSeed`, `EnhancedStageSeed`, `EnhancedStageRng` |
| `intent.rs` | typed IDs and intent records | `LayerId`, `RoomId`, `SocketId`, `RouteId`, `TransitionId`, `ReservationId`, `ZoneId`, `PaletteId`, `IdAllocator`, `RouteIntent`, `TransitionIntent` |
| `error.rs` | typed errors | `EnhancedError` (12 variants) |
| `occupancy.rs` | projected XY occupancy grid | `OccupancyGrid`, `Owner`, `GridCheckpoint` |
| `placement.rs` | two-layer room placement | `PlacedRoom`, `CandidateSocket`, `PlacementResult`, `PlacementJournal` |
| `topology.rs` | MST + loop topology + stair reservations | `TopologyResult`, `build_topology()` |
| `routing.rs` | A* corridor routing | `RouteSegment`, `RouteResult`, `route_sockets()` |
| `reservation.rs` | transactional ownership system | `Transaction`, `TransactionMark`, `OwnerKind` |
| `transition.rs` | grand/wall-edge stair geometry and route-connected reservations | `reserve_transitions()`, `connect_lower_approaches()` |
| `theme.rs` | theme package and palette assignment | `ThemePackage`, `PaletteDefinition`, `RoomRole`, `AssignmentStrategy`, `ThemeAssignment` |
| `features.rs` | corridor/ceiling/pillar variance | `CorridorWidthSelection`, `FeatureResult`, `apply_features()` |
| `emission.rs` | map text emission | `emit_map()` |
| `metadata.rs` | metadata re-export | (re-exports `EnhancedMetadata`) |
| `pipeline.rs` | top-level entry point | `generate_enhanced()`, `EnhancedMetadata` |

### 14.3 RNG Domains

Enhanced v2 uses domain separator `"dungeon-gen/v2"` — **completely independent**
from Legacy v1's `"dungeon-gen/v1"`. Same seed values produce different output.

| tag | stage | stream consumed by |
|-----|-------|-------------------|
| `layer-placement` | room placement on two layers | `place_rooms()` |
| `vertical-topology` | topology edge and transition selection | `build_topology()` |
| `vertical-routing` | reserved | (future use) |
| `theme-assignment` | palette assignment | `assign_uniform()` / `assign_by_zone()` |
| `feature-placement` | pillar positions, ceiling selection | `apply_features()` |
| `corridor-variance` | per-route corridor width selection | `apply_features()` |

### 14.4 Geometry Contracts

**Room placement:**
- Rooms placed across two layers with balanced membership (max diff = 1)
- All rooms on both layers projected onto a shared XY occupancy grid — no XY overlap
- Room spans: 112–256 Quake units per axis (7–16 quanta)
- Socket portals: 64 units wide, 32-unit corner margins, derived from committed rooms only
- Transactional journal: full checkpoint/rollback on failed placement attempts

**Topology:**
- Per-layer MST via Kruskal's algorithm (candidate pairs sorted by center distance)
- Loop edges added from non-MST pairs, canonical backtracking on failed routes
- Stair transitions: one stair per `vertical_edges`, connecting a lower-room socket to an upper-room socket
- Type A reserves a 192-unit run across the lower host's full wall-free width; Type B reserves a 192×64 wall-edge strip
- Both types own 12 exact tread columns with 16-unit tread/riser increments, 80-unit headroom, and lower/upper approaches
- The lower approach joins a committed lower route; the vertical ceiling exit and upper approach reach a split upper-room wall aperture
- Topology backtracks deterministically across transition candidates when an early reservation prevents complete horizontal routing
- Global connectivity validated post-commit (every room reachable from every other)

**Corridor routing:**
- A* on quantum grid with 64-unit corridor envelope
- Route envelopes checked against occupancy grid (allow room endpoints, reject intervening rooms)
- Socket approach reservations: one exterior cell + 64-unit tangent strip

**Theme:**
- CC0 Dungeon v2 theme: checked-in typed data in `theme.rs`
- Palettes: base palette (all roles) + optional zone palettes + connector palette (no accent)
- Room role derivation: Entry (lowest RoomId), Hub (max graph degree), DeadEnd (degree=1), Side (remaining)
- Strategies: `Uniform` (all rooms get base palette) or `ByZone` (zones get distinct palettes)

**Feature variance:**
- Corridor widths: 64, 80, or 96 Quake units per route (RNG-selected, must fit reserved envelopes)
- Ceiling heights: 128, 144, or 176 Quake units per room (RNG-selected, min 80 headroom preserved)
- Pillars: freestanding 32×32×80 axis-aligned boxes, accent-textured, connectivity oracle validates accessibility
- Spawn: center of the canonical 64×64 lower stair landing at floor-top + 24, leaving the 16-unit Quake hull radius plus a 16-unit safety margin from both landing sides and every tread; cardinal yaw faces the stair opening

**Emission:**
- Room shells: floor slab + ceiling slab + 4 wall masks with aperture cutouts
- Corridor union: open cells for 64×64 turns and intersections
- Stairwells: host floor cells are replaced by exact tread columns; the inter-layer slab aperture omits the complete 192-unit run so no ceiling edge or shaft return crosses the flight
- Transition shells union lower-route approaches and upper connectors, split both room-wall apertures, retain 80-unit standing headroom over the supported upper landing, preserve a full 64-unit crest throat, and seal the 176→192 ceiling bridge without capping it
- Canonical `.map` serialization: worldspawn first, alphabetical keys (including spawn `angle`), canonical face order
- Enhanced-only `_minlight 16` in worldspawn ensures fully occluded connector and stair
  faces receive baked lightmap data from the pinned ericw `light` stage

**Publication:**
- `engine_pack enhanced-dungeon` publishes the compiled BSP, .lit, palette, WAD,
  metadata, and exactly the normal/gloss companions for eligible referenced identities
  under a `textures/` subdirectory; it never publishes source basecolor PNGs
- Albedo remains WAD-backed; companion discovery uses the frozen `<miptex>_norm.png` /
  `<miptex>_gloss.png` naming convention
- Enhanced publication verifies that every eligible referenced identity staged both
  canonical companions. Companion PNGs must be regular, complete PNGs with valid CRCs
  and IHDRs whose dimensions exact-match the WAD miptex mip-0 resolution (1024×1024)
- This required-closure rule is Enhanced-only: generic `compile-bsp` retains optional
  companion semantics and its deterministic legacy fallback
- A strict isolated authorization validates the complete staged closure before atomic
  publication; malformed, dimension-mismatched, or missing required assets fail before
  the package becomes visible

### 14.5 Key Differences from Legacy v1

| aspect | Legacy v1 | Enhanced v2 |
|--------|-----------|-------------|
| domain | `"dungeon-gen/v1"` | `"dungeon-gen/v2"` |
| map classes | M1 + M2 | M2 only |
| layers | 1 (flat, all rooms at Z=0) | 2 (lower Z=0, upper Z=192) |
| vertical connections | none | 12-tread room-scale grand or wall-edge narrow stairs (sealed shells) |
| config validation | `DungeonConfig::validate()` | validates at construction |
| RNG isolation | 4 stage tags | 6 stage tags (enhanced-only) |
| theme | CC0 Stone Beta | CC0 Dungeon v2 (separate WAD) |
| room roles | none | Entry / Hub / DeadEnd / Side |
| placement | same-layer, flat | two-layer, balanced membership, projected XY |
| topology | Kruskal MST + loops (flat) | per-layer MST + loops + stair reservations |
| corridor width | fixed 64 | 64/80/96 per route |
| ceiling height | fixed (config.z_span) | 128/144/176 per room |
| pillars | none | up to 8 per room, connectivity-verified |

## 15. See Also

- [BSP Generator Usage Guide](../guide/19-bsp-generator.md) — how to generate and compile
- [BSP Generator API Reference](../api/19-bsp-generator.md) — public type and function docs
- [Dungeon Generation Specification](../../.internal-dev/specifications/bsp-dungeon-generation.md) — frozen contract
- [Evidence Matrix](../../.internal-dev/plans/bsp-dungeon-contract-evidence/evidence-matrix.md) — evidence campaign
- [BSP Runtime and Lifetime](18-bsp-runtime-and-lifetime.md) — downstream BSP pipeline

## 16. Enhanced v3 Proof (Private — 2026-07-31)

### 16.1 Status

A private, test-only proof package demonstrated the architectural feasibility of cardinal/45°
chamfered-octagonal geometry, pointed-arch portal apertures, grounded assemblies, and dense
M2-budget compositions. The proof is **not a production profile** — no `GenerationProfile`
variant, public export, or production code was changed.

### 16.2 Evidence

| claim | fixture | key result |
|-------|---------|------------|
| 45° diagonal walls | `convex-45-shell.map` | Warning-free compile; thickness 32/√2 ≈ 22.63; spatial witnesses pass |
| Pointed-arch portals | `pointed-portal.map` | Full-depth shell omission; 100% throat witnesses non-solid |
| Grounded assemblies | `grounded-assembly.map` | Acyclic support graph; coplanar contact; atomic dependent removal |
| Segmented-arch portals | `segmented-portal.map` | Focused-only; throat witnesses pass |
| M2 budget compliance | `dense-rich.map` | 2,404 faces, 6 entities, 4 batches — well within ceilings |
| v1/v2 compatibility | 24-entry corpus | All byte-identical to baseline; theme assets unchanged |
| Live GPU startup | `bsp_beta` | Swapchain acquired; 21,574 frames; 0 panics/errors |

### 16.3 Known Gaps

- **ericw-tools small-map segfault**: `qbsp` crashes on maps < ~5 brushes. Focused fixtures
  use 73–75 brushes to avoid this. The integrated thin slice (2 brushes) provides source-level
  proof only.
- **Semantic pipeline density**: The private proof pipeline cannot yet emit compiler-safe
  structural density at generation time; the dense Rich fixture is hand-authored.
- **Proof-era deferred capabilities**: Diagonal portals, concave rooms, accessible upper
  features, segmented-arch integration, additional room families, and other capabilities
  were excluded from this historical proof boundary per G-12. The later explorer
  authorization integrates cardinal segmented surrounds with an interface seal; it does
  not rewrite the focused fixture's historical evidence class.

### 16.4 Reference

- Decision package: `.internal-dev/plans/enhanced-v3-proof/decision-package.md`
- Evidence matrix: `.internal-dev/plans/enhanced-v3-proof/evidence-matrix.md`
- Specification: `.internal-dev/specifications/bsp-dungeon-generation.md` §19
- Knowledge: `.internal-dev/knowledge/bsp-enhanced-v3-proof.md`

## 17. Enhanced v3 Production Pipeline

### 17.1 Architecture

The Enhanced v3 profile (`src/bsp_generator/src/enhanced_v3/`) is a structurally
disjoint pipeline from Legacy v1 and Enhanced v2. It produces M2-only, two-layer
dungeons with cardinal + 45° chamfered-octagonal rooms, selectable cardinal
portal surrounds, grounded assemblies, and Sparse/Moderate/Rich density presets.
Pointed is the byte-compatible default; rectangular and sealed segmented
surrounds are explorer overrides. The profile reuses the CC0 Dungeon v2 theme
without modification.

```text
V3Config  (`new()` validates defaults; mutated explorer fields use `validate()`)
    │
    │  V3Seed::new(seed)  →  stage_seed("v3-placement")
    ▼
build_footprints()  ──► FootprintLayout
    │
    │  stage_seed("v3-topology")
    ▼
build_topology()  ──► CommittedTopology
    │
    │  stage_seed("v3-features")
    ▼
plan_composition()  ──► CompositionPlan
    │
    │  stage_seed("v3-detail")
    ▼
build_assembly()  ──► Assembly
    │
    ▼
emit_map_text()  ──► String  (.map bytes)
```

### 17.2 Module Map

| module | responsibility | key types |
|--------|---------------|-----------|
| `config.rs` | configuration, presets, normal classification, frozen constants | `V3Config`, `V3Preset`, `NormalClass`, `CONSTRUCTION_QUANTUM` |
| `error.rs` | typed errors (45 variants, 9 categories) | `V3Error` |
| `rng.rs` | deterministic RNG (domain `"dungeon-gen/v3"`) | `V3Seed`, `V3StageSeed`, `CandidateSelector` |
| `geometry.rs` | i128 Rational exact geometry | `Point3`, `Rational`, `Vector3`, `CanonicalPlane`, `ConvexBrush`, `FaceRole` |
| `ids.rs` | typed newtype IDs and committed topology | `RoomId`, `PortalId`, `SurfaceId`, `FeatureId`, `CommittedTopology`, `V3IdAllocator` |
| `footprint.rs` | cardinal + 45° footprint generation | `Footprint`, `FootprintLayout`, `build_footprints()` |
| `reservation.rs` | transactional volume reservation | `Reservation`, `ReservationSet`, `build_reservations()` |
| `topology.rs` | per-layer MST + transition selection | `CommittedRoute`, `CommittedTransition`, `build_topology()`, `compute_reservations()` |
| `intent.rs` | composition planning with grammar descriptors | `CompositionPlan`, `GrammarDescriptor`, `plan_composition()` |
| `composition.rs` | composition planning types | `CompositionPlan`, `GrammarDescriptor` |
| `assembly.rs` | grounded support-graph assembly | `Assembly`, `AssemblyBrush`, `BrushRole`, `Interface`, `ProtectedVolume`, `Support` |
| `emission.rs` | canonical .map text emission | `emit_map_text()`, `texture_for_role()` |
| `metadata.rs` | production metadata (22 read-only accessors) | `EnhancedV3Metadata` |
| `pipeline.rs` | top-level entry point | `run_pipeline()`, `V3PipelineOutput` |

### 17.3 RNG Domain

Enhanced v3 uses domain separator `"dungeon-gen/v3"` — fully independent from
v1's `"dungeon-gen/v1"` and v2's `"dungeon-gen/v2"`. Same master seed values
produce cryptographically independent output streams across all three domains.

| tag | stage | stream consumed by |
|-----|-------|-------------------|
| `v3-placement` | two-layer room placement with 45° footprint support | `build_footprints()` |
| `v3-topology` | topology and transition selection | `build_topology()` |
| `v3-features` | chamfered footprints, pointed arches, grounded assemblies | `plan_composition()` |
| `v3-detail` | preset-driven feature density, pillar placement | `build_assembly()` |

### 17.4 Geometry Contracts

**Coordinate system:**
- All authored coordinates use i128 integer arithmetic through the `Rational` type
- No floating-point in the geometry path
- Construction quantum: 16 Quake units
- All coordinates are quantum-aligned (multiples of 16)

**Approved normals:**
- Cardinal: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
- Diagonal 45° in XY: (±1, ±1, 0) in lowest integer terms
- All other normals produce `V3Error::UnapprovedNormal`

**ConvexBrush:**
- Defined as a system of half-space inequalities, not as explicit face lists
- Each half-space is a `CanonicalPlane` with an approved normal
- The brush is the intersection of all half-spaces
- Validated for: non-empty, bounded, positive volume, minimum face area,
  minimum edge length, and minimum directional thickness

**Footprint generation:**
- `build_footprints()` generates room footprints from the seed and preset
- Cardinal footprints use standard axis-aligned rectangles
- 45° footprints produce chamfered/octagonal shapes by adding diagonal
  half-spaces to the basic cardinal volume
- Both layers share a projected XY occupancy grid — no XY overlap
- Footprints are immutable after construction

**Assembly contract:**
- Support graph is acyclic (`SupportGraphCycle` error on cycles)
- Every brush has a transitive support path to a floor surface
  (`UnsupportedBrush` error on orphan brushes)
- Contact is coplanar shared-face (zero-volume) between supporting and
  supported brushes
- Positive-volume overlap between distinct brushes is prohibited
  (`PositiveVolumeOverlap` error)
- Protected volumes (portals, stair wells) cannot be intruded upon
  (`ProtectedVolumeIntrusion` error)

**Cardinal portal surrounds:**
- Full-depth shell omission through the wall — no separate opening brush
- 64×80 swept clearance at the throat for `None`, `Pointed`, and `Segmented`
- On cardinal walls only — diagonal portals are deferred
- Aperture ownership: the shell wall's omission IS the portal
- `Pointed` remains the byte-compatible default; `None` emits a flat rectangular opening
- `Segmented` retains two stepped crown bands. The corridor roof seals the first band;
  a one-quantum cap immediately outside the remaining 32-unit centre recess seals the
  Z=112–128 interface without reducing the navigable throat

### 17.5 Composition Planning

`plan_composition()` selects features from 6 grammar descriptor families:

| family | description |
|--------|-------------|
| `PortalChamber` | room with cardinally-aligned pointed-arch portals |
| `ButtressedHall` | elongated room with buttress-like wall features |
| `ColumnGrove` | room interior with freestanding pillar clusters |
| `FracturedVault` | ceiling features with non-planar upper surfaces |
| `TerracedShrine` | stepped floor elevation changes within a room |
| `MonolithicChamber` | large open room with minimal interior subdivision |

These are real integrated feature generators, not planning descriptors.
Each family materializes grounded, family-distinct brushes. Preset
`minimum_families` enforces that at least N families are materialized in the output.

### 17.6 Preset Details

| parameter | Sparse | Moderate | Rich |
|-----------|--------|----------|------|
| `min_rooms` (exact emitted count) | 12 | 20 | 28 |
| same-layer routes (exact emitted count) | 10 | 20 | 30 |
| `target_loops` | 0 | 2 | 4 |
| `minimum_families` | 1 | 3 | 6 |
| `minimum_assemblies` | 1 | 3 | 6 |
| `minimum_feature_brushes` | 2 | 6 | 12 |
| `face_budget` | 3,000 | 5,000 | 8,000 |

The measured default-extent seed matrix (0, 42, 99, 255) emits 1,856–1,883
Sparse, 3,275–3,310 Moderate, and 4,725–4,782 Rich source faces. These values
are below their preset ceilings and the M2 10,000-face ceiling.

### 17.7 Key Differences from Enhanced v2

| aspect | Enhanced v2 | Enhanced v3 |
|--------|-------------|-------------|
| domain | `"dungeon-gen/v2"` | `"dungeon-gen/v3"` |
| geometry arithmetic | i32 / f32 mix | i128 Rational only |
| approved normals | axis-aligned only | cardinal + 45° diagonal |
| room shape | rectangular only | chamfered/octagonal + rectangular |
| portal shape | rectangular (gap in wall) | cardinal rectangular/pointed/segmented full-depth omission |
| assemblies | none (brush-per-primitive) | grounded support graph |
| grammar families | 2 strategies (Uniform/ByZone) | 6 descriptor families |
| density presets | none (single EnhancedConfig) | Sparse/Moderate/Rich |
| minimum-identity | none | typed `MinimumIdentityFailure` |
| config type | `EnhancedConfig` (10 fields) | `V3Config` (19 fields, including explorer overrides; `layers` is validated as exactly 2) |
| stage tags | 6 tags | 4 tags |
| error type | `EnhancedError` (12 variants) | `V3Error` (45 variants) |
| theme | cc0_dungeon_v2 | cc0_dungeon_v2 (reused) |
| entry point | `generate_enhanced(seed, config)` | `generate_v3(&config)` |

### 17.8 Testing

| test file | coverage |
|-----------|----------|
| `enhanced_v3_contract_baseline.rs` | 25 frozen contract identity tests: RNG framing, stage tags, preset tags, seed vectors, geometry policy, layers, corpus matrix, serialization, rejection records, v1/v2 compatibility |
| `enhanced_v3_public_api.rs` | public API surface coverage |
| `enhanced_v3_generation.rs` | can-generate tests for all presets |
| `enhanced_v3_semantic_core.rs` | geometry, footprint, topology unit tests |
| `enhanced_v3_geometry.rs` | ConvexBrush, CanonicalPlane, half-space tests |
| `enhanced_v3_budget.rs` | M2 budget evidence (faces < 10,000, entities < 300) |
| `enhanced_v3_compiled_space.rs` | compiled spatial witnesses |
| `enhanced_v3_compatibility.rs` | v1/v2 baseline byte-identical preservation |
| `enhanced_v3_integrated.rs` | integrated pipeline tests |
| `enhanced_v3_qualification.rs` | full qualification suite |
| `enhanced_v3_production_acceptance.rs` | exact preset topology, 12-entry source matrix, real plan identity, and source evidence |
| `enhanced_v3_compiled_space.rs` | pinned 12-entry compiler matrix, strict reload, budgets, and compiled witnesses |
| `enhanced_v3_explorer.rs` | all explorer overrides, arch variants, exact topology meanings, and validation |
| `tools/dungeon_gen/tests/enhanced_v3_cli.rs` | all m3 options, exact summaries, deterministic replay, seed distinction, and v1/v2 flag isolation |
| `tools/engine_pack/tests/enhanced_dungeon_v3_candidate.rs` | full-config segmented generation, warning-free ericw compilation, strict package publication, and override records |

Engine-owned headless capture evidence is indexed at `.internal-dev/captures/enhanced-v3-production/manifest.md`; it remains distinct from live WSI validation.
