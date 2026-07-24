# bsp_generator Crate Agent Guide

> Development guide for agents working in `src/bsp_generator/`.

## Scope

Pure-Rust offline dungeon generator producing Quake 1 `.map` text from `(u64 seed, DungeonConfig)` pairs. This crate depends **only** on `bsp` (format types) and `sha2` — it must never gain a dependency on `renderer`, `vulkan`, `winit`, `physics`, `audio`, `scripting`, `bsp_runtime`, or `engine_pack`.

## Frozen Contract

All construction parameters, bounds, and serialization rules are frozen in `.internal-dev/specifications/bsp-dungeon-generation.md`. **Do not change** any of the following without explicit re-review:

- Construction quantum (16), wall thickness (16), corridor width (64), corridor height (80)
- M1/M2 room count, loop count, XY max, Z max ranges
- Stage tags (`"room-placement"`, `"corridor-routing"`, `"entity-placement"`, `"light-placement"`)
- Domain separator `"dungeon-gen/v1"`
- Canonical serialization rules (entity order, key order, face order, axis format, line endings)
- Output guarantees (sealed, walkable, no overlap, BSP2 only, open arches only, single-layer)

## Source Layout

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
  junction.rs     — make_brush(), build_l_junction, build_t_junction,
                      build_x_junction, build_room_portal
  emission.rs     — build_emission() — brush construction, entity creation
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

## Key Design Rules

1. **No renderer/Vulkan/windowing deps**: This crate is an offline pipeline. If you need BSP loading or rendering, use `bsp_runtime` + `renderer` downstream.
2. **Pure functions**: Every stage is a function from immutable input to `Result<IR, Error>`. No global state, no lazy initialization, no hidden mutation.
3. **Typed IRs**: Each pipeline stage produces a named struct (e.g., `LayoutIntent`, `RoutedIntent`, `EmissionIntent`). These are pure data records — no generation logic lives in them.
4. **Determinism by construction**: The only source of randomness is `StageRng`. No `HashMap`/`HashSet` iterators affect output order. All serialization uses pre-sorted keys and faces.
5. **Exhaustion, not panic**: Every bounded loop has a hard limit and returns a typed error on exhaustion. Never use `unwrap()`, `expect()`, or `panic!()` in production code paths.
6. **Frozen values**: Constants from the specification are declared as `const` items. Changing a `const` is a contract violation.
7. **No inline texture generation**: The CC0 Stone Beta theme assets are pre-built by `build.py`. The generator references WAD texture names — it does not produce textures.

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
- Successful compilation (no qbsp/vis/light errors)
- BSP2 magic in output headers
- Sealed maps (no pointfile leaks)
- Deterministic byte-identical output across duplicate runs
- M2 exceeds M1 on at least one metric (faces, entities, or batches)

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
- **Canonical face order**: bottom, top, north, south, west, east. This order is frozen. Changing it breaks byte-compatibility.
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
