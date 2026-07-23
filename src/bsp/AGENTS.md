# BSP Crate Agent Guide

## Scope

The `bsp` crate is a pure byte-level trust boundary for Quake 1 BSP29, BSP2, WAD2, BSPX, `.lit`, and entity-string parsing. It produces owned, immutable `BspWorld` records. It has zero renderer, Vulkan, physics, app, windowing, async, or filesystem-watcher dependencies.

## Crate Contract

- **Neutral boundary**: No GPU handles, no scene nodes, no physics objects, no engine runtime.
- **Fail-closed**: Every parsing error produces a `BspReport` with a stable `DiagnosticCode` and severity. Invalid data is rejected before allocation when possible.
- **Immutable output**: `BspWorld` is read-only after construction. No mutable accessors.
- **Deterministic**: Source index ordering, normalized-name ordering, stable duplicate ordinals.
- **Checked arithmetic**: All allocation sizes are validated through `limits.rs` helpers before allocation.
- **No unsafe**: Byte decoding uses checked indexing; no transmute, pointer casts, or unchecked access.
- **Default builds must not link bsp**: The root `engine` crate does not depend on `bsp`.

## Module Map

| module | role |
|--------|------|
| `lib.rs` | `BspLoader::load()` entry point, `BspWorld`, public re-exports |
| `diagnostic.rs` | `DiagnosticCode` enum, `BspReport`, `Severity` |
| `limits.rs` | Hard/aggregate budget constants and checked helpers |
| `profile.rs` | BSP29/BSP2 magic/version detection, profile identity |
| `decode.rs` | LE integer/float decoders from byte slices |
| `lumps.rs` | 15 standard lump parsers + cross-lump validation |
| `bspx.rs` | BSPX directory discovery and extension lump decoding |
| `companions.rs` | `.lit`, palette, WAD companion binding |
| `wad.rs` | WAD2 header/directory/lump parsing |
| `resources.rs` | Texture/resource resolution order |
| `entities.rs` | Entity grammar tokenizer/parser, classification |
| `world.rs` | `BspWorld` construction and validation |

## Dependencies

- `glam` (no default features, `libm` only): `Vec3` for parsed plane normals and vertices.
- No renderer, Vulkan, physics, app, windowing, async, filesystem watcher deps.
- No `thiserror` — error types are hand-implemented for zero-dep purity.

## Testing

```bash
cargo check -p bsp
cargo test -p bsp
```

Tests live in `tests/`:
- `parse_golden.rs` — valid BSP29/BSP2 fixtures, lumps, extensions, companions, entities, resources
- `parse_adversarial.rs` — table-driven truncation/overflow/cycle/bad-index mutations
- `entities_and_resources.rs` — entity grammar edge cases, resource resolution, companion binding
