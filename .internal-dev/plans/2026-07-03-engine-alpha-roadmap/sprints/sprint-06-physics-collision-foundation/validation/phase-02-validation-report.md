# Phase 02 Validation Report: Package/Scene Collision Metadata

## Result

Passed.

## Accepted Schema

Package assets may include optional TOML metadata:

```toml
[assets.metadata.collision]
body_id = "body.wall"
collider_id = "collider.wall"
body_kind = "static"
trigger = false
shape = { kind = "box", half_extents = [0.5, 1.25, 0.5] }
```

Scene nodes may include optional JSON collision components:

```json
"collision": {
  "body": { "id": "body.wall", "kind": "static" },
  "colliders": [
    {
      "id": "collider.wall",
      "shape": { "kind": "box", "half_extents": [0.5, 1.25, 0.5] },
      "trigger": false,
      "asset": "core.collision.wall",
      "offset": [0.0, 0.0, 0.0]
    }
  ]
}
```

Supported body kinds are `static`, `dynamic`, and `kinematic`. Supported shape
kinds are `box`, `cuboid`, `sphere`, `capsule`, and `capsule_y`.

## Implementation Summary

- Added package collision metadata validation through renderer package validation.
- Added scene node collision metadata serialization/parsing and validation.
- Added durable collision ID, duplicate ID, body kind, shape kind, dimension, trigger, offset, runtime-handle, and unknown collision asset diagnostics.
- Added `engine_pack` CLI coverage proving collision metadata failures surface through existing renderer validation.
- Kept old package and scene files backward-compatible when collision metadata is absent.

## Diagnostic Codes

| Area | Codes |
|---|---|
| Package | `asset.collision_invalid_schema`, `asset.collision_invalid_id`, `asset.duplicate_collision_id`, `asset.collision_invalid_body_kind`, `asset.collision_invalid_trigger`, `asset.collision_missing_shape`, `asset.collision_invalid_shape`, `asset.collision_invalid_dimension` |
| Scene | `scene.collision_invalid_id`, `scene.duplicate_collision_id`, `scene.collision_invalid_body_kind`, `scene.collision_missing_collider`, `scene.collision_invalid_shape`, `scene.collision_invalid_dimension`, `scene.collision_invalid_offset`, `scene.unknown_collision_asset_id` |

## Files Changed

| File | Lines | Notes |
|---|---:|---|
| `src/renderer/src/data/asset_registry.rs` | 1948 | Package collision metadata validation and tests |
| `src/renderer/src/api/scene.rs` | 3164 | Scene collision metadata schema, validation, tests |
| `tools/engine_pack/tests/cli_validation.rs` | 485 | CLI validation test for collision metadata failure |

## Commands

| Command | Result | Notes |
|---|---|---|
| `cargo fmt --check` | Pass | Formatter clean |
| `cargo test -p renderer` | Pass | 156 lib tests, 17 integration tests, 5 ignored doc tests |
| `cargo test -p engine_pack` | Pass | 14 CLI tests |
| `cargo check -p engine_pack` | Pass | Existing renderer warning noise remains |
| `cargo check` | Pass | Existing renderer warning noise remains |

## Acceptance Criteria

| Criteria | Status | Evidence |
|---|---|---|
| Package collision metadata validates and rejects invalid cases | Pass | Package tests accept valid metadata and reject invalid dimensions, duplicate IDs, runtime-shaped IDs |
| Scene collision metadata validates and round-trips | Pass | Scene test parses, pretty-serializes, and reparses collision metadata |
| Scene invalid cases rejected | Pass | Scene tests cover invalid dimensions, duplicate IDs, unknown collision asset IDs, runtime handle-shaped collision IDs |
| Runtime handle-shaped collision identities rejected | Pass | Package and scene tests cover collision-specific handle-shaped data |
| `engine_pack` surfaces collision metadata failures | Pass | CLI test expects `asset.collision_invalid_dimension` |
| Backward-compatible files without collision metadata still validate | Pass | Existing package/scene/CLI fixtures continue to pass |

## Capture

No visible renderer/editor behavior changed in Phase 02. This phase changed file
schema validation only, so headless draw capture was not required. Future visual
validation must use true engine-owned capture with `--headless --capture_target draw`.

## Residual Notes

- Full renderer/workspace checks still emit existing renderer dead-code warning noise.
- Collision metadata is validated and persisted in scene JSON, but no editor UI authoring or runtime physics simulation wiring was added in this phase.
