# Target Design

## Boundary Model

- `physics` owns simulation, runtime Rapier integration, durable wrapper IDs, descriptor structs, query result structs, and event extraction primitives.
- `engine_events` owns cross-system event vocabulary. Add to it only if Sprint 05 event shapes are insufficient.
- Renderer package/scene contracts own authored collision metadata and validation, not simulation.
- `engine_pack` exposes validation by calling renderer validators.
- Dogfood remains a game/app consumer and should not define engine-level contracts.

## Physics API Shape

Planned concepts:

- `PhysicsWorld`
- `PhysicsBodyId(String)` or equivalent durable wrapper owned by `physics`
- `PhysicsColliderId(String)` or equivalent durable wrapper owned by `physics`
- `BodyKind::{Static, Dynamic, Kinematic}` where support is tested or explicitly narrowed
- `ColliderShape::{Cuboid { half_extents }, Sphere { radius }, Capsule { half_height, radius }}` with finite positive validation
- `BodyDescriptor { id, kind, translation, rotation?, gravity_scale?, ... }`
- `ColliderDescriptor { id, parent_body, shape, is_trigger, offset?, material? }`
- Query types such as `RayQuery`, `RayHit`, `PointOverlap`, `ShapeOverlap`
- Contact/trigger event records that can map to `engine_events::PhysicsEvent`

The exact names may follow local style, but the implementation must keep serialized/durable IDs separate from Rapier handles.

## Metadata Schema Direction

Scene metadata should be node/component-oriented:

```json
{
  "collision": {
    "body": { "id": "body.player", "kind": "dynamic" },
    "colliders": [
      {
        "id": "collider.player.capsule",
        "shape": { "kind": "capsule", "radius": 0.35, "half_height": 0.9 },
        "trigger": false
      }
    ]
  }
}
```

Package metadata should describe reusable collision assets or collision defaults:

```toml
[[assets]]
id = "dungeon.wall.collision"
kind = "prefab"
path = "prefabs/wall.prefab"

[assets.metadata.collision]
body_kind = "static"
shape = { kind = "box", half_extents = [0.5, 1.25, 0.5] }
trigger = false
```

Workers may choose a different exact field layout if it better matches existing structs, but must preserve durable IDs, typed validation, and backward-compatible defaults.

## Event Bridge

- Physics crate may expose engine-neutral `PhysicsContactEvent`/`PhysicsQueryHit` records.
- A bridge function can convert those records to `engine_events::EngineEvent::Physics`.
- Avoid making `engine_events` depend on `physics`; if a dependency is necessary, it should be `physics -> engine_events` only if that does not create a cycle. A separate bridge module in `physics` with an optional dependency is acceptable only if it remains non-renderer.
- Contact phases should map to Sprint 05 `ContactPhase::{Enter, Stay, Exit}`.

## Dogfood Gate

Decision order:

1. Preserve all existing dogfood collision tests.
2. If a narrow adapter can produce package/scene collision metadata or a small non-render physics proof without replacing ramp stepping, implement it.
3. If parity would require a rewrite, create `reports/dogfood-physics-migration-debt.md` with current behavior, blockers, migration slices, and validation required later.
4. Do not claim dogfood is migrated unless player movement uses the new physics foundation for a real behavior path and tests prove it.

## Validation Design

- Phase validation reports are authoritative and must be reconciled into `artifacts/validation-summary.json`.
- Each phase validator checks architecture boundaries, tests, docs drift, evidence consistency, and protected-path hygiene.
- Final quality review is required after all phase validators pass.
- Headless draw capture is not required by default for this sprint because planned changes are non-visual. If a phase changes visible renderer/editor behavior, add true `--headless --capture_target draw` evidence under `.internal-dev/captures/sprint-06-physics-collision-foundation/`.
