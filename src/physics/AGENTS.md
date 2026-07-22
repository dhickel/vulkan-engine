# Physics Crate Agent Guide (`src/physics`)

Use this guide for physics body/collider identity, simulation stepping, and engine event integration.

## Crate Role

`physics` provides a renderer-independent alpha physics API built on Rapier:

- Durable `PhysicsBodyId` and `PhysicsColliderId` for authored identity
- `PhysicsWorld` wrapping Rapier's rigid body and collider sets
- Collision, trigger, and ray-query event converters that bridge to `engine_events` types
- Ray casts plus durable-ID body/collider lookups

## Public API

- `PhysicsBodyId`, `PhysicsColliderId` — stable authored identity
- `BodyDescriptor`, `ColliderDescriptor` — creation descriptors
- `PhysicsWorld` — simulation container (step, add/remove, query)
- `BodyKind` — Static, Dynamic, Kinematic
- `ColliderShape` — Cuboid, Sphere, CapsuleY, TriMeshStatic, ConvexHull
- `RayHit` — ray cast result with body/collider IDs
- `PhysicsContactRecord` — collision contact record
- `to_engine_event()` — converters that produce `EngineEvent::Physics` variants
- `emit_contact_records()` — helper that emits contact events into an external `EventBus`

## Architecture

- Wraps `rapier3d` internals; Rapier handles are not exposed
- `PhysicsWorld` does **not** own an `EventBus` — callers pass an external bus to `emit_contact_records()` for event emission
- Bodies and colliders are identified by durable string IDs
- Simulation step is explicit and synchronous
- `PhysicsBodyId` and `PhysicsColliderId` are re-exported from `engine_events`; there are no crate-local wrapper IDs

## Current Alpha Status

- Core simulation (step, add/remove, collision events) works
- No engine-level collision metadata or scene integration
- No joint/constraint support
- Dogfood app has separate collision (`apps/dungeon_dogfood/src/collision.rs`)
- Track for full engine integration: Track E (future sprint)

## Deprecation Status

No public item is declared deprecated. The public authored API uses durable
`PhysicsBodyId` / `PhysicsColliderId` IDs exclusively. Rapier handles are
internal runtime detail and are not exposed in public signatures.

## Working Rules

- Do not expose Rapier types in the public API
- Keep body/collider IDs as durable string identifiers
- Emit events through `engine_events::EngineEvent::Physics` variant using the converter functions
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p physics`
- `cargo test -p physics`
