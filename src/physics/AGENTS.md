# Physics Crate Agent Guide (`src/physics`)

Use this guide for physics body/collider identity, simulation stepping, and engine event integration.

## Crate Role

`physics` provides a renderer-independent alpha physics API built on Rapier:

- Durable `PhysicsBodyId` and `PhysicsColliderId` for authored identity
- `PhysicsWorld` wrapping Rapier's rigid body and collider sets
- Collision and contact force event converters that bridge to `engine_events` types
- Query interfaces for ray casts and spatial lookups

## Public API

- `PhysicsBodyId`, `PhysicsColliderId` — stable authored identity
- `BodyDescriptor`, `ColliderDescriptor` — creation descriptors
- `PhysicsWorld` — simulation container (step, add/remove, query)
- `BodyKind` — Static, Dynamic, Kinematic
- `ColliderShape` — Cuboid, Sphere, CapsuleY, TriMeshStatic
- `RayHit` — ray cast result with body/collider IDs
- `PhysicsContactRecord` — collision contact record
- `to_engine_event()` — converters that produce `EngineEvent::Physics` variants
- `emit_contact_records()` — helper that emits contact events into an external `EventBus`

## Architecture

- Wraps `rapier3d` internals; Rapier handles are not exposed
- `PhysicsWorld` does **not** own an `EventBus` — callers pass an external bus to `emit_contact_records()` for event emission
- Bodies and colliders are identified by durable string IDs
- Simulation step is explicit and synchronous
- `From` impls exist between crate-local IDs (`PhysicsBodyId`, `PhysicsColliderId`) and `engine_events` ID types

## Current Alpha Status

- Core simulation (step, add/remove, collision events) works
- No engine-level collision metadata or scene integration
- No joint/constraint support
- Dogfood app has separate collision (`apps/dungeon_dogfood/src/collision.rs`)
- Track for full engine integration: Track E (future sprint)

## Deprecation Notice

Compatibility helpers exist for the original smoke tests that use Rapier handles directly (`create_dynamic_body`, `create_static_body`, `attach_cuboid`, `body_position`). New code should use `PhysicsBodyId`/`PhysicsColliderId`.

## Working Rules

- Do not expose Rapier types in the public API
- Keep body/collider IDs as durable string identifiers
- Emit events through `engine_events::EngineEvent::Physics` variant using the converter functions
- If docs and code diverge, treat code as logical truth

## Validation

- `cargo check -p physics`
- `cargo test -p physics`
