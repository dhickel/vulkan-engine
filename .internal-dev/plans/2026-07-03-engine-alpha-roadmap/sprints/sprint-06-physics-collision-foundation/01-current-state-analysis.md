# Current State Analysis

## Verified Code Facts

- `src/physics/src/lib.rs` currently wraps Rapier with `PhysicsWorld`, raw `RigidBodyHandle` and `ColliderHandle` return values, cuboid attachment, body position lookup, stepping, and two tests.
- `src/events/src/lib.rs` already owns `PhysicsBodyId`, `ColliderId`, `ContactPhase`, and `PhysicsEvent::{Collision, Trigger, QueryHit}` as renderer-independent event contracts.
- `src/renderer/src/data/asset_registry.rs` owns package manifests, durable asset IDs, asset kinds, arbitrary metadata maps, package validation, runtime-handle diagnostics, and source-file checks.
- `src/renderer/src/api/scene.rs` owns serialized scene contracts, stable scene node IDs, asset references, validation, runtime-handle diagnostics, and unknown asset checks.
- `tools/engine_pack/src/main.rs` delegates validation to renderer package/project/scene APIs and has CLI tests under `tools/engine_pack/tests/`.
- `apps/dungeon_dogfood/src/collision.rs` owns custom wall/ramp/floor collision with ramp stepping and behavior tests; `main.rs` builds `CollisionWorld::from_level` and calls `collision::resolve_player_step`.
- Docs currently index renderer/input/events/runtime/packaging but do not expose physics/collision as a supported alpha contract.

## Architecture Fit

- The cleanest boundary is `physics` as the core simulation/query crate, `engine_events` as the event vocabulary crate, and renderer/package/scene as durable metadata owners.
- `engine_pack` should continue using renderer validation APIs; duplicating collision validation in CLI code would create drift.
- Scene metadata belongs beside serialized nodes/assets because collision components are authored data, not runtime renderer handles.
- Package metadata can either use typed fields on `PackageAssetRecord` or a typed parse/validation layer over `metadata`; Phase 02 must choose one and test it.

## Gaps

- No durable physics body/collider IDs in `physics`.
- No descriptor API for body kind, shape, trigger/sensor behavior, or collider metadata.
- No query API.
- No event extraction or bridge from Rapier contacts/intersections to Sprint 05 physics events.
- No package/scene collision metadata schema.
- No validation for collision dimensions, body kinds, shape kinds, trigger flag semantics, or runtime handle leakage in collision metadata.
- Dogfood collision is local to dogfood and not represented as engine-level package/scene data.
- Docs still mark physics/collision integration as deferred.

## Risks

- Rapier handles are tempting but not durable; workers must keep runtime handles private or explicitly marked runtime-only.
- Trigger/contact extraction can be subtle because collision events require active event handlers or contact-pair inspection; Phase 01 should prefer deterministic tests over overbroad API.
- Scene/package schema changes may need backward-compatible defaults.
- Dogfood ramp stepping is specialized and likely not equivalent to generic rigid body collision in this sprint.
- Renderer scene currently uses proxy AABBs for picking and does not own CPU mesh bounds; generated mesh bounds must remain a placeholder unless implemented intentionally.

## Validation Blind Spots

- `cargo check` can pass while schema validation misses runtime-handle-shaped nested metadata; tests must include negative JSON/TOML cases.
- Physics tests can pass while IDs accidentally expose Rapier handles in serialized examples; docs and validation must check the file contract.
- Dogfood compile checks do not prove gameplay parity; existing collision tests must remain meaningful if touched.
- Headless capture is irrelevant for pure physics unless visible runtime behavior changes.

