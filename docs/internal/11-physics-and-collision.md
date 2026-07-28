# Physics and Collision

## 1. Purpose & Audience

This page is for contributors working on the alpha physics crate, collision metadata validation, or future scene/editor/runtime physics loading.

The current implementation is a renderer-independent foundation. It provides stable engine-facing IDs and validation-ready metadata contracts without claiming a complete gameplay physics stack.

## 2. Boundary Map

```text
package manifests / scene files
  -> renderer validators
      -> durable collision metadata diagnostics

app crates
  -> physics crate
      -> Rapier runtime state behind durable IDs
      -> optional EngineEvent::Physics emission

renderer/root runtime
  -> validate and preserve collision metadata
  -> do not automatically instantiate live physics worlds yet
```

`src/physics` may depend on `engine_events` to translate its own records into event payloads. `engine_events` must remain independent of `physics`, renderer, editor, audio, and scripting crates.

## 3. Current Implementation

The alpha `physics` crate wraps Rapier while exposing engine-owned IDs:

- `PhysicsBodyId` and `PhysicsColliderId` are durable strings used by authored/app code.
- `BodyDescriptor` supports `Static`, `Dynamic`, and `Kinematic` bodies.
- `ColliderDescriptor` supports cuboid, sphere, Y-capsule, convex hull, and static triangle mesh shapes, with explicit translation and rotation (applied as a Rapier `UnitQuaternion` isometry).
- `PhysicsWorld::create_body` and `create_collider` validate duplicate IDs, missing parents, finite transforms, and positive dimensions.
- `PhysicsWorld::cast_ray` returns `RayHit` using durable body/collider IDs.
- `PhysicsWorld::step` records collision and trigger `Enter`, `Stay`, and `Exit` transitions.
The crate does not depend on renderer internals, windowing, scene graph storage, editor UI, or package loading.

### 3.1 Convex Hull

`ColliderShape::ConvexHull { points: Vec<[f32; 3]> }` builds a Rapier convex hull collider
from an arbitrary point cloud. It is valid on static, dynamic, and kinematic bodies.

Validation (performed before any Rapier call):

- Rejects empty input with `ConvexHullEmpty`.
- Rejects any non-finite (`NaN` / infinity) vertex with `ConvexHullNonFiniteVertex { index }`.
- Deterministically deduplicates exact points, treating IEEE `-0.0` and `+0.0` as equal.
- Requires at least four unique points; fewer returns `ConvexHullInsufficientPoints { unique_count }`.
- Rejects coplanar, collinear, or numerically zero-volume sets with a rotation-independent
  affine-basis check. Point-to-line and point-to-plane distances are computed in `f64` and
  compared with a `1e-6` tolerance relative to the largest model-space extent, mapped to
  `ConvexHullDegenerate`.
- Delegates final degeneracy detection to Parry's `convex_hull` builder, which may also
  return `None` for zero-volume or otherwise unconstructable input.

Construction is transactional: validation runs and a Rapier collider is built before
insertion into the collider set. A failure leaves no side effects on `PhysicsWorld`.

The public payload preserves model-space coordinates; the app-owned bridge owns
instance transforms.

### 3.1 Character Controller

A Rapier-backed kinematic character controller is exposed via `CharacterController`
in `character.rs`. It wraps `KinematicCharacterController` and provides
`move_and_slide` with slope/step configuration. The controller references
durable `PhysicsBodyId` / `PhysicsColliderId`; Rapier handles are private.

### 3.2 Atomic Registration and Reconfiguration

`register_body(BodyRegistrationRequest)` validates body + collider descriptors,
builds Rapier objects, and commits atomically. `reconfigure_body_mode` changes a
body's kind in place. `replace_collider` swaps a collider's shape while preserving
the parent body.

### 3.3 Targeted Removal

`remove_body_with_outcome` / `remove_collider_with_outcome` return `RemovalOutcome`
with sorted exit records for active pairs. The existing `remove_body` / `remove_collider`
bool-returning methods remain as compatibility shims.

### 3.4 Force, Impulse, and Velocity

Typed `apply_force`, `apply_impulse`, `apply_torque_impulse`, `wake_body`,
`sleep_body`, `set_linear_velocity`, `set_angular_velocity`, and `teleport_body`
methods are available. Static/kinematic bodies are silently ignored for mutation
operations without error.

### 3.5 Queries

`query.rs` provides `sweep_test`, `overlap_sphere`, and `overlap_aabb` with
deterministic result ordering by Rapier entity handle.

### 3.6 Versioned Config DTOs

`components.rs` defines serde-enabled `BodyConfigV1`, `ColliderConfigV1`,
`CharacterConfigV1` and shape config DTOs. No Rapier handles or engine-private
types are included.

## 4. Collision Metadata Validation

Package manifests may include manually authored collision metadata under `assets.metadata.collision`:

```toml
[assets.metadata.collision]
body_id = "body.wall_stone_2m"
collider_id = "collider.wall_stone_2m"
body_kind = "static"
trigger = false
shape = { kind = "box", half_extents = [1.0, 1.0, 0.125] }
```

Scene nodes may include a `collision` component:

```json
{
  "collision": {
    "body": { "id": "body.wall_north_001", "kind": "static" },
    "colliders": [
      {
        "id": "collider.wall_north_001",
        "shape": { "kind": "box", "half_extents": [1.0, 1.0, 0.125] },
        "trigger": false,
        "asset": "core.collision.wall",
        "offset": [0.0, 0.0, 0.0]
      }
    ]
  }
}
```

Validators reject non-durable collision IDs, duplicate IDs, invalid body kinds, invalid dimensions, invalid offsets, unknown referenced collision asset IDs, and serialized runtime handles. Supported primitive shape names are `box`, `cuboid`, `sphere`, `capsule`, and `capsule_y`.

## 5. Event Bridge

Physics records can opt in to the alpha event vocabulary:

- `RayHit::to_engine_event` emits `PhysicsEvent::QueryHit`.
- `PhysicsContactRecord::to_engine_event` emits `PhysicsEvent::Collision` or `PhysicsEvent::Trigger`.
- `contact_records_to_engine_events` converts a slice of records for app-owned dispatch.
- `emit_contact_records` emits contact records into a caller-provided `EventBus`.

This is a bridge, not an engine-wide scheduler. Apps own when they step physics, when they emit records, and how they mutate gameplay state in response.

## 6. Deferred Integration

The following are intentionally outside the current foundation:

- automatic scene collision component loading into `PhysicsWorld`;
- editor UI for collision body/collider authoring;
- mesh-derived bounds or convex decomposition;
- character controller migration for `apps/dungeon_dogfood`;
- runtime fixed-step scheduling and interpolation;
- physics debug drawing;
- broad gameplay collision/audio/script event orchestration.

Dogfood now has an app-owned `MeshColliderBridge` that consumes DTO recipes and
runs `PhysicsWorld` steps synchronized with gameplay ticks. Recipe handles invalidate
immediately on unload/cancellation/failure; generic renderer fence serials delay payload
release and slot reuse. Dynamic writeback preserves body translation/rotation plus the
model's original scale/orientation. This is a proof of integration, not a replacement for
the bespoke `CollisionWorld`.

### 6.1 Collider Rotation

`ColliderDescriptor` supports `rotation: [f32; 4]` (quaternion `[x, y, z, w]`) with a
`rotation()` builder method. The pose is applied through `na::UnitQuaternion` in
`create_collider`, enabling colliders to be instantiated with explicit orientation
independent of the parent body's translation.

Dogfood migration currently remains a debt item because the app has bespoke capsule, tile, ramp, step, and camera-intent collision behavior that needs a focused migration plan before replacement.

## 7. Validation Guidance

Core checks for physics/collision changes:

```sh
cargo fmt --check
cargo test -p physics
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p physics
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
cargo check
```

Use true headless draw-target capture only when a physics/collision change affects visible renderer or editor behavior. Metadata validation, event bridge, and docs-only changes should be validated with compile/tests and targeted CLI cases.

## 8. Cross-Module Links

- Physics crate: `src/physics/src/lib.rs`
- Event crate: `src/events/src/lib.rs`
- Package validators: `src/renderer/src/data/asset_registry.rs`
- Scene validators: `src/renderer/src/api/scene.rs`
- Packaging CLI tests: `tools/engine_pack/tests/cli_validation.rs`
- Public scene contract: `docs/api/03-scene-graph-and-fragment-workflows.md`
- Public asset/package contract: `docs/api/04-assets-sync-deferred-and-handles.md`
- Public event contract: `docs/api/12-events-and-lifecycle.md`
