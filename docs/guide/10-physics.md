# 10 — Physics

> Provenance: `G-10`

> **Alpha Status**: The `physics` crate is alpha-stage. Core simulation (step, add/remove bodies and colliders, collision events, ray casting) works, but engine-level collision metadata, joints, constraints, and scene integration are not yet implemented. This chapter documents what exists today without production-readiness promises. Track E (future sprint) will bring full engine integration.

This chapter covers the renderer-independent physics crate: creating rigid bodies and colliders, running simulation steps, querying contacts, casting rays, emitting physics events, and validating collider geometry.

For the full API reference, see [`src/physics/src/lib.rs`](../../src/physics/src/lib.rs). The crate wraps Rapier internally but exposes only engine-facing types.

## Architecture

The `physics` crate is **renderer-independent**. It depends only on `engine_events` (for ID types and event contracts) and `rapier3d` (wrapped internally). It does not depend on the renderer, Vulkan, windowing, or any other engine crate.

```
┌─────────────────────────────────┐
│         PhysicsWorld            │
│                                 │
│  body_handles: BTreeMap<        │
│    PhysicsBodyId,               │
│    RigidBodyHandle  (internal)  │
│  >                              │
│                                 │
│  collider_handles: BTreeMap<    │
│    PhysicsColliderId,           │
│    ColliderHandle    (internal) │
│  >                              │
│                                 │
│  step(dt) ──► contact records   │
│  cast_ray() ──► RayHit          │
│                                 │
│  emit_contact_records(bus, ..)  │
└─────────────────────────────────┘
```

Rapier handles are never exposed in the public API. All authored code uses durable `PhysicsBodyId` and `PhysicsColliderId` values.

## Key Types

| Type | Purpose |
|------|---------|
| `PhysicsWorld` | Simulation container with step, add/remove, query |
| `PhysicsBodyId` | Durable string ID for a rigid body (re-exported from `engine_events`) |
| `PhysicsColliderId` | Durable string ID for a collider (re-exported from `engine_events`) |
| `BodyDescriptor` | Creation descriptor: ID, kind, initial translation |
| `BodyKind` | `Static`, `Dynamic`, or `Kinematic` |
| `ColliderDescriptor` | Creation descriptor: ID, parent body, shape, trigger flag, offset |
| `ColliderShape` | `Cuboid`, `Sphere`, `CapsuleY`, `TriMeshStatic`, `ConvexHull` |
| `RayQuery` | Ray cast request: origin, direction, max TOI |
| `RayHit` | Ray cast result: body, collider, time of impact |
| `PhysicsContactRecord` | Per-pair contact: phase, kind, collider IDs |
| `PhysicsContactPhase` | `Enter`, `Stay`, `Exit` |
| `PhysicsContactKind` | `Collision` or `Trigger` |
| `PhysicsError` | Validation error with typed variants |
| `BodyPose` | Translation + rotation (quaternion `[x, y, z, w]`) |

## Creating a World

> Provenance: `G-10-WORLD` — Excerpt

```rust
use physics::PhysicsWorld;

let mut world = PhysicsWorld::new();

// Customize gravity (default: (0.0, -9.81, 0.0))
world.set_gravity(0.0, -10.0, 0.0);
```

## Bodies

### Creating Bodies

> Provenance: `G-10-BODIES` — Excerpt

```rust
use physics::{BodyDescriptor, BodyKind, PhysicsBodyId};

// Static body (does not move, e.g., floor, wall)
let floor_id: PhysicsBodyId = world.create_body(BodyDescriptor::new(
    "body.floor",
    BodyKind::Static,
    [0.0, 0.0, 0.0],
))?;

// Dynamic body (affected by gravity and forces)
let player_id = world.create_body(BodyDescriptor::new(
    "body.player",
    BodyKind::Dynamic,
    [0.0, 2.0, 0.0],
))?;

// Kinematic body (moved by code, not affected by forces)
let platform_id = world.create_body(BodyDescriptor::new(
    "body.platform",
    BodyKind::Kinematic,
    [5.0, 1.0, 0.0],
))?;
```

| Kind | Gravity? | Forces? | Moved by Code? |
|------|:--------:|:-------:|:--------------:|
| `Static` | No | No | No |
| `Dynamic` | Yes | Yes | Via forces/impulses |
| `Kinematic` | No | No | Yes (set position) |

### Querying Body State

```rust
// Get position
if let Some(pos) = world.body_position_by_id(&player_id) {
    println!("player at {:?}", pos);
}

// Get full pose
if let Some(pose) = world.body_pose_by_id(&player_id) {
    println!(
        "translation={:?} rotation=[{:.2}, {:.2}, {:.2}, {:.2}]",
        pose.translation, pose.rotation[0], pose.rotation[1],
        pose.rotation[2], pose.rotation[3],
    );
}
```

### Removing Bodies

```rust
// Removes the body and all attached colliders
world.remove_body(&player_id);
```

## Colliders

### Creating Colliders

> Provenance: `G-10-COLLIDERS` — Excerpt

```rust
use physics::{ColliderDescriptor, ColliderShape, PhysicsColliderId};

// Box collider
world.create_collider(ColliderDescriptor::new(
    "collider.player",
    player_id.clone(),
    ColliderShape::Cuboid {
        half_extents: [0.5, 1.0, 0.5],
    },
))?;

// Sphere collider
world.create_collider(ColliderDescriptor::new(
    "collider.ball",
    "body.ball",
    ColliderShape::Sphere { radius: 0.5 },
))?;

// Capsule (Y-axis aligned)
world.create_collider(ColliderDescriptor::new(
    "collider.character",
    "body.character",
    ColliderShape::CapsuleY {
        half_height: 0.8,
        radius: 0.4,
    },
))?;
```

### Offset Colliders

```rust
world.create_collider(
    ColliderDescriptor::new(
        "collider.offset",
        "body.player",
        ColliderShape::Cuboid {
            half_extents: [0.3, 0.3, 0.3],
        },
    )
    .translation([0.0, 1.5, 0.0])  // offset from body origin
    .rotation([0.0, 0.0, 0.0, 1.0]), // offset rotation
)?;
```

### Triggers (Sensors)

Triggers detect overlap without physical collision response:

```rust
world.create_collider(
    ColliderDescriptor::new(
        "collider.detection_zone",
        "body.zone",
        ColliderShape::Sphere { radius: 3.0 },
    )
    .trigger(true),  // sensor: detects overlap, no physics response
)?;
```

### Removing Colliders

```rust
world.remove_collider(&collider_id);
```

## TriMeshStatic (Triangle Mesh Colliders)

> Provenance: `G-10-TRIMESH` — Excerpt

Static triangle meshes enable precise collision with complex geometry like terrain and buildings:

```rust
world.create_collider(ColliderDescriptor::new(
    "collider.terrain",
    "body.terrain",
    ColliderShape::TriMeshStatic {
        vertices: vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        ],
        indices: vec![[0, 1, 2], [1, 3, 2]],
    },
))?;
```

**Constraints:**
- TriMeshStatic colliders may only be attached to **static** bodies. Attaching to dynamic or kinematic bodies returns `TrimeshOnDynamicBody`.
- Vertices must be finite; indices must be in-bounds and non-degenerate.
- Empty meshes and degenerate triangles are rejected at creation time.

## ConvexHull Colliders

> Provenance: `G-10-HULL` — Excerpt

Convex hull colliders are created from point clouds. The engine validates that points form a non-degenerate volume:

```rust
world.create_collider(ColliderDescriptor::new(
    "collider.rock",
    "body.rock",
    ColliderShape::ConvexHull {
        points: vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ],
    },
))?;
```

**Validation rules:**
- At least 4 unique points (non-coplanar)
- Points must be finite
- Must form a non-degenerate volume (tested by computing plane distances from an affine basis)
- Duplicate and interior points are silently deduplicated

### Offline Validation

Validate collider geometry without creating a world:

```rust
use physics::{validate_collider_shape, BodyKind};

// Check a convex hull is valid for a dynamic body
validate_collider_shape(
    &ColliderShape::ConvexHull { points: my_points },
    BodyKind::Dynamic,
)?;

// TriMeshStatic on dynamic body will be rejected:
let result = validate_collider_shape(
    &ColliderShape::TriMeshStatic { vertices, indices },
    BodyKind::Dynamic,
);
assert!(result.is_err());
```

## Simulation Stepping

> Provenance: `G-10-STEP` — Excerpt

```rust
// Step the simulation forward by a fixed timestep
world.step(1.0 / 60.0)?;

// After stepping, read contact records
for record in world.last_contact_records() {
    println!(
        "contact: {:?} {:?} between {} and {}",
        record.phase, record.kind,
        record.a, record.b,
    );
}
```

The step function:
1. Advances the Rapier physics pipeline by `dt` seconds
2. Collects collision and trigger contact records
3. Returns `PhysicsError::NonPositiveDeltaTime` if `dt <= 0.0`

### App-Owned Integration

The physics world does **not** own an `EventBus`. Your app controls when to step and how to emit events:

```rust
use physics::{PhysicsWorld, emit_contact_records};
use engine_events::{EventBus, EventStage};

fn update_physics(
    world: &mut PhysicsWorld,
    events: &mut EventBus,
    dt: f32,
) -> Result<(), physics::PhysicsError> {
    world.step(dt)?;

    // Emit contact records as EngineEvent::Physics events
    emit_contact_records(events, EventStage::PostUpdate, world.last_contact_records());

    Ok(())
}
```

## Ray Casting

> Provenance: `G-10-RAY` — Excerpt

```rust
use physics::RayQuery;

let hit = world.cast_ray(RayQuery::new(
    [0.0, 1.5, 0.0],   // origin (e.g., camera position)
    [0.0, 0.0, -1.0],  // direction (normalized)
    100.0,              // max time of impact
))?;

match hit {
    Some(hit) => {
        println!(
            "hit body={} collider={} at distance={}",
            hit.body, hit.collider, hit.time_of_impact,
        );

        // Convert to engine event
        let event = hit.to_engine_event();
        events.emit(EventStage::PostUpdate, None, event);
    }
    None => {
        // Ray missed all colliders
    }
}
```

Ray casting validates:
- Origin and direction must be finite
- Direction must be non-zero (returns `PhysicsError::ZeroDirection`)
- `max_time_of_impact` must be positive and finite

## Contact Records

### Record Structure

```rust
pub struct PhysicsContactRecord {
    pub phase: PhysicsContactPhase,  // Enter | Stay | Exit
    pub kind: PhysicsContactKind,    // Collision | Trigger
    pub a: PhysicsColliderId,        // First collider
    pub b: PhysicsColliderId,        // Second collider
}
```

For triggers, `a` is always the trigger (sensor) collider and `b` is the other collider.

### Converting to Engine Events

```rust
use physics::{contact_records_to_engine_events, emit_contact_records};

// Convert records to EngineEvent::Physics variants:
let events_list: Vec<EngineEvent> = contact_records_to_engine_events(
    world.last_contact_records()
);

// Or emit directly into an EventBus:
emit_contact_records(
    &mut events,
    EventStage::PostUpdate,
    world.last_contact_records(),
);
```

Conversion rules:
- `Collision` + `Enter`/`Stay`/`Exit` → `PhysicsEvent::Collision { phase, a, b }`
- `Trigger` + `Enter`/`Stay`/`Exit` → `PhysicsEvent::Trigger { phase, trigger, other }`

## Error Handling

All errors are typed `PhysicsError` variants:

| Variant | When |
|---------|------|
| `DuplicateBodyId` | Body ID already exists in the world |
| `DuplicateColliderId` | Collider ID already exists |
| `MissingBody` | Collider references a body ID that doesn't exist |
| `MissingCollider` | Collider ID does not exist in the world |
| `NonFiniteValue` | A numeric field contains NaN or infinity |
| `NonPositiveDimension` | A dimension (radius, half_extent) is ≤ 0 |
| `NonPositiveDeltaTime` | Step dt is ≤ 0 |
| `ZeroDirection` | Ray direction has zero length |
| `InvalidRotation` | Quaternion is non-finite or zero-length |
| `TrimeshNonFiniteVertex` | Triangle mesh vertex is non-finite |
| `TrimeshIndexOutOfBounds` | Index exceeds vertex count |
| `TrimeshEmpty` | Trimesh has no vertices or indices |
| `TrimeshDegenerateTriangle` | Two or more indices in a triangle are equal |
| `TrimeshOnDynamicBody` | Trimesh attached to non-static body |
| `ConvexHullEmpty` | Convex hull has no points |
| `ConvexHullNonFiniteVertex` | A hull point is non-finite |
| `ConvexHullInsufficientPoints` | Fewer than 4 unique points |
| `ConvexHullDegenerate` | Points are coplanar or zero-volume |

## Atomic Body+Collider Registration

> Provenance: `G-10-ATOMIC` — Excerpt

Register a body and its colliders in one validated, transactional call:

```rust
use physics::{BodyRegistrationRequest, BodyDescriptor, ColliderDescriptor, BodyKind, ColliderShape};

let outcome = world.register_body(BodyRegistrationRequest {
    body: BodyDescriptor::new("body.hero", BodyKind::Dynamic, [0.0, 2.0, 0.0]),
    colliders: vec![
        ColliderDescriptor::new(
            "collider.hero_torso",
            "body.hero",
            ColliderShape::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
        ),
        ColliderDescriptor::new(
            "collider.hero_head",
            "body.hero",
            ColliderShape::Sphere { radius: 0.25 },
        )
        .translation([0.0, 1.2, 0.0]),
    ],
})?;

// outcome.body_id, outcome.collider_ids
```

All IDs, shapes, parent references, transforms, and body-kind compatibility are
validated before any Rapier object is inserted.  On failure, the world is
unchanged.

## Body Reconfiguration

Change a body's kind in place without losing pose, velocities, or sleep state:

```rust
use physics::BodyMode;

world.reconfigure_body_mode(&player_id, BodyMode::Kinematic)?;
```

Bodies with attached `TriMeshStatic` colliders cannot be changed to `Dynamic`
or `Kinematic` — the call returns `TrimeshOnDynamicBody`.

## Collider Replacement

Swap a collider's shape and properties without disrupting the parent body:

```rust
use physics::ColliderReplacementRequest;

world.replace_collider(ColliderReplacementRequest {
    collider_id: collider_id.clone(),
    shape: ColliderShape::Cuboid {
        half_extents: [0.4, 0.8, 0.4],
    },
    is_trigger: false,
    translation: [0.0; 3],
    rotation: [0.0, 0.0, 0.0, 1.0],
})?;
```

Validation runs before the old collider is removed; a failed replacement leaves
the world intact.

## Targeted Removal with Outcomes

Both `remove_body_with_outcome` and `remove_collider_with_outcome` return a
[`RemovalOutcome`] containing sorted exit records for every active pair that
ended:

```rust
let outcome = world.remove_body_with_outcome(&player_id).unwrap();
// outcome.removed_body, outcome.removed_colliders, outcome.exited_pairs

// Active pairs are removed individually — no global contact clearing.
// The bool-returning remove_body / remove_collider remain as compat shims.
```

## Force, Impulse, Velocity, and Teleport

```rust
// Apply forces (dynamic bodies only)
world.apply_force(&player_id, [0.0, 100.0, 0.0])?;
world.apply_impulse(&player_id, [0.0, 50.0, 0.0])?;
world.apply_torque_impulse(&player_id, [1.0, 0.0, 0.0])?;

// Velocity control
world.set_linear_velocity(&player_id, [5.0, 0.0, 0.0])?;
world.set_angular_velocity(&player_id, [0.0, 1.0, 0.0])?;

// Sleep / wake
world.wake_body(&player_id)?;
world.sleep_body(&player_id)?;

// Teleport (kinematic-safe)
world.teleport_body(&player_id, BodyPose {
    translation: [10.0, 0.0, 0.0],
    rotation: [0.0, 0.0, 0.0, 1.0],
})?;
```

Forces and impulses on static or kinematic bodies are silently ignored.

## Body Introspection

```rust
world.body_is_static(&id);
world.body_is_dynamic(&id);
world.body_is_kinematic(&id);
world.body_exists(&id);
world.body_linear_velocity(&id);   // None if missing
world.body_angular_velocity(&id);  // None if missing
```

## Sweep and Overlap Queries

```rust
use physics::ColliderShape;

// Sweep a shape through space
let hit = world.sweep_test(
    &ColliderShape::Sphere { radius: 0.5 },
    BodyPose { translation: [0.0, 5.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] },
    [0.0, -10.0, 0.0],
)?;

// Overlap queries
let overlaps = world.overlap_sphere([0.0, 0.0, 0.0], 5.0)?;
let overlaps = world.overlap_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])?;

// Results are deterministic: sorted by Rapier handle.
```

## Character Controller

A Rapier-backed kinematic character controller with step/slope handling:

```rust
use physics::{CharacterConfig, CharacterController};

let config = CharacterConfig {
    max_slope_climb_angle: 0.785398,  // radians
    slide: true,
    autostep_max_height: 0.3,
    autostep_min_width: 0.2,
    autostep: true,
    ..Default::default()
};

let mut controller = CharacterController::new(
    &world, body_id, collider_id, config,
)?;

// Each fixed step:
let actual = controller.move_and_slide(
    &mut world,
    [desired_x, desired_y, desired_z],
    1.0 / 60.0,
)?;

if controller.is_on_floor() {
    // grounded
}
```

## Versioned Config DTOs

Serializable, Rapier-free configuration DTOs for bodies, colliders, and
character controllers:

```rust
use physics::components::{
    BodyConfigV1, ColliderConfigV1, CharacterConfigV1,
    BodyKindConfigV1, ColliderShapeConfigV1,
};

let body = BodyConfigV1 {
    body_id: "body.hero".into(),
    kind: BodyKindConfigV1::Dynamic,
    ..Default::default()
};

let json = serde_json::to_string(&body)?;
let round: BodyConfigV1 = serde_json::from_str(&json)?;
```

## What's Not Yet Implemented

- Joints and constraints
- CCD configuration via public API
- Engine-level collision metadata (material properties, sound triggers)
- Scene-persistence integration for physics components
- Automatic scene-to-physics body instantiation
- Physics debug drawing (gated behind `debug-lines` feature, not yet integrated)

These are tracked in future sprint work. No public item in the physics crate is currently marked deprecated.

## Runnable Verification

Run the physics crate test suite:

```sh
cargo test -p physics
```

Expected: all tests pass (body/gravity, descriptor validation, durable ID mapping, ray queries, contact records, trigger detection, TriMeshStatic constraints, convex hull validation, engine event bridging, renderer-independent step+query+events).

Build the physics crate standalone:

```sh
cargo check -p physics
```

### Physics Test Walkthrough

The test suite demonstrates a complete renderer-independent workflow:

1. Create a `PhysicsWorld`
2. Add static and dynamic bodies with colliders
3. Run `world.step(1.0/60.0)`
4. Read `world.last_contact_records()`
5. Convert records to `EngineEvent::Physics` variants
6. Emit into an `EventBus` and dispatch

All of this runs without a renderer, window, or GPU — the physics crate is fully standalone.

## Next

Continue to [11 — Audio](11-audio.md) to learn about the alpha audio crate: clip loading, device-backed playback, and the dogfood audio event bridge.
