# Editor Object System — API Reference

## 1. Purpose & Audience

This chapter documents the unified object identity, query, selection, component, and command system for editor and inspector workflows. It covers the types and APIs that replace per-kind ad-hoc identity with one unforgeable `ObjectId` space spanning nodes, point lights, directional lights, and spot lights.

It is aimed at Rust developers building editor tools, inspectors, outliners, and command-backed mutation workflows on the engine's scene facade.

## 2. Where This Fits in Engine Flow

```
Object creation / mutation
  → ObjectId minted (unforgeable, provenance-bound)
  → ObjectRecord co-located (persistent identity, visibility, collision, components)
  → Reverse index maintained (persistent_id ↔ ObjectId lookups)
  → Commands executed through prepare/commit lifecycle
  → Queries (raycast, volume, pick) consume scene bounds + reverse index
  → Selection updated deterministically
```

The object system lives in `src/renderer/src/object/` and is re-exported through the renderer facade and the root `engine::object` / `engine::prelude` paths.

## 3. Key Concepts

### ObjectId

`ObjectId` is the unforgeable runtime identity for every scene object (node, point light, directional light, spot light). It is:

- **Private-field**: callers cannot construct IDs from raw parts; only the scene's internal mutation paths can mint them.
- **Copy + Eq + Ord + Hash**: usable as map keys, set members, and selection entries.
- **Provenance-bound**: every `ObjectId` carries a `SceneRuntimeId` minted once per `SceneWorld`. Two IDs with different provenance are never equal — preventing cross-scene identity forgery.
- **Non-serializable**: no `Serialize`/`Deserialize` derives. Runtime-only.
- **Kind-aware**: `ObjectId::kind()` returns the `ObjectKind` without needing a scene lookup.

```rust
use engine::prelude::ObjectId;

fn inspect(id: &ObjectId) {
    println!("kind={:?} slot={} gen={}", id.kind(), id.slot(), id.generation());
}
```

### ObjectKind

`ObjectKind` is the persistent enum defined in `engine_events` (dependency-neutral):

```rust
pub enum ObjectKind {
    Node,
    PointLight,
    DirectionalLight,
    SpotLight,
}
```

It is used for filtering queries, validating capabilities, and routing typed mutation paths.

### SceneObjectId

`SceneObjectId` is the persistent, dependency-neutral string identity (`"node.000001"`, `"object.<64 hex>"`) defined in `engine_events`. It survives save/load round-trips and anchors redo-by-persistent-identity in the command system.

### ObjectCapabilities

Each object kind has a declared capability set via `ObjectCapabilities::for_kind(kind)`:

| Kind | Transform | Children | Grouping | Duplication | Subtree Removal | Persistent ID |
|------|-----------|----------|----------|-------------|-----------------|---------------|
| Node | FullAffine | yes | false | yes | yes | yes |
| PointLight | TranslationOnly | false | true | yes | false | yes |
| DirectionalLight | RigidDirectionOnly | false | true | yes | false | yes |
| SpotLight | RigidWithPosition | false | true | yes | false | yes |

### ObjectTransform

The unified `ObjectTransform` enum encodes the canonical transform for each kind without caller-level branching:

```rust
pub enum ObjectTransform {
    Node(Mat4),                                // full local matrix
    PointLight(Vec3),                          // world translation
    SpotLight { position: Vec3, direction: Vec3 }, // position + direction
    DirectionalLight(Vec3),                    // normalized direction
}
```

### ObjectSummary

`ObjectSummary` provides outliner/inspector enumeration without exposing internal slot/generation details:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `ObjectId` | Runtime identity |
| `persistent_id` | `SceneObjectId` | Durable identity |
| `stable_id` | `Option<String>` | Scene-local stable ID |
| `kind` | `ObjectKind` | Object kind |
| `name` | `String` | Display name |
| `tags` | `Vec<String>` | Editor/game tags |
| `mesh_count` | `usize` | Mesh count (nodes only) |
| `child_count` | `usize` | Child count (nodes only) |
| `group_parent` | `Option<SceneObjectId>` | Group parent (lights only) |
| `visible` | `bool` | Visibility flag |
| `layer` | `Option<String>` | Layer name |
| `component_count` | `usize` | Attached component count |

### ObjectMutationOutcome

Every mutation that creates, removes, restores, or duplicates objects returns an `ObjectMutationOutcome`:

- `remaps`: old-to-new `ObjectId` remaps for editor selection remapping.
- `snapshots`: persistent `SceneObjectLifecycleSnapshot` values for event emission.
- `created_roots`: new root `ObjectId`s (e.g., for selecting duplicated subtree roots).

Convert to `ObjectLifecycleOutcome` for event emission:

```rust
let outcome: ObjectMutationOutcome = scene.remove_node(node_id)?;
let events = outcome.into_lifecycle_events(SceneObjectLifecycleAction::Removed);
for event in events {
    event_bus.emit(EventStage::PostUpdate, Some(frame_id),
        EngineEvent::Scene(SceneEvent::ObjectLifecycle(event)));
}
```

## 4. Queries and Picking

### RayHit

`RayHit` is a rich raycast result:

| Field | Type | Description |
|-------|------|-------------|
| `object` | `ObjectId` | Hit object |
| `persistent_id` | `SceneObjectId` | Persistent identity |
| `kind` | `ObjectKind` | Object kind |
| `distance` | `f32` | t-value along ray |
| `point` | `Vec3` | World-space hit point |
| `normal` | `Option<Vec3>` | Entry-face normal (None if ray started inside) |
| `is_proxy` | `bool` | Whether the hit used a proxy AABB |

### EditorPickResult

`Scene::editor_pick` casts from screen coordinates using the last camera matrices:

```rust
pub struct EditorPickResult {
    pub hits: Vec<RayHit>,
    pub indeterminate: Vec<ObjectId>,
}
```

- `hits` contains objects with known/proxy bounds sorted by distance.
- `indeterminate` contains conservative-visible objects that could not be precisely tested.

### ObjectQueryFilter

`ObjectQueryFilter` controls which objects participate in queries:

```rust
pub struct ObjectQueryFilter {
    pub kind_set: BTreeSet<ObjectKind>,  // empty = all
    pub visible_only: bool,
    pub layer: Option<String>,
    pub tags: Vec<String>,               // objects must have ALL listed tags
    pub exclude_tags: Vec<String>,        // objects must have NONE of these
}
```

### VolumeQuery

Volume queries test objects against spherical, AABB, or frustum volumes and return matching `ObjectId`s with optional distance sorting.

## 5. Selection

`Selection` is a caller-owned ordered set of `ObjectId`s:

- **Non-serializable** — no serde derives.
- **Provenance-scoped** — cross-scene IDs are rejected.
- **Ordered insertion** — duplicates are skipped, order is preserved.
- **Primary selection** — the first entry is the "primary" selection.
- **Stale cleanup** — `cleanup_stale(predicate)` removes objects no longer in the scene.

```rust
use engine::prelude::{Selection, SelectionChange};

let mut selection = Selection::new();

// Add objects (deduplicated, order-preserving)
selection.add(id_a);
selection.add(id_b);

// Set exact selection (replaces all)
selection.set_exact(vec![id_b, id_c]);

// Remove objects
let removed = selection.remove(&[id_a]);

// Clear
selection.clear();

// Primary
let primary = selection.primary();

// Iterate
for id in selection.iter() {
    // ...
}

// Cleanup stale
selection.cleanup_stale(|id| scene.is_valid_object(*id));
```

### SelectionChange

Each mutation returns a `SelectionChange`:

```rust
pub struct SelectionChange {
    pub before_primary: Option<ObjectId>,
    pub after_primary: Option<ObjectId>,
    pub added: Vec<ObjectId>,
    pub removed: Vec<ObjectId>,
}
```

## 6. EditorCamera

`EditorCamera` provides a viewpoint-independent editor camera model for inspector and editor tooling. It is re-exported through `engine::prelude`:

```rust
pub struct EditorCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,         // radians
    pub near: f32,
    pub far: f32,
    pub projection: EditorProjection,
}

pub enum EditorProjection {
    Perspective,
    Orthographic { height: f32 },
}
```

## 7. Component System

The component document model (`src/renderer/src/object/component.rs`) provides a canonical-JSON-authoritative multi-instance component store:

### Constraints

| Limit | Value |
|-------|-------|
| Attachments per object | ≤ 256 |
| Envelope data size | ≤ 1 MiB |
| Nesting depth | ≤ 64 |
| Migration steps | ≤ 32 |

### ComponentKey and ComponentInstanceId

- `ComponentKey`: typed string key identifying a component type (e.g., `"editor_sample.health"`).
- `ComponentInstanceId`: per-object instance discriminator (default `"default"`).

### ComponentEnvelope

```rust
pub struct ComponentEnvelope {
    pub key: ComponentKey,
    pub instance_id: ComponentInstanceId,
    pub schema_version: u32,
    pub data: serde_json::Value,
}
```

### ComponentRegistry and ComponentAdapter

Adapters are registered per `ComponentKey` + schema version:

```rust
pub trait ComponentAdapter: Send + Sync {
    fn key(&self) -> &ComponentKey;
    fn schema_version(&self) -> u32;
    fn migrate(&self, from_version: u32, data: &Value) -> Result<Value, ComponentError>;
    fn hydrate(&self, data: &Value) -> Result<Box<dyn Any + Send>, ComponentError>;
    fn serialize(&self, instance: &dyn Any) -> Result<Value, ComponentError>;
    fn get_property(&self, instance: &dyn Any, property: &str) -> Result<ComponentPropertyValue, ComponentError>;
    fn set_property(&self, instance: &mut dyn Any, property: &str, value: &ComponentPropertyValue) -> Result<(), ComponentError>;
    fn properties(&self) -> Vec<ComponentPropertyDescriptor>;
}
```

### ComponentPropertyValue and ComponentPropertyDescriptor

```rust
pub enum ComponentPropertyValue {
    Bool(bool),
    I32(i32),
    F32(f32),
    String(String),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

pub struct ComponentPropertyDescriptor {
    pub name: String,
    pub property_type: ComponentPropertyType,
    pub default_value: ComponentPropertyValue,
    pub description: String,
}
```

### Core Operations

- `prepare_property_edit` / `prepare_full_state_replacement` / `prepare_reference_remap`: staging operations that validate before commit.
- `commit_full_state_replacement`: applies a staged full-state replacement.
- `hydrate_envelope`: deserialize one envelope through its registered adapter.
- `hydrate_all` / `hydrate_all_by_key`: bulk hydration.
- `canonical_bytes`: deterministic canonical-JSON serialization (sorted keys).
- `component_properties`: reflection for inspector UIs.
- `enforce_limits`: validate attachment count and per-envelope data size.

## 8. Commands

Built-in undoable object commands live in `src/renderer/src/scene/object_commands.rs` and are accessible through the `Command` trait via the scene facade. All commands use a `Prepared → Executed → Undone` state machine:

### Failure Contract

- **Failed execute**: world unchanged, redo stack NOT cleared.
- **Failed undo**: command stays at undo top.
- **Failed redo**: command stays at redo top.

All commands use the prepare/commit lifecycle for failure atomicity.

### Built-in Commands

| Command | Description |
|---------|-------------|
| `SetObjectTransformCommand` | Set transform for any object kind, anchored by persistent ID |
| `SetObjectParentCommand` | Reparent node or regroup light |
| `RemoveObjectCommand` | Remove any object kind (subtree for nodes, ungroup for lights) |
| `DuplicateObjectCommand` | Duplicate nodes or lights with identity remapping |
| `CreateObjectCommand` | Create a new object of any kind |
| `SetObjectNameCommand` | Set display name |
| `SetObjectTagsCommand` | Replace tag list |
| `SetComponentPropertyCommand` | Set a single component property on an object |
| `CommitComponentStateCommand` | Commit a full component state replacement |

### CommandHistory Integration

```rust
let mut history = CommandHistory::new(128);
let cmd = Box::new(SetObjectTransformCommand::new(
    persistent_id, ObjectKind::Node, new_transform,
));
scene.execute_command(&mut history, cmd)?;

// Undo/redo
scene.undo_command(&mut history)?;
scene.redo_command(&mut history)?;
```

## 9. Code Walkthrough

Snippet Type: Real

```rust
// Create objects through the unified scene API
use engine::prelude::{ObjectId, ObjectKind, SceneObjectId};

// Create a node
let node_id: ObjectId = scene.create_object(ObjectKind::Node, None, "My Node", None)?;

// Create a point light grouped under a node
let light_id: ObjectId = scene.create_object(
    ObjectKind::PointLight,
    Some(ObjectParent::Node(node_id)),
    "Torch",
    None,
)?;

// Query all objects of a kind
let summaries: Vec<ObjectSummary> = scene.object_summaries(Some(ObjectKind::Node));

// Find by persistent ID
if let Some(found) = scene.find_object_by_persistent_id(&SceneObjectId::new("node.000001")) {
    println!("Found: {:?}", found);
}
```

Snippet Type: Real

```rust
// Raycast picking
use engine::prelude::{EditorPickResult, Ray};

let ray = Ray::from_screen(
    cursor_x, cursor_y,
    screen_width, screen_height,
    &view_matrix, &projection_matrix,
);

// Immutable read-only pick
let result: EditorPickResult = scene.editor_pick(&ray)?;

// Pick using last camera matrices from render_scene
let result: EditorPickResult = scene.pick_last_camera(cursor_x, cursor_y)?;

for hit in &result.hits {
    println!("Hit {} at distance {}", hit.persistent_id, hit.distance);
}
```

Snippet Type: Real

```rust
// Volume query
use engine::prelude::{VolumeQuery, VolumeShape, ObjectQueryFilter};

let filter = ObjectQueryFilter {
    kind_set: [ObjectKind::Node, ObjectKind::PointLight].into_iter().collect(),
    visible_only: true,
    ..Default::default()
};

let query = VolumeQuery {
    shape: VolumeShape::Aabb(aabb),
    filter,
    sort_by_distance: true,
};

let hits: Vec<ObjectId> = scene.query_volume(&query)?;
```

## 10. Best Practices

- Use `SceneObjectId` for durable identity in serialization and editor selection anchors; never persist `ObjectId`.
- Convert `ObjectMutationOutcome` to lifecycle events at the app's frame boundary, not inside mutation callbacks.
- Build editor selection remapping from `ObjectMutationOutcome::remaps` after undo/redo/duplicate operations.
- Use `ObjectCapabilities::for_kind()` to validate operations before attempting them.
- Keep `Selection` provenance-scoped; call `set_provenance` when switching scenes.
- Register component adapters early (before scene construction) and never change schema versions without migration paths.

## 11. Gotchas & Failure Modes

- `ObjectId` fields are private — attempting to construct one from raw parts will not compile.
- Cross-scene identity forgery is impossible because `SceneRuntimeId` comparison rejects IDs from different scenes.
- Stale `ObjectId` values after removal return `ObjectError::StaleGeneration` or `ObjectError::VacantObject`.
- `ObjectError::WrongKind` is returned when the object exists but has the wrong `ObjectKind`.
- `ObjectError::DuplicatePersistentId` is returned when two objects claim the same `SceneObjectId`.
- `ObjectError::GenerationExhausted` means the slot has exhausted all 2^32 generation values.
- Component operations fail with typed `ComponentError` variants — never silently corrupt state.
- Command state machine violations (double-execute, undo without execute) return `CommandError`.
- Selection mutations that add IDs from a different scene provenance are silently rejected.
- Volume queries may return empty results when all objects in range are conservative-visible with `UnknownBoundsPolicy::Exclude`.

## 12. Debugging Playbook

- Step 1: inspect `ObjectId` display format (`ObjectId(Node, slot=0, gen=1)`) for stale slot/generation.
- Step 2: check provenance — `ObjectError::WrongScene` means the object belongs to a different scene.
- Step 3: verify `ObjectKind` with `id.kind()` before calling kind-specific APIs.
- Step 4: for command failures, read the `CommandError` message — it includes the persistent ID and operation context.
- Step 5: for component hydration failures, inspect `ComponentError::HydrationFailed { key, version, message }`.
- Step 6: for pick/query misses, check `UnknownBoundsPolicy` — conservative-visible objects may be excluded.

## 13. Cross-Module Links

- Object identity kernel: `src/renderer/src/object/identity.rs`
- Query DTOs: `src/renderer/src/object/query.rs`
- Selection: `src/renderer/src/object/selection.rs`
- Component system: `src/renderer/src/object/component.rs`
- Object store (records + reverse index): `src/renderer/src/scene/object_store.rs`
- Object commands: `src/renderer/src/scene/object_commands.rs`
- Root facade re-exports: `src/object.rs`
- Event contracts: `src/events/src/lib.rs`

## 14. Standard References

- Rust ownership and type-safety: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
- JSON Schema (component validation): https://json-schema.org/
- Canonical JSON (deterministic serialization): https://datatracker.ietf.org/doc/html/rfc8785

## 15. See Also

- [Scene Graph & Fragment Workflows](03-scene-graph-and-fragment-workflows.md) — scene construction and mutation patterns
- [Events & Lifecycle](12-events-and-lifecycle.md) — event emission and scene object lifecycle events
- [Internal Scene Object Lifecycle](../internal/20-scene-object-lifecycle.md) — internal prepare/commit, ObjectRecord, reverse index
- [App-Owned Loop](15-app-owned-loop.md) — the app-owned frame boundary where lifecycle events should be emitted
