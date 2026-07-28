# Editor Object System — API Reference

## Purpose

The scene object system gives editor and tooling code one runtime identity type for nodes and lights, while retaining the existing typed scene APIs. Use it for object enumeration, selection, queries, component documents, and undoable mutations.

The public entry points are `engine::object`, `engine::command`, `engine::camera`, and `engine::render::Scene`. `engine::prelude` deliberately includes only the common identity, selection, and query types; import advanced types from their named modules.

## Identity and object summaries

`ObjectId` is the runtime-only identity for nodes, point lights, directional lights, and spot lights. Its fields and raw constructors are private. It is `Copy + Eq + Ord + Hash`, has a public `kind()` accessor, and carries a `SceneRuntimeId` provenance token. IDs from different scene worlds compare unequal.

`SceneObjectId` is the durable, dependency-neutral identity. Persist `SceneObjectId`, never `ObjectId`.

```rust
use engine::prelude::{ObjectId, SceneObjectId};
use engine::render::Scene;
use glam::Mat4;

let mut scene = Scene::new();
let node = scene.create_node_default(None)?;
let id: ObjectId = scene.object_id(node)?;
let persistent: SceneObjectId = scene.object_summary(id)?.persistent_id;

scene.set_object_transform(id, &Mat4::from_translation([1.0, 0.0, 0.0].into()))?;
assert_eq!(scene.find_object_by_persistent_id(&persistent), Some(id));
# Ok::<(), engine::render::SceneError>(())
```

`Scene::objects`, `objects_of_kind`, `object_summary`, and `object_summaries` expose the unified view. `ObjectSummary` includes the runtime and persistent IDs, kind, display metadata, node mesh/child counts, light grouping, visibility/layer metadata, and component count without exposing slot or generation fields.

## Capabilities, transforms, and parenting

`ObjectCapabilities::for_kind(ObjectKind)` declares whether a kind supports transforms, children, grouping, duplication, subtree removal, and persistent identity. Transform capabilities are `FullAffine` (nodes), `TranslationOnly` (point lights), `RigidWithPosition` (spot lights), and `RigidDirectionOnly` (directional lights).

`Scene::get_object_transform` returns an `ObjectTransform`; `set_object_transform` accepts a canonical `Mat4` and validates it for the target kind. The typed `node_*_transform`, `point_light_*_transform`, `spot_light_*_transform`, and `directional_light_*_transform` methods remain available.

`Scene::set_object_parent` accepts `ObjectParent::{None, Node(ObjectId)}`. Nodes use normal scene-graph parenting; lights record a non-inheriting organizational group parent and keep their world-space payload unchanged.

## Queries and picking

All query methods are read-only.

- `Scene::raycast` returns the nearest `RayHit`; `raycast_all` returns deterministic distance-ordered hits.
- `Scene::query_volume` accepts `VolumeQuery` and returns deterministic `VolumeHit` values.
- `Scene::screen_to_ray` builds a ray from the last renderer-supplied camera.
- `Scene::editor_pick(&Ray, EditorProxyPolicy)` returns the nearest eligible `EditorPickResult`, if any.

`EditorPickResult` contains one `object` and an optional AABB hit. It is not a multi-hit result. Directional lights are unbounded and excluded; point and spot lights require an explicit `EditorProxyPolicy` opt-in. `UnknownBoundsPolicy` controls volume-query treatment of conservative-visible objects.

```rust
use engine::object::{EditorProxyPolicy, ObjectQueryFilter, VolumeQuery};

let ray = scene.screen_to_ray(cursor_x, cursor_y, width, height);
if let Some(pick) = scene.editor_pick(&ray, EditorProxyPolicy::NodesOnly)? {
    println!("picked {:?}", pick.object.kind());
}

let query = VolumeQuery::aabb(aabb)
    .with_filter(ObjectQueryFilter::kinds([engine::object::ObjectKind::Node]));
let hits = scene.query_volume(&query);
# Ok::<(), engine::render::SceneError>(())
```

## Caller-owned editor state

`Selection` is a caller-owned, non-serializable ordered set of `ObjectId`s. Bind it to `scene.provenance_token()` when selection is specific to a scene. `add`, `set`, and `toggle` reject a different provenance; `cleanup_stale` accepts the caller's liveness predicate. `SelectionChange` reports the primary selection before/after and added/removed IDs.

`EditorCamera` is an independent camera model in `engine::camera`. It wraps an `OrbitCamera`, supports perspective and orthographic projections, and supplies view/projection and screen-ray helpers. It is caller-owned; `Scene` neither stores nor updates it.

## Components

Component persistence is canonical-JSON authoritative. `ComponentEnvelope` stores a `ComponentKey`, `ComponentInstanceId`, schema version, and JSON data. Each object record owns a component store; the public scene attachment and hydration methods currently operate on scene nodes.

A caller-owned `ComponentRegistry` maps keys to `ComponentAdapter`s. The adapter supplies its current version, one-step migrations, hydration/serialization, property reflection, property edits, and persistent-ID reference remapping. Adapter callbacks are panic-contained. Unknown envelopes remain opaque and preserve their canonical JSON.

Limits are 256 attachments per object, 1 MiB canonical data per envelope, nesting depth 64, and 32 migration steps. Use `attach_component`, `component_envelopes`, `hydrate_component`, and `hydrate_components` on `Scene`; use `ComponentKey::new` and `ComponentInstanceId::new` to validate IDs.

## Commands and lifecycle events

Built-in object commands implement the existing `Command` trait and use `Prepared → Executed → Undone` state. The public command set is:

- `SetObjectTransformCommand`, `SetObjectParentCommand`
- `RemoveObjectsCommand`, `DuplicateObjectsCommand`
- `AttachComponentCommand`, `RemoveComponentCommand`
- `ReplaceComponentStateCommand`, `SetComponentPropertyCommand`

Commands resolve redo targets by `SceneObjectId`, not a stale runtime handle. Execute them through `Scene::execute_command`; undo and redo use the same `CommandHistory`.

Object creation, restoration, duplication, and removal can produce `ObjectMutationOutcome`. Convert it to `ObjectLifecycleOutcome` or directly to events, then explicitly emit those events on the caller-owned `EventBus`. Scene mutation never owns a bus and does not automatically emit legacy scene events.

```rust
use engine::command::{CommandHistory, SetObjectTransformCommand};
use engine::object::{ObjectKind, SceneObjectLifecycleAction};
use glam::Mat4;

let summary = scene.object_summary(id)?;
let command = Box::new(SetObjectTransformCommand::new(
    summary.persistent_id,
    ObjectKind::Node,
    Mat4::IDENTITY,
));
let mut history = CommandHistory::new(128);
scene.execute_command(&mut history, command)?;
# Ok::<(), engine::render::SceneError>(())
```

## Persistence and compatibility

Scene format v2 serializes `ObjectRecord` metadata and component envelopes, never runtime `ObjectId` values. Loading reconstructs runtime IDs and the reverse index while retaining serialized `SceneObjectId` values. V1 loads migrate to v2 and deterministically derive missing persistent IDs. New readers accept v1 and v2; older v1 readers do not understand enriched v2 files.

## See also

- [Scene Graph & Fragment Workflows](03-scene-graph-and-fragment-workflows.md)
- [Events & Lifecycle](12-events-and-lifecycle.md)
- [Internal Scene Object Lifecycle](../internal/20-scene-object-lifecycle.md)
- [Scene construction guide](../guide/08-scene-construction.md)
