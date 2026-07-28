# Scene Object Lifecycle — Internal Implementation

## Purpose

This chapter describes the renderer-internal object lifecycle kernel. It is for changes under `src/renderer/src/object/`, `src/renderer/src/scene/object_store.rs`, and `src/renderer/src/scene/object_commands.rs`. Public usage belongs in [Editor Object System](../api/18-editor-object-system.md).

## Ownership and identity

Each occupied node, point-light, directional-light, or spot-light entry owns one co-located `ObjectRecord`. A vacant slot owns neither payload nor record. `ObjectRecord` holds the persistent `SceneObjectId`, stable ID, optional light group parent, visibility/collision/prefab/shadow metadata, and component store. Do not add parallel maps for record metadata.

`SceneWorld` mints one opaque `SceneRuntimeId` from `getrandom`; every `ObjectId` includes it. `ObjectId::from_parts` and raw slot/generation accessors are `pub(crate)`. The only public identity operation is reading its kind/provenance. Runtime IDs are never serialized.

The persistent reverse index is a `HashMap<SceneObjectId, ObjectHandle>`. `ObjectHandle` is a typed internal handle for a node or one of the three light kinds. The index is updated with its payload during mutation and rebuilt while loading; it is not serialized.

## Lifecycle rules

Object operations use a prepare/commit boundary:

1. **Prepare** validates provenance, kind, finite values, hierarchy/grouping rules, capacities, persistent-ID uniqueness, and generation availability; it reserves and builds typed plan data without visible mutation.
2. **Commit** applies the prepared plan, updates payload/record/index state, and produces the necessary snapshots or remaps. A prepared commit is non-fallible.

Generation reuse uses checked increments. Exhaustion returns `GenerationExhausted` rather than wrapping into a possibly live identity. Failed preparation must leave the scene unchanged.

`SceneWorld::audit_object_invariants()` verifies slot/record/index bijection, persistent-ID uniqueness, typed handle consistency, hierarchy and grouping validity, directional-shadow ownership, free-list integrity, and root consistency. Call it after complex lifecycle tests.

## Mutation boundaries

- Node subtree removal snapshots nodes and detaches grouped lights that survive the removal.
- Restoration mints fresh runtime IDs, retains persistent identities, and returns remaps for stale selection entries.
- Duplication mints new runtime and persistent IDs, then produces deterministic remaps and snapshots.
- Light grouping is metadata only: it does not create transform inheritance.
- Object commands are stateful `Prepared → Executed → Undone` commands. Redo resolves current runtime objects by persistent identity.

The concrete command types are `SetObjectTransformCommand`, `SetObjectParentCommand`, `RemoveObjectsCommand`, `DuplicateObjectsCommand`, `AttachComponentCommand`, `RemoveComponentCommand`, `ReplaceComponentStateCommand`, and `SetComponentPropertyCommand`.

## Components

`object/component.rs` keeps canonical JSON as the authoritative persisted component state. Hydrated typed values are runtime views. Validate structure and fixed limits before adapter calls; migration, hydration, serialization, reflection, property edits, and reference remapping enter the panic guard. Preserve unknown envelopes rather than dropping them.

When duplicating or restoring component-bearing objects, use the prepared reference-remap path and commit the candidate envelope/view pair together. Do not rerun migration after a successful candidate has been prepared.

## Persistence

Format v2 writes record metadata and component envelopes, not `ObjectId`, slots, generations, or the reverse index. Loading reconstructs records, runtime IDs, and the index while preserving v2 persistent IDs. V1 migration derives missing persistent IDs deterministically before the normal reconstruction path. This is backward compatible for new readers; old v1 readers cannot interpret enriched v2 files.

## Contributor checklist

- Keep records co-located with their payloads and update the reverse index in the same commit.
- Never expose a raw `ObjectId` constructor, serde implementation, `Default`, or conversion from a typed runtime handle.
- Keep public API docs aligned with the actual typed creation/removal APIs; there is no generic `create_object` facade.
- Keep lifecycle event payloads persistent-only. Scene and renderer do not own an event bus or automatically call the legacy adapter.
- Run the focused renderer object tests and `audit_object_invariants` checks after lifecycle changes.

## Source map

- Object identity: `src/renderer/src/object/identity.rs`
- Object types, capabilities, lifecycle outcomes: `src/renderer/src/object/mod.rs`
- Components: `src/renderer/src/object/component.rs`
- Queries and selection: `src/renderer/src/object/query.rs`, `src/renderer/src/object/selection.rs`
- Object storage and audit: `src/renderer/src/scene/object_store.rs`
- Scene facade: `src/renderer/src/api/scene.rs`
- Undoable commands: `src/renderer/src/scene/object_commands.rs`
- Event vocabulary: `src/events/src/lib.rs`

## See also

- [Editor Object System](../api/18-editor-object-system.md)
- [Scene Flattening and Culling](08-scene-flattening-and-culling.md)
- [Event System and Lifecycle](10-event-system-and-lifecycle.md)
