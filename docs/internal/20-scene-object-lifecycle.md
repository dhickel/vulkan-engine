# Scene Object Lifecycle — Internal Implementation

## 1. Purpose & Audience

This chapter documents the internal object lifecycle kernel: how `ObjectId` values are minted, how `ObjectRecord` metadata is co-located with typed payloads, how the reverse index maps persistent IDs to runtime handles, how the prepare/commit lifecycle ensures failure atomicity, and how invariant audits detect corruption. It is for contributors working inside `src/renderer/src/scene/object_store.rs`, `src/renderer/src/object/`, or `src/renderer/src/scene/object_commands.rs`.

## 2. Where This Fits in Engine Flow

```
SceneWorld construction
  → mint_provenance() — one SceneRuntimeId per SceneWorld
  → Object slots initialized (nodes, point lights, directional lights, spot lights)

Mutation:
  → prepare_* — validate, reserve slot, build plan (no mutation)
  → commit_* — apply plan, update reverse index, produce ObjectLifecycleOutcome
  → Invariant audit available post-mutation

Query:
  → find_object_by_persistent_id — reverse index lookup
  → editor_pick / query_volume — traverse scene bounds + reverse index

Persistence:
  → Save — serialize ObjectRecord fields (NOT ObjectId)
  → Load — reconstruct ObjectRecord, mint new ObjectId + SceneObjectId
```

## 3. Key Concepts

### ObjectRecord

Every occupied typed slot (node, point light, directional light, spot light) carries exactly one `ObjectRecord` co-located with its payload. Vacant slots have `None` — there is never a record without a payload.

```rust
pub struct ObjectRecord {
    pub persistent_id: SceneObjectId,          // durable identity
    pub stable_id: Option<String>,             // scene-local ID (e.g. "node.000001")
    pub light_group_parent: Option<SceneObjectId>, // grouping for lights
    pub visibility: Option<SerializedVisibility>,   // editor visibility/lock/layer
    pub collision: Option<SerializedCollisionComponent>,
    pub prefab: Option<serde_json::Value>,
    pub directional_shadow_config: Option<DirectionalShadowConfig>,
    pub component_store: ComponentStore,         // canonical-JSON component attachments
}
```

This eliminates the parallel side-map pattern used before Phase 02 — records are always co-located with their payload, so stale-record and mismatched-entry bugs are structurally impossible.

### SceneRuntimeId

`SceneRuntimeId` is an opaque `u64` minted once per `SceneWorld` via `getrandom`. It survives moves and is embedded in every `ObjectId`. Two `ObjectId`s with different `SceneRuntimeId` compare unequal, preventing cross-scene identity forgery.

Minting is internal-only via `mint_provenance()`; there is no public constructor.

### ObjectId Minting

`ObjectId` is constructed from internal parts only:

```rust
ObjectId::from_parts(provenance, kind, slot, generation)
```

- `provenance`: the owning scene's `SceneRuntimeId`
- `kind`: `ObjectKind`
- `slot`: typed-slot index within the kind's storage
- `generation`: checked-incremented counter (prevents ABA reuse)

The `from_parts` constructor is `pub(crate)` — only renderer-internal paths can mint IDs. A `#[cfg(test)]` constructor exists for tests.

### Reverse Index

`SceneWorld` maintains a `HashMap<SceneObjectId, ObjectHandle>` reverse index that maps every persistent ID to its current typed runtime handle:

```rust
pub enum ObjectHandle {
    Node(SceneNodeId),
    PointLight(PointLightId),
    DirectionalLight(DirectionalLightId),
    SpotLight(SpotLightId),
}
```

The reverse index is updated atomically during commit and is the single source of truth for `find_object_by_persistent_id`. It is never persisted — it is rebuilt at scene load time.

### Prepare/Commit Lifecycle

Every object mutation follows a two-phase prepare/commit pattern:

1. **Prepare**: validate inputs (finite transforms, kind constraints, slot availability), check generation exhaustion, reserve slots, build a typed plan struct. No visible state mutation occurs.
2. **Commit**: apply the plan — insert payload, mint `ObjectId`, update the reverse index, build `ObjectLifecycleOutcome` snapshots. Commit is non-fallible after successful prepare.

On failure, prepare returns an `ObjectError` and leaves the world unchanged.

### Generation Management

Generation counters use `checked_add` and return `GenerationExhausted` before collision. Free lists track vacant slots; reuse bumps the generation so stale `ObjectId` values always fail validation.

### Invariant Audit

`SceneWorld::audit_object_invariants()` checks:

- **Slot/record/index bijection**: every occupied slot has exactly one `ObjectRecord` and one reverse-index entry.
- **Unique persistent IDs**: no two occupied slots share a `SceneObjectId`.
- **Typed handle consistency**: `ObjectHandle` kind matches the storage it was drawn from.
- **Valid hierarchy links**: every non-root node's parent exists and is a node.
- **Valid grouping parents**: every light's `light_group_parent` refers to an existing node.
- **Shadow-owner validity**: shadow-casting directional lights exist and have valid shadow configs.
- **Free-list integrity**: no exhausted slot in the free list, no occupied slot in the free list.
- **Single root node**: exactly one parentless node exists when nodes are present.

Returns `Ok(())` or a detailed `Err(String)` describing the first violation found.

## 4. Code Walkthrough

### Provenance Minting

```rust
pub(crate) fn mint_provenance() -> SceneRuntimeId {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("getrandom must succeed during SceneWorld construction");
    SceneRuntimeId::new(u64::from_le_bytes(buf))
}
```

### Persistent ID Minting

```rust
pub(crate) fn mint_persistent_id() -> SceneObjectId {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).expect("getrandom must succeed during persistent ID minting");
    let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    SceneObjectId::new(format!("object.{hex}"))
}
```

### ObjectHandle Enum

```rust
pub enum ObjectHandle {
    Node(SceneNodeId),
    PointLight(PointLightId),
    DirectionalLight(DirectionalLightId),
    SpotLight(SpotLightId),
}

impl ObjectHandle {
    pub fn kind(&self) -> ObjectKind {
        match self {
            Self::Node(_) => ObjectKind::Node,
            Self::PointLight(_) => ObjectKind::PointLight,
            Self::DirectionalLight(_) => ObjectKind::DirectionalLight,
            Self::SpotLight(_) => ObjectKind::SpotLight,
        }
    }
}
```

### Prepare/Create Node Plan

```rust
pub(crate) struct CreateNodePlan {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    pub(crate) node: SceneNode,
    pub(crate) record: ObjectRecord,
    pub(crate) parent: Option<SceneNodeId>,
    pub(crate) is_new_slot: bool,
}
```

### Prepare/Remove Node Plan

```rust
pub(crate) struct RemoveNodePlan {
    /// Post-order (children before parent) list of (id, payload, record).
    pub(crate) snapshots: Vec<SceneNodeRemovalSnapshot>,
    pub(crate) root_replaced: bool,
}

pub(crate) struct SceneNodeRemovalSnapshot {
    pub(crate) id: SceneNodeId,
    pub(crate) node: SceneNode,
    pub(crate) record: ObjectRecord,
    pub(crate) parent: Option<SceneNodeId>,
    pub(crate) parent_index: usize, // index into snapshots (or usize::MAX for root)
}
```

### Subtree Restoration Plan

```rust
pub(crate) struct RestoreSubtreePlan {
    /// (parent_index_in_plan, node, record) in pre-order.
    pub(crate) items: Vec<(Option<usize>, SceneNode, ObjectRecord)>,
    pub(crate) root_slot: Option<u32>,
}
```

Restoration is used by undo of `RemoveNodeCommand` — old `ObjectId` values are NOT reused; fresh IDs are minted with bumped generations, and the `ObjectMutationOutcome::remaps` records the old→new mapping.

### Outlier Cases

#### Light Detachment During Node Removal

When a node subtree is removed, any lights whose `light_group_parent` referred to a removed node are detached (regrouped to scene root) rather than silently deleted. The `RemovalSnapshotData::detached_lights` vector records these for restoration during node undo.

#### Root Replacement

When the scene root is removed, `RemoveNodePlan::root_replaced` is set to `true`. During removal cleanup, the root slot is vacated but the `SceneWorld::root` is set to `None` — the scene is left without a root node.

## 5. Lightning Round: Prepare/Commit State Machine

| Operation | Prepare | Commit |
|-----------|---------|--------|
| Create node | Validate parent, reserve slot, check generation, build node+record | Insert node, update parent's children, update reverse index, snapshot |
| Remove node | Collect subtree post-order, check root-replacement | Vacate slots, update reverse index, detach grouped lights, snapshot |
| Create light | Validate cap, reserve slot, check generation, build payload+record | Insert payload, update reverse index, snapshot |
| Remove light | Validate existence/generation | Vacate slot, update reverse index, snapshot |
| Duplicate objects | Validate all exist, reserve slots for each duplicate | Clone payloads, mint new ObjectIds+ObjectRecords, remap references, snapshot |
| Restore subtree | Validate parent exists, reserve slots | Insert restored nodes, update parent's children, update reverse index, remap/snapshot |

## 6. Best Practices

- Never construct `ObjectId` outside of `object_store.rs`. Use the `#[cfg(test)]` constructor for tests.
- Always call `audit_object_invariants()` in tests after complex mutation sequences.
- Keep `ObjectRecord` fields in sync with scene persistence serialization — adding a field to the record without updating the save/load path will cause silent data loss.
- Use `SceneObjectId` comparisons (NOT `ObjectId` comparisons) when checking identity across scene save/load cycles.
- The reverse index is not thread-safe and must only be accessed through `&mut SceneWorld`.

## 7. Gotchas & Failure Modes

- **Cross-scene ID forgery**: prevented by `SceneRuntimeId` comparison — two `ObjectId`s from different `SceneWorld` instances are never equal.
- **Persistent ID collision**: `mint_persistent_id()` uses 256 bits of randomness — collision is astronomically unlikely, but `audit_object_invariants` still checks for duplicates.
- **Stale ObjectId after restore**: undo of `RemoveNodeCommand` mints NEW `ObjectId` values — the old ones are permanently stale. UI must remap selection from `ObjectMutationOutcome::remaps`.
- **Vacant slot in free list after exhaustion**: handled by `checked_add` generation increment with error return.
- **Record-payload mismatch**: structurally impossible because `ObjectRecord` is co-located with the payload — there is no separate side-map to drift.

## 8. Debugging Playbook

- Step 1: Run `audit_object_invariants()` to get a precise error description.
- Step 2: Check `ObjectError` display messages — they include the `ObjectId` and context.
- Step 3: For command failures, inspect the `CommandError` message for the persistent ID and operation type.
- Step 4: For stale `ObjectId` errors, check the slot and generation against what `find_object_by_persistent_id` returns.
- Step 5: For cross-scene errors, verify the `SceneRuntimeId` provenance of both objects.

## 9. Cross-Module Links

- Object store implementation: `src/renderer/src/scene/object_store.rs`
- Object identity types: `src/renderer/src/object/identity.rs`
- Object commands: `src/renderer/src/scene/object_commands.rs`
- Component system: `src/renderer/src/object/component.rs`
- Scene world (payload storage): `src/renderer/src/scene/scene_world.rs`
- Event contracts: `src/events/src/lib.rs`
- Object system tests: `src/renderer/tests/object_identity.rs`, `src/renderer/tests/object_commands.rs`, `src/renderer/tests/object_api.rs`, `src/renderer/tests/object_queries.rs`

## 10. Standard References

- Slot+generation handle pattern (ABA prevention): common pattern in game engines and Vulkan resource management
- Two-phase commit (prepare/commit): https://en.wikipedia.org/wiki/Two-phase_commit_protocol

## 11. See Also

- [Editor Object System API](../api/18-editor-object-system.md) — public API surface
- [Scene Flattening and Culling](08-scene-flattening-and-culling.md) — how object transforms flow into draw submission
- [Event System and Lifecycle](10-event-system-and-lifecycle.md) — how lifecycle events are emitted from outcomes
