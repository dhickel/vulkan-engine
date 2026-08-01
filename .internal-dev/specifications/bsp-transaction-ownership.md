---
schema_version: 1
document_type: transaction-ownership-specification
status: active
owner: bsp-beta
created: 2026-07-23
approval: evidence re-audited 2026-07-26 — Phase 04 filesystem publication repaired; Phase 05 runtime mount ownership BLOCKED because Scene detachment has no renderer fence-aware queue acknowledgement and committed bridges lack active teardown receipts (GitHub #59, #60)
---

# BSP Transaction and Ownership Contract

## 1. Scope

This specification defines two distinct transaction domains, the resource ownership graph, generation-token guards, app bridge hooks, cache identity, unload/reload semantics, source-link persistence, and lock ordering constraints for BSP map data in the engine.

**Transaction Domain A — Runtime Mount Publication** (§4): The prepare/validate/commit/rollback lifecycle for mounting a parsed BSP into the live engine scene. This is a runtime, in-memory transaction.

**Transaction Domain B — Filesystem Artifact-Set Publication** (§4.6): The atomic publication of compiler outputs (`.bsp`, `.lit`, `.wad`, companion files, manifest) to a destination directory. This is a filesystem transaction with different ownership, edge cases, and evidence requirements.

Passing one transaction domain does not satisfy the other's evidence cell. Every resource owner is enumerated; the contract is the single source of truth for who creates, exposes, replaces, and destroys each resource.

## 2. Crate and Module Ownership

### 2.1 Crate Layout

| crate | location | role |
|-------|----------|------|
| `bsp` | `src/bsp/` | format parsing, BspWorld queries, neutral DTO extraction, diagnostics |
| `bsp_runtime` | `src/bsp_runtime/` | integration coordinator, prepare/commit/rollback, cache ownership, source-link lifecycle |
| `renderer` | `src/renderer/` (existing) | GPU resources, BSP materials, lightmap atlas, BSP pipeline variants, PVS-aware submission |
| `physics` | `src/physics/` (existing) | Rapier world, collider creation from app-provided recipes |
| `engine` | `src/` (existing, thin facade) | optional BSP entry-point re-exports |
| app (e.g., `dungeon_dogfood`) | `apps/dungeon_dogfood/` | app-owned physics bridge, structural behavior adapters |

### 2.2 Dependency Rules

| rule | constraint |
|------|-----------|
| `bsp` depends on | nothing engine-internal; only `glam` for DTO math types |
| `bsp_runtime` engine-internal dependencies | `bsp`, `renderer` (with `bsp` feature), `engine_events`. Does NOT depend on `physics`. |
| `bsp_runtime` external dependencies | `glam`, `log`, `serde`, `serde_json` for DTO math, diagnostics, and source-link JSON. |
| `renderer` optionally depends on | `bsp` (behind `renderer/bsp` feature). Does NOT depend on `bsp_runtime` or `physics`. |
| `physics` does NOT depend on | `bsp`, `bsp_runtime`, `renderer` |
| `engine` re-exports narrow BSP entry APIs from `bsp_runtime` | does not own BSP state |

## 3. Resource Ownership Enumeration

### 3.1 Who Creates/Exposes/Replaces/Destroys

| resource | creator | owner | exposer | replacer | destroyer |
|----------|---------|-------|---------|----------|-----------|
| `BspWorld` (parsed, validated, read-only) | `bsp::from_bytes()` | `bsp_runtime::BspCoordinator` | `bsp_runtime` (via DTO extraction) | `bsp_runtime` (reparse) | `bsp_runtime` (unload) |
| BSP entity descriptors (neutral DTOs) | `bsp::extract_entities()` | `bsp_runtime::BspCoordinator` | `bsp_runtime` | `bsp_runtime` | `bsp_runtime` |
| BSP mesh geometry (neutral DTOs) | `bsp::extract_geometry()` | `bsp_runtime::BspCoordinator` (staging) | N/A (consumed by renderer upload) | N/A | `bsp_runtime` (after upload) |
| scene nodes (`SceneNodeId`) | `bsp_runtime::BspCoordinator` via `Scene::add_mesh_with_bounds` etc. | renderer (`SceneWorld`) | `bsp_runtime` | `bsp_runtime` (reimport) | `bsp_runtime` (unload: scene node removal + retirement) |
| BSP materials (`MaterialHandle`) | renderer via `bsp_runtime` request | renderer (`MaterialCache`) | `bsp_runtime` | N/A (new material on reimport) | renderer (fence-aware retirement) |
| lightmap atlas textures | renderer via `bsp_runtime` upload | renderer (`TextureCache`) | `bsp_runtime` | N/A (new atlas on reimport) | renderer (fence-aware retirement) |
| scene lights (`PointLightId` etc.) | `bsp_runtime::BspCoordinator` via `Scene::create_point_light` | renderer (`SceneWorld`) | `bsp_runtime` | `bsp_runtime` (reimport) | `bsp_runtime` (unload: light removal) |
| world static collider | app bridge via `PhysicsWorld` | app bridge / `PhysicsWorld` | app bridge | app bridge (reimport) | app bridge (unload) |
| entity convex colliders | app bridge via `PhysicsWorld` | app bridge / `PhysicsWorld` | app bridge | app bridge (reimport) | app bridge (unload) |
| behavior state (door/button/platform) | app bridge | app bridge | app bridge | app bridge | app bridge |
| source-link metadata | `bsp_runtime::BspCoordinator` | `bsp_runtime::BspCoordinator` | `bsp_runtime` | `bsp_runtime` | `bsp_runtime` |
| PVS/leaf data (GPU-side) | renderer via `bsp_runtime` upload | renderer | `bsp_runtime` | `bsp_runtime` | renderer (fence-aware) |
| package metadata | `engine_pack` / `AssetRegistry` | `AssetRegistry` | `AssetRegistry` | `AssetRegistry` | `AssetRegistry` |

### 3.2 Ownership Boundaries

- **`bsp` crate**: Owns format knowledge and read-only `BspWorld`. Never owns GPU resources, scene nodes, physics objects, or behavioral state. DTOs are neutral: no Vulkan, Rapier, or engine handles.
- **`bsp_runtime` crate**: Owns the integration transaction, the active `BspCoordinator`, and the source-link lifecycle. Coordinates resources owned by renderer and app bridge.
- **Renderer**: Owns GPU resources. Provides upload APIs consumed by `bsp_runtime`. Never creates physics objects.
- **App bridge**: Owns physics objects and behavioral state. Receives neutral DTOs and query data from `bsp_runtime`. Never reaches into renderer internals.

## 4. Two-Step Transaction

### 4.1 Prepare → Validate → Commit → Rollback

```
           ┌──────────────────┐
           │   prepare(map)   │
           │  (hidden state)  │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │  validate(gen)   │
           │  (all-or-nothing)│
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐     ┌──────────────┐
           │   commit(gen)    │────▶│  published   │
           │  (non-fallible)  │     │  scene state  │
           └────────┬─────────┘     └──────────────┘
                    │ (on failure)
           ┌────────▼─────────┐
           │   rollback()     │
           │  (idempotent)    │
           └──────────────────┘
```

### 4.2 Prepare Phase

1. Parse BSP bytes into validated `BspWorld`.
2. Resolve companion files (.lit, palette, WADs, replacement textures).
3. Extract neutral DTOs: geometry, entities, lights, collider recipes.
4. **App bridge prepare hook**: app receives entity/collider/light DTOs and creates Rapier bodies/colliders (not yet added to simulation).
5. Stage renderer resources: materials, lightmap atlas, meshes (not yet added to scene).
6. Resolve optional-resource fallbacks.
7. Record all staged resources in the coordinator.

### 4.3 Validate Phase

1. Check generation token: must match the coordinator's current `load_generation`.
2. Verify every required resource is ready.
3. **App bridge validate hook**: app confirms physics objects created and valid.
4. Preflight all fallible scene-publication checks against the target scene before commit, including BSP point-light capacity, finite-value payload readiness, and storage reservation needed for non-growing light publication.
5. All-or-nothing: any failure → rollback.

### 4.4 Commit Phase

1. **Must be non-fallible**: after validation passes, publication must succeed.
2. Commit must be publication-only: no parsing, package resolution, external asset loading, GPU/app allocation, upload, lookup, serialization, bridge validation, restored-state validation, or app-world capacity reservation.
3. Atomically swap active scene nodes, lights, materials into the scene.
4. **App bridge commit hook**: app publishes physics objects (adds to `PhysicsWorld`).
5. Publish pre-serialized source-link metadata.
6. Update coordinator's `active_generation`.

### 4.5 Rollback

1. **Idempotent**: can be called multiple times; subsequent calls are no-ops.
2. Remove staged scene nodes (not yet visible; should be none).
3. Release staged renderer resources (fence-aware retirement for any GPU payloads).
4. **App bridge rollback hook**: app removes created physics objects.
5. Clear cached insertions.
6. Return coordinator to pre-prepare state.

### 4.6 Filesystem Artifact-Set Publication Transaction (Domain B)

This section defines the atomic publication of compiler outputs to a destination directory. This is a **separate transaction domain** from runtime mount publication (§4.1–§4.5). Evidence for one domain does not satisfy the other.

#### 4.6.1 Publication Model

```
  ┌──────────────────────────┐
  │   compile(map, profile)  │
  │   → staging directory    │
  └───────────┬──────────────┘
              │
  ┌───────────▼──────────────┐
  │   validate_artifact_set  │
  │   (staging dir)          │
  └───────────┬──────────────┘
              │
  ┌───────────▼──────────────┐     ┌──────────────────┐
  │   publish(staging→dest)  │────▶│  published       │
  │   (atomic directory mv)  │     │  artifact set    │
  └───────────┬──────────────┘     └──────────────────┘
              │ (on failure)
  ┌───────────▼──────────────┐
  │   abort(staging)         │
  │   (cleanup staging dir)  │
  └──────────────────────────┘
```

#### 4.6.2 Compile Phase

1. Invoke pinned ericw-tools `qbsp`, `vis`, `light` in a temporary staging directory.
2. Compiler identity and version MUST match the expected pinned values (SHA-256 verified).
3. Staging directory is a system temp directory with a random suffix; no pre-existing content.
4. Compiler stdout/stderr captured for diagnostics.
5. Compiler exit code checked; nonzero → stage failure.

#### 4.6.3 Validate Phase

1. Verify every required output artifact exists in staging:
   - `.bsp` (required; missing → stage failure)
   - `.lit` (optional; missing → diagnosed, not a failure)
   - `.prt` (optional; missing → diagnosed, not a failure)
   - `.pts` (optional; missing → diagnosed, not a failure)
2. Verify `.bsp` magic matches the requested profile (BSP29 or BSP2).
3. Verify `.bsp` passes structural validation (no truncation, overlap, cycle).
4. Verify `.lit` content (if present): valid QLIT header, payload length matches lightmap luxel count.
5. Verify companion file hashes and record them.
6. Verify the destination directory does not contain a conflicting artifact set (see edge cases below).
7. All-or-nothing: any validation failure → abort staging, no partial publication.

#### 4.6.4 Publish Phase

1. Atomically move or copy the validated artifact set from staging to the destination directory.
2. If the destination directory already contains a valid, complete artifact set from a prior publish:
   - **Pre-existing valid destination**: compare content hashes. If identical → no-op success (idempotent). If different → `LateCollision` diagnostic; publication blocked.
3. Manifest closure: write or update `fixture-manifest.toml` (or equivalent) with compiler provenance, source hashes, and output hashes. An ownership marker may identify an in-progress staging directory, but it is removed before final closure validation and atomic rename; it is never a published artifact.
4. Publication is non-fallible after validation passes (no compilation, no external I/O beyond the atomic move/copy).

#### 4.6.5 Abort Phase

1. Remove staging directory and all contents.
2. Idempotent: may be called multiple times; subsequent calls are no-ops.
3. Destination directory is never modified during abort.

#### 4.6.6 Edge Cases

| edge case | behavior | diagnostic |
|-----------|----------|------------|
| Compiler not found or wrong version | stage failure before compile | `CompilerIdentityMismatch` |
| Compiler nonzero exit | stage failure | `CompilerExecutionFailed(code, stderr)` |
| Compiler produces no `.bsp` | validation failure | `MissingRequiredOutput("bsp")` |
| Compiler produces malformed `.bsp` | validation failure | `BSP-STRUCT-CORRUPT-*` (see bsp-compatibility.md §7) |
| Compiler produces `.lit` with invalid header or wrong payload length | validation failure | `CompanionContentMismatch` |
| Compiler produces `.lit` with zero payload (QLIT header only) | validation passes; `.lit` recorded as empty companion | `EmptyCompanionWarning` |
| Interrupted staging (crash/power loss mid-compile) | staging directory orphaned; next publish cleans up via temp dir suffix or TTL | `OrphanedStagingDetected` |
| Late collision: destination already has a different artifact set | publication blocked | `LateCollision(existing_hash, new_hash)` |
| Pre-existing valid destination (same content hash) | no-op success (idempotent) | none |
| Destination directory is read-only or out of space | publication failure after validation | `PublicationIOError` |
| Partial prior publication (destination has `.bsp` but no manifest) | treated as pre-existing content; collision check runs | `IncompleteDestination` (warning) |
| Manifest closure fails (disk full, permission) | publication failure; staging preserved for retry | `ManifestWriteError` |

#### 4.6.7 Ownership

| resource | owner |
|----------|-------|
| staging directory | `engine_pack` / compiler driver |
| published artifact set | `AssetRegistry` / package system |
| `fixture-manifest.toml` | `engine_pack` (writes), `AssetRegistry` (reads) |
| compiler provenance record | `engine_pack` (writes), `bsp_runtime` (validates on load) |

#### 4.6.8 Relationship to Runtime Mount Transaction

- The filesystem publication transaction produces artifacts that the runtime mount transaction consumes.
- A successful publication does NOT imply a successful mount (e.g., published `.bsp` may have valid structure but fail renderer preflight due to missing GPU resources).
- A successful mount does NOT imply a successful publication (e.g., a hand-crafted `.bsp` may load fine but was never produced by the publication pipeline).
- Evidence cells for the two domains are separate. The evidence matrix (`evidence-matrix.md`) carries rows for both.

## 5. Generation Tokens and Cancellation

### 5.1 Generation Model

| property | value |
|----------|-------|
| token type | `u64` monotonically increasing `load_generation` |
| increment on | each `prepare()` call |
| check on | `validate()` and `commit()` |
| stale rejection | if the generation at validation/commit time ≠ the prepare generation, the operation is rejected |
| cancellation | incrementing the generation cancels any in-flight prepare for the previous generation |

### 5.2 Cancellation and Stale Completion

- A cancelled prepare (by newer `prepare()` call) discards its staged resources on completion.
- CPU-side staged allocations are freed immediately.
- GPU-used payloads retire through fence-observed retirement.
- `commit()` for a stale generation returns `StaleGeneration` error.
- Concurrent `prepare()` calls are serialized by the coordinator.

## 6. App Bridge Hooks

### 6.1 Hook Interface

```rust
// Conceptual contract — actual API may differ in shape
trait BspAppBridge {
    /// Called during prepare. App creates Rapier bodies/colliders from provided DTOs.
    /// Returns a token that identifies the prepared physics state.
    fn prepare_physics(
        &mut self,
        world_collider: &BspColliderRecipe,
        entity_colliders: &[(EntityIndex, BspColliderRecipe)],
    ) -> Result<PhysicsPrepareToken, AppBridgeError>;

    /// Called during validate. App confirms physics objects are valid.
    fn validate_physics(
        &self,
        token: &PhysicsPrepareToken,
    ) -> Result<(), AppBridgeError>;

    /// Called during commit. App publishes physics objects to the simulation.
    fn commit_physics(
        &mut self,
        token: PhysicsPrepareToken,
        physics_world: &mut PhysicsWorld,
    ) -> Result<(), AppBridgeError>;

    /// Called during rollback. App removes any created physics objects.
    /// Idempotent.
    fn rollback_physics(
        &mut self,
        token: PhysicsPrepareToken,
    );
}
```

### 6.2 Bridge Failure Handling

| failure point | coordinator behavior |
|--------------|---------------------|
| `prepare_physics` fails | rollback full prepare; `PrepareFailed(AppBridgeFailed)` |
| `validate_physics` fails | rollback; `ValidateFailed(AppBridgeFailed)` |
| `commit_physics` panics | coordinator poisoned; `CommitPanic` terminal state |
| `rollback_physics` panics | coordinator poisoned; `RollbackPanic` terminal state |

### 6.3 Poisoning Behavior

If any bridge hook panics during commit or rollback, the coordinator enters a poisoned state:
- No further map loads are accepted.
- The current active state (if any) remains available.
- Recovery requires coordinator recreation.
- `BspCoordinator::is_poisoned() -> bool` exposed.

## 7. Cache Identity

### 7.1 Cache Key Components

The BSP cache identity includes every setting that changes extracted output:

```
cache_identity = SHA-256(
    bsp_content_hash ||
    dialect_profile_tag ||
    bsp_scale ||
    palette_content_hash ||
    companion_identities ||
    texture_resolution_roots ||
    replacement_mappings ||
    light_calibration ||
    atlas_policy ||
    collision_policy
)
```

### 7.2 Component Details

| component | source |
|-----------|--------|
| `bsp_content_hash` | SHA-256 of raw .bsp bytes |
| `dialect_profile_tag` | `"q1-portable-ericw"` or exact variant |
| `bsp_scale` | resolved scale as canonical f32 bytes |
| `palette_content_hash` | SHA-256 of palette bytes |
| `companion_identities` | sorted list of `(companion_kind, content_hash)` |
| `texture_resolution_roots` | sorted canonical paths |
| `replacement_mappings` | sorted `(texture_name, resolved_path_hash)` |
| `light_calibration` | calibration parameters as canonical bytes |
| `atlas_policy` | atlas page size, padding, style count |
| `collision_policy` | hull indices, convex decomposition limits |

## 8. Unload / Reload / Reimport

### 8.1 Unload

| step | action |
|------|--------|
| 1 | increment `load_generation` (cancels any in-flight prepare) |
| 2 | remove BSP scene nodes from scene |
| 3 | queue scene node payloads for fence-aware retirement |
| 4 | remove BSP lights from scene |
| 5 | app bridge: remove physics objects |
| 6 | app bridge: reset behavior state |
| 7 | renderer: mark BSP materials/textures for retirement |
| 8 | clear `BspWorld` and coordinator state |

### 8.2 Reload (same source)

| step | action |
|------|--------|
| 1 | full unload |
| 2 | prepare/validate/commit with same source |

### 8.3 Reimport (different source, same logical map)

| step | action |
|------|--------|
| 1 | prepare new map (hidden) |
| 2 | compute source-link reconciliation (see §9) |
| 3 | on commit: atomic swap old → new |
| 4 | old resources retire through fence-aware queue |
| 5 | report reconciliation events |

## 9. Source-Link Persistence

### 9.1 Stored Metadata

| field | description |
|-------|-------------|
| `bsp_asset_id` | durable asset ID from package |
| `bsp_content_hash` | SHA-256 of last loaded .bsp |
| `compiler_provenance` | compiler identity, version, arguments |
| `companion_hashes` | hashes of loaded companions |
| `import_settings` | scale, palette, roots, calibration |
| `entity_identity_map` | UUID → stable entity handle mapping |
| `override_layer` | app-applied overrides (light colors, model assignments, etc.) |

### 9.2 Scene Persistence

Scene files store a **BSP source reference**, not an expanded copy:

```json
{
  "bsp_source": {
    "asset_id": "maps/e1m1",
    "content_hash": "sha256:abcd1234...",
    "compiler_provenance": { ... },
    "import_settings": { ... }
  },
  "bsp_overrides": {
    "entity_overrides": [ ... ],
    "light_overrides": [ ... ]
  }
}
```

The scene file does NOT serialize generated BSP world nodes as an editable copy. On load, the scene re-imports the BSP and applies overrides.

### 9.3 Override Reconciliation on Reimport

When reimporting with a different source hash:

| condition | behavior |
|-----------|----------|
| UUID-matched entity exists in new BSP | override applied to matched entity |
| UUID-matched entity deleted | override orphaned, reported |
| new entity with existing UUID | override ambiguous, reported |
| structural change (classname/origin changed) | override cleared, reported |
| no UUID, fingerprint match found | override applied to matched entity |
| no UUID, no fingerprint match | override orphaned |

## 10. Lock Ordering Constraints

### 10.1 Lock Hierarchy

```
1. BspCoordinator::load_lock (prepare serialization)
2. AssetManager::cache_lock (texture/material/mesh cache)
3. SceneWorld::scene_lock (scene graph mutation)
4. PhysicsWorld::simulation_lock (physics step synchronization)
```

- Locks acquired in strictly increasing order.
- No lock may be held while waiting for another lock at the same or lower level.
- Prepare acquires locks in order during validation, then holds during commit.
- GPU fences are external to this ordering (observed without holding engine locks).
- Renderer BSP retirement acquires internal stores in `mesh_cache → texture_cache → bsp_surface_cache → VMA allocator` order. Texture retirement under that sequence must use the existing allocator guard; it must not re-lock the non-reentrant allocator mutex.

### 10.2 Transaction Constraints

| constraint | rule |
|-----------|------|
| prepare duration | no lock held across I/O or compilation |
| commit duration | scene_lock held; minimal work within lock |
| reentrancy | `prepare()` cannot be called from within a bridge hook |
| bridge ordering | app bridge hooks are called with no engine lock held |

## 11. Versioned Persistence Schema (Frozen)

### 11.1 Source-Reference Based Persistence

Scene files store a BSP **source reference**, not an expanded copy of generated world nodes:

```json
{
  "bsp_source": {
    "asset_id": "maps/e1m1",
    "content_hash": "sha256:abcd1234...",
    "compiler_provenance": {
      "compiler": "ericw-tools",
      "version": "2.0.0-alpha3",
      "qbsp_hash": "...",
      "vis_hash": "...",
      "light_hash": "...",
      "qbsp_args": [...],
      "vis_args": [...],
      "light_args": [...]
    },
    "companion_hashes": {
      "palette": "sha256:...",
      "lit": null
    },
    "import_settings": {
      "bsp_scale": 0.0254,
      "texture_roots": [...],
      "wad_roots": [...]
    },
    "entity_identity_map": {}
  },
  "bsp_overrides": {
    "entity_overrides": [...],
    "light_overrides": [...]
  }
}
```

### 11.2 Identity and Uniqueness

| field | behavior |
|-------|----------|
| UUID-backed identity | **Not viable** — ericw-tools qbsp strips `_tb_id` from compiled entity lumps. No UUID survives compilation. |
| Normalized semantic fingerprint | `(classname, origin, targetname, target)` key set. Used as primary identity for all entities. |
| Duplicate ordinal | Entities with identical fingerprints are assigned ordinal 0, 1, 2, ... within fingerprint group. Source-link stable handles include the fingerprint plus ordinal so identical fingerprints remain distinct. Ordinal is stable across reloads when fingerprint match is preserved. |
| Ambiguity | Multiple current entities matching one stored override handle → `IdentityAmbiguous` diagnostic and restore cancellation before commit. |
| Mismatch | New entity with no fingerprint match → `IdentityInserted`. Old entity with no new match → `IdentityOrphaned`. |
| Migration | Structural change (classname or origin changed) → override cleared with `IdentityStructureChanged` diagnostic. |
| Rejection | Source hash mismatch without explicit migration approval → `SourceMismatch` diagnostic. App may override. |

### 11.3 Banned Fields in Persistence

The following must NEVER appear in serialized persistence:

| banned field | reason |
|-------------|--------|
| GPU handles (VkImage, VkBuffer, VkImageView, VkSampler, VkDescriptorSet, VkPipeline, VkPipelineLayout) | non-serializable, host-local-only |
| Descriptor pool allocations | non-serializable, pool-owned |
| GPU cache slot indices | non-serializable, runtime-only |
| Transient generation handles (scene node IDs, material handles, mesh handles, light IDs) | runtime-only, regenerated on reload |
| Expanded generated geometry (face vertices, lightmap atlas pixels, collider triangulations) | derived from source; must be regenerated |

### 11.4 Mutable Behavior Fields (Phase 08)

The following field classes in `bsp_overrides` are mutable across reloads:

| field class | stored? | behavior |
|------------|---------|----------|
| Entity light color overrides | yes | reapplied to matched entity on reload |
| Entity model assignment overrides | yes | reapplied to matched entity on reload |
| Door/platform/button pose+state | yes (Phase 08) | persisted in `mutable_behavior.doors`/`platforms`/`buttons`; restored on load |
| Trigger/target activation state | yes (Phase 08) | persisted in `mutable_behavior.triggers`; restored on load |
| Light-style intensity table | yes (Phase 08) | persisted in `mutable_behavior.light_styles`; restored on load |
| Timers/counters | yes (Phase 08) | persisted in `mutable_behavior.timers`; restored on load |
| External model override identities | yes (Phase 08) | persisted in `mutable_behavior.external_model_overrides`; restored on load |
| Physics body transforms | no | regenerated from collider recipes on reload |
| PVS state | no | regenerated from VIS data on reload |

### 11.5 Source-Link Restore Validation

On scene load with BSP source reference:

1. Verify `content_hash` matches the loaded .bsp bytes.
2. Build/upload renderer readiness while hidden; no publication occurs.
3. Reconcile entity and light overrides against UUID or fingerprint+ordinal handles.
4. Validate companion hashes and model-mapping identity against the current candidate.
5. Validate mutable behavior payloads (finite poses/timers, legal phases, valid light-style range, valid external model paths).
6. Run scene publication preflight, then commit.
7. On any failure before commit, roll back the hidden candidate and preserve the active scene/source-link payload.
8. **Do not** attempt to restore transient handles from the stored reference.

## 12. Approval and Evidence Matrix

| contract | status | evidence basis | blocker | reviewer |
|----------|--------|---------------|---------|----------|
| Crate dependency graph (§2) | PASS (Phase 09) | Cross-checked against workspace Cargo.toml files: `bsp`→`glam` only; `bsp_runtime`→`bsp`+`renderer`(`bsp` feature)+`engine_events` plus external `glam`/`log`/`serde`/`serde_json`; `renderer` optionally depends on `bsp` behind `renderer/bsp`; no `bsp_runtime`→`physics`, `renderer`→`bsp_runtime`, or `renderer`→`physics` dependency. | none | dhickel (2026-07-23) |
| Resource ownership enumeration (§3) | PARTIAL (Phase A EnhancedV3, 2025-07-24) | `PreparedBspMount` is move-only and Scene detachment is wired. Every `DetachedBspMount` is now queued in `BspCoordinator::pending_retirements` and never silently dropped. The coordinator exposes `drain_pending_retirements()` / `requeue_retirement()` for explicit caller handoff to `Renderer::retire_bsp_mount`. Renderer fence-aware queue acceptance itself is still a caller responsibility; the coordinator does not own the GPU retirement queue. | GitHub #59 (coordinator handoff resolved); renderer retirement queue ownership remains external | dhickel (2025-07-24) |
| Prepare/validate/commit/rollback flow (§4) | PARTIAL (Phase A EnhancedV3, 2025-07-24) | Move-only candidate/Scene transfer, stale-completion detachment, and pre-retirement candidate work are repaired and tested. Rollback, unload, replacement, stale upload, and teardown now deposit every `DetachedBspMount` into the coordinator's pending queue rather than dropping it. Committed bridge tokens are still consumed without an active teardown receipt. | GitHub #60 (bridge active-lifecycle boundaries still required); coordinator detachment handoff resolved | dhickel (2025-07-24) |
| Generation token model (§5) | PASS (Phase 05) | `u64` monotonic counter; incremented on each `prepare()`; checked on `validate()` and `commit()`; `StaleGeneration` error on mismatch; newer prepare invalidates previous candidate | none | dhickel (2026-07-23) |
| App bridge hooks (§6) | PASS (Phase 05) | `AppBridge` trait with `prepare`/`validate`/`commit`/`rollback` methods; `BridgeAggregator` manages multiple bridges; bridge commit panics → coordinator poisoned | none | dhickel (2026-07-23) |
| Poisoning behavior (§6.3) | PASS (Phase 05) | `BspCoordinator::is_poisoned()` gates `prepare`/`validate`/`commit`/`rollback`; `teardown()` bypasses poison check (terminal cleanup). 10 lifecycle fault tests pass. | none | dhickel (2026-07-23) |
| Cache identity components (§7) | PASS (Phase 07) | `CacheIdentity` computes SHA-256 from 10 components; stored in source-link; verified on restore | none | dhickel (2026-07-23) |
| Unload/reload/reimport semantics (§8) | PARTIAL (Phase A EnhancedV3, 2025-07-24) | Scene/source-link references are detached and stale candidates do not mutate the active mount. Detached mounts are queued for explicit handoff rather than silently dropped. No generic active bridge teardown runs on unload/replacement. | GitHub #60 (bridge lifecycle revisions still required); coordinator detachment handoff resolved | dhickel (2025-07-24) |
| Source-link persistence (§9, §11 frozen) | PASS (Phase 08) | Source reference not expanded copy; fingerprint+ordinal identity; `BspPersistenceEnvelope` schema V1; 21 persistence tests + 9 app persistence tests pass | none | dhickel (2026-07-23) |
| Versioned persistence schema (§11 frozen) | PASS (Phase 08) | Schema version enum with `from_u32`; banned fields enforced; `MutableBehaviorState` stores door/button/platform/trigger state, light styles, timers, external model overrides | none | dhickel (2026-07-23) |
| Lock ordering constraints (§10) | PASS (Phase 05) | Prepare holds no lock across I/O; commit holds `SceneWorld` lock briefly for publication only; no reentrancy from bridge hooks | deadlock analysis pending | dhickel (2026-07-23) |
| **Filesystem publication — compiler identity verification** | PASS (Phase 02 repair, 2026-07-24) | `engine_pack compile-bsp` verifies executable SHA-256 before invocation; pinned hashes match ericw-tools 2.0.0-alpha3 | `engine_pack` tests pass SHA-256 verification | dhickel (2026-07-24) |
| **Filesystem publication — stage/validate/publish/abort flow** | PASS (Phase 02 repair, 2026-07-24) | `engine_pack compile-bsp` staging→validate→publish→abort pipeline exercised for `dungeon_evidence_standard.map`; duplicate compile runs produce identical output | `engine_pack` tests pass full pipeline | dhickel (2026-07-24) |
| **Filesystem publication — malformed output handling** | NOT-RUN (blocking, Phase 01) | requires adversarial compiler output simulation (truncated `.bsp`, invalid `.lit`, missing required artifacts) | **Phase 01 resolution**: blocks_generator=true retained. Publication integrity is a required product gate — the end-to-end integrity chain from generator→compiler→publication must be proven. Adversarial publication suite not yet authored. | — |
| **Filesystem publication — missing `.lit` handling** | PASS (Phase 02 repair, 2026-07-24) | empty-lit fixture validates parser-side empty-companion handling; `engine_pack compile-bsp` handles missing `.lit` correctly | publication proceeds without `.lit`; monochrome lighting fallback | dhickel (2026-07-24) |
| **Filesystem publication — interrupted staging recovery** | NOT-RUN (blocking, Phase 01) | requires staged crash/power-loss simulation and orphaned-directory cleanup | **Phase 01 resolution**: blocks_generator=true retained. Publication integrity is a required product gate — the end-to-end pipeline must survive interrupted staging without corrupting the destination. Publication robustness suite not yet authored. | — |
| **Filesystem publication — late collision detection** | NOT-RUN (blocking, Phase 01) | requires two independent publishes to the same destination with different content | **Phase 01 resolution**: blocks_generator=true retained. Publication integrity is a required product gate — concurrent or conflicting publishes must be detected. Evidence campaign not yet executed. | — |
| **Filesystem publication — pre-existing valid destination (idempotent)** | PASS (Phase 02 repair, 2026-07-24) | duplicate `engine_pack compile-bsp` runs with identical inputs produce byte-identical outputs; second publish would be no-op | duplicate compile outputs matched checked-in fixture hashes | dhickel (2026-07-24) |
| **Filesystem publication — complete-directory publication + manifest closure** | PASS (revalidated 2026-07-26) | `engine_pack compile-bsp` publishes only manifest-declared payloads plus the canonical manifest; staging ownership metadata is removed before closure validation/rename, and the validator/publish primitive reject a surviving marker | focused `engine_pack` closure test and isolated nominal-M1 publication pass | dhickel (2026-07-26) |
| Versioned persistence envelope (§11.1) | PASS (Phase 08) | `BspPersistenceEnvelope` with `schema_version` field; only V1 approved; unknown versions rejected at deserialization boundary | none | dhickel (2026-07-23) |
| Canonical serialization and hashing (§11.2) | PASS (Phase 08) | `CanonicalFloat` for deterministic float encoding; sorted BTreeMap iteration; ordered Vec preservation; runtime handle rejection on deserialize | none | dhickel (2026-07-23) |
| Mutable behavior state persistence (§11.4) | PASS (Phase 08) | `MutableBehaviorState` stores door/button/platform pose+state, trigger activation, light-style table, timers/counters, external model overrides; no GPU/transient handles | none | dhickel (2026-07-23) |
| Restore validation and cancellation (§11.5) | PASS (Phase 08 validator repair) | `restore_from_persistence` validates schema→source hash→upload readiness→identity reconcile→companion/model-mapping identity→mutable behavior→scene preflight→commit; post-readiness failures roll back the hidden candidate and preserve active source link/scene payload | none | dhickel (2026-07-23) |
| Schema migration dispatch | PASS (Phase 08) | `SchemaVersion::approved_prior()` returns empty set (V1 is first version); `from_u32` rejects unknown versions; future migrations go through explicit migration functions | none | dhickel (2026-07-23) |
| Save/restore round-trip tests | PASS (Phase 08 validator repair) | 32 persistence tests across `bsp_runtime/tests/reload_and_persistence.rs` (23 total) and `apps/bsp_beta/tests/persistence.rs` (9 total): schema stability, content hash mismatch, companion/mapping mismatch, invalid behavior after upload-readiness rollback, GPU handle exclusion, bridge integration, mutable behavior round-trips | all passing (Phase 08) | dhickel (2026-07-23) |

**Re-review trigger**: Any change to persistence schema, dependency graph, resource ownership, or transaction flow requires owner re-review.

**Unresolved blocked cells**: None at specification level. All remaining cells are implementation pending, not decision blocked.

## 13. EnhancedV3 Transaction and Ownership

### 13.1 Scope

The EnhancedV3 production profile (`src/bsp_generator/src/enhanced_v3/`) adds
a third generation path to the `bsp_generator` crate. This section records the
transaction and ownership additions for the v3 profile without altering any
existing v1 or v2 ownership rule.

### 13.2 Crate and Module Ownership

| resource | location | role |
|----------|----------|------|
| v3 profile dispatch | `src/bsp_generator/src/enhanced/profile.rs` (`GenerationProfile::EnhancedV3`) | profile identity and tag dispatch |
| v3 generation engine | `src/bsp_generator/src/enhanced_v3/` | placement, topology, features, emission |
| v3 proof code (historical) | `src/bsp_generator/tests/enhanced_v3_proof/` | test-only evidence; not a production path |
| v1 generator | `src/bsp_generator/src/` | unchanged |
| v2 generator | `src/bsp_generator/src/enhanced/` | unchanged |

The `enhanced` (v2) and `enhanced_v3` modules are peers — neither depends on
the other. They share theme assets (`cc0_dungeon_v2`) and the compiler profile
(`ericw-q1-bsp2-generated`) but have independent RNG domains and generation
pipelines.

### 13.3 Dependency Rules

| rule | constraint |
|------|-----------|
| `enhanced_v3` depends on | shared `bsp_generator` infrastructure (config, error, geometry primitives, serialization); does NOT depend on `enhanced` (v2) module |
| `enhanced` (v2) depends on | shared `bsp_generator` infrastructure; does NOT depend on `enhanced_v3` |
| `bsp_generator` public exports | `GenerationProfile::EnhancedV3` variant; profile dispatch in `lib.rs` |
| `engine_pack` v3 packaging | `enhanced-dungeon-v3` command reuses existing BSP2 publication infrastructure from v2 path |
| theme assets | shared read-only access to `cc0_dungeon_v2/`; no v3-exclusive theme modifications |

### 13.4 Resource Ownership

The v3 profile creates the same resource types as the v2 profile (`.map` text,
metadata JSON, compiled `.bsp`/`.lit`, companion PNGs) through the same
owner-authorized `engine_pack` compilation and publication pipeline. No new
resource type, owner, or transaction domain is introduced.

| resource | creator | owner | notes |
|----------|---------|-------|-------|
| v3 `.map` text | `enhanced_v3::emit()` | `bsp_generator` (deterministic output) | canonical serialization identical to v1/v2 grammar |
| v3 metadata JSON | `enhanced_v3::metadata()` | `bsp_generator` | schema-v3 metadata with preset, capability, and budget fields |
| compiled v3 `.bsp`/`.lit` | `engine_pack compile-bsp` (pinned ericw-tools) | `engine_pack` publication | same `ericw-q1-bsp2-generated` profile as v2 |
| v3 companion PNGs | `engine_pack enhanced-dungeon-v3` | `engine_pack` publication | same companion completeness requirement as v2 |

### 13.5 Transaction Isolation

- v3 generation is a pure function: `(seed, preset, config) → (.map, metadata)`.
  It is stateless, deterministic, and requires no runtime transaction.
- v3 compilation and publication reuse the existing BSP2 compiler infrastructure
  and `engine_pack` atomic filesystem publication (§4.6).
- v3 BSP loading at runtime uses the existing `bsp_runtime` coordinator and
  two-step prepare/validate/commit transaction (§4). No new transaction domain
  is created.

### 13.6 Compatibility Freeze

The following are immutable through all v3 production phases:

| contract | freeze scope |
|----------|-------------|
| v1 12-entry corpus `.map`/metadata bytes | SHA-256 must match frozen baseline |
| v2 12-entry corpus `.map`/metadata bytes | SHA-256 must match frozen baseline |
| v1 RNG domain `"dungeon-gen/v1"` | separator and 4 stage tags frozen |
| v2 RNG domain `"dungeon-gen/v2"` | separator and 6 stage tags frozen |
| `cc0_dungeon_v2` theme assets | all SHA-256 hashes must match baseline |
| `cc0_stone_beta` theme assets | all SHA-256 hashes must match baseline |
| `GenerationProfile::LegacyV1` tag | `"legacy-v1"` |
| `GenerationProfile::EnhancedV2` tag | `"enhanced-v2"` |
| compiler profile | `ericw-q1-bsp2-generated` args frozen |

Any drift in v1 or v2 corpus output is a blocking regression and must be
escalated immediately.

## 14. Detached-Mount Retirement Handoff (Phase A)

### 14.1 Scope

This section defines the handoff contract for `DetachedBspMount` receipts
produced by the `bsp_runtime` coordinator. It is a subset of the larger
renderer fence-aware retirement problem (GitHub #59) and covers only the
coordinator-side queue and drain API.

### 14.2 Contract

1. Every `DetachedBspMount` produced by replacement, unload, rollback, stale
   upload, or teardown is deposited into the coordinator's pending-retirement
   queue exactly once.
2. No normal replacement, unload, or rollback path may silently drop a
   detached receipt.
3. The coordinator exposes an explicit drain/pop API so app code can retrieve
   pending mounts and submit them to `Renderer::retire_bsp_mount`.
4. A rejected retirement (via `BspRetirementRejection`) can be reconstructed
   into a `DetachedBspMount` via `BspRetirementRejection::into_detached()` and
   requeued via `BspCoordinator::requeue_retirement()`.

### 14.3 API Surface

| method | role |
|--------|------|
| `BspCoordinator::pending_retirement_count() -> usize` | current queue depth |
| `BspCoordinator::drain_pending_retirements() -> Vec<DetachedBspMount>` | take all queued mounts, leaving queue empty |
| `BspCoordinator::requeue_retirement(DetachedBspMount)` | return a previously drained receipt to the queue |
| `BspRetirementRejection::into_detached() -> DetachedBspMount` | reconstruct an intact mount from a rejection for retry |
| `Scene::clear_bsp_mount() -> Option<DetachedBspMount>` | direct scene clear returns its receipt rather than dropping it |

### 14.4 Diagnostics Compatibility

- `retired_mount_count()` / `retirement_diagnostics()` continue to report the
  cumulative detachment count for backward compatibility, but their semantics
  are now documented as detachment diagnostics rather than retirement evidence.
- `pending_retirement_count()` provides the live queue depth.

### 14.5 Evidence

- 9 focused handoff tests in `src/bsp_runtime/tests/retirement_handoff.rs`
- Focused transaction tests cover stale and duplicate completion queueing plus
  `PublishedQuarantined` and `CleanupBlocked` terminal teardown queueing
- Renderer tests cover rejection reconstruction and `Scene::clear_bsp_mount()`
  receipt return
- Existing transaction tests pass

### 14.6 Renderer Completion Contract

1. Normal frame-slot fence observation reaps accepted BSP retirement closures
   through the latest completed submission serial.
2. `VkRenderCore::drop` performs `device_wait_idle`; on success it terminally
   reaps pending BSP closures through the latest submitted serial before data
   cache and VMA destruction.
3. Device-loss teardown abandons Vulkan/VMA destruction and does not manufacture
   completion or invoke the terminal reap.
4. Queue removal is transactional with respect to lock acquisition: every cache
   and allocator lock needed for destruction is acquired before a closure leaves
   renderer ownership.
