# BSP Runtime and Lifetime — Internal Reference

> Architecture, ownership graph, protocol details, and failure modes for the BSP integration layers. For engine maintainers working inside `bsp_runtime`, `renderer` (BSP paths), and the `bsp` parser crate.

## 1. Purpose & Audience

This chapter is for contributors modifying or debugging the coordinator transaction, renderer BSP upload paths, descriptor ABI, or app bridge lifecycle. Assumes Rust proficiency and familiarity with the existing [Vulkan Frame Lifecycle](05-vulkan-sync-and-frame-lifecycle.md), [Descriptor ABI](14-renderer-descriptor-abi.md), and [Asset Lifecycle](03-asset-lifecycle-and-io.md).

## 2. Ownership Graph

```
┌─────────────────────────────────────────────────────────┐
│ App (bsp_beta / dungeon_dogfood / custom)               │
│  owns: event loop, input, camera, physics world,        │
│        behavior adapter, simulation tick, snapshot      │
└────────────┬────────────────────────────────────────────┘
             │ registers bridges, calls prepare/commit/unload
             ▼
┌─────────────────────────────────────────────────────────┐
│ bsp_runtime::BspCoordinator                             │
│  owns: active BspWorld (parsed), staged BspCandidate,   │
│        generation token, bridge registry, source-link   │
│  coordinates: renderer upload, app bridge lifecycle      │
│  does NOT own: GPU resources, physics objects, scene     │
└──────┬──────────────────────┬───────────────────────────┘
       │ DTO extraction       │ renderer mount upload
       ▼                      ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ bsp crate    │    │ renderer (bsp feature)               │
│  owns:       │    │  owns: GPU meshes, materials,        │
│   BspWorld   │    │        lightmap atlas textures,      │
│   (immutable)│    │        BSP pipeline variants,        │
│  produces:   │    │        BspMountState (PVS, leaves),  │
│   neutral    │    │        descriptor sets/layouts       │
│   DTOs       │    │  never owns: scene nodes, physics    │
└──────────────┘    └─────────────────────────────────────┘
```

### Key Boundaries

- **`bsp` crate**: Pure byte-level trust boundary. Depends only on `glam`. Produces immutable `BspWorld`. Zero GPU, Vulkan, physics, or app dependencies.
- **`bsp_runtime`**: Integration coordinator. Depends on `bsp`, `renderer` (with `bsp` feature), `engine_events`. Does NOT depend on `physics` or any app crate.
- **Renderer** (`bsp` feature): Owns GPU resources (meshes, materials, atlas, pipelines). Does NOT depend on `bsp_runtime` — one-way dependency.
- **App**: Owns physics objects and behavior state. Receives neutral DTOs from `bsp_runtime`. Never reaches into renderer internals.

## 3. Package Trust Boundary

The `package_io::PackageResolver` boundary used by BSP package loading enforces:

| check | mechanism |
|-------|-----------|
| Source identity | SHA-256 of raw bytes, compared against manifest |
| Budget enforcement | Aggregate allocation limit (256 MiB), per-category sub-limits |
| Path traversal | `..` detection with percent-encoding awareness |
| Symlink escape | Non-regular file rejection |
| Content hash verification | `ContentIdentity` vs manifest `expected_hashes` |

`package_io::AuthorizedBytes::new` is `pub(crate)` — external crates must go through `PackageResolver::resolve` before passing companion/source bytes into BSP parsing or extraction. `load_bsp_package` derives `<texture>_norm.png` and `<texture>_gloss.png` candidates from sanitized miptex identities, resolves them only under confined package roots, and stores the resulting `ConfinedResource`s on `LoadedBspPackage`. `prepare_from_loaded_package` converts those resources to neutral owned bytes at the coordinator boundary.

## 4. Neutral Extraction ABI

The `bsp::extract` module produces renderer-agnostic DTOs:

| DTO | contents | consumer |
|-----|----------|----------|
| `ExtractedBsp` | top-level container | `bsp_runtime` |
| `FaceGeometry` | indexed triangle geometry in engine space | `renderer::build_face_meshes` |
| `BspMaterial` | resolved texture + lightmap + surface class | `renderer::build_bsp_material_descs` |
| `ExtractedTexture::pbr_companions` | optional authorized normal/gloss PNG bytes and diagnostic logical paths | renderer upload preflight |
| `FaceLightmapLayout` | atlas offset, Quake-grid-snapped luxel extents, half-luxel-centered UVs, per-style layers | `renderer` atlas upload |
| `ExtractedVisibility` | VIS bytes, world model 0 `visleaf_count`, nodes/leaves/planes; PVS bit `i` maps raw leaf `i + 1` | renderer PVS submission and light selection |
| `LightDescriptor` | point light position, color, intensity | `renderer` light publication |
| `CollisionRecipe` | clipnode-derived hull data | app bridge → Rapier |
| `ColliderRecipe` | convex decomposition output | app bridge → Rapier |
| `BehaviorEntityRecipe` | door/button/platform/trigger params | app bridge → behavior adapter |

DTOs carry no Vulkan handles, Rapier types, or engine runtime handles.

## 5. Renderer Resource Graph

### GPU Resources Created for BSP

```
BspExtractionRequest
       │
       ▼
renderer::prepare_bsp_mount()
       │
       ├─► plan_bsp_upload()          → bounded merged batches (≤2,048), shared material plans
       │                              → validated/packed material data (R mask, GB normal, A gloss)
       ├─► merged batch upload        → MeshHandle[]         (VkMeshBuffers, VkSubAlloc)
       ├─► shared material register   → BspMaterialHandle[]   (descriptor sets 1+2)
       ├─► BspSurfaceCache            → VkImage (lightmap atlas, 4-layer array)
       │                              → VkImageView (array view)
       │                              → VkSampler (linear clamp)
       │                              → BspSurfaceUniform UBO per material
       │                              → BspFrameValuesUniform UBO (one per frame slot)
       ├─► BspMountState              → leaf→node map, PVS decompressed bytes
       └─► PreparedBspMount           → consumed by Scene::set_bsp_mount
```

### Retirement

All GPU resources follow the existing fence-aware retirement contract (`GpuRetirementQueue<T>`, keyed by `FrameSerial`):

- Mesh handles: generation-bump on unload, `VkMeshBuffers` + suballocations retained until `completed_serial >= retire_after`
- Material handles: descriptor sets returned to pool via `vkFreeDescriptorSets`, `meta_alloc` suballocation retired
- Lightmap atlas: `VkImage`/`VkImageView` destroyed after fence observation
- Surface UBOs: `VkSubAlloc` deallocation after referencing frames complete

Reserved default slots for BSP resources do not yet exist — all BSP resources are dynamically created.

## 6. Descriptor and Frame ABI

### Set 0 — `BspScene` (shared with PBR)

Identical six-binding layout as `SceneData`. BSP and PBR paths can share the same set 0 descriptor set at bind time:

| binding | type | content |
|---------|------|---------|
| 0 | UBO | `SceneDataUBO` (144 B) |
| 1 | UBO | `EnvironmentUBO` (2048 B) |
| 2 | samplerCube | irradiance |
| 3 | samplerCube | prefiltered env |
| 4 | sampler2D | BRDF LUT |
| 5 | sampler2DArrayShadow | CSM shadow array |

### Set 1 — `BspMaterial` (BSP-specific)

| binding | type | content |
|---------|------|---------|
| 0 | sampler2D | albedo texture (one array layer) |
| 1 | sampler2D | packed material data: R fullbright mask, G/B normal X/Y, A gloss |
| 2 | sampler2DArray | lightmap atlas (4 face-slot-local layers) |
| 3 | UBO | `BspSurfaceUniform` (80 B) |

### Set 2 — `BspFrameValues` (frame-varying)

| binding | type | content |
|---------|------|---------|
| 0 | UBO | `BspFrameValuesUniform` (288 B) |

### Frame-Varying Update Rule (frozen)

- In-flight descriptors are **never** mutated.
- Set 2 UBO is written once per frame max, after frame-slot fence wait.
- A mount starts static-only: style 0 intensity is 1.0 and styles 1–63 are 0.0 until an app snapshot activates them.
- Static textures (albedo, fullbright mask) use one array layer — animation frame changes are communicated via `animationFrame`/`animationTime` uniforms, not by rewriting texture bindings.

### BSP Fragment Color Path

- Lightmap images are `R8G8B8A8_UNORM`; fragment shaders decode sampled bytes with `pow(encoded, 2.2)` before applying the 2× overbright factor.
- Opaque, liquid, and sky output all use the shared renderer exposure/tone-map/gamma path.
- Fullbright masks multiply sampled albedo RGB. A white scalar is not a valid emissive color because it destroys palette-authored lava/trim hues.
- External PBR companions opt opaque/alpha-mask materials into `bsp_pbr.frag`: baked lightmaps remain diffuse irradiance, `roughness = 1 - gloss`, and set 0 prefiltered environment + BRDF LUT provide dielectric specular. Missing normal/gloss channels use flat/fully-rough defaults.
- `{...}` alpha-mask textures clear alpha and fullbright emission only where palette index 255 occurs; index 255 stays opaque on other texture classes.
- With no PBR companion, packed material-data bytes reproduce the prior fullbright RGBA upload and the surface stays on `bsp_lightmapped.frag`.

### Pipeline Variants

All seven BSP pipeline variants share one `vk::PipelineLayout`:

| variant | blend | depth write | cull | fragment shader |
|---------|-------|-------------|------|-----------------|
| `bsp_opaque` | off | write | back | `bsp_lightmapped.frag` |
| `bsp_fullbright` | off | write | back | `bsp_lightmapped.frag` (fullbright path) |
| `bsp_alpha_mask` | off (alpha test) | write | none (two-sided) | `bsp_lightmapped.frag` |
| `bsp_pbr_opaque` | off | write | back | `bsp_pbr.frag` |
| `bsp_pbr_alpha_mask` | off (alpha test) | write | none (two-sided) | `bsp_pbr.frag` |
| `bsp_sky` | off | no write | back | `bsp_sky.frag` |
| `bsp_liquid` | alpha blend | no write | none (two-sided) | `bsp_liquid.frag` |

## 7. Transfer Readiness

BSP mesh and atlas uploads use the renderer's transfer queue. The upload pipeline:

1. `plan_bsp_upload()` preflights bounded merged geometry, shared materials, textures, atlas image size, and compact staging demand before allocation
2. Merged batch meshes create `VkSubAlloc` entries for vertex/index data and shared material descriptors are registered
3. Sparse atlas rectangles are uploaded via compact staging buffer regions → `vkCmdCopyBufferToImage`
4. Transfer completions are observed through `VkFenceQueue`
5. `PreparedBspMount` is only returned after all uploads reach fence-completion

The coordinator must pump transfer submissions (done automatically by `Renderer::prepare_bsp_mount` via the asset upload manager).

## 8. Logical Commit Protocol

```
prepare(bytes, scale, source_id) / prepare_from_world(world, scale, source_id)
/ prepare_from_loaded_package(package, scale)
  │
  ├─► BspLoader::load() or caller-provided world → BspWorld (validated, immutable)
  ├─► extract(request)          → ExtractedBsp (neutral DTOs)
  ├─► build candidate           → BspCandidate (hidden, staging)
  ├─► run bridge prepare hooks  → physics objects created (not published)
  └─► increment generation      → previous prepare invalidated
       │
validate(token)
  │
  ├─► check token == generation
  ├─► run bridge validate hooks
  ├─► preflight scene capacity (light slots, etc.)
  └─► all-or-nothing: failure → rollback
       │
set_renderer_mount_ready(token, mount)
  │
  ├─► generation check
  ├─► GPU upload complete check
  └─► candidate.renderer_ready = true
       │
commit(token, scene)
  │
  ├─► check token == generation  (stale rejection)
  ├─► check candidate.validated  (must have passed validate)
  ├─► check renderer mount ready
  ├─► scene.set_bsp_mount(mount) (GPU resources published)
  ├─► scene.set_bsp_source_link(json) (persistence envelope)
  ├─► add lights to scene
  ├─► run bridge commit hooks    (physics published)
  ├─► swap active ← staged
  └─► consume candidate
```

### Commit Purity

`commit()` does NOT: parse, resolve packages, load assets, allocate GPU resources, upload, look up handles, serialize, validate bridges, validate restored-state, or reserve app-world capacity. All of those must complete before commit is called. This ensures commit is non-fallible after validation passes.

## 9. Generation Observation

- `BspGenerationToken` is a `u64` monotonic counter.
- Incremented on each `prepare()` call.
- Checked on `validate()` and `commit()` — mismatch produces `StaleGeneration`.
- Cancellation: newer `prepare()` invalidates previous candidate. Bridge tokens from superseded candidates are rolled back.
- `current_generation()` is observable for snapshot versioning.

## 10. Shared Cache

`CacheIdentity` produces a SHA-256 fingerprint from 10 components:

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

This identity is stored in the source-link and used to detect when cached GPU resources need invalidation. Normal and gloss companion bytes are independently hashed into sorted `CompanionId` entries, so adding, removing, or editing either map invalidates the candidate cache identity.

## 11. Fence Retirement

All BSP GPU resources follow `DECISION-20260725-15`:

- `GpuRetirementQueue<T>` with `FrameSerial` keys
- `retire_after = max(last_referenced_serial, latest_submitted_serial)`
- Completion advances only from successful fence observations
- Submit failure cannot fabricate completion
- Mesh/material/atlas retirement uses `RetirementClass` taxonomy

## 12. Immutable Snapshot

`bsp_runtime::SnapshotBuilder` produces a `BspSimulationSnapshot` with:

| field | source |
|-------|--------|
| `generation` | active BSP generation observed by the app |
| `epoch` | fixed-step tick, `dt`, and elapsed simulation time |
| `entity_poses` | app physics/behavior bridge → body transforms |
| `external_instances` | app model-mapping bridge |
| `light_styles` | behavior adapter → active style intensities |
| `liquid_time` | elapsed liquid animation time |
| `activations` | behavior adapter → trigger/door/button/platform state |
| `any_motion` / `any_style_change` | app-owned change flags |

The snapshot is read-only and frame-consistent. Apps consume it through their own adapters; the workspace reference is `apps/bsp_beta/src/scene_sync.rs`, which writes `Scene::set_bsp_frame_values()` and updates entity node transforms.

## 13. Persistence Restore

`BspCoordinator::restore_from_persistence()`:

1. Deserialize `BspPersistenceEnvelope` → validate schema version (only V1 approved)
2. Load source BSP bytes → verify `content_hash`
3. Prepare + extract + upload (hidden, no publication)
4. Validate `content_hash` match against envelope
5. Build upload readiness → hidden mount
6. Reconcile entity/light overrides against identity records
7. Validate companion hashes and model-mapping identity
8. Validate mutable behavior payloads
9. Preflight scene capacity → commit

Any failure before commit rolls back the hidden candidate and preserves the active scene/source-link payload. Post-readiness failures must leave the active scene unchanged.

## 14. Transparent Ordering

BSP draws participate in the existing transparent ordering:

1. Opaque (PBR + BSP opaque + BSP fullbright + BSP alpha-mask)
2. BSP sky (depth-test, no write → engine sky renders behind)
3. Transparent sort: non-BSP blended draws + BSP liquids together, back-to-front

BSP draws are recorded before the geometry dynamic-rendering scope ends. `record_bsp_draw_sequence_impl` binds set 0 on every pipeline-layout switch and set 1 on material change.

## 15. Device Loss / Shutdown

- `VkRenderCore` records device loss immediately. BSP GPU resources follow the same terminal path: skip Vulkan/VMA teardown.
- `BspCoordinator::teardown()` does NOT check poison state (terminal cleanup path).
- Bridge panics during commit or rollback poison the coordinator. Recovery requires coordinator recreation.

## 16. Failure Matrix

| failure point | behavior |
|---------------|----------|
| Invalid magic/version | `BSP-UNSUPPORTED-DIALECT` → whole-asset rejection |
| Lump truncation/overlap | `BSP-STRUCT-CORRUPT-LUMP` → whole-asset rejection |
| Corrupt entity string | `BSP-STRUCT-CORRUPT-ENTITY` → whole-asset rejection |
| Missing palette (strict) | `BSP-MISSING-REQUIRED-PALETTE` → whole-asset rejection |
| Missing palette (dev) | textured maps are rejected; no test/default palette substitution |
| Corrupt VIS | invalid/missing global VIS disables PVS; a corrupt per-leaf row falls back conservatively. Row width is model 0 `visleafs`, and raw leaf 0 is never interpreted as PVS bit 0 |
| Corrupt clipnodes | `BSP-STRUCT-CORRUPT-CLIP` → whole-asset rejection |
| Bridge prepare fails | coordinator rolls back, re-prepare needed |
| Bridge commit panics | coordinator poisoned |
| Stale generation at commit | `StaleGeneration` error |
| Renderer upload fails | mount not created, coordinator rollback |
| Malformed or dimension-mismatched PBR PNG | rejected during upload planning before GPU allocation |
| Content hash mismatch on restore | restore cancelled, active state preserved |
| Source hash mismatch on restore | `ContentHashMismatch` error |

## 17. Cross-Module Links

- Parser crate: `src/bsp/src/lib.rs`, `src/bsp/src/extract.rs`, `src/bsp/src/world.rs`
- Coordinator: `src/bsp_runtime/src/coordinator.rs`, `src/bsp_runtime/src/candidate.rs`
- Bridge: `src/bsp_runtime/src/bridge.rs`, `src/bsp_runtime/src/behavior.rs`
- Source-link: `src/bsp_runtime/src/source_link.rs`, `src/bsp_runtime/src/cache.rs`
- Renderer BSP: `src/renderer/src/api/bsp.rs`, `src/renderer/src/data/bsp_import.rs`, `src/renderer/src/data/bsp_material.rs`
- Vulkan BSP: `src/renderer/src/vulkan/vk_bsp.rs`
- BSP shaders: `src/renderer/src/shaders/bsp_lightmapped.*`, `src/renderer/src/shaders/bsp_pbr.*`, `src/renderer/src/shaders/bsp_sky.*`, `src/renderer/src/shaders/bsp_liquid.*`
- Scene BSP: `src/renderer/src/scene/bsp_visibility.rs`
- Descriptor ABI guards: `src/renderer/tests/descriptor_abi.rs`
- App entrypoint: `apps/bsp_beta/src/main.rs`

## 18. See Also

- [BSP Acceptance Spec](../../.internal-dev/specifications/bsp-acceptance.md)
- [BSP Compatibility Spec](../../.internal-dev/specifications/bsp-compatibility.md)
- [BSP Renderer-Lighting Spec](../../.internal-dev/specifications/bsp-renderer-lighting.md)
- [BSP Spatial-Physics Spec](../../.internal-dev/specifications/bsp-spatial-physics.md)
- [BSP Transaction-Ownership Spec](../../.internal-dev/specifications/bsp-transaction-ownership.md)
- [BSP Dungeon Generation Spec](../../.internal-dev/specifications/bsp-dungeon-generation.md)
- [Renderer Descriptor ABI](14-renderer-descriptor-abi.md)
- [Vulkan Sync and Frame Lifecycle](05-vulkan-sync-and-frame-lifecycle.md)
- [Asset Lifecycle and I/O](03-asset-lifecycle-and-io.md)
- [Guide: BSP Beta](../guide/18-bsp-beta.md)
- [API: BSP Beta](../api/17-bsp-beta.md)
