# Voxel Demo Configuration and Regeneration Internals

This document describes the app-local architecture in `apps/voxel_demo`. The implementation deliberately does not change renderer, Vulkan, shader, descriptor, point-light-limit, or procedural-upload contracts.

## Ownership Boundaries

| Boundary | Owner | May contain | Must not contain |
| --- | --- | --- | --- |
| Preset document | `config.rs` | Version tuple, generator/classifier/UV values, wall/floor material references and factors | Headless/capture/environment options, light budget, editor state, request IDs |
| Resolved configuration | `config.rs` + `main.rs`/`editor.rs` | Normalized document, source context, resolved references, semantic identities, runtime options | Renderer handles or scene nodes |
| CPU package | `scene_package.rs` | Owned wall/floor CPU meshes, light/viewpoint descriptors, identities, counts, timings | `Renderer`, `Scene`, GPU/resource handles |
| Worker state | `regeneration.rs` | One immutable request snapshot and one owned `CpuScenePackage` result | Renderer calls or scene mutation |
| Presentation/commit state | event-loop thread | Active node/mesh/material/light handles, material cache, retirement records | Worker-thread renderer access |
| Editor callback | `editor.rs` | Draft mutation and owned queued commands | File I/O, OS randomness, worker launch, callback registration, renderer mutation |

`CpuScenePackage` is the handoff that makes generation threadable without pretending that GPU upload is asynchronous. Renderer resources remain main-thread-only.

## Configuration Resolution

### Types

`PresetDocument` is a strict serde TOML type:

```text
PresetDocument
├── schema_version
├── generator_version
├── rng_version
├── GeneratorSection
│   ├── topology/shape: seed, resolution, shell, site/tunnel/radius fields
│   ├── maze: density, twistiness, radius, retries, global search budget
│   └── presentation-affecting scene config: floor threshold and UV scales
└── MaterialsSection
    ├── wall: albedo, normal, roughness, AO, base-color/roughness/metallic factors
    └── floor: same fields
```

`#[serde(deny_unknown_fields)]` applies at every document level. There are no serde defaults, so a document must be complete. `validate_preset_document` accepts only version tuples `1/1/1` and `2/2/2`, normalizes signed zero, rejects non-finite values, and applies checked topology/interior/capacity/work gates before generation.

`RuntimeOptions` is deliberately separate:

```rust
RuntimeOptions {
    light_budget, // 9..=16
    headless,
    capture_dir,
    env_path,
}
```

These values are never serialized into a preset or included in semantic identities.

### Resolution sequence

`main::run_v2` follows this order:

1. Select one complete base: embedded `--preset`, external `--config`, or embedded `default`.
2. Attach `DocumentSource` and a source directory.
3. Normalize and resolve the complete base before applying overrides.
4. Apply only CLI values represented by `Some`; omission does not overwrite a base value.
5. Normalize again, resolve all eight asset references, and collect validation errors.
6. Validate the runtime light budget separately.
7. Compute typed identities only after successful normalization/resolution/validation.
8. Dispatch `1/1/1` to the legacy path or build the v2 CPU package for `2/2/2`.

The editor repeats the same normalize/validate/resolve/identity boundary for its draft before it can queue save or regeneration.

### Asset path policy

- Catalog IDs are stable semantic names from `known_catalog_ids()`.
- A relative filesystem reference requires `non_portable=false` and resolves from the source document directory.
- An absolute filesystem reference requires `non_portable=true`.
- Paths are lexically normalized, must remain absolute after resolution, must be UTF-8, and must identify a regular file.
- Save retains catalog IDs but converts filesystem references to normalized absolute non-portable paths.
- Save reloads and validates the emitted document before replacing the editor draft.

The current material uploader maps the wall/floor albedo catalog IDs to KB3D directories and discovers the four texture suffixes there. Although the document resolver understands filesystem references for all roles, arbitrary filesystem-backed material upload is not implemented by `create_wall_floor_materials`.

## Identity System

`CanonicalHasher` is a domain-separated SHA-256 writer with field tags, big-endian integers, length-framed strings, and normalized finite `f32` bits.

| Identity | Includes | Excludes |
| --- | --- | --- |
| `GeometryIdentity` | Generator/RNG versions and geometry-affecting generator fields (seed through maze search budget) | Floor classifier, UV scales, materials, runtime state, source formatting |
| `SceneConfigIdentity` | Geometry digest, floor threshold, normalized resolved wall/floor references, scalar material factors, UV scales, fixed lighting-policy description | Raw TOML bytes, source label, editor state, light budget, headless/capture/env options, asset file bytes |
| `AssetDigest` metadata | SHA-256 content digest when populated by a caller | Semantic identity | 

A path's normalized resolved spelling participates in scene identity; equivalent lexical spellings converge before hashing. Asset content digests remain separate so content reproduction/cache invalidation is not confused with semantic config identity. `CpuScenePackage::asset_digests` currently starts empty and is available for caller-populated reproduction metadata.

## Generator v2

### Named RNG streams

`rng::named_stream(seed, rng_version, stage_tag)` hashes a length-framed domain, seed, version, and UTF-8 stage tag with SHA-256, then initializes a fresh `Pcg32V1` from the first 128 digest bits. There is no shared master-stream nonce in v2.

Consequences:

- a stage's random sequence depends only on `(seed, version, tag)`;
- adding or reordering an unrelated stage does not consume another stage's sequence;
- lists are canonically ordered before deterministic shuffle/output where order matters;
- changing a stage tag is an output-versioning event and can change fixtures.

The v1 `PhaseTaggedRng` remains separate and unchanged for legacy fixtures.

### Transactional generation pipeline

`generate_v2` clones the caller's world, runs all stages against the candidate, and replaces the caller world only after success:

```text
validate dimensions/config
  → derive one operation-aware InteriorRegion
  → place 5–12 sites (five stable core IDs)
  → build connected semantic spline tree + deterministic extras
  → carve caverns
  → carve spline tunnels and retain actual route center cells
  → plan all requested maze links and reserve full footprints
  → carve maze links only after the complete requested set succeeds
  → apply surface roughness inside the shared interior
  → enforce and verify every configured shell layer
  → verify reachability
  → derive/validate core viewpoints and light anchors
  → persist canonical site/route records and actual-route clearance
```

The core IDs are stable: spawn, junction, grand cavern, shaft, and destination. Auxiliary IDs are deterministic and remain separate from the core-role contract.

### Reserve before carve

Maze search uses deterministic six-neighbor A* through solid cells, with endpoint attachment air explicitly permitted. Planning is bounded by one global `maze_search_budget` and a retry count. A successful route's complete radius footprint is checked against prior reservations, with explicit endpoint-zone overlap rules. Accepted routes are sorted canonically and carved only after the target count has been planned.

This produces failure atomicity at two levels:

- maze exhaustion carves none of the tentative maze set;
- any typed generation error leaves the caller-owned `VoxelWorld` unchanged because only the candidate clone was mutated.

### Interior and shell

`InteriorRegion::from_operation_requirements` combines configured shell thickness, maximum cavern/tunnel/maze reach, roughness displacement, and extraction safety margin. Site centers and carve operations use its operation-center bounds. Generation then defensively rewrites and verifies all six shell faces for every layer in `shell_thickness`.

## MC33, Partitioning, and Materials

`build_scene_package` extracts MC33 geometry from the final density lattice, applies the app's current structural validation policy, and partitions the source mesh because the renderer binds one material per procedural mesh.

`partition_mesh`:

1. validates finite classifier/UV/transform options;
2. validates source indices and parallel attributes;
3. computes each triangle's consistently wound geometric normal after `object_to_world` (identity in the app);
4. classifies `world_normal.y >= floor_threshold` as floor and every other triangle as wall;
5. proves source triangle-count conservation;
6. compacts each bucket in source-triangle/first-reference order;
7. copies position, normal, tangent, and color attributes bit-exact and applies exactly one bucket UV multiplication;
8. represents an empty bucket as `None` so it is not uploaded.

The source and partitions currently use `MeshValidationPolicy::AllowOpenEdges`; finite data, index validity, normalized normals/tangents, and nondegenerate triangles remain checked. Do not describe this path as a runtime asymptotic-decider implementation or as strict closed-manifold enforcement.

The PBR mapping is intentionally app-local:

| KB3D file | Load space | Renderer slot/sample |
| --- | --- | --- |
| `_basecolor.png` | sRGB | `base_color_tex` |
| `_normal.png` | linear | `normal_tex` |
| separate `_roughness.png` | linear | `metallic_roughness_tex`, shader roughness from G (grayscale replication) |
| separate `_ao.png` | linear | `ao_tex`, shader AO from R |

Metallic is fixed to zero. ARM, metallic, and height files are excluded. KB3D redistribution remains unapproved; this is a local-development path.

## `CpuScenePackage`

`build_scene_package(&ResolvedAppConfig)` returns owned, renderer-free data:

```text
CpuScenePackage
├── Option<CpuMesh> wall_mesh
├── Option<CpuMesh> floor_mesh
├── Vec<CpuLightDescriptor> (five mandatory site + up to four midpoint lights)
├── Vec<CpuViewpoint> (five core roles)
├── GeometryIdentity / SceneConfigIdentity
├── asset digest map
└── triangle/voxel counts and generation/mesh/partition timings
```

The package is the worker result and the one-shot headless input. `CpuMesh` mirrors procedural-upload attributes without renderer handles. Conversion to `ProceduralVertex`, material creation, uploads, and scene attachment occur on the main thread.

## Regeneration State Machine

### State

`RegenerationState` owns:

- `active: Option<PresentedPackage>` — currently authoritative cave node/resources/lights/identity;
- `worker_handle` plus an immutable `active_worker_request` for panic attribution;
- `latest_request: Option<RegenRequest>` — one latest pending snapshot;
- monotonic `latest_request_id`;
- frame index;
- `MaterialCache`;
- deferred `(MaterialBundle, retire_frame)` records.

`PresentedPackage` retains the cave node, optional wall/floor mesh handles, wall/floor material bundles, stable point-light IDs, current light descriptors, scene identity, and staging frame.

### Coalescing

```text
submit A (id=1) ── launch worker A
submit B (id=2) ── latest pending = B
submit C (id=3) ── latest pending = C (B replaced)
worker A done   ── stale result discarded; launch C
worker C done   ── accepted CPU result returned to event loop
```

The running worker is not cancelled. Stale success, error, or panic attribution cannot commit because request IDs are checked when polling and again around staging/commit. A worker panic is converted to an error result for the request snapshot actually assigned to that worker.

### Main-thread staging and commit

`commit_replacement` uses the following order:

1. Verify `result_id == expected_request_id == state.latest_request_id` before resource side effects.
2. Require a successful package and an existing active package.
3. Snapshot the old node, meshes, material bundles, stable light IDs, and descriptors.
4. Resolve wall/floor bundles through the material cache or load them on the main thread.
5. Create a candidate cave node.
6. Upload and attach each non-empty wall/floor mesh beneath the candidate.
7. Recheck latest request ID before touching stable lights.
8. Update existing light IDs from candidate descriptors, retaining prior descriptors for rollback.
9. Remove the old cave node.
10. Publish the candidate as `state.active`.
11. Record old material bundles for deferred retirement and hand detached old meshes to `AssetManager::unload_mesh`.

The old node remains authoritative during CPU work and staging. No render call occurs inside this commit function.

### Rollback

- Upload/attachment failure removes the candidate and unloads any candidate mesh already uploaded.
- Staleness after staging performs the same candidate cleanup before light mutation.
- Light-update failure restores every previous light descriptor, then removes candidate resources.
- Old-node removal failure restores lights and removes the candidate.
- `state.active` changes only after all commit operations above succeed.

Mesh unload transfers lifetime management to the renderer's fence-aware queue after scene detachment.

### Material cache and retirement

`MaterialCache` maps a `MaterialCacheKey` to a cloneable `MaterialBundle`. The current key contains the four resolved texture-role references (`albedo`, `normal`, `roughness`, `ao`). Cache hits avoid texture/material creation for an identical key.

A key does not currently include scalar PBR factors, UV/classifier values, or content digests. Callers must not infer those distinctions from cache size, and future cache-key work must be validated against semantic identities rather than path spelling alone.

Material and texture unload APIs are app-owned in this path. Old bundles are removed from reusable cache ownership when appropriate, protected while referenced by the active package/cache, and attempted only after `FRAME_GRACE_PERIOD` (three frame boundaries). Duplicate texture unload is suppressed within a reap pass.

This is a frame-count grace, not a fence-completion serial. Do not document it as a renderer transaction or as proof of fence-gated material/texture retirement. The renderer does own fence-aware retirement for meshes after `unload_mesh`.

## Editor Lifecycle

### Callback registration

Windowed v2 creates `Rc<RefCell<EditorModel>>` and registers one app UI callback under `voxel_editor`. Headless v2 constructs neither the model nor the callback.

- Editor visible: callback registered; current renderer UI policy suppresses app camera/gameplay input.
- F1/F2: intercepted before renderer routing and toggles visibility.
- Hide: `unregister_app_ui` removes the callback and restores camera routing.
- Show: movement/action release events are queued before registration to avoid latched controls.

Registered UI does not promise viewport click-through.

### Draft versus active state

The model keeps a complete draft and a distinct active scene identity/statistics snapshot. Editing changes only the draft and marks it dirty. `revalidate` normalizes the draft, aggregates validation/asset errors, and computes a draft identity only when the full snapshot is valid and resolvable.

Load replaces the complete draft. Save and regenerate capture owned snapshots. Randomize uses OS entropy and changes only the draft seed. None of these actions implicitly presents a new cave.

### Command queue

The imgui callback queues one of:

```text
LoadPreset | LoadConfig | Save(snapshot) | RandomizeSeed
| Regenerate(resolved snapshot + identity) | Hide
```

The queue is bounded at 32. The event-loop owner drains it after frame input/worker handling, performs file I/O or randomness, submits regeneration, and changes callback registration. The resolved regeneration snapshot carries the runtime options but keeps them outside the document/identities.

`EditorPhase` reports `Idle`, `Queued`, `Generating { request_id }`, or `Failed`. Success/failure updates are accepted only for the model's latest accepted request ID.

## Headless Lifecycle

Headless v2 reuses configuration, identity, package, material, upload, light, and viewpoint derivation but is intentionally one-shot:

1. build CPU package;
2. create headless renderer and scene;
3. create materials and upload non-empty partitions;
4. add package lights and environment;
5. warm ten frames;
6. capture five fixed core-role views with renderer and enriched sidecars;
7. exit.

It has no editor, coalescing worker, replacement commit, cache lifecycle, or interactive retirement state.

## Performance Evidence Boundary

The retained Phase 07 JSONL profile has 120 records representing two samples for each of 60 preset/seed/resolution cases. Across those raw records, generation is the largest coarse stage (about 73% of the summed generation/MC33/partition time), followed by MC33 (about 21%) and partitioning (about 6%). The artifact does not isolate generator substages or define an objective threshold for one transformation. Phase 08 therefore made no production optimization rather than selecting a speculative loop from source inspection.

Any future optimization must first add or retain deterministic substage evidence, compare identical inputs/build profile/protocol, and recheck v1 goldens plus v2 identities/fixtures/meshes.

## Validation and Change Boundaries

For app-local changes:

```bash
cargo test -p voxel_demo
cargo check -p voxel_demo
git diff --exit-code -- apps/voxel_demo/test_data/goldens/
```

Also verify that no path under `src/renderer/`, renderer shaders, or descriptor code changed. Visible generator/material changes require deterministic headless capture evidence; documentation-only closeout does not claim new visual proof.

## Key Files

- `apps/voxel_demo/src/cli.rs`
- `apps/voxel_demo/src/config.rs`
- `apps/voxel_demo/src/validate.rs`
- `apps/voxel_demo/src/cave_gen/rng.rs`
- `apps/voxel_demo/src/cave_gen/generators/topology_first.rs`
- `apps/voxel_demo/src/meshers/mc33.rs`
- `apps/voxel_demo/src/meshers/partition.rs`
- `apps/voxel_demo/src/scene_package.rs`
- `apps/voxel_demo/src/materials.rs`
- `apps/voxel_demo/src/regeneration.rs`
- `apps/voxel_demo/src/editor.rs`
- `apps/voxel_demo/src/main.rs`
