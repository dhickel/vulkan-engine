# BSP Beta Knowledge

## Topic
Recurring evidence, assumptions, gotchas, and corrections for the BSP Map Support Beta sprint.

## Source References
- BSP compatibility specification: `.internal-dev/specifications/bsp-compatibility.md`
- Renderer-lighting specification: `.internal-dev/specifications/bsp-renderer-lighting.md`
- Spatial/physics specification: `.internal-dev/specifications/bsp-spatial-physics.md`
- Transaction/ownership specification: `.internal-dev/specifications/bsp-transaction-ownership.md`
- Acceptance specification: `.internal-dev/specifications/bsp-acceptance.md`
- Handoff: `.internal-dev/handoffs/20260723-040449-home-dhickel-code-rust-vulkan-engine-internal.md`
- ericw-tools: `https://github.com/ericwa/ericw-tools`
- TrenchBroom: `https://trenchbroom.github.io/`

## Key Takeaways

### Explorer Tool Discovery Requires Executable Binaries (2026-08-01)
- `bsp_beta --m3-generate` discovers `qbsp`, `vis`, and `light` from an explicit directory, `ERICW_TOOLS_DIR`, the HOME default, or `PATH`.
- On Unix, discovery must require a regular file with at least one execute bit, not merely `is_file()`. A non-executable file otherwise passes discovery and fails later at package compilation rather than producing the CLI's typed explicit-directory error.
- Keep this check local to discovery and retain the non-Unix regular-file fallback; `engine_pack` remains responsible for actual process execution and compiler diagnostics.

### 1. BSP29 vs BSP2 Field Widths — FIXTURE CORRECTED (2026-07-23)
- BSP29 uses 16-bit indices for vertices, edges, faces, markfaces, and model fields.
- **ericw-tools BSP2 uniformly widens fields.** Compiled fixture evidence via authoritative `bspinfo` tool:
  - Nodes: 44 bytes (BSP29: 24 bytes). All fields widened: children i16→i32, mins/maxs i16→i32, face_id/face_num u16→u32.
  - Leaves: 44 bytes (BSP29: 28 bytes). mins/maxs i16→i32, mark/markleaf u16→u32.
  - Clipnodes: 12 bytes (BSP29: 8 bytes). children i16→i32.
  - Edges: 8 bytes (BSP29: 4 bytes). Vertex indices u16→u32 unconditionally.
  - Faces: 28 bytes when non-empty (BSP29: 20 bytes). ledge/ledge_num/plane u16→u32.
  - Markfaces: 4 bytes when non-empty (BSP29: 2 bytes). Face index u16→u32.
  - Models: 64 bytes (unchanged — face_id/face_num already i32 in BSP29).
- The BSP2 magic is the only reliable differentiator: `29` (LE i32) vs `"BSP2"` (4-byte magic).
- Lump count alone is NOT a differentiator: both use 15 standard lumps; BSPX extensions are appended outside the standard header.
- **Parser rule**: Compute BSP2 element width as `(lump_size / element_count)` using authoritative element counts from the profile. Use widened widths for empty lumps (faces, markfaces). The `bspinfo` tool output is authoritative for element counts.

### 2. RGB Lightmaps Can Come from Two Sources
- BSPX `RGBLIGHTING` lump: embedded in the BSP file as an extension lump.
- External `.lit` file: `QLIT` magic plus little-endian version 1 (8-byte header), then RGB payload with the same luxel/style count as the base BSP light data.
- BSPX takes precedence over .lit. Both override the base monochrome lump.
- A mismatch in luxel count between .lit header and BSP face data is a fatal diagnostic in strict mode.

### 3. Face Light Styles Are Per-Face, Not Per-Map
- Each face has 4 style slots (`face.styles[0..3]`), value 255 = unused.
- Style 0 is always the static (normal) lightmap.
- Styles 1–63 reference named light animation patterns (flicker, pulse, etc.).
- The base BSP29 lightmap lump stores multiple monochrome luxel pages: `(num_luxels * num_valid_styles * 1)` bytes per face.
- Colored BSPX/`.lit` companions store the same pages as RGB: `(num_luxels * num_valid_styles * 3)` bytes.
- A face with styles `[0, 1, 255, 255]` has 2 lightmap pages; one with all 4 used has 4.

### 4. Texinfo Flags Control Surface Behavior
- `TEX_SPECIAL` (0x01): special surface (sky, liquid).
- `SURF_SKY` (0x04): sky surface (depth-preserving).
- `SURF_WARP` (0x08): liquid/warp surface (animated, two-sided).
- `SURF_TRANS33` (0x10): 33% transparent.
- `SURF_TRANS66` (0x20): 66% transparent.
- `SURF_FLOWING` (0x40): flowing liquid (scrolls texture).
- `SURF_NODRAW` (0x80): not rendered.
- Texture name conventions (`*sky`, `*water`, `clip`, `trigger`, etc.) are ALSO checked, not just flags.

### 5. Clipnodes ≠ Render Faces — Collision Must Be Separate
- The clipnode tree (lump 9) is a separate BSP tree for collision, using a subset of the render planes.
- Clipnodes use `i16` children in BSP29, `i32` in BSP2.
- The clip hulls are: hull 0 (point), hull 1 (player, 32×32×56 Quake units), hull 2 (large monster, 48×48×64).
- Face geometry CANNOT be substituted for collision authority — surfaces with `SURF_SKY`, `SURF_NODRAW` flags are not solid.
- Similarly, clip geometry should not be rendered (nodraw/clip faces intentionally omit rendering).

### 6. Entity `_tb_id` Survival Requires Pinned-Compiler Proof
- TrenchBroom stores per-entity UUIDs in `_tb_id` key.
- **FIXTURE EVIDENCE (2026-07-23)**: The pinned ericw-tools 2.0.0-alpha3 `qbsp` **strips** `_tb_id` from the compiled entity lump. Both compiled BSP fixtures contain zero `_tb_id` keys. The compiler does NOT preserve unknown entity key/value pairs.
- UUID-backed entity identity is **not viable** with this compiler. Identity MUST use structural fingerprint reconciliation.
- The UUID path in code is retained only as a future hook for a different compiler that preserves entity keys.
- Identity reconciliation must handle: fingerprint present, fingerprint ambiguous, structural changes, and duplicate ordinals for entities with identical fingerprints.

### 7. Quake Angle/Pitch/Mangle Conversion Is Subtle
- Quake pitch: positive = look DOWN (Quake forward vector has `z = -sin(p)`). After `QuakeToEngine`, that maps to negative engine Y; engine Euler APIs with positive-up pitch should use `-p`.
- Sentinel `-1` = straight up, `-2` = straight down (in Quake space).
- `mangle` is identical to `angles` in format, used on monster/entity rotation.
- The proof requires fixture entities at known angles with known world-space directions.

### 8. PVS Is Still Useful for Dense Indoor Scenes
- BSP PVS provides exact visibility sets that can cull 70-90% of faces in indoor Quake levels.
- Frustum culling alone cannot match this because indoor walls create occlusion that frustum cannot exploit.
- PVS is complementary to BVH/frustum: PVS determines batch eligibility, frustum/BVH culls within the eligible set.
- Corrupt VIS must fall back to frustum-only (conservative — no false culling).

### 9. WAD2 Miptex Has 4 Mip Levels
- Each Quake miptex entry stores 4 mip levels: the largest (mip 0) is named, and mips 1–3 are progressively halved.
- Texture name matching uses mip 0 name (case-sensitive).
- Texture resolution: embedded miptex mip 0 is the texture size; WAD miptex may differ.
- Mip 0 dimensions are in the miptex header (width, height as u32 LE).

### 10. No Original Quake Assets Can Be Checked In
- Fixtures must be project-authored .map files with a project-authored palette.
- The CC0 palette must not be derived from the id Software Quake palette.
- Reference renderer screenshots are for calibration only; not redistributed as project fixtures.
- ericw-tools must be user-supplied; not bundled with the engine.

### 11. Shell-Free Compiler Invocation Is Direct Subprocess, Not a Sandbox
- `build_fixtures.py` uses `subprocess.run()` with argument vectors (no shell).
- Controlled working/output directories.
- Minimized environment variables.
- Output-size checks and post-build hash verification.
- This is NOT a security sandbox — the compiler executable is outside the parser trust boundary.

### 12. BSP Descriptors Must Be Separate from PBR ABI
- BSP lightmapped surfaces need: lightmap atlas array texture (style-indexed) and packed material data. External BSP PBR companions must stay on this BSP ABI rather than the general mesh PBR sampler ABI.
- The existing PBR material path uses descriptor sets 0–3 with glTF sampler semantics.
- Mixing BSP samplers into PBR descriptor sets would: increase set size, add unused bindings for non-BSP materials, risk ABI drift.
- A dedicated BSP descriptor set 1 (material) with its own pipeline layout is required.
- Scene set 0 (UBO + shadow map) is shared between BSP and PBR paths.
- **In-flight descriptors are never mutated for frame-varying state.** Frame-varying BSP values (style intensities, animation indices, liquid parameters) use set 2 UBO.
- **Static textures use one array layer.** Animated textures use per-frame layer selection via uniform, not descriptor mutation.

### 13. Renderer BSP ABI Frozen (Phase 01)
- Set 0 (BspScene): six-binding layout identical to SceneData — scene UBO, env UBO, IBL samplers, shadow array. Compatible with shared scene bindings.
- Set 1 (BspMaterial): binding 0 = albedo 2D, binding 1 = packed material data (`R=fullbright`, `G/B=normal X/Y`, `A=gloss`), binding 2 = lightmap atlas sampler2DArray, binding 3 = BspSurfaceUniform UBO (80 bytes).
- Set 2 (frame-varying): style intensities, animation indices, liquid parameters, deterministic simulation values. One write per frame max.
- BSP pipeline variants: opaque, fullbright, alpha-mask, PBR opaque, PBR alpha-mask, sky, liquid — all sharing one pipeline layout.
- BSP descriptor bindings tested in `descriptor_abi_bsp_bindings_registered` (feature-gated).

### 14. Generated-Map Compile Success Does Not Prove Sealing or Texture Completeness
- Absence of a `.pts` leak pointfile is not sufficient sealing evidence. If every point entity is inside solid geometry, ericw-tools `qbsp` can warn `No entities in empty space -- no filling performed` for each hull, skip outside fill, still emit a BSP, and produce no leak pointfile.
- Generated point entities must be probed as non-solid before compilation, and corpus validation must reject the no-entity/no-fill warning.
- Additive corridor brushes do not carve apertures through complete room-wall brushes. A portal must be emitted by splitting/omitting wall solids around the opening; adding a solid “portal marker” makes the obstruction worse.
- Every emitted face texture must be cross-checked against WAD directory names. The dungeon generator's `generator_brick` closure texture was absent from the CC0 theme WAD, while compile/reload tests still passed with warnings.
- BSP visual diagnosis on 2026-07-25 showed that a known-good map rendered correctly through the same WAD/lightmap renderer path; generated fragmentation came from authoring geometry/entity placement rather than global brush winding or PBR companions. Evidence: `.internal-dev/captures/diagnostic/findings.md`.

## Engine Relevance
- The existing engine has all integration foundations (slot+generation handles, fence-aware retirement, neutral geometry DTOs, `SceneBounds::Known`, transactional scene persistence, Rapier collider shapes).
- The `AssetKind` enum must gain a `Bsp` variant for package integration.
- The renderer's current light cap (4 directional, 16 point, 16 spot) must be shared between BSP-imported and app-added lights.
- The existing feature flag pattern (`csm`, `instancing`, `scene-bvh`) is followed for `bsp`.

## Phase 08 Implementation Gotchas

### Physics Bridge Token Serialization
- `PreparedPhysics` contains `BodyDescriptor`, `ColliderDescriptor`, `PhysicsBodyId`,
  `PhysicsColliderId`, and `ColliderShape` types that don't derive `Serialize/Deserialize`.
  The physics bridge stores staged state in the bridge struct (`self.staged`) and uses
  the bridge token only as a generation marker (`vec![1u8]`).
- `commit_to_world()` is a separate method from `AppBridge::commit()` because the
  `AppBridge` trait doesn't receive `PhysicsWorld`. The app must call both.

### Structural Behavior Borrowing
- `StructuralBehaviorAdapter::update()` iterates doors/buttons/platforms by sorted
  entity index. Door terminal targets must be collected into a `Vec<String>` before
  calling `queue_target_activation()` to avoid double-borrow conflicts.
- Updates are sorted globally by entity index after all entity types are processed,
  ensuring deterministic ordering regardless of entity type.

### Cycle Detection
- Activation cascade depth is tracked per-entity with `activation_count: HashMap<u32, usize>`.
  `MAX_CASCADE_DEPTH = 3` prevents infinite loops in A→B→A style cycles while allowing
  legitimate multi-hop chains (e.g., trigger → button → door → target).

### BehaviorEntityRecipe Scope
- Phase 08 validator repair kept `BehaviorEntityRecipe` at its existing fields because
  `src/bsp_runtime/src/bridge.rs` and `src/bsp_runtime/src/coordinator.rs` were outside the
  approved write-target set. Do not add `light_style` or concrete collision mesh fields to
  bridge DTOs unless those files are explicitly in scope.
- `RuntimeBridge` registers light styles from `LightEntityRecipe::style`; `targetname` and
  `target` still depend on future entity key-value extraction from compiled BSP descriptors.

### WorldCollisionRecipe Plane-Only DTO
- `WorldCollisionRecipe` remains plane-only in Phase 08. `apps/bsp_beta::PhysicsBridge`
  creates a static world body when planes are present but cannot create a world collider
  until a future scoped runtime API adds concrete mesh or hull data.
- The validator intentionally removed the out-of-target `WorldTriMesh` API addition; future
  world trimesh extraction must include bridge/coordinator files in the phase write contract.

### Kinematic Body Transform Sync
- `PhysicsWorld::set_body_position_by_id()` / `set_body_pose_by_id()` now expose durable-ID
  body transform mutation. Kinematic bodies also receive `set_next_kinematic_position()` before
  their immediate pose is set.
- `PhysicsBridge::sync_body_transform()` now calls the physics API and tests assert the body
  pose changes, so it is no longer a stub.

## Phase 08 (Persistence) Implementation Gotchas

### BspSourceLink Field Renames
- `BspSourceLink.bsp_source` → `asset_id`, `content_hash`, `compiler_provenance`, `companion_hashes`, `import_policy`, `model_mapping_identity`, `entity_identity_records`, `overrides` (was `bsp_overrides`), `mutable_behavior`.
- `BspSourceReference` and the old `BspSourceLink { bsp_source, bsp_overrides }` envelope are retained as legacy compatibility types but new code uses the flat `BspSourceLink` under a `BspPersistenceEnvelope`.
- The scene stores `BspPersistenceEnvelope` as raw JSON via `Scene::set_bsp_source_link`. The coordinator serializes the envelope before commit.

### Canonical Float Uses f64 in JSON
- `CanonicalFloat` serializes as `f64` in JSON to preserve full precision, but converts back to `f32` on deserialize.
- `to_canonical_bytes()` normalizes -0.0 to +0.0 before producing LE bytes for hashing.
- This avoids platform-dependent float serialization in deterministic hashing.

### EntityOverride Uses CanonicalFloat for Numeric Fields
- `EntityOverride.light_intensity` is now `Option<CanonicalFloat>` (was `Option<f32>`).
- `LightOverride.intensity` and `LightOverride.radius` are also `Option<CanonicalFloat>`.
- Test code needs `CanonicalFloat(value)` wrappers.

## Phase 09 Implementation Gotchas

### Adversarial Test API Mismatches
- `validate_lit` doesn't exist — use `validate_lit_header(data, strict)` for magic/version and `validate_lit_against_lightmap(lit_rgb_size, lightmap_size, strict)` for luxel count.
- `BspWorld` has no `visibility` field — PVS state is accessed through `vis_data` and `bspx` fields.
- `Palette` is `[[u8; 3]; 256]` — no `Default` impl; construct explicitly with `[[0u8; 3]; 256]`.

### ResourceKind API
- `ResourceKind` doesn't implement `FromStr` — it uses `From<&str>` which panics on unknown strings.
- The tag for `ResourceKind::Generic` is `"asset"` not `"generic"`.
- `ResourceKind` has `Texture` and `Manifest` variants in addition to the ones documented in tests.

### Parser Tolerance
- The BSP parser does not reject long miptex names or texture name path traversal at load time — these are checked downstream.
- Model out-of-range face references may not be caught at parse time but surface during extraction.
- Path traversal in texture names uses `SecurityPathTraversal` code when caught, but the parser may parse successfully.

### FIFO Detection Without libc
- Use `std::process::Command::new("mkfifo")` instead of `libc::mkfifo` to avoid an external dependency.
- Non-regular file rejection may emit `PackageIoDeviceFile` or `PackageIoMetadataFailed` depending on the code path.

### Performance Test Dependencies
- Performance harness uses `serde_json` and `serde` (already in `bsp_beta` deps).
- Timestamp uses `SystemTime` + `UNIX_EPOCH` instead of `chrono` to avoid adding a dependency.
- Performance tests are non-ignored (CPU-only) and pass within budget on minimal fixtures.

### Renderer Test Feature Gating
- `descriptor_abi.rs` BSP tests use `#[cfg(feature = "bsp")]` — they only compile with `--features bsp`.
- `bsp_lifecycle.rs` uses `#[cfg(feature = "bsp")]` on the entire module — all tests are behind the gate.
- `gpu_smoke.rs` BSP-inactive tests are `#[ignore]` because they require a Vulkan-capable GPU.
- Descriptor ABI guards should verify the actual current push-constant contract, not stale comments: PBR `VkModelPushConsts` carries model matrix, vertex buffer, material metadata, joint count, `has_uv1`, and padding; BSP `BspModelPushConsts` omits material/joint state and is asserted separately as 80 bytes.
- Renderer lifecycle integration tests that match `FrameRenderOutcome` must cover newer non-fatal outcomes (`PresentedSuboptimal`, `SubmittedNotPresented`) so ignored GPU-required tests still compile under `--features bsp`.

### Scene Source-Link Is Now the Envelope
- `Scene::bsp_source_link()` returns the raw JSON of a `BspPersistenceEnvelope`, not a `BspSourceLink`.
- Access path: `link["schema_version"]`, `link["bsp_source"]["asset_id"]`, `link["bsp_source"]["overrides"]["entity_overrides"]`.
- The old path `link["bsp_source"]["import_settings"]["scale"]` is now `link["bsp_source"]["import_policy"]["scale"]` (a float, not an object).

### Restore Order Must Complete Before Commit
- Phase 08 validator repair corrected `restore_from_persistence` to match the accepted order: schema/source validation → prepare parse/extract → content hash check → hidden upload readiness → identity reconcile → companion/model-mapping validation → mutable behavior validation → scene preflight → commit.
- Mutable behavior validation must not run inside the early runtime-handle check; otherwise invalid behavior can fail before upload readiness and mask post-readiness rollback defects.
- Any failure before commit rolls back the staged candidate and returns an error. Post-readiness failures must leave the active scene source-link payload unchanged and clear `staged_extraction()`.

### Mutable Behavior State Is Mostly App-Populated
- `BspCoordinator::capture_mutable_behavior()` currently returns an empty `MutableBehaviorState`.
- The app bridge (e.g., `RuntimeBridge`) is responsible for populating door/button/platform/trigger state from the live `StructuralBehaviorAdapter` before calling `capture_source_link`.
- Until this is wired, restored BSP mounts start with all entity state machines at their defaults (doors closed, buttons up, etc.).

### Fingerprint Stable Handles Include Duplicate Ordinals
- Source-link stable handles for fingerprint identities must be `fingerprint#duplicate_ordinal` (UUIDs remain raw UUID strings). Storing only the normalized fingerprint makes duplicate entities indistinguishable and causes false ambiguity/orphaning.
- Light overrides use the same UUID/fingerprint+ordinal reconciliation path as entity overrides. Do not leave light overrides UUID-only while `_tb_id` is stripped by ericw-tools.

### Schema Version Enum Has u32 Discriminant
- `SchemaVersion::V1 = 1` as a C-like enum with explicit discriminant.
- Serialization uses the u32 value directly; deserialization uses `SchemaVersion::from_u32` which returns `None` for unknown versions.
- `approved_prior()` currently returns an empty slice because V1 is the first version.

## BSP Dungeon Contract Evidence Matrix Validation
- The Phase 01 reconcile-contracts matrix must contain exactly 80 evidence rows with unique IDs and 14 columns: `cell_id`, `category`, `fixture_identity`, `exact_command`, `expected_observation`, `actual_observation`, `tool_hashes`, `input_hashes`, `output_hashes`, `environment`, `evidence_paths`, `affected_spec_cells`, `status`, and `blocks_generator`.
- Pre-execution rows may explicitly record `NO COMMAND IN PHASE 01` when no authorized generator/public entrypoint exists, but downstream phases must replace that blocker with an executable command before promoting a row from `NOT-RUN`.
- Keep the summary table synchronized with row truth. After the Phase 10 closeout validator repair, the finalized matrix has exactly 80 rows, 37 `PASS`, 43 `NOT-RUN`, and 11 `NOT-RUN` rows with `blocks_generator = true`; those 11 are the GPU/WSI-dependent visual/live-navigation rows. Do not leave stale summary text claiming 23/55 blocking rows or stale row-level `blocks_generator = true` on non-GPU follow-ups.
- Use `.internal-dev/debug_reports/bsp-dungeon-generator/phase-08-corpus.json` for final corpus maxima. The final corpus evidence records M1 max 1,404 compiled faces / 18 entities and M2 max 4,917 compiled faces / 42 entities; do not repeat the stale 2,064 M1 face-estimate wording as “within <2,000” evidence.

## 2026-07-26 Integration Revalidation Gotchas

### Published Closure Must Exclude Staging Ownership Metadata
- `.engine-pack-staging` is proof of ownership only while a directory is staging. It must be removed before final manifest-closure validation and atomic rename; allowing it as an undeclared final-tree exception makes a published package contain transaction state and breaks exact-tree comparison.
- Defend at both boundaries: `validate_manifest_closure` rejects an undeclared marker, and `publish_directory_no_replace` rejects a marker that appears after preflight. The CLI integration regression must invoke `engine_pack compile-bsp`, then assert the marker is absent and the final tree validates exactly against its manifest.

### Do Not Promote Preflight Counts to Submission Evidence
- The 2026-07-26 nominal-M1 generated package reaches 364 renderable faces and six neutral/upload-preflight batches under explicit development policy. This is useful current evidence that immutable batching is below the `<100` batch ceiling, but it is not a strict mount or submitted-draw result.
- The same package strictly authorizes then correctly fails neutral extraction at `MissingRequiredLightmap` face 75 (GitHub #58). Development bypasses that release gate only for diagnosis and cannot close strict acceptance.

### GPU Rollback Crash Is a Separate Renderer Blocker
- The development mount above reaches `BspUploadReceipt::rollback` after material registration and SIGSEGVs in RADV descriptor-pool destruction: `BspSurfaceCache::clear_frame_values -> destroy_descriptor_pool`. This is tracked as GitHub #61 and is distinct from closed #55 because it reproduces on a six-batch M1 candidate, not a large-map descriptor-exhaustion case.
- Treat the resulting neutral/preflight count as bounded CPU evidence only. Do not claim GPU mount, scene publication, PVS submission, capture, replacement, or fence-retirement success until #61 is repaired and the typed rollback path is exercised.

## Open Questions
- Can convex reconstruction from clipnodes faithfully recover all brush entity shapes in the fixture maps? Counter-examples: concave brushes, thin brushes, angled geometry near BSP split planes.
- `_tb_id` does **not** survive the pinned ericw-tools 2.0.0-alpha3 compiler. Other Q1 compilers (txqbsp, hmap2, etc.) would require separate compiler-specific fixture proof before enabling a UUID identity path.
- What SSIM tolerance is achievable between engine PBR rendering of BSP lightmaps and reference software Quake rendering? The lighting models differ substantially.
- Will the 2-texel lightmap atlas padding be sufficient at all mip levels, or will atlas bleeding occur at lower mips?

## Phase 10 Sign-Off Evidence (2026-07-23)

### Automated Test Evidence
- `cargo test -p bsp` — ~191 tests pass (parser, adversarial, entities, resources, extraction)
- `cargo test -p bsp_runtime` — ~68 tests pass (coordinator, bridge, persistence, reload, snapshot)
- `cargo test -p renderer --features bsp` — all renderer tests pass including `descriptor_abi_bsp_bindings_registered` and `bsp_lifecycle`
- `cargo test -p dungeon_dogfood` — all existing tests unchanged by BSP feature
- `cargo test -p voxel_demo` — all existing tests unchanged
- `cargo test -p engine_pack` — `compile-bsp` tests pass
- `cargo test -p apps/bsp_beta` — CLI, persistence, and runtime tests pass
- **Total**: ~685 tests pass across all BSP-touched crates

### Compile Matrix
| configuration | status |
|---------------|--------|
| `cargo check` (default features, no BSP linked) | PASS |
| `cargo check -p renderer --features bsp` | PASS |
| `cargo check -p renderer --all-features` | PASS |
| `cargo check -p renderer --examples` | PASS |
| `cargo check -p bsp_runtime` | PASS |
| `cargo check -p dungeon_dogfood` | PASS |
| `cargo check -p voxel_demo` | PASS |

### Non-BSP Regression
- All existing renderer examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `demo_async_loading`, `api_test`) compile and pass timeout-bound smoke with default features.
- `capture_bsp_beta` example compiles behind `--features bsp` only and does not affect default builds.
- No BSP crate linked in default builds.

### Remaining Blockers
1. **No visible-face BSP fixture** — the pinned ericw-tools compiler produces zero face geometry for current test fixtures. Visual captures, deterministic capture acceptance, BSPX RGBLIGHTING proof, and calibration against reference renderer are blocked or unrun until a map produces visible, lightmapped faces.
2. **Live WSI not validated** — headless captures cannot substitute for swapchain lifecycle tests. Requires a live GPU/window-system environment.
3. **Reference renderer calibration blocked** — SSIM comparison vs vkQuake pending visible-face fixture.
4. **Dynamic shadow composition (CSM) for BSP path not yet integrated** — pending CSM+BSP integration.
5. **World trimesh collider from clipnodes not yet implemented** — `WorldCollisionRecipe` remains plane-only; point-contents and hull traces are functional.

### Documentation Complete
- `docs/guide/18-bsp-beta.md` — app-builder how-to
- `docs/api/17-bsp-beta.md` — public API contract
- `docs/internal/18-bsp-runtime-and-lifetime.md` — ownership graph, protocols, failure matrix
- All index files and existing deep dives updated with BSP cross-links
- All 6 module AGENTS.md files updated
- All 5 BSP specifications updated with final Phase 09 evidence
- `decisions.md` includes all 15 BSP architecture decisions

### Sign-Off Disposition
**BLOCKED / not accepted.** Implementation phases 01–09 have automated evidence and the documentation/specifications are reconciled, but Phase 10 sign-off cannot pass while the traceability matrix contains 7 NOT-RUN and 2 BLOCKED rows. The reusable validation rule is that compilation of `capture_bsp_beta` is not a substitute for a raw visible-lightmapped capture artifact.

## Phase 02 Implementation Gotchas

### BSP2 Leaf Stride — Superseded Correction
- Earlier Phase 02 notes assumed ericw-tools BSP2 leaves kept i16 mins/maxs and used a 32-byte stride. That note is superseded by Phase 01 validator evidence: `bspinfo` reports `ericw-bsp2-colored.bsp` has 7 leaves and a 308-byte leaf lump, so the authoritative stride is **44 bytes**.
- Use the approved BSP2 leaf layout from `bsp-compatibility.md`: contents i32, visofs i32, mins i32×3, maxs i32×3, mark u32, markleaf u32, ambient u8[4]. Do not reintroduce the 32-byte leaf assumption.

### BSPX Directory Placement
- The BSPX directory lives at the very end of the file: entries + u32 count + BSPX magic. The entry data must be placed before the directory (at lower file offsets). discover_bspx validates that entry ranges don't overlap the directory itself.
- Standard lump overlap with BSPX entries is caught in validate_bspx_entries, not discover_bspx.

### Entity Parser State Machine
- The tokenizer operates within `TokenState::InEntity` without separate InKey/InValue states. The alternation between key and value is tracked via `current_key.is_none()`.
- This handles all Quake entity quoting including escape sequences (`\n`, `\"`, `\\`) within values.

### Model Lump Is Always 64 Bytes
- The model lump element size is always 64 bytes regardless of BSP29 or BSP2. face_id and face_num are always i32 (4 bytes each), not u16. The BSP2 profile only changes the semantics (32-bit index space), not the storage width for these fields.

### WAD Path Traversal
- WAD texture name sanitization must check for `..` in the full path BEFORE extracting the basename. Otherwise `../escape` becomes basename `escape` and passes the check.

### Palette Decode
- The palette is exactly 768 bytes: 256 × 3 bytes (RGB triples). Index 0 = bytes [0..3), index 255 = bytes [765..768).

### Cross-Lump Validation Order
- All lumps are parsed first, then cross-lump indices are validated during BspWorldBuilder::build(). This ensures whole-asset rejection for structural corruption — individual faces/models/nodes are never silently dropped.
- Fatal diagnostics accumulated by subsystem parsers must be checked before returning `BspWorld`; otherwise strict unsupported BSPX extensions or corrupt WAD companions can be silently returned as a world with an error diagnostic.
- Graph acyclicity is checked via depth-bounded DFS on both the node tree and the clipnode tree.

### BSP2 Stride Checks
- BSP2 nodes, leaves, clipnodes, edges, faces, and markfaces need approved widened strides before allocation. Fixture-verified strides are: nodes 44B, leaves 44B, clipnodes 12B, edges 8B; faces 28B and markfaces 4B when non-empty. Keeping BSP29 strides while reading BSP2 offsets causes hidden out-of-bounds/truncation failures.
- Model lump storage remains 64 bytes and stores face_id/face_num as i32 for BSP29 and BSP2; do not parse BSP29 model face fields as u16.

### Entity Grammar Fail-Closed Checks
- Entity tokenizers must reject unquoted tokens, nested braces, key-without-value, and non-null-terminated entity lumps. Treating these as skipped text breaks the fail-closed parser boundary.
- Empty entities and unknown classes are preserved diagnostics; malformed structural grammar is not preserved.

### Diagnostic Codes Are Stable
- Test assertions match on DiagnosticCode and Severity, never on message text. Message strings may change; codes are the stable contract.

## Phase 03 Implementation Gotchas

### QuakeToEngine Coordinate Transform
- The transform `(x,y,z) → scale*(x,z,-y)` is applied to all positions. Normals use `(nx,nz,-ny)` then renormalize. Plane distances multiply by scale.
- AABB conversion requires converting both corners and recomputing min/max since the axis swap can invert ordering.
- The inverse transform (engine→quake) is needed for stored-hull traces: `(ex,ey,ez) → (ex/s, -ez/s, ey/s)`.

### Angle Conversion Subtleties
- Quake `angle -1` = straight up (engine +Y), `-2` = straight down (engine -Y).
- Quake yaw 0 = east (+X), 90 = north (engine -Z). The direction vector must be `(cos(yaw), 0, -sin(yaw))` not `(cos(yaw), sin(yaw), 0)`.
- Quake pitch positive = look DOWN. For engine Euler APIs with positive-up pitch, use `-quake_pitch`.
- Entity `mangle` uses the same (pitch, yaw, roll) tuple order as `angles`.

### Face Winding from Surfedges
- `reconstruct_winding` collects the START vertex of each surfedge (not the end). Negative surfedge indices indicate reversed edge direction — swap v[0]/v[1].
- Degeneracy checks must happen in Quake space (before scaling) for the 2^15 component limit; the check in engine space would be scale-dependent.
- The planarity check happens in engine space. Non-planar faces with vertex-plane distance > 1e-4 are rejected.

### Lightmap Atlas Layout
- Atlas padding of 2 texels prevents bilinear bleeding between adjacent face blocks.
- The atlas packer uses a simple row-based greedy algorithm. Face luxel blocks are placed left-to-right within a row, with row-wrapping when a block doesn't fit.
- Face luxel dimensions are estimated from texinfo scale: `ceil(max_extent / tex_scale / 16) + 1` per Quake convention, clamped to 256.

### PVS Decompression
- The RLE scheme: byte `0x00` means "next byte × 8 zero bits", non-zero byte means "raw bits for next 8 leaves".
- Decompression must validate the command stream doesn't overrun the PVS buffer or the VIS data.
- Corrupt PVS for a single leaf does NOT disable PVS globally — only that leaf gets the conservative all-visible fallback.
- Missing or empty VIS data (size 0) disables PVS globally.
- The conservative fallback sets all bits for the true leaf count; excess bits in the last byte are masked to 0.

### Stored-Hull Traces
- Traces operate in Quake space, not engine space. The trace function converts engine start/end back to Quake space.
- The recursive hull check follows the Quake `SV_RecursiveHullCheck` algorithm: walk clipnode tree, compute plane intersection fraction, check front side first.
- The point hull check (for start-solid detection) does not expand the hull extents — it checks a single point against the clipnode tree.
- Hull extents are: hull 0 = point, hull 1 = ±(16,16,24) Quake units (player), hull 2 = ±(24,24,32) (large monster).

### Convex Reconstruction from Clipnodes
- `collect_clip_planes` DFS-walks the clipnode tree and converts all encountered planes to engine space.
- `convex_from_planes` computes the convex polyhedron by intersecting all 3-plane combinations, keeping points that satisfy all half-space inequalities.
- Deduplication of planes must ONLY remove same-direction near-parallel planes (dot ≈ 1), never opposite-direction (dot ≈ -1) which define the other side of a box.
- Volume detection requires at least 4 non-coplanar vertices — a flat polyhedron is degenerate.
- The complexity limit of 64 faces and 128 vertices per convex piece is enforced before vertex generation.

### Entity Identity Reconciliation
- With the pinned ericw-tools 2.0.0-alpha3 compiler, UUID-backed identity (`_tb_id`) is unavailable because qbsp strips the key from compiled entity lumps. Treat UUID matching as a future hook only for compilers with separate preservation proof.
- Current reconciliation uses structural fingerprint: (classname, origin, targetname, target). Duplicate ordinals distinguish entities with identical fingerprints.
- Ambiguous matches (multiple old entities with same fingerprint) are reported as `IdentityAmbiguous`.
- `find` closures receive `&&T` (reference to item) but return `Option<&T>` — dereference carefully in match arms vs closure bodies.

### Test Pattern: Iterator Parameter Types
- `Iterator::find()` passes `&Self::Item` to its closure (so `&&T` for `&T` items).
- `Iterator::any()` passes `Self::Item` to its closure (so `&T` for `&T` items).
- This difference caused confusing compile errors in `has_volume` vetting; the fix was `**v` in `find` closures and `*v` in `any` closures.

### Phase 03 Validator Repairs
- Face planarity validation must compare `dot(converted_plane_normal, engine_vertex) - converted_plane_dist`, not `dot(face_normal, vertex)` against zero. Non-origin faces are otherwise falsely rejected.
- PVS RLE decode must fill exactly `(num_leaves + 7) / 8` bytes. Truncated streams, missing zero-run counts, zero-length runs, and zero-runs that overshoot the destination are corrupt and must discard the partial decode in favor of the all-visible fallback.
- Stored hull extents are AABB extents after `QuakeToEngine` axis remapping: player `±(16,16,24)` Quake becomes engine extents `(16,24,16) * scale`; large monster `±(24,24,32)` becomes `(24,32,24) * scale`.
- Inline model render batching must keep model indices separate even when material, lightmap page, render class, and empty leaf signatures match.
- Phase 03 validation found that returning `Ok(ExtractedBsp)` with `Error` diagnostics still hides malformed extraction data. `extract(BspExtractionRequest)` must fail on any error-severity extraction diagnostic after texture, lightmap, entity, visibility, collision, and invariant validation.
- `BspExtractionRequest::palette` is optional only so empty/test worlds can use `Default`; any world with miptex names requires an authorized palette from the request or parsed world. Do not reintroduce an all-black or identity fake palette in callers.
- Neutral lightmap layouts now carry per-style layer metadata (`FaceLightmapLayout::style_layers`). Renderer upload may still expose one GPU layer per packed page until downstream GPU array upload is implemented, but extraction must decode and validate every style byte range.
- `BspCoordinator::prepare` currently has no palette/companion input, so textured BSPs through that API fail closed until package/runtime plumbing supplies authorized bytes.
## Phase 04 Implementation Gotchas

### CoreShaderType COUNT Must Be Public And Feature-Gated
- `CoreShaderType::COUNT` must be `pub const` and feature-gated with all feature combinations because `vk_bsp.rs`'s function signatures reference it as an array size.
- `VkDescType::COUNT` must also be `pub const` because `VkDescLayoutCache::new` uses it as an array size.
- Failing to keep these `pub` causes `E0624` private associated constant errors.
- The `const fn` pattern in `VkDescLayoutCache::bsp_scene_index()` / `bsp_material_index()` avoids `cfg!` which doesn't work in const context.

### OwnedPipeline Shares Layout Across BSP Variants
- All seven BSP pipeline variants share one `vk::PipelineLayout`. Each variant creates an `OwnedPipeline` pointing to the same layout. `OwnedPipeline::disarm()` calls `mem::forget(self)`, so the shared layout is never double-freed.
- `VkPipelineCache::destroy` deduplicates `PipelineLayout` handles in a `HashSet`, so the shared layout is destroyed exactly once.

### BSP Descriptor Layouts Are Separate From PBR
- BSP has `BspScene` (set 0) and `BspMaterial` (set 1). `BspScene` uses the identical six-binding structure as `SceneData` so the scene set can be shared at bind time, but it is a separate cached layout.
- BSP pipelines never reference `PbrSamplers` (set 2) or `SkinData` (set 1 for general mesh PBR). The BSP set 1 is `BspMaterial` with BSP semantics (albedo, packed fullbright/normal/gloss data, lightmap array, surface UBO), including the BSP-specific PBR fragment variants.

### VkSubAlloc Needs Default
- `BspCachedSurface` stores a `surf_ubo_alloc: VkSubAlloc` which default-initializes to zero/null sentinel values before Phase 05 wires real UBO allocations.

### Manifest Parsing Shared Between Core and BSP
- `load_core_shader_manifest` was refactored to call `parse_shader_manifest_lines` for both the core manifest and (conditionally) the BSP manifest. The function no longer contains inline parsing logic.
- Duplicate key detection is shared across both manifests via a `HashSet<CoreShaderType>` seeded from the core manifest.

### GLSL Includes
- BSP shaders `#include "vertex_struct.glsl"` from the same directory using glslc `-I .` flag, which mirrors the core shader build convention.
- BSP material set 1 is fragment-only. Do not read `BspSurfaceParams` from `bsp_lightmapped.vert`; Vulkan pipeline creation/validation requires descriptor stage flags to cover every shader read.
- Keep `vk_bsp.rs` privately included from `vk_pipeline.rs` when the phase write contract excludes `src/renderer/src/vulkan/mod.rs`; otherwise the module declaration itself becomes an out-of-target edit.
- When a GLSL shader changes, rebuild the checked-in `.spv` too; renderer defaults to `compile_shaders: false`, so headless captures otherwise keep using stale bytecode.

### Phase 04 Validator Repairs
- `Renderer::prepare_bsp_mount` must route through the asset upload manager or otherwise pump transfer submissions while BSP mesh uploads run. Calling the raw core upload path directly can deadlock waiting for synchronous transfer completion.
- VMA 0.4 host-visible buffers that are mapped manually need `AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE`; relying only on `HOST_VISIBLE | HOST_COHERENT` triggers `IsMappingAllowed()` assertions.
- `BspSurfaceCache` owns active BSP atlas/UBO payloads at shutdown. Do not lock the allocator mutex inside cache teardown while `VkRenderCore::drop` already holds the allocator guard; pass the allocator reference through to avoid a drop-time deadlock.
- BSP lightmap UVs from extraction are normalized per-face luxel coordinates. `bsp_lightmapped.frag` must apply `BspSurfaceUniform.lightmapScaleBias` before sampling the atlas array, or rendered surfaces sample the wrong/empty atlas region and appear black.
- Validation needs a fixture with visible, lightmapped faces. Synthetic ericw-tools fixtures may need deterministic lightdata patching when the compiler marks test texinfos special/no-lightdata.
- `apps/bsp_beta` is currently Git-ignored by the root `apps/*` ignore rule. Local validator repairs there validate runtime behavior but will not appear in ordinary `git status` unless ignored paths are explicitly inspected.
- Until `BspCoordinator::prepare` accepts authorized palette/companion bytes, `bsp_beta` must extract palette-backed BSP DTOs directly and pass them to `Renderer::prepare_bsp_mount`; using coordinator extraction for textured fixtures fails with `MissingRequiredPalette`.

## Phase 05 Implementation Gotchas

### Candidate Owns All Staged State
- `BspCandidate` replaces the loose `staged_extracted`/`staged_source_link`/
  `staged_cache_identity` fields. The coordinator holds at most one candidate.
- New prepare atomically replaces the previous candidate; the old candidate's
  bridge tokens are rolled back via `BridgeAggregator::rollback_tokens()`.
- After commit, the candidate is consumed (`self.candidate.take()`). The
  active state moves to `active_extracted`/`active_source_link`/`active_lights`.

### Renderer Lease State Machine
- `RendererLease`: NotStarted → Pending → Ready, or NotStarted → Ready (sync).
- `set_renderer_ready` is idempotent for Ready → Ready (replaces mount).
- `commit_with_mount` is a compatibility wrapper: `set_renderer_mount_ready` +
  `commit`. The test pattern can use either.
- `commit` requires: validated candidate + renderer mount ready + generation match.

### Commit Is Pure Publish
- `commit` does NOT: parse, resolve packages, load assets, allocate GPU resources,
  upload, look up handles, serialize, validate bridges, validate restored-state,
  or reserve app-world capacity.
- All of those operations must complete before `commit` is called.
- Bridge activation during commit is non-fallible; a panic poisons the coordinator.
- On commit failure after bridge commit, the coordinator is poisoned (state
  cannot be trustfully rolled back).

### Bridge Commit Must Be Idempotent Across Mount Cycles
- During reload/reimport, the same bridge instance receives a second `commit`
  call for the new mount. Bridges must reset prior committed state before
  accepting the new batch.
- `PhysicsBridge::commit` clears `self.committed`, `self.staged`, and
  `self.published_*` before setting the new batch.
- `RuntimeBridge::commit` resets `self.adapter` before accepting new entities.

### Coordinator Poisoning Checkpoints
- `prepare`, `validate`, `commit`, `rollback`, and `unload` all check
  `self.poisoned` and return `CoordinatorPoisoned` if true.
- `teardown` does NOT check poisoned (it's a terminal cleanup path).
- `rollback` delegates to `rollback_staged` which checks poisoned.
- Bridge rollback panics set `self.poisoned = true` and return `RollbackFailure`.

### Reload/Reimport Beside Active World
- `reload` calls `prepare` (new candidate, hidden) while the active mount
  stays visible. Only after full prepare+validate+set_renderer_ready does
  `commit` atomically swap old→new.
- On any failure during reload, the old world is unchanged.
- Override reconciliation happens against the candidate's extracted identities,
  not the active world's.

### Test Pattern: commit_with_mount vs commit
- Tests using `commit_with_mount` work with the old pattern (pass mount directly).
- Tests using `commit` must call `set_renderer_mount_ready` first.
- Both paths check generation token and validation state.

### PreparedBspMount Does Not Implement Debug
- `PreparedBspMount` doesn't derive `Debug`, so `RendererLease` and
  `BspCandidate` must use manual `Debug` impls that skip the mount display.

### BridgeAggregator Internal Token Storage
- Legacy `prepare`/`validate`/`commit`/`rollback` use `self.prepared_tokens`.
- Candidate-aware `prepare_with_tokens` returns tokens to caller.
- `validate_candidate` reads tokens from `&BspCandidate`.
- `commit_candidate` takes tokens from `&mut BspCandidate`.
- `rollback_tokens` accepts externally-owned tokens and rolls them back.

### Phase 05 Validator Repairs (2026-07-23 Atomic Publication)
- Commit purity means even source-link JSON serialization and BSP light-capacity checks must happen before commit. `BspCandidate` now stores pre-serialized scene source-link JSON and prevalidated point-light payloads.
- BSP candidates that publish point lights must call `validate_for_scene(token, &mut scene)`; plain `validate(token)` is only fully publication-ready for candidates with no scene lights. Scene point-light slot storage is reserved during this validation step so commit does not grow the scene light table.
- Validation failures roll back the current candidate immediately, but stale generation validation still leaves the newer candidate intact.
- Run coordinator-owned fallible candidate payload construction before bridge prepare; otherwise a later local validation failure can leak bridge-prepared hidden resources.

## Phase 05 Implementation Gotchas (continued)

### BspCandidate Does Not Implement Debug
- `BspCandidate` derives `Debug` but `PreparedBspMount` doesn't implement `Debug`,
  so `RendererLease` must use a manual `impl fmt::Debug` that skips the mount body.

### BridgeAggregator Fields Are Private
- `BridgeAggregator::len()` and `has_bridges()` already existed; new methods follow
  the same pattern of being defined on the impl block in `bridge.rs`.


- `bsp::materials::SurfaceClass` uses `Opaque`, `Fullbright`, `AlphaMask`, `Sky`, `Liquid`, `NoDraw`, `Clip`, `Trigger`, `Skip`.
- The renderer's `BspSurfaceClass` uses `Lightmapped` for the Opaque case (standard lightmapped surfaces).
- Mapping: `SurfaceClass::Opaque` → `BspSurfaceClass::Lightmapped`; `SurfaceClass::Fullbright` → `BspSurfaceClass::Fullbright`; `SurfaceClass::NoDraw` → excluded from material descs.

### FaceLightmapLayout Field Names
- `FaceLightmapLayout` uses `atlas_offset: (u32, u32)` (not `page_x`/`page_y`), `luxel_extents: (u32, u32)` (not `luxel_width`/`luxel_height`), `has_data: bool`.

### LightmapAtlas Has No `luxels` Field
- `LightmapAtlas` stores pixel data in `AtlasPage::data` (Vec<u8>, RGB8 packed). It has `pages`, `face_layouts`, `styles` fields. The atlas pages are populated during extraction.

### Module Visibility for Tests
- The `data` and `scene` modules are crate-private. Integration tests must import through the public `api::bsp` re-export path.
- Types needed by tests must be re-exported from `api/bsp.rs`.

### BspMountState Duplicate Import
- When a type is both internally used and re-exported in `api/bsp.rs`, remove the standalone `use` import to avoid E0252 conflicts. The re-export makes the name available within the module.

### BSP Draw Dispatch
- BSP draws use dedicated pipeline variants (BspOpaque, BspFullbright, BspAlphaMask, BspPbrOpaque, BspPbrAlphaMask, BspSky, BspLiquid) with a different descriptor set binding order than general mesh PBR:
  - Set 0: scene data (shared layout with PBR)
  - Set 1: BSP material (albedo, fullbright, lightmap array, surface UBO)
- BSP draws must be recorded before the geometry dynamic-rendering scope ends. Recording after `cmd_end_rendering` compiles but emits draw commands outside rendering.
- `record_bsp_draw_sequence_impl` must bind set 0 on every BSP pipeline-layout switch and bind set 1 whenever the BSP material descriptor changes. Production BSP uploads must create real descriptor sets; null material descriptors are a defect and must be rejected before draw submission, not used as a silent fallback.
- The function accesses `mesh_cache`, `texture_cache`, and `bsp_surface_cache`; BSP mesh and texture handles should be marked referenced with the prospective submit serial just like PBR draws.

### Phase 05 Validator Repairs
- `Scene::set_bsp_mount` must move the prepared face mesh/material arrays, render batches, and light descriptors into `BspMountState`; storing only VIS leaf state silently drops all BSP draw and light submission data.
- `SceneWorld::build_submission` should update BSP PVS and submit BSP draw items even when the regular scene graph has no root. A BSP-only mount with no scene nodes is still renderable.
- Empty or missing batch leaf signatures are conservative-visible under a valid PVS. Treating an empty signature as “intersects nothing” creates false culling.
- Production light-selection tests must call `BspMountState::select_*` instead of testing local helper copies; otherwise scoring, PVS fallback, and hysteresis can regress while tests pass.
- Current `LightmapAtlas::AtlasPage` stores one packed RGB8 page, not one buffer per style layer. Until extraction writes per-style layers, renderer upload DTOs should report one layer rather than `atlas.styles.len()` to avoid a texture-array size/data mismatch.
- `SurfaceClass::Clip`, `Trigger`, and `Skip` are non-renderable like `NoDraw`; material and face mesh construction must skip every `!SurfaceClass::is_visible()` class. Alpha-mask surfaces still attach the lightmap atlas.
- `cargo fmt -p renderer` can format many pre-existing renderer files outside a phase write contract. Revert out-of-scope formatting before closeout when a phase requires a strict write-target set.

## Phase 06 Implementation Gotchas

### TOML Serialization Must Stay Out of `bsp` Crate
- The `bsp` crate has a strict dependency budget: only `glam` for math types. Adding `toml` as a dependency violates this contract.
- All TOML serialization/deserialization for `BspPackageManifest`, `CompilerProfile`, and related types lives in `engine_pack::compiler` (which already depends on `toml`).
- The `bsp::package` module exports only pure data types and the `validate_bsp_package_manifest` function (which requires no TOML).

### Compiler Profile Parsing Has Defaults
- `timeout_seconds` defaults to 120 if not specified.
- `max_output_size` defaults to 128 MiB (134,217,728 bytes) if not specified.
- `expected_hashes` is optional — only populated when reproducibility verification is configured.

### Shell-Free Invocation Pattern
- `compile-bsp` runs `qbsp`, `vis`, `light`, and compiler version probes as direct subprocesses via `std::process::Command` with `.env_clear()` + controlled env. No shell interpolation.
- If profile `expected_hashes` are present, executable SHA-256 hashes are verified before any compiler executable is run.
- Compiler version is verified by running each executable with `--version` (and `-help` fallback) under the same cleared environment and checking for the expected version string in output.
- Stderr and stdout are captured separately and recorded in the provenance.
- Compiler stage and version-probe subprocesses enforce `timeout_seconds`; timeout kills the child and fails closed.
- Output size is checked against `max_output_size` before the BSP is read.
- Post-compile re-validation runs the compiled .bsp through `bsp::BspLoader::load` with `strict: true`.
- Compiler provenance records actual stage arguments plus source and accepted output SHA-256 hashes; do not record only default profile arguments.
- `compile-bsp` should run external tools in an internal staging work directory and remove it before publishing, otherwise copied `.map`/palette files or compiler scratch outputs can leak into the package output.

### AssetKind::Bsp Added to Registry
- The `AssetKind` enum in `src/renderer/src/data/asset_registry.rs` gained a `Bsp` variant.
- Serialization uses `"bsp"` as the string form.
- The renderer API re-exports `AssetKind` automatically — no separate change needed in `api/mod.rs`.
- `engine_pack`'s `classify_asset_kind` recognizes `.bsp` extension.
- `engine_pack`'s `parse_asset_kind` accepts `"bsp"` string.

### TrenchBroom Game Config Format
- TrenchBroom 2024.1+ uses JSON for `GameConfig.cfg` (version 9).
- The FGD uses Quake-style entity definition syntax.
- Engine-recognized structural entities: `worldspawn`, `light`/variants, `func_door`, `func_button`, `func_plat`, `trigger_once`, `trigger_multiple`, `target`, `info_player_start`, `info_teleport_destination`.
- Unknown classnames are preserved as generic entities.

### Sha256 Implementation
- `compiler.rs` includes a minimal, no-dependency SHA-256 implementation (`mod sha2`) for compiler executable hashing.
- This avoids adding a sha2 crate dependency to engine_pack.
- Tests verify against known vectors (empty string, "abc").

### Companion File Kinds
- `CompanionKind` enum: `Lit`, `Palette`, `Wad`.
- Round-trip via `from_str`/`as_str` uses lowercase names: `"lit"`, `"palette"`, `"wad"`.
- `CompanionBinding` stores package-relative path and optional content hash.

### Phase 06 CC0 Theme WAD2 Directory Repair
- Quake WAD2 directory entries use the `lumpinfo_t` order `filepos`, `disksize`, `size`, `type`, `compression`, two padding bytes, then `name[16]`. Do not write name-first directory entries; they may pass local tests that mirror the bug but fail authoritative WAD readers.
- Quake WAD2 miptex lumps use type byte `0x44` (`TYP_MIPTEX`), not `0x43`.
- Theme validation should parse directory entries in the authoritative order and independently verify generated WAD bytes, not just compare deterministic outputs.

### Phase 06 Validator Repairs (Surface/Lighting/Animation)
- GLSL uniform blocks default to std140. A declaration like `float styleIntensity[64]` has a 16-byte array stride and does **not** match a Rust `[f32; 64]` inside a 288-byte UBO. Pack light-style weights as `vec4 styleIntensityPacked[16]` in GLSL while keeping Rust `[f32; 64]` contiguous.
- The renderer's current BSP atlas upload uses four face-slot-local lightmap array layers (slot 0..3), not one sparse layer per global style ID. The surface UBO carries the four style IDs only for weight lookup.
- Keep `surfaceFlags` and `receiveMask` semantically separate. Classification flags are alpha/sky/liquid/fullbright bits; sealed/outdoor receive defaults belong only in `receiveMask`.
- Fullbright emission is additive wherever the fullbright mask is nonzero; do not guard it on a surface class flag because fullbright is per-pixel, not a current `bsp::materials::SurfaceClass` variant.
- Transparent ordering must be one pass/policy: draw opaque and alpha-mask geometry first, draw BSP opaque/sky before transparency, then sort non-BSP blended draws and BSP liquids together back-to-front.
- Frame-varying BSP values should update the current frame slot's UBO bytes after the slot fence wait while leaving set 2 descriptor bindings immutable.

## Phase 07 Implementation Gotchas

### Renderer `api::scene` Is `pub(crate)`
- The `scene` module in `renderer::api` is `pub(crate)`, so external crates cannot use
  `renderer::api::scene::Scene` or `renderer::api::scene::PointLight`.
- Use the re-exports from `renderer::api` instead: `renderer::api::Scene`, `renderer::api::PointLight`,
  `renderer::api::PointLightId`. These are re-exported in `api/mod.rs`.

### PreparedBspMount Is Not Re-exported from api/mod.rs
- `PreparedBspMount` is defined in `api::bsp` and is `pub`, but it is NOT re-exported from
  `api/mod.rs`. External crates must use `renderer::api::bsp::PreparedBspMount`.
- `BspMountState` is also in `api::bsp` (re-exported from `scene::bsp_visibility`).

### BSP Feature Gate Is on renderer, Not bsp_runtime
- The `bsp_runtime` crate does NOT define a `bsp` feature. It unconditionally depends on
  `renderer` with `features = ["bsp"]`.
- Using `#[cfg(feature = "bsp")]` inside `bsp_runtime` produces an `unexpected_cfgs` warning.
- The feature gate lives in `renderer` only.

### Bridge Prepare Failure Does Not Deregister
- If a bridge fails during `prepare`, the bridge remains registered for subsequent
  operations. The coordinator rolls back the staged state and clears bridge tokens,
  but does NOT deregister the failing bridge.
- Repeated `prepare()` calls will hit the same bridge failure until the app removes
  or replaces the bridge.

### ExtractedBsp Clone Required for Coordinator Error Paths
- The coordinator's `commit_with_mount` error path for missing source_link needs to
  re-stage the extracted BSP, requiring `ExtractedBsp` to implement `Clone`. It does
  (via `#[derive(Clone)]` in the `bsp` crate).

### Scene BSP Source-Link Is Raw JSON
- To avoid a `renderer → bsp_runtime` dependency, the scene stores the BSP source-link
  as `Option<serde_json::Value>` (raw JSON). The typed `BspSourceLink` lives in
  `bsp_runtime`. The coordinator serializes `BspSourceLink` to JSON and calls
  `Scene::set_bsp_source_link`.

### BridgeAggregator Fields Are Private
- `BridgeAggregator` has `pub(crate)` visibility for the struct but private fields.
  Helper methods like `len()` and `has_bridges()` must be defined in the same module
  (`bridge.rs`), not in `coordinator.rs`.

### EntityIdentity Public API
- `EntityIdentity` has `entity_index`, `source` (IdentitySource enum), `has_stable_uuid`,
  `duplicate_ordinal`.
- UUID access: match `id.source` for `IdentitySource::TrenchbroomUuid(ref uuid)`.
- There is no `.uuid` field or `.fingerprint_matches()` method on EntityIdentity.
- The `reconcile_identities` function in `bsp::identity` does full identity reconciliation
  between old and new entity sets.

### Phase 07 Validator Repairs
- `BspCoordinator::commit_with_mount` must reject commits whose generation token has not
  completed `validate`; generation equality alone is not enough to enforce the
  prepare → validate → commit transaction.
- Bridge panics must be caught. Commit panics poison the coordinator before scene publication;
  rollback panics poison the coordinator after staged cleanup is attempted.
- Scene source-link persistence is raw JSON owned by `Scene`; the coordinator must serialize
  `BspSourceLink` into `Scene::set_bsp_source_link` during commit and clear it on unload.
- New prepares must roll back any previous staged bridge tokens before replacing staged DTOs,
  otherwise superseded prepares leak app-owned resources.
- BSP point-light publication needs preflight capacity/finite-value checks before scene mutation;
  warning and continuing after `create_point_light` failure creates partial publication.
- Generated BSP strict-reload validation must assert `world.diagnostics.is_empty()`, not just zero
  error-severity diagnostics; warnings or info diagnostics still violate the Phase 07 pipeline acceptance proof.

## Phase 02 Implementation Gotchas

### package_io Budget Atomicity
- All `BudgetLedger` reservation methods must check the aggregate package limit BEFORE mutating the per-category accumulator. Checking after mutation creates partial state on failure.
- `check_aggregate_tentative()` computes the tentative total with the candidate delta before any field is updated.
- File count and image pixels do NOT contribute to the aggregate byte total.
- `PackageResolver` must not reserve file count/source bytes until file read and optional hash verification have succeeded. Failed reads, metadata drift, source-byte budget failure, and hash mismatch must leave the resolver ledger unchanged.
- Package resource identity is SHA-256 so loaded `ContentIdentity` values can be compared directly with manifest content hashes.

### Compiled Fixture Parse Coverage
- `cargo test -p bsp` must parse the checked-in compiled BSP29 and BSP2 fixtures, not only programmatic minimal BSPs. The fixture tests caught stale parser assumptions: BSP29 nodes are 24 bytes, ericw-tools BSP2 nodes/leaves are 44 bytes, and BSP2 leaf bounds are i32 triplets.
- Zero-luxel colored fixtures may need a deterministic empty `QLIT` v1 `.lit` companion (`QLIT` + LE version 1, no payload) so package companion loading remains reproducible until visible-face lightmap fixtures exist.
- Visible compiler-evidence fixtures must be unmodified compiler outputs. Do not patch BSP lightdata/texinfo or synthesize nonempty `.lit` payloads to satisfy proof cells; instead fix the source `.map` and fail the build when a required nonempty `.lit` is missing or header-only.
- ericw-tools `light` produced nondeterministic `.bsp`/`.lit` bytes for the visible BSP2 fixture with automatic threading. Pin evidence profiles to `light -threads 1 -lit`; duplicate `engine_pack compile-bsp --wad` runs then produce byte-identical outputs matching checked-in fixture hashes.

### Percent-Encoded Traversal Detection
- Percent-encoded `..` must be rejected BEFORE lexical normalization. Otherwise `%2e%2e%2f` decodes to `../` which then gets lexically resolved.
- The correct approach: if `raw != decoded` (percent sequences were present), scan the decoded form for `..` components and reject them.
- Explicit (non-encoded) `..` in the raw path IS handled by lexical resolution — only encoded escapes are rejected.

### Windows Drive Detection on Linux
- `Path::new("C:/Windows").components()` on Linux yields `[Normal("C:"), Normal("Windows")]` — no `Prefix` component.
- Must add explicit Windows drive-letter detection (`[A-Za-z]:`) to catch these paths on non-Windows platforms.

### `AuthorizedBytes` Visibility
- `AuthorizedBytes::new` is `pub(crate)` to prevent external crates from bypassing the trust boundary. External crate tests must use `PackageResolver::resolve` to obtain authorized bytes.

### Integration Test vs Lib Test Discrepancies
- When integration tests and lib tests diverge on the same code, check for stale build artifacts. Use `cargo clean` before retrying.

## Phase 03 Implementation Gotchas

### BspExtractionRequest Takes Ownership
- `extract()` takes `BspExtractionRequest` which owns `world: BspWorld`. The world is moved, not borrowed. Callers must not use `world` after extraction.
- `BspExtractionRequest` implements `Default` with an empty world, identity palette, scale 0.0254, fullbright 224-255, and dev-mode strictness.
- External callers using `extract(&world, scale)` must migrate to `extract(BspExtractionRequest { world, scale: scale.unwrap_or(0.0254), ..Default::default() })`.

### SurfaceClass::Fullbright Removed
- `SurfaceClass::Fullbright` variant is gone. Fullbright is now a per-pixel mask (`fullbright_mask: Vec<u8>`) attached to `BspMaterial` and `ExtractedTexture`.
- Renderer code that matches on `Fullbright` must be updated. The renderer's own `BspSurfaceClass::Fullbright` variant remains for pipeline selection (maps to dedicated fullbright pipeline).
- The `face_materials` vector (1:1 with faces) replaces the old `surface_classes` + `material_identities` split.

### Texture Extraction Pipeline
- `resources::resolve_extracted_texture()` implements the approved precedence: embedded miptex → WAD lookup → diagnostic fallback.
- `wad::decode_miptex_pixels()` validates mip-0 offset, dimensions, and pixel count before decoding. Returns `MiptexCorrupt` diagnostic on failure.
- `read_embedded_miptex_entry()` reads the BSP miptex lump header (count + offset table + entries) to locate a specific entry by index.
- Palette index 255 is NOT globally transparent — alpha is determined by surface classification.

### Animation Frame Validation
- `validate_animation_dimensions()` checks all frames have consistent dimensions. Mismatch produces `AnimationDimensionMismatch` diagnostic.
- Animation frames are sorted by base name then frame index. Animation detection uses `materials::detect_animation()`.
- The borrow checker requires collecting animation updates before mutating textures — mutably iterating textures while looking them up by identity causes E0502.

### Atlas Allocation
- `LightmapAtlas::allocate_face()` returns `AtlasPageOverflow` diagnostic when page budget is exceeded, not a fatal error.
- Atlas pages grow dynamically up to `max_atlas_pages` (default 4). Page allocation failure is diagnosed per-face.
- Multi-style lightmap layers are registered via `atlas.add_style()` but actual per-style decoding is deferred until fixtures with multi-style data exist.

### Entity Model Reference Bounds
- `build_entity_descriptors()` bounds-checks `*N` model references: if `N == 0 || N >= models.len()`, `model_ref` is `None`.
- Model 0 is worldspawn and is never an inline model reference.

### Collision Recipe Extraction
- Brush entities use hull 1 (player hull) for collision. `headnode[1]` is used; if negative, the entity has no collision hull.
- `ConvexReconstructionFailed` diagnostic is emitted for entities with degenerate clipnode geometry. Trigger entities always get a recipe even with no collision pieces.
- World collision planes are built from the entire clipnode tree starting at node 0.

### Extraction Invariants
- `validate_extraction_invariants()` checks 7 conditions before returning:
  1. Parallel face vectors (geometries, materials, layouts) all match `num_faces`
  2. Render batch face indices are in bounds
  3. Render batch model indices exist in inline_models
  4. Inline model face indices are in bounds
  5. Collision recipe entity indices are valid
  6. Atlas pages exist if there are lightmap layouts
  7. No unexpected fatal errors in diagnostics
- Invariant violations return `Err(BspReport)` with `ExtractionInvariantViolation` code.

## Phase 03 Authorized Import Validation (2026-07-26)

- `BspWorld::content_hash` is an internal deterministic extraction fingerprint, not the resolver-issued SHA-256. A source link built from `AuthorizedBspImport` must persist `import.bsp.identity`, or it falsely labels a non-SHA value as `sha256:` and breaks durable resource identity.
- A direct BSP launch must reject a missing `--strict`/`--development` selection during CLI parsing; delaying that check until after companion handling can report an unrelated missing-resource error instead of the policy error.
- Generated dungeon artifacts can authorize every strict BSP/palette/WAD/`.lit`/PBR input successfully and still fail strict extraction with `MissingRequiredLightmap`. Keep development-mode fixture tests explicitly tagged as diagnostic-route coverage; do not weaken strict policy to conceal the artifact defect. Track this as GitHub issue #58 and `.internal-dev/bugs/bsp-generated-strict-missing-lightmap/report.md`.

## Phase 04/05 Publication and Mount Ownership Audit (2026-07-26)

- A canonical `engine_pack` BSP manifest must exclude itself and the staging marker from `published_artifacts`; validate the final staged bytes separately, then reject every other undeclared regular file. Record payload byte sizes as well as SHA-256 values.
- PBR selection is a closure, not a texture-inventory copy: derive identities from face → texinfo → source miptex slot, validate selected PNG envelope/CRC/dimensions against base miptex dimensions, and write the selected normal/gloss identities into the manifest. Missing selected companions remain legacy fallback.
- Do not serialize raw host paths in reproducible compiler provenance. Hash the sorted minimized compiler environment and use input basenames plus hashes.
- Orphan staging recovery is only safe for a direct sibling with both the destination-bound marker and a recursively inspectable regular-file/directory tree. A symlinked or mismatched lookalike must be retained.
- **Do not treat `BspCoordinator::retired_mount_count` / `retirement_diagnostics()` as evidence of renderer retirement.** Phase 05 revalidation made `PreparedBspMount` move-only, moved it directly into `Scene`, and routes stale/duplicate/cancelled leases through an opaque scene-detachment receipt. Phase 07's `SceneWorld::retire_bsp_mount()` only removes submission state, however: it does not accept the lease in a renderer/core retirement queue, invalidate cache handles, or observe fences. The renderer lifecycle gap remains GitHub Issue #59. A committed `AppBridge` token is also consumed with no active teardown receipt, so generic bridge unload/replacement remains blocked by GitHub Issue #60.

## Phase 09 — Dungeon Generator Contract Freeze (2026-07-24)

### Generator Authorization Gate
- The generator is **not** authorized by Phase 09. This phase freezes contracts and closes evidence records.
- Next sprint must be a dedicated dungeon generator implementation reading `bsp-dungeon-generation.md` as its principal contract.
- Frozen values (M1/M2 bounds, construction parameters, output ceilings, support corpus) may not be tuned in response to generator results.

### BSP2-Only Output
- Generated dungeon `.map` output is BSP2 only. BSP29 has structural limits that M2 may exceed and has no compiler-produced face-visible fixture.
- Compiler profile: `qbsp -bsp2`, `vis` (default), `light -threads 1 -lit` (deterministic).
- This is the same profile that produced `dungeon-evidence-bsp2.bsp` (41 faces, reproducible).

### Open Arches Only
- Doors excluded from generator beta. 11 door evidence cells are optional and non-blocking.
- All room connections are open arches — no moving geometry, trigger/target wiring, or runtime behavior.
- Future door support requires door evidence cells to pass first.

### Navigation Model
- Point-trace movement against compiler-preexpanded hull 1 clipnodes.
- Corridor minimums: 64 Quake units wide, 80 Quake units tall (conservative for both FGD and symmetric hull interpretations).
- FGD vs symmetric hull dispute unresolved (NAV-HULL-THRESHOLD NOT-RUN).

### Support Corpus
- 12 configurations: 8 nominal seeds (0, 1, 2, 3, 17, 255, 0x5555555555555555, u64::MAX) + 4 boundary configs (M1 min/max, M2 min/max).
- All frozen and unexecuted. Execution requires a generator first.

### Evidence Matrix Final State
- 9 cells PASS (4 compiler + 5 publication). 71 cells NOT-RUN.
- 60 cells block generator authorization and need generator-produced fixtures.
- Zero cells FAIL or BLOCKED.

### Not-Yet-Proven Gates (Carried Forward)
1. Theme licensing — no production theme selected
2. Production KB3D conversion — no pipeline/assets
3. Reference-renderer calibration — no project-owned fixture
4. Support-corpus execution — frozen, unexecuted
5. Generator determinism — no generator exists
6. Live-WSI acceptance — resize/minimize/surface-loss unrun
7. M1/M2 budget evidence — microfixtures and 41-face fixture do not prove budgets
8. ~~BSP2 face-visible output~~ — CLEARED (dungeon-evidence-bsp2.bsp)
9. ~~Nonempty QLIT v1 .lit~~ — CLEARED (dungeon-evidence-bsp2.lit)
10. BSP29 as generated-content support — N/A (all output is BSP2)

## Phase 09 Implementation Gotchas

### App-Owned Loop Pattern
- The bsp_beta app follows the dungeon_dogfood/voxel_demo pattern: app owns input, camera,
  event loop; renderer handles rendering.
- `RendererConfig` requires at minimum `app_name`; all other fields have sensible defaults
  via `..RendererConfig::default()`.
- The `FPSController` needs `update_from_snapshot(snapshot, dt, &mut camera)` with a
  fixed dt (1.0/60.0) for consistent camera feel.

### Register Bridges Before Prepare
- `BspCoordinator::register_bridge(name, bridge)` stores the bridge but does not retroactively call `prepare` on it.
- Register bridges before the candidate prepare whenever their hidden resources are part of the mount. Re-preparing after late registration intentionally creates a new generation and cancels the previous candidate.

### PreparedBspMount Uses Real Upload Path
- The current renderer path uses `Renderer::prepare_bsp_mount()` / `PreparedBspMount::upload_from_extracted()` to upload face meshes, lightmap atlas data, and BSP material descriptors.
- Non-rendered faces may still carry `MeshHandle::new(0, 0)`, but that is a nodraw/invalid-face sentinel, not a stub publication path. Do not describe compilation of `capture_bsp_beta` as visual proof; raw visible-face capture evidence is still required.

### CLI Parsing Is Space-Separated Only
- `--bsp <path>` with a space separator is required. No equals form (`--bsp=<path>`) is
  supported, matching the pattern in dungeon_dogfood.
- Unknown flags cause immediate exit with usage text.

### EnhancedV3 Live Generation Ownership
- `engine_pack` publication is atomic and no-replace. A live regeneration worker must assign a distinct confined package directory to every request; reusing one output directory makes the first changed config fail as a late collision.
- Strict generated imports must explicitly authorize the package's `.lit`, WAD, palette, and texture closure. Sibling discovery is not a substitute for the authorized resource record.
- `BspCoordinator::commit` consumes the staged candidate. Capture spawn and app-owned inline-model/classname/source-model data before commit, then apply it only after successful publication.
- A failed renderer upload or scene validation must call coordinator rollback before draining retirement receipts, otherwise the ready candidate lease remains staged beside the old world.
- Background completions need a queue rather than a single overwrite slot: every stale successful result owns a package directory that must be observed and removed, while only the latest request ID may mount.

### Checked-In Beta Core Is a Structural Microfixture
- The earlier placeholder note is obsolete: `apps/bsp_beta/assets/maps/beta_core.bsp` is now a valid 23,500-byte structural BSP microfixture and `beta_core.lit` is a valid header-only 8-byte QLIT companion. Runtime MCP validation parses and mounts it successfully.
- `beta_core.bsp` has zero renderable faces, so its uniform headless capture proves transport/readback only; it is not visible-geometry evidence.
- The `project_palette.lmp` is a legitimate CC0 palette copied from `src/bsp/tests/fixtures/palettes/`.

### scene_sync Is Snapshot-Driven (Phase 07 Dynamic Content Snapshot)
- `apps/bsp_beta::scene_sync::sync_snapshot_to_scene()` now publishes inline-model batch transforms and bounds from `BspSimulationSnapshot` every snapshot, then applies mapped external/inline scene-node transforms when node mappings exist.
- BSP frame values must be written every snapshot, not only on pose/style changes, because liquid time changes independently.
- `sync_snapshot_to_physics()` delegates to `PhysicsBridge::sync_from_snapshot()`, which updates full kinematic poses from the snapshot transform rather than translation-only state.

### Capture Example Uses BSP Feature Gate
- `capture_bsp_beta.rs` depends on `bsp_runtime`, which requires the `bsp` feature on
  the renderer crate. The example compiles only when `--features bsp` is active.
- This is consistent with the existing BSP feature gate pattern.

### Dynamic Entity Rendering Uses BSP Mount Per-Batch Transforms
- Moving inline brush entities render through the BSP mount's per-batch `model_index` transform map, not ordinary scene nodes. Scene-node mappings are still used for external model instances and any explicitly created inline nodes.
- Renderer BSP frame submissions carry snapshot-derived light-style intensities and liquid time into the BSP frame-values UBO; do not reintroduce submit-serial synthetic animation values for snapshot-driven paths.

### Phase 09 Validator Repairs
- `capture_bsp_beta` must not import `bsp_runtime` from the renderer package. The renderer crate does
  not depend on `bsp_runtime`; the capture example compiles with `--features bsp` by loading and
  extracting through the optional `bsp` dependency, then calling the renderer BSP upload path.
- `bsp_beta` CLI tests should call the fallible parser directly. Struct construction tests do not prove
  flag parsing, equals-form rejection, missing-value handling, or finite-scale validation.
- The maintained app-owned loop proof should use `begin_app_frame`/`end_app_frame`, caller-owned
  `InputSystem`, `InputActionEventEmitter`, `EventBus`, `FrameClock`, `FixedStepClock`, `Camera`, and
  `CameraView` rather than only calling `InputSystem::dispatch_frame()` before render.
- The load-query-physics-behavior-reload startup proof can run without a live renderer by parsing BSP bytes,
  querying `point_contents`, preparing/committing app-owned `PhysicsBridge` and `RuntimeBridge` state, and
  using a temporary `Scene` plus `BspCoordinator` for commit/reload/unload. Runtime execution requires valid
  compiled BSP bytes; the current checked-in `beta_core.bsp` is a valid zero-face structural microfixture.
- Final integration found the root `.gitignore` still ignored `apps/bsp_beta/` via `apps/*`; unignore the app directory (including nested `assets/`) before relying on `git status`/final diff for BSP beta app files.
- Runtime visual evidence commands must carry the same palette/WAD companions used during compile/load. Do not print or validate marker commands that rely on stale implicit discovery; copy `palette.lmp` and the generated-theme WAD into the capture directory and pass explicit `--palette`, `--companion-dir`, and `--wad` paths. An explicit missing `--wad` is a failure, not permission to silently render with unresolved textures.

### MCP Stdio Must Stay Protocol-Clean
- MCP mode reserves stdout for newline-delimited JSON-RPC responses. Renderer startup previously used direct `println!` calls for scene descriptor setup and Vulkan validation callbacks; those must remain logger calls so diagnostics go to stderr instead of corrupting the protocol stream.
- Coordinator commit consumes the staged extraction. Retain only the neutral visibility records, coordinate transform, source/render counts, and BSP byte size needed by `get_info` and `point_contents` before publication.
- Frame capture readback is asynchronous even though the MCP `capture` call is synchronous from the client's perspective. Queue one draw-target capture, pump a bounded number of headless frames, match status by output path to avoid stale results, and return only after `Succeeded` reports the actual dimensions.

## `start.bsp` Rendering Repair Gotchas (2026-07-24)

### Capture Sequences Must Be Verified Bytewise
- The five originally supplied frame captures were visually reported as changing but were byte-identical (`9622a8a9…`). Do not infer temporal instability from filenames or viewing order; hash and pixel-diff every frame first.
- The repaired five-frame sequence at frames 6/8/10/12/14 is also intentionally byte-identical (`952ed76d…`), which is valid stability evidence because the headless camera and simulation state are frozen.

### Never Substitute the Project Test Palette for Game Content
- `bsp_beta` previously fell back silently to the CC0 synthetic fixture palette. That made correctly decoded Quake indices appear blue/noisy and looked like upload corruption.
- Production palette lookup must resolve explicit bytes or the map/game-root `gfx/palette.lmp` and fail when absent. A fixture palette is authorized only for its fixture.
- Original Quake palettes remain third-party local validation inputs and must not be checked into project fixtures.

### Quake Lightmap Extents Are Grid-Snapped, Not Vector-Length-Normalized
- Project each vertex with texinfo to source texture texels, independently floor the minimum and ceil the maximum to the 16-texel lightmap grid, then compute `(max - min) / 16 + 1` samples.
- Texinfo vector length is already embodied in the projection. Dividing projected extents by vector length changes the expected face byte count and shifts all later style/lightmap reads. On `start.bsp`, the old formula mismatched 23,472 of 57,595 faces.
- UV1 should target half-luxel centers inside the snapped rectangle; edge UVs on integer boundaries can sample atlas padding/neighbor rectangles.

### PVS Uses Model-0 Visleaf Space
- The decompressed row width is `models[0].visleafs`, not the full leaf-lump length.
- Raw BSP leaf 0 is reserved solid and omitted from PVS: raw leaf `n >= 1` maps to PVS bit `n - 1`. Face memberships and imported-light leaf IDs must use that bit space before intersection tests.
- Do not write `(leaf_index > 0).then_some(leaf_index - 1)`: `then_some` evaluates eagerly and underflows for reserved leaf 0 in debug builds. Use lazy `(leaf_index > 0).then(|| leaf_index - 1)`.
- Leaves outside model 0's visibility range and empty membership signatures must remain conservative-visible. After the repair, fixed-camera PVS-on and forced-PVS-off captures were byte-identical.

### Utility Suffixes and Palette Fullbrights Can Masquerade as White Geometry
- Tool textures such as `*waterskip`, `*slimeskip`, and `*lavaskip` must be classified as hidden before the generic `*water`/`*slime`/`*lava` rules.
- `{...}` textures use palette index 255 for alpha testing; clear both alpha and emissive mask for those pixels, but do not make index 255 globally transparent.
- A fullbright mask is scalar coverage, not RGB. Multiply it by sampled albedo to preserve palette-authored emission color; adding `vec3(mask)` bleaches lava and lamps white.

### External PBR Companions Preserve the BSP Descriptor ABI
- Discover companions from sanitized miptex identities as `<texture>_norm.png` / `<texture>_gloss.png`; either file opts eligible opaque/alpha-mask materials into PBR.
- Keep binding 1 as one packed UNORM texture: R fullbright, G/B tangent normal X/Y, A gloss. For legacy materials, preserve the prior `[mask, mask, mask, 255]` bytes exactly so old shaders and captures do not change.
- Read PNG dimensions before full decode and apply decoder width/height limits equal to the base texture during CPU upload planning, before any GPU allocation. A malformed optional file should fail the mount rather than silently produce a partially different material.
- Baked lightmaps remain legacy diffuse-light modulation. Normal/gloss drive only the dielectric specular-IBL branch (`roughness = 1 - gloss`); missing channels use flat-normal/fully-rough defaults.
- BSP albedo and packed material-data images must allocate matching complete mip chains through 1×1. Preflight budgets all levels; a 1024² image has 11 levels and 1,398,101 texels.
- Hash normal and gloss bytes independently into sorted cache companion identities. Otherwise editing a map can leave a stale prepared material under the same BSP content hash.
- `glslc` without an explicit `--target-env` reproduces the repository's checked-in BSP SPIR-V; adding `--target-env=vulkan1.3` produces valid but byte-different output.

### BSP Fragment Output Must Match the Renderer Color Pipeline
- ericw/Quake lightmap atlas samples are legacy UNORM modulation bytes: sum style-weighted normalized samples, apply 2× overbright, and multiply albedo. Do not apply `pow(encoded, 2.2)` or baked `/ PI`; those physical-light operations severely darken the classic calibrated range.
- A missing baked-light layout uses neutral modulation and must not sample an unrelated atlas layer. Do not compensate with a global albedo multiplier.
- Opaque, liquid, and sky shaders must use the shared exposure/tone-map/gamma output path. Writing raw linear HDR directly makes capture brightness incomparable with the main renderer.

### Phase 10: Immutable Draw-Identity Batching (2026-07-26)
- **Batch grouping key**: `BatchKey` now groups by `(render_class, material_identity, lightmap_page, model_index)` — immutable draw identity, not exact leaf signature. This is the authoritative mapping owned by neutral extraction.
- **Leaf signature**: Is now an output field on `RenderBatch` computed as the sorted unique union of member face leaf indices. PVS culling checks this union: empty/missing = conservative visible; any leaf visible = batch visible.
- **No renderer regrouping**: `plan_bsp_upload` materializes neutral batches one-for-one. The renderer no longer regroups faces by leaf bucket (`PlannedBatchKey`, `choose_leaf_bucket_span`, `grouped_faces`, `batch_key` removed).
- **Batch count**: Faces with identical material/texture/page/model now merge into one batch regardless of leaf membership. For M1 maps with ~4 unique materials, expected batch count drops from ~183 (one per leaf cluster) to ~4-12 (one per material per page).
- **Invariant**: `merge_batch_mesh` still validates all faces in a batch share the same material plan index. A batch mixing materials or models is a preflight failure.
- **Key constraint**: `material_identity` is computed from `(texture_index << 32 | style_mask)`. If two faces share a texture but diverge in fullbright-mask or PBR-companion status (captured by `PlannedMaterialKey` not by `material_identity`), the renderer's material homogeneity check catches this at preflight.

### Generated-Dungeon Rendering Fallback and First Material Slot (2026-07-26)

- A renderable miptex slot that is structurally valid but unresolved in the authorized WAD closure must map to the shared diagnostic checkerboard in every import mode. Do not leave `slot_to_texture` empty: a concrete fallback material is required for batching, upload, and draw submission.
- A visible face with no lightmap offset must use `FallbackMissingLightmap` and render its resolved material with neutral light modulation in every import mode. This visual fallback is not compiler-publication, lighting-calibration, or release evidence; malformed lightmap byte ranges and other structural data defects still fail closed.
- The first `BspSurfaceCache::add` allocation is `BspMaterialHandle { slot: 0, generation: 0 }`. It is valid when contained in `BspResourceLease`; do not introduce a synthetic null sentinel check. Lease membership and the authoritative cache lookup establish liveness. GitHub #63 was closed after this regression was tested.
- The exact `baseline-artifact.bsp` strict headless mount now reaches 411 faces, six batches, three PBR textures, six recorded draws, and a successful 1920×1080 capture at `.internal-dev/captures/bsp-beta/headless-1426186/bsp_beta_frame_0000.png`. This is a focused visual unblock, not all-corpus acceptance.
- The fresh canonical scale-`0.25` seed-0 artifact has 903 valid lightmapped faces and 221 fallback faces, so the older blanket “missing authored baked-light data” diagnosis does not describe the current artifact. Remaining broad blockers must be evaluated against fresh fingerprinted outputs rather than stale caches.

### Windowed Renderer Teardown Must Drop ImGui Before the Device
- The patched `imgui-rs-vulkan-renderer` owns Vulkan pipelines, descriptors, textures, and an allocator and destroys them from `Drop`. Calling its explicit `destroy()` and then letting the field drop causes double destruction; leaving the field until after `ash::Device::destroy_device` calls through a dead dispatch table and can SIGSEGV.
- `VkRenderCore::drop` must `take()` and drop `VkImgui` exactly once after `device_wait_idle` and before any logical-device destruction. On terminal device loss, continue to forget it because no Vulkan-calling destructor is safe.
- Reproduce lifecycle fixes with a real window close, not only timeout termination: the 2026-07-27 X11/RADV close run handled `WM_DELETE_WINDOW` and exited status 0 after the repair.
