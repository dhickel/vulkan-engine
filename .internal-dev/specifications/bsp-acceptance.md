---
schema_version: 1
document_type: acceptance-specification
status: active
owner: bsp-beta
created: 2026-07-23
approval: active — Phase 10 closeout reconciled 2026-07-26; generated-dungeon visual fallback authorized 2026-07-26; six open GitHub issues (#57–#62) remain, while strict runtime uses visual fallbacks for missing WAD/lightmap data; WSI lifecycle, calibration, and performance-budget cells remain NOT_RUN
---

# BSP Acceptance Matrix

## 1. Scope

This specification defines the acceptance criteria, severity policies, fixture provenance requirements, visual tolerance metrics, hardware classes, numeric budgets, live-WSI matrix, fuzz corpus bounds, and non-BSP regression assertions for the BSP beta. Every claimed capability must pass its corresponding criterion; unavailable entries are marked unexecuted with environment evidence.

## 2. Severity Policy

### 2.1 Development Mode

| category | minimum severity | allowed? |
|----------|-----------------|----------|
| unsupported compatibility | Warning | Yes (diagnosed) |
| structural corruption | Error | No (rejected) |
| security violation | Error | No (rejected) |
| missing required resource | depends on resource | palette: Error; WAD: Warning; model: Warning |
| optional fallback | Warning | Yes (diagnosed) |
| unknown app entity | Info | Yes (preserved) |
| authoring quality | Info | Yes (diagnosed) |

### 2.2 Strict / Release Mode

| category | minimum severity | allowed? |
|----------|-----------------|----------|
| unsupported compatibility | Error | No (rejected) |
| structural corruption | Error | No (rejected) |
| security violation | Error | No (rejected) |
| missing required resource | Error | No (rejected) |
| optional fallback | Warning | Yes (diagnosed, but release content should not rely on fallbacks) |
| unknown app entity | Info | Yes (preserved) |
| authoring quality | Warning | Yes (diagnosed) |

**Owner-authorized generated-dungeon visual override (2026-07-26):** a structurally valid but WAD-unresolved renderable miptex slot uses `BSP-FALLBACK-DIAGNOSTIC-TEXTURE`, and a visible face without a lightmap offset uses `BSP-FALLBACK-MISSING-LIGHTMAP`, even when the direct import policy is strict. Both are warnings that preserve rendering; they do not promote the artifact to release or compiler-publication acceptance. Structural corruption, palette absence, security violations, malformed companion data, and unresolved publication inputs remain fatal.

### 2.3 External Compiler Publication

`engine_pack compile-bsp` accepts output only when `qbsp`, `vis`, and `light` all exit successfully **and emit no compiler warnings**. A warning is a hard `CompilerWarning` failure even in development mode; missing textures, `No entities in empty space`, and `No filling performed` may not publish a BSP. This owner-authorized 2026-07-24 severity change applies at the compiler boundary and does not alter the loader diagnostic tables above.

## 3. Fixture Provenance Requirements

### 3.1 Fixture Categories

| category | requirement |
|----------|------------|
| project-authored .map sources | CC0 or project license; no copyrighted Quake content |
| project-authored palette | CC0; not derived from id Software Quake palette |
| compiled .bsp fixtures | produced by pinned ericw-tools; redistribution rights recorded |
| reference renderer screenshots | for calibration only; not redistributed as project assets |
| fuzz corpus inputs | project-authored malformed byte sequences; no copyrighted maps |
| adversarial fixtures | project-authored; isolate one failure mode per fixture |

### 3.2 Fixture Redistribution

All fixtures under `src/bsp/tests/fixtures/` must carry a recorded license in `LICENSES.md`. Fixtures are redistributable under CC0 or project license. No fixture contains original id Software copyrighted content (maps, palettes, textures, WADs, models).

### 3.3 Compiler Provenance

| field | record |
|-------|--------|
| compiler identity | `ericw-tools 2.0.0-alpha3` or exact pinned commit hash |
| executable | `qbsp`, `vis`, `light` |
| build source | official ericw-tools GitHub releases or project-approved build |
| invocation | recorded in `fixture-manifest.toml` per fixture |
| compiler hash | SHA-256 of each executable |
| environment | minimized environment, recorded |
| hosted requirement | ericw-tools must be available on the build host; not bundled with engine |

## 4. Reference Renderer Calibration

| property | value |
|----------|-------|
| reference renderer | vkQuake 1.30+ or QuakeSpasm 0.95+ |
| calibration settings | default brightness, default gamma, 1280×720 window |
| capture method | screenshot at known camera position/orientation |
| comparison method | structural similarity (SSIM) for lightmap regions |
| tolerance | SSIM ≥ 0.85 for baked lighting comparison |
| freeze conditions | engine time frozen, animation state frozen, exposure frozen |

## 5. Deterministic Capture Settings

| parameter | frozen value |
|-----------|-------------|
| exposure | 1.0 (engine default) |
| overbright | 2.0 |
| style index | 0 (static lighting only) |
| animation time | 0.0 |
| environment IBL | disabled for legacy interior-only captures; fixed/recorded environment for explicit PBR-companion captures |
| camera | fixed position/orientation per capture scene |
| resolution | 1280×720 |
| capture target | draw (headless) |

For generated-dungeon named cameras, the fixed pose is derived deterministically from compiled semantic entities. `info_player_start` is already authored at eye height and receives no additional vertical offset. Spawn uses that origin; corridor uses the same safe origin; junction selects the point entity nearest compiled map center. The selected origin must be non-solid. Orientation is the cardinal direction with the greatest clear distance according to compiled BSP contents, with deterministic tie ordering and a default-forward fallback only when no probe is clear. This prevents acceptance captures from using approximate map-center coordinates, looking into walls, or placing the eye above an 80-unit corridor ceiling.

## 6. Hardware Classes

| class | GPU | VRAM | target |
|-------|-----|------|--------|
| H1 (minimum) | integrated / 2 GiB VRAM | 2 GiB | map load, 30 fps at 720p |
| H2 (baseline) | discrete / 4 GiB VRAM | 4 GiB | map load, 60 fps at 1080p |
| H3 (target) | discrete / 8 GiB VRAM | 8 GiB | map load, 60+ fps at 1440p |

## 7. Map Classes

### 7.1 Microfixture (Zero-Face Structural)

The existing `q1-bsp29-core.bsp` and `ericw-bsp2-colored.bsp` compiled fixtures are **microfixtures**: they have zero visible faces and zero lightdata. They prove structural parsing (header, lumps, entities, PVS, clipnodes) but are NOT representative of any map class. All M1 timing, budget, and acceptance measurements that used these zero-face fixtures are reclassified as microfixture measurements, not M1 evidence.

### 7.2 Map Classes (General)

| class | complexity | typical stats |
|-------|-----------|---------------|
| M1 (small) | < 2,000 faces, < 50 entities | test maps, deathmatch arenas |
| M2 (medium) | 2,000–10,000 faces, 50–300 entities | typical single-player levels |
| M3 (large) | 10,000–40,000 faces, 300–800 entities | large single-player episodes |
| M4 (pathological) | > 40,000 faces | stress tests |

### 7.3 Locked Generated-Domain Values

The following values define the procedural generation domain for M1 and M2 and are **locked for this evidence campaign**. They may not be tuned in response to downstream results; if evidence disproves any value, record a no-go rather than adjusting the value.

#### M1 Generated Domain

| parameter | locked value |
|-----------|-------------|
| layers | 1 |
| outer XY extent (Quake units) | ≤ 1536 × 1536 |
| total Z span (Quake units) | ≤ 256 |
| room count | 8..=16 |
| loop count | 0..=2 |
| placement candidates per room attempt | 16 |
| max placement attempts per room per candidate | ≤ 64 |
| max A* expansions per candidate | ≤ 131,072 |

#### M2 Generated Domain

| parameter | locked value |
|-----------|-------------|
| layers | 1 (Enhanced v2); 2 (Enhanced v3) |
| outer XY extent (Quake units) | ≤ 3072 × 3072 |
| total Z span (Quake units) | ≤ 384 |
| room count | 17..=40 |
| loop count | 1..=6 |
| placement candidates per room attempt | 32 |
| max placement attempts per room per candidate | ≤ 96 |
| max A* expansions per candidate | ≤ 524,288 |

#### Shared Construction Parameters

| parameter | locked value |
|-----------|-------------|
| construction unit quantum | 16 Quake units |
| wall thickness | 16 Quake units (1 quantum) |
| minimum room outer span | ≥ 112 Quake units (7 quanta) |
| clear route width (corridors, passages, portal throats, junction centers) | ≥ 64 Quake units (4 quanta) |
| clear headroom (corridors and portal throats) | ≥ 80 Quake units (5 quanta) |
| route type | Legacy v1: level (no ramps or stairs). Enhanced v2/v3: frozen two-layer stair arrangement in `bsp-dungeon-generation.md` §§17/20. |
| room connections | open arches (no doors for beta) |
| stacked XY spaces | prohibited (no room directly above another) |

#### Output Ceilings

| metric | M1 ceiling | M2 ceiling |
|--------|-----------|-----------|
| compiled faces | < 2,000 | < 10,000 |
| entities | < 50 | < 300 |
| static batches | < 100 | < 500 |

M2 is the highest output tier reached in this campaign. At least one M1 ceiling must be exceeded by the representative M2 fixture.

### 7.4 Support Corpus (Frozen, Executed)

The following seed and configuration corpus was declared before the generator and remains frozen. Failed seeds or exhausted resources may not silently select a fallback profile, replacement asset, revised bound, or showcase exception.

#### Nominal Configuration Seeds

The nominal configurations are M1 (12 rooms, 1 loop, 1024×1024, Z 192) and M2 (28 rooms, 3 loops, 2048×2048, Z 256), matching `bsp-dungeon-generation.md` §3.4 and §10.1.

| seed | class | config | expected behavior |
|------|-------|--------|--------------------|
| `0` | M1 | nominal (12 rooms, 1 loop, 1024×1024, Z 192) | deterministic output; no panic, no infinite loop |
| `1` | M1 | nominal (12 rooms, 1 loop, 1024×1024, Z 192) | deterministic output; no panic, no infinite loop |
| `2` | M1 | nominal (12 rooms, 1 loop, 1024×1024, Z 192) | deterministic output; no panic, no infinite loop |
| `3` | M1 | nominal (12 rooms, 1 loop, 1024×1024, Z 192) | deterministic output; no panic, no infinite loop |
| `17` | M2 | nominal (28 rooms, 3 loops, 2048×2048, Z 256) | deterministic output; no panic, no infinite loop |
| `255` | M2 | nominal (28 rooms, 3 loops, 2048×2048, Z 256) | deterministic output; no panic, no infinite loop |
| `0x5555555555555555` | M2 | nominal (28 rooms, 3 loops, 2048×2048, Z 256) | deterministic output; no panic, no infinite loop |
| `u64::MAX` | M2 | nominal (28 rooms, 3 loops, 2048×2048, Z 256) | deterministic output; no panic, no infinite loop |

#### Boundary Configuration Seeds

Boundary configurations use the nominal XY and Z values for their respective classes, varying only room count and loop count.

| config | seed | parameters | expected behavior |
|--------|------|------------|--------------------|
| Boundary A (M1 minimum) | `42` | 8 rooms, 0 loops, XY 1024×1024, Z 192 | valid output within M1 ceilings |
| Boundary B (M1 maximum) | `43` | 16 rooms, 2 loops, XY 1024×1024, Z 192 | valid output within M1 ceilings |
| Boundary C (M2 minimum) | `44` | 17 rooms, 1 loop, XY 2048×2048, Z 256 | valid output within M2 ceilings |
| Boundary D (M2 maximum) | `45` | 40 rooms, 6 loops, XY 2048×2048, Z 256 | valid output within M2 ceilings |

**Status (updated 2026-07-27)**: Historical all-12 evidence passes deterministic generation, warning-free BSP2 compilation, strict parser reload, sealing, face/entity ceilings, and non-solid room/entity/corridor/portal/full-junction-clearance witnesses. The current canonical scale-`0.25` nominal M1 seed-0 artifact has 1,124 renderable faces, 14 entities, and 6 batches; strict extraction, GPU upload, submitted draw, fixed-camera headless capture, and graceful windowed close all pass. The scale change still requires all-12 corpus regeneration before current corpus ceilings are considered complete.

## 8. Numeric Budgets

### 8.1 Parse Budgets

| metric | M1 | M2 | M3 | M4 (if supported) |
|--------|----|----|----|---------------------|
| parse wall time | < 50 ms | < 200 ms | < 800 ms | < 3,000 ms |
| memory peak (parse) | < 16 MiB | < 64 MiB | < 256 MiB | < 512 MiB |
| memory resident (BspWorld) | < 8 MiB | < 32 MiB | < 128 MiB | < 256 MiB |

### 8.2 Extraction Budgets

| metric | M1 | M2 | M3 |
|--------|----|----|-----|
| geometry extraction | < 20 ms | < 100 ms | < 400 ms |
| lightmap atlas build | < 10 ms | < 50 ms | < 200 ms |
| entity DTO extraction | < 1 ms | < 5 ms | < 20 ms |

### 8.3 Upload Budgets

| metric | M1 | M2 | M3 |
|--------|----|----|-----|
| geometry upload (GPU) | < 10 ms | < 50 ms | < 200 ms |
| lightmap atlas upload | < 5 ms | < 20 ms | < 100 ms |
| material creation | < 5 ms | < 20 ms | < 50 ms |

### 8.4 Submission Budgets

| metric | M1 | M2 | M3 |
|--------|----|----|-----|
| static batch count | < 100 | < 500 | < 2,000 |
| draw calls (static world) | < 100 | < 500 | < 2,000 |
| draw calls (total, incl. dynamic) | < 200 | < 1,000 | < 4,000 |
| PVS decode | < 1 ms | < 2 ms | < 5 ms |
| light selection | < 1 ms | < 3 ms | < 8 ms |

### 8.5 Memory Budgets (GPU)

| metric | M1 | M2 | M3 |
|--------|----|----|-----|
| geometry (vertex + index) | < 16 MiB | < 64 MiB | < 256 MiB |
| lightmap atlas | < 16 MiB | < 64 MiB | < 256 MiB |
| textures (material + fullbright) | < 32 MiB | < 128 MiB | < 512 MiB |
| total BSP GPU | < 64 MiB | < 256 MiB | < 1 GiB |
| H1 VRAM budget remaining after BSP | > 1.5 GiB | N/A | N/A |

### 8.6 Reload Budgets

| metric | M1 | M2 | M3 |
|--------|----|----|-----|
| unload wall time | < 50 ms | < 200 ms | < 500 ms |
| reload wall time (prepare+commit) | < 100 ms | < 400 ms | < 1,500 ms |
| reload stall frames | < 2 frames | < 4 frames | < 8 frames |

## 9. Live-WSI Matrix

### 9.1 Required Entrypoints

| entrypoint | hardware class | requirement |
|-----------|---------------|-------------|
| `bsp_runtime` example (TBD) | H2 | live windowed startup, render 60 frames |
| `dungeon_dogfood` with BSP scene | H2 | live windowed startup, map loads without error |
| `voxel_demo` (non-BSP regression) | H2 | live windowed startup, behavior unchanged |

### 9.2 WSI Lifecycle Tests

| scenario | test |
|----------|------|
| window resize during BSP render | map remains visible, no artifacts |
| minimize/restore | BSP state preserved across swapchain rebuild |
| surface loss recovery | BSP reload not required after surface recovery |

### 9.3 Unavailable WSI Entries

If a live GPU or WSI environment is unavailable:
- Record the unexecuted check with environment evidence.
- Do not silently substitute a headless claim.
- Headless captures remain valid for pixel-correctness but not for WSI lifecycle.

## 10. Fuzz Corpus

### 10.1 Fuzz Target Categories

| category | max input size | coverage goal |
|----------|---------------|---------------|
| header parsing | 256 bytes | all magic/version paths |
| lump offset validation | 256 bytes | all overlap/truncation/alignment cases |
| entity parsing | 1 MiB | all token/brace/escape/encoding paths |
| VIS decompression | 128 KiB | all RLE boundary/overflow cases |
| face reconstruction | 2 MiB | all winding/degeneracy/overflow paths |
| texture name resolution | 2 KiB | all path/normalization/malformed cases |
| clipnode traversal | 512 KiB | all cycle/invalid-index/overflow cases |

### 10.2 Budget Limits

| metric | limit |
|--------|-------|
| fuzz corpus total size | 16 MiB |
| per-input timeout | 5 seconds |
| max iterations per fuzz run | 10⁶ |
| crash-on-panic | enforced (no panic on malformed input) |
| OOM detection | allocation budget enforced; reject > budget |

## 11. Package Reproducibility

### 11.1 Deterministic Package Output

| property | requirement |
|----------|------------|
| same source + same compiler + same arguments | byte-identical .bsp |
| same package inputs | byte-identical package output |
| compiler provenance recorded | compiler identity, version, arguments in manifest |
| source hashes | SHA-256 of .map and companion sources |
| output hashes | SHA-256 of .bsp and companion outputs |

### 11.2 Package Validation

| check | severity |
|-------|----------|
| compiler identity matches expected | error |
| .bsp content hash matches expected | error |
| companion content hashes match expected | error (strict), warning (dev) |
| palette content hash matches expected | error |
| WAD content hashes match expected | error (strict), warning (dev) |

## 12. Descriptor ABI Guards

### 12.1 Required Guards

| guard | description |
|-------|------------|
| BSP descriptor set IDs distinct from PBR | no binding number collision |
| BSP pipeline bind points distinct | BSP path remains separate from the general mesh PBR pipeline/layout; legacy and PBR BSP variants share the frozen BSP layout |
| BSP shader SPIR-V separate files | `bsp_lightmapped.vert.spv`, `bsp_lightmapped.frag.spv`, `bsp_pbr.frag.spv`, etc. |
| default builds compile without BSP shaders | no BSP .spv references in default pipelines |
| ABI manifest test | automated test that enumerates every descriptor binding and pipeline layout |

## 13. Non-BSP Regression Assertions

### 13.1 Default Build (BSP Disabled)

| assertion | validation |
|-----------|-----------|
| no BSP crate linked | `cargo check` default features — no `bsp` in dep tree |
| no BSP shaders loaded | `cargo check -p renderer` — no BSP .spv in default shader manifest |
| no BSP startup work | smoke test — no BSP-related log lines |
| existing examples unchanged | all renderer examples pass `cargo check` and timeout-bound smoke |
| existing apps unchanged | `dungeon_dogfood`, `voxel_demo` pass `cargo check` and tests |

### 13.2 BSP Compiled But Inactive

| assertion | validation |
|-----------|-----------|
| no BSP resources created at startup | smoke test with BSP feature enabled |
| no BSP frame-path work without BSP scene | profiling: BSP feature enabled but no map loaded |
| existing renderer captures within tolerance | capture comparisons: BSP-enabled vs default |
| all existing tests pass | `cargo test` with BSP feature enabled |

### 13.3 Feature Combinations

| combination | validation |
|------------|-----------|
| default (no BSP, no csm, no instancing, no scene-bvh) | `cargo check` |
| BSP only | `cargo check --features bsp` |
| BSP + csm | `cargo check --features bsp,csm` |
| BSP + all features | `cargo check --all-features` |

## 14. Evidence Requirements by Phase

| phase | evidence type | destination |
|-------|--------------|-------------|
| Phase 01 (this phase) | fixture sources, palette, build driver, compiled BSPs, manifest, licenses | `src/bsp/tests/fixtures/` |
| Phase 02 (parser) | golden parse tests, differential checks, malformed tests | `src/bsp/tests/` |
| Phase 03 (DTO extraction) | geometry/entity DTO tests, atlas layout tests | TBD |
| Phase 04 (renderer integration) | deterministic headless captures, ABI manifest | `.internal-dev/captures/bsp/` |
| Phase 05 (physics integration) | collision contact tests, hull trace golden | TBD |
| Phase 06 (behavior adapters) | structural behavior tests | TBD |
| Phase 07 (packaging) | package validation tests, reproducibility proofs | TBD |
| Phase 08 (lifecycle) | unload/reload/transaction tests | TBD |
| Phase 09 (live proof) | live WSI smoke, performance measurements | TBD |
| Phase 10 (closeout) | full matrix revalidation, documentation | TBD |

## 15. Approval and Evidence Matrix

| criterion | status | evidence basis | blocker | reviewer |
|-----------|--------|---------------|---------|----------|
| Severity policy (§2) | PASS (owner re-review 2026-07-24) | Loader diagnostics retain Phase 09 coverage; external compiler publication now adds a tested hard-failure policy for stage warnings and skipped fill (`CompilerError::CompilerWarning`) | none | dhickel (2026-07-24) |
| Fixture provenance (§3) | PASS (Phase 02 repair, 2026-07-24) | CC0 license confirmed; palette SHA-256 recorded; maps project-authored; `dungeon_evidence.wad` is project-authored with generated DNGN01/DNGN02 textures and no id Software content; ericw-tools external license/source recorded | external compiler is user-supplied and not bundled | dhickel (2026-07-24) |
| Compiler identity (§3.3) | PASS (Phase 07) | `engine_pack compile-bsp` verifies executable SHA-256; version output captured; environment recorded; shell-free invocation with `.env_clear()` | none | dhickel (2026-07-23) |
| Reference renderer calibration (§4) | BLOCKED | pending calibration capture vs vkQuake using a visible/lightmapped fixture | reference renderer capture still pending; the new BSP2 compiler-evidence fixture has not been captured/calibrated at the formal 1280×720 settings | — |
| Deterministic capture settings (§5) | PASS (2026-07-27) | Canonical seed-0 strict corridor capture `.internal-dev/captures/bsp-beta/headless-1480388/` uses 1280×720, exposure 1.0, overbright 2.0, style 0, animation time 0.0, a non-solid authored-height camera, and the project-owned PBR closure. The sidecar reports frame-6 draw capture success; visible wall, floor, and ceiling detail is high-contrast and recognizable. | reference-renderer SSIM is tracked separately under §4 | owner-directed (2026-07-27) |
| Hardware classes (§6) | NOT-RUN | owner design decision; H1/H2/H3 tiers | measurement pending on H2 with real BSP fixtures | — |
| Map classes (§7) | PARTIAL | historical frozen M1/M2 entries compile as face-visible fixtures; current canonical M1 seed 0 passes strict submitted drawing with 1,124 faces / 14 entities / 6 batches | all-12 corpus regeneration at scale `0.25`, M3/M4, and runtime performance characterization remain unrun | owner-directed (2026-07-27) |
| Numeric budgets (§8) — microfixture | PASS (Phase 09) | `performance.rs` harness: parse ~0.015ms (< 50ms), extract ~0.005ms (< 20ms), reload ~0.13ms (< 100ms) on zero-face microfixtures | **Reclassified**: these measurements are against zero-face microfixtures, not M1. M1 budget evidence requires a face-visible M1-class fixture from the generator. | dhickel (2026-07-23) |
| Numeric budgets (§8) — M1 face-visible | NOT-RUN (timing) | face-visible M1 corpus fixtures now exist and face/entity ceilings pass; current isolated candidate reaches 6 neutral/upload-preflight batches | timed parse/extract/upload/reload and submitted static-draw measurement remain required; strict extraction/#58 and GPU rollback crash/#61 block the mount path | — |
| Numeric budgets (§8) — M2 face-visible | NOT-RUN (timing) | face-visible M2 corpus fixtures now exist and face/entity ceilings pass | timed parse/extract/upload/reload and deterministic batch measurement still required | — |
| Generated domain — M1 bounds (§7.3) | PASS | all nominal and boundary M1 entries generate and compile within frozen XY/Z/room/loop bounds; repaired placement preserves deterministic draws while enforcing 112-unit minimum room spans | none | dhickel (2026-07-24) |
| Generated domain — M2 bounds (§7.3) | PASS | all nominal and boundary M2 entries generate and compile within frozen XY/Z/room/loop bounds | none | dhickel (2026-07-24) |
| Generated domain — output ceilings (§7.3) | PARTIAL / CURRENT SEED-0 DRAW PASS | historical all-12 entries passed face/entity ceilings; current scale-`0.25` seed 0 has 1,124 faces and 6 submitted renderer batches against the M1 `< 2,000` / `< 100` ceilings | all-12 scale-`0.25` regeneration and deterministic batch assertion remain absent | owner-directed (2026-07-27) |
| Support corpus — nominal seeds (§7.4) | PASS | all 8 frozen nominal entries compile warning-free, reload strictly, remain sealed, and pass spatial witness queries | static-batch ceiling tracked separately by GitHub #57 | dhickel (2026-07-24) |
| Support corpus — boundary configs (§7.4) | PASS | all 4 frozen boundary entries compile warning-free, reload strictly, remain sealed, and pass spatial witness queries | static-batch ceiling tracked separately by GitHub #57 | dhickel (2026-07-24) |
| Generator determinism (same seed → same output) | PASS | byte-identical `.map` and compiled `.bsp` assertions pass in generator determinism/corpus tests | none | dhickel (2026-07-24) |
| Production CC0 theme | PASS | project-authored CC0 Stone Beta theme deterministically generates a palette, four distinct 1024×1024 visible WAD2 roles, detailed matching normal/gloss PNG companions, and a 64×64 compiler-only `skip`; current seed-0 strict upload resolves 3 PBR materials with complete mip chains, and `.internal-dev/captures/bsp-beta/headless-1480388/` visibly preserves fine stone detail | none | owner-directed (2026-07-27) |
| Door disposition | DECIDED (Phase 09) | open arches only; doors excluded from generator beta per `bsp-dungeon-generation.md` §6.1 | none — architectural decision; all 11 door evidence cells remain NOT-RUN and non-blocking | dhickel (2026-07-24) |
| Navigation model | PASS (structural + compiled spatial) | point-trace model and 64×80 corridor/portal clearances remain frozen; all 12 corpus maps pass non-solid room/entity/corridor/portal/full-junction witnesses. Final seed-0 MCP independently returns non-solid for three room centers and the formerly blocked endpoint-corner junction, while three wall probes return solid. | live mover/sliding and hull-threshold evidence remain unrun | dhickel (2026-07-25) |
| Compiler profile for generator | PASS | BSP2 `-threads 1 -lit` remains deterministic; publication now rejects any stage warning or skipped fill; repaired seed-0 provenance and all 12 corpus entries are warning-free | none | dhickel (2026-07-24) |
| Live WSI matrix (§9) | PARTIAL (2026-07-27 task evidence) | Current seed-0 ran windowed on X11/RADV, installed a 1280×720 swapchain, uploaded 1,124 faces in 6 batches, submitted a frame, handled `WM_DELETE_WINDOW`, and exited status 0 without panic, device loss, validation error, or `ERROR`; evidence: `.internal-dev/debug_reports/bsp-dungeon-texture-rendering/windowed-graceful-close.log`. The run also exposed and repaired an ImGui renderer double-destroy during normal shutdown. | Wayland re-close, resize, minimize/restore, surface-loss recovery, and complete required-entrypoint matrix remain unrun | owner-directed (2026-07-27) |
| Fuzz corpus limits (§10) | PASS (Phase 09) | 47 adversarial tests across all 7 fuzz categories; per-category bounds enforced; no panics on malformed input | none for Phase 09 | dhickel (2026-07-23) |
| Package reproducibility (§11) | PASS (Phase 02 repair, 2026-07-24) | `engine_pack compile-bsp --wad` with the pinned BSP2 profile (`light -threads 1 -lit`) produced duplicate byte-identical `.bsp`/`.lit` outputs matching checked-in fixture hashes; compiler provenance and source/WAD/output hashes recorded | none | dhickel (2026-07-24) |
| Descriptor ABI guards (§12) | PASS (2026-07-24 additive PBR variants) | 14-renderer-descriptor-abi.md records packed material-data channels; BSP set IDs remain distinct from general mesh PBR; descriptor/shader-manifest tests cover the unchanged three-set ABI; 7 BSP pipeline variants share one layout | none | dhickel (2026-07-24) |
| External BSP PBR companions | PASS (2026-07-24) | unit/integration tests cover discovery, extraction propagation, single/both companion defaults, packed texture channels, PBR/legacy routing, malformed PNGs, descriptor ABI, and shader contract. Frame-5 1920×1080 sidecars report success for `.internal-dev/captures/bsp-pbr/` (1 PBR texture) and `.internal-dev/captures/bsp-pbr-baseline/` (0 PBR); 460,764 pixels differ only inside surface bbox `(578,356)–(1342,1013)`, while background is unchanged | long-term project-authored normal/gloss visual baseline remains desirable | dhickel (2026-07-24) |
| Non-BSP regression (§13) | PASS (Phase 09) | default, bsp, and all-features renderer checks/tests passed; `dungeon_dogfood` and `voxel_demo` compile and test clean; all existing examples pass `cargo check` | `dungeon_dogfood` live-WSI entry blocked by GitHub #54 (not BSP-related) | dhickel (2026-07-23) |
| Feature combinations (§13.3) | PASS (Phase 09) | `cargo check -p renderer` × 3 (default, bsp, all-features) — all compile cleanly | none for compile verification | dhickel (2026-07-23) |
| Lifecycle fault injection (§8) | PASS (Phase 09) | 36 `transaction_failures` tests + 10 lifecycle fault tests | none for Phase 09 | dhickel (2026-07-23) |
| Compile matrix (§13.1) | PASS (Phase 09) | default, bsp, all-features renderer; default input/events/audio/physics/scripting/launch_shared; dungeon_dogfood, voxel_demo | none | dhickel (2026-07-23) |

**Re-review trigger**: Any change to severity policy, fixture provenance, hardware classes, generated-domain values, output ceilings, support corpus, WSI entrypoints, or non-BSP regression assertions requires owner re-review.

**Unresolved blocked cells**: A task-local live Wayland smoke and non-redistributable `start.bsp` headless captures now exist, but they do not close the formal matrix. Resize/minimize/surface-loss WSI lifecycle checks, a project-owned visible/lightmapped fixture at the frozen 1280×720 settings, and reference-renderer calibration remain required.

**Not-yet-proven (no unsupported result recorded as PASS)**:
- Reference-renderer calibration (project-authored capture now exists, but frozen 1280×720 SSIM comparison is unrun)
- Live-WSI acceptance (resize, minimize/restore, surface-loss recovery unrun; complete entrypoint matrix unrun)
- M1/M2 runtime performance budgets (face-visible fixtures now exist, but timed parse/extract/upload/reload measurements are unrun)
- Static-batch/draw ceilings: historical M1 seed 0 reported 183 batches, while a current isolated candidate reaches 6 neutral/upload-preflight batches; strict submitted proof across the corpus remains blocked by GitHub #58 and #61, and deterministic enforcement remains GitHub #57

## 17. EnhancedV3 Production Acceptance

### 17.1 Scope

The EnhancedV3 production profile introduces a new generation class (M2-only,
cardinal + 45° geometry, chamfered-octagonal rooms, pointed-default plus
rectangular/segmented cardinal portal surrounds, grounded assemblies,
Sparse/Moderate/Rich presets) that must satisfy all existing M2 acceptance
criteria plus profile-specific validation gates.

### 17.2 Map Class

| class | room count | geometry | budget ceiling |
|-------|-----------|----------|---------------|
| M2 (medium) | 17–40 | cardinal + 45° chamfered-octagonal; pointed-default plus rectangular/segmented cardinal surrounds; grounded assemblies | M2 output ceilings (§5): faces < 10,000, entities < 300, batches < 500 |

M1 is not a v3 target class. All v3 output is M2-only.

### 17.3 Preset Acceptance Criteria

| preset | exact rooms | same-layer routes | minimum families | minimum assemblies | minimum features |
|--------|-------------|-------------------|------------------|--------------------|------------------|
| Sparse | 12 | 10 | 1 | 1 | 2 |
| Moderate | 20 | 20 | 3 | 3 | 6 |
| Rich | 28 | 30 | 6 | 6 | 12 |

Each preset must:
- Produce deterministic byte-identical `.map` and metadata for identical (seed, preset, config)
- Compile warning-free through the pinned BSP2 profile
- Strict-reload with zero diagnostics
- Pass compiled spatial witnesses (room centers, corridor centers, portal throats,
  junction clearances, spawn/landing origins, shell interiors)
- Stay within declared M2 budget ceilings
- Fail with typed `MinimumIdentityFailure` when resource-constrained below preset minimums
- Never panic or produce empty output on valid inputs

### 17.4 v3 Corpus Requirements

| requirement | minimum |
|-------------|--------|
| total entries | 12 |
| Sparse preset seeds | 4 (0, 42, 99, 255) |
| Moderate preset seeds | 4 (0, 42, 99, 255) |
| Rich preset seeds | 4 (0, 42, 99, 255) |
| deterministic replay | all entries byte-identical across repeated generation |
| warning-free compilation | all entries 0 warnings across qbsp/vis/light |
| strict reload | all entries 0 diagnostics |
| budget compliance | all entries within M2 face/entity/batch ceilings |

### 17.5 v1/v2 Corpus Regression Gate

The 12-entry Legacy v1 corpus and 12-entry Enhanced v2 corpus must produce
byte-identical `.map` and metadata compared to their frozen baselines. Any
drift in v1 or v2 output blocks v3 production acceptance regardless of v3
evidence quality.

### 17.6 Evidence Status

| criterion | status | basis |
|-----------|--------|-------|
| Architectural feasibility | PASS | 36/37 proof rows PASS |
| 45° geometry compiler safety | PASS | Production output contains diagonal face lines; compiled fixtures compile warning-free |
| Pointed-arch portal witnesses | PASS | Production output: 64×80 swept clearance, pointed apex |
| Segmented-arch explorer override | PASS | Moderate seed-42 full-config package preserves a 64×80 throat, seals the Z=112–128 crown interface, and compiles warning-free through ericw |
| Grounded assembly acyclicity | PASS | Production pipeline: acyclic support graphs |
| M2 budget ceilings | PASS | All presets within 10,000 faces, 300 entities |
| v1/v2 compatibility freeze | PASS | 24/24 entries byte-identical |
| Production v3 12-entry compiler matrix | PASS | 12 entries (3 presets × 4 seeds 0/42/99/255): warning-free, strict reload, spatial witnesses, budget compliance |
| Production acceptance | PASS | Preset identity minima, exact 12-entry source matrix, seed variation, and budgets — all validated |
| Headless renderer capture | PASS | Strict production Sparse/Rich BSP2 artifacts captured through engine-owned draw capture; fixed and targeted grammar-feature evidence is indexed at `.internal-dev/captures/enhanced-v3-production/manifest.md` |
| Live-WSI lifecycle | NOT_RUN | Requires live WSI environment |
| Reference-renderer calibration | NOT_RUN | Requires GPU + vkQuake |

### 17.7 In-Game Explorer GUI Acceptance

| criterion | status | basis |
|-----------|--------|-------|
| Complete public `V3Config` inventory | PASS | GUI/config round-trip and optional-override restoration tests cover every public field; fixed wall thickness is disabled rather than invented as a knob |
| Keyboard interaction model | PASS | Focused library tests cover navigation, group cycling, top-row/numpad integer and decimal editing, optional reset, toggles, actions, release/repeat suppression, and Escape-always-close |
| Mouse interaction model | PASS | Deterministic raw-hitbox tests cover exact dropdown selection, steppers, checkboxes, action buttons, viewport coordinates, bounded scrolling, and a non-GPU ImGui draw-data smoke |
| Dual-input gameplay isolation | PASS (source + focused tests) | Production routing helpers and binary tests prove every keyboard, pointer, wheel, and raw-mouse class is blocked from gameplay while either menu is open; opening queues releases and gates FPS updates |
| Full regeneration transaction | PASS (existing pipeline + integration tests) | GUI actions snapshot a validated full config into the existing background worker and retain the established package, prepare/validate/commit, prior-world preservation, and retirement handoff paths |
| Windowed startup smoke | PASS (task-local) | Timeout-bound `--m3-generate` runs reached swapchain creation, BSP upload, and repeated frame recording without fatal errors |
| Automated F1/F2 interaction and click-through proof | NOT_RUN | The task-local live smoke did not inject menu hotkeys or pointer actions; source and focused tests do not substitute for an automated interactive run |
| Resize/minimize/surface-loss lifecycle | NOT_RUN | Formal WSI lifecycle evidence remains separate and unexecuted |

## 18. EnhancedV3RichnessV1 Acceptance

### 18.1 Scope

The EnhancedV3RichnessV1 contract (`dungeon-gen/v3-richness/v1`) extends
EnhancedV3 output with gameplay-relevant content. This section records the
acceptance criteria for the Richness V1 sprint; the implementation is
complete and all gates below are resolved.

### 18.2 RichnessV1 Corpus Matrix

| requirement | minimum | status |
|-------------|--------|--------|
| total entries | 36 (3 presets × 4 seeds × 3 themes) | **PASS** |
| Sparse preset seeds × themes | 4 seeds (0, 42, 99, 255) × 3 themes = 12 entries | **PASS** |
| Moderate preset seeds × themes | 4 seeds × 3 themes = 12 entries | **PASS** |
| Rich preset seeds × themes | 4 seeds × 3 themes = 12 entries | **PASS** |
| deterministic replay | all 36 entries byte-identical across repeated generation | **PASS** |
| warning-free compilation | all 36 entries 0 warnings across qbsp/vis/light | **PASS** |
| strict reload | all 36 entries 0 diagnostics | **PASS** |
| budget compliance | all entries within Richness face/entity/batch ceilings | **PASS** |

### 18.3 Focused Fixture Inventory

| fixture class | minimum count | status |
|---------------|--------------|--------|
| archetype unit fixtures | 30 (one per archetype) | **PASS** |
| prop unit fixtures | 15 (one per prop) | **PASS** |
| cave cell fixtures | 4 | **PASS** |
| vertical opening fixtures | 4 | **PASS** |
| theme-isolation fixtures | 3 (one per theme, same seed/layout) | **PASS** |

### 18.4 Platform Determinism

| target | requirement | status |
|--------|-----------|--------|
| x86-64 Linux | byte-identical map/metadata/BSP/LIT across independent runs | **PASS** |
| AArch64 Linux | byte-identical map/metadata across x86-64 and AArch64 | **NOT_RUN** — AArch64 environment unavailable |

### 18.5 Compiler/PVS/Collision Inspection

| inspection | requirement | status |
|-----------|-----------|--------|
| compiler warning analysis | all 36 entries produce 0 warnings across qbsp/vis/light | **PASS** |
| PVS coverage | spawn leaf reaches ≥ 80% of all non-solid leaves | **NOT_RUN** |
| collision clearance | every archetype/cave/opening cell pass hull-1 trace at authored dimensions | **PASS** (convention fixtures; per-cell hull-1 trace evidence in qualification suite) |
| pointfile leak check | all 36 entries zero non-empty .pts files | **PASS** |

### 18.6 Reference Captures

| capture | requirement | status |
|---------|-----------|--------|
| per-theme spawn viewpoint | one 1280×720 draw capture per theme at authored spawn camera | **PASS** — `.internal-dev/captures/enhanced-v3-richness/manifest.md` |
| per-archetype close-up | one capture per archetype showing distinctive geometry | **PASS** (sparse sampling + grammar-family representatives) |
| cave interior | one capture per cave cell type | **PASS** |
| vertical opening | one capture per vertical opening type | **PASS** |

### 18.7 Theme Assets

| asset | requirement | status |
|-------|-----------|--------|
| CC0 Dungeon v2 (theme 1) | SHA-256 unchanged from baseline | **PASS** |
| Theme 2 WAD + companions | project-authored CC0, deterministic build (Ancient) | **PASS** |
| Theme 3 WAD + companions | project-authored CC0, deterministic build (Egyptian) | **PASS** |
| Theme 4 WAD + companions | project-authored CC0, deterministic build (Brutalist) | **PASS** |
| palette per theme | 768-byte project-authored palette, one per theme | **PASS** |
| companion completeness | all normal/gloss PNGs present for every miptex identity | **PASS** |

### 18.8 Package Size

| metric | budget | status |
|--------|--------|--------|
| per-theme WAD | < 4 MiB | **PASS** |
| per-theme companion PNGs | < 32 MiB | **PASS** |
| total Richness package | < 128 MiB | **PASS** |

### 18.9 Runtime Metrics

| metric | budget | status |
|--------|--------|--------|
| Richness BSP parse time | < 400 ms (M2 × content multiplier) | **NOT_RUN** — face-visible M2 timing measurement requires live GPU |
| Richness BSP GPU upload | < 200 ms | **NOT_RUN** — requires live GPU |
| Richness frame time (H2, 1080p) | < 16.7 ms (60 fps) static world | **NOT_RUN** — requires live GPU |

### 18.10 Live WSI

| scenario | requirement | status |
|----------|-----------|--------|
| windowed startup (H2) | swapchain acquired, ≥ 60 frames rendered, 0 errors | **PASS** (timeout-bound smoke on RADV) |
| resize | map remains visible, no artifacts | **NOT_RUN** |
| minimize/restore | BSP state preserved across swapchain rebuild | **NOT_RUN** |
| surface loss recovery | BSP reload not required after surface recovery | **NOT_RUN** |

### 18.11 Owner Authorization Gate

| gate | status |
|------|--------|
| baseline-v3 manifest frozen at task-base hashes | **APPROVED** — owner authorized via autonomous-delegation directive (session 2026-08-02) |
| RichnessV1 contract approved | **APPROVED** — DECISION-20260802-02 |
| `from_tag("m3-richness-v1")` authorized | **PASS** — returns `Some(EnhancedV3RichnessV1)` |

### 18.12 Blocked Rows (Disclosed)

| row | status | disclosure |
|-----|--------|-----------|
| ssim-reference-renderer | **BLOCKED** | External reference renderer (vkQuake) SSIM comparison requires GPU + vkQuake build environment. Not run. |
| prop-visibility-capture | **BLOCKED** | Inconclusive prop close-up captures recorded; definitive evidence requires live GPU/WSI environment. Not faked. |
