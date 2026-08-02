---
schema_version: 1
document_type: generation-specification
status: active
owner: bsp-dungeon
created: 2026-07-24
approval: implemented — Phase 05 compiler/controller/measurement conventions frozen 2026-08-02; Richness production exposure and unrelated beta gates remain pending
---

# BSP Dungeon Generation Specification

## 1. Scope

This specification freezes the implemented procedural dungeon generation contract for M1 and M2 map classes. It defines the input domain, topology model, construction parameters, output ceilings, navigation model, door disposition, compiler profile, PBR texture companion naming, and support corpus. Values may change only through explicit owner re-review. The 2026-07-24 portal/clearance repair is an owner-authorized construction review: it adds the 112-unit minimum room span and makes the existing 64×80 open-route requirement apply explicitly at room portals and junction centers.

## 2. Generator Authorization Gate

The original authorization gate is complete. Current conditions are:

| condition | status |
|-----------|--------|
| M1/M2 bounds locked | READY (§3) |
| Construction parameters frozen | READY (§4) |
| Output ceilings declared | READY (§5) |
| Door disposition decided | READY (§6) |
| Navigation model frozen | READY (§7) |
| Compiler profile pinned | READY (§8) |
| PBR texture companion naming frozen | READY (§9) |
| Support corpus declared | READY (§10) |
| All 5 upstream BSP specs reconciled | READY (Phase 09 updates) |
| Generator implementation sprint | **IMPLEMENTED** — `bsp_generator` and `dungeon_gen` delivered 2026-07-24 |

Future changes must read this specification as the principal contract and must not alter frozen values without owner re-review.

## 3. Map Classes

### 3.1 Definition

The dungeon generator targets exactly two output tiers: M1 (small) and M2 (medium). M3 and M4 tiers are deferred.

| class | room count | loop count | typical use |
|-------|-----------|------------|-------------|
| M1 (small) | 8–16 | 0–2 | test maps, deathmatch arenas |
| M2 (medium) | 17–40 | 1–6 | representative single-player dungeon levels |

M2 is the highest output tier in this campaign. At least one M1 ceiling must be exceeded by the representative M2 fixture.

### 3.2 M1 Generated Domain

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

### 3.3 M2 Generated Domain

| parameter | locked value |
|-----------|-------------|
| layers | 1 |
| outer XY extent (Quake units) | ≤ 3072 × 3072 |
| total Z span (Quake units) | ≤ 384 |
| room count | 17..=40 |
| loop count | 1..=6 |
| placement candidates per room attempt | 32 |
| max placement attempts per room per candidate | ≤ 96 |
| max A* expansions per candidate | ≤ 524,288 |

### 3.4 Nominal Configuration

The following nominal ("typical") configurations are the default M1 and M2 inputs used by all eight nominal-seed corpus entries. They represent mid-range configurations well within the frozen bounds for their class, leaving headroom for boundary-configuration stress testing.

| parameter | M1 nominal | M2 nominal |
|-----------|-----------|------------|
| room count | 12 | 28 |
| loop count | 1 | 3 |
| outer XY extent (Quake units) | 1024 × 1024 | 2048 × 2048 |
| total Z span (Quake units) | 192 | 256 |

Boundary configurations (§10.2) use the same nominal XY and Z values for their respective classes, varying only room count and loop count. The nominal configuration does not widen or narrow the generator's construction domain — it defines the default inputs.

## 4. Shared Construction Parameters

| parameter | locked value |
|-----------|-------------|
| construction unit quantum | 16 Quake units |
| wall thickness | 16 Quake units (1 quantum) |
| minimum room outer span | ≥ 112 Quake units (7 quanta; 80-unit clear interior after wall cells) |
| clear route width (corridors, passages, portal throats, junction centers) | ≥ 64 Quake units (4 quanta) |
| clear headroom (corridors and portal throats) | ≥ 80 Quake units (5 quanta) |
| public `Corridor.height` | exactly 80 Quake units; non-80 public input is rejected (owner-approved source: `owner-sprint-20260726`; implementation deferred to Phase 09) |
| route type | Legacy v1: level (no ramps or stairs). Enhanced v2/v3: frozen two-layer stair arrangement (§17/§20). |
| room connections | open arches only (no doors for beta — see §6) |
| stacked XY spaces | prohibited (no room directly above another) |
| room shape | axis-aligned rectangular rooms (no diagonal, polygonal, or curved rooms) |
| corridor shape | axis-aligned straight corridors only (no diagonal, curved, or angled corridors) |
| room Z alignment | Legacy v1: all rooms share a common floor Z and common ceiling Z (single-layer Cartesian). Enhanced v2/v3: two-layer arrangement with lower floor Z=0, upper floor Z=192, room height 176 (see §17 and §20). |

## 5. Output Ceilings

| metric | M1 ceiling | M2 ceiling |
|--------|-----------|------------|
| compiled faces | < 2,000 | < 10,000 |
| entities | < 50 | < 300 |
| static batches | < 100 | < 500 |

At least one M1 ceiling must be exceeded by a representative M2 fixture to prove the tier separation is meaningful.

## 6. Door Disposition

### 6.1 Decision: Open Arches Only

Doors (`func_door`, `func_button`, `func_plat`) are **excluded** from the dungeon generator beta. All room connections are open arches with no moving geometry. Rationale:

1. **Compilation complexity**: Doors require inline brush models, trigger/target graph wiring, clipnode hull geometry, and convex decomposition — each with its own acceptance cells.
2. **Runtime behavior**: Door behavior (activation, timing, collision, persistence, pose-sync, lighting) requires an app-owned behavior adapter that is not yet fully validated.
3. **Evidence budget**: Door cells (11 optional rows) are all NOT-RUN. Opening them for generator output would add 11 blocking cells to the evidence matrix.
4. **Fallback**: Open arches provide continuous walkable navigation with zero runtime behavior and zero additional entity/trigger/target complexity.

### 6.2 Future Door Support

If a future sprint adds door support:
- The 11 door evidence cells in the evidence matrix must pass first.
- Generator room connections must gain an optional door flag per connection.
- `func_door`, `func_button`, `func_plat` brush model geometry must be producible by the generator.
- Trigger/target graph wiring must be generated deterministically from the connection graph.

Until then, open arches are the only connection type.

## 7. Navigation Model

### 7.1 Player Movement Strategy

The generator's corridor/room topology is designed for **point-trace movement against compiler-preexpanded hull 1 clipnodes**. The movement contract is:

| property | value |
|----------|-------|
| trace type | point trace (hull 0) |
| collision target | hull 1 clipnodes (compiler-preexpanded) |
| player dimensions | 32×32×48 Quake units (symmetric, compiler-expanded from ±(16, 16, 24)) |
| minimum passage width | 64 Quake units (player width + 2 wall thicknesses of buffer) |
| minimum vertical clearance | 80 Quake units (player height + headroom) |

### 7.2 Hull Dispute Status

The competing `FGD-documented hull` vs `symmetric hull` dispute (`bsp-spatial-physics.md` §5.2.1) is **unresolved** at the generator specification level. The navigation model assumes the compiler-preexpanded hull 1 hypothesis:

1. Hull 1 stored in the BSP clipnode tree is the expanded collision volume from the player's point-sized origin.
2. The ericw-tools `qbsp` compiler expands hull 1 from the point trace origin, making the effective hull dimensions compiler-output-dependent.
3. Player movement uses **point traces** (hull 0) against hull 1 clipnodes, not box traces with the player's AABB.

The generator must produce corridors whose clear width (≥ 64 Quake units) and height (≥ 80 Quake units) are wide enough for a 32×32×48 player regardless of which hull interpretation is correct. If Phase 06 compiled threshold tests disprove point-trace compatibility, the corridor minimums may need revision.

### 7.3 Reachability Guarantee

| guarantee | M1 | M2 |
|-----------|----|----|
| all rooms reachable from spawn via at least one path | yes | yes |
| all room pairs connected via the corridor graph | yes | yes |
| loop count | 0–2 | 1–6 |
| dead-end rooms permitted | yes (when loops = 0) | yes (with alternative routes) |

A* or equivalent pathfinding must find a path between every pair of room centers that have a connecting corridor. Corridor-to-room adjacency must produce valid walkable connections with no geometry gaps, overlaps, or false solid leaves.

## 8. Compiler Profile

The generator produces Quake 1 `.map` files targeting the pinned compiler profile:

| parameter | value |
|-----------|-------|
| compiler | ericw-tools 2.0.0-alpha3 |
| compatibility family | `q1-portable-ericw` |
| exact publication profile | `ericw-q1-bsp2-generated` at `tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml` |
| format | BSP2 (mandatory for generated dungeon output) |
| qbsp args | `-bsp2 -threads 1` |
| vis args | `-threads 1` |
| light args | `-threads 1 -lit` (deterministic, external `.lit` output) |
| WAD companion | project-authored dungeon theme WAD referenced by basename in the generated `.map`; includes compiler-only `skip` miptex |
| warning policy | any warning from `qbsp`, `vis`, or `light`, including missing textures or skipped fill, fails compilation and prevents publication |

### 8.1 Why BSP2 Only

BSP29 has structural limits (65,535 vertices, 65,535 edges, 32,767 clipnodes) that M2 maps may approach or exceed. BSP2 has effectively unlimited indices and is the BSP evidence baseline (all compiler PASS cells are BSP2). Generated output must use BSP2 exclusively.

### 8.2 Determinism

The generator must produce byte-identical `.map` output for identical (seed, config) pairs. Combined with the `light -threads 1 -lit` profile, this guarantees byte-identical compiled `.bsp` and `.lit` across independent builds. The pinned compiler profile is proven reproducible for BSP2 output (evidence cell `CMP-REPRO` PASS, 2026-07-24).

### 8.3 Development Cache Identity

Development cache reuse is permitted only when a deterministic fingerprint covers the generator, compiler-driver, BSP dependency source, Cargo lock state, explorer script, exact profile, WAD, palette, and selected `qbsp`/`vis`/`light` executables. The cache manifest must verify current BSP, LIT, WAD, palette, and generator identities before reuse. A missing hash or artifact is stale, not an optional cache hit. This prevents output-affecting source changes such as serializer mapping defaults from silently replaying an old BSP.

## 9. PBR Texture Companion Naming

The generator produces BSP maps referencing a project-authored WAD2 texture set. PBR companion discovery uses the frozen naming convention:

| companion | naming rule | format | dimensions |
|-----------|------------|--------|------------|
| normal map | `<texture>_norm.png` | PNG, tangent-space R/G encode X/Y | exact match to base texture |
| gloss map | `<texture>_gloss.png` | PNG, red channel only | exact match to base texture |

Where `<texture>` is the sanitized BSP miptex identity, not the BSP filename or a replacement path. The naming convention is identical to the existing BSP renderer-lighting contract (`bsp-renderer-lighting.md` §2.4).

The generator itself does not produce PNG companions — it produces the `.map` referencing WAD texture names. The theme asset pack provides companion PNGs alongside the WAD at a configured package root. The existing companion discovery, validation, and PBR pipeline consume these according to the frozen BSP renderer-lighting contract.

## 10. Support Corpus

### 10.1 Nominal Configuration Seeds

The following seeds are declared as the nominal support corpus. All must produce valid output within their declared M1 or M2 ceilings:

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

### 10.2 Boundary Configuration Seeds

| config | seed | parameters | expected behavior |
|--------|------|-----------|--------------------|
| Boundary A (M1 minimum) | `42` | 8 rooms, 0 loops, XY 1024×1024, Z 192 | valid output within M1 ceilings |
| Boundary B (M1 maximum) | `43` | 16 rooms, 2 loops, XY 1024×1024, Z 192 | valid output within M1 ceilings |
| Boundary C (M2 minimum) | `44` | 17 rooms, 1 loop, XY 2048×2048, Z 256 | valid output within M2 ceilings |
| Boundary D (M2 maximum) | `45` | 40 rooms, 6 loops, XY 2048×2048, Z 256 | valid output within M2 ceilings |

### 10.3 Execution Requirement

All 12 configurations (8 nominal + 4 boundary) must be compiled through the pinned BSP2 profile and produce valid output within their declared M1/M2 ceilings. No panic, infinite loop, silent fallback, profile substitution, or showcase exception is permitted. Failed seeds or exhausted resources may not silently select a fallback profile, replacement asset, revised bound, or showcase exception.

**Status (2026-07-25 explicit-shell validation): PASS for generation, warning-free compilation, strict reload, sealing, face/entity ceilings, and spatial witnesses.** All 12 frozen entries compile through the pinned BSP2 profile with no warnings. Tests require non-solid BSP `point_contents` at room centers, point-entity origins, corridor centers, portal throats, and the center plus four interior-corner witnesses of every 64×64 junction clearance. Historical static-batch evidence remains tracked by GitHub #57; a 2026-07-26 current nominal-M1 isolated package reaches 6 neutral/upload-preflight batches, but strict extraction (#58) and GPU mount/submission (#61) block the required strict submitted-draw proof.

## 11. Generator Guarantees

The generator must guarantee:

1. **Deterministic output**: byte-identical `.map` for identical (seed, config) inputs.
2. **Sealed map**: no leaks and no skipped-fill compiler path.
3. **Walkable topology**: all rooms reachable from at least one `info_player_start` entity; compiled room, corridor, portal, and junction witnesses are non-solid.
4. **Explicit role-bound room shells**: every room emits a `stone_floor` slab at Z `0..16`, a `stone_ceiling` slab at `(ceiling-16)..ceiling`, and four full-height `stone_wall` masks split around omitted apertures.
5. **Real open arches**: every routed approach reaches its actual room wall; both normal portal rectangles and routed endpoint-footprint wall cells are omitted while preserving a 64×80 clear throat.
6. **Clear corridor junctions**: corridor-only union geometry preserves the complete central 64×64 clear square of every L/T/X junction without extending corridor ceilings into room interiors.
7. **Safe point entities**: spawn and light origins occupy room clear volume above the floor slab.
8. **No overlapping geometry**: no room-on-room and no corridor-through-room without an explicit open connection.
9. **Valid warning-free BSP2 compilation**: output compiles through the pinned profile without errors or warnings; missing textures and skipped fill are hard failures.
10. **Budget compliance**: compiled output remains within declared M1/M2 ceilings. Face/entity ceilings pass; the current nominal-M1 candidate reaches 6 neutral/upload-preflight batches, but static-batch/draw compliance remains unresolved until strict submitted evidence exists for the frozen corpus (GitHub #57, blocked by #58 and #61) and must not be represented as PASS.
11. **Seeded randomness**: the random stream derives directly and exclusively from the seed value with SHA-256 framing per semantic stage, following the `DECISION-20260722-01` pattern.
12. **Open arches only**: no door, button, platform, or trigger entities in generated output.

## 12. Deterministic Random Framing

### 12.1 RNG Derivation

Every random stream in the generator derives directly and exclusively from the master `u64` seed value through SHA-256-based domain separation. The derivation contract:

| property | value |
|----------|-------|
| domain separator | `"dungeon-gen/v1"` (UTF-8, prefixed before seed bytes) |
| seed byte order | little-endian `u64` (8 bytes) |
| framing | `SHA-256(domain_separator || seed_le_bytes || tag)` per semantic stage |
| output | 32-byte SHA-256 digest; consumed as a stream of little-endian `u64` values |

### 12.2 Semantic Tags

The following tags are the frozen stage identifiers. Renaming or reframing a tag is an output-version change.

| tag | stage |
|-----|-------|
| `room-placement` | room candidate generation, overlap rejection, bounds clamping |
| `corridor-routing` | connection topology, A* routing, junction materialization |
| `entity-placement` | spawn point placement, light placement, entity ordering |
| `light-placement` | light entity positioning within rooms and corridors |

### 12.3 Output-Version Rule

Changing the domain separator, seed byte order, tag set, tag spelling, or framing algorithm increments the output version. Maps produced under different versions are independent products; no byte-compatibility is guaranteed.

## 13. Canonical Serialization

### 13.1 Map Text Grammar

The generator emits Standard Quake `.map` text with the following canonical ordering rules:

| rule | value |
|------|-------|
| entity order | `worldspawn` first; remaining entities in creation-index order |
| key order | alphabetical by key string (ASCII byte order) within each entity |
| brush order | by creation index within each entity |
| face order per brush | bottom, top, north, south, west, east (canonical axis order) |
| plane point format | three parenthesized integer triples `( x y z ) ( x y z ) ( x y z )` |
| integer formatting | decimal; no scientific notation; no leading zeros except for `0` itself |
| texture name | double-quoted; follows plane points on same line |
| texture mapping syntax | Standard Quake offset/rotation/scale: `"texture" x_off y_off rotation x_scale y_scale`; canonical bytes use `"texture" 0 0 0 0.25 0.25` |
| line endings | `\n` (LF) |
| terminal newline | exactly one trailing `\n` |

### 13.2 Deterministic Byte Contract

Identical `(seed, config)` inputs must produce byte-identical `.map` text under the canonical serialization rules. Equivalent logical output must serialize identically; any deviation in ordering, whitespace, or formatting is a determinism failure.

## 14. Evidence Status (Post-Implementation)

As of the 2026-07-25 explicit-shell rendering closeout, the following gates have been resolved:

### Resolved (PASS)

1. **Theme licensing**: CC0 Stone Beta theme delivered; `LICENSE` file recorded; zero rights clearance needed. (PASS — 2026-07-24)
2. **Production theme**: CC0 procedural theme replaces KB3D; Pillow-backed `build.py` is deterministic; four distinct 1024×1024 visible stone roles with 12 detailed companion PNGs, a 64×64 compiler-only `skip`, WAD2, and a project-authored palette. ericw-tools compiles the 1024² miptex entries into BSP2 output and a strict headless capture resolves three PBR materials. (PASS — 2026-07-27)
3. **Support-corpus execution**: all 12 configurations generate, compile warning-free through the BSP2 profile, reload strictly, remain sealed, and pass room/entity/corridor/portal/junction `point_contents` witnesses. (PASS — 2026-07-25, `corpus_execution` test)
4. **Generator determinism**: byte-identical `.map` and compiled `.bsp` across independent runs. (PASS — 2026-07-25, `determinism` + `corpus_execution` tests)
5. **M1/M2 face and entity budgets**: all corpus entries remain within those ceilings and M2 exceeds M1. Historical nominal M1 seed 0 reported 183 batches, while the 2026-07-26 current isolated candidate reaches 6 neutral/upload-preflight batches. Static submitted-draw compliance across the frozen corpus is not proven because strict extraction (#58) and GPU upload rollback (#61) stop the path. (PARTIAL — 2026-07-26; GitHub #57)

### Still Not Proven (BLOCKED on external GPU/WSI environment)

6. **Reference-renderer calibration**: SSIM comparison against vkQuake not run. (NOT-RUN — requires GPU + vkQuake)
7. **Live-WSI acceptance**: resize, minimize/restore, surface-loss recovery unrun. (BLOCKED — requires live WSI environment)
8. **Animated texture frame selection**: no face-visible project-authored fixture with animated textures. (NOT-RUN — deferred)
9. **Material-source multi-route fixture**: no project-authored fixture exercising embedded miptex + WAD + loose replacement. (NOT-RUN — deferred)

## 15. Out of Scope for Beta

- Multi-layer dungeons (ramps, stairs, stacked rooms, multiple Z planes)
- Doors, buttons, platforms, triggers, or moving brush entities
- Non-rectangular rooms (diagonal, polygonal, curved)
- Non-straight corridors (diagonal, angled, curved)
- Liquid volumes (water, slime, lava)
- Theme variation (single dungeon theme)
- Ambient sound entities
- Monster/item placement
- Puzzle/trap logic
- Save/load of generated map state (the `.map` is regenerated deterministically)

## 16. Owner-Gated Decision Packets (Phase 01 Baseline)

### 16.1 Decision A — Serializer Grammar

**Status:** **APPROVED — Option A (Standard Quake); mapping scale owner-amended 2026-07-27**
**Baseline:** `.internal-dev/captures/bsp-dungeon-repair-baseline/manifest.json`
**Baseline ID:** `5fda7dae1d1f3da51c064d1d136418dae9c0e79a43ad73396a496bba81270c35`

**Approved contract:** Retain Standard Quake offset/rotation/scale bytes: `"texture" x_off y_off rotation x_scale y_scale`; canonical output is `"texture" 0 0 0 0.25 0.25`. Valve 220 is not implemented. `BrushFace.u_axis` and `BrushFace.v_axis` are dead public-IR fields and must be removed by the authorized generator implementation; they are not part of the approved grammar.

**Justification:** Standard Quake syntax remains compiler-compatible, while scale `0.25` gives the 1024² production stone textures usable authored detail on 64-unit walls and corridors. A stale scale-`1.0` cache sampled only 64 source texels across a 64-unit span and produced visibly oversized, soft features; a fresh scale-`0.25` compile samples 256 texels across that span. The canonical seed-0 rebuild remains below the M1 face/batch ceilings and passes strict headless draw validation.

**Rejected alternative:** Valve 220 would require valid non-collinear face-specific U/V axes, bracket serialization, changed deterministic bytes, regeneration of all corpus entries, and compiler/lightmap/visual revalidation. The current placeholders make that alternative unsafe.

**Caveat:** Mapping scale affects compiler face subdivision and lightmap extents, so any future default change requires corpus ceiling, strict extraction, and fixed-camera visual revalidation. `u_axis`/`v_axis` remain outside the canonical serialized grammar.

**Affected specifications:** §13.1 of this specification; `bsp-compatibility.md` §12.3; decision records `DECISION-20260726-01` and `DECISION-20260727-02`.

**Re-review trigger:** Any change from Standard Quake grammar, texture-coordinate defaults, or the public `BrushFace` contract requires owner re-review before implementation and output-version/corpus revalidation.

### 16.2 Decision B — Vertical Corridor IR Semantics

**Status:** **APPROVED — Option A (freeze at 80); source: `owner-sprint-20260726`**
**Baseline:** `.internal-dev/captures/bsp-dungeon-repair-baseline/manifest.json`

**Approved contract:** Public `Corridor.height` is exactly `80` Quake units. The first fallible public-IR boundary must reject every non-80 value; generated routes always use `80`. `build_corridor_slabs` and `build_corridor_boundary_walls` must derive their vertical geometry from `corridor.height` rather than reconstructing it from global `CORRIDOR_HEIGHT`. The generator constant remains the route-construction source of the approved value, not an emission override.

**Justification:** Every current generated route uses 80 and the frozen construction contract requires 80-unit clear headroom. Fixing the public value eliminates the exposed non-80 inconsistency without expanding the geometry/sealing test surface.

**Rejected alternative:** Variable public heights would require quantized constructibility rules, consistent use through all routing/emission paths, new non-80 tests, and compiled spatial revalidation. It is not authorized for this beta.

**Caveat:** Phase 01 changes no production code. Current source still exposes non-80 `Corridor.height` and reconstructs slabs/walls through the global constant; Phase 09 must implement the approved rejection and field-based emission before code can claim conformance.

**Affected specifications:** §4, §7, and §11 of this specification; `bsp-compatibility.md` §12.4; decision record `DECISION-20260726-02`.

**Re-review trigger:** Any variable-height corridor request, non-80 clear-height proposal, or change to the validated public-IR boundary requires owner re-review and renewed compiled spatial evidence.

### 16.3 Profile Identity Reconciliation

**Status:** RECORDED

Two profile names exist; they are distinct concepts:
- **Compatibility family:** `q1-portable-ericw` — describes the compiler dialect (ericw-tools Q1-family BSP2).
- **Exact publication profile:** `ericw-q1-bsp2-generated` — the checked-in profile file at `tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml`.

These are not interchangeable aliases. The specification uses `q1-portable-ericw` for the family concept and references the profile file for exact args. This reconciliation is recorded without renaming either name.

### 16.4 Exact Artifact Freeze (Retry)

**Status:** **COMPLETE — exact compiled artifact identity**
**Baseline ID:** `5fda7dae1d1f3da51c064d1d136418dae9c0e79a43ad73396a496bba81270c35`

The frozen exact artifact is `.internal-dev/captures/bsp-dungeon-repair-baseline/baseline-artifact.{map,bsp,lit}` with compiler provenance in `baseline-artifact.provenance.toml`. Its BSP SHA-256 is `b24922904beccd06617b47715243993d8f40b4f262b4140200a69a3bdd6326d7` (86,312 B); the map and LIT hashes are recorded in the manifest. The owner supplied that these are deterministic outputs of the current generator source tree and the exact artifact associated with the reported rendering defect. The current generator tree matches reported code commit `8933a041`.

This supersedes only the artifact-identity blocker. The old cached pre-fix BSP and raw MCP transcript remain historical comparison evidence in the manifest; they are not substituted for the exact artifact. No renderer replay, fixed-camera capture, frame-slot observation, or live-WSI claim is implied by this artifact freeze.

### 16.5 Known Code/Spec Divergence

| Item | Spec claim | Code reality | Status |
|------|-----------|-------------|--------|
| Texture coordinate format | §13.1: Standard Quake `"texture" x_off y_off rotation x_scale y_scale`, canonical scale `0.25` | `emit_face` emits Standard `0 0 0 0.25 0.25` | **ALIGNED** — grammar retained; owner-directed scale repair validated 2026-07-27 |
| `u_axis`/`v_axis` fields | Approved IR has no axis fields for Standard syntax | Current `make_brush` retains identical `[1,0,0,0]` placeholders | **APPROVED / implementation pending** — remove in authorized generator phase |
| Corridor height IR | Public `Corridor.height == 80`; field is the emission source of truth | Current public field accepts non-80; slabs/walls use global constant | **APPROVED / implementation pending** — reject non-80 and use field in authorized generator phase |
| BSP artifact identity | §8: pinned profile, deterministic output | Exact `baseline-artifact.bsp` is frozen at `b2492290…` with matching map/LIT/provenance and the current generator tree; the old cached BSP remains a distinct historical cache-invalidation defect | **RESOLVED for the exact artifact freeze**; historical cache defect remains isolated |
| WAD identity | §9: theme WAD referenced by basename | Exact artifact provenance selects the 27,572-B `b239e695…` theme WAD and matching palette; the historical cache's 22,060-B compile-time candidate remains historical only | **RESOLVED for the exact artifact closure**; historical cache provenance remains unresolved |
| Static batch ceiling | §5: M1 < 100 batches | Historical: pre-fix 329, post-fix 183; current isolated nominal-M1 candidate: 6 neutral/upload-preflight batches | PARTIAL — submitted strict draw evidence blocked by #58/#61; GitHub #57 remains open |

## 17. Enhanced v2 Two-Layer Profile

### 17.1 Scope

The Enhanced v2 profile is an additive, structurally disjoint generation path
in `src/bsp_generator/src/enhanced/`. It produces M2-only two-layer dungeons
with stairs, theme palette assignment, and per-room corridor/ceiling/pillar
variance. The Legacy v1 generator and its 12-entry frozen corpus remain
unchanged.

### 17.2 Vertical Contract

Phase 01 feasibility evidence established and froze the following contract
(`DECISION-20260729-02`):

| parameter | frozen value |
|-----------|-------------|
| lower floor Z | 0 |
| upper floor Z | 192 |
| room height (both layers) | 176 |
| stair riser | 16 |
| stair tread (both stair types) | 16 |
| total Z span | 368 (≤ 384 M2_Z_MAX) |
| layer count | 2 (frozen, not configurable) |

All four candidate fixtures from Phase 01 pass warning-free BSP2 compilation,
strict load, spatial witnesses, deterministic replay, and the frozen legacy
corpus remains byte-identical.

### 17.3 Placement Contract

- Rooms placed on two layers with **balanced membership** (max difference of 1 between layers)
- All rooms on both layers projected onto a shared XY **occupancy grid** — no two rooms may overlap in XY
- Room horizontal spans: **112–256** Quake units per axis
- Sockets derived from committed room walls only; walls must be ≥ 128 units long to host a socket
- Socket aperture: **64** units wide, **32-unit** corner margins
- Placement uses transactional journal checkpoint/rollback — failed attempts restore prior state

### 17.4 Topology Contract

- **Per-layer MST**: rooms on each layer form a minimum spanning tree via Kruskal's algorithm
- **Loop edges**: `loop_count` extra edges added from non-MST candidate pairs
- **Stair transitions**: exactly `vertical_edges` (1–3) direct lower-to-upper room stair connections
- **Type A — Room-Scale Grand**: a 192-unit run across the lower host room's full wall-free width
- **Type B — Wall-Edge Narrow**: a 192×64-unit run hugging a lower host room wall
- Both types reserve 12 exact 16-unit tread/riser columns, 80-unit headroom, lower and upper approaches, and lower-wall/ceiling/upper-wall apertures
- The inter-layer slab aperture covers the complete 192-unit tread run; the supported upper landing retains 80-unit headroom and joins the aperture through a full 64-unit crest throat
- Lower approaches join committed lower routes; upper approaches leave through the split host ceiling and reach the selected upper-room wall aperture
- **Transactional**: all topology operations use mark/rollback/commit with loop budget tracking and bounded canonical transition backtracking
- **Global connectivity**: all rooms in both layers reachable from any other (validated post-commit)

### 17.5 Theme Contract

- CC0 Dungeon v2 theme at `src/bsp_generator/themes/cc0_dungeon_v2/`
- Separate WAD from Legacy v1's CC0 Stone Beta
- The checked-in deterministic source closure includes `LICENSE`, `theme.toml`, the project palette, WAD2, and 15 visible identities with matching 1024² basecolor, normal, and gloss PNGs; `skip` has no PNG companions
- `engine_pack enhanced-dungeon` keeps albedo WAD-backed and publishes both normal and gloss companions for every eligible referenced miptex identity; a missing, malformed, or dimension-mismatched expected companion rejects Enhanced publication before atomic rename
- Generic `engine_pack compile-bsp` retains its optional-companion behavior; the complete companion requirement is specific to the authored Enhanced profile
- Palettes: typed, immutable, checked-in Rust data (not parsed from TOML at runtime)
- Room role derivation: Entry (lowest RoomId), Hub (max degree), DeadEnd (degree=1), Side (remaining)
- Assignment strategies: Uniform (all rooms same palette) or ByZone (zone-distinct palettes)

### 17.6 Feature Variance Contract

- **Corridor widths**: 64, 80, or 96 Quake units per route (RNG-selected)
- **Ceiling heights**: 128, 144, or 176 Quake units per room (RNG-selected)
- **Pillars**: up to N per room (0–8), freestanding 32×32×80 axis-aligned boxes, connectivity-verified
- **Spawn origin**: exactly one `info_player_start` centered on the canonical 64×64 lower stair landing, at floor-top + 24; the center retains a 16-unit Quake hull radius plus a 16-unit safety margin from landing sides and tread solids, and its cardinal `angle` faces the stair opening
- **Light origins**: one per room in clear volume above floor slab
- Exclusion regions computed before pillar placement (walls, apertures, corridors, transitions)

### 17.7 Emission Contract

- Explicit room shells with floor/ceiling slabs and wall aperture masks
- Corridor-only union geometry for turns and intersections
- Stairwell shells split room floors beneath tread columns, omit the host inter-layer slab across the complete 192-unit run, omit both wall apertures, preserve a supported 80-unit-headroom upper landing and full 64-unit crest throat, and seal the 176→192 bridge plus lower/upper approaches without positive-volume overlap
- Enhanced worldspawn emits `"_minlight" "16"` so the pinned ericw `light` stage assigns static style-0 baked data to fully occluded connector and stair faces instead of omitting their lightmaps
- Canonical `.map` output compatible with the pinned BSP2 compiler profile

### 17.8 RNG Isolation

Enhanced v2 uses domain separator `"dungeon-gen/v2"` — fully independent from
Legacy v1's `"dungeon-gen/v1"`. Six frozen stage tags:

| tag | stage |
|-----|-------|
| `layer-placement` | two-layer room placement |
| `vertical-topology` | topology and transition selection |
| `vertical-routing` | reserved for future vertical routing |
| `theme-assignment` | palette assignment |
| `feature-placement` | pillars, ceiling variance |
| `corridor-variance` | per-route corridor width |

### 17.9 Evidence Status

| contract | status |
|----------|--------|
| Vertical contract selection | **PASS** — Phase 01 feasibility evidence |
| Config validation | **PASS** — all field-rejection tests pass |
| Determinism | **PASS** — `generate_deterministic` test |
| Nominal generation | **PASS** — seed 42 generates 28 rooms, route-connected transitions, and lights; the pinned compiler path is warning/leak-free |
| Minimal generation | **PASS** — seed-search succeeds within 0..100 |
| Maximal generation | **PASS** — seed-search succeeds within 0..200 |
| Metadata population | **PASS** — all fields populated |
| RNG independence | **PASS** — Legacy and Enhanced domains produce distinct output |
| Baked lightmap coverage | **PASS** — seed 42 full pinned compiler pipeline produces 7,213 faces, all with nonnegative light offsets and static style 0 after Enhanced-only `_minlight 16` emission |
| Theme resource closure | **PASS** — deterministic rebuild byte-matches the checked-in WAD, palette, static files, and all 45 PNGs; seed-42 Enhanced publication stages and strictly authorizes the exact 12 normal/gloss companions for its six referenced identities |
| Stair spawn clearance | **PASS** — seed-42 and the full Enhanced corpus place the authored spawn at a lower landing center; compiled `StoredHull::Player` traces are not start-solid |
| Full-run stair headroom | **PASS** — Type A/Type B source masks omit the full 192-unit run, preserve a 64-unit crest throat, and compiled witnesses remain clear at 56-unit standing height and the frozen 80-unit headroom |

## 18. Approval and Evidence Matrix (Post-Implementation)

| contract | status | evidence basis | blocker | reviewer |
|----------|--------|---------------|---------|----------|
| M1/M2 bounds (§3) | **PASS** | all 12 corpus configs generate within bounds; boundary A–D exercise min/max room+loop counts successfully (corpus_execution test, 2026-07-24) | none — bounds proven by implementation | dhickel (2026-07-24) |
| Construction parameters (§4) | **PASS (owner re-review 2026-07-24)** | `CONSTRUCTION_QUANTUM=16`, `WALL_THICKNESS=16`, minimum room span 112, and corridor/portal 64×80 minimums are enforced; L/T/X tests preserve the full central 64×64 clear square. The user's authorized repair request approved the new room-span minimum and explicit portal/junction interpretation. | none | dhickel (2026-07-24) |
| Output ceilings (§5) | **PARTIAL / SEED-0 SUBMITTED-DRAW PASS** | Historical warning-free corpus evidence records M1 max 558 compiled faces / 18 entities and M2 max 1,826 compiled faces / 42 entities before the canonical scale change. A fresh 2026-07-27 nominal M1 seed-0 scale-`0.25` artifact has 1,124 renderable faces, 20 entities, 6 renderer batches, and a successful strict submitted headless draw. | seed-0 remains below M1 ceilings; all-12 corpus regeneration and a deterministic batch assertion remain required for complete evidence | owner-directed (2026-07-27) |
| Door disposition (§6) | DECIDED | open arches only; doors excluded from beta | none — decision is architectural, not evidence-dependent | dhickel (2026-07-24) |
| Navigation model (§7) | **PASS (structural + compiled spatial witnesses)** | corridor/portal minimums 64×80 are enforced; topology tests prove reachability; all 12 compiled corpus maps require non-solid room, point-entity, corridor, portal-throat, and full junction-clearance witnesses. The final seed-0 MCP session independently confirms three room centers and the formerly blocked endpoint-corner junction are non-solid. Live movement remains BLOCKED (NAV-FIXED-STEP-TRAVERSAL, NAV-SLIDING, NAV-LIVE-MOVER NOT-RUN). | live navigation evidence BLOCKED on GPU/WSI environment | dhickel (2026-07-25) |
| Compiler profile (§8) | PASS | BSP2 `-threads 1 -lit` profile remains deterministic; `engine_pack` and corpus helpers now reject warnings from every compiler stage, including missing textures and skipped fill. Fresh seed-0 provenance contains zero warnings. | none | dhickel (2026-07-24) |
| PBR texture companion naming (§9) | PASS | `<texture>_norm.png` / `<texture>_gloss.png` convention proven (MAT-PBR-* companion tests pass 2026-07-24) | none — naming convention is frozen in renderer-lighting spec | dhickel (2026-07-24) |
| Support corpus (§10) | **PASS (except separately tracked runtime batch/draw proof)** | all 12 configurations (8 nominal + 4 boundary) generate, compile with zero warnings, reload strictly, remain sealed, pass face/entity ceilings, and return non-solid contents at room/entity/corridor/portal/junction witnesses. Byte-identical BSP determinism remains covered. | current M1 neutral/preflight batch count is below ceiling, but all-12 strict submitted batch/draw evidence is blocked by #58/#61 and tracked by #57 | dhickel (2026-07-26) |
| Generator guarantees (§11) | **PARTIAL** | deterministic map/BSP output, explicit role-bound room shells, sealed warning-free compilation, real open portals, 64-unit clear junctions, safe point entities, structural reachability, face/entity budgets, SHA-256-framed RNG, and open-arch-only output pass. Current nominal M1 reaches 6 neutral/upload-preflight batches. | strict extraction/#58, GPU upload rollback/#61, all-12 submitted batch/draw proof/#57, live movement/WSI, and reference calibration remain blocked | dhickel (2026-07-26) |

## 19. Enhanced v3 Architectural Proof (Private — 2026-07-31)

### 19.1 Scope

A private proof-only package demonstrated the architectural feasibility of cardinal/45°
chamfered-octagonal geometry, pointed-arch portal apertures, grounded assemblies, and
dense M2-budget compositions. All work was confined to test-only code; no production
profile, dispatch, public export, or living specification was changed. Its segmented-arch
row records the historical proof boundary; the later owner-authorized explorer integration
is governed by §20.7 and does not retroactively change that fixture's evidence class.

### 19.2 Evidence Summary

| claim | status | key result |
|-------|--------|------------|
| 45° diagonal wall geometry | PASS | `convex-45-shell.map`: warning-free compile, thickness 32/√2 ≈ 22.63 ≥ 16, spatial witnesses pass |
| Pointed-arch portal aperture | PASS | `pointed-portal.map`: full-depth shell omission, 100% throat witnesses non-solid |
| Grounded assembly support contract | PASS | `grounded-assembly.map`: acyclic graph, coplanar contact, atomic dependent removal |
| Segmented-arch portal (focused-only) | PASS | `segmented-portal.map`: throat witnesses pass; deferred from integration |
| M2 budget compliance | PASS | Dense Rich: 2,404 faces, 6 entities, 4 batches — well within all ceilings |
| v1/v2 compatibility freeze | PASS | 24/24 entries byte-identical (12 Legacy + 12 Enhanced) |
| Deterministic output | PASS | Byte-identical .map and metadata across repeated generation |
| Live GPU startup | PASS | Swapchain acquired, 21,574 frames, 0 panics/errors |

### 19.3 Known Gaps

- **ericw-tools small-map segfault**: Tool limitation on maps < ~5 brushes. Focused
  fixtures enlarged to 73–75 brushes. Integrated.map (2 brushes) provides source-level
  proof; compiled claims use focused fixtures.
- **Integrated portal capture (EV-070)**: NOT_RUN due to the above. Focused portal
  fixtures provide compiled spatial proof.
- **Private semantic pipeline density**: Cannot yet emit compiler-safe structural
  density at generation time; dense Rich fixture is hand-authored.

### 19.4 Reference

- Decision package: `.internal-dev/plans/enhanced-v3-proof/decision-package.md`
- Full evidence matrix: `.internal-dev/plans/enhanced-v3-proof/evidence-matrix.md`
- Changelog: `.internal-dev/changelogs/2026-07-31-enhanced-v3-proof-closeout.md`
- Knowledge: `.internal-dev/knowledge/bsp-enhanced-v3-proof.md`

**Owner decision gate**: **APPROVED** — Decision A (Standard Quake serializer grammar) and Decision B (public corridors fixed at 80) were approved with source `owner-sprint-20260726`. See §16.1, §16.2, `DECISION-20260726-01`, and `DECISION-20260726-02`. Phase 01 records the contract only; authorized implementation remains subsequent work.

**Generator authorization gate**: IMPLEMENTED. The `bsp_generator` crate exists at `src/bsp_generator/`; functional topology, compiler cleanliness, determinism, and face/entity ceilings pass. The historical M1 batch breach is no longer the current neutral/preflight count (6), but the beta gate remains NO-GO: strict generated extraction is blocked by #58, GPU mount/submission by #61, and all-12 submitted batch/draw proof remains open in #57. See `.internal-dev/plans/bsp-dungeon-contract-evidence/evidence-matrix.md`.

**Re-review trigger**: Any change to M1/M2 bounds, construction parameters, output ceilings, door disposition, navigation model, support corpus, or generator guarantees requires owner re-review.

## 20. EnhancedV3 Production Profile

### 20.1 Scope

The EnhancedV3 profile is an additive, structurally disjoint generation path
in `src/bsp_generator/src/enhanced_v3/`. It produces M2-only two-layer dungeons
with cardinal + 45° chamfered-octagonal rooms, pointed-default plus
rectangular/segmented cardinal portal surrounds, grounded assemblies, and
Sparse/Moderate/Rich density presets. The Legacy v1
generator (12-entry frozen corpus), Enhanced v2 generator (12-entry frozen
corpus), theme assets, and compiler profile remain unchanged.

### 20.2 Authorization

This profile is authorized by the Enhanced v3 architectural proof
(`DECISION-20260731-01`), the production authorization decision
(`DECISION-20260731-02`), and the later segmented explorer integration decision
(`DECISION-20260731-05`). The proof demonstrated 36/37 evidence rows PASS
with compiled spatial, budget, compatibility, and live-GPU validation.
The production pipeline and CLI integration are implemented. This living
contract records their owner re-review requirements and acceptance evidence.

### 20.3 Profile Identity

| field | value |
|-------|-------|
| `GenerationProfile` variant | `EnhancedV3` |
| production tag | `"m3"` |
| dispatch | `GenerationProfile::from_tag("m3")` → `Some(EnhancedV3)` |
| CLI | `dungeon_gen --class m3` |
| packaging | `engine_pack enhanced-dungeon-v3` |
| explorer | `dungeon_explore.sh --class m3` |
| production module | `src/bsp_generator/src/enhanced_v3/` |
| proof code (historical) | `src/bsp_generator/tests/enhanced_v3_proof/` |

The proof-only tag `"enhanced-v3"` is not a production dispatch token.
`from_tag("enhanced-v3")` returns `None`. The production tag `"m3"` is the
single authorized production dispatch token for this profile.

### 20.4 Vertical Contract

Identical to the Enhanced v2 vertical contract (§17.2, `DECISION-20260729-02`):

| parameter | frozen value |
|-----------|-------------|
| lower floor Z | 0 |
| upper floor Z | 192 |
| room height (both layers) | 176 |
| total Z span | 368 (≤ 384 M2_Z_MAX) |
| layer count | 2 |

### 20.5 Geometry Contract

| property | value |
|----------|-------|
| approved normals | cardinal (axis-aligned) and exact 45° diagonal in XY plane |
| wall thickness (cardinal) | 16 Quake units (1 quantum) |
| wall thickness (45° diagonal) | ≥ 16 Quake units perpendicular (32/√2 ≈ 22.63 at minimum diagonal span) |
| construction quantum | 16 Quake units |
| integer arithmetic only | no floating-point in authored geometry path |
| non-compliant normals rejected | 15°, 30°, arbitrary-angle, lattice-slope normals produce typed errors |

### 20.6 Room Contract

- **Chamfered/octagonal footprint**: rooms may use cardinal + 45° footprint families
- **Axis-aligned rectangular rooms** remain supported (compatible with v1/v2)
- **Minimum room outer span**: ≥ 112 Quake units (7 quanta; 80-unit clear interior)
- **Maximum room outer span**: ≤ 448 Quake units per axis (owner re-review 2026-08-01: intentionally revised from 256 to 448 for EnhancedV3 only; Legacy v1 and Enhanced v2 remain 112–256)
- **Balanced layer membership**: max difference of 1 between lower and upper layers
- **Shared XY occupancy grid**: no two rooms may overlap in XY projection

### 20.7 Portal Contract

- **Pointed-arch portal**: full-depth omitted shell volume through the wall, with
  64×80 swept clearance at the throat, preserved across the complete 192-unit tread
  run for stair transitions
- **Aperture ownership**: no separate opening brush; the shell wall's omission IS
  the portal; no decorative arch over an intact lintel
- **Cardinal-wall only**: pointed-arch portals on diagonal walls are deferred
- **Segmented-arch portal**: owner-authorized explorer-integrated override on
  cardinal walls. It preserves the complete 64×80 core and the historical two-band
  segmented recess. The corridor roof seals the first crown band through Z=112; a
  one-quantum cap immediately outside the remaining 32-unit centre opening seals the
  Z=112–128 arch/corridor interface. `V3Config::validate()`, CLI generation, and
  full-config package publication accept `ArchType::Segmented`.
- **Compatibility identity**: `ArchType::Pointed` remains the default. The segmented
  seal is emitted only for an explicit segmented override, so default and pointed map
  bytes remain unchanged.

### 20.8 Grounded Assembly Contract

- **Support graph**: acyclic directed graph with `Floor`, `Wall`, `Ceiling`,
  and `SupportedBy(instance)` edges
- **Contact**: coplanar shared-face (zero-volume) contact between supporting and
  supported brushes
- **Atomic dependent removal**: removing a parent removes all dependents
- **Overlap prohibition**: no two distinct assembly brushes may share positive-volume
  overlap
- **Convex validation**: every assembly brush must pass convexity, boundedness,
  positive volume, and minimum feature size checks

### 20.9 Preset Contract

Three density presets, with minima updated per the owner's 2026-10-19 re-review
authorizing the fully materialized production pipeline:

| preset | exact rooms | same-layer routes | minimum families | minimum assemblies | minimum features | face budget |
|--------|-------------|-------------------|------------------|--------------------|------------------|-------------|
| Sparse | 12 | 10 | 1 | 1 | 2 | 3,000 |
| Moderate | 20 | 20 | 3 | 3 | 6 | 5,000 |
| Rich | 28 | 30 | 6 | 6 | 12 | 8,000 |

Under-resourced configurations produce typed `MinimumIdentityFailure` errors,
never panics or empty output. Presets select features from the approved capability
set; they do not silently integrate deferred capabilities.

### 20.10 Approved Capabilities

| capability | integrated | focused evidence only | deferred |
|-----------|-----------|----------------------|----------|
| Chamfered/octagonal footprint | ✓ | — | — |
| Pointed-arch portal (cardinal) | ✓ | — | — |
| Grounded assembly | ✓ | — | — |
| Segmented-arch portal | ✓ (explorer override) | ✓ (historical focused fixture) | — |
| Grammar families (PortalChamber, ButtressedHall, ColumnGrove, FracturedVault, TerracedShrine, MonolithicChamber) | ✓ — real integrated feature generators | — | — |
| Twisted pillars | ✓ | — | — |
| Fractured ceilings | ✓ | — | — |
| Diagonal portals | — | — | ✓ |
| Lattice-slope walls (15°/30°) | — | — | ✓ |
| Concave rooms (T/L/alcove) | — | — | ✓ |

### 20.11 Theme Contract

- **cc0_dungeon_v2** theme at `src/bsp_generator/themes/cc0_dungeon_v2/`
- Identical to the Enhanced v2 theme — no new theme is created
- SHA-256 hashes of all theme assets match the frozen baseline
- `engine_pack enhanced-dungeon-v3` enforces the same normal/gloss companion
  completeness requirement as `enhanced-dungeon` for v2

### 20.12 RNG Contract

| property | value |
|----------|-------|
| domain separator | `"dungeon-gen/v3"` (UTF-8, prefixed before seed bytes) |
| seed byte order | little-endian `u64` (8 bytes) |
| framing | `SHA-256(domain_separator || seed_le_bytes || tag)` per semantic stage |
| output | 32-byte SHA-256 digest; consumed as a stream of little-endian `u64` values |

Frozen stage tags:

| tag | stage |
|-----|-------|
| `v3-placement` | two-layer room placement with 45° footprint support |
| `v3-topology` | topology and transition selection |
| `v3-features` | chamfered footprints, pointed arches, grounded assemblies |
| `v3-detail` | preset-driven feature density, pillar placement |

Renaming or reframing a tag is an output-version change for the v3 profile only.
v1 (`"dungeon-gen/v1"`, 4 tags) and v2 (`"dungeon-gen/v2"`, 6 tags) domains
remain frozen and independent. Same master seed across all three domains
produces cryptographically independent output streams.

### 20.13 Compiler Contract

Identical to the frozen BSP2 compiler profile (§8). v3 output targets the
same `ericw-q1-bsp2-generated` profile at
`tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml`. Deterministic
`-bsp2 -threads 1` compilation, warning-free `qbsp`/`vis`/`light` stages,
and strict reload with zero diagnostics are required. All corpus entries
must satisfy these criteria.

### 20.14 v3 Support Corpus

A 12-entry compiler matrix is frozen and validated:

| # | preset | seed | extent |
|---|--------|------|--------|
| 1–4 | Sparse | 0, 42, 99, 255 | 2048 |
| 5–8 | Moderate | 0, 42, 99, 255 | 2048 |
| 9–12 | Rich | 0, 42, 99, 255 | 3072 |

The corpus must satisfy:

- All three presets covered with 4 seeds each
- Deterministic byte-identical replay across independent runs
- Warning-free BSP2 compilation (zero qbsp/vis/light warnings)
- Strict reload with zero diagnostics
- Compiled face count < 10,000, entity count < 300
- Spatial witnesses: spawn, room centers, portal throats, corridor
  midpoints, stair treads, upper landing all non-solid
- All entries within M2 output ceilings (§5)

### 20.15 v1/v2 Compatibility Freeze

| contract | requirement |
|----------|------------|
| Legacy v1 12-entry corpus | byte-identical `.map` and metadata to frozen baseline |
| Enhanced v2 12-entry corpus | byte-identical `.map` and metadata to frozen baseline |
| `GenerationProfile::LegacyV1` tag | `"legacy-v1"` unchanged |
| `GenerationProfile::EnhancedV2` tag | `"enhanced-v2"` unchanged |
| v1 RNG domain | `"dungeon-gen/v1"` unchanged |
| v2 RNG domain | `"dungeon-gen/v2"` unchanged |
| cc0_stone_beta theme | SHA-256 unchanged |
| cc0_dungeon_v2 theme | SHA-256 unchanged |

Any drift in v1 or v2 corpus output is a blocking regression. The v3
profile is additive and structurally isolated — no production code changed
outside `src/bsp_generator/src/enhanced_v3/` and `src/bsp_generator/src/enhanced/profile.rs`
(the `GenerationProfile` enum addition).

### 20.16 Evidence Status

| contract | status | basis |
|----------|--------|-------|
| Architectural feasibility | **PASS** | 36/37 proof rows PASS; 1 NOT_RUN (tool limitation) |
| Production pipeline | **PASS** | `generate_v3()`, `run_pipeline()`, all 15 source modules in `src/bsp_generator/src/enhanced_v3/` |
| Cardinal + 45° geometry | **PASS** | Production output contains diagonal (45°) face lines; compiled fixtures compile warning-free |
| Pointed-arch portals | **PASS** | Production output: 64×80 swept clearance, pointed apex above rectangular core |
| Segmented-arch explorer override | **PASS** | Full 64×80 source throat, sealed 32-unit crown interface, exact Moderate seed-42 full-config package compiles warning-free and strict-loads through ericw |
| Grounded assemblies | **PASS** | Production pipeline produces acyclic support graphs; output brushes transitively supported |
| Grammar families — real integrated generators | **PASS** | All 6 families materialize grounded, family-distinct brushes in Rich preset |
| Twisted pillars | **PASS** | Production Rich preset includes twisted pillar features |
| Fractured ceilings | **PASS** | Production Rich preset includes fractured ceiling features |
| M2 budget compliance | **PASS** | All presets: faces < 10,000, entities < 300 |
| v1/v2 compatibility freeze | **PASS** | 24/24 entries byte-identical (12 Legacy + 12 Enhanced) |
| Deterministic output | **PASS** | Byte-identical .map and metadata across repeated generation |
| Live-WSI lifecycle | **NOT_RUN** | Headless draw capture is PASS, but no live-WSI lifecycle claim is made for this production repair. |
| Production code delivery | **PASS** | `V3Config`, `V3Preset`, `V3Error` (45 variants), `EnhancedV3Metadata`, CLI `--class m3 --preset`, all delivered and passing |
| v3 contract baseline | **PASS** | 25 frozen contract identity tests pass consistently |
| v3 12-entry compiler matrix | **PASS** | 12 entries (3 presets × 4 seeds 0/42/99/255): warning-free compilation, strict reload, spatial witnesses, budget compliance |
| Production acceptance | **PASS** | Preset identity minima, exact 12-entry source matrix, diagonal/pointed/stair/grammar evidence, budgets, and seed variation — all validated |
| Headless renderer capture | **PASS** | Production Sparse/Rich BSP2 artifacts rendered through strict engine-owned headless draw capture on RADV; fixed spawn/corridor/junction captures and targeted twisted-pillar, fractured-vault, and terraced-shrine captures are recorded in `.internal-dev/captures/enhanced-v3-production/manifest.md` |

### 20.17 Windowed In-Game Explorer GUI

The windowed `bsp_beta --m3-generate` path provides one in-game EnhancedV3
configuration overlay with two mutually exclusive interaction modes. Initial
startup configuration accepts `--seed`, `--preset`, exact room/corridor/loop
overrides, chamfer enable/disable, arch type, and a grammar-family allowlist.
An omitted seed derives from the current system time; the remaining defaults are
Moderate, chamfer enabled, pointed arches, and all six grammar families eligible.
The startup package and GUI draft must reflect the same resolved configuration.

Interaction modes:

- **F1 Keyboard mode**: arrows navigate, Enter edits or activates, Tab and
  Shift+Tab move between groups, Space toggles checkboxes, `+`/`-` adjust
  numeric fields, decimal keys edit numbers, and Escape closes. Keyboard input
  is consumed by the menu and every mouse interaction is discarded.
- **F2 Mouse mode**: pointer clicks select fields, exact dropdown options,
  steppers, checkboxes, and actions; the wheel scrolls bounded content. Mouse
  input is consumed by the menu and keyboard input is discarded except Escape
  and the globally consumed F1/F2 mode controls.
- **No menu**: the normal app-owned gameplay input route remains authoritative.
  Opening either menu queues releases for gameplay bindings and pauses FPS
  controller updates; closing restores gameplay without synthesizing presses.

The overlay exposes every public `V3Config` field: seed, preset, XY extent,
rooms, corridors, loops, vertical edges, chamfer, arch type, stairs, room-span
bounds, grammar allowlist and mode, all five feature flags, feature density,
minlight, and optional light count. The fixed 16-unit wall thickness is shown
as a disabled structural invariant and is not represented as a fabricated
configuration field.

`Generate` snapshots the complete validated draft and leaves the menu open.
`Apply & Close` closes only after its matching latest request successfully
publishes. Work remains off the event thread and follows the existing pipeline:
EnhancedV3 generation and `.map` staging, pinned ericw compilation and atomic
package publication, strict authorization, hidden renderer preparation,
coordinator validation/commit, and detached-mount retirement handoff. A failed
or stale request never replaces the active world. Successful generation shows
a two-second overlay or noncapturing title indication.

Source-level GUI, routing, pipeline, and binary tests pass. Timeout-bound
windowed startup proves startup, swapchain creation, BSP upload, and frame
recording in the task environment; automated F1/F2 interaction, resize,
minimize/restore, and surface-loss lifecycle evidence remain unexecuted and
must not be inferred from that smoke.

### 20.18 Re-Review Triggers

Any change to the following requires owner re-review:

- `GenerationProfile` variant set (adding, removing, or renaming a variant)
- v3 vertical contract (floor Z, upper Z, room height, layer count)
- v3 geometry contract (approved normal classes, thickness rules)
- v3 preset definitions (count, minimums, face budgets)
- v3 RNG domain separator or frozen stage tags
- v3 approved capability set (promoting a deferred capability)
- v1 or v2 corpus output (any byte drift is a blocking regression)
- cc0_dungeon_v2 theme asset changes

## 21. EnhancedV3RichnessV1 Contract (Freeze-Only — APPROVED 2026-08-02 via autonomous-delegation directive)

### 21.1 Scope

The EnhancedV3RichnessV1 contract is a separate immutable contract domain
`dungeon-gen/v3-richness/v1` that extends EnhancedV3 output with gameplay-
relevant content (archetypes, props, lighting recipes, theme variation,
cave cells, multi-storey vertical openings, ladder/drop controller
semantics). It is additive to — and structurally isolated from — the
baseline v3 generation profile. No Richness implementation code, CLI, GUI,
or public surface is authorized by this freeze; this section records the
contract that subsequent sprint phases must implement.

### 21.2 Profile Identity

| field | value |
|-------|-------|
| domain separator | `"dungeon-gen/v3-richness/v1"` |
| framing | `SHA-256(domain \|\| seed_le_bytes \|\| stage_tag)` |
| relationship to v3 | additive extension; v3 map geometry byte stream is unchanged |
| baseline-v3 manifest | remains `enhanced-v3-corpus/v1` — not renamed to imply Richness |
| algorithm revision | `enhanced-v3-richness-algorithm/v1` |
| content revision | `enhanced-v3-richness-content/v1` |
| preset revision | `enhanced-v3-richness-presets/v1` |
| theme revision | `enhanced-v3-richness-themes/v1` |
| asset revision | `enhanced-v3-richness-assets/v1` |
| convention revision | `enhanced-v3-richness-conventions/v1` |
| v3 RNG | `"dungeon-gen/v3"` domain independent from `"dungeon-gen/v3-richness/v1"` |

### 21.3 Request and Profile Tag Strategy

| concept | value |
|---------|-------|
| request tag | `"richness-v1"` — reserved; `GenerationProfile::from_tag` returns `None` until final owner-authorized exposure |
| profile variant | `GenerationProfile::EnhancedV3RichnessV1` — planned additive enum member, not yet created |
| dispatch | `dungeon_gen --class richness-v1` is reserved and rejected until final exposure |
| packaging | `engine_pack enhanced-dungeon-v3-richness` is reserved and rejected until final exposure |
| explorer | `dungeon_explore.sh --class richness-v1` is reserved and rejected until final exposure |
| implementation domain | `src/bsp_generator/src/enhanced_v3_richness/` (not yet created) |
| baseline V3Config | frozen — Richness must not alter `V3Config`, `V3Preset`, or `NormalClass` |
| tags | baseline M1/M2 tags, CLI documents, and package manifests remain frozen |

### 21.4 Same-XY Multi-Storey Reservation Rules

Richness V1 introduces explicit composite reservations that may occupy the same
XY projection across both layers. This is a controlled exception to the
baseline projection-exclusive rule:

| rule | value |
|------|-------|
| composite reservation type | `RichnessCompositeReservation` owning both lower and upper layers |
| connecting void | every intervening Z span between reserved layers is explicitly owned |
| ordinary baseline footprints | remain projection-exclusive (unchanged from §20.6) |
| overlap prohibition | two distinct composite reservations must not overlap in XY |
| composite-vs-ordinary | a composite reservation and an ordinary footprint must not overlap in XY |
| stair transitions | composite reservations may host vertical openings; baseline stair reservations remain independent |

### 21.5 Vertical Openings

| concept | value |
|---------|-------|
| opening types | ladder shaft, drop hole, open stairwell, spiral-stair opening |
| minimum clearance | 64×64 XY, 80-unit Z headroom at every standing surface |
| ladder semantics | compiler-preserved explicit brush-AABB descriptor → app-owned `BspPlayerMovementController` climb state; no surface-flag dependency |
| drop controller | compiler-preserved explicit brush-AABB descriptor → app-owned fixed-step one-way-drop state and player-hull landing trace |
| moving brush models | forbidden; Richness V1 does not emit `func_plat` or any cargo lift |
| safety margin | 16-unit Quake hull radius preserved around every vertical opening boundary |

### 21.6 Ladder/Drop Controller Semantics

| concept | value |
|---------|-------|
| controller boundary | `BspPlayerMovementController` in `apps/bsp_beta/src/player_navigation.rs`; app-owned, not generator-owned |
| integration target | shipped `apps/bsp_beta` fixed-step loop for maps carrying revision-qualified Richness descriptors |
| collision | origin line trace through compiler-preexpanded stored player hull 1 |
| ladder entry | origin inside compiled descriptor AABB, positive forward input, and horizontal wish dot entry normal ≥ 0.5 |
| ladder behavior | constant 1.5 engine units/s; forward/back maps to up/down; no gravity/lateral authority; retained horizontal velocity restored on exit |
| ladder exits | top, bottom, jump, volume loss, teleport, or regeneration; exact states and velocities are frozen in `bsp-spatial-physics.md` §11 |
| ladder overlap | priority descending, then compiled entity order, then stable ID |
| drop behavior | entry horizontal direction retained; subsequent input ignored; gravity/terminal velocity applied until an eligible lower landing |
| drop non-return | landing must be ≥32 Quake units below entry; characterization proves ordinary jump cannot regain upper platform |
| reset | teleport/external synchronization and generation replacement clear velocity, active ID, diagnostics, and state to `Airborne` |

### 21.7 Cave Eligibility

| concept | value |
|---------|-------|
| cave cell type | `CaveCell` — additional non-room navigable cell |
| eligibility predicate | seed-and-layout-dependent, not theme-dependent |
| minimum cave cells per Richness preset | 2 (Sparse/Moderate), 4 (Rich) |
| maximum cave cells | 6 |
| cave BSP role | carved subtractive void with unadorned stone trim |
| cave lighting | minimum ambient `_minlight 32` within cave bounds |

### 21.8 Theme Invariance

| rule | value |
|------|-------|
| blueprint candidate keys | invariant under theme selection |
| cave decisions | theme must not gate, suppress, or redirect cave placement |
| archetype selection | theme colors/skins archetypes but does not gate their eligibility |
| route topology | invariant under theme selection |
| assembly graph | invariant under theme selection |
| compliance gate | a v3 blueprint regenerated with a different theme must produce identical `.map` geometry bytes |

### 21.9 Exact Content Counts

| category | count |
|----------|-------|
| archetypes (gameplay-bearing prefabs) | exactly 30 shared semantic archetypes |
| props (decorative non-collidable entities) | exactly 15 shared semantic props |
| lighting recipes | exactly 12 shared semantic recipes |
| themes | 3 (CC0 Dungeon v2, plus two additional project-authored CC0 themes) |
| cave cell archetypes | 4 (spawn cave, dead-end gallery, junction grotto, transition tunnel) |
| vertical opening archetypes | 4 (ladder shaft, drop hole, open stairwell, spiral-stair opening) |

### 21.10 Source and Compiled Budgets

| metric | budget |
|--------|--------|
| Richness entity ceiling | < 500 (extends M2 baseline) |
| Richness compiled face ceiling | < 15,000 (extends M2 baseline) |
| Richness light entity ceiling | < 100 |
| Richness static batch ceiling | < 800 |
| per-archetype source brush budget | < 64 brushes |
| per-prop source brush budget | < 16 brushes |
| per-theme WAD budget | < 64 miptex identities |
| per-theme companion PNG budget | < 128 files (basecolor × normal × gloss per identity) |

### 21.11 Stable Errors

| error variant | condition |
|---------------|----------|
| `RichnessThemeAssetMissing` | a required theme WAD, palette, or companion PNG is absent at generation time |
| `RichnessArchetypeBudgetExceeded` | per-archetype brush count exceeds budget |
| `RichnessPlacementExhausted` | no valid placement found within bounded attempts |
| `RichnessCaveEligibilityFailed` | seed/layout cannot place minimum cave cells |
| `RichnessVerticalOpeningBlocked` | composite reservation rejected due to conflict with ordinary footprint |
| `RichnessThemeInvarianceViolated` | theme selection altered blueprint candidate keys or cave decisions |

All errors are typed, never panics, and never produce partial output. A
`RichnessThemeInvarianceViolated` error is a generator-level assertion failure
and blocks publication.

### 21.12 Atomic Public Exposure

| rule | value |
|------|-------|
| `GenerationProfile::EnhancedV3RichnessV1` | exposed only when all RichnessV1 acceptance gates pass |
| `from_tag("richness-v1")` | returns `None` until the acceptance gate is approved |
| CLI dispatch | `dungeon_gen --class richness-v1` rejected with "not yet available" until approved |
| package command | `engine_pack enhanced-dungeon-v3-richness` rejected until approved |
| public API surface | no `RichnessConfig`, `RichnessPreset`, or `RichnessMetadata` type exposed until approved |

### 21.13 Re-Review Triggers

Any change to the following requires owner re-review:

- RichnessV1 domain separator, revision identifiers, or stage tag spelling
- Same-XY multi-storey reservation rules
- Cave eligibility predicate
- Theme invariance rule (particularly the compliance gate)
- Exact content counts (archetypes/props/lighting recipes/themes)
- Source or compiled budgets
- Stable error variants
- Atomic public exposure gating policy
- Vertical opening clearance minimums or ladder/drop controller semantics

### 21.14 Phase 05 Convention Qualification (FROZEN 2026-08-02)

The sealed convention and controller fixtures compile twice through pinned
ericw-tools 2.0.0-alpha3 with byte-identical BSP/LIT output. Every stage is
warning-free; leaks, skipped fill, missing textures, malformed output, and
strict-load diagnostics fail closed. The convention fixture proves an
authorized lowercase `hint`/`hintskip` sloped split, visible-tool-surface
omission, independent solid-`skip` collision, independent player-hull `clip`
collision while omitted from rendering, detail consumption,
colored QLIT output, and exact preservation behavior for custom climb/drop entities.
`_tb_id` remains unsupported and uses structural fingerprints. The frozen
result table is `bsp-spatial-physics.md` §11.7.

The convention test stages `hint`, `hintskip`, and `clip` as deterministic
compiler-only derivatives of the existing authorized CC0 `skip` miptex. Only
the miptex and WAD directory identity fields differ. These assets are scoped
to the offline qualification fixture, are never visible, have no PBR
companions, and do not revise production theme selection or geometry.

The controller fixture drives the actual
`BspPlayerMovementController::fixed_step` boundary invoked from the shipped
`apps/bsp_beta` loop, not `PlayerMover` test helpers. It freezes the 60 Hz
standard movement constants and all ladder, overlap, drop, reset, collision,
and non-return semantics in `bsp-spatial-physics.md` §11.

The measurement method in
`tools/engine_pack/tests/enhanced_v3_richness_measurement.rs` is fail-closed:
it canonicalizes and hashes corpus order, excludes labeled warmups, selects
median while also recording min/max/mean/nearest-rank p95/population standard
deviation, requires generation/qbsp/vis/light/parse/extract/load/render stage
metrics, all 15 BSP lumps, artifact hashes/bytes, package bytes,
faces/entities/batches, process/runtime memory, and validates a closed JSON
schema before atomic publication. Measurement observations are canonicalized
post-generation and cannot enter the generation-decision boundary. These are
method rules only: Richness ceilings remain unfrozen until Richness maps exist.

`engine_pack` concurrently drains stdout/stderr under one combined output
ceiling, applies a bounded timeout, places each process in a Unix process
group, terminates the whole group on timeout/output breach and after direct
child exit, and joins both reader threads. Adversarial tests cover hangs,
combined noisy streams, nonzero exits, descendant termination, and the
successful-parent/inherited-pipe case.
