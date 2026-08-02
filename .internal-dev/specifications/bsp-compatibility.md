---
schema_version: 1
document_type: compatibility-specification
status: active
owner: bsp-beta
created: 2026-07-23
approval: evidence-finalized — Phase 05 convention, compiler-supervisor, and measurement method cells frozen 2026-08-02; production Richness profile remains gated
---

# BSP Compatibility Specification

## 1. Scope and Authority

This specification defines the exact BSP profile, semantics, ownership, limits, companion-file binding, entity grammar, diagnostic categories, and fallback policy for the engine's Q1-family BSP beta. Every supported cell requires fixture evidence; every unsupported cell has explicit status. No code implementation is authorized without an approved cell in this matrix.

## 2. Approved Profile

### 2.1 Profile Identity

| field | value |
|-------|-------|
| canonical name | `q1-portable-ericw` |
| display name | Quake 1 BSP29 + ericw-tools QBSP2/Vis/Light |
| dialect family | Quake 1 (id Tech 1) |
| status | beta |
| compiler identity | `ericw-tools` release `2.0.0-alpha3` (pinned) |
| compiler executables | `qbsp`, `vis`, `light` (ericw-tools build) |
| TrenchBroom game config | `Quake` / `Quake (ericw-tools)` |
| FGD baseline | `quake.fgd` (project-authored subset) |
| compiler invocation | shell-free via `build_fixtures.py` subprocess |
| minimum compiler version | `ericw-tools 2.0.0-alpha3` or exact pinned hash |
| profile identity checks | compiler executable SHA-256, version string match, supported extension enumeration |
| `engine_pack` publication warning policy | any `qbsp`, `vis`, or `light` warning (including missing textures or skipped fill) is `CompilerWarning`; no artifact set is published |

### 2.2 Portable Core — BSP29

| field | value |
|-------|-------|
| magic | little-endian `29` (0x1D 0x00 0x00 0x00) |
| version | 29 (int32 LE) |
| header size | 124 bytes (4-byte version + 15 × 8-byte lump descriptors) |
| lump count | 15 standard lumps (indices 0–14) |
| lump alignment | 4-byte (entities), 1-byte (others) |
| field widths | all offsets and sizes are `i32` LE |

#### Standard Lump Layout (BSP29)

| index | name | element size | notes |
|-------|------|-------------|-------|
| 0 | Entities | variable (null-terminated string) | UTF-8 with Latin-1 fallback |
| 1 | Planes | 20 bytes | `(normal: vec3 f32, dist: f32, type: i32)` |
| 2 | Miptex | variable | embedded texture lump |
| 3 | Vertices | 12 bytes | `(x, y, z: f32)` |
| 4 | Visinfo | variable | compressed visibility data |
| 5 | Nodes | 24 bytes | `(plane: i32, children[2]: i16, mins/maxs: i16×3 each, face_id: u16, face_num: u16)` |
| 6 | Texinfo | 40 bytes | `(vecS, distS, vecT, distT: f32×4, miptex: u32, flags: u32)` |
| 7 | Faces | 20 bytes | `(plane: u16, side: u16, ledge: u32, ledge_num: u16, texinfo: u16, styles[4]: u8, lightofs: i32)` |
| 8 | Lightmaps | 1 byte per luxel per valid style | monochrome grayscale; decoded to RGB by replication |
| 9 | Clipnodes | 8 bytes | `(plane: i32, children[2]: i16)` |
| 10 | Leaves | 28 bytes | `(contents: i32, visofs: i32, mins/maxs: i16×3, mark: u16, markleaf: u16, ambient: u8[4])` |
| 11 | Markfaces | 2 bytes | `u16` face indices |
| 12 | Edges | 4 bytes | `(v[2]: u16)` |
| 13 | Surfedges | 4 bytes | `i32` (negative = reverse edge) |
| 14 | Models | 64 bytes | `(mins/maxs: f32×3, origin: f32×3, headnode: i32[4], visleafs: i32, face_id, face_num: i32)` |

### 2.3 Named TrenchBroom Profile — ericw-tools BSP2

| field | value |
|-------|-------|
| magic | little-endian `BSP2` (0x42 0x53 0x50 0x32) |
| version | `BSP2` magic + BSP29-format lumps with conditionally extended field widths |
| header size | 124 bytes (4-byte magic/version + 15 × 8-byte lump descriptors) |
| lump count | 15 standard lumps (indices 0–14); BSPX extensions, if present, are appended outside this header |
| field widths | offsets are `i32` LE, sizes are `i32` LE |
| key difference from BSP29 | edges use 32-bit vertex indices unconditionally (4→8 bytes); nodes, leaves, and clipnodes use uniformly widened fields in ericw-tools BSP2; faces, markfaces, surfedges use 32-bit indices when the per-lump element count exceeds the 16-bit limit |

#### BSP2 Field Widths — Fixture-Verified (ericw-tools 2.0.0-alpha3)

Verified via authoritative `bspinfo` tool and byte inspection of `ericw-bsp2-colored.bsp`.

| lump | BSP29 width | BSP2 width (ericw-tools 2.0.0-alpha3) | verification |
|------|-----------|--------------------------------------|-------------|
| vertices (3) | `f32×3` = 12 bytes | `f32×3` = 12 bytes (same) | 12/1 = 12 ✓ |
| edges (12) | vertex indices `u16×2` = 4 bytes | vertex indices `u32×2` = 8 bytes (always) | 8/1 = 8 ✓ |
| surfedges (13) | edge index `i32` = 4 bytes | `i32` = 4 bytes (unchanged width; refers to u32 edge pool) | 0/0 = N/A (empty) |
| faces (7) | plane u16(2), side u16(2), ledge u32(4), ledge_num u16(2), texinfo u16(2), styles[4] u8(4), lightofs i32(4) = 20 bytes | plane u32(4), side u16→u32(4), ledge u32(4), ledge_num u32(4), texinfo u16→u32(4), styles[4] u8(4), lightofs i32(4) = 28 bytes (when non-empty) | 0/0 = N/A (empty) |
| markfaces (11) | face index `u16` = 2 bytes | face index `u32` = 4 bytes (when non-empty) | 0/0 = N/A (empty) |
| leaves (10) | contents i32(4), visofs i32(4), mins i16×3(6), maxs i16×3(6), mark u16(2), markleaf u16(2), ambient u8[4](4) = 28 bytes | contents i32(4), visofs i32(4), mins i32×3(12), maxs i32×3(12), mark u32(4), markleaf u32(4), ambient u8[4](4) = **44 bytes** | 308/7 = 44 ✓ |
| nodes (5) | plane i32(4), children[2] i16(4), mins i16×3(6), maxs i16×3(6), face_id u16(2), face_num u16(2) = 24 bytes | plane i32(4), children[2] i32(8), mins i32×3(12), maxs i32×3(12), face_id u32(4), face_num u32(4) = **44 bytes** | 264/6 = 44 ✓ |
| clipnodes (9) | plane i32(4), children[2] i16(4) = 8 bytes | plane i32(4), children[2] i32(8) = **12 bytes** | 288/24 = 12 ✓ |
| models (14) | 64 bytes (face_id/face_num are i32 in BOTH BSP29 and BSP2) | 64 bytes (same) | 64/1 = 64 ✓ |

**Detection rule**: BSP2 field widths are determined per-lump from authoritative element counts. The parser must compute element width as `(lump_size / element_count)`. For empty lumps (faces, markfaces, surfedges), use the known widened width as default. The `bspinfo` tool output is authoritative for element counts.

### 2.4 Companion Formats

| format | extension | version/constraints | precedence |
|--------|-----------|-------------------|------------|
| Colored light data | BSPX `RGBLIGHTING` lump | 3 bytes per base lightmap luxel/style, matching the monochrome lightmap layout after RGB expansion | overrides monochrome lighting lump |
| External light data | `.lit` | `QLIT` + version `1` 8-byte header, then RGB payload with the same luxel/style count as BSP | lower precedence than BSPX; content-mismatch diagnosed |
| WAD2 texture archive | `.wad` | Quake WAD2 format, 4-mip miptex entries | loose replacement textures preferred |
| Palette | `.lmp` | 768 bytes (256 × RGB), raw binary | required; engine does not embed default Quake palette |
| External normal map | `<texture>_norm.png` | PNG, dimensions exactly match resolved base texture; tangent-space R/G encode X/Y and positive Z is reconstructed | optional; either PBR companion opts eligible surfaces into PBR |
| External gloss map | `<texture>_gloss.png` | PNG, dimensions exactly match resolved base texture; red channel is gloss and `roughness = 1 - gloss` | optional; either PBR companion opts eligible surfaces into PBR |

#### Companion Precedence and Mismatch

Deterministic resolution order for colored lighting:

1. Explicit package manifest choice (if configured).
2. Supported, content-compatible BSPX `RGBLIGHTING` lump.
3. Supported, content-compatible package-provided `.lit` file.
4. Base monochrome lighting lump (BSP29 lump 8).

Conflicting valid sources are diagnosed at strict policy level. Mismatch (wrong RGB payload length relative to the base lightmap luxel/style count, invalid header, version mismatch) for a `.lit` companion is a `CompanionContentMismatch` diagnostic at policy-selected severity.

#### Resource Resolution Order (Textures)

1. Explicit package mapping/override.
2. Loose replacement texture + companion PBR maps in configured package roots.
3. Embedded miptex (lump 2).
4. Sanitized WAD basename lookup in configured WAD roots.
5. Diagnostic fallback for a structurally valid but unresolved source slot in either import mode.

#### External PBR Companion Binding

- Filenames derive from the sanitized BSP miptex identity, not from a replacement path or BSP filename.
- Matching is exact-case first, then ASCII case-insensitive; configured/package root order remains authoritative.
- Package discovery is confined through `PackageResolver`; neutral extraction accepts only caller-authorized owned bytes and performs no filesystem I/O.
- Eligible surface classes are opaque and alpha mask. Sky, liquid, nodraw, and tool surfaces keep their existing specialized behavior.
- A normal-only material defaults gloss to 0 (fully rough). A gloss-only material defaults to a flat tangent-space normal.
- Malformed PNGs and dimension mismatches fail renderer preflight before GPU allocation. Missing files are optional and preserve the legacy material route.
- Companion content hashes participate in BSP cache identity.

## 3. Limits

### 3.1 Structural Limits

| limit | BSP29 | BSP2 | diagnostic on exceed |
|-------|-------|------|---------------------|
| max vertices | 65,535 | 2³¹‑1 | `StructuralVertexCount` |
| max edges | 65,535 | 2³¹‑1 | `StructuralEdgeCount` |
| max surfedges | 65,535 | 2³¹‑1 | `StructuralSurfedgeCount` |
| max faces | 65,535 | 2³¹‑1 | `StructuralFaceCount` |
| max markfaces | 65,535 | 2³¹‑1 | `StructuralMarkfaceCount` |
| max nodes | 32,767 | 2³¹‑1 | `StructuralNodeCount` |
| max leaves | 8,191 (markface index) | 2³¹‑1 | `StructuralLeafCount` |
| max clipnodes | 32,767 | 2³¹‑1 | `StructuralClipnodeCount` |
| max models | 256 (index 255 = reserved) | 2³¹‑1 | `StructuralModelCount` |
| max light styles | 4 per face | 4 per face | `UnsupportedStyleSlot` |
| max entity string length | 2²⁰ bytes (1 MiB) | 2²⁰ bytes | `EntityStringTooLarge` |

### 3.2 Aggregate Allocation Budgets

| budget | limit | diagnostic |
|--------|-------|------------|
| total lump allocation | 2²⁸ bytes (256 MiB) | `AllocationExceeded` |
| total face vertex allocation | 2²⁴ bytes (16 MiB) | `AllocationExceeded` |
| total entity count | 2¹⁶ (65,536) | `EntityCountExceeded` |
| total texture count (miptex) | 2¹² (4,096) | `TextureCountExceeded` |
| WAD entry count per archive | 2¹² (4,096) | `WadEntryCountExceeded` |

## 4. Magic and Version Detection

```rust
// Detection is exact: magic bytes must match, version must match.
const BSP29_MAGIC: u32 = 29; // LE bytes: [0x1D, 0x00, 0x00, 0x00]
const BSP2_MAGIC: [u8; 4] = *b"BSP2"; // LE bytes: [0x42, 0x53, 0x50, 0x32]
```

- If the first `i32` LE is 29 → BSP29 path.
- If the first 4 bytes are `BSP2` → BSP2 path.
- All other magic values → `UnsupportedDialect` diagnostic. No guessing.
- The engine never attempts BSP30 (HL, magic=30), BSP38 (Q2, `"2PSB"`), or BSP46 (Q3/IBSP, `"IBSP"`). These are distinct products and produce `UnsupportedDialect`.

## 5. Entity Grammar

### 5.1 Format

Entities are null-terminated strings with:

```
{
"classname" "value"
"key" "value"
}
{
"classname" "value2"
...
}
```

### 5.2 Parsing Rules

| rule | behavior |
|------|----------|
| encoding | UTF-8 preferred; Latin-1 bytes are preserved (non-UTF-8 is diagnosed) |
| key ordering | preserved from source |
| duplicate keys | preserved, all values recorded; typed singleton access uses last-value-wins with `DuplicateKey` diagnostic |
| unknown keys | preserved, no failure |
| unknown classnames | classified as `EntityClass::Unknown`, preserved in raw, treated as generic tagged nodes |
| token quoting | double-quoted values with `\"` escape and `\n` newline support |
| unquoted values | rejected as `EntityTokenUnquoted` error |
| unterminated entity | rejected as `EntityUnterminated` error |
| nested braces | rejected as `EntityNestedBraces` error |
| empty classname | `EntityClasslessWithKeys` warning (preserved) |
| key-only (no value) | `EntityValueMissing` error |
| zero-length entity `{}` | `EntityEmpty` diagnostic, preserved as empty node |

### 5.3 Recognized Entity Classes

The BSP crate classifies only format-level concepts:

| classification | classname patterns | produces |
|----------------|-------------------|----------|
| worldspawn | `worldspawn` | world model, BSP tree root, world collision |
| light | `light`, `light_fluoro`, `light_flame_large_yellow`, `light_torch_small_walltorch`, etc. | light descriptor |
| point entity | any non-worldspawn, non-brush-model entity | generic node with origin/angle |
| inline brush model | `func_door`, `func_button`, `func_plat`, `func_wall`, `func_illusionary`, etc. | inline model node + collider recipe |
| trigger | `trigger_once`, `trigger_multiple`, `trigger_push`, etc. | trigger collider recipe |
| spawn marker | `info_player_start`, `info_player_deathmatch`, `info_teleport_destination` | spawn marker node |
| unknown | anything not matched above | preserved generic entity |

## 6. Explicit Exclusions

The following format families are **excluded** and produce `UnsupportedDialect`:

| dialect | magic | reason |
|---------|-------|--------|
| Half-Life BSP30 | version 30 | distinct product |
| Quake 2 BSP38 | `"2PSB"` | distinct product |
| Quake 3 / IBSP BSP46 | `"IBSP"` | distinct product |
| Valve/Source VBSP | varied | distinct product |
| TrenchBroom `.map` source | N/A | runtime loading not supported |
| external entity `.ent` sidecars | N/A | not in approved profile |
| external visibility `.vis` sidecars | N/A | not in approved profile |

## 7. Diagnostic Categories

### 7.1 Stable Diagnostic Codes

Every diagnostic carries a stable machine-readable code (not only message text).

| code | category | severity (dev) | severity (strict) | description |
|------|----------|---------------|-------------------|-------------|
| `BSP-UNSUPPORTED-DIALECT` | unsupported compatibility | error | error | magic/version not in approved profile |
| `BSP-STRUCT-CORRUPT-LUMP` | structural corruption | error | error | lump offset/size invalid, overlap, truncation |
| `BSP-STRUCT-CORRUPT-INDEX` | structural corruption | error | error | cross-lump index out of valid range |
| `BSP-STRUCT-CORRUPT-CYCLE` | structural corruption | error | error | cyclic tree/leaf graph |
| `BSP-STRUCT-CORRUPT-OVERFLOW` | structural corruption | error | error | integer overflow in count/layout arithmetic |
| `BSP-STRUCT-CORRUPT-ALIGNMENT` | structural corruption | error | error | required alignment violated |
| `BSP-STRUCT-CORRUPT-ENTITY` | structural corruption | error | error | entity string structurally malformed |
| `BSP-STRUCT-CORRUPT-FACE` | structural corruption | error | error | face winding/plane/texinfo invalid |
| `BSP-SECURITY-PATH-TRAVERSAL` | security | error | error | path escape attempt in resource reference |
| `BSP-SECURITY-SYMLINK-ESCAPE` | security | error | error | symlink escape in package root |
| `BSP-SECURITY-DEVICE-FILE` | security | error | error | non-regular file at expected resource path |
| `BSP-COMPAT-UNSUPPORTED-EXT` | unsupported compatibility | warning | error | unknown BSPX extension name |
| `BSP-COMPAT-AMBIGUOUS-EXT` | unsupported compatibility | warning | error | conflicting valid extensions |
| `BSP-COMPAT-COMPANION-VERSION` | unsupported compatibility | warning | error | companion file version unsupported |
| `BSP-COMPAT-COMPANION-MISMATCH` | unsupported compatibility | warning | error | companion data mismatch (e.g., luxel count) |
| `BSP-COMPAT-STALE-COMPANION` | unsupported compatibility | warning | warning | companion file exists but hash doesn't match expected |
| `BSP-MISSING-REQUIRED-PALETTE` | missing required | error | error | no palette available |
| `BSP-MISSING-REQUIRED-WAD` | missing required | error | error | referenced WAD not found in configured roots |
| `BSP-MISSING-REQUIRED-MODEL` | missing required | warning | error | external model not found in release mappings |
| `BSP-MISSING-REQUIRED-LIGHTMAP` | missing required | error | error | face has no valid lightmap data |
| `BSP-FALLBACK-DEFAULT-PALETTE` | optional fallback | warning | warning | using project palette fallback |
| `BSP-FALLBACK-EMBEDDED-MIPTEX` | optional fallback | warning | warning | using embedded miptex for texture |
| `BSP-FALLBACK-DIAGNOSTIC-TEXTURE` | optional fallback | warning | warning | using diagnostic checkerboard for missing texture |
| `BSP-FALLBACK-MISSING-LIGHTMAP` | optional fallback | warning | warning | face missing lightmap, using fullbright fallback |
| `BSP-ENTITY-UNKNOWN-CLASS` | unknown app entity | info | info | entity classname not recognized by engine |
| `BSP-ENTITY-DUPLICATE-KEY` | authoring quality | info | info | duplicate key in entity |
| `BSP-ENTITY-EMPTY` | authoring quality | info | info | empty entity `{}` |

### 7.2 Severity Policy

- **Development mode** (`strict = false`): optional fallbacks and unknown app entities are allowed. Unsupported compatibility becomes `warning`. Security and structural corruption remain `error`.
- **Strict / release mode** (`strict = true`): unsupported compatibility, security, structural corruption, and missing required resources without an approved fallback are `error`. Expected custom entities remain preservable data (not import failures).

**Owner-authorized generated-dungeon visual override (2026-07-26):** a renderable source miptex slot that is structurally valid but unresolved by its authorized WADs produces `BSP-FALLBACK-DIAGNOSTIC-TEXTURE` in both modes. A visible face with no lightmap offset produces `BSP-FALLBACK-MISSING-LIGHTMAP` and renders unlit. These warnings preserve a concrete material and draw path for generated BSPs; malformed slots, bad indices, missing palettes, security violations, and all other structural failures remain errors. This override is not compiler-publication evidence.

## 8. Whole-Asset vs Conservative-Subsystem Fallback

| failure class | policy |
|---------------|--------|
| invalid magic/version | whole-asset rejection |
| lump truncation/overlap | whole-asset rejection |
| corrupt entity string | whole-asset rejection |
| missing palette (strict) | whole-asset rejection |
| missing palette (dev) | whole-asset rejection for textured maps; the required palette is never substituted with a test/default palette |
| unresolved structurally valid WAD/miptex source slot (strict or dev) | conservative diagnostic-texture fallback (`BSP-FALLBACK-DIAGNOSTIC-TEXTURE`) |
| visible face missing a lightmap offset (strict or dev) | conservative unlit fallback (`BSP-FALLBACK-MISSING-LIGHTMAP`) |
| missing .lit (strict) | conservative fallback: monochrome lighting (diagnosed) |
| missing .lit (dev) | conservative fallback: monochrome lighting (diagnosed) |
| missing PBR texture companion | no diagnostic required; legacy BSP lightmapped rendering remains unchanged |
| malformed or dimension-mismatched PBR texture companion | whole-mount rejection during renderer preflight; no partial PBR fallback |
| corrupt VIS | conservative fallback: PVS disabled, frustum/BVH culling only |
| corrupt clipnodes | whole-asset rejection (collision is structural) |
| corrupt face winding | conservative fallback: face omitted from geometry (diagnosed) |
| invalid texinfo index | conservative fallback: face uses diagnostic texture |
| unreferenced lumps | no failure; diagnosed as authoring quality |

Structural geometry (planes, face winding, node graph cycles) and collision (clipnodes) corruption is never handled by silently dropping individual elements — it produces `BSP-STRUCT-CORRUPT-*` errors.

## 9. Fixture Evidence Matrix

Each supported cell must have a fixture. Fixtures are in `src/bsp/tests/fixtures/`.

### 9.1 Source Map Fixtures

| fixture | source | covers |
|---------|--------|--------|
| `q1_profile_core.map` | project-authored | BSP29 headers, entities, worldspawn, basic brush source geometry, embedded textures; visible face/lightmap/sky/liquid compiled coverage is blocked by `__TB_empty` output |
| `q1_profile_structural.map` | project-authored | doors, buttons, platforms, triggers, targets, inline models, target/targetname graph |
| `q1_profile_spatial.map` | project-authored | visibility boundaries, collision shapes, hull traces, contents regions |
| `dungeon_evidence_standard.map` | project-authored | sealed visible BSP2 compiler-output proof using project-authored WAD2 textures `DNGN01`/`DNGN02`, nonempty lightdata, and external QLIT v1 `.lit` output |

### 9.2 Compiled Fixtures

| fixture | source | compiler | expected profile cells |
|---------|--------|----------|----------------------|
| `q1-bsp29-core.bsp` | `q1_profile_core.map` | ericw-tools qbsp/vis/light | BSP29 magic/header, texture lump, entities, PVS bytes, clipnodes; visible faces/lightmaps/sky/liquids blocked (compiled output has zero faces/lightdata). **Reclassified**: this is a zero-face structural fixture, not a visible/lightmapped fixture. |
| `q1-bsp29-visible.bsp` | `q1_profile_visible.map` | ericw-tools qbsp/vis/light | **Reclassified**: derived (post-processed) renderer fixture. `build_fixtures.py` patches texinfo and lightdata into the compiler output after the compiler run. This fixture proves renderer integration paths but is NOT primary compiler-output evidence for face layout, lightmap layout, or BSPX colored-light output. Valid for: parser acceptance of face-visible BSP29, renderer upload paths, lightmap atlas layout (post-processed), and headless capture validation. Not valid for: compiler provenance of visible geometry, unmodified compiler output claims. |
| `ericw-bsp2-colored.bsp` | `q1_profile_core.map` | ericw-tools qbsp (BSP2) + light -bsp2 -lit | BSP2 magic/header, texture lump, entities, PVS bytes, clipnodes, BSP2 field widths (nodes 44B, leaves 44B, clipnodes 12B, edges 8B); colored lightmaps/BSPX blocked (compiled output has zero faces/lightdata). **Reclassified**: this is a zero-face structural fixture, not a colored-light or face-layout fixture. |
| `dungeon-evidence-bsp2.bsp` | `dungeon_evidence_standard.map` + `dungeon_evidence.wad` | ericw-tools qbsp `-bsp2`, vis, light `-threads 1 -lit` | Unmodified compiler-produced visible BSP2 proof: 41 faces at 28 B, 55 markfaces at 4 B, 4,256 B lightdata, 3 entities, strict parser reload with 0 diagnostics. Duplicate `engine_pack compile-bsp --wad` runs are byte-identical and match the checked-in fixture. |

### 9.3 Companion Fixtures

| fixture | format | paired with | content |
|---------|--------|-------------|---------|
| `project_palette.lmp` | 768-byte raw RGB palette | all maps | project-authored, CC0-licensed palette (not Quake-derived) |
| `ericw-bsp2-colored.lit` | QLIT v1 (8 bytes) | `ericw-bsp2-colored.bsp` | **Reclassified**: empty companion-path fixture. Contains the 8-byte QLIT header (`QLIT` magic + version `1`) with zero RGB payload bytes. Proves the companion-path resolution and empty-lit handling but is NOT colored-light evidence. Valid for: companion-file discovery, `.lit` header validation, empty-payload diagnosis. Not valid for: colored lightmap rendering, nonempty QLIT v1 content proof. |
| `dungeon-evidence-bsp2.lit` | QLIT v1 (12,776 bytes) | `dungeon-evidence-bsp2.bsp` | Unmodified compiler-produced external colored-light companion: 8-byte QLIT v1 header plus 12,768 RGB bytes, exactly `BSP lightdata 4,256 × 3`. |

## 10. Source Identity Policy

### 10.1 TrenchBroom UUID Stability — FIXTURE EVIDENCE: `_tb_id` STRIPPED BY COMPILER

TrenchBroom assigns per-entity UUIDs in the `_tb_id` key.

**Fixture evidence (2026-07-23)**: The pinned ericw-tools 2.0.0-alpha3 `qbsp` **strips** `_tb_id` from the compiled entity lump. Both `q1-bsp29-core.bsp` and `ericw-bsp2-colored.bsp` compiled from `q1_profile_core.map` (which carries `_tb_id` on every entity) contain zero `_tb_id` keys in their entity strings. The compiler does NOT preserve unknown entity key/value pairs.

**Approved policy**: UUID-backed entity identity is **not viable** with this compiler. Identity MUST use structural fingerprint reconciliation:
1. `(asset_id, entity_index)` — structural position in BSP.
2. Normalized semantic fingerprint: `(classname, origin, targetname, target)` key set.
3. Source evidence: map file path and compiler provenance.
4. Duplicate ordinal for entities with identical fingerprints.

### 10.2 Backup Identity Reconciliation

When UUID is unavailable, identity reconciliation uses:

1. `(asset_id, entity_index)` — structural position in BSP.
2. Normalized semantic fingerprint: `(classname, origin, targetname, target)` key set.
3. Source evidence: map file path and compiler provenance.
4. Duplicate ordinal for entities with identical fingerprints.

The reconciliation must report:

| event | diagnostic |
|-------|-----------|
| new entity matched to existing | `IdentityMatched` |
| existing entity not in new load | `IdentityOrphaned` |
| entity in new load not in existing | `IdentityInserted` |
| structural change detected | `IdentityStructureChanged` |
| ambiguous match (multiple candidates) | `IdentityAmbiguous` |
| deleted entity confirmed | `IdentityDeleted` |

## 11. Compiler Evidence Appendix

### 11.1 Compiler Executables

| executable | SHA-256 | version output |
|-----------|---------|---------------|
| `qbsp` | `4a05974acf9e59f73a9c8f4e8236f3d1e0961be477dae002837e166278882f17` | `ericw-tools 2.0.0-alpha3` |
| `vis` | `f7f429e0ad9bebbb0ebdefea8d6cd5e13a6ad6ff9f6893126772e00b66e364ea` | `ericw-tools 2.0.0-alpha3` |
| `light` | `1210ee9bed8990f67e3be7e28fbfd8d210329b052ac57dba017200d2da1ca5e5` | `ericw-tools 2.0.0-alpha3` |

Distribution: `ericw-tools-2.0.0-alpha3-Linux` from official GitHub releases. Located at `~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/`. User-supplied; not bundled with the engine.

### 11.2 Fixture Compilation Arguments

| fixture | qbsp args | vis args | light args |
|---------|-----------|----------|------------|
| `q1-bsp29-core` | (none) | (none) | (none) |
| `ericw-bsp2-colored` | `-bsp2` | (none) | `-bsp2 -lit -colored` |
| `dungeon-evidence-bsp2` | `-bsp2` | (none) | `-threads 1 -lit` |

### 11.3 Duplicate-Build Reproducibility

Two independent `build_fixtures.py` runs in separate temp directories produced byte-identical zero-face structural outputs. For the visible BSP2 fixture, duplicate `engine_pack compile-bsp` runs using `ericw-q1-bsp2-generated-profile.toml` (`light -threads 1 -lit`) produced byte-identical `.bsp` and `.lit` outputs, and both matched the checked-in `dungeon-evidence-bsp2` fixture hashes. This confirms the pinned compiler/profile combination is deterministic for the evidence fixture.

### 11.4 Source and Output Hashes

| artifact | SHA-256 | size |
|----------|---------|------|
| `project_palette.lmp` | `056225760af497967cb7fc67aba0757e915e2f461ba909d440e8073d591c5dd0` | 768 B |
| `q1_profile_core.map` | `2d65d0381fe6f4e03929b751340e1208c3d382e5a06cfa9a6150072d8fcde96a` | — |
| `q1_profile_spatial.map` | `96796b534c1de1b62770cfc92153e2be3f3094ff93b38f75faeb798545e9fa19` | — |
| `q1_profile_structural.map` | `2146fb0f4b0fdaa19fb98f67010d14dd7d2ffb0d22ebabd7038a2f1115e21494` | — |
| `q1_profile_visible.map` | `5c46e35a86d727ab035f597eaf608847d44b7ede304ed098ad7c6b4ff3cbb436` | — |
| `dungeon_evidence_standard.map` | `036f28cd941c2eb4434827131ba54db0cbe56b4b5c76360a3a9deef8fbb28d6a` | — |
| `dungeon_evidence.wad` | `a41973a96de9bc49d6f817e5442e6b392d8a09c54338a1faba010a0b82f8fd6c` | 66,156 B |
| `q1-bsp29-core.bsp` | `10f4bcade9b4e6f2afccaca8cd40be5ac5d380a2e41982f5e41dfcccb9d8e268` | 23,500 B |
| `q1-bsp29-visible.bsp` | `07249e4fd28d4108128f8be37b48510da69444bcbc6a8c6ea3bc5270d3d2483e` | 123,424 B |
| `ericw-bsp2-colored.bsp` | `d4f8552fd6cbc9d0eea067b583a5b9b2ae2c5fa98d25a6f8614209f76b652e71` | 23,832 B |
| `ericw-bsp2-colored.lit` | `0b46b6b85a8ec0a4a3b905c059f73485e06e6bf452df80823f4ebb87b029fe36` | 8 B |
| `dungeon-evidence-bsp2.bsp` | `1889a6c7959b3e00e9a907243b61d35b6c7d32525f098e13b5b059bd17ba5f15` | 26,424 B |
| `dungeon-evidence-bsp2.lit` | `a732a09039ff31485c3d91346d778d5ad41ba563bddca76876fb81c0eb3f1805` | 12,776 B |

## 12. Profile Identity Reconciliation (Phase 01 Baseline)

### 12.1 Compatibility Family vs Publication Profile

Two distinct identity concepts exist and must not be used interchangeably:

| concept | value | description |
|---------|-------|-------------|
| Compatibility family | `q1-portable-ericw` | The compiler dialect family: ericw-tools Q1-family with BSP29 portable core and BSP2 extended profile |
| Exact publication profile | `ericw-q1-bsp2-generated` | The checked-in profile file at `tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml` with pinned compiler args: `qbsp -bsp2 -threads 1`, `vis -threads 1`, `light -threads 1 -lit` |

**Baseline:** `.internal-dev/captures/bsp-dungeon-repair-baseline/manifest.json`
**Baseline ID:** `5fda7dae1d1f3da51c064d1d136418dae9c0e79a43ad73396a496bba81270c35`
**Status:** RECORDED — no renaming performed; both names retained as distinct concepts.

### 12.2 Exact Artifact Identity Freeze (Retry)

The Phase 01 retry freezes the exact current-code artifact at `.internal-dev/captures/bsp-dungeon-repair-baseline/baseline-artifact.{map,bsp,lit}`. Its BSP is 86,312 bytes with SHA-256 `b24922904beccd06617b47715243993d8f40b4f262b4140200a69a3bdd6326d7`; its map, LIT, compiler provenance, selected 27,572-byte `b239e695…` WAD, palette, publication profile, and pinned compiler hashes are content-addressed in the manifest.

The owner supplied that the artifact is deterministically regenerated from the current generator source tree and is the exact output associated with the reported rendering defect. The source tree matches reported code commit `8933a041`. This resolves the exact-artifact identity blocker without claiming a renderer replay, fixed-camera capture, frame-slot observation, or live-WSI result.

The 159,012-byte cached pre-fix `m1-seed-0.bsp`, its raw MCP transcript, its 22,060-byte WAD candidate, and the stale-cache behavior of `tools/dungeon_explore.sh` remain historical comparison evidence. They are retained under `historical_comparison_baseline` in the manifest and must not be substituted for the exact artifact. The cache invalidation defect remains relevant to the historical route, but it no longer blocks the exact artifact freeze.

### 12.3 Generator Serializer Grammar

**Status:** **APPROVED — Option A (Standard Quake); source: `owner-sprint-20260726`**

The approved grammar is Standard Quake offset/rotation/scale syntax: `"texture" x_off y_off rotation x_scale y_scale`; Valve 220 is not implemented. `BrushFace.u_axis`/`v_axis` are dead fields and must be removed in the authorized generator implementation. Current code still retains them, so this Phase 01 specification reconciliation does not claim implementation completion. See `bsp-dungeon-generation.md` §16.1 and `DECISION-20260726-01` for justification, rejected Valve 220 alternative, caveats, affected specs, and re-review trigger.

### 12.4 Corridor Vertical IR

**Status:** **APPROVED — Option A (freeze at 80); source: `owner-sprint-20260726`**

Public `Corridor.height` must equal `80`; non-80 input must be rejected at the first fallible public boundary, and generated routes remain 80 high. `build_corridor_slabs` and `build_corridor_boundary_walls` must use `corridor.height` rather than a global-height reconstruction. Current code has not yet implemented that approved behavior, so Phase 01 does not claim code conformance. See `bsp-dungeon-generation.md` §16.2 and `DECISION-20260726-02` for justification, rejected variable-height alternative, caveats, affected specs, and re-review trigger.

## 13. Approval and Evidence Matrix

| cell | status | fixture | evidence | reviewer |
|------|--------|---------|----------|----------|
| BSP29 magic/version detection | PASS (Phase 02) | `q1-bsp29-core.bsp` (compiled, hash verified) | parser fixture tests passed; `cargo test -p bsp` parses both BSP29 and BSP2 fixtures | dhickel (2026-07-23) |
| BSP29 all 15 lumps parse | PASS (Phase 02) | `q1-bsp29-core.bsp` | golden parse tests + adversarial corpus covering all lump types | dhickel (2026-07-23) |
| BSP2 magic/version detection | PASS (Phase 02) | `ericw-bsp2-colored.bsp` (compiled, hash verified) | parser fixture tests passed | dhickel (2026-07-23) |
| BSP2 widened node layout (44B) | PASS (Phase 02) | `ericw-bsp2-colored.bsp` (bspinfo-verified: 6 nodes × 44B = 264B) | `264 / 6 = 44` ✓; parser computes `lump_size / element_count` | dhickel (2026-07-23) |
| BSP2 widened leaf layout (44B) | PASS (Phase 02) | `ericw-bsp2-colored.bsp` (bspinfo-verified: 7 leaves × 44B = 308B) | `308 / 7 = 44` ✓ | dhickel (2026-07-23) |
| BSP2 widened clipnode layout (12B) | PASS (Phase 02) | `ericw-bsp2-colored.bsp` (bspinfo-verified: 24 clipnodes × 12B = 288B) | `288 / 24 = 12` ✓ | dhickel (2026-07-23) |
| BSP2 widened edge layout (8B, u32 vertex indices) | PASS (Phase 02) | `ericw-bsp2-colored.bsp` (edges 8B vs BSP29 4B) | parser uses u32 vertex indices for BSP2 edges unconditionally; `lump_size / element_count` = 8 ✓ | dhickel (2026-07-23) |
| BSP2 nonempty face layout (28B) | PASS (Phase 02 repair, 2026-07-24) | `dungeon-evidence-bsp2.bsp` has 41 faces; face lump 1,148 B = 41 × 28 B | strict parser tests and `bspinfo` stats verified nonempty BSP2 face stride | dhickel (2026-07-24) |
| BSP2 nonempty markface layout (4B) | PASS (Phase 02 repair, 2026-07-24) | `dungeon-evidence-bsp2.bsp` has 55 markfaces; markface lump 220 B = 55 × 4 B | strict parser tests and `bspinfo` stats verified nonempty BSP2 markface stride | dhickel (2026-07-24) |
| BSP2 Standard-map compile | PASS (Phase 02 repair, 2026-07-24) | `dungeon_evidence_standard.map` + `dungeon_evidence.wad` compiled through pinned `engine_pack compile-bsp --wad` profile | duplicate compile outputs matched checked-in `.bsp` and `.lit` hashes | dhickel (2026-07-24) |
| BSP2 strict reload | PASS (Phase 02 repair, 2026-07-24) | `dungeon-evidence-bsp2.bsp` + project palette + `dungeon-evidence-bsp2.lit` | `cargo test -p bsp -p engine_pack`; `engine_pack validate-bsp --strict` reported BSP2, 3 entities, 41 faces, 0 diagnostics | dhickel (2026-07-24) |
| BSP2 duplicate-build reproducibility | PASS (Phase 02 repair, 2026-07-24) | Two `engine_pack compile-bsp` runs with `light -threads 1 -lit` produced identical BSP/LIT hashes (`1889a6c7…`, `a732a090…`) | duplicate outputs also matched checked-in fixture bytes | dhickel (2026-07-24) |
| BSP2 clean failure handling | NOT-RUN (reclassified non-blocking, Phase 01) | requires adversarial BSP2 fixtures (truncated lumps, invalid indices, cycle corruption) | **Phase 01 reclassification**: blocks_generator=false. This cell exercises parser-level adversarial testing, not generator-dependent behavior. The 47 existing adversarial tests cover all 7 fuzz categories for BSP29/BSP2. Generator does not produce malformed output. BSP2-specific adversarial corpus remains desirable but does not block generator authorization. | — |
| BSP2 provenance (compiler identity, version, args recorded) | PASS (Phase 02 repair, 2026-07-24) | `fixture-manifest.toml` and `engine_pack` provenance record source/WAD/palette hashes, pinned executable SHA-256 hashes, and `qbsp -bsp2` / `light -threads 1 -lit` args | duplicate compile + strict reload validated the visible BSP2 path | dhickel (2026-07-24) |
| BSP2 external `.lit` colored-light output | PASS (Phase 02 repair, 2026-07-24) | `dungeon-evidence-bsp2.lit` has QLIT v1 header plus 12,768 RGB bytes matching `dungeon-evidence-bsp2.bsp` lightdata × 3 | strict load selects `ColoredLightSource::LitFile`; size relation asserted in golden/evidence tests | dhickel (2026-07-24) |
| BSPX RGBLIGHTING extension | BLOCKED (generator gate) | no fixture — compiler-produced face-visible BSP2 fixture (`dungeon-evidence-bsp2.bsp`) has nonempty `.lit` but no BSPX lump | blocked pending face-visible compiler-produced BSPX fixture; not required for generator output | — |
| BSP29 face-visible parse (derived fixture) | NOT-RUN (deferred) | `q1-bsp29-visible.bsp` (post-processed by `build_fixtures.py`) | BSP29 is not required for dungeon generation — all generated output is BSP2; this cell is deferred | — |
| Entity grammar (all rules) | PASS (Phase 02) | `q1-bsp29-core.bsp` (5 valid entities) + adversarial tests | 47 adversarial tests + entity parser validation | dhickel (2026-07-23) |
| `_tb_id` stripped by compiler | PASS (Phase 01) | both BSP29/BSP2 entity lumps lack `_tb_id` | byte inspection Phase 01; identity uses fingerprint reconciliation | dhickel (2026-07-23) |
| WAD2 texture lookup | PASS (Phase 02) | `project_palette.wad` | WAD parser validates header, directory, lump bounds | dhickel (2026-07-23) |
| Palette loading | PASS (2026-07-24 repair) | `project_palette.lmp` validates the 256×3 parser; local `start.bsp` validation resolved its game-root `gfx/palette.lmp` explicitly | production content never falls back to the synthetic project fixture; missing palette fails | dhickel (2026-07-24) |
| External normal/gloss companions | PASS (2026-07-24) | deterministic filename/discovery tests, package confinement test, extraction propagation test, PNG decode/dimension preflight tests, PBR route/material packing tests, and a GPU headless capture with synthetic companions | project-owned authored normal/gloss fixture remains desirable for long-term visual regression | dhickel (2026-07-24) |
| Unsupported dialect rejection | PASS (Phase 02) | fixture independent | adversarial test covers BSP30/HL, BSP38/Q2, BSP46/Q3, Source/VBSP magic values | dhickel (2026-07-23) |
| Diagnostic code stability | PASS (Phase 02) | fixture independent | all 47 adversarial tests assert on `DiagnosticCode` + `Severity`, never message text | dhickel (2026-07-23) |
| Duplicate build byte-identical | PASS (Phase 01) | Two-temp-dir build verified | build_fixtures.py evidence; `engine_pack compile-bsp` produces reproducible output | dhickel (2026-07-23) |
| Whole-asset rejection for invalid magic | PASS (Phase 02) | fixture independent | parser returns `BspReport` with `BSP-UNSUPPORTED-DIALECT` | dhickel (2026-07-23) |
| Whole-asset rejection for corrupt lumps | PASS (Phase 02) | 47 adversarial tests | truncation, overlap, cycle, overflow, alignment cases all produce `BSP-STRUCT-CORRUPT-*` errors | dhickel (2026-07-23) |
| Versioned persistence schema (V1 envelope) | PASS (Phase 08) | `BspPersistenceEnvelope` round-trip tests: `SchemaVersion` enum, `from_u32` rejection of unknown versions, `validate_schema` for approved versions | 23 reload/persistence tests passed | dhickel (2026-07-23) |
| Canonical float encoding | PASS (Phase 08) | `CanonicalFloat` normalizes -0.0→+0.0, deterministic LE bytes for hashing, JSON f64 for full precision | `canonical_float_normalizes_neg_zero` + `canonical_float_round_trips` passed | dhickel (2026-07-23) |
| Source-link persistence excludes GPU handles | PASS (Phase 08) | `save_capture_excludes_gpu_handles` test: verified VkImage/VkBuffer/VkDescriptorSet/cache_slot/generated_geometry absent from serialized payload | all 9 persistence tests passed | dhickel (2026-07-23) |
| Restore cancellation preserves active generation | PASS (Phase 08) | `restore_with_content_hash_mismatch_fails` and `restore_cancelled_proves_active_unchanged`: tampered hash → restore cancelled → active generation unchanged → coordinator still usable | 23 reload/persistence tests passed | dhickel (2026-07-23) |
| Identity reconciliation (fingerprint+ordinal) | PASS (Phase 08 validator repair) | `build_identity_records` uses `EntityIdentity` data; fingerprint stable handles include duplicate ordinal; `reconcile_overrides` matches UUID first, then fingerprint+ordinal for entity and light overrides; orphaned/ambiguous detected | 23 reload/persistence tests passed | dhickel (2026-07-23) |

**Re-review trigger**: Any change to pinned compiler version or hash, any new BSP format addition, or any fixture replacement requires re-review of all approved cells.

## 13. EnhancedV3 Generation Profile Compatibility

### 13.1 Scope

The EnhancedV3 profile (`GenerationProfile::EnhancedV3`, tag `"m3"`) produces
M2-only two-layer dungeons with cardinal + 45° chamfered-octagonal geometry,
pointed-default plus rectangular/segmented cardinal portal surrounds, and
grounded assemblies. It targets the same pinned BSP2 compiler profile as
Enhanced v2 (§12.1). This section records the compatibility contract for the
v3 profile; it does not alter any existing v1 or v2 compatibility rule.

### 13.2 Generation Profile Identity

| field | value |
|-------|-------|
| `GenerationProfile` variant | `EnhancedV3` |
| production tag | `"m3"` |
| proof tag (historical only) | `"enhanced-v3"` — unrecognized in production dispatch |
| `from_tag("m3")` | `Some(EnhancedV3)` |
| `from_tag("enhanced-v3")` | `None` (proof tag, not a production dispatch token) |

The existing `LegacyV1` (`"legacy-v1"`) and `EnhancedV2` (`"enhanced-v2"`)
profiles are unchanged. `EnhancedV3` is additive only.

### 13.3 Compiler Compatibility

| property | value |
|----------|-------|
| compiler | ericw-tools 2.0.0-alpha3 (pinned) |
| format | BSP2 only (identical to v2 requirement) |
| exact publication profile | `ericw-q1-bsp2-generated` at `tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml` |
| qbsp args | `-bsp2 -threads 1` |
| warning policy | any warning from `qbsp`, `vis`, or `light` fails compilation |
| geometry constraints | cardinal + 45° normals only; all vertices integer multiples of 16 |

### 13.4 RNG Domain

| property | value |
|----------|-------|
| domain separator | `"dungeon-gen/v3"` |
| framing | SHA-256(domain || seed_le || stage_tag) |
| stage tags | `v3-placement`, `v3-topology`, `v3-features`, `v3-detail` |
| isolation from v1 | cryptographically independent from `"dungeon-gen/v1"` |
| isolation from v2 | cryptographically independent from `"dungeon-gen/v2"` |

### 13.5 Frozen v1/v2 Corpus Identity

The Legacy v1 12-entry corpus and Enhanced v2 12-entry corpus must remain
byte-identical to their frozen baselines through all v3 production phases.
Any drift in v1 or v2 output is a blocking regression. The v3 profile is
structurally isolated — no v1 or v2 source file is modified to add v3
behavior.

### 13.6 Evidence Basis

The Enhanced v3 architectural proof (`DECISION-20260731-01`) demonstrated:
- 45° diagonal wall geometry compiles warning-free with thickness ≥ 16
- Pointed-arch portal apertures preserve 64×80 swept clearance
- Grounded assemblies satisfy acyclic support graph contract
- Dense M2 fixture: 2,404 faces, 6 entities, 4 batches — all within ceilings
- 24/24 v1+v2 corpus entries byte-identical to baseline
- Live GPU startup: swapchain acquired, 21,574 frames, 0 errors

## 14. EnhancedV3RichnessV1 Compatibility Profile (Freeze-Only — APPROVED 2026-08-02 via autonomous-delegation directive)

### 14.1 Scope

The EnhancedV3RichnessV1 profile (`GenerationProfile::EnhancedV3RichnessV1`,
tag `"richness-v1"`) extends EnhancedV3 output with gameplay content. It is an
additive profile structurally isolated from Legacy v1, Enhanced v2, and
EnhancedV3 baseline. This section records the compatibility contract; no
implementation code, CLI, GUI, or public surface exists as of this freeze.

### 14.2 Profile Identity

| field | value |
|-------|-------|
| `GenerationProfile` variant | `EnhancedV3RichnessV1` (not yet created) |
| production tag | `"richness-v1"` — returns `None` from `from_tag` until owner-authorized |
| CLI | `dungeon_gen --class richness-v1` (rejected until authorized) |
| packaging | `engine_pack enhanced-dungeon-v3-richness` (rejected until authorized) |
| relation to baseline v3 | additive extension; v3 map geometry byte stream unchanged |

### 14.3 RNG Domain

| property | value |
|----------|-------|
| domain separator | `"dungeon-gen/v3-richness/v1"` |
| framing | `SHA-256(domain \|\| seed_le \|\| stage_tag)` |
| stage tags | `richness-archetype`, `richness-prop`, `richness-lighting`, `richness-cave`, `richness-opening`, `richness-theme` (6 frozen tags) |
| isolation from v1 | cryptographically independent from `"dungeon-gen/v1"` |
| isolation from v2 | cryptographically independent from `"dungeon-gen/v2"` |
| isolation from v3 | cryptographically independent from `"dungeon-gen/v3"` |

### 14.4 Compiler Compatibility

Identical to the frozen BSP2 compiler profile (§12.1) and EnhancedV3
compiler contract (§13.3). RichnessV1 output targets the same pinned
ericw-tools 2.0.0-alpha3 profile with `-bsp2 -threads 1` compilation.
Warning-free `qbsp`/`vis`/`light` stages and strict reload with zero
diagnostics are required.

### 14.5 Theme Invariance Compatibility

| rule | value |
|------|-------|
| map geometry bytes | invariant under theme selection |
| blueprint candidate keys | invariant under theme selection |
| cave decisions | invariant under theme selection |
| route topology | invariant under theme selection |
| compliance gate | `RichnessThemeInvarianceViolated` error if theme changes any geometry decision |

### 14.6 Frozen Compatibility Baseline

| contract | requirement |
|----------|------------|
| Legacy v1 12-entry corpus | byte-identical to frozen baseline |
| Enhanced v2 12-entry corpus | byte-identical to frozen baseline |
| EnhancedV3 12-entry corpus | byte-identical to Phase 01 baseline-freeze manifest |
| baseline V3Config constructors | unchanged |
| baseline V3Config equality | unchanged |
| baseline metadata schema | unchanged |
| profile tags | `"legacy-v1"`, `"enhanced-v2"`, `"m3"` remain recognized; `"richness-v1"` gated |
| CLI documents | unchanged |
| package fingerprints | unchanged |

### 14.7 Evidence Basis

No production Richness generator/profile exposure exists. The Phase 01
baseline-v3 freeze remains authoritative and Richness must not disturb it.
Phase 05 now supplies frozen compiler-convention, active controller, process
supervisor, and measurement-method evidence only; it does not satisfy later
Richness content/profile acceptance rows in `bsp-acceptance.md` §18.

## 15. Phase 05 Convention Evidence (FROZEN 2026-08-02)

### 15.1 Convention Fixture

| fixture | path |
|---------|------|
| conventions.map | `src/bsp_generator/tests/fixtures/enhanced_v3_richness/conventions.map` |
| compiler | ericw-tools 2.0.0-alpha3 (pinned) |
| profile | `ericw-q1-bsp2-generated` |
| WAD | qualified fixture closure from authorized `cc0_dungeon_v2.wad` |
| tool miptex authorization | production lowercase `skip`; fixture-scoped lowercase `hint`, `hintskip`, and `clip` deterministically clone the exact CC0 skip pixels/mips and alter only internal/directory identity fields |
| palette | `palette.lmp` (project-authored) |
| source construction | sealed structural room with independently addressable convention cells; compiler-safe for all pinned hulls |
| failure policy | any warning, leak/pointfile, skipped fill, missing texture, signal/nonzero exit, malformed BSP/LIT, or strict-load diagnostic fails the test |

### 15.2 Compiler Transformation Evidence

| convention | source | compiler output | strict reload |
|-----------|--------|----------------|---------------|
| hint/hintskip portal | one exact lowercase sloped `hint` plane with five `hintskip` sides, all on 16-unit coordinates | independently addressable plane survives as a BSP split; all hint/hintskip faces omitted | exact split plane present; zero visible tool faces; 0 diagnostics |
| skip omission and collision | separate exact lowercase skip cells, including a solid pillar | skip surfaces omitted while solid skip brush still contributes stored collision | zero visible skip faces; pillar center is solid |
| clip omission and collision | separate exact lowercase clip solid pillar | clip surfaces are omitted from rendering and hull-0 leaf point contents while retained in the player stored-hull collision tree | zero visible clip faces; hull-0 point is non-solid; player stored-hull point is start-solid |
| func_detail compiler control | func_detail entity with bs_accent brush | compiler consumes control and merges brush into world model | func_detail absent; one world model and nonempty PVS |
| colored light entity | light entity with `_color "1.0 0.3 0.1"` | key retained and colored QLIT v1 payload emitted | key present; valid nonempty LIT RGB payload |
| custom climb point entity | `func_ladder` with origin and climb metadata | unknown classname and custom keys retained; no surface flag or inline-model claim | exact classname/keys present |
| custom drop point entity | `trigger_multiple` with origin and one-way metadata | trigger classname and custom keys retained | exact classname/keys present |
| active climb/drop brush descriptors | `controller.map` revision-qualified `trigger_multiple` brush entities with IDs, normals, priorities, and one-way keys | inline-model bounds and keys retained | app parses compiled AABBs and drives the active controller |
| `_tb_id` key | `_tb_id` source identity | compiler strips it | absent; structural fingerprint reconciliation remains required |

### 15.3 Supported/Unsupported Summary

| convention | status | structural equivalent (if unsupported) |
|-----------|--------|---------------------------------------|
| hint/hintskip split and surface omission | SUPPORTED | — |
| skip face omission and independent solid collision | SUPPORTED | — |
| clip face omission and independent player-hull collision | SUPPORTED | — |
| func_detail compiler control | SUPPORTED | — |
| colored light (`_color`) | SUPPORTED | — |
| custom climb/drop classname and key preservation | SUPPORTED | — |
| revision-qualified trigger brush-model bounds | SUPPORTED | app-owned descriptor parsing |
| `_tb_id` preservation | UNSUPPORTED | fingerprint-based entity reconciliation (§10) |
| `func_ladder` surface-flag semantics | UNSUPPORTED / not used | compiled brush-AABB climb descriptor (`bsp-spatial-physics.md` §11) |

### 15.4 Deterministic Recompile

Two independent `qbsp → vis → light` compiles of `conventions.map` in
separate temp directories produce byte-identical BSP and LIT bytes. The
controller fixture applies the same duplicate-compile gate before active-path
movement characterization. Both strict-load with zero diagnostics.

### 15.5 Compiler Supervisor and Measurement Compatibility

The `engine_pack` runner uses shell-free commands, a cleared explicit
environment, concurrent pipe drains, one combined byte ceiling, bounded
polling timeout, Unix process groups, whole-group termination, and joined
reader cleanup. Group cleanup also runs after a successful direct-child exit,
preventing an inherited descendant pipe from hanging the caller. Adversarial
hang, noisy-combined-stream, nonzero-exit, descendant, and orphaned-pipe tests
are mandatory.

The Phase 05 measurement report has closed identity
`enhanced-v3-richness-measurement/v1`. It requires canonical corpus SHA-256,
explicit warmup/sample labels, the median selected statistic, all eight frozen
stages, all 15 BSP lumps, artifact sizes/hashes, package size,
faces/entities/batches, and process/runtime memory. The harness validates the
closed JSON schema before atomic write and fails on missing observations,
wrong run count, duplicate/missing stages or lumps, inconsistent byte counts,
or timeout breach. Measurement output is observational only and is
canonicalized after generation decisions. No compatibility ceiling is inferred
from the synthetic Phase 05 observations.

The Phase 05 tables are accepted and frozen by the implementation
authorization. Any compiler version/hash/profile change, tool-miptex identity
change, convention result change, measurement schema/method change, or
controller semantic change requires owner re-review.
