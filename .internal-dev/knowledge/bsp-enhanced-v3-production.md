# Knowledge: BSP EnhancedV3 Production

## Topic
EnhancedV3 production profile architecture, contracts, evidence, and known limitations.

## Source References
- Specification: `.internal-dev/specifications/bsp-dungeon-generation.md` §20
- Proof evidence: `.internal-dev/plans/enhanced-v3-proof/decision-package.md`
- Production code: `src/bsp_generator/src/enhanced_v3/`
- Crate guide: `src/bsp_generator/AGENTS.md`
- User guide: `docs/guide/19-bsp-generator.md`
- API reference: `docs/api/19-bsp-generator.md`
- Internal docs: `docs/internal/19-bsp-generator.md`
- Owner re-review directive: Phase 05 EnhancedV3 production integration review
- Visual evidence: `.internal-dev/captures/enhanced-v3-production/manifest.md`

## Key Takeaways

### Profile Identity
- `GenerationProfile::EnhancedV3` with production tag `"m3"`
- Proof-only tag `"enhanced-v3"` returns `None` from `from_tag()`
- CLI: `dungeon_gen --class m3 --seed 42`
- Pack: `engine_pack enhanced-dungeon-v3 --seed 42 --preset moderate --out /tmp/pkg`
- Explore: `./tools/dungeon_explore.sh --class m3 --seed 42`

### Architecture
- Two-layer M2 arrangement: lower floor Z=0, upper floor Z=192, room height 176
- Cardinal (axis-aligned) + 45° diagonal geometry only
- i128 Rational arithmetic — no floating-point in geometry path
- Full pipeline: V3Config → Footprints → CommittedTopology → CompositionPlan → Assembly → .map

### Presets
- Sparse: exactly 12 rooms, 10 same-layer routes, 1 family / 1 assembly / 2 features, 3,000 face budget
- Moderate: exactly 20 rooms, 20 same-layer routes, 3 families / 3 assemblies / 6 features, 5,000 face budget
- Rich: exactly 28 rooms, 30 same-layer routes, all 6 families / 6 assemblies / 12 features, 8,000 face budget

The measured default-extent seed matrix (0, 42, 99, 255) emits 1,856–1,883
Sparse, 3,275–3,310 Moderate, and 4,725–4,782 Rich source faces. Those source
counts remain below their preset ceilings and the 10,000-face M2 ceiling.

### RNG
- Domain separator `"dungeon-gen/v3"` — independent from v1/v2
- 4 frozen stage tags: `v3-placement`, `v3-topology`, `v3-features`, `v3-detail`
- Same seed across v1/v2/v3 produces cryptographically independent output

### Key Differences from Enhanced v2
- Geometry: i128 Rational (not i32/f32); cardinal + 45° (not axis-aligned only)
- Rooms: chamfered/octagonal (not just rectangular)
- Portals: cardinal full-depth omissions with pointed default and rectangular/segmented explorer surrounds (not opening brushes)
- Assemblies: grounded support graph (new capability)
- Families: 6 real integrated feature generators (vs 2 strategies)
- Presets: Sparse/Moderate/Rich (vs single EnhancedConfig)
- Error type: V3Error with 45 variants (vs EnhancedError with 12)

### Budget Evidence
- Dense Rich proof fixture: 2,404 faces, 6 entities, 4 batches
- All presets well within M2 ceilings (10,000 faces, 300 entities)
- The measured 12-entry default-extent source matrix ranges from 1,856 to 4,782 faces
- Source brush count measures authored structural density; it is not a renderer static-batch count and must not be compared with the 500-batch ceiling. Measure `ExtractedBsp::render_batches.len()` after strict compiled BSP extraction.
- Strict headless Rich seed-42 extraction produced 5,448 compiled faces in 4 batches; targeted non-solid MCP cameras visibly proved the stepped twisted pillar, ceiling-attached fracture pieces, and three-level terraced shrine
- `build_v3_package` publishes with atomic no-replace semantics. A `TempDir::path()` already exists and is therefore not a valid new publication destination; use a nonexistent child such as `temp.path().join("package")`.

### Compatibility
- The authorized explorer inventory is exact: rooms, corridor segment count, loops, vertical edges, chamfer, arch type, stairs, room-span bounds, grammar families/mode, feature flags/density, minlight, and light count, plus the existing seed/preset/extent. The fixed two-layer layout and 64×80 route clearance remain structural invariants, not configurable or provenance-only public fields. Do not infer additional knobs merely because a constant appears in the generator.
- v1 12-entry corpus: byte-identical to frozen baseline
- v2 12-entry corpus: byte-identical to frozen baseline
- Theme: cc0_dungeon_v2 reused without modification
- Compiler profile: same ericw-tools BSP2 profile
- `tests/fixtures/enhanced_v3_corpus/manifest.json` was frozen before the production repair that closed GitHub #68 and still records 3/4/6-room thin-slice hashes. Until GitHub #69 refreshes that corpus deliberately, `enhanced_v3_qualification::corpus_entries_deterministic` is a stale-baseline failure, not a valid current default-byte oracle. For scoped compatibility work, compare the current default matrix directly against the task-base commit and keep the 24-entry v1/v2 frozen baseline as the legacy gate.

### Portal and Route Assembly
- `CommittedPortal` data is structural input, not metadata: assembly must omit its aperture from both endpoint room walls. A solid opening/throat brush cannot carve Quake additive geometry.
- Preserve the declared width and height by emitting side wall segments plus any required sill/lintel. The current 64×80 portals begin at the room floor-slab top and require a lintel above Z=96.
- `CommittedRoute.envelopes` describe corridor clear bounds. Emit floor and ceiling slabs extending beneath/over side walls, then emit the two side walls around the clear span.
- Route envelopes can overrun an endpoint room's outer cross-axis span (the Rich preset does this). Seal the overrun with full-height terminal caps, and seal corner portions above the 80-unit opening with corner lintels. Otherwise ericw-tools reports `Reached occupant ... no filling performed` and writes a leak pointfile.
- The historical segmented surround leaves a 32-unit centre recess in its Z=112–128 crown band, above the corridor roof ending at Z=112. Directly integrating that surround exposed exterior void; seed-42 Moderate leaked from a light through `(64, 448, 120)`. The accepted explorer implementation retains the recess but backs it with a one-quantum cap immediately outside the room wall. The cap starts at the corridor roof top, does not enter the wall-depth silhouette, and leaves the complete 64×80 throat untouched. `ArchType::Segmented` is therefore valid through integrated generation and full-config package publication. The old focused fixture remains historical proof, not evidence that the uncapped integrated interface was sealed.
- Exact assembly validation requires interfaces for every positive-area shared face. For axis-aligned pipeline boxes, deriving interfaces from deterministic AABB face contacts avoids stale interface lists when portal walls split into several brushes.

## Engine Relevance
- The EnhancedV3 profile is the highest-capability production dungeon generator
- It shares the two-layer vertical contract from Enhanced v2
- It uses the same theme, compiler, and serialization contracts from v1/v2
- All three profiles coexist; selection is by `--class` tag at the CLI boundary
- The user's `dungeon_gen --class m3 --seed 42` defaults to Sparse preset at 2048²
- CLI explorer options cover preset/extent, exact rooms/corridor segments/loops/vertical edges, chamfer, `none|pointed|segmented` arch type, stairs, room spans, grammar allowlist/mode, feature flags/density, minlight, and light count
- All six grammar families (PortalChamber, ButtressedHall, ColumnGrove,
  FracturedVault, TerracedShrine, MonolithicChamber) are real integrated
  feature generators in production
- `dungeon_gen`'s stderr summary includes the requested output path, so
  deterministic CLI checks must compare map bytes (and same-path summaries),
  not summaries from distinct `--out` paths.
- `bsp_beta --m3-generate` resolves an omitted startup seed from system time and
  defaults to Moderate, chamfer enabled, pointed arches, and all six grammar
  families. CLI overrides must seed both the initial package and the GUI draft;
  otherwise the first editor state misrepresents the mounted world.

### Proof-vs-Production Gotcha
The old 160-brush proof model was hand-authored for budget ceiling evidence.
Production independently implements the same capabilities through the full
pipeline (V3Config → footprints → topology → composition → assembly → .map).
The production pipeline produces varying source face counts depending on seed
and geometry density. The measured default-extent 12-entry matrix ranges from
1,856 to 4,782 source faces. Do not expect production output to match proof
fixture brush/face counts exactly.

## Open Questions
- Live fixed-step movement through v3 pointed-arch portals not yet validated
- Broader segmented override corpus coverage beyond the focused Moderate seed-42 package is not yet frozen
- Diagonal portals and concave rooms remain deferred per G-12
