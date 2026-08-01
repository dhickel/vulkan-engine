# Date

2026-07-31

# Change Summary

Repaired the EnhancedV3 segmented arch/corridor leak without narrowing the accepted explorer contract. Full-config package generation now validates the genuine segmented override through warning-free ericw compilation and strict publication.

# Files

- `src/bsp_generator/src/enhanced_v3/config.rs`
- `src/bsp_generator/src/enhanced_v3/pipeline.rs`
- `src/bsp_generator/tests/enhanced_v3_explorer.rs`
- `tools/dungeon_gen/src/main.rs`
- `tools/dungeon_gen/tests/enhanced_v3_cli.rs`
- `tools/engine_pack/src/enhanced_dungeon_v3.rs`
- `tools/engine_pack/tests/enhanced_dungeon_v3_candidate.rs`
- `docs/{api,guide,internal}/19-bsp-generator.md`
- `.internal-dev/specifications/{bsp-acceptance,bsp-compatibility,bsp-dungeon-generation,decisions}.md`
- `.internal-dev/knowledge/{bsp-enhanced-v3-production,bsp-generator-compiled-spatial-validation}.md`
- `.internal-dev/debug_reports/enhanced-v3-segmented-default-byte-compare.txt`
- `.internal-dev/bugs/enhanced-v3-stale-qualification-manifest/report.md` (pre-existing validation defect, GitHub #69)

# Behavioral Impact

Integrated generation, cycling, CLI parsing/help, and full-config package publication accept `ArchType::Segmented`. The existing two-band surround now has a one-quantum cap immediately outside its 32-unit Z=112–128 centre recess, where the corridor roof ends at Z=112. This seals the exterior path without filling the visible recess or reducing the complete 64×80 throat. The package test exercises rooms, physical corridor segments, loops, segmented arch type, minlight, and light count and records those overrides in metadata and the canonical manifest. Pointed/default output bytes are unchanged.

# Specification Impact

Promoted cardinal segmented surrounds from historical focused-only evidence to an owner-authorized explorer-integrated override. Historical proof records remain historical; diagonal portals, concave rooms, and other G-12 capabilities remain deferred.

# Risks

The new cap is emitted only for explicit segmented configurations, limiting compatibility risk. Validation covers the accepted Moderate seed-42 full-config package; a broader segmented seed/preset compiler corpus is not frozen. The checked-in v3 qualification manifest is independently stale from the pre-#68 thin-slice generator and is tracked by GitHub #69; direct 12-entry task-base comparison was used for current default-byte evidence.

# Follow-up Items

- Optional future evidence widening: add segmented overrides to a multi-seed/preset compiler corpus if segmented becomes a preset default rather than an explicit explorer choice.
