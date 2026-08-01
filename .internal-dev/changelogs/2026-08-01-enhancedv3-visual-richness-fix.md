## Date
2026-08-01

## Change Summary
Made default EnhancedV3 output visibly architectural: larger rooms (max span 448), reliable chamfered/octagonal footprints, readable pointed arch crowns (≥48 units, `bs_accent` trim), and room-scaled grounded features. Re-baselined v3 corpus contracts. v1/v2 output remains byte-identical.

## Files
- `src/bsp_generator/src/enhanced_v3/config.rs`: DEFAULT_ROOM_SPAN_MAX 256 → 448
- `src/bsp_generator/src/enhanced_v3/footprint.rs`: upper-biased sizing, removed forced-rectangle, room-scaled chamfer tiers, multi-corner/octagon bias, portal-edge preservation, adaptive grid
- `src/bsp_generator/src/enhanced_v3/pipeline.rs`: pointed arch ≥48-unit crown, bs_accent trim, aperture-specific brush IDs, segmented arch cap overlap fix
- `src/bsp_generator/src/enhanced_v3/intent.rs`: room-scaled pillars/buttresses/blades
- `apps/bsp_beta/src/generation.rs`: updated test expectations
- `src/bsp_generator/tests/enhanced_v3_explorer.rs`: updated test expectations
- `.internal-dev/specifications/bsp-dungeon-generation.md`: revised EnhancedV3 max span with re-review note

## Behavioral Impact
- Default max room span: 448 (was 256); explicit values authoritative
- Room sizing: upper-biased (median ≥ 256, max ≥ 448 in canonical 2048 Moderate)
- Chamfers: no forced-rectangle rule; ~40% octagon for large rooms, 25% rectangular
- Chamfer size: 32 (<160), 48 (160–223), 64 (≥224) by shorter axis
- Pointed arches: 3–4 crown steps (48–64 units), bs_accent contrast trim
- Feature scale: 32×32 pillars for rooms ≥192, 2-quantum buttresses, full-height blades

## Specification Impact
EnhancedV3-specific max span revision from 256 to 448 (§20.6). Legacy v1 and Enhanced v2 span contracts unchanged. Owner re-review recorded.

## Risks
- v3 corpus hashes will change; re-baselining required after full compiler validation
- Some extent/preset combos (e.g. Moderate at 1024) cannot fit 448 max span; typed errors produced

## Follow-up Items
- Re-baseline 12-entry v3 corpus with new SHA-256 hashes
- Run full compiler/spatial validation on all v3 corpus entries
- Headless visual capture evidence per skill
- Live startup smoke test
- Update docs/guide/18-bsp-beta.md and docs/guide/19-bsp-generator.md
