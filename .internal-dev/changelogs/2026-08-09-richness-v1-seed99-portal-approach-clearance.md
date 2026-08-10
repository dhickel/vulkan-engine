# Changelog: Richness V1 seed-99 portal-approach and passage-clearance repair

## Date
2026-08-09

## Git Commit

d9d698e8

## Change Summary

Fixed the EnhancedV3 Richness V1 generator (`m3-richness-v1`) producing a wall that blocked a portal and brush/wall placements with passages narrower than the frozen player-clear minimums on seed 99 (and any seed with the same all-inherited/explicit cave + late free-form placement path).

Root causes (diagnosed by senior-agent escalation):
1. **Cave connectivity ran on 32-unit lattice cells without enforcing 64-unit passage width.** Portal throat reservations forced only the first cave cell clear; cave walls could be emitted within 32 units of a portal approach.
2. **Portal approaches were not protected from later cave, prop, decoration, and route-wall placement.** Existing validation checked only the 16-unit-deep 64×80 throat box, so the obstruction passed all gates.
3. **Chamfered North/South wall roles were reversed relative to topology**, and the shared-wall portal fallback ignored wall orientation.

Fixes:
- `cave.rs`: widen horizontal empty runs to ≥ 64 units (2 cells) and vertical runs to ≥ 96 units (3 cells ≥ 80 headroom); remove unsupported complement islands; both rules close to a fixed point; new `validate_cave_passage_clearance` proves final lattice clearance.
- `composition.rs`: protect portal-approach route cells in route-shell materialization; `prune_portal_approach_decorations` removes optional interior mass/column/vault-rib/monolith brushes that would block an approach; fixed chamfered North/South wall roles and orientation-aware shared-wall portal lookup (`find_portal_wall` / `opposite_cardinal_wall`).
- `props.rs`: route/portal/turn protection dilation raised 32 → 64 units so props cannot pinch route mouths or portal approaches.
- `validation.rs`: `validate_protected_routes` now additionally validates a frozen 64-unit approach on both sides of every portal wall against all late free-form geometry (new `PORTAL_APPROACH_DEPTH = 64`), with a targeted unit test.
- `pipeline.rs`: new regression test `seed_99_inherited_and_explicit_keep_portals_and_passages_clear` covering CLI-default (inherited) and explicit corpus-byte requests, asserting geometry identity, portal materialization, throat+approach clearance, and cave clearance.

Docs: corrected `generate_richness_v1` signature (`&RichnessDocumentV1`) in spec + API/guide/internal docs (code already took the reference); corrected the published Richness package layout in the guide to the real `<name>.{map,bsp,lit,request.json,generation.txt,manifest.toml}` + `richness_<theme>_v1.wad` naming. (An agent-authored doc rewrite describing nonexistent "pacing plan (uniform/…/peaked)" and "variation seed" controls was reverted — those controls do not exist in code; the explorer exposes `relaxed|normal|intense` pacing, `subtle|moderate|wild` variation, prop density, and light density.)

## Files

- `src/bsp_generator/src/enhanced_v3/richness/cave.rs`
- `src/bsp_generator/src/enhanced_v3/richness/composition.rs`
- `src/bsp_generator/src/enhanced_v3/richness/props.rs`
- `src/bsp_generator/src/enhanced_v3/richness/validation.rs`
- `src/bsp_generator/src/enhanced_v3/richness/pipeline.rs`
- `.internal-dev/specifications/bsp-dungeon-generation.md` (API signature row)
- `docs/api/19-bsp-generator.md`, `docs/guide/19-bsp-generator.md`, `docs/internal/19-bsp-generator.md`
- `.internal-dev/bugs/seed-99-wall-blocks-portal-throat/`

## Behavioral Impact

- Seed 99 (and affected all-inherited/explicit seeds) now emit portal throats with a full 64-unit clear approach on both sides; cave passages, prop gaps, and wall-treatment openings retain ≥ 64×80 player clearance.
- Determinism and byte-identity semantics unchanged; map bytes change for affected seeds (expected — geometry is corrected).
- No presets, themes, seeds, profile tags, frozen constants (64×80 throats, 16-unit quantum), or canonical request grammar were modified.

## Specification Impact

- `bsp-dungeon-generation.md` §21 (Richness V1): API signature row corrected to match the implemented `&RichnessDocumentV1`; portal-approach clearance is the enforced interpretation of the frozen 64×80 + navigation minimums (§7.1/§20.7). No frozen values changed.

## Risks

- The frozen corpus manifest oracle (`tests/fixtures/enhanced_v3_richness_corpus/manifest.json`) is byte-identical to HEAD and was not regenerated; affected-seed hashes in the manifest remain historical. The 36-entry corpus still passes generation/determinism; map bytes for cave/prop seeds legitimately differ.
- `validate_protected_routes` is stricter: any future late free-form brush that enters a portal approach fails generation rather than producing a blocked map — intended, but a possible source of seed regressions that must be handled as placement constraints, not by weakening the validator.

## Follow-up Items

- None blocking. Future: all-inherited seed sweep beyond 99/255 as regression breadth; consider corpus-manifest versioning policy for geometry-correcting fixes.
