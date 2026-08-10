# Bug Report: Richness V1 seed 99 — wall blocks portal and narrow passages below player clearance

## Summary

On seed 99, the EnhancedV3 Richness V1 generator (`m3-richness-v1`) produced a wall that blocked a portal (door) and multiple brush/wall placements whose openings and inter-brush gaps were narrower than the frozen player-clear minimums. Fixed and validated by senior agent escalation (2026-08-09).

## Scope

- Profile: `m3-richness-v1` (EnhancedV3 Richness V1), `dungeon-gen/v3-richness/v1`
- Affected modules: `src/bsp_generator/src/enhanced_v3/richness/{cave,composition,props,validation,pipeline}.rs`
- Also affected: chamfered North/South wall role convention (reversed relative to topology) and shared-wall portal lookup orientation

## Reproduction

1. `cargo run -p dungeon_gen -- --seed 99 --class m3-richness-v1 --out /tmp/seed99.map`
2. Inspect portal 4: throat `(1536,656,16)..(1552,720,96)`.
3. Cave floor/wall brushes began at `x=1584`, leaving only 32 units of room-side approach.

## Expected

- Every portal throat keeps the frozen exact 64×80 clear swept volume **plus** a 64-unit clear approach on each side of the owning wall.
- Every cave passage, prop gap, and wall-treatment opening retains ≥ 64 units horizontal and ≥ 80 units vertical clearance (player passage contract, spec §7.1/§20.7/§21).

## Actual

- Seed 99 portal 4 approach pinched to 32 units by a cave wall starting at `x=1584`.
- Existing validation only checked the 16-unit-deep 64×80 throat box, so the obstruction passed all gates; the frozen corpus missed it because its request bytes were explicit while the CLI/all-inherited path differed (same latent-cause class as the phase-17 `WallToDiagCorner` defect).

## Evidence

- Pre-fix map SHA-256: `ca20b76c…`; post-fix map SHA-256: `cd824b29…` (warning-free `qbsp → vis → light`, portal point witnesses and stored player-hull traces pass).
- Pre/post artifacts in `/tmp/seed99-before.map`, `/tmp/seed99-final.map`, `/tmp/seed99-compiled-final/`.
- Regression test: `pipeline::tests::seed_99_inherited_and_explicit_keep_portals_and_passages_clear` (also asserts inherited CLI defaults == explicit corpus bytes geometry identity).
- Validation: `cargo check -p bsp_generator`, `cargo test -p bsp_generator --release --lib` (1082 passed), 36-entry corpus 36/36, legacy v1/v2 closure 24/24, richness compatibility/portal/cave/vertical suites.

## Impact

- Player-blocked dungeon paths and doorways on affected seeds with cave/prop/late-free-form placement; reachability and navigation guarantees (§7.3) violated at runtime.

## Status

FIXED — committed with the seed-99 portal-approach clearance repair.

## Next Action

None. Follow-up: none blocking; monitor corpus manifest (unchanged oracle) and future all-inherited seed sweeps.
