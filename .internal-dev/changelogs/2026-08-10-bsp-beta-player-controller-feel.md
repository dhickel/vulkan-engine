# Changelog: BSP beta player controller feel — asymmetric box, faster walk, wall slide, no-clip

## Date
2026-08-10

## Git Commit

11c8e95a
## Change Summary

Adjusted the BSP beta player movement controller: the player is now a skinnier asymmetric box (narrower sideways than forward), walks faster, slides along walls instead of stopping dead, and supports a no-clip toggle (C to enable, Escape to disable).

Implementation history in one change:
1. Initial implementation added asymmetric extents (`PLAYER_HALF_EXTENTS_QUAKE = (10, 24, 16)`), `WALK_SPEED_ENGINE = 1.8`, `NO_CLIP_SPEED_ENGINE = 2.8`, multi-sample clipnode box traces, multi-clip sliding, and C/ESC no-clip wiring in both windowed paths.
2. Review probes proved a collision defect: in current ericw-compiled fixtures, stored hull 0 and hull 1 share the same pre-expanded player clipnode tree (world headnodes `[0, 0, <hull2>, 0]`), so sampling an app-owned box through `StoredHull::Point` double-expanded horizontal collision (nominal 20×32 box became effectively 52 wide) and broke step-up; the step-up regression was initially masked by weakened test assertions.
3. The fix moved horizontal collision to the unexpanded leaf tree: `point_contents_with_transform` box clearance over 8 corners + 6 face centers, binary-searched max clear horizontal fraction (8 iterations), axis-separated wall slide (remaining X then remaining Z), and a step-up fallback that permits the leaf-clear lift when the stored hull reports `starts_solid` beside a riser. Vertical movement (grounded probe, jump, headroom, climb, drop, ladder) stays on the stored player hull.

## Files

- `apps/bsp_beta/src/player_navigation.rs`
- `apps/bsp_beta/src/main.rs`
- `apps/bsp_beta/tests/richness_movement_contract.rs`
- `.internal-dev/knowledge/bsp-beta.md`

## Behavioral Impact

- Player box is asymmetric: 20 Quake units side-to-side (strafe), 32 forward/back, 48 tall; strafing fits 24-unit gaps that forward walking (32-unit length) cannot pass.
- Walk speed 1.0 → 1.8 engine units/s; no-clip flies at 2.8 with vertical input (Space up / Shift down).
- Wall contact slides along the blocking axis instead of stopping dead.
- C toggles no-clip on/off; Escape disables no-clip — in both `run_windowed` and `run_m3_generate_windowed`, only when no GUI mode owns input; free-camera (no Richness volumes) maps also fly at no-clip speed.
- Step-up onto 24-unit platforms restored (original `active_step_cell_climbs_bounded_platform` assertions x>72, z>=55 pass again).
- No frozen BSP trace contracts, `QuakeToEngine`, `StoredHull`, generator, compiler profile, or canonical request grammar changed.

## Specification Impact

Specification Impact: none. The change is scoped to BSP beta runtime gameplay feel; the frozen `bsp-spatial-physics.md` §5 stored-hull trace contracts and generator corridor/profile contracts are deliberately untouched (horizontal collision is app-owned against the unexpanded leaf tree, which is an existing public query, not a new BSP contract).

## Risks

- Step-up uses a narrow fallback when the stored player hull reports `starts_solid` beside a riser but the leaf-sampled asymmetric box is clear; intentionally limited to step lifting — jump/headroom/climb/drop vertical behavior remains on stored player-hull traces.
- The leaf tree is coarser than clipnodes for highly concave geometry; the generator's cardinal + 45° brush output is within the sampled fidelity, but non-generated third-party maps could exhibit edge-case differences at thin slabs.
- Live WSI/window smoke was not run for this change (no GPU/WSI validation required for the non-rendering defect; the full `bsp_beta` test suite passes).

## Follow-up Items

- None blocking. Future: live windowed feel-check of slide/no-clip, and consider a true point-hull clipnode tree if arbitrary-box clipnode tracing is ever needed.
