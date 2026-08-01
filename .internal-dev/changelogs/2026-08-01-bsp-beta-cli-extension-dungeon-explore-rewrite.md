## Date
2026-08-01

## Change Summary
Completed the `--m3-generate` CLI extension for `bsp_beta` with `--seed`, `--preset`, `--rooms`, `--corridors`, `--loops`, `--chamfer`/`--no-chamfer`, `--arch-type`, and `--grammar-families` flags. Rewrote `tools/dungeon_explore.sh` to remove the interactive menu in favor of the in-game GUI, with architectural mode (m3) as the default.

## Files
- `apps/bsp_beta/src/cli.rs` — new generation flags, validation, conflict checks, `system_time_seed()` helper
- `apps/bsp_beta/src/main.rs` — `initial_m3_config()` wires CLI config into startup package and GUI draft
- `apps/bsp_beta/tests/runtime_cli.rs` — populated new `CliArgs` fields in struct literal
- `tools/dungeon_explore.sh` — complete rewrite: removed interactive menu, camera/stats/all-visible; architectural (m3) is default; m1/m2/m3 aliases preserved; cache system preserved for classic/enhanced
- `docs/guide/18-bsp-beta.md` — documented new generation flags, defaults, and examples
- `docs/guide/19-bsp-generator.md` — updated `dungeon_explore.sh` example syntax
- `.internal-dev/specifications/bsp-dungeon-generation.md` — recorded CLI contract in §20.17
- `.internal-dev/knowledge/bsp-enhanced-v3-production.md` — recorded launch workflow knowledge

## Behavioral Impact
- `bsp_beta --m3-generate` alone now defaults to system-time seed, moderate preset, chamfer enabled, pointed arches, all six grammar families
- `--development` is accepted alongside `--m3-generate` (previously rejected); generated packages always authorize strictly at runtime
- M3-specific options (`--seed`, `--preset`, etc.) are rejected without `--m3-generate`
- `dungeon_explore.sh` with no args launches directly into the architectural GUI editor; interactive terminal menu removed
- Classic/enhanced modes retain full cache-backed generation and validation

## Specification Impact
New CLI contract for `--m3-generate` flags recorded in `.internal-dev/specifications/bsp-dungeon-generation.md` §20.17.

## Risks
None. Backward-compatible: m1/m2/m3 aliases preserved, cache system preserved, `--bust`/`--cache-only`/`--development` flags preserved.

## Follow-up Items
None.
