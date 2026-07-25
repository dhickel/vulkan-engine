# Date
2026-07-24

# Change Summary
Added a repository launch helper that generates or reuses cached BSP dungeon maps by seed/class and launches `bsp_beta` with the CC0 stone beta companions.

# Files
- `tools/dungeon_explore.sh`
- `tools/dungeon_gen/Cargo.toml`
- `tools/dungeon_gen/src/main.rs`
- `Cargo.toml`
- `Cargo.lock`

# Behavioral Impact
Users can run `./tools/dungeon_explore.sh [seed] [m1|m2]` to explore generated BSP dungeons. Cached `.bsp` files under `.internal-dev/captures/bsp-dungeon-generator/` skip generation and compilation.

# Specification Impact
Specification Impact: none. This automates the existing BSP dungeon generator and BSP2 compile contract without changing the frozen generation parameters or runtime acceptance criteria.

# Risks
The first uncached run still requires the pinned ericw-tools binaries to be available either at the default local install path or on `PATH`.

# Follow-up Items
None.
