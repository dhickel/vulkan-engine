# BSP Beta EnhancedV3 Generation Explorer

## Date

2026-07-31

## Change Summary

Added `bsp_beta --m3-generate`, strict full-closure generation and loading through `engine_pack`, off-thread hot regeneration, live F5/F6/F7/F8/F9/Ctrl+R controls, transactional replacement, and explicit renderer retirement handoff.

## Files

- `apps/bsp_beta/Cargo.toml`
- `apps/bsp_beta/src/cli.rs`
- `apps/bsp_beta/src/generation.rs`
- `apps/bsp_beta/src/lib.rs`
- `apps/bsp_beta/src/main.rs`
- `apps/bsp_beta/tests/m3_generation.rs`
- `apps/bsp_beta/tests/runtime_cli.rs`
- `docs/guide/18-bsp-beta.md`
- `.internal-dev/specifications/bsp-transaction-ownership.md`

## Behavioral Impact

`bsp_beta --m3-generate` starts with seed 42, Sparse, extent 2048, builds a complete package with pinned ericw-tools, and authorizes BSP, LIT, WAD, palette, and PBR texture companions in strict mode. Explicit, environment, HOME, and PATH tool discovery accepts only executable `qbsp`, `vis`, and `light` files on Unix. Windowed controls regenerate unique package closures on a background worker: F5 increments seed, F6 cycles preset/extent, F7 toggles chamfer, F8 cycles arch type, F9 toggles stairs, and Ctrl+R rebuilds unchanged state. Stale results are discarded, prepublication failures preserve the old world, successful replacement refreshes camera and app-owned entity state, and detached mounts enter renderer fence-aware retirement.

## Specification Impact

Updated the BSP transaction ownership contract for explicit coordinator-to-renderer retirement custody and normal/terminal closure reaping. Generator defaults and frozen v1/v2 output contracts are unchanged.

## Risks

Automated tests cover key semantics, request coalescing, stale-result custody, full ericw publication, and strict authorization. Real WSI startup and continuous rendering pass, but no input-injection utility was available under the Wayland session to automate physical hotkey presses.

## Follow-up Items

- GitHub #70 was resolved by measuring strict-extracted static render batches instead of source brushes; the full BSP beta suite passes.
- GitHub #60 remains the generic committed-bridge teardown concern.
