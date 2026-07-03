# Phase 03 Validation Report: Event Bridge And Dogfood Proof Gate

## Result

Passed.

## Implementation Summary

- Added `physics -> engine_events` dependency.
- Added pure bridge helpers from `RayHit` and `PhysicsContactRecord` to `EngineEvent::Physics`.
- Added helper functions to convert and emit contact records into an `EventBus`.
- Preserved `engine_events` independence from `physics`.
- Corrected trigger records so the trigger/sensor collider is first in trigger event records.
- Made the dogfood migration decision explicit with a debt artifact.

## Dogfood Decision

Migration deferred. See:

`.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/reports/dogfood-physics-migration-debt.md`

Dogfood's current collision path includes gameplay-specific capsule movement,
ramp height solving, step limits, and camera-intent handling. A narrow proof
would not preserve the current behavior without becoming a broad movement
rewrite.

## Files Changed

| File | Lines | Notes |
|---|---:|---|
| `src/physics/Cargo.toml` | 8 | Adds `engine_events` dependency |
| `src/physics/src/lib.rs` | 932 | Event bridge helpers and tests |
| `reports/dogfood-physics-migration-debt.md` | 62 | Dogfood migration decision and future slices |

## Commands

| Command | Result | Notes |
|---|---|---|
| `cargo fmt --check` | Pass | Formatter clean |
| `cargo test -p physics` | Pass | 11 unit tests |
| `cargo test -p engine_events` | Pass | 7 unit tests |
| `cargo check -p dungeon_dogfood` | Pass | Existing renderer/dogfood warning noise remains |
| `cargo check -p editor` | Pass | Existing renderer warning noise remains |
| `cargo check` | Pass | Existing renderer warning noise remains |
| `cargo tree -p engine_events` | Pass | No dependency on `physics` |
| `cargo test -p dungeon_dogfood` | Pass | 40 tests, run as behavior baseline even though dogfood source was not touched |

## Acceptance Criteria

| Criteria | Status | Evidence |
|---|---|---|
| Physics outcomes map to `EngineEvent::Physics` | Pass | Collision, trigger, and query conversion tests |
| Bridge behavior is renderer/Vulkan-free | Pass | Physics unit tests use `EventBus` only |
| `engine_events` remains independent from `physics` | Pass | `cargo tree -p engine_events` has no dependencies |
| Dogfood decision is explicit | Pass | Migration debt artifact created |
| Dogfood still compiles | Pass | `cargo check -p dungeon_dogfood` |
| Dogfood current behavior remains green | Pass | `cargo test -p dungeon_dogfood`, 40 tests |

## Capture

No visible renderer/editor behavior changed in Phase 03. No headless draw capture
was required. Future visual validation must use true engine-owned capture with
`--headless --capture_target draw`.

## Residual Notes

- Dogfood gameplay still uses its legacy custom collision solver.
- Physics event bridge exposes current event vocabulary; query distance remains in `RayHit`, not in `EngineEvent::Physics::QueryHit`.
