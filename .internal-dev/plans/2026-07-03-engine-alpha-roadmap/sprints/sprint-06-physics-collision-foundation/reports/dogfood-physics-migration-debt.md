# Dogfood Physics Migration Debt

## Decision

Dogfood physics migration is deferred for Sprint 06 Phase 03.

The current dogfood collision path is not just generic rigid-body collision. It
contains gameplay-specific capsule movement, tile-neighborhood wall resolution,
ramp height solving, step-up and step-down limits, and camera-intent guards.
Replacing that with the new alpha physics crate inside this phase would be a
broad movement rewrite, not a narrow proof.

## Current Proven Behavior

Current dogfood collision tests cover:

- wall sliding against nearby wall tiles;
- ramp ascent;
- ramp descent without large vertical snap;
- no wall penetration after iterative solver;
- finite/clamped player camera intent;
- level/ramp reachability and generated multi-layer traversal assumptions.

Validation command:

```bash
cargo test -p dungeon_dogfood
```

Result: pass, 40 tests.

## Blockers To Full Migration

- The alpha physics crate has durable IDs, basic shapes, ray query, contact/trigger records, and event bridging, but it does not yet provide a character controller abstraction.
- Dogfood's ramp logic solves floor height from authored tile ramps. The physics crate does not yet expose equivalent slope stepping semantics.
- Dogfood resolves player motion from renderer camera intent. A migration needs a stable app-level movement contract before replacing the solver.
- Current dogfood walls are tile AABBs and ramps are gameplay surfaces. Turning the level into authored collision metadata is possible, but not yet wired through a runtime scene/physics loader.
- Contact events alone do not preserve enough information for dogfood movement parity.

## Migration Slices

1. Add dogfood collision metadata export proof from `ParsedLevel` without changing gameplay movement.
2. Add a non-render physics-world builder from exported dogfood collision descriptors.
3. Add a character/capsule query helper in `physics` for sweep/slide or grounded movement.
4. Add ramp/slope validation cases to the physics crate before dogfood consumes it.
5. Add a dogfood adapter behind a feature or test-only path that compares legacy solver output to physics output for wall and floor cases.
6. Migrate one non-ramp wall collision path first, preserving all existing dogfood tests.
7. Migrate ramp behavior only after parity tests cover ascent, descent, step-up, step-down, and multi-layer reachability.

## Required Future Validation

- `cargo test -p physics`
- `cargo test -p dungeon_dogfood`
- dogfood movement parity tests for wall sliding, no penetration, ramp ascent, ramp descent, and camera-intent clamping;
- runtime smoke for dogfood if app behavior changes;
- true `--headless --capture_target draw` evidence only if visible renderer/editor behavior changes.

## Accepted Residual Risk

Sprint 06 Phase 03 makes physics events observable and preserves dogfood's
existing custom solver. It does not claim dogfood gameplay has migrated to the
new physics foundation.
