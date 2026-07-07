# Phase 03 Worker Directive: Event Bridge And Dogfood Proof Gate

## Objective

Bridge physics query/contact/trigger outcomes to Sprint 05 `EngineEvent::Physics` contracts, then make the dogfood migration decision through a narrow proof or an explicit debt record.

## User-Visible Outcome

Physics events are observable through the engine event vocabulary, and dogfood collision migration status is no longer implicit.

## Editable Targets

- `src/physics/src/lib.rs` and optional physics modules
- `src/physics/Cargo.toml` if an `engine_events` dependency or feature is needed
- `src/events/src/lib.rs` only if a real contract gap is found
- Optional non-render sample/test under an appropriate existing crate/test location
- Dogfood proof targets only if narrow and low risk:
  - `apps/dungeon_dogfood/src/collision.rs`
  - `apps/dungeon_dogfood/src/player.rs`
  - `apps/dungeon_dogfood/src/main.rs`
  - `apps/dungeon_dogfood/src/layout.rs`
  - `apps/dungeon_dogfood/src/scene_seed.rs`
- If migration is deferred: `reports/dogfood-physics-migration-debt.md`

## Forbidden Scope

- Do not rewrite dogfood movement/collision broadly.
- Do not make `engine_events` depend on `physics`.
- Do not introduce renderer/Vulkan dependency into event bridge tests.
- Do not claim dogfood is migrated unless player movement uses the new foundation in a real path and tests prove it.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/events/src/lib.rs`
- `apps/dungeon_dogfood/src/collision.rs`
- Relevant dogfood tests in `apps/dungeon_dogfood/src/`

## Senior-Engineer Guidance

- Preferred dependency direction is `physics -> engine_events` or a small bridge helper outside `engine_events`; never `engine_events -> physics`.
- Bridge functions should be pure and unit-testable.
- Existing `PhysicsEvent::QueryHit` has body/collider IDs but no ray distance; keep extra physics-specific query detail in physics records unless event vocabulary must change.
- Dogfood's ramp/floor stepping is not generic rigid body physics. Treat it as legacy gameplay behavior until a targeted adapter proves parity.
- A good debt record is an acceptable phase outcome if it names blockers, migration slices, and validation commands.

## Ordered Implementation Steps

1. Inspect Phase 01 event/query records and Sprint 05 `engine_events` physics event shapes.
2. Add bridge conversions from physics contact/trigger/query records to `EngineEvent::Physics` using durable IDs and `ContactPhase`.
3. Add unit tests for collision, trigger, and query conversion paths.
4. Add a small non-render sample/test demonstrating a physics step/query/event bridge without Vulkan.
5. Inspect dogfood collision tests and current movement call sites.
6. Decision gate:
   - If a narrow adapter/proof can preserve current tests, implement the proof and add tests.
   - If a safe proof is not bounded, write `reports/dogfood-physics-migration-debt.md` with current custom behaviors, blockers, proposed migration slices, and future validation.
7. Run validation commands.

## Acceptance Criteria

- Physics outcomes can map to `engine_events::EngineEvent::Physics` without renderer/Vulkan.
- Bridge behavior is tested.
- `engine_events` remains independent from `physics`.
- Dogfood decision is explicit: proof path with tests or migration debt report.
- Dogfood still compiles; if touched, existing dogfood tests for collision/layout/player behavior still pass.

## Negative Checks

- No broad dogfood rewrite.
- No silent deletion of ramp/floor stepping tests.
- No event vocabulary changes that break Sprint 05 docs/tests without corresponding docs updates planned for Phase 04.
- No renderer/Vulkan dependency in bridge tests.

## Validation Commands

```bash
cargo fmt --check
cargo test -p physics
cargo test -p engine_events
cargo check -p dungeon_dogfood
cargo check -p editor
cargo check
```

If dogfood collision/layout/player code is touched, also run the relevant package tests:

```bash
cargo test -p dungeon_dogfood
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/validation/phase-03-validation-report.md`
- If dogfood migration is deferred, expected artifact: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/reports/dogfood-physics-migration-debt.md`
- Implementation handoff should state the dogfood decision and commands run.

## Stop Conditions

- Stop if event bridge requires `engine_events` to depend on `physics`.
- Stop if dogfood proof grows into a broad migration.
- Stop if existing dogfood tests fail due to proof work and repair is not narrow.

## Do Not Close Unless

- Event bridge tests pass.
- Dogfood gate has a concrete proof or debt artifact.
- Dependency direction is verified.
- The phase is ready for validator review.

