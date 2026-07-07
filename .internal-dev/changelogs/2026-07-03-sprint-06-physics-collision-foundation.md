# Sprint 06 Changelog: Physics and Collision Foundation

Date: 2026-07-03

Branch: `sprint/alpha-06-physics-collision-foundation`

## Completed

- Expanded `src/physics` into a renderer-independent alpha physics API with durable body/collider IDs, descriptors, primitive collider shapes, ray queries, contact records, and event bridge helpers.
- Added package collision metadata validation through renderer asset registry validators and `engine_pack`.
- Added scene collision metadata validation for durable body/collider IDs, primitive shapes, offsets, triggers, and project-known collision asset references.
- Wired physics records to optional `engine_events` emission without making `engine_events` depend on `physics`.
- Recorded dogfood migration debt instead of replacing the bespoke capsule/tile/ramp/step solver during this sprint.
- Updated public and internal docs for physics/collision contracts, CLI validation, runtime deferrals, and true headless draw capture policy.

## Validation

- `cargo fmt --check`
- `cargo check`
- `cargo test -p physics`
- `cargo test -p engine_events`
- `cargo test -p renderer`
- `cargo test -p engine_pack`
- `cargo check -p physics`
- `cargo check -p renderer --examples`
- `cargo check -p editor`
- `cargo check -p dungeon_dogfood`
- `cargo check -p engine_pack`
- `cargo test -p dungeon_dogfood`
- stale-reference sweep over `docs` and the Sprint 06 plan directory

All commands passed. Existing renderer/editor/dogfood warning noise remains outside this sprint.

## Deferred

- Runtime scene collision loading into `physics::PhysicsWorld`.
- Editor UI collision authoring.
- Mesh-derived collision generation.
- Dogfood migration to the new physics crate.
- Physics debug rendering and broader gameplay orchestration.
