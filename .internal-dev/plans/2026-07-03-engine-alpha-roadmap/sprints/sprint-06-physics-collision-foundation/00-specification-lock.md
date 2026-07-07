# Specification Lock

## Locked Objective

Build the alpha foundation for physics and collision contracts across the physics crate, package/scene metadata, event integration, docs, and dogfood migration decision records.

## Acceptance Criteria

- Physics crate remains renderer-independent and testable with `cargo test -p physics`.
- Public physics API uses engine-owned durable wrapper IDs and descriptors, not raw Rapier handles as serialized identity.
- Basic shapes include at least cuboid and one rounded/curved actor shape such as sphere or capsule. Generated bounds may be a descriptor placeholder only if clearly documented as deferred.
- Supported body kinds include static, dynamic, and kinematic if the implementation can test them safely; otherwise static/dynamic must land and kinematic must be explicitly deferred.
- Queries include at least one deterministic Vulkan-free query path: ray cast, point overlap, shape cast, or shape overlap.
- Contact and trigger extraction is available from physics stepping or a documented event drain path.
- Package/scene collision metadata can round-trip through serialization and validation with typed body/shape descriptors.
- Validators reject invalid dimensions, unknown enum values, runtime handle-shaped identities, duplicate durable collision IDs, and unknown referenced collision assets when package records are available.
- Physics outcomes can be translated to `engine_events::PhysicsEvent` without renderer/Vulkan.
- Dogfood migration is gated by risk: narrow proof if it preserves current tests, otherwise a migration debt artifact is required.
- Public/internal docs describe implemented alpha contracts and deferred limits.

## Validation Criteria

- Required phase reports:
  - `validation/phase-01-validation-report.md`
  - `validation/phase-02-validation-report.md`
  - `validation/phase-03-validation-report.md`
  - `validation/phase-04-validation-report.md`
  - `validation/final-quality-review.md`
- `artifacts/validation-summary.json` must remain conservative until every required validator passes.
- Pure physics/metadata tests must not require Vulkan or visual capture.
- Runtime smoke is required only if phase work changes runtime/app behavior.
- True headless draw capture using `--headless --capture_target draw` is required only if visible renderer/editor behavior changes; desktop screenshots are not acceptable evidence.

## Negative Criteria

- No broad dogfood rewrite in Phase 01 or Phase 02.
- No editor UI claims for collision authoring.
- No serialization of `RigidBodyHandle`, `ColliderHandle`, slot/generation handle shapes, or other runtime identity.
- No hidden dependency from `physics` to renderer/editor/dogfood.
- No broad warning cleanup outside new sprint-caused warnings.
- No stale plan/evidence language claiming implementation is done during planning.

## Non-Goals

- Full gameplay physics replacement.
- Mesh collider generation from renderer mesh data.
- Editor property panels for collision components.
- Cross-thread physics service, async simulation, rollback, networking, or persisted replay.
- Production-ready character controller.

## Constraints

- Branch: `sprint/alpha-06-physics-collision-foundation`.
- Protected local state: do not touch `.idea/engine.iml` or `.reasonix/`.
- Main thread commits/pushes/emails each phase after validation; workers do not own those steps.
- `.internal-dev` is the durable planning/evidence store; do not read it broadly outside named artifacts.

## Assumptions

- `engine_events` from Sprint 05 is fully validated and available as the event vocabulary owner.
- Existing package/scene validation patterns in renderer should be extended instead of duplicated in `engine_pack`.
- Dogfood ramp/floor logic is behaviorally important and should be preserved unless a scoped proof demonstrates parity.
- Existing warnings may appear during validation; reports should distinguish existing noise from new warnings.

## User-Decision Gates

- If kinematic bodies, capsule shape support, or trigger extraction requires a broad Rapier redesign, stop and ask whether to defer or expand scope.
- If collision metadata requires a schema version bump with backward compatibility impact, stop for approval before locking format changes.
- If dogfood migration cannot preserve existing tests with a narrow adapter, record migration debt and do not force migration.
- If visible renderer/editor behavior changes unexpectedly, add `--headless --capture_target draw` validation before closeout.

## Stop Rules

- Stop on dependency cycles involving `physics`, `renderer`, or `engine_events`.
- Stop on any plan criteria conflict with current code contracts that cannot be resolved locally.
- Stop if a required validator model/tool is unavailable and record `TOOLING_CONSTRAINT`.
