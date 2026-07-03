# Phase 01 Worker Directive: Physics Crate Alpha Contract

## Objective

Expand `src/physics` from a thin Rapier wrapper into a renderer-independent alpha physics API with durable wrapper IDs, body/collider descriptors, basic shapes, deterministic queries, and contact/trigger event records.

## User-Visible Outcome

Engine users can create a small physics world, register durable body/collider IDs, run simple queries, step simulation, and inspect collision/trigger outcomes without starting Vulkan.

## Editable Targets

- `src/physics/Cargo.toml`
- `src/physics/src/lib.rs`
- Optional new modules under `src/physics/src/`
- Optional phase notes under `reports/phase-01-*`

## Forbidden Scope

- Do not edit renderer, package/scene validation, engine_pack, editor, or dogfood in this phase.
- Do not add dependencies from `physics` to `renderer`, `ash`, `winit`, `imgui`, editor, dogfood, or Vulkan-facing crates.
- Do not serialize Rapier handles or expose them as durable identity.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- Top-level `AGENTS.md`

## Senior-Engineer Guidance

- Keep Rapier handles internal. A public runtime lookup token may exist only if clearly not serialized; durable authored identity must be string/newtype based.
- Add validation constructors or `Result`-returning creation paths for finite positive dimensions.
- Prefer a small tested API over a large incomplete facade.
- Contact/trigger event extraction must be deterministic enough for unit tests; if Rapier event handlers are too broad, inspect contact/intersection pairs after stepping and document the alpha limits.
- Preserve the existing simple create/attach/step behavior through compatibility wrappers if practical; otherwise provide a migration note in the phase report.

## Ordered Implementation Steps

1. Audit current `src/physics/src/lib.rs` tests and Rapier usage.
2. Introduce durable ID newtypes or structs for bodies and colliders.
3. Introduce body/collider descriptor types, body kind enum, shape enum, trigger/sensor flag, transform/offset representation, and validation errors.
4. Rework or wrap body/collider creation so callers can create static/dynamic bodies and attach cuboid plus at least one of sphere/capsule.
5. Add a deterministic query API such as ray cast or point/shape overlap.
6. Add contact/trigger record extraction with durable IDs and contact phases where practical.
7. Add unit tests for:
   - existing body fall behavior;
   - descriptor validation rejects invalid dimensions;
   - durable IDs map to runtime bodies/colliders;
   - query hit/miss behavior;
   - contact or trigger record extraction;
   - dependency-free compile expectations.
8. Run validation commands and record notable existing warning noise for the validator.

## Acceptance Criteria

- `cargo test -p physics` passes.
- `cargo check -p physics` passes.
- Public physics API has durable wrapper IDs/descriptors and does not force raw Rapier handles into authored code.
- At least cuboid plus one sphere/capsule-style shape is supported and tested.
- At least one query path is supported and tested.
- Contact/trigger records exist and are tested or a clearly documented stop condition explains why the phase must be revised.
- No renderer/Vulkan dependency is introduced.

## Negative Checks

- `rg -n "renderer|ash|vulkan|winit|imgui|dungeon_dogfood|editor" src/physics` must not show imports/dependencies except explanatory comments if unavoidable.
- No `Serialize`/`Deserialize` implementation should serialize Rapier `RigidBodyHandle` or `ColliderHandle`.
- No tests should require a window, GPU, or renderer startup.

## Validation Commands

```bash
cargo fmt --check
cargo test -p physics
cargo check -p physics
cargo check
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/validation/phase-01-validation-report.md`
- Implementation handoff should list commands run and whether warning noise is existing or new.
- Do not mark `artifacts/validation-summary.json` as passed from implementation alone.

## Stop Conditions

- Stop if durable IDs create a dependency cycle or require renderer types.
- Stop if query/event extraction cannot be made deterministic without broad redesign.
- Stop if supporting capsule/kinematic/trigger behavior would substantially expand beyond this phase; document the smallest safe subset and ask for planning revision.

## Do Not Close Unless

- Physics tests prove the new API behavior without Vulkan.
- Dependency hygiene has been checked.
- Runtime handles remain internal or clearly runtime-only.
- The phase is ready for validator review.

