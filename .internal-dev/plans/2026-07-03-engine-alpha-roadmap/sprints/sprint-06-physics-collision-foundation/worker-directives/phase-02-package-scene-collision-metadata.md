# Phase 02 Worker Directive: Package/Scene Collision Metadata

## Objective

Add typed collision metadata to package and scene contracts, plus validation through renderer APIs and `engine_pack`.

## User-Visible Outcome

Authors can store collision descriptors in package/scene files, and validation rejects bad collision metadata before runtime.

## Editable Targets

- `src/renderer/src/data/asset_registry.rs`
- `src/renderer/src/api/scene.rs`
- `tools/engine_pack/src/main.rs` only if CLI output/wiring needs adjustment
- `tools/engine_pack/tests/cli_validation.rs`
- Optional fixtures under existing test temp paths if local pattern uses them
- Optional `reports/phase-02-*`

## Forbidden Scope

- Do not change physics simulation behavior in this phase except small shared type imports if Phase 01 exposed reusable descriptor types and dependency direction is safe.
- Do not add renderer dependency to `physics`.
- Do not implement editor UI authoring.
- Do not add CPU mesh-bound generation unless explicitly required by tests and scope is revised.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/data/AGENTS.md`
- Existing validation tests in `src/renderer/src/data/asset_registry.rs` and `src/renderer/src/api/scene.rs`

## Experience Contract

This phase has no UI surface. The authoring experience is file/schema validation:

- valid files should continue to validate when collision metadata is absent;
- invalid collision data should fail with specific diagnostics;
- CLI users should see validation failures through existing `engine_pack` commands.

## Senior-Engineer Guidance

- Extend existing validation paths instead of duplicating logic in `engine_pack`.
- Keep schema additions backward-compatible with old scene/package files.
- If using generic metadata maps, parse them into typed validation structs before accepting them.
- Runtime-handle rejection must recurse into collision metadata, not only existing asset fields.
- Use durable IDs for collision bodies/colliders; no slot/generation handle-shaped IDs.

## Ordered Implementation Steps

1. Inspect existing package and scene serialization/validation tests.
2. Choose the minimal schema shape for package collision metadata and scene node collision components.
3. Add typed structs/enums or typed parsing helpers for body kind, shape kind, dimensions, trigger flag, and durable IDs.
4. Add validation for:
   - missing/blank IDs where IDs are required;
   - duplicate collision IDs within a scene/package scope;
   - invalid dimensions/non-finite values;
   - unknown body/shape kind;
   - runtime handle-shaped identities or nested handle fields;
   - unknown referenced collision assets where known package records are available.
5. Ensure old package/scene files without collision metadata still validate.
6. Add package manifest tests for valid and invalid collision metadata.
7. Add scene serialization/validation tests for valid round-trip and invalid metadata.
8. Add or update `engine_pack` CLI tests that prove validation reports collision metadata failures.
9. Run validation commands.

## Acceptance Criteria

- Package collision metadata validates and rejects invalid cases.
- Scene collision metadata validates, round-trips, and rejects invalid cases.
- Runtime handle-shaped collision IDs/fields are rejected.
- `engine_pack` validation surfaces collision metadata failures.
- Backward-compatible files without collision metadata still validate.

## Negative Checks

- No Rapier handles in serialized examples or tests.
- No CLI-only validation drift from renderer validation.
- No editor UI claims or generated mesh-bound completion claims.
- No broad renderer scene graph refactor.

## Validation Commands

```bash
cargo fmt --check
cargo test -p renderer
cargo test -p engine_pack
cargo check -p engine_pack
cargo check
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-06-physics-collision-foundation/validation/phase-02-validation-report.md`
- Implementation handoff should name the exact schema accepted and diagnostic codes added/changed.
- `artifacts/validation-summary.json` remains non-final until validator reconciliation.

## Stop Conditions

- Stop if schema changes require a format-version bump with compatibility consequences.
- Stop if renderer must depend on `physics` in a way that creates a cycle or heavy runtime dependency.
- Stop if generated mesh bounds become necessary to satisfy acceptance criteria; return for plan revision.

## Do Not Close Unless

- Positive and negative schema tests exist.
- CLI validation has at least one collision-specific failure test.
- Old files remain accepted.
- The phase is ready for validator review.

