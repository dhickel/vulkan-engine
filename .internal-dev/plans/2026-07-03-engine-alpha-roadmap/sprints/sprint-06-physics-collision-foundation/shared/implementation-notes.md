# Implementation Notes

## Repo Contracts To Preserve

- Top-level repo guide requires `.internal-dev` for durable planning/evidence and controlled reads.
- `src/renderer/AGENTS.md`, `src/renderer/src/data/AGENTS.md`, and package-level guides should be read by workers before editing those areas.
- `src/input/AGENTS.md` is relevant only if event/input interactions are touched.
- Do not edit `.idea/engine.iml` or `.reasonix/`.

## Suggested Phase Dependencies

1. Phase 01 creates core physics descriptors, IDs, queries, and event records.
2. Phase 02 reuses or mirrors Phase 01 descriptor concepts in package/scene metadata validation.
3. Phase 03 bridges physics records to `engine_events` and makes the dogfood proof/debt decision.
4. Phase 04 updates docs, reconciles validation, and runs final quality review.

## Testing Notes

- Add physics tests close to `src/physics/src/lib.rs` or new physics modules.
- Add scene/package validation tests in the same modules that own existing validation tests.
- Add CLI validation tests in `tools/engine_pack/tests/cli_validation.rs` only after renderer validation has collision cases to call.
- Preserve existing dogfood collision tests if dogfood files are touched.

## Evidence Notes

- Phase workers may create brief implementation notes in `reports/` if useful, but validators own validation reports.
- `artifacts/validation-summary.json` starts as planning-only and must not claim pass/final status until validators update it.
- Main thread owns email/report sending and phase commit/push steps.

