# EnhancedV3 Static-Batch Evidence Repair

## Date

2026-08-01

## Change Summary

Corrected EnhancedV3 qualification, package-corpus, and BSP beta runtime evidence to distinguish source brush density from renderer static batches. Repaired the package corpus's atomic publication destination and made compiler-available publication/extraction failures fail closed.

## Files

- `apps/bsp_beta/tests/enhanced_v3_runtime.rs`
- `src/bsp_generator/tests/enhanced_v3_qualification.rs`
- `tools/engine_pack/tests/enhanced_dungeon_v3_corpus.rs`
- `.internal-dev/bugs/.archive/enhanced-v3-runtime-evidence-brush-budget/report.md`

## Behavioral Impact

Evidence now strict-loads and strict-extracts compiled BSP/LIT/WAD/palette closures and compares `render_batches.len()` with the `<500` static-batch ceiling. Source brush count remains descriptive structural-density data. The engine-pack corpus publishes to a nonexistent child of its temporary root, satisfying atomic no-replace semantics, and fails when any compiler-available entry does not publish, strict-extract, or meet budgets.

## Specification Impact

Specification Impact: none. The change makes tests measure the already-specified renderer static-batch contract rather than an unrelated source-brush count.

## Risks

Real ericw corpus execution remains host/tool dependent. When the pinned compiler is unavailable, the source-only cells remain distinct from compiled static-batch evidence rather than being reported as compiled passes.

## Follow-up Items

GitHub #70 is closed. GitHub #69 remains the separate stale qualification-manifest issue and was not rebaselined.
