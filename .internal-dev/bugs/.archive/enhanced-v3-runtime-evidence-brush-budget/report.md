# EnhancedV3 Runtime Evidence Uses a Stale Source-Brush Ceiling

## Summary

`apps/bsp_beta/tests/enhanced_v3_runtime.rs::runtime_budget_and_spatial_evidence` labels EV-080 as a static-batch budget but fails current production maps by comparing source brush count to an undocumented `<500` ceiling. Current Moderate seed 4 and Rich seed 8 outputs contain 545 and 795 source brushes while remaining within the specified face/entity budgets and producing four runtime render batches in live validation.

## Scope

The defect is confined to the pre-existing BSP beta qualification/evidence assertion and its generated evidence record. The EnhancedV3 explorer implementation does not modify this test or the default map bytes.

## Reproduction

```bash
cargo test -p bsp_beta --test enhanced_v3_runtime runtime_budget_and_spatial_evidence -- --nocapture
```

## Expected

EV-080 should measure its declared static-batch criterion, or use a documented source-brush budget aligned with the EnhancedV3 specification. Current byte-compatible production maps should pass when they satisfy the governing M2 face/entity ceilings and runtime batch criterion.

## Actual

The test sets `brushes_ok: source_brushes < 500` and fails at least `v3-moderate-seed-4` (545 brushes); Rich seed 8 reports 795 brushes. The same live generated map uploads as four renderer batches with no failed draws.

## Evidence

- Failing assertion: `apps/bsp_beta/tests/enhanced_v3_runtime.rs:602`
- Misclassified threshold: `apps/bsp_beta/tests/enhanced_v3_runtime.rs:462`
- Test header declares EV-080 as `Static Batch Budget (< 500 static batches)`.
- Default EnhancedV3 output was independently compared byte-for-byte against the task base during the explorer work.
- Live `bsp_beta --m3-generate` reports four mounted/rendered batches.

## Impact

The broad `cargo test -p bsp_beta` suite cannot pass even when the runtime implementation is correct, and the generated evidence conflates source authoring complexity with runtime batch count.

## Status

Resolved by commit `7f5bbfae`; mirrored GitHub issue [#70](https://github.com/dhickel/vulkan-engine/issues/70) is closed.

## Next Action

None. EV-080 now extracts each compiled representative map and measures `render_batches.len()` against the `<500` static-batch ceiling while retaining source brush count as descriptive evidence. Sparse, Moderate, and Rich pass without changing frozen generator bytes.
