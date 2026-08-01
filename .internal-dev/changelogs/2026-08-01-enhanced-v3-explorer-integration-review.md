# EnhancedV3 Explorer Integration Review

## Date
2026-08-01

## Change Summary
Completed final integration review of the EnhancedV3 explorer commit series through `0000f2da`. Repaired BSP beta tool discovery so Unix discovery accepts only executable `qbsp`, `vis`, and `light` files; regular non-executable files now fail at the declared discovery boundary. Corrected runtime and package corpus evidence to measure strict-extracted static batches, and removed speculative late config fields that were not in the authorized explorer inventory.

## Files
- `apps/bsp_beta/src/generation.rs`
- `.internal-dev/knowledge/bsp-beta.md`

## Behavioral Impact
Generated BSP beta launches fail early and clearly when an explicit or discovered tool directory contains non-executable compiler files. Valid executable tool directories retain the existing discovery and generation behavior.

## Specification Impact
none. The implementation now enforces the existing documented requirement that the discovered compiler files are executable.

## Risks
Unix execute-bit semantics are intentionally platform-specific; non-Unix platforms retain the previous regular-file check. Existing compiler invocation remains the final authority for executable compatibility.

## Follow-up Items
- Renderer `bsp_lifecycle` evidence tests still abort during VMA teardown on this host. The test source predates the reviewed commits and the failure is consistent with GitHub #64's headless-renderer VMA allocation-leak class; it requires separate renderer lifecycle investigation.
- GitHub #69 remains the independent stale EnhancedV3 qualification-manifest issue.
- GitHub #70 is resolved: `cargo test -p bsp_beta` now passes with actual static-batch evidence.
