# Work Units

## Phase 01: Shared Validation Contract

Expose or add shared Rust validation for project/package/scene files. Strengthen project validation and make scene validation available for CLI use without Vulkan/runtime asset upload.

## Phase 02: CLI Validation Commands

Create `tools/engine_pack`, wire workspace membership, implement `validate-package`, `validate-project`, and `validate-scene`, and add valid/invalid fixtures.

## Phase 03: Authoring And Pack Commands

Implement `new-project`, `new-package`, `scan-assets`, `add-asset`, and `pack` with tests proving generated output revalidates.

## Phase 04: Docs And Final Evidence

Update docs, run final command set, reconcile capture decision, update evidence summary, run final quality validation, and prepare closeout.

## Dependency Order

Phase 02 depends on Phase 01. Phase 03 depends on Phase 02. Phase 04 depends on all implementation phases.

Do not parallelize dependent phases.
