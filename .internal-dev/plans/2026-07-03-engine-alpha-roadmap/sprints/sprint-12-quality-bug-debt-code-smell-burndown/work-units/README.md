# Work Units

## Phase 01: Residual Inventory And Triage Lock

Directive:

`worker-directives/phase-01-residual-inventory-triage.md`

Outcome:

- verified inventory of current residuals;
- priority classification;
- accepted/deferred/fix-now split;
- recommended phase scope adjustments before code changes.

## Phase 02: Vulkan Lifecycle, Destroy, Swapchain, And Shutdown Hardening

Directive:

`worker-directives/phase-02-vulkan-lifecycle-hardening.md`

Outcome:

- confirmed or remediated shutdown/destroy/swapchain risks;
- runtime smoke evidence for touched behavior;
- updated docs/bug records for lifecycle residuals.

## Phase 03: Runtime Panic, Error, Frame, And Asset Stall Hardening

Directive:

`worker-directives/phase-03-runtime-panic-stall-hardening.md`

Outcome:

- high-risk runtime panics converted or justified;
- bounded/measured stall behavior where touched;
- focused tests and debug-record evidence.

## Phase 04: Docs, Examples, Public-Contract Drift, And Test Gaps

Directive:

`worker-directives/phase-04-docs-examples-test-drift.md`

Outcome:

- docs/examples align with current alpha contracts;
- stale historical residual claims corrected;
- test gaps filled where scope allows.

## Phase 05: Final Validation Matrix And Residual Acceptance

Directive:

`worker-directives/phase-05-final-residual-acceptance.md`

Outcome:

- complete evidence index;
- residual acceptance ledger;
- final validation readiness;
- no unearned `fully_validated` status.

## Sequencing

Run phases sequentially. Phase 01 can stop or reshape later phases if the inventory proves the burn-down is too broad or if a critical defect needs a dedicated sprint.
