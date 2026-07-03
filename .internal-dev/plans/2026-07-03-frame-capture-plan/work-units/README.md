# Work Units

Dispatch phases in order. Each phase mutates product code or durable artifacts and must be validated before dependent phases proceed.

## Phase 01: Capture Contracts, Scheduler, And Launch Parsing

Directive: `worker-directives/phase-01-capture-contracts-and-cli.md`

Purpose: add typed capture configuration, scheduling semantics, public facade entrypoints, example/editor parser support, focused tests, and no-op backend plumbing.

Dependency: none.

## Phase 02: Windowed Vulkan Capture Path

Directive: `worker-directives/phase-02-vulkan-capture-windowed.md`

Purpose: harden image readback, split terminal present transition, add capture execution for present/draw targets, write PNG/sidecar output, and prove windowed capture.

Dependency: phase 01 validation pass.

## Phase 03: Headless/Offscreen Capture

Directive: `worker-directives/phase-03-headless-offscreen-capture.md`

Purpose: implement true headless/offscreen rendering/capture using engine-owned images and a no-present path, or stop at the explicit user-decision gate with blocker evidence.

Dependency: phase 02 validation pass.

## Phase 04: Manual Input And Editor Integration

Directive: `worker-directives/phase-04-manual-and-editor-integration.md`

Purpose: wire F12/manual capture defaults and editor launch/runtime support without disrupting existing input/debug behavior.

Dependency: phase 03 validation pass or approved headless fallback.

## Phase 05: Docs, Validation Harness, And Evidence Index

Directive: `worker-directives/phase-05-docs-validation-evidence.md`

Purpose: finalize docs, helper validation tooling if needed, full PNG proof matrix, and canonical evidence index.

Dependency: phase 04 validation pass.

