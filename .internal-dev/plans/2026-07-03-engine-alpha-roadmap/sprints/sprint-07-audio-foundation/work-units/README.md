# Work Units

## Phase 01: Audio Crate Alpha Contract

Directive: `worker-directives/phase-01-audio-crate-alpha-contract.md`

Build the renderer-independent audio facade, replace required device-dependent tests with device-independent coverage, and gate any device smoke.

## Phase 02: Package/Scene Audio Metadata

Directive: `worker-directives/phase-02-package-scene-audio-metadata.md`

Add durable audio metadata validation through package/scene/project/CLI flows.

## Phase 03: Event Bridge And Sample/Dogfood Proof

Directive: `worker-directives/phase-03-event-bridge-dogfood-proof.md`

Bridge audio status into events and provide a narrow opt-in sample/dogfood playback proof or explicit debt artifact.

## Phase 04: Docs And Final Validation

Directive: `worker-directives/phase-04-docs-final-validation.md`

Update public/internal docs, evidence summary, stale-reference sweep, and final closeout preparation.

## Dependency Order

Phase 01 must pass before Phase 03. Phase 02 can begin after Phase 01 design is stable enough to know the durable clip shape. Phase 04 must run last.
