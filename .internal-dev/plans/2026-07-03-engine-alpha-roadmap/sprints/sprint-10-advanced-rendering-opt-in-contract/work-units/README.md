# Work Units

## Phase 01: Contract Audit And Sprint Spec Confirmation

Directive: `worker-directives/phase-01-contract-audit.md`

Audit current source/docs/examples and produce `reports/phase-01-advanced-api-audit.md`. This phase should not implement product changes.

## Phase 02: Feature-Gate And Documentation Hardening

Directive: `worker-directives/phase-02-feature-gate-docs.md`

Harden the default vs advanced documentation/export contract and add focused checks where practical.

## Phase 03: Minimal Named Advanced Surface Or Deliberate Defer

Directive: `worker-directives/phase-03-named-advanced-surface.md`

Add a narrowly safe named advanced surface only if Phase 01/02 prove it can be done without raw handles or hidden sync contracts. Otherwise record a deliberate defer with validation.

## Phase 04: Final Docs, Evidence, And Quality Review Prep

Directive: `worker-directives/phase-04-final-validation-closeout.md`

Reconcile docs/evidence, run the final validation matrix, and prepare final validator handoff.

## Dependencies

- Phase 02 depends on Phase 01 audit findings.
- Phase 03 depends on Phase 02 feature/doc hardening.
- Phase 04 depends on all implementation phases and validators passing or producing scoped remediation.
