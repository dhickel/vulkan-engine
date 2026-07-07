# Work Units

## Dispatch Order

1. Phase 01: `worker-directives/phase-01-state-and-command-hardening.md`
2. Phase 02: `worker-directives/phase-02-save-reload-validation.md`
3. Phase 03: `worker-directives/phase-03-headless-capture-proof.md`
4. Phase 04: `worker-directives/phase-04-docs-final-closeout.md`

Phase 02 depends on Phase 01 because persistence tests should use the hardened placement contract. Phase 03 depends on Phase 02 because capture should use a saved scene copy or equivalent package-backed scene data path. Phase 04 depends on all earlier phase validation reports and capture evidence.

## Phase Boundaries

- Phase 01 owns editor state/action and command hardening. It may touch renderer command tests if needed, but not docs closeout.
- Phase 02 owns saved scene durability, reload behavior, and `engine_pack` validation. It should avoid UI redesign and visual capture work.
- Phase 03 owns deterministic capture proof and any minimal harness needed for visual validation. It should not broaden editor architecture.
- Phase 04 owns docs, stale-reference cleanup, validation summary finalization, changelog, and final quality review preparation.

## Main Thread Gates

After each phase validator passes, the main thread records commit, push, and email report evidence in `artifacts/validation-summary.json`. The phase worker should produce enough evidence for that gate but should not perform out-of-band coordination.
