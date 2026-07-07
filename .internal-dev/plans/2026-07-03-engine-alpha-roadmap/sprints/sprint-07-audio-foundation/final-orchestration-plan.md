# Final Orchestration Plan

## Dispatch Order

1. Dispatch Phase 01 using `worker-directives/phase-01-audio-crate-alpha-contract.md`.
2. Run Phase 01 validator and require `validation/phase-01-validation-report.md`.
3. After pass, main thread commits/pushes/reports Phase 01 if following sprint protocol.
4. Dispatch Phase 02 using `worker-directives/phase-02-package-scene-audio-metadata.md`.
5. Run Phase 02 validator and require `validation/phase-02-validation-report.md`.
6. After pass, main thread commits/pushes/reports Phase 02.
7. Dispatch Phase 03 using `worker-directives/phase-03-event-bridge-dogfood-proof.md`.
8. Run Phase 03 validator and require `validation/phase-03-validation-report.md`.
9. After pass, main thread commits/pushes/reports Phase 03.
10. Dispatch Phase 04 using `worker-directives/phase-04-docs-final-validation.md`.
11. Run Phase 04 validator and require `validation/phase-04-validation-report.md`.
12. Run final quality validator and require `validation/final-quality-review.md`.

## Remediation Routing

- `code_defect`: fresh scoped repair worker against the failed phase target.
- `docs_or_evidence_defect`: fresh scoped repair worker unless it is a trivial validator-owned typo/stale-link correction.
- `browser_harness_defect` or capture harness defect: repair harness/evidence first; change product code only after evidence proves a product bug.
- `plan_defect`: return to advanced planning for revised criteria/directives.
- `validator_error`: correct checklist or use a fresh validator before product repair.
- Same targeted issue failing twice: escalate to fresh high-reasoning repair worker per planning policy.

## Validation Gates

- Every mutating phase must pass its phase validator before dependent work proceeds.
- `artifacts/validation-summary.json` is the canonical evidence index and must remain conservative.
- Required final commands are listed in `shared/validation-matrix.md`.
- Device smoke is optional and host-dependent; core validation must pass without it.
- Runtime smoke is required only if runtime/app behavior changes, and any device-backed command must be explicitly enabled.
- True headless draw capture using `--headless --capture_target draw` is required only if visible renderer/editor behavior changes; no desktop screenshots.
- Final quality review is required for this large sprint.

## Dogfood/Sample Gate

Phase 03 must produce one of:

- narrow sample/dogfood proof path with packaged audio reference, opt-in playback, tests/checks, and device status reporting; or
- `reports/dogfood-audio-proof-debt.md` with blockers, future migration slices, and validation expectations.

Do not proceed to final closeout without one of these outcomes.

## Main-Thread Responsibilities

- Phase commits and pushes after validation.
- Email/report sending after each validated phase.
- Sprint tracker updates.
- Changelog timing and any `.internal-dev/changelogs/` entry after user confirmation.
- Asking before logging out-of-scope future considerations in `.internal-dev/notes/`.

## Closeout Gates

- All phase validation reports present.
- Final quality review passes.
- Evidence summary status is internally consistent.
- Docs do not overclaim device support, production audio features, or editor placement.
- Device smoke status is recorded as pass/skipped/blocked.
- Protected `.idea/engine.iml` and `.reasonix/` remain untouched.
- Sprint tracker updated only by main thread if needed.
