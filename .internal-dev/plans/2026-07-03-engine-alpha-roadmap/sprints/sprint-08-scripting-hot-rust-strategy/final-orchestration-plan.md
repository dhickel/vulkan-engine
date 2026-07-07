# Final Orchestration Plan

## Dispatch Order

1. Dispatch Phase 01 using `worker-directives/phase-01-current-state-contract-audit.md`.
2. Run Phase 01 validator and require `validation/phase-01-validation-report.md`.
3. After pass, main thread commits/pushes/reports Phase 01 if following sprint protocol.
4. Dispatch Phase 02 using `worker-directives/phase-02-rust-app-template-path.md`.
5. Run Phase 02 validator and require `validation/phase-02-validation-report.md`.
6. After pass, main thread commits/pushes/reports Phase 02.
7. Dispatch Phase 03 using `worker-directives/phase-03-script-asset-event-boundary.md`.
8. Run Phase 03 validator and require `validation/phase-03-validation-report.md`.
9. After pass, main thread commits/pushes/reports Phase 03.
10. Dispatch Phase 04 using `worker-directives/phase-04-docs-final-validation.md`.
11. Run Phase 04 validator and require `validation/phase-04-validation-report.md`.
12. Run final quality validator and require `validation/final-quality-review.md`.

## Remediation Routing

- `code_defect`: fresh scoped repair worker for the failed target.
- `docs_or_evidence_defect`: fresh scoped repair worker unless it is a trivial validator-owned typo/stale-link correction.
- `capture_harness_defect`: repair capture harness/evidence first; change product code only after evidence proves a product bug.
- `plan_defect`: return to advanced planning for revised criteria/directives.
- `validator_error`: correct checklist or use a fresh validator before product repair.
- Same targeted issue failing twice: escalate to fresh high-reasoning repair worker.

## Validation Gates

- Every mutating phase must pass its phase validator before dependent work proceeds.
- `artifacts/validation-summary.json` is the canonical evidence index and must remain conservative.
- Required final commands are listed in `shared/validation-matrix.md`.
- `cargo test -p dungeon_dogfood` is conditional; inherited `russimp_sys` blocker must be recorded if encountered.
- True headless draw capture using `--headless --capture_target draw` is required only if visible renderer/editor behavior changes.
- Desktop screenshots do not count.
- Final quality review is required for this large sprint.

## Branch, Push, And Email Expectations

The user wants phased sprint execution after planning:

- main thread commits and pushes after each validated phase;
- main thread sends HTML email after each validated phase;
- each worker drafts `reports/phase-XX-email.md` for that phase;
- planning artifacts only define these expectations and do not perform email/wait mechanics.

## Closeout Gates

- All phase validation reports present.
- Final quality review passes or records explicit remediation.
- Evidence summary status is internally consistent.
- Docs state app crates primary, scripts experimental, hot Rust reload deferred/tooling-only.
- Any app-template claim is backed by generated app build evidence.
- Any script asset/event claim is backed by tests.
- Capture status is `not_applicable` unless visible behavior changed.
- Protected `.idea/engine.iml` and `.reasonix/` remain untouched.
- Changelog and sprint tracker updates are main-thread closeout responsibilities after user confirmation.
