# Final Orchestration Plan

Date: 2026-07-07
Status: execution guidance for main-thread orchestrator

## Dispatch Sequence

Run phases sequentially. Validate each phase before dispatching the next.

1. Phase 00: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/worker-directives/phase-00-preflight-drift.md`
   - Validation report: `validation/phase-00-validation-report.md`
2. Phase 01: `worker-directives/phase-01-root-facade.md`
   - Validation report: `validation/phase-01-validation-report.md`
3. Phase 02: `worker-directives/phase-02-renderer-view-path.md`
   - Validation report: `validation/phase-02-validation-report.md`
4. Phase 03: `worker-directives/phase-03-app-owned-input.md`
   - Validation report: `validation/phase-03-validation-report.md`
5. Phase 04: `worker-directives/phase-04-app-owned-events.md`
   - Validation report: `validation/phase-04-validation-report.md`
6. Phase 05: `worker-directives/phase-05-dogfood-migration.md`
   - Validation report: `validation/phase-05-validation-report.md`
7. Phase 06: `worker-directives/phase-06-compat-docs-closeout.md`
   - Validation report: `validation/phase-06-validation-report.md`

## Validation Routing

- Use phase validation/red-team agent after each worker.
- Use final large-suite quality validator after Phase 06 passes.
- No browser/Playwright validation applies.
- Use headless capture validation only if implementation claims visual/camera output proof or materially changes visible camera behavior.

## Remediation Routing

- `code_defect`: fresh scoped repair worker for the failed phase.
- `docs_or_evidence_defect`: fresh scoped repair worker unless validator-safe one-place edit.
- `plan_defect`: return to advanced planning and revise affected directives before more coding.
- `validator_error`: fix checklist or use fresh validator.
- Same targeted issue failing twice: escalate to fresh scoped high-reasoning repair with the escalation model.

## Stop Gates

Stop orchestration and ask the user/main thread before continuing if:

- root bin+lib is blocked by a real dependency cycle and a new crate is needed;
- dogfood audio drift requires broad redesign;
- preserving legacy APIs conflicts with new path correctness;
- a support crate, renderer, `launch_shared`, tool, or pack tooling would need to depend on root `engine`;
- validation tool/model fallback is required.

## Commit/Closeout Gates

If commit policy applies:

- Commit after each validated phase or after logical validated bundles if main-thread policy prefers fewer commits.
- Include implementation and `.internal-dev` artifacts for that phase together.
- Do not archive this plan until final quality review and user/main-thread closeout are complete.

Closeout must include:

- updated specs/docs/knowledge/changelog;
- final evidence index at `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json`;
- final quality review at `validation/final-quality-review.md`;
- GitHub issue closeout handled by main thread if requested/approved.

## Final Quality Review Criteria

Final validator passes only if:

- all phase reports pass or have approved residuals;
- final command suite and required runtime smokes pass or have approved environmental constraints;
- dogfood active path no longer uses renderer-owned input/event/camera lifecycle APIs;
- specs/docs do not leave stale renderer-owned lifecycle as the only intended contract;
- raw primitive support remains documented and functional;
- crate graph evidence distinguishes allowed app/example -> root `engine` facade edges from forbidden lower/support crate -> root `engine` reverse edges;
- evidence index status is conservative and consistent.
