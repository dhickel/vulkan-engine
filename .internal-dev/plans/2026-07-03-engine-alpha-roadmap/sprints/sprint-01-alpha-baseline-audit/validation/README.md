# Validation Instructions

Validators must treat Sprint 01 as a process/docs baseline sprint. Product-code changes should fail validation unless the relevant phase directive explicitly allowed a tiny docs/process repair.

## Report Paths

- Phase 01: `validation/phase-01-validation-report.md`
- Phase 02: `validation/phase-02-validation-report.md`
- Phase 03: `validation/phase-03-validation-report.md`
- Phase 04: `validation/phase-04-validation-report.md`
- Final quality review: `validation/final-quality-review.md`
- Evidence index: `artifacts/validation-summary.json`

## Validator Checklist

- Read `00-specification-lock.md`, `shared/validation-matrix.md`, and the relevant phase directive.
- Check branch, dirty state, and staged files.
- Verify phase outputs against acceptance and negative criteria.
- Inspect command evidence, not only worker claims.
- Confirm pushed commit/ref and GitHub links when available.
- Confirm post-phase AgentMail HTML report was sent and includes required content.
- Confirm validation-summary status does not overclaim.
- Write the phase validation report before the phase is considered complete.

## Stale-Reference Sweep

Before final quality pass, run targeted searches over current docs and sprint artifacts:

```bash
rg -n "gap-report|known limitations|not implemented|pending|planned|TODO|/tmp|agent id|stale|fully_validated" AGENTS.md README.md docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit
```

Treat intentional historical references as pass only when clearly labeled historical or superseded.

## Capture Validation

If no visual behavior changed, validators should record capture status as `not_required_docs_process_only`.

If visual proof is required, validation cannot pass until a headless capture command produces inspectable PNG/JSON evidence and the validator reconciles it with explicit visual criteria.
