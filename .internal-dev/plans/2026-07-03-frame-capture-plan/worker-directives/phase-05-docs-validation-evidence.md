# Phase 05 Worker Directive: Docs, Validation Harness, And Evidence Index

## Objective

Finalize documentation, validation tooling, full runtime PNG proof, and the canonical evidence index for the frame capture suite.

## User-Visible Outcome

The repository documents how to use frame capture, and `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json` records conservative final evidence for compile checks, capture matrix, headless, manual, validators, and residual risks.

## Editable Targets

- `docs/api/07-engine-arguments.md`
- `docs/api/08-debug.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `src/renderer/AGENTS.md` only if runtime validation commands materially change
- validation helper scripts/artifacts under `.internal-dev/plans/2026-07-03-frame-capture-plan/` if useful
- `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-05-validation-report.md`

## Forbidden Scope

- Do not implement new renderer behavior beyond validation harness fixes.
- Do not claim desktop screenshots or timing JSONL satisfy image proof.
- Do not mark final evidence `fully_validated` until all required gates pass and validator reconciliation is complete.
- Do not create changelog/knowledge/notes closeout artifacts without asking the user at closeout, per repo instructions.

## Supporting Docs To Read

- all previous phase validation reports;
- `shared/validation-matrix.md`;
- `validation/README.md`;
- `00-specification-lock.md`;
- `docs/api/07-engine-arguments.md`;
- `docs/api/08-debug.md`;
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`.

## Senior Engineer Guidance

- Documentation must follow the actual implemented flag names, not the initial desired examples if they diverged.
- Evidence index status must be conservative. Missing matrix rows or pending validators are failures or non-final statuses.
- Prefer a small local image sanity helper if repeated manual `file`/metadata/pixel checks become error-prone.
- Keep generated capture outputs under `.internal-dev/debug_reports/` and reference them from the evidence index.

## Implementation Steps

1. Update user-facing docs for:
   - single capture;
   - N-frame capture;
   - headless/offscreen capture;
   - manual F12 capture;
   - default artifact paths;
   - image sanity expectations;
   - relationship to timing JSONL.
2. Update internal frame lifecycle docs for capture pass/order/layout behavior.
3. Add or finalize a validation helper for PNG metadata/nonblank checks if useful.
4. Run compile/check gates:
   - `cargo check`;
   - `cargo check -p renderer`;
   - `cargo check -p renderer --examples`;
   - `cargo check -p input`.
5. Run required windowed capture matrix for all canonical examples and custom environment path.
6. Run N-frame validation.
7. Run headless validation matrix or record approved gate/fallback status.
8. Run manual capture validation or direct API fallback proof.
9. Write `artifacts/validation-summary.json`.
10. Write phase validation report.

## Acceptance Criteria

- Docs accurately describe final behavior and commands.
- Every required matrix row has a PNG artifact and sanity check result.
- Evidence index exists and is internally consistent.
- Compile gates are recorded with pass/fail details.
- Residual risks are explicit and not hidden behind final status.

## Negative Checks

- No stale flag names in docs or plan artifacts.
- No `/tmp` evidence paths unless explicitly marked temporary/superseded.
- No pending/planned/TODO wording implying incomplete required work if status claims validation.
- No final status contradiction, such as `fully_validated` with a failed/missing row.

## Validation Commands

Required:

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`

Required runtime matrix: see `shared/validation-matrix.md`.

## Evidence Expectations

Write:

- `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`
- `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/phase-05-validation-report.md`

The validation report must include command outputs summarized, artifact paths, image sanity results, and any residual risk.

## Stop Conditions

- Stop if any required capture matrix row cannot produce a PNG after one focused repair attempt; route remediation through the orchestrator.
- Stop if evidence status would need to overstate certainty.
- Stop if docs reveal a plan/spec mismatch requiring planner revision.

## Do Not Close Unless

- validation summary JSON exists;
- all required compile and capture evidence is recorded or blockers are routed;
- stale-reference sweep is complete;
- phase validation report is written.

