# Final Orchestration Plan

## Dispatch Order

1. Phase 01 worker: `worker-directives/phase-01-current-dogfood-project-audit.md`
2. Phase 01 validator writes `validation/phase-01-validation-report.md`
3. Phase 02 worker: `worker-directives/phase-02-packaged-content-sample-scene.md`
4. Phase 02 validator writes `validation/phase-02-validation-report.md`
5. Phase 03 worker: `worker-directives/phase-03-runtime-gameplay-input-camera.md`
6. Phase 03 validator writes `validation/phase-03-validation-report.md`
7. Phase 04 worker: `worker-directives/phase-04-true-headless-visual-baseline.md`
8. Phase 04 validator reconciles capture evidence and writes `validation/phase-04-validation-report.md`
9. Phase 05 worker: `worker-directives/phase-05-docs-final-validation-prep.md`
10. Phase 05 validator writes `validation/phase-05-validation-report.md`
11. Final quality validator writes `validation/final-quality-review.md`

## Model Guidance

- Implementation workers: default `gpt-5.3` high reasoning unless the user overrides.
- Phase validators: default `gpt-5.5` high reasoning unless the user overrides.
- Final quality validator: default `gpt-5.5` xhigh reasoning unless the user overrides.
- Visual proof agent: engine headless capture validation, default `gpt-5.5` high reasoning; this is not browser/Playwright work.
- If any requested model/tool is unavailable, record `TOOLING_CONSTRAINT` in `artifacts/validation-summary.json` and stop for user approval before fallback.

## Phase Gates

Each mutating phase must pass validation before dependent work proceeds.

Phase 01 gate:

- Reports exist.
- Contract/migration decision is clear.
- Sprint 09/Sprint 10 hazards are recorded.

Phase 02 gate:

- Dogfood package/project/scene validates or blocker is explicit.
- Migration debt report updated.

Phase 03 gate:

- Dogfood run path works or blocker is explicit.
- Input/camera/gameplay loop covered by tests or runtime evidence.

Phase 04 gate:

- True `--headless --capture_target draw` evidence exists and sidecar/PNG are inspected.
- No desktop screenshot evidence is used.

Phase 05 gate:

- Docs match commands and limitations.
- Evidence summary is consistent.
- Final report draft exists.

## Remediation Routing

- `code_defect`: fresh scoped repair worker on the failed phase target.
- `docs_or_evidence_defect`: fresh scoped repair worker unless it is a trivial one-place report typo.
- `capture_harness_defect`: repair capture harness/evidence first; change product code only if evidence proves a product bug.
- `plan_defect`: return to planning for revised criteria/directives.
- `validator_error`: correct checklist or use fresh validator.
- Same targeted issue failing twice: escalate to fresh `gpt-5.5` high-reasoning repair worker.

## Required Final Commands

Run or reconcile phase evidence for:

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo check -p engine_pack
cargo test -p engine_pack
cargo check -p dungeon_dogfood
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

Required runtime/capture evidence:

- dogfood windowed/generated smoke;
- dogfood authored level/ramp smoke;
- debug timing JSONL under `.internal-dev/debug_reports/sprint-11-dogfood-vertical-slice/`;
- draw-target capture evidence under `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`.

## Final Quality Review Checklist

- Compare code/docs against `00-specification-lock.md`.
- Inspect every phase validation report.
- Inspect `reports/api-friction.md`, `reports/migration-debt.md`, and `reports/final-report.md`.
- Inspect `artifacts/validation-summary.json` for internal consistency.
- Inspect capture sidecars and PNGs.
- Confirm no tracker update was made by workers.
- Confirm no `.idea/engine.iml` or `.reasonix/` edits were made.
- Confirm out-of-scope bugs are filed.
- Confirm changelog timing remains a main-thread/user gate.

## Closeout Gates

The main thread owns:

- updating `SPRINT-TRACKER.md` after review;
- asking user whether to create changelog;
- writing changelog after confirmation;
- committing/pushing/PR if requested;
- sending final report email through AgentMail if requested.

Do not mark the sprint closed until these gates are reconciled.
