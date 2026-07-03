# Specification Lock

## Classification

Large. Sprint 02 introduces a new workspace CLI, shared validation contracts, fixtures/tests, docs, and sprint evidence gates across renderer/editor/package surfaces.

## Locked Objective

Build an execution-ready Rust CLI path for alpha package authoring and validation while preserving the renderer/editor durable identity contracts.

## Acceptance Criteria

- `tools/engine_pack` is a Rust CLI workspace member.
- `validate-package`, `validate-project`, and `validate-scene` are implemented and covered by valid and invalid fixtures.
- `new-project`, `new-package`, `scan-assets`, `add-asset`, and `pack` are implemented unless a phase records a concrete blocker and narrows scope with validator approval.
- `apps/editor/sample_project/engine.project.toml` validates through the CLI.
- CLI and editor/runtime validation share Rust logic or have explicit compatibility tests that fail on disagreement.
- Errors are deterministic enough for tests: stable prefix/diagnostic code or stable message text for missing version, unsupported version, duplicate IDs, invalid path identity, missing package manifest, missing asset path, unknown asset ID, invalid scene graph, and runtime handle serialization attempts.
- Every phase has a validation report, commit/push gate, AgentMail HTML report gate, and changed files/line counts/git links matrix requirement.

## Validation Criteria

- Phase validation reports:
  - `validation/phase-01-validation-report.md`
  - `validation/phase-02-validation-report.md`
  - `validation/phase-03-validation-report.md`
  - `validation/phase-04-validation-report.md`
- Final evidence:
  - `validation/final-quality-review.md`
  - `artifacts/validation-summary.json`
- Required checks by the end of Sprint 02:
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p input`
  - `cargo check -p engine_pack`
  - `cargo test -p engine_pack`
  - focused renderer/shared-validator tests added by workers
  - CLI validation against `apps/editor/sample_project/engine.project.toml`
- Capture decision:
  - Record `not_required_cli_schema_only` unless visual/rendering behavior changes or visual readiness is claimed.
  - If capture is required, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` and store outputs under `.internal-dev/captures/` or `.internal-dev/headless_capture_tests/`.

## Negative Criteria

- Do not use Python as the canonical package validator/tool.
- Do not serialize runtime handles as durable project/package/scene identity.
- Do not commit `.idea/engine.iml` or `.reasonix/`.
- Do not resolve Sprint 01 blocked changelog confirmation.
- Do not mark `fully_validated` while residuals, missing pushes, missing email reports, failed validators, skipped required checks, or unmade capture decisions remain.

## Constraints And Assumptions

- Branch is `sprint/alpha-02-packaging-tools`.
- Push after every phase to `origin sprint/alpha-02-packaging-tools`.
- Existing dirty state `.idea/engine.iml` and `.reasonix/` must be preserved and excluded from commits.
- Current package validation is in `src/renderer/src/data/asset_registry.rs`.
- Current scene persistence validation is in private `SerializedScene` implementation inside `src/renderer/src/api/scene.rs`.
- Project validation is currently weaker than package and scene validation; Sprint 02 should strengthen it in Rust.
- Sprint 01 is still blocked on changelog timing and should be noted, not closed.

## Stop Rules

- Stop if branch is not `sprint/alpha-02-packaging-tools` or push target cannot be verified.
- Stop if a required command needs a product/schema decision not covered by this spec.
- Stop if implementing a CLI command would require broad renderer/editor architecture changes outside packaging validation.
- Stop if durable identity would need to be replaced by runtime handles.
- Stop after two failed remediation cycles for the same targeted issue and escalate to a fresh repair worker.
