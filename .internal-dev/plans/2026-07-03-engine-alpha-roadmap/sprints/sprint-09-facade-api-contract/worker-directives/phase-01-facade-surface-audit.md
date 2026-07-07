# Phase 01 Worker Directive: Facade Surface Audit

## Objective

Audit and classify the current public renderer API surface so Sprint 09 can distinguish supported beginner facade, compatibility exports, advanced interop, internals, and deferred gaps without breaking existing users.

## User-Visible Outcome

The docs and phase report clearly state what is alpha-supported for beginners and what remains public only for compatibility or advanced use. No accidental public export is presented as beginner-stable just because it is reachable.

## Editable Targets

- `docs/api/00-index.md`
- Optional small supporting doc under `docs/api/` if needed, such as a concise facade contract chapter.
- `src/renderer/src/lib.rs` only for comments/rustdoc/module organization that does not remove existing exports.
- `src/renderer/src/api/mod.rs` only for comments/rustdoc/module organization that does not remove existing exports.
- `src/renderer/tests/integration.rs` only for compile assertions or import classification tests that do not depend on a GPU.
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/reports/phase-01-export-audit.md`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/artifacts/validation-summary.json`

## Forbidden Scope

- Do not remove public exports.
- Do not create a broad prelude in this phase.
- Do not implement new renderer, project runtime, material, camera, or input behavior.
- Do not touch `.idea/engine.iml` or `.reasonix/`.
- Do not run or require headless capture unless this phase unexpectedly changes visible runtime behavior.

## Supporting Docs To Read

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- Plan files: `00-specification-lock.md`, `01-current-state-analysis.md`, `02-target-design.md`, `shared/senior-engineer-guidance.md`
- `src/renderer/src/lib.rs`
- `src/renderer/src/api/mod.rs`
- `docs/api/00-index.md`
- `src/renderer/tests/integration.rs`

## Senior Engineer Guidance

- Current tests import legacy root exports; this is compatibility evidence, not a reason to break tests.
- The immediate bug is doc/API contract ambiguity. Fix the promise before changing the machinery.
- If a type is public but too low-level for beginner docs, classify it as compatibility or advanced-adjacent.
- Keep the audit concrete: table of symbol groups, source path, current exposure, intended classification, and action.
- Avoid semver language stronger than alpha support. This sprint locks community-alpha intent, not long-term semver.

## Implementation Steps

1. Enumerate current re-exports from `src/renderer/src/api/mod.rs` and extra root exports from `src/renderer/src/lib.rs`.
2. Classify each group, not necessarily every individual symbol, into:
   - alpha beginner facade;
   - compatibility public;
   - advanced interop;
   - internal but exposed through legacy path;
   - deferred.
3. Update `docs/api/00-index.md` so it no longer claims every public symbol is beginner-stable.
4. If useful, add a short facade contract table in `docs/api/00-index.md` or one new small doc linked from the index.
5. Add narrow compile-oriented tests only if they clarify the contract without adding GPU/runtime dependencies.
6. Write `reports/phase-01-export-audit.md` with the audit table, decisions, validation run, and residuals.
7. Update `artifacts/validation-summary.json` phase 01 status to an honest intermediate state such as `implementation_complete_validation_pending`.

## Acceptance Criteria

- The API index distinguishes supported alpha beginner surface from compatibility/advanced exports.
- The audit report names the root export mismatch and the chosen classification.
- Current non-`api` root exports are preserved unless a user-approved exception exists.
- The worker report includes any docs/code divergences discovered.

## Negative Checks

- No public export removal.
- No broad docs rewrite.
- No new runtime behavior.
- No final validation claims in `validation-summary.json`.

## Validation Commands

Run and record results:

```sh
cargo fmt --check
cargo check -p renderer
rg -n "stable public surface|Everything below api|advanced-interop|AnimationPlayer|SceneWorld|CommandHistory" docs/api src/renderer/src src/renderer/tests
```

Run if rustdoc or re-export docs changed materially:

```sh
cargo doc -p renderer --no-deps
```

## Stop Conditions

- Stop if classification would require removing or renaming public exports.
- Stop if docs cannot be made truthful without implementing new API behavior.
- Stop if validation failures suggest pre-existing broad doc/test issues outside this phase; record and hand back.

## Evidence Expectations

- Worker report: `reports/phase-01-export-audit.md`
- Validator report path: `validation/phase-01-validation-report.md`
- Update `artifacts/validation-summary.json` without claiming final pass.

## Do Not Close Unless

- The audit report exists.
- The API docs no longer overpromise the beginner surface.
- Validation commands and results are recorded.
- Protected local state is untouched.
