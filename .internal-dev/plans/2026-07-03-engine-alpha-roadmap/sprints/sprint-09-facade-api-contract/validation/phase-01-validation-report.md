# Phase 01 Validation Report: Facade Surface Audit

## Verdict

Pass. Sprint 09 Phase 01 may proceed to the next phase.

Validator: Codex validation agent, no explicit user model override provided for this turn.

## Findings

No blocking findings.

## Criterion Results

| Criterion | Result | Evidence |
|-----------|--------|----------|
| API index distinguishes supported alpha beginner surface from compatibility/advanced exports | Pass | `docs/api/00-index.md:31` defines a tiered public API contract; `docs/api/00-index.md:65` classifies alpha beginner facade, compatibility public, advanced interop, internal implementation detail, and deferred gaps. |
| Audit report names the root export mismatch and chosen classification | Pass | `reports/phase-01-export-audit.md` includes the root export mismatch and classifies `AnimationPlayer`, camera helpers, command history, and `SceneWorld` as compatibility public rather than beginner facade. |
| Current non-`api` root exports are preserved | Pass | `git diff -- src/renderer/src/lib.rs src/renderer/src/api/mod.rs src/renderer/tests/integration.rs` produced no diff. Current source still exports `AnimationPlayer`, camera helpers, command history, and `SceneWorld` from `src/renderer/src/lib.rs:41`. |
| Worker report includes docs/code divergences discovered | Pass | Worker report records the previous inaccurate claim that the full re-export list lived in `api/mod.rs` and that all symbols below `api::*` in `lib.rs` were stable public surface. |
| Negative check: no public export removal | Pass | No source diff in `src/renderer/src/lib.rs` or `src/renderer/src/api/mod.rs`; `cargo check -p renderer` passed. |
| Negative check: no broad docs rewrite | Pass | Diff is limited to the API index contract section plus plan evidence artifacts. |
| Negative check: no new runtime behavior | Pass | No Rust source changes in renderer runtime/facade files. |
| Negative check: no final validation claims before validation | Pass | Worker left `phase_01.status` as `implementation_complete_validation_pending`; this validator updates it to `validated` only after passing checks. |
| Protected local state preserved | Pass | `.idea/engine.iml` remains modified and `.reasonix/` remains untracked; neither was touched by validation. |

## Commands And Evidence

| Command | Result | Notes |
|---------|--------|-------|
| `git status --short` | Inspected | Confirmed protected unrelated local state: `.idea/engine.iml` modified and `.reasonix/` untracked. |
| `git diff -- docs/api/00-index.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/reports/phase-01-export-audit.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-09-facade-api-contract/artifacts/validation-summary.json` | Inspected | Product-facing change is the API index contract section; summary records pending validation and command evidence. |
| `git diff -- src/renderer/src/lib.rs src/renderer/src/api/mod.rs src/renderer/tests/integration.rs` | Pass | No diff, confirming no export or compile-contract source edits in the inspected facade files. |
| `git diff --check` | Pass | No whitespace errors. |
| `cargo fmt --check` | Pass | Formatting unchanged. |
| `cargo check -p renderer` | Pass | Passed with existing renderer dead-code warnings. |
| `rg -n "stable public surface|Everything below api|advanced-interop|AnimationPlayer|SceneWorld|CommandHistory" docs/api src/renderer/src src/renderer/tests` | Pass for phase intent | No stale overpromise phrases remain. Remaining hits are expected compatibility-symbol and `advanced-interop` references in docs/code/tests. |

`cargo doc -p renderer --no-deps` was not run. The phase changed Markdown docs and plan evidence only; rustdoc and re-export organization were not materially changed.

Headless capture was not run. This phase is non-visual docs/API contract work and introduced no visible renderer behavior change.

## Evidence Reconciliation

The canonical validation summary matched observed worker evidence before validator closeout: `phase_01.status` was `implementation_complete_validation_pending`, command evidence was recorded, and capture was marked not applicable. No conflicting browser, capture, or runtime evidence exists for this phase.

After this report, `artifacts/validation-summary.json` should update only `phase_reports.phase_01.status` to `validated`.

## Residual Risk

The root compatibility surface remains broad by design. Later phases still need to keep examples, docs, and any supported import path aligned with the tiered contract. Existing renderer dead-code warnings remain out of scope for this phase.
