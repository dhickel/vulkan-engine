# Phase 04 Worker Directive: Docs, Examples, Public-Contract Drift, And Test Gaps

## Objective

Align docs, examples, public-contract language, and focused tests with current alpha behavior after the quality fixes from Phases 02-03.

## User-Visible Outcome

Alpha users and future agents see docs/examples that match the engine's actual supported path, known residuals, and validation expectations. Stale historical warnings are corrected or tracked.

## Editable Targets

Docs:

- `docs/api/`
- `docs/internal/`
- `README.md`
- `AGENTS.md` and module `AGENTS.md` only if validation/process guidance materially changed.

Examples/tests:

- `src/renderer/examples/`
- `src/renderer/tests/`
- existing test modules in touched crates.

Artifacts:

- `reports/phase-04-docs-examples-test-drift.md`
- `artifacts/validation-summary.json`
- `.internal-dev/bugs/<bug-id>/report.md` only for newly confirmed out-of-scope defects.

## Forbidden Scope

- Do not rewrite all docs.
- Do not implement new feature behavior just because docs mention it.
- Do not change public exports unless Phase 01 and the user decision gate approved it.
- Do not update `SPRINT-TRACKER.md`.
- Do not edit `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- Phase 01-03 reports and validation reports.
- Sprint 09 plan/evidence relevant to public facade classification.
- `docs/api/00-index.md`
- `docs/internal/00-index.md`
- docs called out by stale scans.

## Senior Engineer Guidance

- Direct target: truthfulness and validation coverage, not prose polish.
- Approach: update docs after code behavior is known.
- Gotcha: legacy duplicate docs may still exist; route readers to current alpha docs rather than deleting broad history.
- Gotcha: scans for `TODO` and `planned` will find legitimate residuals; classify rather than blindly removing.
- Best practice: make examples compile and match the documented supported APIs.
- Likely failure mode: turning the phase into a documentation rewrite and missing actual test gaps.

## Implementation Steps

1. Read prior phase reports and extract every docs/examples/test follow-up.
2. Run stale scans:

```sh
rg -n "pending|planned|not implemented|/tmp|desktop screenshot|gap-report|old image views|destroy paths|VkSubAllocator::destroy|fence\\[0\\]|double free" docs/api docs/internal README.md AGENTS.md src/renderer/AGENTS.md
rg -n "TODO|FIXME|todo!\\(|panic!\\(|unwrap\\(|expect\\(" src/renderer/examples src/renderer/tests docs/api docs/internal
```

3. Update docs to match code and prior sprint contracts.
4. Adjust examples only where they contradict the supported alpha path or contain high-risk user-facing panic behavior.
5. Add or adjust focused tests for scene/package/runtime flows that Phase 01 identified as gaps and Phases 02-03 did not cover.
6. Run compile/test/doc checks.
7. Write phase report and update evidence index.

## Acceptance Criteria

- Docs no longer claim known fixed lifecycle bugs remain open.
- Docs no longer imply unsupported alpha behavior is implemented.
- Examples compile and align with supported alpha usage.
- Test gaps identified for this sprint are filled or accepted with mitigation.
- Any remaining stale docs are listed as residuals with reason.

## Negative Checks

- No broad docs rewrite.
- No feature implementation hidden inside docs cleanup.
- No unapproved public API breakage.
- No protected path edits.

## Validation Commands

Required:

```sh
cargo fmt --check
cargo check -p renderer --examples
cargo test -p renderer
```

Run if public rustdoc or re-export docs changed:

```sh
cargo doc -p renderer --no-deps
```

Run if touched:

```sh
cargo test -p engine_pack
cargo test -p editor
cargo test -p dungeon_dogfood
```

Final stale scans:

```sh
rg -n "pending|planned|not implemented|/tmp|desktop screenshot|gap-report|old image views|destroy paths|VkSubAllocator::destroy|fence\\[0\\]|double free" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
```

## Stop Conditions

- Stop if docs cannot be made truthful without implementing a new feature.
- Stop if docs reveal a critical defect not in Phase 01 inventory; route through planning/remediation.
- Stop if `cargo test -p renderer` fails for broad doctest/prose issues that exceed this phase; record and ask whether to expand scope.

## Evidence Expectations

- Worker report: `reports/phase-04-docs-examples-test-drift.md`
- Validator report: `validation/phase-04-validation-report.md`
- Evidence index updated with docs/tests status and residuals.

## Do Not Close Unless

- Docs/examples touched by Phases 02-03 are reconciled.
- Stale residual claims are corrected or explicitly listed.
- Validation results are recorded.
