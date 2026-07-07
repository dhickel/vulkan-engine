# Phase 01 Worker Directive: Residual Inventory And Triage Lock

## Objective

Build the verified Sprint 12 residual inventory and classify what must be fixed now, what can be accepted with mitigation, and what requires a dedicated follow-up sprint.

## User-Visible Outcome

The sprint no longer relies on vague bug debt. It has a concrete quality ledger that names alpha blockers, high-risk issues, stale residuals, and accepted non-critical debt before implementation begins.

## Editable Targets

- `reports/phase-01-residual-inventory.md`
- `artifacts/validation-summary.json`
- Existing or new `.internal-dev/bugs/<bug-id>/report.md` only for confirmed out-of-scope bugs discovered during inventory.
- Optional: `shared/validation-matrix.md` only if the inventory proves a required validation command is missing or impossible.

Read-only targets:

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `src/renderer/src/vulkan/AGENTS.md`
- roadmap and prior sprint plan/evidence files as targeted needed.
- source/docs/tests needed to classify residuals.

## Forbidden Scope

- Do not implement product code.
- Do not edit docs, source, tests, `SPRINT-TRACKER.md`, `.idea/engine.iml`, or `.reasonix/`.
- Do not broad-scan huge ignored/build directories.
- Do not close archived bugs without verifying current source/runtime evidence.

## Supporting Docs To Read

- `00-specification-lock.md`
- `01-current-state-analysis.md`
- `02-target-design.md`
- `shared/senior-engineer-guidance.md`
- `shared/implementation-notes.md`
- `.internal-dev/bugs/renderer-double-free-on-shutdown/.archive/report.md`
- Prior sprint validation summaries only as needed, especially Sprint 09 current state and any final residuals from Sprints 02-08.

## Senior Engineer Guidance

- Fact: many scan hits are test-only. Classify by runtime reachability before assigning work.
- Fact: docs may still mention stale Vulkan residuals. Compare to current code before treating a doc warning as a defect.
- Fact: archived bug reports are historical evidence, not closed truth. Re-open only when current reproduction or source risk supports it.
- Logic: this phase prevents later workers from solving the wrong problem.
- Likely failure mode: overfitting to `rg` output and creating a phase for every `unwrap`.

## Implementation Steps

1. Verify local state and record protected dirty paths in the report.
2. Read the required plan and repo guidance.
3. List existing bug records and prior sprint residuals that are relevant to alpha stability.
4. Run targeted scans:

```sh
rg -n "TODO|FIXME|todo!\\(|unimplemented!\\(|panic!\\(|unwrap\\(|expect\\(" src/renderer/src src/renderer/examples src/runtime.rs src/launch.rs apps docs/api docs/internal
rg -n "destroy path|double free|swapchain|old image view|VkSubAllocator|VkHostBuffer|fence\\[0\\]" src/renderer/src/vulkan src/renderer/src/data docs/internal .internal-dev/bugs
rg -n "public API|pub use|pub mod|advanced-interop|prelude|SceneWorld|CommandHistory|AnimationPlayer" docs/api src/renderer/src
```

5. Group findings into the classification table from `02-target-design.md`.
6. For each critical/high finding, include source path, evidence, expected phase, and validation needed.
7. Identify stale residuals that should be corrected in docs/evidence rather than code.
8. Identify stop-rule issues that require user decision or a dedicated follow-up sprint.
9. Write `reports/phase-01-residual-inventory.md`.
10. Update `artifacts/validation-summary.json` to `phase_01_implementation_complete_validation_pending`.

## Acceptance Criteria

- Report includes a residual inventory table with class, evidence, target phase, and action.
- Critical residuals are not vague; each has a reproduction/source reference and a proposed disposition.
- Stale residuals are separated from live defects.
- Phase 02-05 scope is either confirmed or explicitly adjusted.
- Evidence index is updated conservatively.

## Negative Checks

- No product code changes.
- No tracker updates.
- No protected path changes.
- No `fully_validated` status.

## Validation Commands

Validators should rerun or inspect the targeted scans above. Worker should also run:

```sh
cargo check -p renderer
cargo check -p renderer --examples
```

The compile checks are orientation checks, not a full sprint pass.

## Stop Conditions

- Stop if inventory shows more than two critical defect families that each need broad remediation; return to planning for sprint split.
- Stop if a critical defect requires public API breakage or broad renderer architecture change.
- Stop if current worktree state makes it impossible to distinguish Sprint 09 active changes from Sprint 12 target source.

## Evidence Expectations

- Worker report: `reports/phase-01-residual-inventory.md`
- Validator report: `validation/phase-01-validation-report.md`
- Evidence index updated with findings and residuals.

## Do Not Close Unless

- Every roadmap priority class has at least one explicit disposition.
- Phase 02-05 work is scoped from verified current evidence.
- Protected local state is untouched.
