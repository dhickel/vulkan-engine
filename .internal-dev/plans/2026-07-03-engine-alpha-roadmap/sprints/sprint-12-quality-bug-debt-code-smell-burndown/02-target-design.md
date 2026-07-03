# Target Design

## Sprint Shape

Sprint 12 is a burn-down sprint, not a feature sprint. The target design is an evidence-backed quality ledger plus focused repairs in the areas most likely to destabilize alpha.

The sprint succeeds when:

- critical defects are fixed or accepted with mitigation;
- non-critical code smells are not allowed to distract from alpha stability;
- validation is repeatable;
- residual status is conservative and machine-readable.

## Residual Classification Model

Every finding in Phase 01 must be classified:

| Class | Definition | Required Action |
| --- | --- | --- |
| Critical alpha blocker | Crash, data loss, invalid Vulkan lifetime, public contract contradiction, or validation blocker that makes alpha unsafe | Fix in Sprint 12 or stop for user decision/dedicated sprint |
| High alpha risk | Runtime panic/stall/error path likely to hit users, examples, editor, dogfood, or package/project workflows | Fix if scoped; otherwise accept only with mitigation and follow-up |
| Medium code smell | Maintainability issue with plausible runtime risk but no current repro | Fix only if local and low risk, otherwise track |
| Test-only/invariant | Panic/unwrap inside tests or intentionally unreachable invariant | Leave alone unless message or guard is misleading |
| Stale documentation | Docs contradict code or prior sprint status | Fix if alpha-facing; record if out of scope |
| Already resolved/stale residual | Historical note no longer matches code | Close or mark stale in report/evidence |

## Phase Boundaries

Phase 01 locks the inventory and decides what actually belongs to Sprint 12.

Phase 02 handles Vulkan lifecycle and shutdown. It is intentionally separated because cleanup ordering needs a different validation posture from docs or API cleanup.

Phase 03 handles runtime panics and stalls. It should convert high-risk failures to `Result` or clear bounded behavior without changing public API shape more than necessary.

Phase 04 handles docs/examples/test drift. It should align documentation with code and prior sprint contracts after implementation realities are known.

Phase 05 reconciles validation, residuals, and final evidence. It should not hide unresolved critical issues inside a closing report.

## Evidence Design

Canonical evidence index:

`artifacts/validation-summary.json`

The index must record:

- top-level conservative status;
- phase report paths and validator status;
- commands run and result summaries;
- runtime smoke evidence paths;
- capture artifact directory and status if applicable;
- residual risks with class, owner/follow-up path, and acceptance state;
- model/tooling constraints;
- superseded artifacts;
- final quality review result.

Valid status examples:

- `planning_locked`
- `phase_01_inventory_validated`
- `phase_02_validation_failed`
- `phase_03_repair_in_progress`
- `final_quality_pending`
- `final_quality_review_passed_with_residuals`
- `fully_validated`
- `blocked_tooling_constraint`

Use `fully_validated` only when every required validator passes, required captures pass, and no unresolved residual risks remain.

## Runtime Validation Design

Use renderer examples as smoke harnesses:

- `demo_pbr`
- `demo_unlit`
- `demo_model_load`
- `demo_async_loading`
- `api_test`
- `api_test --env src/renderer/src/assets/sky_maps/indoor_4k.exr`

Use debug-record flags for timing and stall evidence:

```sh
--record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/<name>.jsonl
```

Treat successful startup logs with no fatal errors before timeout as smoke pass only when the tested behavior does not require clean process exit. For shutdown/destroy fixes, the validator must also inspect the tail for allocator crashes, Vulkan validation/fatal errors, and double-free symptoms.

## Capture Design

If a phase changes visible rendering, use the project-local skill:

`.internal-dev/skills/engine-headless-capture-validation/SKILL.md`

Evidence path:

`.internal-dev/captures/sprint-12-quality-burndown/<scenario>/`

Capture is not required for pure docs, error propagation, or shutdown-only cleanup unless output correctness is affected.

## Public Contract Design

Sprint 12 should respect Sprint 09 facade outcomes:

- keep beginner facade small;
- classify or document compatibility exports rather than removing them;
- do not implement advanced rendering opt-in from Sprint 10;
- treat accidental public contracts as documentation/classification issues unless a safe, user-approved compatibility path exists.

## Dedicated Follow-Up Sprint Threshold

Create a stop condition instead of burying the issue if any finding requires:

- broad Vulkan resource ownership redesign;
- rendergraph scheduling rewrite;
- asset streaming architecture rewrite;
- public API breakage;
- dogfood project migration;
- new editor workflow;
- more than one phase's scope to fix safely.
