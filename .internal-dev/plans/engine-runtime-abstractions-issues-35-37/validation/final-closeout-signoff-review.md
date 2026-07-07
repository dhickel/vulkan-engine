# Final Closeout Signoff Review

Status: passed after evidence remediation
Date: 2026-07-07
Reviewer role: x-high senior engineer closeout validation

## Findings

1. Resolved evidence defect: the canonical headless capture sidecar originally did not satisfy the validation matrix metadata contract.
   - Matrix requirement: `.internal-dev/plans/engine-runtime-abstractions-issues-35-37/shared/validation-matrix.md:73-77` requires the PNG/JSON sidecar to identify target, frame, source, format, extent, requested camera/view, applied camera/view path, residual risks, and to distinguish the caller-provided view path from the legacy renderer-owned camera path.
   - Remediated sidecar: `.internal-dev/captures/engine-runtime-abstractions-issues-35-37/phase-05-dogfood/dungeon-dogfood-frame-0-draw-seq-0000.json` now records requested app-owned `CameraView` construction, submitted `render_scene_headless_with_view` path, unused legacy renderer camera path, metadata reconciliation, and residual risks.
   - Impact: `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json` may now mark the suite `fully_validated` without conflicting with the written matrix.

## Non-Blocking Observations

- Implementation shape is sound: the root `engine` facade remains lightweight, raw crates remain directly usable, and `CameraView` stays in the renderer facade boundary.
- Legacy renderer-owned input/events/camera APIs are preserved and active docs/specs label them as compatibility rather than the only intended path.
- `dungeon_dogfood` owns app input, app events, frame clock, and camera/controller state before submitting a caller-provided `CameraView`.
- Swapchain acquire retry warnings and known dead-code warning noise are documented as residual risks instead of hidden.

## Pass/Fail By Criterion

| Criterion | Result | Evidence |
| --- | --- | --- |
| No overclaim in validation-summary.json or Phase 06 report. | PASS | The capture metadata gap is now disclosed and remediated; `fully_validated` is consistent with the evidence contract. |
| Active docs/specs do not present renderer-owned input/events/camera as the only intended path. | PASS | Docs/specs distinguish renderer-owned compatibility and app-owned facade paths. |
| Legacy APIs are preserved/labeled compatibility. | PASS | Renderer APIs remain present and docs/specs label compatibility behavior. |
| Root engine facade remains lightweight and raw primitive access remains available. | PASS | Root facade modules re-export or lightly compose support-crate primitives; tests prove raw crate imports. |
| Dogfood migration and validation evidence are sufficient to sign off for #35-#37. | PASS | Product-code, report evidence, runtime smokes, and enriched canonical capture sidecar now satisfy the plan's required evidence contract. |
| Residual risks are appropriately documented. | PASS | Swapchain acquire retry warnings and known warning noise appear in validation summary, changelog, and phase reports. |

## Checks Run

- `cargo fmt --check`: passed.
- `cargo test -p engine --test facade_imports --test input_action_events --test runtime_event_dispatcher --quiet`: passed; known renderer warning noise printed.
- `cargo test -p renderer --test integration --quiet`: passed, 21 tests; known renderer warning noise printed.
- `cargo check -p dungeon_dogfood --quiet`: passed; known renderer and dogfood warning noise printed.
- `jq` consistency check for final status/residuals in `validation-summary.json`: passed.
- Stale-reference grep over `docs .internal-dev src apps`: hits are implementation terms, compatibility APIs, unrelated tests, examples, or historical artifacts; no active docs/specs present renderer-owned runtime as the only intended path.
- Capture artifact inspection: PNG and JSON exist; JSON includes requested/app-owned camera view, applied caller-view render path, unused legacy camera path, metadata reconciliation, and residual risks.
- Validator self-remediation: corrected one evidence string in the capture sidecar from nonexistent `Renderer::render_scene_with_view_using` to existing `Renderer::render_scene_internal_with_view`.

## Remediation Handoff

Failure class: docs_or_evidence_defect, remediated.

Completed repair:

1. Canonical capture evidence now records:
   - requested app camera/view inputs or an explicit app-owned `CameraView` descriptor;
   - applied path as `render_scene_headless_with_view` / caller-provided `CameraView`;
   - legacy renderer-owned camera path as not used for this capture;
   - residual risks, including swapchain acquire retry warnings being a windowed-smoke residual and not observed as a fatal headless capture failure.
2. `artifacts/engine-runtime-abstractions-issues-35-37/validation-summary.json` was updated after the sidecar metadata was added.
3. Focused evidence validation re-inspected the updated JSON, verified PNG presence, and reran jq/status consistency checks.

No product-code remediation is required by this review unless the team later chooses to make the renderer capture writer emit camera/view metadata natively instead of using a task-scoped evidence sidecar.

## Sign-Off

PASS. Senior-engineer signoff is granted for the current scope of GitHub issues #35-#37.

The implementation keeps the root `engine` facade appropriately thin, preserves raw primitive access and renderer-owned compatibility APIs, proves the app-owned runtime path through dogfood, and now has reconciled canonical evidence for the caller-provided camera/view capture path. Residual swapchain acquire retry warnings and known warning noise are documented as accepted follow-up risk, not hidden validation failures.
