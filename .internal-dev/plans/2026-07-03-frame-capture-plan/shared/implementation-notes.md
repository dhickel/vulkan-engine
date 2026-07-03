# Shared Implementation Notes

## Repository Governance

- Preserve unrelated local changes.
- Use `.internal-dev/debug_reports/` for generated debug output.
- Use `.internal-dev/plans/2026-07-03-frame-capture-plan/validation/` for phase reports.
- Keep email/remote coordination out of worker directives and implementation notes.
- Ask before logging out-of-scope future considerations in `.internal-dev/notes/`.

## Suggested Types

Use names that fit local style, but preserve these concepts:

- `CaptureTarget`: `Present`, `Draw`.
- `FrameCaptureRequest`: one concrete output path.
- `FrameCaptureSequence`: count/start/interval/output-dir.
- `FrameCaptureScheduler`: resolves due captures for a frame.
- `FrameCaptureReport`: success metadata.
- `FrameCaptureError`: structured failure.
- `CaptureOutputNaming`: shared helper for deterministic paths.

## CLI Parsing Rules

- Keep both `--flag value` and `--flag=value` conventions where existing parser supports both.
- Reject zero values for counts and intervals.
- Reject conflicting single and sequence output options where ambiguity would cause lost captures.
- Preserve existing `--env`, `--model`, and `--record_debug*` semantics.
- Treat unknown args consistently with current examples unless the worker deliberately tightens parser behavior and updates docs/tests.

## Capture Scheduling Rules

- Frame numbers use the public renderer frame counter observed by examples.
- Single capture at frame N fires once.
- Sequence capture starts at `start_frame` and captures every `interval` frames until exactly `count` images are written.
- Manual capture queues one capture for the next renderable frame.
- Capture should defer or fail clearly during resize/minimized/invalid extent instead of writing a stale image.

## Vulkan Readback Rules

- Prefer command recording in the active graphics command buffer, then consume readback after the frame fence signals.
- A bounded immediate-submit path is acceptable only if it restores layouts and does not race with in-flight frame commands.
- Ensure readback allocations are destroyed through the same allocator path that created them.
- Record source image current and restored layouts in the sidecar.
- Treat PNG writing errors as capture failures, not renderer panics.

## Evidence Index Skeleton

The final evidence index must exist at `.internal-dev/plans/2026-07-03-frame-capture-plan/artifacts/validation-summary.json`.

Minimum shape:

```json
{
  "plan_slug": "2026-07-03-frame-capture-plan",
  "status": "implementation_checks_passed",
  "model_tooling_constraints": [],
  "compile_checks": [],
  "capture_matrix": [],
  "headless_matrix": [],
  "manual_capture": {},
  "phase_validations": [],
  "final_quality_review": {},
  "artifact_roots": [
    ".internal-dev/debug_reports/",
    ".internal-dev/plans/2026-07-03-frame-capture-plan/validation/"
  ],
  "superseded_artifacts": [],
  "residual_risks": []
}
```

Do not set `status` to `fully_validated` unless every required phase validator, capture proof, and final quality review has passed and no unresolved blocking residual remains.

