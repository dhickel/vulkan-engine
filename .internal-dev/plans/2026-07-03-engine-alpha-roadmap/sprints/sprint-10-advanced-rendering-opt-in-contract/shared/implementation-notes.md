# Implementation Notes

## Branch And Worktree

- Intended execution branch: `sprint/alpha-10-advanced-rendering-opt-in-contract`.
- Start after the main thread reconciles Sprint 09. Do not treat the planning-time dirty worktree as Sprint 10-owned.
- Do not edit `.idea/engine.iml`, `.reasonix/`, or `SPRINT-TRACKER.md`.

## Suggested Phase Flow

1. Audit current exports/docs/examples and write the Sprint 10 API audit report.
2. Harden feature gates and documentation for safe vs advanced extension points.
3. Add a minimal named advanced surface only if the audit shows it is safe and useful; otherwise record a deliberate defer.
4. Run final validation, stale-reference sweep, and evidence reconciliation.

## Feature-Gate Checks

Run both:

```sh
cargo check -p renderer
cargo check -p renderer --features advanced-interop
```

Also run examples in both modes:

```sh
cargo check -p renderer --examples
cargo check -p renderer --examples --features advanced-interop
```

## Runtime/Capture Rules

Runtime smoke for API/runtime behavior changes:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-10-api-test-timing.jsonl
```

Headless capture only when visible renderer behavior or capture/readback behavior changes:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_dir .internal-dev/captures/sprint-10-advanced-rendering-opt-in-contract/headless-draw
```

Capture evidence must include PNG and sidecar JSON with successful draw-target status.

## Documentation Consistency

Use one canonical extension chapter. If keeping both `05-render-hooks-and-extension-points.md` and `05-hooks.md`, make one a concise redirect/compatibility page or keep both semantically identical. Do not leave one page claiming command-buffer access through safe hooks.

## Evidence Hygiene

- Keep phase reports under `validation/`.
- Keep audit/final reports under `reports/`.
- Keep canonical status in `artifacts/validation-summary.json`.
- Use conservative statuses such as `planned`, `phase_validation_pending`, `runtime_smoke_pending`, `final_quality_pending`, or `final_quality_review_passed_with_residuals`.
