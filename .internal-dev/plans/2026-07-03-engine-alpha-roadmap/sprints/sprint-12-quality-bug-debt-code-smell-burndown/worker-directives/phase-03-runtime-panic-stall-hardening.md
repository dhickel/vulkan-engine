# Phase 03 Worker Directive: Runtime Panic, Error, Frame, And Asset Stall Hardening

## Objective

Reduce alpha-facing runtime panic and stall risk by converting high-risk user/runtime paths to recoverable errors or bounded behavior, while leaving test-only/invariant panics alone unless they are misleading.

## User-Visible Outcome

Project/runtime/example failures produce actionable errors instead of surprise panics where feasible, and frame/asset stalls are measured, bounded, or explicitly tracked.

## Editable Targets

Only as confirmed by Phase 01:

- `src/runtime.rs`
- `src/launch.rs`
- `src/renderer/examples/`
- `src/renderer/src/api/`
- `src/renderer/src/data/`
- `src/renderer/src/vulkan/` only for error propagation adjacent to Phase 02 work.
- `apps/editor/` and `apps/dungeon_dogfood/` only for high-risk alpha runtime paths confirmed by inventory.
- focused tests in existing test modules.

Artifacts:

- `reports/phase-03-runtime-panic-stall-hardening.md`
- `artifacts/validation-summary.json`
- relevant `.internal-dev/bugs/<bug-id>/report.md`

## Forbidden Scope

- Do not mechanically replace every `unwrap`.
- Do not rewrite the app/runtime architecture.
- Do not implement Sprint 10 or Sprint 11.
- Do not remove public APIs.
- Do not edit `SPRINT-TRACKER.md`, `.idea/engine.iml`, or `.reasonix/`.

## Supporting Docs To Read

- Phase 01 report and validation report.
- Phase 02 report if Vulkan error handling changed.
- `docs/api/07-engine-arguments.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- relevant source for confirmed runtime panic/stall targets.

## Senior Engineer Guidance

- Direct target: non-test runtime paths users can hit.
- Approach: prefer existing error types and `Result` return paths.
- Gotcha: `Mutex::lock().unwrap()` may represent poisoned internal state; classify separately from user input errors.
- Gotcha: examples are both diagnostics and user learning material; top-level `expect` may be acceptable, nested panic usually is not.
- Best practice: preserve context in error messages; do not flatten all failures to strings without path/action context.
- Likely failure mode: broad panic cleanup changes behavior but leaves no tests.

## Implementation Steps

1. Start from Phase 01's high/critical runtime panic and stall findings.
2. Separate findings into:
   - user input/project/package errors;
   - runtime device/renderer errors;
   - frame/asset stall risks;
   - test-only or explicit invariants.
3. Convert selected high-risk paths to existing `Result`/error boundaries.
4. Add negative tests for user input/project/package/argument cases where Vulkan is not needed.
5. For stall-related changes, run debug-record evidence before/after if feasible, or record why only post-change evidence is available.
6. Update docs only if user-visible error behavior or command behavior changes.
7. Write the phase report and update the evidence index.

## Acceptance Criteria

- High-risk runtime panic findings assigned to Phase 03 are fixed or explicitly accepted.
- Error messages retain useful path/context.
- Tests cover converted error paths where possible without GPU.
- Stall changes have debug-record evidence or a documented reason no measurement was possible.
- No new broad API contract is introduced.

## Negative Checks

- Test-only panic cleanup does not displace runtime risk fixes.
- No swallowed errors.
- No unbounded waits introduced.
- No protected path edits.
- No final status overclaim.

## Validation Commands

Required:

```sh
cargo fmt --check
cargo check
cargo test -p renderer
cargo check -p renderer --examples
```

Run when relevant:

```sh
cargo test -p input
cargo test -p engine_pack
cargo test -p editor
cargo test -p dungeon_dogfood
```

Runtime/timing smokes for touched paths:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_async_loading-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_model_load-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-env-timing.jsonl
```

## Stop Conditions

- Stop if fixing a stall requires asset streaming architecture redesign.
- Stop if converting an error path would require breaking public APIs.
- Stop if a runtime crash appears related to Phase 02 lifecycle work; route back to targeted repair.
- Stop after two failed repair attempts for the same issue and escalate.

## Evidence Expectations

- Worker report: `reports/phase-03-runtime-panic-stall-hardening.md`
- Validator report: `validation/phase-03-validation-report.md`
- Debug reports for runtime/stall behavior.
- Tests and command outputs summarized in evidence index.

## Do Not Close Unless

- Every Phase 03 assigned finding has a disposition.
- Converted runtime failures have tests or a clear reason tests are infeasible.
- Stall claims have evidence.
- Evidence index remains conservative.
