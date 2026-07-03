# Phase 02 Worker Directive: Vulkan Lifecycle, Destroy, Swapchain, And Shutdown Hardening

## Objective

Fix or explicitly classify the highest-risk Vulkan lifecycle defects confirmed by Phase 01, focusing on shutdown/destroy ordering, swapchain/presentation cleanup, host-buffer/fence cleanup, and known double-free risk.

## User-Visible Outcome

Renderer examples and dogfood/runtime shutdown no longer hide allocator crashes or lifecycle warnings caused by Sprint 12's target defects. Remaining lifecycle residuals are named and mitigated.

## Editable Targets

Primary code targets, only as confirmed by Phase 01:

- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/vulkan/vk_types.rs`
- `src/renderer/src/vulkan/vk_storage.rs`
- `src/renderer/src/vulkan/vk_util.rs`
- `src/renderer/src/vulkan/vk_init.rs`
- `src/renderer/src/data/data_cache.rs`
- focused renderer tests under existing test modules or `src/renderer/tests/`

Docs/artifacts:

- `docs/internal/04-vulkan-subsystem.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `docs/internal/06-data-suballocation-and-transfer.md`
- relevant `.internal-dev/bugs/<bug-id>/report.md`
- `reports/phase-02-vulkan-lifecycle-hardening.md`
- `artifacts/validation-summary.json`

## Forbidden Scope

- Do not redesign rendergraph or the whole Vulkan ownership model.
- Do not remove public APIs.
- Do not change shader/material behavior unless lifecycle evidence requires it.
- Do not edit `SPRINT-TRACKER.md`, `.idea/engine.iml`, or `.reasonix/`.
- Do not claim visual correctness without headless capture if output changed.

## Supporting Docs To Read

- Phase 01 report and validation report.
- `src/renderer/src/vulkan/AGENTS.md`
- `docs/internal/04-vulkan-subsystem.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `docs/internal/06-data-suballocation-and-transfer.md`
- relevant source around confirmed defects.

## Senior Engineer Guidance

- Direct target: fix confirmed lifecycle bugs, not every Vulkan TODO.
- Approach: change one ownership boundary at a time and validate before expanding.
- Gotcha: swapchain images are externally owned; present image views are engine-owned.
- Gotcha: fences and command pools may be referenced by pending transfers; check queue/fence ownership before destruction.
- Best practice: keep `device_wait_idle` ordering explicit and avoid hiding errors with broad `let _ =`.
- Likely failure mode: fixing double-free by skipping destruction and creating a leak.

## Implementation Steps

1. Start from Phase 01's confirmed Vulkan findings. Do not add new lifecycle scope without recording why.
2. For each target resource, document owner, creator, destroyer, and in-flight synchronization assumptions in the phase report.
3. Fix the smallest confirmed defect set:
   - stale `todo!()` reachable at runtime;
   - missing fence/resource destruction;
   - duplicate destroy or swapchain view leak;
   - shutdown double-free reproduction if still current.
4. Add focused tests where behavior can be tested without Vulkan.
5. Update internal docs only where code/docs diverged.
6. Run required compile checks.
7. Run runtime smokes for touched renderer paths with debug-record output.
8. If visible output changed, run headless capture via the project skill.
9. Write phase report and update evidence index.

## Acceptance Criteria

- Confirmed critical lifecycle defect(s) assigned to Phase 02 are fixed or explicitly accepted with mitigation.
- No new broad ownership model is introduced.
- Runtime smokes do not show allocator double-free, fatal Vulkan errors, or teardown regressions for touched paths.
- Docs no longer repeat stale lifecycle claims for the fixed items.
- Evidence records any residual lifecycle risks.

## Negative Checks

- No leaked critical residual hidden as "TODO".
- No blanket `unwrap` removal without error handling plan.
- No desktop screenshots as proof.
- No protected path edits.

## Validation Commands

Required:

```sh
cargo fmt --check
cargo check -p renderer
cargo check -p renderer --examples
```

Run if tests were added/changed:

```sh
cargo test -p renderer
```

Runtime smokes for touched renderer behavior:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_pbr-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-timing.jsonl
```

Run additional smoke commands from `shared/validation-matrix.md` when the touched code affects async loading, model loading, unlit, or environment loading.

## Stop Conditions

- Stop if the confirmed issue requires a full resource ownership rewrite.
- Stop if shutdown double-free remains after two scoped repair attempts; escalate to fresh targeted repair worker.
- Stop if runtime validation cannot run and no approved fallback exists.
- Stop if a fix requires public API breakage.

## Evidence Expectations

- Worker report: `reports/phase-02-vulkan-lifecycle-hardening.md`
- Validator report: `validation/phase-02-validation-report.md`
- Debug reports under `.internal-dev/debug_reports/`.
- Capture artifacts under `.internal-dev/captures/sprint-12-quality-burndown/` only if visual output changed.
- Updated bug reports for lifecycle residuals.

## Do Not Close Unless

- Phase 01 Vulkan findings are each fixed, accepted, or escalated.
- Required commands are recorded with results.
- Runtime evidence matches the behavior touched.
- Evidence index status is conservative.
