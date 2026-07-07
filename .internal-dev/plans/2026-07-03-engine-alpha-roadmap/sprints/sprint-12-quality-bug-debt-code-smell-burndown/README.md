# Sprint 12: Quality, Bug Debt, And Code Smell Burn-Down

Status: planned

## Objective

Remove or explicitly classify the defects and code smells that would make the alpha engine unstable, surprising, or hard to validate.

## User-Visible Outcome

An alpha user or dogfood developer gets a more reliable engine: shutdown paths do not hide allocator crashes, renderer/runtime failures become actionable errors where practical, docs/examples match the supported alpha surface, and remaining critical residuals are named with mitigation instead of hidden behind a green status.

## In Scope

- Residual inventory and triage across current bug records, prior sprint evidence, targeted source scans, renderer docs, runtime examples, and validation summaries.
- Vulkan lifecycle hardening for destroy paths, swapchain/presentation cleanup, transfer/host-buffer cleanup, deletion queue ordering, and shutdown crash risk.
- Runtime panic/unwrap triage for non-test paths, especially renderer examples, root runtime/project loading, asset/frame paths, and Vulkan calls that can fail at runtime.
- Frame or asset loading stalls that can block alpha workflows, with debug-record evidence when behavior is touched.
- Stale docs/examples and accidental public-contract drift left by alpha sprint work.
- Focused tests and validation matrix updates for scene/package/runtime flows where gaps are confirmed.
- Conservative residual acceptance for issues that are real but too large or risky to bury inside burn-down.

## Out Of Scope

- Sprint 09 facade implementation or active Sprint 09 files during this planning task.
- Sprint 10 advanced rendering opt-in implementation.
- Sprint 11 dogfood vertical slice work except where dogfood exposes quality defects that need classification.
- Broad renderer architecture rewrite.
- New engine features unless the smallest correct bug fix requires a tiny helper or test seam.
- Public API breaking changes without a user decision gate.
- Updating `SPRINT-TRACKER.md`; the main thread owns tracker reconciliation.
- Editing `.idea/engine.iml` or `.reasonix/`.

## Target Surfaces

Code:
- `src/renderer/src/vulkan/`
- `src/renderer/src/data/`
- `src/renderer/src/api/`
- `src/renderer/examples/`
- `src/runtime.rs`
- `src/launch.rs`
- `apps/editor/`
- `apps/dungeon_dogfood/`
- focused tests under existing crate test modules or `src/renderer/tests/`

Docs:
- `docs/api/`
- `docs/internal/`
- `README.md`
- package/module `AGENTS.md` only if process or runtime guidance materially changes.

`.internal-dev` artifacts:
- this plan suite;
- `reports/phase-*.md`;
- `validation/phase-*-validation-report.md`;
- `validation/final-quality-review.md`;
- `artifacts/validation-summary.json`;
- new or updated `.internal-dev/bugs/` reports only for real out-of-scope defects discovered during execution.

## Assumptions

- Later execution will happen on `sprint/alpha-12-quality-bug-debt-code-smell-burndown` after the main thread reconciles Sprint 09 state.
- Current dirty Sprint 09 files are not planning outputs and must be preserved.
- Some residuals may be valid alpha-accepted risks, but critical residuals cannot be silently carried.
- Code is logical truth; docs are intended truth.
- Desktop screenshots are not valid renderer proof. Visible renderer behavior requires engine-owned headless capture where applicable.

## Risks And Gotchas

- Many `unwrap`, `expect`, and `panic!` occurrences are test-only or invariant checks; blanket replacement would make the code worse.
- Vulkan lifecycle bugs can turn into double-free, leak, or use-after-free failures if cleanup ordering is changed without fence and idle reasoning.
- `VkSubAllocator::destroy` appears implemented in current source, but prior evidence and docs still refer to destroy-path residuals; the sprint must verify current truth before acting.
- Docs contain stale line references and duplicate chapter families. Fixing every stale sentence is not required unless it changes alpha-facing behavior.
- Runtime smoke can take 20-30 seconds to initialize; use timeout-bound commands and debug records.
- `cargo test -p renderer` may expose doctest/prose failures; classify pre-existing failures separately from Sprint 12 regressions.

## Acceptance Criteria

- Critical residuals are either closed with code/tests/docs/evidence or explicitly accepted for alpha with owner, mitigation, reproduction, and follow-up path.
- Vulkan lifecycle/destroy/swapchain/shutdown issues have focused code review, tests where feasible, and runtime smoke evidence when touched.
- Non-test runtime panic paths are triaged and the highest-risk sites are converted to recoverable errors or documented invariant checks.
- Frame or asset stall risks are measured or bounded when modified; evidence is written under `.internal-dev/debug_reports/`.
- Docs/examples/test surfaces no longer contradict the supported alpha contracts created by previous sprints.
- Validation summary stays conservative until all phase validators and final quality review pass.

## Negative Criteria

- Do not claim `fully_validated` while residuals remain.
- Do not hide critical bugs as generic TODOs.
- Do not rewrite low-level Vulkan ownership without proving synchronization and destruction order.
- Do not break public API compatibility without user approval.
- Do not treat desktop screenshots as visual validation.
- Do not update `SPRINT-TRACKER.md`.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Validation Plan

Compile/test:
- `cargo fmt --check`
- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- `cargo test -p input`
- `cargo test -p renderer`
- targeted tests for touched support crates/apps, such as `cargo test -p engine_pack`, `cargo test -p physics`, `cargo test -p audio`, `cargo test -p scripting`, `cargo test -p dungeon_dogfood`, or `cargo test -p editor` only when relevant.

Runtime smoke:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_pbr-timing.jsonl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_unlit-timing.jsonl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_model_load-timing.jsonl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_async_loading-timing.jsonl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-timing.jsonl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-env-timing.jsonl`

Visual/capture proof:
- Required for visible renderer/editor output changes.
- Use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Store evidence under `.internal-dev/captures/sprint-12-quality-burndown/`.

Docs/process checks:
- targeted `rg` scans for stale sprint, TODO, `/tmp`, pending/planned claims, and known residual phrases.
- final stale-reference sweep before final validation.

## Advanced-Planner Handoff

Execution phases:

1. Residual inventory and triage lock.
2. Vulkan lifecycle, destroy, swapchain, and shutdown hardening.
3. Runtime panic/error and frame/asset stall hardening.
4. Docs, examples, public-contract drift, and test gap cleanup.
5. Final validation matrix, residual acceptance, and evidence closeout.

Each phase has a worker directive under `worker-directives/` and a validator report path under `validation/`.

## Closeout Checklist

- Validation evidence recorded in `artifacts/validation-summary.json`.
- Known residuals tracked with mitigation.
- Final quality review written.
- Changelog timing confirmed with user if repo guidance requires it.
- Sprint tracker update left to main thread.
