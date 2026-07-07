# Specification Lock

## Acceptance Criteria

- Sprint 12 starts with a verified residual inventory, not stale assumptions.
- Critical quality residuals are closed or explicitly accepted for alpha with mitigation, reproduction/evidence, and follow-up routing.
- Vulkan lifecycle risks are addressed in priority order: shutdown crash/double free, incomplete destroy paths, swapchain/presentation cleanup, host-buffer/fence destruction, deletion queue ordering, and resize/rebuild regressions.
- Non-test runtime `unwrap`, `expect`, and `panic!` sites are classified by risk. High-risk user/runtime paths are converted to error propagation or guarded invariants where feasible.
- Frame and asset stall fixes use debug-record evidence or explicit measurement when behavior changes.
- Docs/examples/tests are made consistent with the alpha contracts from prior sprints, especially Sprint 09 facade classification, without rewriting unrelated documentation.
- Validation commands, runtime smokes, capture evidence if applicable, phase validator reports, residual risks, and tooling constraints are recorded in `artifacts/validation-summary.json`.
- Final status remains conservative when residuals remain.

## Validation Criteria

Required static checks across the suite:

```sh
cargo fmt --check
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo test -p renderer
```

Required runtime smokes when renderer runtime behavior is touched:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_pbr-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_unlit-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_model_load-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-demo_async_loading-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-timing.jsonl
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-12-api_test-env-timing.jsonl
```

Required visual proof when visible renderer/editor output changes:

- Read `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Use engine-owned headless capture, not desktop screenshots.
- Store captures under `.internal-dev/captures/sprint-12-quality-burndown/`.

Required stale/reference scans before final quality review:

```sh
rg -n "TODO|FIXME|todo!\\(|unimplemented!\\(|panic!\\(|unwrap\\(|expect\\(" src/renderer/src src/renderer/examples src/runtime.rs src/launch.rs apps docs/api docs/internal
rg -n "pending|planned|not implemented|/tmp|sprint-0[1-9]|Sprint 0[1-9]|gap-report|desktop screenshot|headless-draw" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-12-quality-bug-debt-code-smell-burndown
rg -n "destroy path|double free|swapchain|old image view|VkSubAllocator|VkHostBuffer|fence\\[0\\]" src/renderer/src/vulkan src/renderer/src/data docs/internal .internal-dev/bugs
```

## Negative Criteria

- Do not update `SPRINT-TRACKER.md`.
- Do not touch `.idea/engine.iml` or `.reasonix/`.
- Do not edit current Sprint 09 active files during this planning task.
- Do not remove public exports or narrow public API support without explicit user approval.
- Do not replace test-only `unwrap`/`panic!` mechanically unless it improves test clarity.
- Do not accept critical runtime crash or data-loss bugs without mitigation and follow-up.
- Do not use desktop screenshots for visual proof.
- Do not claim `fully_validated` if any required validator/capture is missing or residual critical risk remains.

## Non-Goals

- Advanced rendering feature implementation.
- Dogfood vertical slice expansion.
- Public semver stabilization.
- Whole-codebase lint crusade.
- Full documentation rewrite.
- Dependency modernization unless required for a confirmed quality defect.

## Constraints

- Rust 2021 workspace.
- Code is logical truth; docs are intended truth.
- Intended later execution branch: `sprint/alpha-12-quality-bug-debt-code-smell-burndown`.
- Current dirty worktree includes Sprint 09 active files. Preserve local state.
- `.internal-dev/` is ignored; main thread may need to force-add artifacts.
- Runtime validation must be timeout-bound.
- Browser/Playwright validation does not apply; renderer visual validation uses the engine headless capture skill.

## Assumptions To Verify

- Sprint 09 will finish or be reconciled before Sprint 12 execution begins.
- The archived shutdown double-free report may still be relevant and must be re-tested before closure.
- Current source may already have fixed some historically named residuals; the first phase must separate verified current defects from stale records.
- `cargo test -p renderer` may include pre-existing doctest/prose failures; do not hide them as Sprint 12 pass/fail noise.

## User Decision Gates

- Stop and ask before breaking or removing public APIs.
- Stop and ask before expanding into a dedicated renderer architecture refactor.
- Stop and ask before accepting an alpha-critical crash/data-loss defect as residual.
- Stop and ask if a fix requires broad dependency upgrades, workspace membership changes, or non-trivial app rewrites.
- Stop and ask if true headless capture is required but blocked and a substitute proof path is proposed.

## Stop Rules

- Stop for plan revision if residual inventory shows Sprint 12 must split into multiple burn-down sprints.
- Stop implementation if Vulkan cleanup requires unsafe ownership changes that cannot be proven with current tests/runtime evidence.
- Stop phase progression on unresolved validation failures unless they are explicitly classified as pre-existing and accepted.
- Stop final closeout if `artifacts/validation-summary.json` contradicts phase reports or claims final validation early.
