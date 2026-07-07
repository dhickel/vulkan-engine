# Sprint 10: Advanced Rendering Opt-In Contract

Status: planned

## Objective

Define and harden the alpha contract for advanced rendering so power users have explicit opt-in paths without expanding the beginner facade or encouraging unsafe Vulkan/rendergraph mutation as the default extension model.

## User-Visible Outcome

After execution, a renderer user can tell which extension surfaces are supported by default, which surfaces require `advanced-interop`, what misuse risks exist, and which advanced rendergraph/material/readback capabilities are still deferred until synchronization and resource contracts are stronger.

## In Scope

- Audit current advanced rendering, hook, debug view, capture/readback, and rendergraph exposure.
- Preserve the Sprint 09 beginner facade boundary and ensure facade examples do not depend on advanced APIs.
- Document safe default extension points separately from feature-gated alpha/unstable interop.
- Harden `advanced-interop` compile/documentation gates and add tests or checks that prove default builds do not require it.
- Add only a minimal named advanced extension/read-only diagnostic surface if the audit proves it can be implemented without exposing raw Vulkan handles or hidden synchronization contracts.
- Use true engine-owned headless capture only if implementation changes visible renderer behavior or capture/readback behavior.

## Out Of Scope

- Replacing the current linear rendergraph with dependency-derived scheduling.
- General raw Vulkan command buffer, descriptor, swapchain, allocator, or queue access for users.
- Production custom pass ABI, shader/plugin system, material graph, or hot shader reload.
- Beginner facade expansion for advanced-only needs.
- Sprint 09 implementation or tracker reconciliation.
- Edits to `.idea/engine.iml`, `.reasonix/`, or `SPRINT-TRACKER.md`.

## Target Surfaces

- Code: `src/renderer/src/api/advanced.rs`, `src/renderer/src/api/mod.rs`, `src/renderer/src/api/hooks.rs`, `src/renderer/src/api/renderer.rs`, `src/renderer/src/lib.rs`, `src/renderer/src/rendergraph/mod.rs`, selected capture/debug modules only if a named surface is added.
- Docs: `docs/api/00-index.md`, `docs/api/05-render-hooks-and-extension-points.md`, `docs/api/05-hooks.md`, `docs/api/08-debug.md`, `docs/internal/07-rendergraph-dependencies-and-aliasing.md`, any source-level docs needed for `advanced-interop`.
- Tests/examples: renderer compile checks, renderer examples, focused API/hook/advanced feature checks, no beginner example requiring `advanced-interop`.
- `.internal-dev` artifacts: this sprint suite, phase validation reports, audit report(s), final validation summary, optional capture/debug evidence.

## Assumptions

- Sprint 09 will be reconciled by the main thread before Sprint 10 execution starts.
- Existing dirty Sprint 09 work is not owned by this plan creation task.
- `advanced-interop` may remain unsafe and unstable for alpha, but it must be explicit and documented.
- Named extension points are preferred over raw Vulkan handles until synchronization, resource ownership, and pass-order contracts are validated.
- Headless capture evidence is required only for visible renderer/capture/readback behavior changes, not for docs-only or compile-only contract work.

## Risks And Gotchas

- `renderer::rendergraph` becomes public when `advanced-interop` is enabled, so docs must not imply it is stable or safe by default.
- Current rendergraph execution is linear pass order, not a hazard-checked DAG; custom pass registration is risky unless heavily constrained.
- One hooks doc currently overclaims command-buffer style extension despite live hooks exposing only API-level frame/depth context.
- Exposing raw backend handles could create unsound frame synchronization, descriptor lifecycle, swapchain, or allocator misuse.
- Default examples or docs could accidentally pull users into advanced APIs and undermine Sprint 09.

## Acceptance Criteria

- Advanced APIs are feature-gated or explicitly classified as default-safe facade hooks/debug views.
- Documentation marks advanced interop as alpha/unstable and lists misuse risks.
- Beginner facade docs/examples compile without `advanced-interop` and do not require advanced modules.
- Any new advanced API uses named extension points/read-only descriptors instead of raw Vulkan handles unless it remains inside the existing unsafe escape hatch.
- Rendergraph pass registration is either deferred with clear rationale or implemented only behind feature gates with explicit ordering/resource constraints and validation.
- Validation evidence records default and `advanced-interop` compile checks, example checks, docs drift review, and runtime/capture evidence when applicable.

## Negative Criteria

- Do not expose command buffers, raw `ash` handles, allocator handles, descriptor sets, swapchain images, or queue submission controls as the normal advanced path.
- Do not make `advanced-interop` a default feature.
- Do not add advanced imports to beginner facade examples or `renderer::prelude`.
- Do not claim `fully_validated` while runtime/capture evidence, phase validation, or residual risks are missing.
- Do not use desktop/compositor screenshots for renderer proof.

## Validation Plan

- Compile/test:
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p renderer --examples`
  - `cargo check -p renderer --features advanced-interop`
  - `cargo check -p renderer --examples --features advanced-interop`
  - focused tests added or selected by implementation workers.
- Runtime smoke:
  - `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-10-api-test-timing.jsonl`
  - Add advanced-feature runtime smoke only if a runnable advanced example/test surface is added.
- Visual/capture proof:
  - Not required for docs-only/compile-only contract hardening.
  - Required if visible renderer behavior or capture/readback behavior changes:
    `RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_dir .internal-dev/captures/sprint-10-advanced-rendering-opt-in-contract/headless-draw`
- Docs/process checks:
  - Stale-reference sweep over changed docs and this sprint directory.
  - Verify `artifacts/validation-summary.json` status is conservative and cross-field consistent.

## Advanced-Planner Handoff

Use the worker directives in `worker-directives/` and the dispatch order in `final-orchestration-plan.md`. Every mutating phase requires a phase validation report under `validation/` before dependent work proceeds.

## Closeout Checklist

- Validation evidence recorded.
- Known residuals tracked.
- Changelog timing confirmed with user if required by repo guidance.
- Main thread reconciles `SPRINT-TRACKER.md`; Sprint 10 workers must not update it unless explicitly instructed later.
