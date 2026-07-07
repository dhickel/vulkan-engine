# Current State Analysis

## Verified Planning Inputs

- `AGENTS.md` defines the renderer examples as canonical runtime entrypoints and requires headless capture skill usage for renderer visual proof.
- `.internal-dev/AGENTS.md` requires controlled `.internal-dev` access, conservative evidence, bug reports for out-of-scope defects, and changelog timing confirmation.
- `src/renderer/AGENTS.md` identifies high-risk renderer areas: `vk_render.rs`, `data_cache.rs`, descriptor/pipeline binding order, and incomplete destroy paths.
- `src/renderer/src/vulkan/AGENTS.md` names lifecycle, sync, resource ownership, frame execution, destroy TODOs, swapchain cleanup FIXME areas, and queue assumptions as current risks.
- The roadmap defines Sprint 12 around Vulkan lifecycle issues, runtime panics, stale docs/examples, frame or asset stalls, unintended public contracts, and test gaps.
- `SPRINT-TRACKER.md` currently marks Sprint 09 as active/planned and Sprint 12 proposed. The user explicitly said not to update the tracker.

## Current Worktree Constraint

At planning time the worktree had unrelated local changes:

- `.idea/engine.iml` modified.
- `.reasonix/` untracked.
- Sprint 09 active renderer API/example/test files modified or untracked.

This suite must not alter those files during planning. Later Sprint 12 execution should start only after the main thread reconciles Sprint 09 state and creates/switches to `sprint/alpha-12-quality-bug-debt-code-smell-burndown`.

## Existing Residual Records

`.internal-dev/bugs/` contains only archived bug reports at planning time:

- `renderer-double-free-on-shutdown/.archive/report.md` reports a shutdown double free after `dungeon_dogfood` runs and points to renderer shutdown/destruction ordering.
- `.archive/2026-02-16-workspace-missing-dungeon-dogfood/report.md` is historical workspace membership context.

Prior memory and validation context also indicate conservative residual handling around renderer doctest/prose failures, destroy-path risk, and `demo_unlit` behavior. Treat this as stale-prone context that Phase 01 must verify.

## Targeted Source Findings

Targeted scans found these current signals:

- `src/renderer/src/vulkan/vk_util.rs` still contains `find_memory_type(...) -> todo!()`.
- `src/renderer/src/vulkan/vk_storage.rs` implements `VkSubAllocator::destroy`, but allocation growth still has a TODO and must be verified against docs that mention destroy-path residuals.
- `src/renderer/src/vulkan/vk_types.rs` destroys only `self.fence[0]` in `VkHostBuffer::destroy`, while internal docs mention two submission fences.
- `src/renderer/src/vulkan/vk_types.rs` still has runtime `unwrap()` in command buffer reset and fence status/reset paths.
- `src/renderer/src/vulkan/vk_render.rs` has explicit shutdown ordering in `Drop`, including device idle, transfer destroy, presentation destroy, data cache destroy, deletion queue, swapchain, allocator, device, surface, debug, and instance teardown.
- `src/renderer/src/vulkan/vk_render.rs` currently comments that old present views are destroyed by `replace_present_images`, so docs claiming old image views are not destroyed may be stale.
- Docs under `docs/internal/` still mention destroy TODOs, old image view cleanup, `VkHostBuffer::destroy` fence behavior, and stale line references.
- Runtime/example paths contain `expect(...)` at loop boundaries, which may be acceptable for examples if the failure message is clear, but not if those examples are the alpha user path.

## Architecture Fit

Sprint 12 should not invent a new architecture. It should tighten existing contracts:

- Vulkan ownership cleanup belongs under `src/renderer/src/vulkan/` and must respect `VkDestroyable`, frame fence/deletion queue ordering, and `VkRenderCore::drop`.
- Data/cache fixes belong under `src/renderer/src/data/` only when they directly relate to asset stalls, default resource safety, or cleanup.
- Runtime error conversion belongs at existing API boundaries: `RendererError`, asset/runtime project loading errors, launch parsing, and example `Result` returns.
- Public contract cleanup must align with Sprint 09's facade classification rather than introducing Sprint 10 advanced API policy.

## Validation Blind Spots

- Unit tests can cover many package/scene/runtime error paths without Vulkan.
- Vulkan lifecycle cleanup still needs bounded runtime smoke because compile checks cannot prove teardown correctness.
- Shutdown crashes may appear only after timeout/INT or window close; use smoke logs and, if needed, focused debug reports.
- Headless capture is required only for visible output changes. Lifecycle-only cleanup may not require capture if runtime smokes prove startup/present/shutdown health.
- Docs can pass scans while still stale; validators must compare claims to source and prior phase reports.

## Planning Consequence

The sprint must begin with a classification phase. Without that, workers may waste time fixing test-only unwraps, stale doc warnings, or historical residuals already resolved in code. After classification, implementation phases should target the highest risk confirmed defects and produce conservative evidence.
