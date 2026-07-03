# Senior Engineer Guidance

## Operating Posture

- Start from current source, not old docs. Several residuals may be stale, and stale bug labels are themselves part of Sprint 12 cleanup.
- Fix defects by contract boundary. Vulkan lifetime bugs belong in Vulkan ownership code; user-facing failures belong in API/runtime error paths; docs drift belongs in docs.
- Prefer scoped repair over cleanup enthusiasm. Burn-down does not mean rewriting every TODO or every test `unwrap`.
- Keep evidence conservative. A green compile check does not prove lifecycle correctness; a runtime smoke does not prove visual correctness.

## Vulkan Lifecycle Guidance

- Device idle in `VkRenderCore::drop` is necessary but not enough. Teardown order still matters for allocator/device/surface ownership.
- Before changing a destroy path, identify the owning type, whether the resource is swapchain-owned or engine-owned, and whether any frame fence/deletion queue can still reference it.
- `VkDestroyable` implementations should be idempotent in practice where feasible, or guarded by ownership state so duplicate destroy is impossible.
- Do not destroy swapchain images owned by the swapchain. Destroy image views created by the engine.
- Treat `Arc<Mutex<Allocator>>` lock unwraps as different from recoverable user errors; if poisoning is possible at runtime, classify risk before changing.
- If a shutdown double-free still reproduces, narrow the resource class before changing multiple teardown layers.

## Panic And Error Guidance

- Tests can use `unwrap` and `panic!` when they assert fixtures or failure variants.
- Runtime paths should return existing error types where they already exist. Avoid inventing broad new error hierarchies unless a local enum is clearly missing.
- Example top-level `expect("failed to run ...")` may be acceptable if the example returns `Result`; internal example loop panics are higher risk.
- Vulkan FFI calls often return `VkResult`; convert high-risk calls near runtime boundaries to `Result` when callers can reasonably handle or report them.
- `todo!()` in compiled runtime code is never alpha-stable unless provably unreachable and documented as such.

## Stall Guidance

- Do not guess about frame or asset stalls. Use debug-record timing evidence before and after behavior changes.
- If a long wait is intentional, make the bound and user-visible behavior clear.
- If a worker finds unbounded blocking that requires streaming architecture work, stop and file a dedicated follow-up rather than doing a risky partial rewrite.

## Docs And Examples Guidance

- Docs should describe what the current alpha supports, not what a future sprint intends.
- When source and docs disagree, update docs or file a tracking artifact; do not make code chase stale docs without reason.
- Keep old/duplicate doc families navigable but do not spend Sprint 12 rewriting all legacy docs.
- Examples are part of the alpha contract because they are canonical runtime entrypoints. If examples diverge from docs, fix either the example or the docs.

## Likely Failure Modes

- A worker marks historical residuals fixed without rerunning targeted scans or smoke checks.
- Vulkan cleanup fixes compile but introduce double-destroy on resize.
- Runtime panic cleanup accidentally erases useful error context.
- Docs stale sweep removes deliberate TODOs that should remain as accepted residuals.
- Evidence index claims `fully_validated` while accepted residuals remain.

## Protected Local State

- Do not touch `.idea/engine.iml`.
- Do not touch `.reasonix/`.
- During this planning task, do not edit Sprint 09 active files. Later execution must start from a reconciled Sprint 12 branch.
