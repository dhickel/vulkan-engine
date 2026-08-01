# BSP Renderer Retirement Reap Repair

## Date

2026-07-31

## Change Summary

Repaired BSP GPU retirement so normal frame reaping does not recursively acquire the VMA allocator mutex and renderer shutdown destroys every accepted BSP closure after `device_wait_idle`.

## Files

- `src/renderer/src/vulkan/vk_frame.rs`
- `src/renderer/src/vulkan/vk_render.rs`
- `src/renderer/src/data/data_cache.rs`
- `src/renderer/src/data/retirement.rs`
- `docs/internal/18-bsp-runtime-and-lifetime.md`
- `.internal-dev/specifications/bsp-transaction-ownership.md`
- `.internal-dev/knowledge/renderer-vulkan-lifecycle-and-audit-gotchas.md`

## Behavioral Impact

BSP retirement acquires cache and allocator locks in a fixed order and reuses the held allocator guard for texture destruction. A successful terminal device-idle wait now reaps pending BSP arena closures through the latest submitted serial before data-cache and VMA teardown. Device-loss paths continue to avoid manufacturing completion or invoking unsafe destruction.

## Specification Impact

Updated the BSP transaction ownership contract to record renderer acknowledgement, normal fence reaping, terminal post-idle reaping, and the required lock order.

## Risks

Poisoned locks are rejected during normal frame reaping so queue ownership remains retryable. Terminal post-idle teardown recovers poisoned guards because no later retry boundary exists and renderer-owned resources must not be dropped without destruction.

## Follow-up Items

- Keep the BSP lock-order rule aligned with any future cache or allocator restructuring.
- GitHub #60 remains the separate committed-bridge teardown concern.
