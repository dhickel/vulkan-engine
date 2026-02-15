# 10 - Synchronization and Performance Gotchas

This chapter focuses on high-value failure modes while working on alpha renderer internals.

## Fence/Semaphore Contract

Current frame model uses per-frame sync objects:
- Fence: CPU reuse safety for frame resources.
- Acquire semaphore: image-available dependency.
- Render semaphore: render-complete dependency for present.

If fence wait/reset ordering is wrong, expect flicker, frame reuse corruption, or deadlocks.

## Image Layout Transition Sensitivity

Common transition chain includes:
- draw/depth target prep
- color attachment rendering
- transfer/copy to present image
- optional imgui overlay on present image
- final `PRESENT_SRC_KHR`

When debugging visuals, first validate layout transitions and stage masks.

Reference:
- https://github.khronos.org/Vulkan-Site/guide/latest/layout_transitions.html
- https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html

## Deferred Loading Progress and Stalls

Current asset tracker behavior:
- Bounded concurrency for deferred jobs (default max 4 in flight).
- Progress still requires render/begin-frame/manual pumping.
- Texture uploads finalize through poll-based completion; intermediate large batches may still block due to single staging buffer reuse.

Performance implication:
- Bulk loads improve over strict serialization, but heavy upload sets can still hitch when staging-buffer turnover is high.

## Known Hotspots

- `vk_render.rs` is high-blast-radius orchestration code.
- Environment preparation can cause visible frame hitches on first activation.
- Startup and sync model loads may stall while waiting on transfer completion.
- Texture compression in `Auto/Force` modes can add CPU cost during ingest.

## Debug Workflow

1. Reproduce with validation layers enabled.
2. Capture first validation error and fix that first.
3. Re-test on all example scenarios.
4. Re-test resize + environment transition specifically.

## Learn More

- Vulkan sync overview:
  - https://github.khronos.org/Vulkan-Site/guide/latest/synchronization.html
- `vkguide` descriptor/sync architecture patterns:
  - https://vkguide.dev/
