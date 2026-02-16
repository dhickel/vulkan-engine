# Internal Documentation Index

## 1. Purpose & Audience
This index is for contributors working inside renderer internals (frame loop, synchronization, caches, and pass sequencing). It assumes Rust proficiency and basic graphics familiarity, but explains Vulkan-heavy topics in engine-specific terms.

## 2. Where This Fits in Engine Flow
These docs map to the internal runtime path:
`Renderer::render_scene(...)` -> `SceneWorld::build_submission()` -> `VkRender::render_with_hooks(...)` -> rendergraph passes -> submit/present.

## 3. Key Concepts
- Internal docs explain how and why systems are implemented, not just how to call them.
- Current state is alpha: docs include known risks and non-hardened behavior.
- Every chapter follows the same 10-section structure and labels snippet types explicitly.
- Template contract (exact order): Purpose & Audience; Where This Fits in Engine Flow; Key Concepts; Code Walkthrough; Best Practices; Gotchas & Failure Modes; Debugging Playbook; Cross-Module Links; Standard References; See Also.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/rendergraph/mod.rs
pub fn default_graph() -> Self {
    Self::new(vec![
        Box::new(PrepareTargetsPass),
        Box::new(SkyboxPass),
        Box::new(GeometryPass),
        Box::new(PresentCopyPass),
        Box::new(ImguiPass),
    ])
}
```

Reading order for current `00`-`16` implementation scope:
1. [`docs/internal/01-rendering-pipeline-mental-model.md`](01-rendering-pipeline-mental-model.md)
2. [`docs/internal/02-synchronization-and-fencing.md`](02-synchronization-and-fencing.md)
3. [`docs/internal/03-asset-lifecycle-and-io.md`](03-asset-lifecycle-and-io.md)
4. [`docs/internal/04-api-to-backend-handoff.md`](04-api-to-backend-handoff.md)
5. [`docs/internal/05-vulkan-sync-and-frame-lifecycle.md`](05-vulkan-sync-and-frame-lifecycle.md)
6. [`docs/internal/06-data-suballocation-and-transfer.md`](06-data-suballocation-and-transfer.md)
7. [`docs/internal/07-rendergraph-dependencies-and-aliasing.md`](07-rendergraph-dependencies-and-aliasing.md)
8. [`docs/internal/08-scene-flattening-and-culling.md`](08-scene-flattening-and-culling.md)
9. [`docs/internal/09-input-winit-integration.md`](09-input-winit-integration.md)

Verification audit table (phase `16`):

| Audit check | Status | Notes |
|---|---|---|
| Mandatory 10-section order in internal docs | Pass | All internal chapters use `1..10` heading order |
| Snippet markers (`Real`/`Pseudocode`) present | Pass | Each chapter includes labeled snippets |
| `See Also` section present with >= 2 links | Pass | Validated for all internal chapters |
| Vulkan + glTF links in `Standard References` | Pass | All internal chapters include both |
| Root-to-chapter traversal from this index | Pass | Reading-order list uses direct relative links |

Snippet Type: Real
```markdown
[Frame Sync](05-vulkan-sync-and-frame-lifecycle.md)
[Facade Lifecycle](../api/02-renderer-lifecycle-and-frame-api.md)
```

## 5. Best Practices
- Start from frame-level mental model before editing fine-grained barriers.
- Keep handle validation and generation semantics intact when touching caches.
- Preserve pass order assumptions unless transitions and ABI contracts are updated together.

## 6. Gotchas & Failure Modes
- Changing pass order without matching transitions can silently break output.
- Swapchain rebuild has known cleanup sharp edges.
- `todo!()` destroy paths still exist in some Vulkan wrappers.

## 7. Debugging Playbook
- Use `cargo check -p renderer` first.
- Use headless smoke runs with timeout to verify startup and frame loop stability.
- Enable validation layers via renderer startup options before reasoning about sync correctness.

## 8. Cross-Module Links
- Top-level contributor orientation: `AGENTS.md`
- Renderer package guide: `src/renderer/AGENTS.md`
- Vulkan deep guide: `src/renderer/src/vulkan/AGENTS.md`
- Data/cache deep guide: `src/renderer/src/data/AGENTS.md`

## 9. Standard References
- Vulkan Guide: https://github.khronos.org/Vulkan-Site/guide/latest/
- Vulkan Spec index: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/index.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- vkguide: https://vkguide.dev/
- Sascha Willems Vulkan glTF PBR baseline: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- [`docs/internal/02-synchronization-and-fencing.md`](02-synchronization-and-fencing.md)
- [`docs/internal/03-asset-lifecycle-and-io.md`](03-asset-lifecycle-and-io.md)
- [`docs/internal/01-rendering-pipeline-mental-model.md`](01-rendering-pipeline-mental-model.md)
- [`docs/internal/04-api-to-backend-handoff.md`](04-api-to-backend-handoff.md)
- [`docs/internal/05-vulkan-sync-and-frame-lifecycle.md`](05-vulkan-sync-and-frame-lifecycle.md)
- [`docs/internal/06-data-suballocation-and-transfer.md`](06-data-suballocation-and-transfer.md)
- [`docs/internal/07-rendergraph-dependencies-and-aliasing.md`](07-rendergraph-dependencies-and-aliasing.md)
- [`docs/internal/08-scene-flattening-and-culling.md`](08-scene-flattening-and-culling.md)
- [`docs/internal/09-input-winit-integration.md`](09-input-winit-integration.md)
