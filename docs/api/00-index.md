# API Documentation Index

## 1. Purpose & Audience
This index is for students, hobbyists, and indie developers who are comfortable with Rust but new to engine architecture and rendering pipelines. It focuses on facade-level workflows first, then points to internal deep dives when needed.

## 2. Where This Fits in Engine Flow
This API track covers the public entry path used by runtime examples:
`Renderer::new(...)` -> per-frame input/camera update -> `Renderer::render_scene(...)` (or explicit `begin_frame` / `render_scene_in_frame` / `end_frame`).

## 3. Key Concepts
- Audience split: `/docs/api` explains stable usage patterns; `/docs/internal` explains implementation details and risks.
- Learning path: start with API usage, then use internal docs to debug behavior or extend engine systems.
- New learner first-stop: `docs/api/01-student-quickstart.md` gets you from setup to first frame and first model load.
- Documentation contract: all chapter docs follow the same 10-section structure and include labeled snippets.
- Template contract reference: [`docs/api/00-index.md`](00-index.md) and [`docs/internal/00-index.md`](../internal/00-index.md).
- Interlink contract: every chapter is reachable from this index and must include at least two `See Also` links.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/api/renderer.rs (simplified usage)
let mut renderer = Renderer::new(config, &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_default();

loop {
    renderer.render_scene(&window, &mut scene)?;
}
```

Snippet Type: Real
```rust
// Explicit frame API (src/renderer/src/api/renderer.rs)
let mut frame = renderer.begin_frame(&window)?;
renderer.render_scene_in_frame(&mut frame, &mut scene)?;
renderer.end_frame(frame)?;
```

Migration mapping matrix (implemented scope `00`-`16`):

| Legacy topic | New destination |
|---|---|
| Documentation style system and split contract | [`docs/api/00-index.md`](00-index.md) and [`docs/internal/00-index.md`](../internal/00-index.md) |
| Foundation structure and migration | [`docs/api/00-index.md`](00-index.md) and [`docs/internal/00-index.md`](../internal/00-index.md) |
| Synchronization and fencing (internal KB) | [`docs/internal/02-synchronization-and-fencing.md`](../internal/02-synchronization-and-fencing.md) |
| Asset lifecycle and I/O (internal KB) | [`docs/internal/03-asset-lifecycle-and-io.md`](../internal/03-asset-lifecycle-and-io.md) |
| Rendering pipeline mental model (internal KB) | [`docs/internal/01-rendering-pipeline-mental-model.md`](../internal/01-rendering-pipeline-mental-model.md) |
| API facade: renderer lifecycle and frame flow | [`docs/api/02-renderer-lifecycle-and-frame-api.md`](02-renderer-lifecycle-and-frame-api.md) |
| Internal API-to-backend handoff | [`docs/internal/04-api-to-backend-handoff.md`](../internal/04-api-to-backend-handoff.md) |
| Internal Vulkan sync and frame lifecycle | [`docs/internal/05-vulkan-sync-and-frame-lifecycle.md`](../internal/05-vulkan-sync-and-frame-lifecycle.md) |
| Internal data sub-allocation and transfer queue | [`docs/internal/06-data-suballocation-and-transfer.md`](../internal/06-data-suballocation-and-transfer.md) |
| API render hooks and extension points | [`docs/api/05-render-hooks-and-extension-points.md`](05-render-hooks-and-extension-points.md) |
| Engine launch/runtime arguments | [`docs/api/07-engine-arguments.md`](07-engine-arguments.md) |
| Internal rendergraph dependencies and aliasing | [`docs/internal/07-rendergraph-dependencies-and-aliasing.md`](../internal/07-rendergraph-dependencies-and-aliasing.md) |
| API scene graph and fragment workflows | [`docs/api/03-scene-graph-and-fragment-workflows.md`](03-scene-graph-and-fragment-workflows.md) |
| Internal scene flattening and culling | [`docs/internal/08-scene-flattening-and-culling.md`](../internal/08-scene-flattening-and-culling.md) |
| API input polling and layered dispatch model | [`docs/api/06-input-polling-and-listeners.md`](06-input-polling-and-listeners.md) |
| Internal input winit integration | [`docs/internal/09-input-winit-integration.md`](../internal/09-input-winit-integration.md) |
| Student quickstart guide | [`docs/api/01-student-quickstart.md`](01-student-quickstart.md) |

Current API chapter map (facade-first reading order):

| Topic | Chapter |
|---|---|
| Student bootstrap to first frame | [`docs/api/01-student-quickstart.md`](01-student-quickstart.md) |
| Renderer setup + frame lifecycle | [`docs/api/02-renderer-lifecycle-and-frame-api.md`](02-renderer-lifecycle-and-frame-api.md) |
| Scene graph + fragment mount workflows | [`docs/api/03-scene-graph-and-fragment-workflows.md`](03-scene-graph-and-fragment-workflows.md) |
| Sync/deferred assets + ticket lifecycle | [`docs/api/04-assets-sync-deferred-and-handles.md`](04-assets-sync-deferred-and-handles.md) |
| Render hooks + extension boundaries | [`docs/api/05-render-hooks-and-extension-points.md`](05-render-hooks-and-extension-points.md) |
| Input polling + layered dispatch model | [`docs/api/06-input-polling-and-listeners.md`](06-input-polling-and-listeners.md) |
| Engine launch/runtime arguments | [`docs/api/07-engine-arguments.md`](07-engine-arguments.md) |

Verification audit table (phase `16`):

| Audit check | Status | Notes |
|---|---|---|
| Mandatory 10-section order in API docs | Pass | All API chapters follow `1..10` heading order |
| Snippet markers (`Real`/`Pseudocode`) present | Pass | Each chapter includes labeled snippets |
| `See Also` section present with >= 2 links | Pass | Validated for all API chapters |
| Root-to-chapter traversal from this index | Pass | Chapter map uses direct relative links |

Snippet Type: Real
```markdown
[Renderer Lifecycle](02-renderer-lifecycle-and-frame-api.md)
[Input Internals](../internal/09-input-winit-integration.md)
```

## 5. Best Practices
- Keep render loop logic at the facade boundary; avoid direct Vulkan calls from app code.
- Pump deferred asset work regularly when using async loading paths.
- Treat handles (`MeshHandle`, `TextureHandle`, `EnvironmentHandle`) as opaque IDs, not array indices.

## 6. Gotchas & Failure Modes
- Calling `render_scene` while an explicit frame is open returns an invalid state error.
- Calling `render_scene_in_frame` twice for the same `FrameContext` is invalid.
- Rendering is skipped while resize is pending (`resize_requested`).

## 7. Debugging Playbook
- Start by reproducing with `cargo run -p renderer --example api_test`.
- If frame submission fails, verify call order: `begin_frame` -> `render_scene_in_frame` -> `end_frame`.
- If assets appear missing, poll load tickets and verify load state transitions before drawing.

## 8. Cross-Module Links
- Scene submission builder: `src/renderer/src/scene/scene_world.rs`
- Render submission payload: `src/renderer/src/scene/render_submission.rs`
- Vulkan execution path: `src/renderer/src/vulkan/vk_render.rs`

## 9. Standard References
- GitHub Markdown docs: https://docs.github.com/en/get-started/writing-on-github
- Vulkan Guide: https://github.khronos.org/Vulkan-Site/guide/latest/
- Vulkan Spec index: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/index.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Sascha Willems Vulkan glTF PBR baseline: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- [`docs/api/01-student-quickstart.md`](01-student-quickstart.md)
- [`docs/api/02-renderer-lifecycle-and-frame-api.md`](02-renderer-lifecycle-and-frame-api.md)
- [`docs/api/03-scene-graph-and-fragment-workflows.md`](03-scene-graph-and-fragment-workflows.md)
- [`docs/api/04-assets-sync-deferred-and-handles.md`](04-assets-sync-deferred-and-handles.md)
- [`docs/api/05-render-hooks-and-extension-points.md`](05-render-hooks-and-extension-points.md)
- [`docs/api/06-input-polling-and-listeners.md`](06-input-polling-and-listeners.md)
- [`docs/api/07-engine-arguments.md`](07-engine-arguments.md)
- [`docs/internal/00-index.md`](../internal/00-index.md)
- [`docs/internal/01-rendering-pipeline-mental-model.md`](../internal/01-rendering-pipeline-mental-model.md)
