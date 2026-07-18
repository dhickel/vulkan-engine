# Rendergraph Dependencies and Resource Aliasing

## 1. Purpose & Audience
This chapter is for contributors editing rendergraph pass order, pass boundaries, or image transition behavior in backend rendering. It focuses on the current linear pass model and the planned (not yet implemented) graph scheduling and aliasing direction.

## 2. Where This Fits in Engine Flow
Current flow:
`Renderer::render_scene(...)` -> `SceneWorld::build_submission()` -> `VkRenderCore::render_with_hooks(...)` -> `execute_rendergraph_for_frame(...)` -> `RenderGraph::execute(...)` -> fixed pass list.

## 3. Key Concepts
- Current state is linear pass execution, not dependency-derived scheduling.
- `RenderGraph` only stores ordered `Vec<Box<dyn RenderPassNode>>`; it does not compute a DAG.
- Pass dependencies are mostly implicit through image layout/state carryover:
  - draw/depth target preparation must happen before shadow/skybox/geometry.
  - `ShadowPass` must finish its D32 depth write and shader-read transition before PBR geometry samples scene binding 5.
  - present image transition path differs depending on whether draw targets are used.
  - `DebugCapturePass` reads due draw/present captures after UI recording.
  - `TerminalPresentPass` owns the final windowed `COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR` handoff; it is a no-op headless.
- Resource aliasing is not implemented today:
  - no lifetime analysis for transient attachments
  - no allocator-level rendergraph alias planner
- Roadmap direction (not implemented): explicit pass resource declarations, topological sort, and transient resource aliasing with hazard checks.

## 4. Code Walkthrough
Snippet Type: Real
```rust
// src/renderer/src/rendergraph/mod.rs
pub fn default_graph() -> Self {
    Self::new(vec![
        Box::new(PrepareTargetsPass),
        Box::new(ShadowPass),
        Box::new(SkyboxPass),
        Box::new(GeometryPass),
        Box::new(PresentCopyPass),
        Box::new(ImguiPass),
        Box::new(DebugCapturePass),
        Box::new(TerminalPresentPass),
    ])
}

pub fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
    for pass in self.passes.iter() {
        pass.execute(ctx)
            .map_err(|err| format!("render pass '{}' failed: {err}", pass.name()))?;
    }
    Ok(())
}
```

Snippet Type: Real
```rust
// src/renderer/src/rendergraph/passes/present_copy_pass.rs
fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String> {
    let has_draw_targets = ctx.submission.has_draw_targets();
    let mut recording = ctx.present_copy_ctx();
    if has_draw_targets {
        recording.copy_draw_to_present();
    } else {
        recording.prepare_present_color_attachment();
    }
    Ok(())
}
```

Current pass contract and transition ownership matrix:

| Pass | Reads | Writes | Required incoming state | Outgoing state guarantee |
|---|---|---|---|---|
| `PrepareTargetsPass` | submission flags | `frame.draw.image`, `frame.depth.image` layouts | none | draw in `COLOR_ATTACHMENT_OPTIMAL`, depth in `DEPTH_ATTACHMENT_OPTIMAL` when `has_draw_targets()` |
| `ShadowPass` | optional directional light + opaque resolved draws | frame-slot D32 shadow image | prior contents discarded | shadow map in `SHADER_READ_ONLY_OPTIMAL` for PBR sampling |
| `SkyboxPass` | skybox mesh/material/env descriptors | draw color attachment | draw image already in `COLOR_ATTACHMENT_OPTIMAL` | draw color attachment remains renderable for later passes |
| `GeometryPass` | submission draw items + copied material records + scene shadow descriptor | draw color + depth | draw/depth attachments prepared; shadow map shader-readable | draw color/depth complete, draw color still in `COLOR_ATTACHMENT_OPTIMAL` |
| `PresentCopyPass` | draw image when enabled | present image layout + copy/blit result | if draw path: draw color in `COLOR_ATTACHMENT_OPTIMAL`; present image available as acquired/offscreen image | present image in `COLOR_ATTACHMENT_OPTIMAL` for UI/capture |
| `ImguiPass` | optional imgui draw data | present image | present image in `COLOR_ATTACHMENT_OPTIMAL` | present image remains `COLOR_ATTACHMENT_OPTIMAL`; headless/no-context path records no dynamic rendering region |
| `DebugCapturePass` | due capture requests + draw/present image | readback commands/status | source image rendered | due requests consumed before submit |
| `TerminalPresentPass` | present image | present image layout | windowed present image in `COLOR_ATTACHMENT_OPTIMAL` | windowed image in `PRESENT_SRC_KHR`; headless unchanged |

Snippet Type: Pseudocode
```text
# Future direction (not implemented):
passes = register_passes_with_declared_io()
graph = build_dependency_graph(passes) # edges from declared read/write hazards
order = topo_sort(graph)

for resource in transient_resources:
  lifetime = union(pass_index_first_use, pass_index_last_use)
  alias_candidate = find_non_overlapping_allocation(lifetime, format/usage)
  assign_or_allocate(resource, alias_candidate)

for pass in order:
  emit_required_barriers(pass.declared_inputs, pass.declared_outputs)
  execute(pass)
```

## 5. Best Practices
- Treat pass order as ABI until dependency metadata exists.
- Keep transition ownership local and explicit:
  - prepare pass owns draw/depth prep.
  - shadow pass owns D32 attachment-to-sampled transitions.
  - present-copy owns the renderable present layout and terminal-present owns the final swapchain handoff.
- When adding a pass, document:
  - required incoming layout/state
  - produced outgoing layout/state
  - whether it is reorderable and under what conditions.
- Separate language for current behavior from roadmap behavior to avoid false assumptions.

## 6. Gotchas & Failure Modes
- Hidden dependencies through implicit state carryover:
  - a pass may seem independent but still requires a specific prior transition.
- Incorrect reorder assumptions:
  - moving `GeometryPass` before `PrepareTargetsPass` breaks attachment layout expectations.
  - moving `ShadowPass` after `GeometryPass` makes geometry sample stale/unwritten shadow data.
  - moving `DebugCapturePass` after `TerminalPresentPass` invalidates present-image readback assumptions.
- Aliasing proposals without synchronization/lifetime analysis can create write-after-read and read-after-write hazards.
- Branch-dependent present path:
  - `has_draw_targets() == false` bypasses draw image and still requires present image transition for ImGui/present.

## 7. Debugging Playbook
- Step 1: verify pass order in `RenderGraph::default_graph()` before debugging barriers.
- Step 2: trace submission flags (`draw_skybox`, `draw_geometry`, `draw_imgui`) to understand which pass branches executed.
- Step 3: inspect layout transitions in:
  - `prepare_draw_targets(...)`
  - `copy_draw_to_present(...)`
  - `prepare_present_color_attachment(...)`
  - `transition_present_for_present(...)`
  - `ShadowPass::execute(...)`
- Step 4: if validation reports layout hazards, map the failing image to pass boundary ownership from the matrix above.
- Step 5: reproduce with fixed scene flags (all on vs all off) to isolate branch-specific dependency gaps.

## 8. Cross-Module Links
- Rendergraph core and pass ordering: `src/renderer/src/rendergraph/mod.rs`
- Pass implementations: `src/renderer/src/rendergraph/passes/mod.rs`
- Frame execution integration: `src/renderer/src/vulkan/vk_render.rs` (coordinator), `src/renderer/src/vulkan/vk_commands.rs` (recording)
- Submission flags and draw-target branch condition: `src/renderer/src/scene/render_submission.rs`
- Frame synchronization context: `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`

## 9. Standard References
- Vulkan synchronization chapter: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization
- Vulkan synchronization examples: https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html
- glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Frostbite framegraph article: https://www.ea.com/frostbite/news/framegraph-extensible-rendering-architecture-in-frostbite
- Baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/internal/01-rendering-pipeline-mental-model.md`
- `docs/internal/02-synchronization-and-fencing.md`
- `docs/internal/05-vulkan-sync-and-frame-lifecycle.md`
- `docs/internal/04-api-to-backend-handoff.md`
- `src/renderer/src/vulkan/AGENTS.md`
