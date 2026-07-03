# Rendergraph — Pass Orchestration

> Source: [`src/renderer/src/rendergraph/mod.rs`](../src/renderer/src/rendergraph/mod.rs), [`src/renderer/src/rendergraph/passes/`](../src/renderer/src/rendergraph/passes/) — no legacy docs consulted.

## RenderPassNode Trait

All passes implement this trait at [`rendergraph/mod.rs:31`](../src/renderer/src/rendergraph/mod.rs:31):

```rust
pub trait RenderPassNode {
    fn name(&self) -> &'static str;
    fn execute(
        &mut self,
        ctx: &mut RenderGraphContext,
        core: &VkRenderCore,
        frame_idx: usize,
    ) -> Result<(), RendererError>;

    // Attachment transitions
    fn input_attachment_transitions(&self) -> Vec<AttachmentTransition>;
    fn output_attachment_transitions(&self) -> Vec<AttachmentTransition>;
}
```

Each pass records Vulkan commands into the current frame's command buffer (accessible via `core.current_command_buffer(frame_idx)`).

## Default Pass Chain

```rust
// rendergraph/mod.rs
RenderGraph::new(vec![
    Box::new(PrepareTargetsPass),   // step 1
    Box::new(SkyboxPass),           // step 2
    Box::new(GeometryPass),         // step 3
    Box::new(PresentCopyPass),      // step 4
    Box::new(ImguiPass),            // step 5
])
```

Passes execute sequentially — no parallel execution or automatic dependency resolution. Attachment transitions declare layout changes between passes.

### 1. PrepareTargetsPass

Sets up the color and depth attachments for the frame:
- Transitions swapchain image to `COLOR_ATTACHMENT_OPTIMAL`
- Creates/transitions depth buffer
- Clears color (to sky color or black) and depth (to 1.0)

### 2. SkyboxPass

Renders the environment cubemap as a full-screen skybox:
- Uses `skybox.vert` + `skybox.frag` shaders
- Samples the prefiltered environment cubemap
- Writes to the color attachment with depth test enabled (writes to far plane)

### 3. GeometryPass

The primary draw pass:
- Iterates `RenderSubmission::render_objects`
- For each `RenderObject`:
  - Binds the mesh vertex/index buffers (via buffer device address)
  - Binds the material descriptor set (textures + PBR params)
  - Sets push constants (`VkModelPushConsts` — model matrix, buffer addresses, joint count)
  - Issues `vkCmdDrawIndexed`
- Uses `pbr_base.vert` + `material_pbr.frag` (or `material_unlit.frag` in unlit mode)
- Supports GPU skinning via joint buffer + `joint_count` in push constants

### 4. PresentCopyPass

Copies the rendered color attachment to the swapchain image for presentation:
- Image layout transitions
- `vkCmdBlitImage` or `vkCmdCopyImage` (depending on format compatibility)

### 5. ImguiPass

Renders the imgui draw data:
- Uploads font atlas
- Records imgui draw commands (vertex/index buffers, texture bindings)
- Does not use `VkRenderPass` — uses dynamic rendering

## Adding Custom Passes

The `rendergraph` module is currently **private** (`mod rendergraph;` in `lib.rs`). Users cannot add custom passes through the public API. The `advanced-interop` feature gate exposes `Renderer::raw_core_mut() -> &mut VkRender` but this is documented as unsafe/internal-use-only.

To add a pass internally:
1. Create a struct implementing `RenderPassNode`
2. Add it to the `RenderGraph::new()` vec in the desired order
3. Ensure attachment transitions are correct — wrong transitions cause validation errors

## Attachment Aliasing

The rendergraph supports attachment aliasing (reusing the same GPU memory for different attachments across passes). This is configured via `AttachmentTransition` descriptors. Currently, the default pass chain doesn't alias heavily — each pass writes to the same color attachment.

## See Also

- [02-renderer-internals.md](02-renderer-internals.md) — where rendergraph fits in frame loop
- [08-shaders.md](08-shaders.md) — shaders used by each pass
- [src/renderer/src/rendergraph/mod.rs](../src/renderer/src/rendergraph/mod.rs) — implementation
