# Rendergraph — Pass Orchestration

> Source: [`src/renderer/src/rendergraph/mod.rs`](../../src/renderer/src/rendergraph/mod.rs), [`src/renderer/src/rendergraph/passes/`](../../src/renderer/src/rendergraph/passes/).

## RenderPassNode Trait

The current rendergraph is a fixed sequential pass list. Every pass receives the frame submission, current `VkFrame`, and mutable `VkRenderCore` through `RenderGraphContext`:

```rust
pub trait RenderPassNode {
    fn name(&self) -> &'static str;
    fn execute(&self, ctx: &mut RenderGraphContext) -> Result<(), String>;
}
```

The module is public only with the `advanced-interop` feature. That exposure is alpha-unstable and does not provide safe public pass registration; mutable core access can violate synchronization and descriptor contracts.

## Default Pass Chain

```rust
RenderGraph::new(vec![
    Box::new(PrepareTargetsPass),
    Box::new(ShadowPass),
    Box::new(SkyboxPass),
    Box::new(GeometryPass),
    Box::new(PresentCopyPass),
    Box::new(ImguiPass),
    Box::new(DebugCapturePass),
    Box::new(TerminalPresentPass),
])
```

Passes execute in this exact order. There is no dependency-derived scheduling or transient attachment alias planner.

### 1. PrepareTargetsPass

Transitions the offscreen draw/depth targets into writable attachment layouts when the submission has draw work.

### 2. ShadowPass

For one optional scene directional light, resolves opaque draw objects, fits a conservative light-space orthographic volume, clears/records the current frame slot's 2048² D32 map, and transitions it to shader-read layout. PBR geometry samples this map from scene set 0 binding 5 with fixed 3×3 comparison PCF.

### 3. SkyboxPass

Renders the selected environment into the draw color target.

### 4. GeometryPass

Resolves mesh handles and copies material binding fields under cache guards into `CopiedMaterialDrawRecord`. It partitions opaque/masked/blended lists, binds scene/skin/material descriptor sets, pushes model and GPU-address data, and records indexed PBR or unlit draws. Blended objects are sorted back-to-front.

### 5. PresentCopyPass

Copies/blits the offscreen draw image into the present image, or prepares a present color attachment when no draw target path ran. Its outgoing state is suitable for optional UI and capture.

### 6. ImguiPass

Records ImGui dynamic rendering only when the submission requests UI and a context exists. Headless rendering has no ImGui context, so this pass records no unmatched begin/end rendering region.

### 7. DebugCapturePass

Consumes due draw/present capture requests and records readback commands before submission.

### 8. TerminalPresentPass

Transitions a windowed present image to `PRESENT_SRC_KHR`. It is a no-op for headless offscreen present images.

## Adding or Reordering Passes

Internal pass changes must update all of the following together:

1. fixed order in `RenderGraph::default_graph()`;
2. incoming/outgoing image layouts and stage/access masks;
3. scene descriptor and shader ABI when a pass adds sampled resources;
4. frame-transaction failure behavior for any new fallible recording path;
5. focused validation-layer headless capture evidence.

Do not treat `RenderGraph::new` as a safe application extension API. It accepts unrestricted pass objects but performs no resource declaration, topological sort, hazard analysis, or synchronization validation.

## Resource Aliasing

Rendergraph resource aliasing is not implemented. Frame draw/depth, present, and directional shadow images have explicit owners and frame-slot lifetimes. Any future alias planner requires resource declarations, lifetime analysis, and hazard-aware scheduling before allocations can be shared safely.

## See Also

- [02-renderer-internals.md](02-renderer-internals.md) — frame lifecycle
- [07-rendergraph-dependencies-and-aliasing.md](07-rendergraph-dependencies-and-aliasing.md) — current transition ownership and future graph direction
- [08-shaders.md](08-shaders.md) — shader and descriptor contracts
- [`src/renderer/src/rendergraph/mod.rs`](../../src/renderer/src/rendergraph/mod.rs) — implementation
