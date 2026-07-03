# Current State Analysis

## Verified Inputs

- Roadmap Sprint 10 asks for advanced rendering opt-in, feature-gated unstable interop, and named extension points over raw Vulkan handles.
- Sprint 09 target design keeps beginner facade small and reserves advanced rendergraph access for Sprint 10.
- Current dirty worktree contains Sprint 09-looking edits in renderer API/examples/tests and `.idea/engine.iml`; this Sprint 10 planning task did not mutate them.

## Current Advanced Surface

- `src/renderer/src/api/advanced.rs:5-14` exposes `unsafe fn renderer_core_mut(renderer: &mut Renderer) -> &mut VkRenderCore`.
- `src/renderer/src/api/mod.rs:10-11` gates `api::advanced` behind `advanced-interop`.
- `src/renderer/src/lib.rs:14-17` makes `rendergraph` private by default and public when `advanced-interop` is enabled.
- `src/renderer/Cargo.toml` defines `advanced-interop = []` and no default features.

## Current Safe Extension Surface

- `RenderHookContext` exposes frame index, viewport size, and optional depth texture, with private construction.
- Hook invocation catches panics and wraps hook errors into `HookError::Invocation`.
- Debug UI supports custom debug views through `DebugViewDescriptor`, `DebugViewCallback`, and renderer registration methods.
- Frame/debug capture paths already exist and are documented for engine-owned headless draw validation.

## Current Rendergraph Contract

- `RenderGraph` stores ordered `Vec<Box<dyn RenderPassNode>>`.
- `RenderGraph::default_graph()` currently orders prepare targets, skybox, geometry, present copy, imgui, debug capture, terminal present.
- `RenderGraph::execute()` iterates linearly and records pass timing; it does not validate declared read/write resources or perform topological scheduling.
- Internal docs explicitly say dependency-derived scheduling and aliasing are roadmap direction, not current behavior.

## Docs Drift

- `docs/api/05-render-hooks-and-extension-points.md` largely matches the safer API-level hook contract and warns against backend resource ownership.
- `docs/api/05-hooks.md` is stale: it claims hooks are the primary path for custom Vulkan commands/post-processing and lists obsolete `HookError` variants. It must be corrected or demoted to a compatibility/redirect page.
- `docs/api/00-index.md` already classifies advanced interop as feature-gated, but Sprint 10 should add sharper guidance about unstable alpha status and deferred rendergraph custom-pass work.

## Architecture Fit

- Default-safe hooks/debug views fit the facade: they do not expose backend mutation.
- Existing unsafe `renderer_core_mut` is acceptable only as an explicit escape hatch because it bypasses facade invariants.
- Public rendergraph under `advanced-interop` is high-risk because pass nodes receive `&mut VkRenderCore`; documentation and tests must avoid implying stability.
- A minimal named advanced surface can fit only if it is read-only/descriptor-oriented or internally validates ordering/resource ownership.

## Validation Blind Spots

- Compile checks alone do not prove visual/capture correctness if a change touches readback or frame capture.
- Default `cargo check -p renderer --examples` is needed to prove beginner examples do not depend on `advanced-interop`.
- Feature-enabled checks are needed because `rendergraph` and `api::advanced` compile only under `advanced-interop`.
- Docs can easily drift across duplicate hook chapters; validators must inspect both canonical and older docs.

## Residuals To Track If Not Fixed In Sprint

- Production custom rendergraph pass registration remains deferred until pass resource declarations and synchronization contracts exist.
- Material/shader override registration may remain deferred if manifest validation and shader asset contracts are not ready.
- Read-only frame/depth/debug texture exposure may remain limited to existing handles if safe lifetime and usage contracts are unclear.
