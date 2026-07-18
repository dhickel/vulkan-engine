# Renderer Package Agent Guide (`src/renderer`)

Use this guide for package-level renderer work. Use child module guides for subsystem implementation details.

## Package Role

`renderer` owns:

- public facade API (`src/renderer/src/api/`)
- runtime examples (`src/renderer/examples/`)
- scene ownership and submission build
- Vulkan frame/render orchestration
- rendergraph pass execution

Primary entrypoint type exports are in `src/renderer/src/lib.rs` and `src/renderer/src/api/mod.rs`.

## Current Runtime Path

1. Example constructs `Renderer`.
2. App updates input and scene each frame.
3. Scene emits `RenderSubmission`.
4. `VkRender` executes `PrepareTargets -> Shadow -> Skybox -> Geometry -> PresentCopy -> Imgui -> DebugCapture -> TerminalPresent`, then submits/presents.

## Documentation Routing

- API index: `docs/api/00-index.md`
- Internal index: `docs/internal/00-index.md`
- Facade lifecycle and frame API: `docs/api/02-renderer-lifecycle-and-frame-api.md`
- Scene workflows: `docs/api/03-scene-graph-and-fragment-workflows.md`
- Asset workflows: `docs/api/04-assets-sync-deferred-and-handles.md`
- Render hooks: `docs/api/05-render-hooks-and-extension-points.md`
- API-to-backend handoff: `docs/internal/04-api-to-backend-handoff.md`
- Rendergraph internals: `docs/internal/07-rendergraph-dependencies-and-aliasing.md`

Module guides:

- Data/cache/scene internals: `src/renderer/src/data/AGENTS.md`
- Vulkan internals: `src/renderer/src/vulkan/AGENTS.md`
- Shader lineage/contracts: `src/renderer/src/shaders/AGENTS.md`

## High-Risk Areas

- `src/renderer/src/vulkan/vk_render.rs`: highest blast radius orchestration.
- `src/renderer/src/data/data_cache.rs`: handle validity and lifetime-sensitive caches.
- render-path correctness is sensitive to descriptor/pipeline binding order, especially scene binding 5 for the frame-local directional shadow map.
- material draw records must remain by-value copies made while the texture-cache lock is held; do not reintroduce cache-owned raw pointers.

## Working Rules

- Keep stable handle contracts (slot + generation) intact unless deliberately migrating all consumers.
- Treat `.spv` artifacts and GLSL sources as paired assets.
- Prefer small scoped edits and validate with `cargo check -p renderer --examples`.
- If docs and code disagree, treat code as logical truth and record the divergence.

## Runtime Commands

- `cargo run -p renderer --example api_test`
- `cargo run -p renderer --example api_test -- --env <path>`
- `cargo run -p renderer --example demo_pbr`
- `cargo run -p renderer --example demo_unlit`
- `cargo run -p renderer --example demo_model_load`
- `cargo run -p renderer --example demo_async_loading`
- `cargo run -p renderer --example capture_culling -- --headless --culling=on`
- `cargo run -p renderer --example capture_shadows -- --headless`

## Headless Capture Validation

- Use the project skill `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` for renderer changes that need visual proof.
- Prefer timeout-bound `--headless` frame captures over desktop screenshots; agents do not fully control the user's windowing environment.
- Agents may create focused validation scenes or examples under `src/renderer/examples/capture_tests/` when existing examples are too ambiguous.
- Use `.internal-dev/headless_capture_tests/` for temporary scene specs, notes, and investigation artifacts.
- Use `.internal-dev/captures/` for generated PNG/JSON capture evidence.

## Runtime Debug Capture

- Use launch recording flags to capture timing JSONL without opening the debug menu.
- Baseline diagnosis command (recommended default):
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl`
- Baseline `api_test` with environment:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/api_test-timing.jsonl`
- Engine startup can take ~20-30 seconds; keep timeout at `60s` unless a task specifically needs longer.
- Agents should default timing-report output to `.internal-dev/debug_reports/` to avoid polluting project root.
- Adjust as needed:
- `--record_debug=<seconds>` to capture longer/shorter windows.
- `--record_debug_interval=<ms>` to trade sample density vs. file size.
- `--record_debug_path=<path>` to force a known output file path.
- For complete argument reference, see `docs/api/07-engine-arguments.md`.
