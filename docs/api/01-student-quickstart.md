# Student Quickstart Guide

## 1. Purpose & Audience
This guide is for students, hobbyists, and indie developers who are comfortable with Rust but new to engine and rendering workflows. The goal is to get from clone to first rendered frame, then to first model load, in under 30 minutes.

## 2. Where This Fits in Engine Flow

> **renderer compatibility path** (good for demos, examples, and smoke testing):
choose example binary -> create `EventLoop` and window -> `Renderer::new(...)` -> `take_startup_scene()` (or build your own `Scene`) -> per-event `update_input(...)` -> per-frame `render_scene(...)` or explicit frame API.

> **current app-owned path** (recommended for custom apps):
custom app crate -> root `engine` facade helpers -> `Renderer::route_platform_input(...)` -> app-owned input/events/camera/frame update -> caller-provided `CameraView` render API. See [15-app-owned-loop.md](15-app-owned-loop.md).

Project quickstart path:
author/validate project package data -> run the root launcher with `cargo run -- --project <path>` -> use `--headless --capture_target draw` for validation captures.

## 3. Key Concepts
- Use the root launcher for data-driven projects: `cargo run -- --project apps/dungeon_dogfood/engine.project.toml`.
- Use example binaries from the `renderer` crate to learn the facade and diagnose renderer behavior.
- Use app crates under `apps/<name>` for custom Rust application behavior. App crates may use `engine::prelude` when they want to own input, events, frame clock, and camera state.
- Facade-first API boundary:
  - `Renderer` owns rendering runtime, assets, capture, resize, and platform/UI side effects.
  - App-owned loops can own input dispatch, lifecycle events, frame clock, and camera state.
  - `Scene` owns runtime graph content to draw.
  - `AssetManager` loads models/textures/environments.
- Two frame styles (renderer compatibility path):
  - one-call: `render_scene(...)`
  - explicit: `begin_frame(...)` -> `render_scene_in_frame(...)` -> `end_frame(...)`
- App-owned render path: `render_scene_with_view(...)` or `render_scene_headless_with_view(...)` with a caller-provided `CameraView`
- Frame outcomes are explicit:
  - `FrameRenderOutcome::Rendered`
  - `FrameRenderOutcome::SkippedResizePending`
  - `FrameRenderOutcome::SubmittedNotPresented`
  - `FrameRenderOutcome::PresentedSuboptimal`
- Model loading can be:
  - sync (`load_model`) for easiest first integration
  - deferred (`request_model_load` + `poll_model_load`) for non-blocking loads

## 4. Code Walkthrough
Snippet Type: Real
```bash
# Run renderer diagnostic examples:
cargo run -p renderer --example api_test
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading

# Run the sample project through the root launcher:
cargo run -- --project apps/dungeon_dogfood/engine.project.toml

# Generate a standalone Rust support-crate app scaffold:
cargo run -p engine_pack -- new-app /tmp/engine-app --id app.example --name "Example App"
cargo check --manifest-path /tmp/engine-app/Cargo.toml

# Produce true headless draw-target validation evidence:
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/runtime-launcher/headless-draw

# Optional: custom environment map with api_test
cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
```

Snippet Type: Real (renderer compatibility path)
```rust
// Minimal explicit-frame loop shape (from src/renderer/examples/api_test.rs)
let mut renderer = Renderer::new(config.clone(), &window)?;
let mut scene = renderer.take_startup_scene().unwrap_or_else(Scene::new);

let mut frame = renderer.begin_frame(&window)?;
let outcome = renderer.render_scene_in_frame(&mut frame, &mut scene)?;
renderer.end_frame(frame)?;
```

Snippet Type: Real
```rust
// First model load path (from src/renderer/examples/common/mod.rs)
let mut scene = Scene::new();
let fragment = {
    let mut assets = renderer.assets();
    assets.load_model("src/renderer/src/assets/DamagedHelmet.glb")?
};
scene.merge_fragment(None, fragment)?;
```

Snippet Type: Pseudocode
```text
Custom app structure idea:
  apps/my_app/
    Cargo.toml
    src/main.rs        // window + event loop + renderer ownership
    src/game_state.rs  // gameplay/editor state
    src/scene_setup.rs // initial scene + model loads

Renderer compatibility path frame tick:
  1) ingest OS events
  2) renderer.update_input(...)
  3) mutate game/scene state
  4) render exactly one frame

Current app-owned frame tick (recommended for custom apps):
  1) renderer.route_platform_input(...)
  2) queue uncaptured input into app InputSystem
  3) dispatch app input and emit app lifecycle/events
  4) update app camera/player state
  5) render with CameraView
```

## 5. Best Practices
- Start from `api_test` to understand ownership and error handling, then move to `demo_model_load` and `demo_async_loading`.
- Keep one frame API style per tick; do not mix single-call and explicit frame APIs in the same frame.
- Handle render outcomes exhaustively and keep the loop alive while resize is pending.
- Begin with sync model load for correctness, then adopt deferred tickets for larger assets.
- Keep asset loading code separate from draw submission code so failures are easier to isolate.
- Keep custom Rust behavior in app crates under `apps/<name>` or start from `engine_pack new-app` for a standalone support-crate scaffold. Dynamic Rust hot reload, production scripting runtime scheduling, package-level script assets, runtime physics scene loading, root-runtime audio playback, production audio mixing/spatialization/streaming, broad dogfood migration to project manifests, and renderer-window generated app templates are deferred.
- For app-owned gameplay/camera loops, follow the `dungeon_dogfood` ownership shape before introducing a larger runtime abstraction.

## 6. Gotchas & Failure Modes
- Running `cargo run` at workspace root requires `--project <path>` for the launcher.
- Running the wrong target is still a common startup miss: use `cargo run -- --project ...` for project data, `cargo run -p renderer --example ...` for diagnostics, and `cargo run -p <app>` for custom app crates.
- Missing Vulkan runtime/driver support causes renderer initialization failure.
- If shader compilation is enabled and tools are missing, startup can fail (`glslc` or `glslangValidator` not found).
- Calling facade APIs in invalid order can produce `RendererError::InvalidState`.
- Deferred loads stay `Pending` if your app never pumps progress (render loop/pump path not advancing).

## 7. Debugging Playbook
- If the app does not launch rendering:
  - Check command target first: `cargo run -- --project apps/dungeon_dogfood/engine.project.toml` for the project launcher, or `cargo run -p renderer --example api_test` for renderer diagnostics.
- If initialization fails immediately:
  - Check Vulkan runtime/driver setup and then re-run with logs:
  - `RUST_LOG=debug cargo run -p renderer --example api_test`
- If model does not appear:
  - Confirm asset path exists and `scene.merge_fragment(...)` succeeds.
  - Confirm you are rendering the same `Scene` you modified.
- If async loads never complete:
  - Log `LoadStatus` transitions and ensure per-frame progression is happening.
- If resizing causes apparent freeze:
  - Expect temporary `SkippedResizePending` outcomes while resize is being handled.

## 8. Cross-Module Links
- Canonical facade loop: `src/renderer/examples/api_test.rs`
- Demo scenarios: `src/renderer/examples/common/mod.rs`
- Async loading example: `src/renderer/examples/demo_async_loading.rs`
- Renderer API docs: `docs/api/02-renderer-lifecycle-and-frame-api.md`
- Runtime launcher docs: `docs/api/11-runtime-project-launcher.md`
- Scene workflow docs: `docs/api/03-scene-graph-and-fragment-workflows.md`
- Asset workflow docs: `docs/api/04-assets-sync-deferred-and-handles.md`

## 9. Standard References
- Rust install and toolchain: https://www.rust-lang.org/tools/install
- Vulkan SDK/runtime guidance: https://vulkan.lunarg.com/
- winit docs: https://docs.rs/winit/latest/winit/
- Vulkan Guide: https://github.khronos.org/Vulkan-Site/guide/latest/
- Engine baseline reference: https://github.com/SaschaWillems/Vulkan-glTF-PBR

## 10. See Also
- `docs/api/00-index.md`
- `docs/api/02-renderer-lifecycle-and-frame-api.md`
- `docs/api/03-scene-graph-and-fragment-workflows.md`
- `docs/api/04-assets-sync-deferred-and-handles.md`
- `docs/internal/00-index.md`
