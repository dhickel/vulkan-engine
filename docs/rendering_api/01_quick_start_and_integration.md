# 01 - Quick Start and Integration

## Minimal run path (external API today)

Today, the stable external path is:
- `src/main.rs` -> `renderer::run()`.

Code example (external/public):
```rust
fn main() {
    renderer::run();
}
```

Best practice:
- Start from this path first; verify your Vulkan environment before deeper changes.

Learn more:
- Runtime entrypoint: `src/main.rs`
- Runtime loop orchestration: `src/renderer/src/lib.rs`

## Build and run commands

```bash
cargo check
cargo check -p renderer
cargo run
```

If you edit GLSL and want fresh SPIR-V:
```bash
cargo run -- --rebuild-shaders
```
or
```bash
ENGINE_REBUILD_SHADERS=1 cargo run
```

Best practice:
- Run `cargo check -p renderer` first when iterating render code.

Learn more:
- Shader compile hook: `src/renderer/src/vulkan/vk_render.rs:500`
- Compiler helper: `src/renderer/src/vulkan/vk_util.rs:987`

## Runtime debug selector (pipeline/material testing)

Use this to validate startup material routing:
```bash
cargo run -- debug_runtime testpbr
cargo run -- debug_runtime testunlit
cargo run -- --debug-runtime=testunlit
```

Code path:
- `parse_debug_runtime_mode` in `src/renderer/src/lib.rs`
- startup scenario logic in `src/renderer/src/scene/debug_scenarios.rs`

Best practice:
- Use `testunlit` to quickly isolate material/pipeline issues from PBR/IBL complexity.

Learn more:
- Debug scenario loader: `src/renderer/src/scene/debug_scenarios.rs`

## Validation layers during development

Current runtime sets validation off (`with_validation = false`).

Code example (in-tree/internal):
```rust
// src/renderer/src/lib.rs
match vk_render::VkRender::new(
    window_state,
    true, // enable validation during development
    runtime_flags.compile_shaders,
    runtime_flags.debug_runtime_mode,
)
```

Best practice:
- Enable validation while changing pass order, descriptors, transitions, or swapchain flow.

Learn more:
- Vulkan validation overview: https://github.khronos.org/Vulkan-Site/guide/latest/validation_overview.html

## Integration model into a larger engine

Current in-tree frame structure:
1. Process input into `InputManager`.
2. Update camera/controller.
3. Build `RenderSubmission` from `SceneWorld`.
4. Call `app.render(frame, &submission)`.

Code example (in-tree/internal):
```rust
state.input_manager.update();
state.app.core.window_state.controller.borrow_mut().update(dt);
state.scene_world.update_camera(view, proj, cam_pos);

let submission = state.scene_world.build_submission();
state.app.render(state.frame, &submission);
```

Best practice:
- Keep simulation state and render submission separate; submission should be frame-local and immutable.

Learn more:
- Redraw handler: `src/renderer/src/lib.rs` (`handle_redraw_requested`)
- Fixed timestep discussion: https://gafferongames.com/post/fix_your_timestep/
