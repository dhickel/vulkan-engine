# Rendering API Handbook (Alpha)

This handbook documents the current renderer API for students, hobbyists, and indie developers integrating the engine as a library.

Audience:
- Rust users who are new to renderer architecture.
- Intermediate graphics learners who want practical Vulkan-backed API usage.
- Engine contributors who need the facade contract and internal mental model in one place.

Current API status (as implemented):
- Facade-first runtime API is available and used by all example binaries.
- Public types are exported from `renderer` (`src/renderer/src/api/mod.rs`).
- Root `engine` binary is a migration stub; runtime entrypoints are `renderer` examples.

Primary runtime commands:
```bash
cargo run -p renderer --example api_test
cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading
```

Build/test baseline commands:
```bash
cargo check
cargo check -p renderer --examples
cargo test -p renderer --lib --no-run
```

## Start Here

If this is your first pass, read in this order:
1. `01_quickstart_facade_bootstrap.md`
2. `02_renderer_lifecycle_and_frame_api.md`
3. `03_scene_graph_and_fragment_workflows.md`
4. `04_assets_sync_deferred_and_handles.md`
5. `08_examples_dogfooding_playbook.md`

Then use these for deeper architecture/debug work:
1. `09_internal_render_pipeline_mental_model.md`
2. `10_sync_and_performance_gotchas.md`
3. `11_alpha_limits_and_roadmap.md`

## Document Index

- `01_quickstart_facade_bootstrap.md`
- `02_renderer_lifecycle_and_frame_api.md`
- `03_scene_graph_and_fragment_workflows.md`
- `04_assets_sync_deferred_and_handles.md`
- `05_environment_and_skybox_runtime.md`
- `06_render_hooks_and_extension_points.md`
- `07_error_model_and_recovery_patterns.md`
- `08_examples_dogfooding_playbook.md`
- `09_internal_render_pipeline_mental_model.md`
- `10_sync_and_performance_gotchas.md`
- `11_alpha_limits_and_roadmap.md`

## External References

- Vulkan Guide: https://github.khronos.org/Vulkan-Site/guide/latest/
- Vulkan Specification: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/index.html
- Khronos glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Sascha Willems Vulkan glTF PBR baseline:
  - https://github.com/SaschaWillems/Vulkan-glTF-PBR
