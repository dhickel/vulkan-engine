# Shader Agent Guide (`src/renderer/src/shaders`)

Use this guide for shader source ownership and Rust-side shader interface alignment.

## Directory Role

This directory stores active GLSL and precompiled SPIR-V (`.spv`) artifacts consumed by runtime shader loading.

Core path map is in `src/renderer/src/shaders/core_shader_manifest.txt`.

## Documentation Routing

- Internal index: `docs/internal/00-index.md`
- Rendering pipeline model: `docs/internal/01-rendering-pipeline-mental-model.md`
- Rendergraph dependencies: `docs/internal/07-rendergraph-dependencies-and-aliasing.md`
- Vulkan module guide: `src/renderer/src/vulkan/AGENTS.md`
- Renderer package guide: `src/renderer/AGENTS.md`

## Interface Contracts

- Keep GLSL stage interfaces aligned with Rust push constant and descriptor expectations.
- Descriptor set ordering must remain compatible with draw path binding order.
- Keep `.vert/.frag/.comp` changes synchronized with required `.spv` updates.

## Current Focus Areas

- PBR and unlit material paths (`material_pbr.frag`, `material_unlit.frag`)
- Environment/IBL generation (`env_irradiance_cube.frag`, `env_prefilter_cube.frag`, `gen_brd_flut.*`)
- Skybox path (`skybox.vert`, `skybox.frag`)

External conceptual baseline:

- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

## Working Rules

- Treat shader and pipeline/descriptor updates as one contract change.
- Validate any interface mutation against Vulkan pipeline/descriptor definitions.
- If docs and code diverge, treat code as logical truth and record the divergence.

## Validation

- `cargo check -p renderer`
- Run targeted example smoke when shader behavior changes.
