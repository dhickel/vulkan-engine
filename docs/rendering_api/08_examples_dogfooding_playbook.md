# 08 - Examples Dogfooding Playbook

This chapter is the practical runtime validation guide for alpha dogfooding.

## Canonical Example Binaries

- `api_test`
- `demo_pbr`
- `demo_unlit`
- `demo_model_load`
- `demo_async_loading`

Run all:
```bash
cargo run -p renderer --example api_test
cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
cargo run -p renderer --example demo_pbr
cargo run -p renderer --example demo_unlit
cargo run -p renderer --example demo_model_load
cargo run -p renderer --example demo_async_loading
```

## What Each Example Validates

- `api_test`
  - Explicit frame lifecycle (`begin_frame`, `render_scene_in_frame`, `end_frame`)
  - Handles `FrameRenderOutcome` (`Rendered` vs `SkippedResizePending`) in redraw loop
  - Input + resize integration shape
  - Environment-source loading via `--env <path>`
- `demo_pbr`
  - Startup scene PBR path
- `demo_unlit`
  - Runtime debug mode forcing unlit material path
- `demo_model_load`
  - Sync model load + fragment merge + transform updates
- `demo_async_loading`
  - Deferred model load ticket request/poll/mount loop

## Headless/CI-style Smoke Pattern

In terminal-only environments, run bounded sessions:
```bash
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_pbr
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_unlit
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_model_load
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_async_loading
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test
RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr
```

Interpretation:
- Startup logs + no fatal errors before timeout = smoke pass.

## Compile/Test Gate

```bash
cargo check
cargo check -p renderer --examples
cargo test -p renderer --lib --no-run
```

## Dogfooding Checklist

1. Confirm app can run with startup scene and custom scene.
2. Confirm sync model load and deferred model load both work.
3. Confirm resize path remains stable during continuous rendering.
4. Confirm environment transitions and fallback behavior are understandable in logs/UI.
5. Confirm errors are surfaced clearly to gameplay-layer logging.
6. Confirm environment source formats used by your app (`.exr`, `.hdr`, strip, or faces) load through `EnvironmentSource::Auto` or explicit mode.

## Learn More

- Example sources: `src/renderer/examples/`
- Runtime risks/limits: `11_alpha_limits_and_roadmap.md`
