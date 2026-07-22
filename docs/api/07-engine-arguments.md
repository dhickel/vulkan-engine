# Engine Configuration and Arguments

## 1. Purpose & Audience
This page is the canonical reference for `RendererConfig` and renderer-example launch arguments. Use it for understanding engine configuration, running diagnostics, automation smoke runs, or reproducing rendering issues from the command line.

For the root project launcher, see [11-runtime-project-launcher.md](11-runtime-project-launcher.md). The root launcher runs data-driven projects with `cargo run -- --project <path>` and owns the alpha headless draw capture path. Renderer examples remain examples/diagnostics, not the primary alpha app runtime path.

## 2. Where This Fits in Engine Flow
`RendererConfig` is passed to `Renderer::new(config, &window)` at initialization time and controls Vulkan backend settings, visual tuning, asset policies, shader compilation, and headless mode. Example launch arguments are parsed at renderer example startup before `Renderer::new(...)` enters the per-frame loop. Debug-record arguments configure/trigger the same timing recorder used by the in-engine debug UI.

## 3. Key Concepts

### RendererConfig

```rust
pub struct RendererConfig {
    pub window_width: u32,           // default: 1920
    pub window_height: u32,          // default: 1080
    pub app_name: String,            // window title, default: "renderer"
    pub validation_layer: bool,      // enable Vulkan validation layers, default: false
    pub shader_debug_mode: DebugRuntimeMode,  // default: DebugRuntimeMode::Default
    pub compile_shaders: bool,       // compile .glsl to .spv at startup, default: false
    pub preload_startup_scene: bool, // preload default PBR scene, default: true
    pub visual_tuning: VisualTuning,
    pub headless: bool,              // set by Renderer::new_headless for offscreen validation
    pub asset_policy: AssetPolicyConfig,
}
```

Defined at [`src/renderer/src/api/config.rs`](../../src/renderer/src/api/config.rs#L127). Use `RendererConfig::default()` for sensible defaults. Windowed apps should call `Renderer::new(config, window)` with `headless = false`; offscreen validation should call `Renderer::new_headless(config)` and render with `render_scene_headless(...)`.

`RendererConfig::headless` is controlled by `Renderer::new_headless`. For visual validation, use the root project launcher with `--headless --capture_target draw`; a desktop screenshot is not renderer-owned proof. Windowed examples can still request present/draw captures through the frame-capture API, but Sprint validation evidence must use true headless draw capture when visible behavior changes.

### Shader Compilation

When `compile_shaders: true`, the engine invokes `glslc` (or `glslangValidator`) to compile `.glsl` → `.spv` at startup. This requires the tools to be in `PATH`. Pre-compiled `.spv` files are included in the repository; use this only during shader development.

### Validation Layers

`validation_layer: true` enables `VK_LAYER_KHRONOS_validation` with `VK_EXT_debug_utils`. Significant performance impact — development only.

### DebugRuntimeMode

```rust
pub enum DebugRuntimeMode {
    Default,   // full PBR + IBL pipeline
    TestPbr,   // PBR without some debug overhead
    TestUnlit, // unlit material pipeline (no lighting)
}
```

Defined at [`src/renderer/src/api/config.rs`](../../src/renderer/src/api/config.rs#L9). Affects shader variant selection and material evaluation.

### VisualTuning

```rust
pub struct VisualTuning {
    pub exposure: f32,           // default: 4.5
    pub gamma: f32,              // default: 2.2
    pub ibl_ambient_scale: f32,  // default: 1.0
}
```

Defined at [`src/renderer/src/api/config.rs`](../../src/renderer/src/api/config.rs#L69). These are passed to the `EnvironmentUBO` each frame — changes take effect immediately.

### AssetPolicyConfig

```rust
pub struct AssetPolicyConfig {
    pub manifest_mode: AssetManifestMode,     // default: BestEffort
    pub allow_filename_heuristics: bool,      // default: true
    pub compression: CompressionConfig,       // default: Disabled + quality 50
}

pub enum AssetManifestMode {
    Disabled,   // ignore .meta files entirely
    BestEffort, // use .meta if present, fall back to heuristics
    Strict,     // require .meta for every asset, error on missing
}
```

Defined at [`src/renderer/src/api/config.rs`](../../src/renderer/src/api/config.rs#L48). Manifest files are TOML sidecars (e.g., `model.glb.meta`) that override texture sampling parameters.

### Resolution Chain for Texture Parameters

When loading a texture, the engine resolves `FilterMode` and `WrapMode` via a priority chain:

1. **API override** — `TextureLoadOptions` passed to the load call
2. **Manifest sidecar** — `.meta` file if present and `manifest_mode` allows
3. **Filename heuristics** — e.g., `_normal` suffix → linear filtering, `_roughness` → linear (if `allow_filename_heuristics` is true)
4. **Engine defaults** — `FilterMode::LinearMipmapLinear`, `WrapMode::Repeat`

### Renderer Example Launch Arguments

- Pass example arguments after `--` when using Cargo.
- The current shared parser lives in `src/renderer/examples/common/mod.rs`.
- The same argument set applies across runtime examples (`demo_pbr`, `demo_unlit`, `demo_model_load`, `demo_async_loading`, `api_test`).
- Custom Rust behavior belongs in app crates under `apps/<name>` and runs with `cargo run -p <app>`.
- For off-workspace app-control scaffolding, use `engine_pack new-app`; it generates a support-crate app and does not add renderer-window integration or runtime reload.
- Dynamic Rust hot reload, production scripting runtime scheduling, package-level script assets, runtime physics scene loading, root-runtime audio playback, production audio mixing/spatialization/streaming, broad dogfood migration to project manifests, and renderer-window generated app templates are deferred.
- `--record_debug=<seconds>` starts capture immediately at launch.
- `--record_debug_interval` and `--record_debug_path` can be supplied with or without `--record_debug`.
- If interval/path are supplied without `--record_debug`, values are configured but capture is not auto-started.

## 4. Code Walkthrough

Snippet Type: Real
```rust
// src/renderer/examples/common/mod.rs
if let Some(value) = arg.strip_prefix("--record_debug=") {
    options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
}
if let Some(value) = arg.strip_prefix("--record_debug_interval=") {
    options.record_debug_interval_ms =
        Some(parse_positive_u64("--record_debug_interval", value)?);
}
if let Some(value) = arg.strip_prefix("--record_debug_path=") {
    options.record_debug_path = Some(value.to_string());
}
```

### Argument Reference

| Argument | Value | Meaning | Notes |
|---|---|---|---|
| `--env` / `--env=<path>` | file path | Environment map path for skybox/environment loading | Typically used with `api_test` |
| `--record_debug` / `--record_debug=<seconds>` | integer `>= 1` | Starts timing JSONL capture immediately | Example: `10` |
| `--record_debug_interval` / `--record_debug_interval=<ms>` | integer `>= 1` | Snapshot interval in milliseconds | Example: `50` |
| `--record_debug_path` / `--record_debug_path=<path>` | file path | Output JSONL path override | If omitted, default timestamped filename is used |

## 5. Best Practices
- Use `RendererConfig::default()` for initial setup; tune only the fields you need to change.
- Windowed apps use `Renderer::new(config, &window)`. Do not set `config.headless = true` manually.
- For automated diagnosis, use:
  `--record_debug=10 --record_debug_interval=50`
- Keep run timeout at `60s` because startup commonly takes ~20-30 seconds.
- Use an explicit `--record_debug_path` in scripts so downstream parsing can target a known file.
- Contributor/agent default path: `.internal-dev/debug_reports/<example>-timing.jsonl`.
- For offscreen validation evidence, use `Renderer::new_headless(config)` and the root project launcher `--headless` path.

## 6. Gotchas & Failure Modes
- Missing values (for example bare `--record_debug`) are treated as argument errors and example startup exits early.
- Zero values are invalid for record duration/interval.
- Passing `--record_debug_interval` without `--record_debug` does not start capture.
- If shader compilation is enabled and tools are missing, startup can fail (`glslc` or `glslangValidator` not found).
- Validation layers have significant performance impact — development only.
- Setting `config.headless = true` on the windowed `Renderer::new(config, window)` path is an error; use `Renderer::new_headless(config)` instead.

## 7. Debugging Playbook
- Basic capture on startup:
  `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50`
- Capture with known output path:
  `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/demo_pbr-timing.jsonl`
- `api_test` with custom environment:
  `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50`

Root project launcher visual validation:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/runtime-launcher/headless-draw
```

This command is intentionally rooted at `cargo run --`, not `cargo run -p renderer --example ...`, because the root binary is the data-driven project launcher.

Use `--capture_target draw` for validation evidence when a change affects rendered output. Desktop screenshot capture does not count because it records the compositor/windowing environment rather than the renderer-owned draw target.

## 8. Cross-Module Links
- Configuration structs: `src/renderer/src/api/config.rs`
- Shared argument parser: `src/renderer/examples/common/mod.rs`
- Example with environment arg usage: `src/renderer/examples/api_test.rs`
- Debug recorder internals: `src/renderer/src/debug_ui/mod.rs`
- Renderer launch hooks for debug recording: `src/renderer/src/api/renderer.rs`
- Asset manifest format: `src/renderer/src/data/asset_manifest.rs`

## 9. Standard References
- Cargo run argument forwarding: https://doc.rust-lang.org/cargo/commands/cargo-run.html
- Vulkan validation layers: https://github.khronos.org/Vulkan-Site/guide/latest/validation_layers.html
- Vulkan Guide index: https://github.khronos.org/Vulkan-Site/guide/latest/

## 10. See Also
- [`docs/api/00-index.md`](00-index.md)
- [`docs/api/01-student-quickstart.md`](01-student-quickstart.md)
- [`docs/api/02-renderer-lifecycle-and-frame-api.md`](02-renderer-lifecycle-and-frame-api.md)
- [`src/renderer/AGENTS.md`](../../src/renderer/AGENTS.md)
- [`AGENTS.md`](../../AGENTS.md)
