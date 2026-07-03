# Configuration Reference

> Source: [`src/renderer/src/api/config.rs`](../src/renderer/src/api/config.rs) — no legacy docs consulted.

## RendererConfig

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
    pub headless: bool,              // UNSUPPORTED — returns error if true
    pub asset_policy: AssetPolicyConfig,
}
```

Defined at [`config.rs:127`](../src/renderer/src/api/config.rs:127). Use `RendererConfig::default()` for sensible defaults.

### Shader Compilation

When `compile_shaders: true`, the engine invokes `glslc` (or `glslangValidator`) to compile `.glsl` → `.spv` at startup. This requires the tools to be in `PATH`. Pre-compiled `.spv` files are included in the repository; use this only during shader development.

### Validation Layers

`validation_layer: true` enables `VK_LAYER_KHRONOS_validation` with `VK_EXT_debug_utils`. Significant performance impact — development only.

## DebugRuntimeMode

```rust
pub enum DebugRuntimeMode {
    Default,   // full PBR + IBL pipeline
    TestPbr,   // PBR without some debug overhead
    TestUnlit, // unlit material pipeline (no lighting)
}
```

Defined at [`config.rs:9`](../src/renderer/src/api/config.rs:9). Affects shader variant selection and material evaluation.

## VisualTuning

```rust
pub struct VisualTuning {
    pub exposure: f32,           // default: 4.5
    pub gamma: f32,              // default: 2.2
    pub ibl_ambient_scale: f32,  // default: 1.0
}
```

Defined at [`config.rs:69`](../src/renderer/src/api/config.rs:69). These are passed to the `EnvironmentUBO` each frame — changes take effect immediately.

## AssetPolicyConfig

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

Defined at [`config.rs:48`](../src/renderer/src/api/config.rs:48). Manifest files are TOML sidecars (e.g., `model.glb.meta`) that override texture sampling parameters.

## Resolution Chain for Texture Parameters

When loading a texture, the engine resolves `FilterMode` and `WrapMode` via a priority chain ([`asset_manifest.rs:390`](../src/renderer/src/data/asset_manifest.rs:390)):

1. **API override** — `TextureLoadOptions` passed to the load call
2. **Manifest sidecar** — `.meta` file if present and `manifest_mode` allows
3. **Filename heuristics** — e.g., `_normal` suffix → linear filtering, `_roughness` → linear (if `allow_filename_heuristics` is true)
4. **Engine defaults** — `FilterMode::LinearMipmapLinear`, `WrapMode::Repeat`

## See Also

- [02-renderer.md](02-renderer.md) — Renderer lifecycle
- [04-assets.md](04-assets.md) — asset loading and manifests
- [src/renderer/src/data/asset_manifest.rs](../src/renderer/src/data/asset_manifest.rs) — manifest format
