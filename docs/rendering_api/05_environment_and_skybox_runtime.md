# 05 - Environment and Skybox Runtime

This chapter covers public environment APIs and what to expect during runtime switching.

## Public Environment Surface

Types:
- `EnvironmentSource`
  - `EquirectFile(PathBuf)`
  - `CubemapDir(PathBuf)`
- `EnvironmentState`
  - `Unloaded`
  - `Loading`
  - `Ready`
  - `Failed(AssetError)`

Methods:
- `load_environment(source) -> Result<EnvironmentHandle, AssetError>`
- `environment_state(env) -> Result<EnvironmentState, AssetError>`
- `default_environment() -> EnvironmentHandle`
- `Scene::set_skybox(env)`
- `Renderer::environment_runtime_status()`

## Typical Flow

1. Load environment asset:
```rust
use renderer::EnvironmentSource;

let env = {
    let mut assets = renderer.assets();
    assets.load_environment(EnvironmentSource::EquirectFile("path/to/file.hdr".into()))?
};
```
2. Set it on the scene:
```rust
scene.set_skybox(env);
```
3. Poll state/debug transition:
```rust
let state = {
    let assets = renderer.assets();
    assets.environment_state(env)?
};
let rt = renderer.environment_runtime_status();
```

## Runtime Semantics (Important)

- Environment load registers source data and skybox asset.
- IBL generation (irradiance/prefilter) can happen on first activation.
- During activation, state may report `Loading` before `Ready`.
- On preparation failure, runtime keeps prior active environment and records failure.

Implication for game UX:
- Treat environment switch as potentially visible transition.
- Use status fields in debug UI to avoid "why didn’t it switch?" confusion.

## Input Format Notes

- `EquirectFile`: HDR/equirectangular input path.
- `CubemapDir`: directory mode with engine-expected cubemap loading format.

If unsupported format/path is provided, load returns typed `AssetError`.

## Learn More

- Public asset API source: `src/renderer/src/api/assets.rs`
- Runtime status source: `src/renderer/src/api/renderer.rs`
- PBR/IBL background:
  - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
  - https://learnopengl.com/PBR/IBL/Specular-IBL
