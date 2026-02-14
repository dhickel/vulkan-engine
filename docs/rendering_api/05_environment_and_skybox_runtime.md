# 05 - Environment and Skybox Runtime

This chapter covers public environment APIs and what to expect during runtime switching.

## Public Environment Surface

Types:
- `EnvironmentSource`
  - `Auto(PathBuf)` - auto-detect layout from aspect ratio (2:1 = equirect, 6:1 = strip)
  - `Equirectangular(PathBuf)` - equirectangular (2:1) HDR/EXR/image file
  - `CubeStrip(PathBuf)` - horizontal 6:1 cube strip file
  - `FaceDirectory { path, pattern }` - directory of 6 face images
- `FacePattern`
  - `AutoAliases` - tries all known face name aliases (px/right/posx/+x etc.)
  - `PxNxPyNyPzNz` - expects px/nx/py/ny/pz/nz naming
  - `PosxNegxPosyNegyPoszNegz` - expects posx/negx/posy/negy/posz/negz naming
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
use renderer::{EnvironmentSource, FacePattern};

// Auto-detect from file aspect ratio
let env = {
    let mut assets = renderer.assets();
    assets.load_environment(EnvironmentSource::Auto("assets/env/studio.hdr".into()))?
};

// Explicit equirectangular
let env = {
    let mut assets = renderer.assets();
    assets.load_environment(EnvironmentSource::Equirectangular("assets/env/studio.exr".into()))?
};

// Cube strip (6:1 horizontal)
let env = {
    let mut assets = renderer.assets();
    assets.load_environment(EnvironmentSource::CubeStrip("assets/env/strip.hdr".into()))?
};

// Face directory
let env = {
    let mut assets = renderer.assets();
    assets.load_environment(EnvironmentSource::FaceDirectory {
        path: "assets/env/sky_faces".into(),
        pattern: FacePattern::AutoAliases,
    })?
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
- For equirectangular sources, GPU-side conversion to cubemap happens during first activation.
- IBL generation (irradiance/prefilter) can happen on first activation.
- During activation, state may report `Loading` before `Ready`.
- On preparation failure, runtime keeps prior active environment and records failure.

Implication for game UX:
- Treat environment switch as potentially visible transition.
- Use status fields in debug UI to avoid "why didn't it switch?" confusion.

## Supported Input Formats

- **HDR** (Radiance `.hdr`): native float linear data
- **EXR** (OpenEXR `.exr`): native float linear data (requires `exr` crate)
- **LDR** (PNG/JPG/etc.): allowed for development, but IBL quality is limited (warning logged)

## Layout Detection (Auto mode)

Given decoded image `w x h`:
- `w == 2 * h` -> Equirectangular
- `w == 6 * h` -> CubeStrip (horizontal strip)
- otherwise -> error with actionable message

## Face Directory Aliases

Face directory loading supports these case-insensitive stem aliases:
- **+X**: px, posx, +x, right
- **-X**: nx, negx, -x, left
- **+Y**: py, posy, +y, top, up
- **-Y**: ny, negy, -y, bottom, down
- **+Z**: pz, posz, +z, front
- **-Z**: nz, negz, -z, back

If unsupported format/path is provided, load returns typed `AssetError`.

## Learn More

- Public asset API source: `src/renderer/src/api/assets.rs`
- Environment import logic: `src/renderer/src/data/environment_import.rs`
- Runtime status source: `src/renderer/src/api/renderer.rs`
- PBR/IBL background:
  - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
  - https://learnopengl.com/PBR/IBL/Specular-IBL
