# Runtime Project Launcher

## 1. Purpose & Audience

This page documents the root `engine` launcher for alpha data-driven projects. Use it when running a project manifest outside the editor, validating packaged scene data, or producing headless draw-target capture evidence.

For custom Rust gameplay or tool behavior, use an app crate under `apps/<name>` and run it with `cargo run -p <app>`. The root launcher emits alpha lifecycle/package/scene/shutdown events for validation and diagnostics, but it is not dynamic Rust hot reload, scripting runtime execution, physics integration, audio integration, generated app templates, or a full gameplay lifecycle framework.

## 2. Basic Launch

Run commands from the workspace root:

```sh
cargo run -- --project apps/editor/sample_project/engine.project.toml
```

The `--project` argument is required. The launcher validates the project, enabled package manifests, and startup scene before rendering.

To override the project startup scene:

```sh
cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --scene apps/editor/sample_project/scenes/start.engine.scene.json
```

## 3. Headless Draw Capture

Visual validation must use the true headless draw-target path:

```sh
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

Accepted Sprint 04 evidence comes from `Renderer::new_headless` and `render_scene_headless`. Sidecar JSON must report `status = "succeeded"`, `capture_target = "draw"`, a draw-target image format such as `R16G16B16A16_SFLOAT`, positive extent, and an existing PNG path. Desktop screenshots, compositor screenshots, and present-target captures are not validation evidence for this path.

## 4. Argument Reference

Live help is available with:

```sh
cargo run -- --help
```

Supported launcher arguments:

| Argument | Value | Meaning |
|---|---|---|
| `--help`, `-h` | none | Print help and exit `0` |
| `--project`, `--project=<path>` | project manifest path | Required project manifest to launch |
| `--scene`, `--scene=<path>` | scene path | Optional startup scene override |
| `--headless` | none | Use the headless runtime path |
| `--capture_target`, `--capture_target=<present\|draw>` | `present` or `draw` | Select capture source |
| `--capture_frame`, `--capture_frame=<n>` | frame number | Request one frame capture |
| `--capture_frame_path`, `--capture_frame_path=<path>` | path | Output path for one frame capture |
| `--capture_frames`, `--capture_frames=<n>` | count | Request a capture sequence |
| `--capture_frame_start`, `--capture_frame_start=<n>` | frame number | First sequence frame |
| `--capture_frame_interval`, `--capture_frame_interval=<n>` | frame count | Interval between sequence frames |
| `--capture_dir`, `--capture_dir=<dir>` | directory | Output directory for sequence captures |
| `--manual_capture_dir`, `--manual_capture_dir=<dir>` | directory | Manual capture output directory |
| `--record_debug`, `--record_debug=<seconds>` | seconds | Start debug timing capture at launch |
| `--record_debug_interval`, `--record_debug_interval=<ms>` | milliseconds | Debug timing sample interval |
| `--record_debug_path`, `--record_debug_path=<path>` | path | Debug timing JSONL output path |

Usage errors exit with code `2`. Runtime validation errors exit non-zero with actionable stderr. Unknown flags fail instead of being ignored.

## 5. Data Flow

The launcher uses the same project/package/scene contracts as the packaging CLI and editor:

1. Load and validate `engine.project.toml`.
2. Resolve the project root from the manifest path.
3. Load enabled package manifests with expected package IDs.
4. Resolve the startup scene from `--scene` or project `startup_scene`.
5. Load the scene through the renderer asset manager.
6. Render through `Renderer::new` for windowed launch or `Renderer::new_headless` for headless launch.

The launcher also records lifecycle events around these boundaries: app start, project loading, package loading success/failure, scene loading, headless shutdown completion, and windowed shutdown-requested intent.

Project `name`, `window_width`, and `window_height` feed `RendererConfig`. Other project settings such as fullscreen and vsync may exist in project data, but the root launcher only documents behavior that is currently wired.

## 6. Custom Rust App Loop

Use an app crate when the project needs custom Rust behavior:

```sh
cargo run -p dungeon_dogfood
```

App crates can depend on `renderer`, `input`, and other workspace support crates directly. They own their Rust control flow and may choose whether to consume project/package/scene data. `apps/dungeon_dogfood` is the current custom app path and has not been migrated to project manifests. App crates can consume renderer events through the public facade. Dynamic Rust hot reload, a scripting runtime, physics/collision gameplay integration, audio gameplay integration, broad dogfood migration, and generated app templates are later roadmap work, not part of the current root launcher.

## 7. See Also

- [Packaging CLI](10-packaging-cli.md)
- [Editor Asset Browser and Wall Chunk Placement](09-editor-asset-browser-and-wall-chunks.md)
- [Engine Arguments](07-engine-arguments.md)
- [Events and Lifecycle](12-events-and-lifecycle.md)
- [Dungeon Dogfood README](../../apps/dungeon_dogfood/README.md)
