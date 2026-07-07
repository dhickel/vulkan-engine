# Sprint 04 Target Design

## Design Summary

The root `engine` binary becomes the alpha project runtime launcher:

```bash
cargo run -- --project apps/editor/sample_project/engine.project.toml
```

It loads an `engine.project.toml`, validates enabled packages and startup scene, initializes a renderer, loads the startup scene through package-backed durable asset IDs, and runs either:

- a simple windowed render loop for interactive inspection; or
- a bounded headless render loop for validation/capture.

Custom Rust applications remain workspace app crates under `apps/<name>`, run with:

```bash
cargo run -p <app>
```

## CLI Contract

Required supported arguments:

- `--help` / `-h`: print usage and exit `0`.
- `--project <path>` and `--project=<path>`: required project manifest path.
- `--scene <path>` and `--scene=<path>`: optional startup scene override.
- `--headless`: use `Renderer::new_headless` and `render_scene_headless`.
- `--capture_target <present|draw>` and `--capture_target=<present|draw>`.
- `--capture_frame <n>` and `--capture_frame=<n>`.
- `--capture_frame_path <path>` and `--capture_frame_path=<path>`.
- `--capture_frames <n>` and `--capture_frames=<n>`.
- `--capture_frame_start <n>` and `--capture_frame_start=<n>`.
- `--capture_frame_interval <n>` and `--capture_frame_interval=<n>`.
- `--capture_dir <dir>` and `--capture_dir=<dir>`.
- `--manual_capture_dir <dir>` and `--manual_capture_dir=<dir>`.
- `--record_debug <seconds>` and `--record_debug=<seconds>`.
- `--record_debug_interval <ms>` and `--record_debug_interval=<ms>`.
- `--record_debug_path <path>` and `--record_debug_path=<path>`.

Recommended default:

- `--project` is required for the root runtime launcher. Do not silently fall back to the editor sample project in the root runtime path, because root runtime should behave like an app launcher rather than an editor convenience mode.

Controlled error rules:

- usage errors exit `2`;
- validation/runtime errors exit non-zero and print stable, actionable stderr;
- unknown flags fail instead of being ignored in the root launcher;
- invalid capture target must mention accepted values `present` and `draw`;
- `--capture_frame_path` requires `--capture_frame`;
- `--capture_dir`, `--capture_frame_start`, and `--capture_frame_interval` require `--capture_frames`;
- `--capture_frame` and `--capture_frames` are mutually exclusive.

## Runtime Loading Contract

Runtime loading must use current project/package/scene contracts:

1. Resolve the project path relative to the current process working directory.
2. Validate the project file with file checks before rendering.
3. Load `Project`.
4. Determine `project_root` from the project file parent.
5. Resolve scene path:
   - use `--scene` when supplied;
   - otherwise use `project.startup_scene`;
   - fail if neither is available.
6. Validate the resolved scene against the project asset registry where practical using existing renderer validators.
7. Initialize `RendererConfig`:
   - `app_name = project.name` or a stable `"engine"` fallback;
   - `window_width = project.settings.window_width`;
   - `window_height = project.settings.window_height`;
   - `headless = launch_options.headless`;
   - `preload_startup_scene = false` for app-driven scene loading unless a worker proves this regresses a necessary default;
   - asset policy aligned with editor runtime package loading.
8. Initialize renderer:
   - `Renderer::new(config, &window)` for windowed;
   - `Renderer::new_headless(config)` for headless.
9. Load enabled package manifests with expected package IDs before scene load.
10. Load the startup scene with `Scene::load(scene_path, &mut renderer.assets())`.
11. Render:
   - windowed path feeds winit events into `renderer.update_input`, handles resize/close/escape, and calls `renderer.render_scene`;
   - headless path calls `renderer.render_scene_headless` until requested captures finish or bounded frame budget expires.

## Capture Contract

The root launcher must wire launch-time capture options to the existing renderer capture APIs:

- `Renderer::configure_manual_frame_capture_dir`;
- `Renderer::request_frame_capture_at`;
- `Renderer::configure_frame_capture_sequence`.

For Sprint 04 validation:

- required command includes `--headless --capture_target draw`;
- output directory is `.internal-dev/captures/sprint-04-runtime-launcher/headless-draw`;
- sidecar predicates must pass:
  - `status == "succeeded"`;
  - `capture_target == "draw"`;
  - `format == "R16G16B16A16_SFLOAT"` or an explicitly documented draw image format from current renderer code;
  - extent width/height match the project settings unless the implementation documents an explicit headless sizing reason;
  - PNG file exists and is non-empty.

Desktop screenshots and present-target captures are not accepted proof.

## Debug Timing Contract

The root launcher should preserve the existing debug-record behavior:

- configure timing when any record-debug option is provided;
- start timing immediately only when `--record_debug` is supplied;
- write debug JSONL under `.internal-dev/debug_reports/sprint-04-runtime-launcher/` in validation commands.

## Module Boundary

Preferred implementation structure:

- Keep `src/main.rs` as a thin entrypoint.
- Add root-binary modules under `src/` if needed, for example:
  - `src/launch.rs` for CLI parsing and validation;
  - `src/runtime.rs` for project loading and runtime loops.
- Do not make the root binary depend on editor app internals.
- If duplication with editor launch parsing becomes too risky, extract only the small common non-UI pieces. Do not create a broad shared engine app framework in this sprint.

## Docs Design

Docs should describe two alpha loops:

1. Data-driven runtime project launcher:
   - author/validate project/package/scene data;
   - run with root `cargo run -- --project ...`;
   - use headless draw-target capture for validation.
2. Custom Rust app crate:
   - create workspace app under `apps/<name>`;
   - depend on `renderer` and other support crates as needed;
   - iterate with `cargo run -p <app>`;
   - use project/package/scene data where it fits;
   - dynamic Rust hot reload and scripting are later sprint topics.

Docs should update or add:

- `README.md`;
- `docs/api/00-index.md`;
- `docs/api/07-engine-arguments.md` or a new root-runtime guide such as `docs/api/11-runtime-project-launcher.md`;
- `docs/api/10-packaging-cli.md`;
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`;
- `apps/dungeon_dogfood/README.md` if needed to label dogfood as a custom app crate path.

## Architecture Decisions

- Use root `engine` binary for project launcher because it is currently a stub and matches Track C.
- Require explicit `--project` in root launcher because root runtime should not pick editor sample by magic.
- Keep renderer examples as diagnostic/sample code, not the alpha app runtime path.
- Reuse existing validators and facade types; do not build a second schema parser.
- Keep dogfood custom in Sprint 04; document divergence instead of migrating.
- Treat unsupported project settings honestly. `window_width`, `window_height`, and `name` must be applied; fullscreen/vsync can be documented if not supported by current API.

## Residual Risks

- Renderer startup may take 20-30 seconds; all runtime validation commands must use `timeout --signal=INT 60s`.
- Host Vulkan/headless support may fail. If so, record `TOOLING_CONSTRAINT` and do not accept alternate proof without user approval.
- If `cargo test -p renderer` exposes existing unrelated failures, validators should isolate whether Sprint 04 changed behavior and record residuals conservatively.
