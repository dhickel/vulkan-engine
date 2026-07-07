# Sprint 04 Current State Analysis

Date: 2026-07-03

## Verified Inputs

- Root repo guide identifies the root binary as a migration stub and renderer examples as current canonical renderer runtime entrypoints.
- Sprint 04 brief locks the goal: root `engine` binary becomes the alpha project launcher, sample project runs outside the editor, and headless proof must use draw-target capture.
- Sprint tracker lists Sprint 04 as `ironing-out` on branch `sprint/alpha-04-runtime-launcher`.
- Deep Track C recommends:
  - app crates under `apps/<app_name>`;
  - root launcher opens `engine.project.toml` for non-custom projects;
  - package tool may generate an app template later;
  - runtime project loading uses the same project/package/scene contracts as the editor;
  - default Rust loop is `cargo run -p <app>`.

## Root Launcher State

Current `src/main.rs`:

- imports only `std::env` and `std::process`;
- prints migration guidance for renderer examples;
- exits with code `2` even when invoked with no arguments;
- rejects all runtime arguments as unsupported.

This file is the direct implementation target for Phase 01 and Phase 02.

## Editor Launch State

Current editor surfaces:

- `apps/editor/src/launch.rs` defines `LaunchOptions` with project, scene, debug timing, capture, target, headless, and manual capture directory options.
- `LaunchOptions::parse` accepts both `--flag value` and selected `--flag=value` forms.
- It validates incompatible capture shapes such as `--capture_frame` with `--capture_frames`, zero counts, missing capture dependencies, and invalid `--capture_target` values.
- Parser tests already cover project/scene/debug flags, capture flags, sequence flags, and invalid capture values.
- `apps/editor/src/main.rs` applies debug timing options through `Renderer::configure_debug_timing_recording` and `Renderer::start_debug_timing_recording`.
- `apps/editor/src/main.rs` applies frame capture options through `Renderer::request_frame_capture_at`, `Renderer::configure_frame_capture_sequence`, and `Renderer::configure_manual_frame_capture_dir`.
- `run_headless_editor` uses `Renderer::new_headless`, `render_scene_headless`, expected-capture counting, and bounded frame budget behavior.
- Editor project loading is private helper code in `apps/editor/src/main.rs`: `load_project_context`, `load_enabled_project_packages`, and `load_startup_scene`.

Implementation implication:

- Reuse the editor launch behavior as a proven pattern, but avoid keeping the root runtime path coupled to editor UI code.
- A small shared root module is likely enough for the launcher; a new crate is not justified by current scope unless Rust package boundaries make sharing from `apps/editor` impossible without duplication.

## Renderer Facade State

Current renderer exports include:

- `Renderer`, `RendererConfig`, `FrameRenderOutcome`;
- `Renderer::new` and `Renderer::new_headless`;
- `Renderer::render_scene` and `Renderer::render_scene_headless`;
- `Renderer::take_startup_scene`;
- `Renderer::assets`;
- debug timing configuration and start APIs;
- frame capture scheduling APIs;
- `Project`, `ProjectPackage`, `ProjectSettings`, `ProjectValidationOptions`;
- package/project/scene validation functions;
- `Scene::load`.

`RendererConfig` currently includes:

- `window_width`;
- `window_height`;
- `app_name`;
- validation/shader/debug/preload/visual/headless fields;
- asset policy fields.

`Project` currently includes:

- durable `project_id`;
- `name`;
- `asset_root`;
- optional `startup_scene`;
- optional `default_environment`;
- enabled packages;
- `ProjectSettings` containing `window_width`, `window_height`, `fullscreen`, and `vsync`.

Implementation implication:

- The sprint can use current public facade APIs. A renderer internals redesign is not required.
- Fullscreen/vsync application should be treated carefully. Window size and app name are straightforward. Fullscreen may be a narrow winit window flag in Phase 02 if needed. Vsync appears not directly configurable through `RendererConfig`; document as currently deferred unless a narrow existing setting is found.

## Project/Package/Scene Validation State

`Project::load` calls `validate_project_file(...check_files(false))`, so it parses schema but does not enforce all file existence checks by itself.

`engine_pack` already validates:

- project files and startup scene existence;
- enabled package manifests and expected package IDs;
- scene references against project asset registries;
- path traversal and runtime-handle contamination boundaries.

`AssetManager::load_package_manifest_with_expected_id` is already used by the editor to load enabled packages before `Scene::load`.

Implementation implication:

- Runtime loading should explicitly validate project file and startup scene/package boundaries before rendering, not rely only on `Project::load`.
- If the root launcher needs reusable project-load diagnostics, prefer a small local helper that calls existing renderer validators and asset manager methods over adding a second schema implementation.

## Sample Project State

`apps/editor/sample_project/engine.project.toml`:

- `project_id = "project.editor_sample"`;
- `name = "Editor Sample Project"`;
- `asset_root = "assets"`;
- `startup_scene = "scenes/start.engine.scene.json"`;
- one enabled package: `editor_sample` at `assets/editor_sample.package.toml`;
- settings: `1440 x 900`, fullscreen false, vsync true.

The startup scene uses durable asset ID `editor_sample.model.block` and a path hint `models/block_prop.obj`.

Implementation implication:

- Phase 02 should use this sample as the root-launcher fixture.
- The sample proves durable package-backed scene loading is the key behavior to preserve.

## Capture State

Renderer capture support already writes PNG plus sidecar JSON.

Sidecar JSON includes:

- `status`;
- `frame_number`;
- `sequence_index`;
- `capture_target`;
- `source`;
- `png_path`;
- `extent`;
- `format`;
- `color_conversion`;
- `row_layout`;
- `captured_at_unix_ms`.

Sprint 03 docs accepted draw-target sidecars with:

- `capture_target = "draw"`;
- `format = "R16G16B16A16_SFLOAT"`;
- `status = "succeeded"`;
- `extent = 1440 x 900`.

Implementation implication:

- Sprint 04 proof must inspect sidecar predicates, not only existence of PNGs.
- Present-target captures are not acceptable proof for this sprint.

## Documentation State

Current README says:

- root `engine` binary is a migration stub;
- `cargo run` prints migration guidance and exits;
- renderer examples are runtime entrypoints.

Current API index centers the canonical example path on renderer examples and lists packaging CLI, editor docs, and renderer facade docs.

`docs/api/10-packaging-cli.md` currently lists `runtime project launcher` as deferred.

`docs/api/09-editor-asset-browser-and-wall-chunks.md` currently lists runtime project launcher as not included.

`docs/api/07-engine-arguments.md` is for renderer runtime examples and does not yet describe root project launcher arguments.

Implementation implication:

- Phase 03 must update stale docs after code lands.
- The docs should keep renderer examples as diagnostics/examples while pointing project users to `cargo run -- --project ...`.
- The docs must preserve deferred boundaries for hot reload, scripting, event system, physics, audio, and dogfood migration.

## Dogfood State

`apps/dungeon_dogfood` is a custom app crate with:

- custom level/content/generator paths;
- direct renderer facade usage;
- custom procedural scene seeding;
- custom content pack assets;
- its own README and validation commands.

Implementation implication:

- Sprint 04 should document dogfood as the custom Rust app path for now.
- Do not migrate dogfood to project manifests in this sprint.

## Architecture Fit

The lowest-risk target shape is:

- root `engine` binary owns the runtime project launcher;
- launch argument parsing is local to the root binary or shared via a small module that does not depend on editor UI;
- project/package/scene loading uses existing renderer facade types and validators;
- windowed runtime loop mirrors the editor/example pattern but omits editor UI and editor command history;
- headless runtime loop mirrors editor headless capture behavior;
- docs explain the data-driven root launcher and custom app-crate path as separate alpha loops.

## Main Risks

- Duplicating editor-private project load logic could create divergent behavior. Mitigate with small shared helpers or deliberate duplication with tests.
- Relying only on `Project::load` can miss file validation. Mitigate by calling existing validation paths before rendering.
- Fullscreen/vsync settings may not map cleanly today. Mitigate by applying what current APIs support and documenting unsupported settings honestly.
- Capture can appear to pass by creating present-target or desktop proof. Mitigate with sidecar predicate validation.
- Docs can overclaim hot reload/app lifecycle readiness. Mitigate with stale-claim sweep in Phase 04.

## Validation Blind Spots To Close

- Root binary currently has no tests. Add focused parser/loading tests.
- Runtime proof must run root `cargo run`, not editor or renderer examples.
- Negative CLI checks must verify controlled errors.
- Evidence summary must remain conservative until validators reconcile code checks and capture proof.
