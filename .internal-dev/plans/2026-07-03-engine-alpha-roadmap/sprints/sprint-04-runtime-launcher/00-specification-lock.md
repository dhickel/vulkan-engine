# Sprint 04 Specification Lock: Runtime Project Launcher And App Dev Loop

Date: 2026-07-03
Classification: medium
Branch: `sprint/alpha-04-runtime-launcher`
Plan root: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/`

## Locked Objective

Make the root `engine` binary the alpha runtime project launcher and define the supported application development loop.

The launcher must run `apps/editor/sample_project/engine.project.toml` outside the editor, support true headless draw-target capture for validation, and document app crates under `apps/<name>` as the default Rust custom-code development loop.

## User-Visible Outcome

From the workspace root, a project author can:

```bash
cargo run -- --project apps/editor/sample_project/engine.project.toml
```

to run a packaged scene outside the editor, and can run a validation capture with:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/headless-draw
```

The docs must also explain that custom Rust behavior is developed in workspace app crates under `apps/<name>` using `cargo run -p <app>`, not by editing renderer internals or relying on dynamic Rust hot reload.

## Acceptance Criteria

- `cargo run -- --project apps/editor/sample_project/engine.project.toml` starts the sample project through the root `engine` binary outside the editor.
- The root launcher supports explicit `--project <path>` and `--project=<path>`.
- The root launcher rejects missing project arguments, unreadable project files, missing startup scenes, unknown/missing enabled package manifests, invalid package IDs, and invalid capture options with controlled stderr and non-zero exit codes.
- Project runtime loading validates project/package/startup scene boundaries before rendering and loads enabled package manifests before loading the startup scene.
- The launcher applies `ProjectSettings.window_width`, `ProjectSettings.window_height`, and project/app name to `RendererConfig` for windowed and headless launch. Fullscreen/vsync may remain documented as currently not wired unless implementation can support them narrowly without renderer redesign.
- The launcher uses `Renderer::new` for windowed runs and `Renderer::new_headless` plus `render_scene_headless` for headless runs.
- Headless validation proof uses `--headless --capture_target draw`; sidecar JSON must report `status = "succeeded"`, `capture_target = "draw"`, and a draw-target source format such as `R16G16B16A16_SFLOAT`.
- The launcher exits automatically after requested headless captures complete and reports failure if captures are not completed within a bounded frame budget.
- Debug timing launch options remain supported where planned: `--record_debug`, `--record_debug_interval`, and `--record_debug_path`.
- Documentation updates identify:
  - root `engine` launcher as the alpha project runtime path for data-driven projects;
  - app crates under `apps/<name>` as the supported Rust custom-code loop;
  - `apps/dungeon_dogfood` as a custom app path for now unless a narrow implementation decision proves otherwise;
  - renderer examples as renderer diagnostics/examples, not the primary app runtime path;
  - dynamic Rust hot reload, scripting, event system, physics, audio, and broad dogfood migration as deferred.
- `artifacts/validation-summary.json` is updated conservatively and does not claim `fully_validated` until all phase validation and capture evidence have passed.

## Negative Criteria

- Do not implement dynamic Rust hot reload, runtime Rust compilation, plugin ABI, scripting integration, event system, physics, audio, or custom gameplay lifecycle APIs.
- Do not require the editor to run projects.
- Do not leave renderer examples as the documented alpha runtime path after the root launcher works.
- Do not accept desktop screenshots, compositor screenshots, or present-target captures as proof for Sprint 04 visual validation.
- Do not serialize runtime handles such as `MeshHandle`, `TextureHandle`, `EnvironmentHandle`, `SceneNodeId`, `PointLightId`, or `LoadTicket` into project/package/scene data.
- Do not redesign renderer internals, swapchain/presentation, asset cache lifetime, or dogfood gameplay.
- Do not change `apps/dungeon_dogfood` code unless a later worker proves a tiny docs-aligned compatibility edit is needed and the directive allows it. The default is docs-only for dogfood.
- Do not close the sprint with stale docs still claiming `cargo run` prints migration guidance or that the runtime launcher is deferred.

## Validation Criteria

Required compile and test checks:

```bash
cargo fmt --check
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p editor
cargo check -p engine_pack --locked
cargo test -p engine
cargo test -p renderer
cargo test -p engine_pack --locked
git diff --check
```

Required runtime and CLI checks:

```bash
cargo run -- --help
cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames 1 --capture_frame_start 5 --capture_dir .internal-dev/captures/sprint-04-runtime-launcher/smoke-draw
cargo run -- --project .internal-dev/does-not-exist/engine.project.toml
cargo run -- --project apps/editor/sample_project/engine.project.toml --capture_target swapchain
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
```

Required capture proof:

- Use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Capture command must include `--headless --capture_target draw`.
- Capture output must be under `.internal-dev/captures/sprint-04-runtime-launcher/`.
- Capture sidecar predicates:
  - `.status == "succeeded"`;
  - `.capture_target == "draw"`;
  - `.format` is not a present/swapchain-only format; expected draw-target format is currently `R16G16B16A16_SFLOAT`;
  - `.extent.width > 0` and `.extent.height > 0`;
  - `.png_path` exists.
- Desktop screenshots and present-target captures are invalid evidence for this sprint.

## Constraints

- Preserve unrelated local state and do not revert user work.
- Plan and evidence files remain under the Sprint 04 plan directory except runtime capture/debug output, which should use `.internal-dev/captures/sprint-04-runtime-launcher/` and `.internal-dev/debug_reports/sprint-04-runtime-launcher/`.
- Code is the logical source of truth; docs are intended truth. If code and docs diverge during execution, record the mismatch in validation output and either fix in-scope docs or log a follow-up artifact.
- `.internal-dev` is untracked and should be used with controlled access only.
- Ask the user before logging out-of-scope future considerations in `.internal-dev/notes/`.
- Create a changelog under `.internal-dev/changelogs/` only when final closeout reaches the repo-required changelog timing gate.

## Assumptions

- The root `engine` binary can be repurposed because current `src/main.rs` is only a migration stub.
- The editor sample project is the canonical first runtime fixture.
- The editor's current launch parser and project-loading helpers are useful source material, but workers should avoid keeping project runtime behavior private to `apps/editor`.
- Existing renderer APIs are sufficient for this sprint: `Renderer::new`, `Renderer::new_headless`, `render_scene`, `render_scene_headless`, frame capture scheduling, `Project`, package validation/load APIs, and `Scene::load`.
- `engine_pack` remains the package/project/scene validation CLI and does not need to become the runtime launcher.
- Dogfood remains a custom app path unless the implementation finds a very small shared-runtime reuse opportunity. A broad dogfood migration is out of scope.

## User Decision Gates

- Stop and return to the main thread if implementing the root launcher requires broad renderer API redesign or Vulkan/presentation changes.
- Stop and return to the main thread if true headless draw-target capture cannot be produced; record a tooling blocker instead of substituting desktop or present proof.
- Stop and return to the main thread if satisfying dogfood parity would require event, physics, audio, scripting, or gameplay lifecycle work.
- Stop and return to planning if validation criteria are found to be materially wrong or impossible under live code.

## Non-Goals

- Dynamic Rust hot reload.
- Scripting runtime.
- Event system.
- Physics or collision authoring/runtime integration.
- Audio authoring/runtime integration.
- Dogfood migration to project manifests.
- Binary package archive support.
- Asset thumbnails.
- Editor UI redesign.
- Renderer internals cleanup unrelated to launcher execution.

## Stop Rules

- Stop code work if package/project/scene identity regresses into runtime handle serialization.
- Stop closeout if required validation reports or `artifacts/validation-summary.json` are missing or contradictory.
- Stop final status promotion if any required validator is missing, failed, or unreconciled.
- Stop final status promotion if capture proof is present-target, desktop screenshot based, or lacks passing sidecar predicates.
