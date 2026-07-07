# Sprint 04: Runtime Project Launcher And Application Dev Loop

Status: closed

## Objective

Turn project manifests and package-backed scenes into a runnable alpha application path outside the editor, while defining the default Rust application development loop without overclaiming hot reload, scripting, physics, audio, or custom gameplay lifecycle work.

## User-Visible Outcome

A project author can run the sample project through the engine runtime launcher with a documented command, capture it headlessly for validation, and understand how to create a Rust app crate under `apps/` without modifying renderer internals.

## In Scope

- Decide and implement the root `engine` binary as the alpha project launcher unless planning finds a hard blocker.
- Support `cargo run -- --project <engine.project.toml>` from the root binary.
- Load project manifest, enabled package manifests, startup scene, project window settings, default environment, and package asset references through a runtime path that is not editor-specific.
- Support the same debug-record and frame-capture launch options needed for validation, including `--headless --capture_target draw`.
- Provide a small reusable runtime-launch module if that avoids duplicating editor project loading logic.
- Add focused tests for argument parsing, project path resolution, missing/invalid project errors, and package/startup scene validation boundaries.
- Add documentation for the alpha build/run loop: root launcher for data-driven projects, app crates under `apps/<name>` for custom Rust behavior, incremental `cargo run -p <app>` as the default Rust iteration loop, and asset/data hot reload as a future scoped capability.
- Run and capture the editor sample project outside the editor through the root launcher.

## Out Of Scope

- Dynamic Rust plugin ABI, live Rust code reload, or runtime recompilation.
- Scripting runtime integration.
- Event system, lifecycle bus, physics, collision, audio, or gameplay API design beyond documenting why they are later sprints.
- Full migration of `apps/dungeon_dogfood` to project manifests if that would expand the sprint; document its current custom path and any intentional divergence.
- Editor UI changes beyond shared helper extraction if needed.
- Binary package archives or package thumbnail generation.

## Target Surfaces

- Code:
  - `src/main.rs`
  - potential new root module files under `src/`
  - `apps/editor/src/main.rs` and/or `apps/editor/src/launch.rs` only if shared loading/capture code is extracted cleanly
  - `src/renderer/src/api/*` only for small facade additions needed by runtime launch
  - `tools/engine_pack/src/main.rs` only if template or validation integration is selected by the advanced plan
- Docs:
  - `README.md`
  - `docs/api/00-index.md`
  - new or existing API/runtime launcher guide under `docs/api/`
  - `apps/dungeon_dogfood/README.md` if documenting the dogfood path
- `.internal-dev` artifacts:
  - sprint plan suite under this directory
  - validation reports under `validation/`
  - capture output under `.internal-dev/captures/sprint-04-runtime-launcher/`
  - final evidence summary under `artifacts/validation-summary.json`

## Assumptions

- The root `engine` binary can become the launcher because it is currently only a migration stub.
- The editor sample project is the canonical first data-driven runtime fixture.
- Root launcher runtime should default to a simple windowed render loop for interactive use and a headless render loop for validation.
- Current project settings are enough for initial window size/title-ish behavior; missing settings should fall back to `RendererConfig::default()`.
- Rust app crates remain the alpha path for custom code; hot Rust reload is deferred.

## Risks And Gotchas

- Editor-local project loading may need extraction to avoid divergent runtime/editor behavior.
- The root binary must not reintroduce stale renderer example semantics or serialize runtime handles.
- Headless proof must use true headless renderer creation and draw-target capture, not desktop screenshots or present-target evidence.
- Dogfood currently has a custom content/level/generation path; forcing it into the project launcher too early could create an oversized sprint.
- Existing renderer warnings are likely to remain and should be recorded as residuals, not hidden.

## Acceptance Criteria

- `cargo run -- --project apps/editor/sample_project/engine.project.toml` starts the sample project outside the editor.
- `cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw --capture_frames ...` produces draw-target sidecar evidence for the sample project.
- The launcher validates or clearly errors on missing project, missing startup scene, unknown package, and invalid capture options.
- Project/package/scene loading uses durable asset IDs and does not introduce runtime handle identity into persisted scene data.
- Documentation states the supported alpha run loop and the deferred hot-code/plugin/script boundaries.
- Dogfood app status is documented as either using the same runtime path or intentionally remaining a custom app path for now.

## Negative Criteria

- Do not claim dynamic Rust hot reload, scripting, event system, physics, audio, or dogfood migration is complete.
- Do not make the editor the only way to run a project.
- Do not make users launch renderer examples as the alpha runtime path.
- Do not accept present-target capture as the visual proof for headless validation.
- Do not close the sprint if final report/email evidence and validation summary are missing.

## Validation Plan

- Compile/test:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo check -p renderer`
  - `cargo check -p editor`
  - `cargo check -p engine_pack --locked`
  - focused launcher unit/integration tests added by the plan
  - existing package/scene validation tests touched by shared loading changes
- Runtime smoke:
  - timeout-bound root launcher startup against `apps/editor/sample_project/engine.project.toml`
  - negative launcher command checks for missing project and invalid capture options
- Visual/capture proof:
  - true headless draw-target capture from the root launcher into `.internal-dev/captures/sprint-04-runtime-launcher/`
  - sidecar predicates must include `capture_target = "draw"`, `status = "succeeded"`, and non-present source format
- Docs/process checks:
  - `git diff --check`
  - stale-reference sweep for renderer examples as the only runtime path, unsupported hot reload claims, present-target proof, and stale sprint statuses
  - validation summary JSON parse check

## Advanced-Planner Handoff

Classification: medium.

Expected phases:

- Phase 01: Runtime launcher skeleton and argument contract.
- Phase 02: Project/package/startup scene runtime loading and interactive/headless loop.
- Phase 03: App development loop docs and dogfood path decision.
- Phase 04: Headless draw capture proof, final quality review, changelog, report evidence, and tracker closeout.

Worker boundaries:

- Keep root launcher/runtime code separate from editor UI.
- Extract shared project-loading helpers only if the boundary is small and keeps editor behavior stable.
- Do not alter dogfood gameplay code unless the plan identifies a narrow docs or launch compatibility need.

Stop conditions:

- Stop if root launcher requires broad renderer API redesign.
- Stop if a capture path cannot produce true headless draw-target evidence; record the tooling blocker instead of accepting desktop proof.
- Stop if dogfood migration expands into event/physics/audio/scripting work.

## Closeout Checklist

- Validation evidence recorded.
- Known residuals tracked.
- Changelog created during final closeout.
- Final report email sent.
- Sprint tracker updated.
