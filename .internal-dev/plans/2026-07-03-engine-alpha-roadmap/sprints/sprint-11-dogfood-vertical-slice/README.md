# Sprint 11: Dogfood Vertical Slice

Status: planned

Intended execution branch: `sprint/alpha-11-dogfood-vertical-slice`

Plan directory: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice/`

## Objective

Turn `apps/dungeon_dogfood` from a useful renderer workbench into a small explorable alpha vertical slice that uses the project/package/scene contracts wherever practical and exposes API friction honestly.

## User-Visible Outcome

A clean checkout can validate and run a dungeon dogfood slice with documented commands. The slice loads packaged content, renders a saved startup scene or deterministic scene data through the alpha contracts, supports camera/input-driven exploration, shows materials/lights/environment, and produces true engine-owned headless draw capture evidence.

## In Scope

- Audit current dogfood app, content pack, package/project runtime, and Sprint 09/Sprint 10 overlap before mutating code.
- Add or normalize dogfood package/project/scene artifacts under `apps/dungeon_dogfood/` when the live contract supports it.
- Keep any remaining dogfood-only manifest behavior explicit as migration debt with an artifact, not hidden app-only logic.
- Exercise input, camera, scene loading, materials, lighting, environment, props, audio metadata, and one exploration/gameplay loop.
- Add targeted tests for dogfood content/project/runtime glue where they can run without Vulkan.
- Add true `--headless --capture_target draw` visual baseline support and validation evidence.
- Update public docs and `.internal-dev` sprint reports for run instructions, validation, residuals, and closeout.

## Out Of Scope

- Do not implement Sprint 09 facade API cleanup or Sprint 10 advanced rendering API work here.
- Do not edit active Sprint 09 files until the main thread confirms that work is merged or safe to build on.
- Do not touch `.idea/engine.iml` or `.reasonix/`.
- Do not update `SPRINT-TRACKER.md`; the main thread owns tracker reconciliation.
- Do not build a full game, editor workflow, dynamic Rust hot reload, production physics migration, production audio mixer, or binary package format.
- Do not use desktop screenshots as visual proof.

## Target Surfaces

Code:

- `apps/dungeon_dogfood/`
- `tools/engine_pack/`
- `src/runtime.rs`
- `src/launch.rs`
- `src/renderer/src/api/scene.rs`
- `src/renderer/src/data/asset_registry.rs`
- `src/input/`
- only after Sprint 09/Sprint 10 are reconciled: `src/renderer/src/api/mod.rs`, `src/renderer/src/lib.rs`, `src/renderer/examples/*`

Docs:

- `apps/dungeon_dogfood/README.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- `docs/api/00-index.md`
- optional: `docs/api/14-dogfood-vertical-slice.md`

`.internal-dev` artifacts:

- this plan suite;
- `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`;
- `.internal-dev/debug_reports/sprint-11-dogfood-vertical-slice/`;
- `.internal-dev/headless_capture_tests/sprint-11-dogfood-vertical-slice/` if needed;
- `.internal-dev/bugs/` for out-of-scope API friction or defects;
- `.internal-dev/changelogs/` only when the main thread confirms closeout timing.

## Assumptions

- Sprint 09 is active and may be changing renderer facade exports/examples; implementation workers must refresh before touching shared renderer API/example files.
- Sprint 10 is not planned in the tracker; if Sprint 10 lands before Sprint 11 execution, workers must inspect its contract before using advanced rendering hooks.
- The current dogfood app uses `apps/dungeon_dogfood/assets/content_pack.toml`, which is app-specific and not the canonical package/project contract.
- The root launcher already supports project/package/scene validation and true headless draw capture for data-driven projects.
- Existing `cargo test -p dungeon_dogfood` may remain blocked by the known renderer test-profile `russimp_sys` binding issue; use `cargo check -p dungeon_dogfood` plus focused non-Vulkan tests unless live source proves the blocker is fixed.

## Risks And Gotchas

- Package/project contracts may not yet express every dogfood concept. Produce a migration debt artifact instead of inventing another permanent app manifest.
- Headless visual proof must come from engine-owned draw-target capture metadata and images, not compositor screenshots.
- Runtime/project scene loading may be data-only while dogfood custom gameplay still needs an app crate. Keep that boundary documented.
- Asset paths must remain package/project relative where contracts require it; do not serialize runtime handles.
- Visual changes can be slow to validate. Use timeout-bound commands and deterministic capture settings.

## Acceptance Criteria

- Dogfood has a validated package/project/scene path or a documented, deliberate migration-debt exception for each unsupported concept.
- Dogfood runtime exercises input, camera, scene/content loading, materials, lighting, environment, and at least one exploration/gameplay loop.
- Clean-checkout instructions exist for validation and run paths.
- Headless visual baseline uses `--headless --capture_target draw`; sidecar metadata is inspected for success, draw target, positive extent, and existing PNGs.
- API friction discovered during dogfood is filed into alpha backlog/docs rather than hidden behind app-only hacks.
- Final evidence is indexed in `artifacts/validation-summary.json` with conservative status.

## Negative Criteria

- Runtime handles or local absolute paths appear in project/package/scene files.
- The dogfood slice depends on undocumented one-off manifests with no migration debt artifact.
- Visual validation relies on desktop screenshots, present-target capture, or compile checks alone.
- The app bypasses input/frame boundaries or camera contracts with hard-coded per-platform hacks.
- Docs claim `fully_validated` while capture proof, validators, residual tracking, or closeout evidence is missing.

## Validation Plan

Compile/test:

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- `cargo test -p input`
- `cargo check -p engine_pack`
- `cargo test -p engine_pack`
- `cargo check -p dungeon_dogfood`
- focused package/project/scene validation tests added by phases

Runtime smoke:

- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level generated_sprawl`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level level_02_ramps`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level level_03_lighting`
- dogfood project path smoke, exact command to be locked by Phase 02/03 after live CLI shape is implemented.

Visual/capture proof:

- Must use a true engine-owned headless draw path, expected final shape:
  - `RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --project apps/dungeon_dogfood/engine.project.toml --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5 --capture_dir .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-baseline`
- If the app cannot accept that exact shape without broader API work, Phase 04 must implement the narrowest equivalent dogfood-owned headless path or stop with a migration-debt/blocker report.

Docs/process checks:

- `rg -n "/tmp|pending|planned|not implemented|TODO|desktop screenshot|present-target" .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-11-dogfood-vertical-slice docs/api apps/dungeon_dogfood/README.md`
- Validate all referenced evidence paths exist before final quality review.

## Advanced-Planner Handoff

This suite is large because Sprint 11 crosses package/project schema, dogfood app runtime, content assets, renderer capture, docs, and validation evidence. Execute phases in order. Do not mutate product code from this planning pass.

## Closeout Checklist

- Validation evidence recorded in `artifacts/validation-summary.json`.
- Known residuals tracked in `.internal-dev/bugs/`, plan reports, or docs.
- Changelog timing confirmed with user before writing `.internal-dev/changelogs/`.
- Final closeout report/email content staged under `reports/` for main-thread/agentmail handling.
- Main thread updates `SPRINT-TRACKER.md` after review, not during planning.
