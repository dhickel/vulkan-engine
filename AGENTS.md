# Engine Repository Agent Guide

Use this file for repo-level orientation. For implementation details, jump to module-level `AGENTS.md` files.

## Scope and Runtime

- Language/runtime: Rust 2021, desktop Vulkan renderer plus alpha-stage engine support crates/apps.
- Root binary: `engine` (`src/main.rs`) is a migration stub that prints current example commands.
- Canonical renderer runtime entrypoints: `src/renderer/examples/*.rs`.
- Workspace members:
  - `src/renderer` (`renderer`): renderer runtime and API facade.
  - `src/input` (`input`): frame-buffered input broadcast/listener crate.
  - `src/audio` (`audio`): alpha audio crate.
  - `src/physics` (`physics`): alpha physics crate.
  - `src/scripting` (`scripting`): alpha scripting crate.
  - `apps/dungeon_dogfood` (`dungeon_dogfood`): dogfood application.
  - `apps/editor` (`editor`): alpha editor application.

## Source-of-Truth Policy

- Code is the logical source of truth.
- Documentation is intended truth.
- When code and docs diverge, note it in the relevant task output and record follow-up in `.internal-dev/`.

## Documentation Map

- API usage docs: `docs/api/00-index.md`
- Internal implementation docs: `docs/internal/00-index.md`
- Renderer package guide: `src/renderer/AGENTS.md`
- Input package guide: `src/input/AGENTS.md`
- Data internals: `src/renderer/src/data/AGENTS.md`
- Vulkan internals: `src/renderer/src/vulkan/AGENTS.md`
- Shader internals: `src/renderer/src/shaders/AGENTS.md`

## Project Skills

- Headless capture validation: `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Use this skill when validating renderer, scene, shader, camera, material, asset, or Vulkan behavior with screenshot evidence. Prefer timeout-bound engine headless captures over desktop screenshots so agents can validate without taking over the user's screen.

## `.internal-dev` Development Document Store

`.internal-dev/` is the persistent engineering document store for plans, bugs, changelogs, reviews, notes, and reusable knowledge.

### When you are finish task you must use internal-dev for (after asking the user it if time to first):
- Making a changelog to: `.internal-dev/changelogs/`:
- Add any general knowledge to : `.internal-dev/knowledge/`
- Add any notes to : `.internal-dev/notes/`, using or creating the futuer_consideration.md for future improvement/concerns that should be addressed
- Add any out of scope bugs to:`.internal-dev/bugs/`


When generating plans or reviews you are to always use  `.internal-dev/plans/` or `.internal-dev/reviews/`, large multistep plans should have their own directory.
 
- Operating guide and templates: `.internal-dev/AGENTS.md`
- `.internal-dev/` is intentionally untracked in this repo so the workflow can stay stable across repos.
- Structure:
- `.internal-dev/bugs/`: out-of-scope bugs found during other work (log immediately).
- `.internal-dev/plans/`: active plans in nested plan directories with phase files.
- `.internal-dev/reviews/`: review outputs.
- `.internal-dev/notes/`: deferred ideas/future considerations.
- `.internal-dev/knowledge/`: reusable research and learner-facing summaries.
- `.internal-dev/changelogs/`: finalized change records.
- Do not read `.internal-dev` broadly by default.
- Use controlled access: read only files needed for the active task.
- Ask before logging future considerations in `notes/` when they are out of scope.
- Move finalized bug/plan artifacts to sibling `.archive/` directories.
- Create changelog entries for finalized work.
- Keep AGENTS and `.internal-dev` documentation aligned with major architecture/process changes.

## Repository Layout

- `Cargo.toml`: workspace root (`engine`, `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, `apps/editor`)
- `src/main.rs`: migration stub that prints example commands
- `src/renderer/`: rendering runtime crate
- `src/input/`: input crate
- `src/audio/`: audio crate
- `src/physics/`: physics crate
- `src/scripting/`: scripting crate
- `apps/dungeon_dogfood/`: dogfood application
- `apps/editor/`: editor application
- `docs/api/`: facade/API learning and usage path
- `docs/internal/`: internal implementation references
- `.internal-dev/`: development document store

## Runtime Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`

Headless smoke pattern:

- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr`

Treat successful startup logs with no fatal errors before timeout as a smoke pass.

Headless capture validation:

- For visual proof, use `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`.
- Agents may create focused capture test scenes/specs for themselves when needed:
- source-controlled reusable scenes/examples under `src/renderer/examples/capture_tests/`
- temporary investigation specs/evidence under `.internal-dev/headless_capture_tests/`
- capture output under `.internal-dev/captures/`

Debug-record smoke pattern (agent should use this for runtime diagnosis by default):

- Engine startup can take ~20-30 seconds; use `timeout --signal=INT 60s`.
- Default debug capture profile: `--record_debug=10 --record_debug_interval=50`.
- Write capture output under `.internal-dev/debug_reports/` to avoid polluting repo root.
- Command template:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example <example_name> -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/<example_name>-timing.jsonl`
- With custom environment:
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/api_test-timing.jsonl`
- Optional output path override:
- `--record_debug_path /tmp/engine_timing.jsonl`
- Tune `--record_debug` seconds and `--record_debug_interval` ms when investigating longer or denser captures.

## External Baseline

When behavior is ambiguous in glTF/PBR/IBL flows, use:

- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

as conceptual lineage, not a drop-in implementation source.
