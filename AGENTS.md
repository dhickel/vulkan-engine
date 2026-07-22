<!-- BEGIN INTERNAL-DEV WORKFLOW -->
## `.internal-dev` Development Document Store

`.internal-dev/` is the persistent engineering document store for specifications, plans, bugs, changelogs, reviews, reusable knowledge, and durable validation evidence.

### Required workflow
- Before non-trivial implementation or planning, read `.internal-dev/specifications/AGENTS.md`.
- Before changing services, APIs, web pages or fragments, UI modules, architecture, persistence, workflow behavior, or product contracts, read the relevant files in `.internal-dev/specifications/`.
- Before non-trivial work, list or search `.internal-dev/knowledge/` filenames and read only files whose domain matches the task.
- When lost, blocked by project context, or correcting a false assumption, search `.internal-dev/knowledge/` filenames again, then run a deeper grep across `.internal-dev/knowledge/` before inventing a new explanation.
- Use web or official documentation when missing information is external framework, library, tool, protocol, or platform behavior and local knowledge is absent or stale.
- Mid-workflow, route intended contracts to specifications, durable tradeoffs to `specifications/decisions.md`, reusable learning to knowledge, prior edit context to changelogs, defects to bugs, and scoped handoffs to plans or reviews.
- User hints like "future", "eventually", "later", or "this will become" go to `specifications/horizon-ideas.md` unless accepted as deferred product capability.
- Accepted future product capability goes to `specifications/deferred-features.md`.
- Durable architecture, design, product, and workflow decisions go to `specifications/decisions.md` with justification, alternatives or tradeoffs when known, caveats, affected specs, source, and review timing.
- After each feature implementation or non-trivial fix, complete the full `.internal-dev` closeout: update affected specifications, knowledge, bugs, changelogs, plans, and reviews; do not route active workflow material to retired focus, notes, research, or inbox stores.
- If an implementation has no impact on specifications, the changelog must say `Specification Impact: none` with one sentence explaining why.
- For any feature or non-trivial fix, update relevant project documentation when behavior, architecture, APIs, schema, configuration, or user workflows change.
- Plans and reviews are written to `.internal-dev/plans/` and `.internal-dev/reviews/`.
- Out-of-scope bugs found during work are logged immediately in `.internal-dev/bugs/`.
- If this project has a GitHub repository, every `.internal-dev/bugs/` report must be mirrored directly to the GitHub repository as a GitHub Issue when it is created or compiled.
- When adding or updating a local bug report in a project with a GitHub repository, check for related closed GitHub Issues before finishing; if the corresponding issue is already closed, move the local bug report to `.internal-dev/bugs/.archive/` instead of leaving it active.
- Finalized work gets a changelog entry in `.internal-dev/changelogs/`.
- Move finalized bug/plan artifacts to sibling `.archive/` directories.
- When repository or task policy requires a commit, include both implementation changes and related `.internal-dev` updates.
- **Stopgap commits**: Before starting any new phase, plan, or major task, commit all unfinished work in the working tree as a stopgap commit with a message like `wip: stopgap before <next-task>`. If uncommitted work from a prior session or agent is detected, commit it first before dispatching new work. This prevents working-tree edits from being clobbered when multiple agents touch the same files. Stopgap commits are never pushed without review.
- Inbound AgentMail or remote-work coordination uses the global `mailctl status`, `mailctl next`, and `mailctl wait` workflow. Do not create a repo-local `.internal-dev/inbox` ledger.

### Controlled access
- Do not read `.internal-dev` broadly by default.
- Read only the files required for the active task.

### Reference guide
- Process and templates: `.internal-dev/AGENTS.md`
- Specification routing and schemas: `.internal-dev/specifications/AGENTS.md`
<!-- END INTERNAL-DEV WORKFLOW -->

# Engine Repository Agent Guide

Use this file for repo-level orientation. For implementation details, jump to module-level `AGENTS.md` files.

## Scope and Runtime

- Language/runtime: Rust 2021, desktop Vulkan renderer plus alpha-stage engine support crates/apps.
- Root binary: `engine` (`src/main.rs`) is the data-driven project runtime launcher; the root library (`src/lib.rs`) is a thin app-facing facade; renderer examples remain diagnostics and capture entrypoints.
- Workspace members (membership is not production-readiness):
  - `src/input` (`input`): frame-buffered input broadcast/listener crate.
  - `src/renderer` (`renderer`): renderer runtime and API facade.
  - `src/audio` (`audio`): alpha audio crate.
  - `src/physics` (`physics`): alpha physics crate.
  - `src/scripting` (`scripting`): alpha scripting crate.
  - `src/events` (`engine_events`): event bus and lifecycle contracts.
  - `src/launch_shared` (`launch_shared`): shared launch infrastructure.
  - `apps/dungeon_dogfood` (`dungeon_dogfood`): dogfood application.
  - `apps/voxel_demo` (`voxel_demo`): configurable procedural-cave application.
  - `tools/engine_pack` (`engine_pack`): packaging CLI and new-app scaffolding.

## Source-of-Truth Policy

- Code is the logical source of truth.
- Documentation is intended truth.
- When code and docs diverge, note it in the relevant task output and record follow-up in `.internal-dev/`.

## Documentation Map

- API usage docs: `docs/api/00-index.md`
- Internal implementation docs: `docs/internal/00-index.md`
- Renderer package guide: `src/renderer/AGENTS.md`
- Input package guide: `src/input/AGENTS.md`
- Events package guide: `src/events/AGENTS.md`
- Audio package guide: `src/audio/AGENTS.md`
- Physics package guide: `src/physics/AGENTS.md`
- Scripting package guide: `src/scripting/AGENTS.md`
- Data internals: `src/renderer/src/data/AGENTS.md`
- Vulkan internals: `src/renderer/src/vulkan/AGENTS.md`
- Shader internals: `src/renderer/src/shaders/AGENTS.md`
- Tools guide: `tools/AGENTS.md`

## Project Skills

- Headless capture validation: `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Use this skill when validating renderer, scene, shader, camera, material, asset, or Vulkan behavior with screenshot evidence. Prefer timeout-bound engine headless captures over desktop screenshots so agents can validate without taking over the user's screen.

## Repository Layout

- `Cargo.toml`: workspace root (`engine`, `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `src/events`, `src/launch_shared`, `apps/dungeon_dogfood`, `apps/voxel_demo`, `tools/engine_pack`)
- `src/main.rs`: data-driven project launcher binary
- `src/lib.rs`: thin app-facing library facade
- `src/launch.rs`: launch command parsing and dispatch
- `src/runtime.rs`: runtime orchestration
- `src/input/`: input crate
- `src/renderer/`: rendering runtime crate
- `src/audio/`: alpha audio crate
- `src/physics/`: alpha physics crate
- `src/scripting/`: alpha scripting crate
- `src/events/`: event contracts and lifecycle bus
- `src/launch_shared/`: shared launch infrastructure
- `apps/dungeon_dogfood/`: dogfood application
- `apps/voxel_demo/`: configurable procedural-cave application
- `tools/engine_pack/`: packaging CLI and new-app scaffolding
- `docs/api/`: facade/API learning and usage path
- `docs/internal/`: internal implementation references
- `.internal-dev/`: development document store

## Runtime Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`
- `cargo check -p engine_events`
- `cargo check -p audio`
- `cargo check -p physics`
- `cargo check -p scripting`
- `cargo check -p launch_shared`
- `cargo check -p engine_pack`
- `cargo check -p dungeon_dogfood`
- `cargo check -p voxel_demo`
- `cargo test -p engine`
- `cargo test -p input`
- `cargo test -p engine_events`
- `cargo test -p audio`
- `cargo test -p physics`
- `cargo test -p scripting`
- `cargo test -p launch_shared`
- `cargo test -p renderer`
- `cargo test -p dungeon_dogfood`
- `cargo test -p voxel_demo`

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
