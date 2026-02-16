# Engine Repository Agent Guide

Use this file for repo-level orientation. For implementation details, jump to module-level `AGENTS.md` files.

## Scope and Runtime

- Language/runtime: Rust 2021, desktop Vulkan renderer.
- Root binary: `engine` (`src/main.rs`) is a migration stub.
- Canonical runtime entrypoints: `src/renderer/examples/*.rs`.
- Workspace crates:
- `src/renderer` (`renderer`): renderer runtime and API facade.
- `src/input` (`input`): input broadcast/listener crate.

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

## `.internal-dev` Development Document Store

`.internal-dev/` is the persistent engineering document store for plans, bugs, changelogs, reviews, notes, and reusable knowledge.

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

- `Cargo.toml`: workspace root (`engine`, `src/input`, `src/renderer`)
- `src/main.rs`: migration stub that prints example commands
- `src/renderer/`: rendering runtime crate
- `src/input/`: input crate
- `docs/api/`: facade/API learning and usage path
- `docs/internal/`: internal implementation references
- `.internal-dev/`: development document store

## Runtime Validation Commands

- `cargo check`
- `cargo check -p renderer`
- `cargo check -p renderer --examples`
- `cargo check -p input`

Headless smoke pattern:

- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_pbr`
- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_unlit`
- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_model_load`
- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example demo_async_loading`
- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test`
- `RUST_LOG=debug timeout --signal=INT 45s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr`

Treat successful startup logs with no fatal errors before timeout as a smoke pass.

## External Baseline

When behavior is ambiguous in glTF/PBR/IBL flows, use:

- `https://github.com/SaschaWillems/Vulkan-glTF-PBR`

as conceptual lineage, not a drop-in implementation source.
