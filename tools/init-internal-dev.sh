#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
ROOT="${TARGET_DIR%/}/.internal-dev"

mkdir -p "$ROOT"
mkdir -p "$ROOT/bugs" \
         "$ROOT/plans" \
         "$ROOT/reviews" \
         "$ROOT/notes" \
         "$ROOT/knowledge" \
         "$ROOT/changelogs" \
         "$ROOT/debug_reports" \
         "$ROOT/bugs/.archive" \
         "$ROOT/plans/.archive" \
         "$ROOT/reviews/.archive" \
         "$ROOT/notes/.archive" \
         "$ROOT/knowledge/.archive" \
         "$ROOT/changelogs/.archive"

# Create seed files only when missing (never overwrite user content).
if [[ ! -f "$ROOT/notes/future_consideration.md" ]]; then
  cat > "$ROOT/notes/future_consideration.md" <<'NOTE_EOF'
# Future Considerations

Use this file for deferred improvements and concerns that should be addressed later.
NOTE_EOF
fi

if [[ ! -f "$ROOT/AGENTS.md" ]]; then
  cat > "$ROOT/AGENTS.md" <<'AGENTS_EOF'
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

- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test`
- `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr`

Treat successful startup logs with no fatal errors before timeout as a smoke pass.

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
AGENTS_EOF
fi

echo "Initialized (without overwriting existing files): $ROOT"
echo
cat <<'MSG_EOF'
Add this to your top level AGENTS.md : ## `.internal-dev` Development Document Store

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
MSG_EOF
