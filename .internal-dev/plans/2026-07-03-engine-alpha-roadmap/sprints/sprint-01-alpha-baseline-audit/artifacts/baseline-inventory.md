# Phase 01 Baseline Inventory

Date: 2026-07-03

## Git Baseline

- Branch: `sprint/alpha-01-baseline-audit`
- HEAD: `ce43008098fb19bad0cce8fd965f908870d8b988`
- Remote: `origin https://github.com/dhickel/vulkan-engine.git`

## Dirty State To Preserve

Initial `git status --short --branch -uall` showed only the user-preserved unrelated dirt:

```text
## sprint/alpha-01-baseline-audit
 M .idea/engine.iml
?? .reasonix/truncated-results/1782968061211-b71854bf-run_command.txt
?? .reasonix/truncated-results/1782974551246-1abc8cd2-run_command.txt
```

These files/directories were not edited:

- `.idea/engine.iml`
- `.reasonix/`

## Workspace Members

Root `Cargo.toml` declares these workspace members:

- `src/input`
- `src/renderer`
- `src/audio`
- `src/physics`
- `src/scripting`
- `apps/dungeon_dogfood`
- `apps/editor`

Available package manifests observed:

- `Cargo.toml` (`engine`)
- `src/input/Cargo.toml` (`input`)
- `src/renderer/Cargo.toml` (`renderer`)
- `src/audio/Cargo.toml` (`audio`)
- `src/physics/Cargo.toml` (`physics`)
- `src/scripting/Cargo.toml` (`scripting`)
- `apps/dungeon_dogfood/Cargo.toml` (`dungeon_dogfood`)
- `apps/editor/Cargo.toml` (`editor`)

## Process Guide Status

- Before Phase 01, `.internal-dev/AGENTS.md` was missing.
- `.internal-dev/.archive/AGENTS.md` was available and contained the active process contract needed for restoration.
- Phase 01 restored `.internal-dev/AGENTS.md` from the archived guide and updated wording only for current access discipline, archive status, evidence directories, and capture readiness.

## Docs And Process Drift For Later Phases

Evidence-only observations for later sprint phases:

- Root `AGENTS.md` says the workspace crates are only `src/renderer` and `src/input`, but root `Cargo.toml` now declares seven workspace members.
- Root `AGENTS.md` repository layout lists `Cargo.toml` as workspace root with `engine`, `src/input`, and `src/renderer`; it omits `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, and `apps/editor`.
- Root `AGENTS.md` references `.internal-dev/AGENTS.md`; Phase 01 restored that active guide.
- Stale gap-report defects were not treated as current because this phase did not verify historical gap reports.

## Validation Capability Baseline

- Phase 01 required process/document validation only.
- Compile checks were not required because only allowed `.internal-dev` documentation/artifact files were edited.
- Capture validation was not run because Phase 01 changed no visual rendering behavior.
- Capture readiness exists via `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`; future visual validation should define expected visual behavior first, run timeout-bound headless captures, and record PNG plus sidecar JSON evidence under `.internal-dev/captures/`.
