# Phase 02 Docs Drift Audit

Date: 2026-07-03

## Scope

Phase 02 repaired public documentation drift caused by root docs and indexes treating the workspace as renderer/input-only and by `docs/gap-report.md` presenting stale historical claims as current truth.

## Repaired Drift

- `AGENTS.md` now lists all root `Cargo.toml` workspace members: `src/input`, `src/renderer`, `src/audio`, `src/physics`, `src/scripting`, `apps/dungeon_dogfood`, and `apps/editor`.
- `README.md` now identifies the root binary as a migration stub, points runtime use to renderer examples, and links the current alpha readiness baseline.
- `docs/api/00-index.md` and `docs/internal/00-index.md` now include workspace context and point known-limitations traffic to the current alpha readiness baseline instead of a stale raw gap list.
- `docs/gap-report.md` keeps its stable path but is rewritten as the current alpha readiness baseline. Stale subsystem absence claims are labeled superseded and routed to Phase 03 residual classification.
- `docs/internal/04-vulkan-subsystem.md` no longer points to a numbered gap-report item that disappeared when the gap report became the alpha readiness baseline, and its touched source links now resolve from `docs/internal/`.

## Deferred Classification

Phase 02 did not verify every historical gap item. The rewritten report keeps historical candidates visible but marks them as residual inputs only. Phase 03 must classify each relevant item as verified, stale, unknown, accepted debt, or blocked validation using live source evidence.

## Governance Note

The Phase 02 directive named `.internal-dev` artifact edits but did not list `.internal-dev/AGENTS.md` as a supporting input. The repo guide requires that governance file for `.internal-dev` work, so it was read narrowly before editing sprint artifacts.

## Product Code

No Rust source was edited.
