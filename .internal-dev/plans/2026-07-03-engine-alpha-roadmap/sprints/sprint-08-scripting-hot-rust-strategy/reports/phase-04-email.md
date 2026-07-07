# Sprint 08 Phase 04 Report: Docs And Final Validation Prep

Date: 2026-07-03

## Summary

Phase 04 aligned public docs and closeout evidence with the implemented Sprint 08 scope. The docs now state that Rust app crates remain the primary custom behavior path, `engine_pack new-app` is a standalone support-crate scaffold, scripting is experimental around script-ID-aware eval/log/event/error boundaries, package-level script assets remain deferred, and dynamic Rust/hot Rust runtime reload remains deferred/tooling-only.

No product behavior was added. No desktop screenshots or renderer captures were used because Sprint 08 Phase 04 was non-visual and did not change renderer/editor visible behavior.

## Files Changed

| File | Reason |
|---|---|
| `README.md` | Replaced stale blanket "generated app templates deferred" wording with the implemented `engine_pack new-app` support scaffold status, experimental scripting scope, and deferred runtime reload/package-script boundaries. |
| `docs/api/00-index.md` | Added the `new-app` support scaffold to the API overview and narrowed deferred wording to renderer-window templates, production scripting scheduling, package-level script assets, and hot Rust reload. |
| `docs/api/01-student-quickstart.md` | Replaced generic scripting-runtime deferred wording with the narrower production scheduling/package-script deferred language. |
| `docs/api/07-engine-arguments.md` | Added a pointer to `engine_pack new-app` for support scaffolding and preserved deferred status for renderer-window templates and runtime reload. |
| `docs/api/09-editor-asset-browser-and-wall-chunks.md` | Clarified the editor limitation is scripting UI and renderer-window generated app templates, not all app scaffolding. |
| `docs/api/11-runtime-project-launcher.md` | Clarified that the root launcher is not production scripting scheduling and that generated support scaffolds do not imply renderer-window app generation. |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/validation-summary.json` | Updated the evidence index with Phase 04 commands, capture status, residuals, and conservative pending-validator status. |
| `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-04-email.md` | This phase closeout draft. |

## Command Matrix

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --check` | Pass | No formatting drift. |
| `cargo check` | Pass | Existing renderer warning noise remains. |
| `cargo test -p scripting` | Pass | 9 tests passed. |
| `cargo test -p engine_events` | Pass | 7 tests passed. |
| `cargo test -p renderer` | Pass | 160 unit tests and 17 integration tests passed; 5 doctests ignored; existing renderer warnings remain. |
| `cargo test -p engine_pack` | Pass | 20 CLI tests passed, including generated `new-app` standalone check; existing renderer warnings remain. |
| `cargo check -p renderer --examples` | Pass | Existing renderer warnings remain. |
| `cargo check -p editor` | Pass | Existing renderer warnings plus one editor dead-code warning. |
| `cargo check -p dungeon_dogfood` | Pass | Existing renderer warnings plus dogfood dead-code warnings. |
| `cargo test -p dungeon_dogfood` | Not applicable | Sprint 08 did not change dogfood expectations, and the directive says to run this only if dogfood expectations changed. |
| `rg -n "/tmp\|pending\|planned\|not implemented\|TODO\|desktop screenshot\|generated app templates\|scripting runtime\|hot Rust\|dynamic Rust" docs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy` | Pass with expected matches | Current docs matches are command examples, explicit deferred status, or unrelated renderer/internal wording. Historical sprint artifacts still contain earlier phase wording and were left unchanged. |

## Capture Status

Not applicable. Phase 04 changed documentation and evidence only. No visible renderer/editor behavior changed, so true headless draw capture was not run. No desktop screenshots were taken.

## Residuals

- Phase 04 independent validation is still pending: `validation/phase-04-validation-report.md` does not exist yet.
- Final quality review is still pending: `validation/final-quality-review.md` does not exist yet.
- `artifacts/validation-summary.json` deliberately keeps `fully_validated` as `false`.
- Protected local state remains visible and was not touched: `M .idea/engine.iml` and `?? .reasonix/`.
- Renderer-window generated app templates, package-level script assets, production scripting scheduling, and dynamic Rust/runtime reload remain deferred.

## Final-Review Readiness

Ready for Phase 04 validator and final quality review. The final reviewer should verify the docs wording against the implemented code/tests, confirm the stale-reference matches are either expected current deferred wording or historical artifacts, and keep the evidence index conservative until both validation reports pass.
