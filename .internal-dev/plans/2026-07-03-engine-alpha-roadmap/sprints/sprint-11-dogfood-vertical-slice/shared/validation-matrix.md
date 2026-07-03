# Validation Matrix

| Gate | Required Evidence | Command Or Inspection | Owner | Status At Plan Creation |
|---|---|---|---|---|
| Phase 01 audit | Current-state report and API friction inventory | Read target files, `git status --short`, contract comparison | Phase 01 worker + validator | pending |
| Package/project data | Dogfood project/package/scene validate | `engine_pack validate-*` commands | Phase 02 worker + validator | pending |
| Dogfood compile | App compiles with project contract path | `cargo check -p dungeon_dogfood` | Phase 03 worker + validator | pending |
| Workspace compile | Shared changes do not break workspace | `cargo check`, scoped package checks | Phase validators | pending |
| Input/camera gameplay | Focused tests and runtime smoke | unit tests plus timeout dogfood runs | Phase 03 worker + validator | pending |
| Runtime debug | Timing JSONL written under `.internal-dev/debug_reports/` | dogfood `--record_debug*` command | Phase 03/04 worker + validator | pending |
| Headless visual | Draw-target captures and sidecars inspected | dogfood `--headless --capture_target draw` command | Phase 04 worker + validator | pending |
| Docs | Public docs match commands and limitations | docs review plus stale-reference sweep | Phase 05 worker + validator | pending |
| Evidence index | Conservative JSON status matches reports | inspect `artifacts/validation-summary.json` | Final quality validator | pending |
| Closeout | Report/email draft ready, changelog timing gate noted | `reports/final-report.md`, no tracker edit | Main thread after validation | pending |

## Minimum Command Set Before Final Review

```sh
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p input
cargo check -p engine_pack
cargo test -p engine_pack
cargo check -p dungeon_dogfood
cargo run -p engine_pack -- validate-package apps/dungeon_dogfood/assets/dogfood_dungeon.package.toml --expected-package-id dogfood_dungeon
cargo run -p engine_pack -- validate-project apps/dungeon_dogfood/engine.project.toml
cargo run -p engine_pack -- validate-scene apps/dungeon_dogfood/scenes/start.engine.scene.json --project apps/dungeon_dogfood/engine.project.toml
```

## Runtime/Capture Gates

Runtime smoke commands must be timeout-bound and report whether startup reached the event loop/render path without fatal errors before timeout.

Headless capture proof must inspect:

- at least one PNG;
- corresponding sidecar JSON;
- `status = "succeeded"` or equivalent live field;
- `capture_target = "draw"`;
- positive extent;
- non-empty, non-blank visual content;
- full-content flags enabled for final baseline.

## Validation Report Paths

- `validation/phase-01-validation-report.md`
- `validation/phase-02-validation-report.md`
- `validation/phase-03-validation-report.md`
- `validation/phase-04-validation-report.md`
- `validation/phase-05-validation-report.md`
- `validation/final-quality-review.md`
