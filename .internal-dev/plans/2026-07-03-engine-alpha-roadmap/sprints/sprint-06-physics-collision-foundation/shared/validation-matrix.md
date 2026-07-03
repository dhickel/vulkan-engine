# Validation Matrix

| Gate | Phase | Required Evidence | Pass Criteria | Failure Routing |
|---|---:|---|---|---|
| Physics crate API | 01 | `cargo test -p physics`, API/tests review | Durable IDs/descriptors/queries/events compile and pass without Vulkan | Fresh scoped code repair worker |
| Physics dependency hygiene | 01 | import/dependency scan, `cargo tree -p physics` if useful | No renderer/Vulkan/editor/dogfood dependency in physics | Fresh scoped code repair worker |
| Workspace sanity | 01 | `cargo check` | Workspace still checks; new warnings are explained/fixed | Fresh scoped code repair worker |
| Package metadata validation | 02 | `cargo test -p renderer` targeted tests | Valid metadata accepts; bad shapes/body kinds/dimensions/runtime handles reject | Fresh scoped code repair worker |
| Scene metadata validation | 02 | `cargo test -p renderer` targeted tests | Scene collision metadata round-trips and rejects invalid/duplicate/handle-shaped data | Fresh scoped code repair worker |
| CLI validation | 02 | `cargo test -p engine_pack`, `cargo check -p engine_pack` | `engine_pack` reports renderer validation failures for collision metadata | Fresh scoped code repair worker |
| Event bridge | 03 | `cargo test -p physics`, `cargo test -p engine_events` | Physics records map to `EngineEvent::Physics` with durable IDs/phases | Fresh scoped code repair worker |
| Dogfood gate | 03 | `cargo check -p dungeon_dogfood`, dogfood tests if touched, debt/proof report | Either narrow proof works with tests or migration debt is recorded | Fresh scoped repair or docs/evidence repair |
| Docs | 04 | docs review and stale-reference sweep | Docs match implemented alpha contracts and deferred limits | Docs/evidence repair worker |
| Full compile/test | 04 | required final commands | All required checks pass or blockers are recorded conservatively | Repair by failed target |
| Runtime smoke | 04 | debug report only if runtime/app behavior changed | Startup has no fatal errors before timeout | Runtime repair worker |
| Headless draw capture | 04 | capture dir only if visible behavior changed | True engine-owned `--headless --capture_target draw` capture succeeds and is reconciled by validator | Harness repair first; product repair only after proven bug |
| Final quality | 04 | `validation/final-quality-review.md`, validation summary | Plan criteria, reports, code/docs/tests/evidence are consistent | Planning revision or scoped repair depending on defect |

## Required Final Commands

```bash
cargo fmt --check
cargo check
cargo test -p physics
cargo test -p engine_events
cargo test -p renderer
cargo test -p engine_pack
cargo check -p physics
cargo check -p renderer --examples
cargo check -p editor
cargo check -p dungeon_dogfood
```

If runtime/app behavior changed:

```bash
RUST_LOG=debug timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-06-physics-collision-foundation/root-runtime-physics-timing.jsonl
```

If visible renderer/editor behavior changed, additionally run true headless draw capture through `.internal-dev/skills/engine-headless-capture-validation/SKILL.md` with `--headless --capture_target draw`; no desktop screenshots.

## Evidence Consistency Rules

- If any required phase validator is missing or failed, top-level status must not be `fully_validated`.
- If visible behavior changed and capture proof is pending, top-level status must not be `fully_validated`.
- If dogfood migration is deferred, the debt artifact must be listed as an accepted residual, not hidden.
- If a tooling fallback is used, record `TOOLING_CONSTRAINT` and main-thread approval.
- Superseded artifacts must be listed in `artifacts/validation-summary.json`.
