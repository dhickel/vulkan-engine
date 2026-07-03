# Validation Matrix

| Area | Required Evidence | Command Or Check | Pass Criteria | Stop Condition |
|---|---|---|---|---|
| Default compile | Command result | `cargo check -p renderer` | Succeeds without `advanced-interop` | Any Sprint 10 compile error |
| Default examples | Command result | `cargo check -p renderer --examples` | Beginner/diagnostic examples compile without advanced feature | Any example requires advanced imports |
| Feature compile | Command result | `cargo check -p renderer --features advanced-interop` | Feature-gated advanced module/rendergraph compile | Any feature-only compile error |
| Feature examples | Command result | `cargo check -p renderer --examples --features advanced-interop` | Examples compile when feature is enabled | Any feature-mode regression |
| Workspace smoke | Command result | `cargo check` | Workspace still compiles or known blocker recorded | New broad compile failure from Sprint 10 |
| Hooks/tests | Test result | Focused `cargo test -p renderer <filter>` or full renderer test if practical | Changed hook/advanced logic covered | Missing test for changed behavior |
| Runtime smoke | Debug report | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --record_debug=10 --record_debug_interval=50 --record_debug_path=.internal-dev/debug_reports/sprint-10-api-test-timing.jsonl` | Startup reaches frames and writes timing JSONL, no fatal errors before timeout | Runtime initialization failure or missing report |
| Headless capture | PNG/sidecar evidence | Conditional headless draw command from implementation notes | Required only for visible/capture/readback changes; sidecar status succeeded and PNG exists | Missing/inconclusive capture when required |
| Docs drift | Manual validator review | Inspect changed docs plus both hook docs | No contradictory hook/advanced claims remain | Docs conflict with live code |
| Evidence consistency | JSON/report review | Inspect `artifacts/validation-summary.json` and phase reports | Top-level status matches phase/command/capture reality | Any false `fully_validated`/passed claim |

## Justified Omissions

- Browser/Playwright validation does not apply; this is a Rust desktop renderer/API sprint.
- Desktop screenshots are forbidden for renderer proof; use engine-owned headless capture when proof is required.
- Headless capture can be omitted for docs-only, compile-only, or pure feature-gate classification changes.
