# Sprint 01 Validation Matrix

| Gate | Required For | Commands / Evidence | Pass Criteria | Stop / Fail Criteria |
|---|---|---|---|---|
| Branch hygiene | Every phase | `git status --short --branch` | On `sprint/alpha-01-baseline-audit`; only in-scope changes plus preserved `.idea/engine.iml` and `.reasonix/` | Unexpected dirty product code or wrong branch |
| Process docs | Phase 01 | `.internal-dev/AGENTS.md` review, root guide consistency check | Active guide exists and matches internal-dev contract | Only archived guide exists or root docs point to missing active process |
| Workspace docs | Phase 02 | Compare docs to root `Cargo.toml` | Docs mention all live workspace members and correct runtime entrypoints | Docs continue to describe renderer/input-only workspace as current |
| Stale gap report | Phase 02 | `rg "gap-report|known limitations|no audio|no physics|no scripting|no scene serialization|headless"` targeted sweep | Stale report is superseded/replaced or links no longer treat it as current truth | Future sprint docs can still cite stale report as current truth |
| Residual register | Phase 03 | Register file under `.internal-dev/reviews/` or `.internal-dev/bugs/` | Findings classified as verified, stale, unknown, accepted debt, or blocked validation | Historical claims are copied as current defects without verification |
| Compile baseline | Phase 04, earlier if useful | `cargo check`; package checks | Commands pass or blockers recorded with exact failure evidence | Missing command evidence, hidden failure, or untracked blocker |
| Input tests | Phase 04 | `cargo test -p input` | Passes or blocker recorded | Failure not recorded or in-scope regression introduced |
| Renderer examples compile | Phase 04 | `cargo check -p renderer --examples` | Passes or blocker recorded | Failure hidden or incorrectly marked pass |
| Runtime smoke | Phase 04 if selected | Debug-record command under `.internal-dev/debug_reports/` | Startup has no fatal error before timeout; JSONL path recorded | Fatal error, missing debug output when claimed, or command omitted without reason |
| Capture readiness | Phase 04 only if needed | Headless capture skill evidence under `.internal-dev/captures/` | PNG/JSON produced and inspected against expected criteria | Capture required but not run, output missing, or visual result inconclusive |
| Evidence index | Every phase | `artifacts/validation-summary.json` | Status names match actual completed gates | Top-level status overclaims validation |
| Commit/push | Every phase | `git rev-parse HEAD`, `git push`, remote links | In-scope commit pushed to branch | Push failure, missing hash, missing ref, or unrelated files committed |
| Email report | Every phase | AgentMail sent item/thread evidence | HTML report includes all required fields | Email missing, plain report missing required fields, or send failure |

## Final Quality Gate

After all phases pass, run a final quality validation pass that reconciles:

- phase directives;
- validation reports;
- command evidence;
- pushed commits;
- email reports;
- stale-reference sweep;
- validation-summary status.

Final pass may be `final_quality_review_passed_with_residuals` when all residuals are explicitly tracked and accepted for later sprints.
