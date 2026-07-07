# Sprint 01 Validation Matrix

| Gate | Required For | Commands / Evidence | Pass Criteria | Stop / Fail Criteria |
|---|---|---|---|---|
| Branch hygiene | Every phase | `git status --short --branch` | On `sprint/alpha-01-baseline-audit`; only in-scope changes plus preserved `.idea/engine.iml` and `.reasonix/` | Unexpected dirty product code or wrong branch |
| Process docs | Phase 01 | `.internal-dev/AGENTS.md` review, root guide consistency check | Active guide exists and matches internal-dev contract | Only archived guide exists or root docs point to missing active process |
| Workspace docs | Phase 02 | Compare docs to root `Cargo.toml` | Docs mention all live workspace members and correct runtime entrypoints | Docs continue to describe renderer/input-only workspace as current |
| Stale gap report | Phase 02 | `rg "gap-report|known limitations|No audio|No physics|No scripting|No project system|No scene serialization|headless" README.md AGENTS.md docs` | Stale report is superseded/replaced or links no longer treat it as current truth | Future sprint docs can still cite stale report as current truth |
| Residual register | Phase 03 | `.internal-dev/reviews/2026-07-03-alpha-baseline-register.md`; `rg -n "verified_current|stale_resolved|unknown_needs_audit|accepted_alpha_debt|blocked_validation" .internal-dev/reviews .internal-dev/bugs .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit` | Findings classify historical claims as verified, stale, unknown, accepted debt, or blocked validation; each item includes evidence, impact, next action, and likely owner when known | Historical claims are copied as current defects without verification, or out-of-scope bugs are silently discarded |
| Markdown/path integrity | Docs or artifact phases that modify Markdown links | Prefer local markdown tooling if available; otherwise manually inspect modified relative links and record method in the phase report | Modified links/path references resolve or are explicitly non-clickable evidence paths | Link tooling exists but is skipped, or modified links are claimed valid without an inspection method |
| Compile baseline | Phase 04, earlier if useful | `cargo check`; package checks | Commands pass or blockers recorded with exact failure evidence in `validation/phase-04-validation-report.md` and `artifacts/validation-summary.json` | Missing command evidence, hidden failure, or untracked blocker |
| Input tests | Phase 04 | `cargo test -p input` | Passes or blocker recorded in the validation report and evidence index | Failure not recorded or in-scope regression introduced |
| Renderer examples compile | Phase 04 | `cargo check -p renderer --examples` | Passes or blocker recorded in the validation report and evidence index | Failure hidden or incorrectly marked pass |
| Runtime smoke | Phase 04 if selected | Debug-record command under `.internal-dev/debug_reports/` using `--record_debug=10 --record_debug_interval=50` | Startup has no fatal error before timeout; JSONL path recorded in report and evidence index | Fatal error, missing debug output when claimed, or command omitted without reason |
| Capture readiness | Phase 04 or visual sprint only if needed | Headless capture skill evidence under `.internal-dev/captures/` | PNG/JSON produced and inspected against expected criteria; reason recorded when capture is not required | Capture required but not run, output missing, or visual result inconclusive |
| Evidence index | Every phase | `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-01-alpha-baseline-audit/artifacts/validation-summary.json`; `python -m json.tool ... >/dev/null` when JSON changes | Status names match actual completed gates; phase reports, command results, capture status, pushed refs, email evidence, and residual risks are indexed | Top-level status overclaims validation, JSON is invalid, or required evidence paths are missing |
| Commit/push | Every phase unless the user explicitly assigns gates to parent/main thread | `git rev-parse HEAD`, `git push`, `git ls-remote origin refs/heads/sprint/alpha-01-baseline-audit`, remote links | In-scope commit pushed to branch, or worker report explicitly records that parent/main thread owns the gate | Push failure, missing hash, missing ref, unrelated files committed, or worker claims a gate it did not perform |
| Email report | Every phase unless the user explicitly assigns gates to parent/main thread | AgentMail sent item/thread evidence | HTML report includes all required fields, or worker report explicitly records that parent/main thread owns the gate | Email missing, plain report missing required fields, send failure, or worker claims an email it did not send |

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
