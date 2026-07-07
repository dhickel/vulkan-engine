# Phase 03 Validator Review: Dev Loop Docs

Date: 2026-07-03
Branch: `sprint/alpha-04-runtime-launcher`
Validator scope: Sprint 04 Phase 03 docs-only validation.

## Decision

PASS

Phase 03 passes after remediation. The previously blocking stale guidance in `docs/api/02-renderer-lifecycle-and-frame-api.md` now distinguishes the root project launcher, renderer facade diagnostics, and custom app crates. The requested command set passed, and remaining stale-sweep hits are accepted contextual/deferred wording rather than contradictory runtime guidance.

## Findings

### Blocking

No blocking findings remain.

The prior finding against `docs/api/02-renderer-lifecycle-and-frame-api.md` is remediated:

- line 12 now says data-driven projects run through `cargo run -- --project apps/editor/sample_project/engine.project.toml`;
- line 13 now classifies renderer examples as facade lifecycle diagnostics;
- line 14 now points custom Rust apps to `cargo run -p <app>`;
- line 114 now describes wrong-target startup as a three-path choice rather than root `cargo run` as a migration trap;
- lines 119 and 133-134 now link project-launcher issues to the root launcher and renderer diagnostics to `api_test`.

### Non-Blocking Residuals

- `cargo run -- --help` emits existing renderer dead-code warnings before printing launcher help. The Phase 03 report already classifies these as existing warnings.
- The canonical evidence index remains conservative with `fully_validated: false` and Phase 04 pending.
- `apps/dungeon_dogfood/README.md` contains the required dogfood custom-app wording, but it is not currently shown as a modified file in `git status`. This is not a Phase 03 blocker because the live content satisfies the criterion.
- The broader stale wording sweep still reports accepted context hits for explicit deferred feature lists, schema/content migration wording, internal legacy stub documentation, and correct root-launcher guidance.

## Criteria Results

| Criterion | Result | Evidence |
| --- | --- | --- |
| Root launcher sample command is documented exactly | Pass | `README.md:21-22`, `docs/api/11-runtime-project-launcher.md:11-15`, `docs/api/01-student-quickstart.md:39-41` |
| True headless draw-target capture command is documented exactly | Pass | `README.md:23-24`, `docs/api/11-runtime-project-launcher.md:29-40`, `docs/api/01-student-quickstart.md:43-51` |
| Capture proof language rejects desktop/compositor/present evidence | Pass | `docs/api/11-runtime-project-launcher.md:42` |
| App crates under `apps/<name>` are documented as the custom Rust loop | Pass | `README.md:25-26`, `docs/api/11-runtime-project-launcher.md:87-95` |
| Dogfood is documented as a custom app crate and migration remains deferred | Pass | `apps/dungeon_dogfood/README.md:7-13` |
| Deferred hot reload/scripting/event/physics/audio/generated templates are explicit and not overclaimed | Pass | `README.md:27`, `docs/api/11-runtime-project-launcher.md:7`, `docs/api/11-runtime-project-launcher.md:95`, `docs/api/10-packaging-cli.md:124-143` |
| Runtime project launcher is no longer listed as deferred | Pass | Targeted stale sweep had no `runtime project launcher.*deferred` hits; `docs/api/09-editor-asset-browser-and-wall-chunks.md:143-146` points to root launcher instead. |
| Docs no longer claim root `cargo run` is a migration stub or renderer examples are the runtime path | Pass | `docs/api/02-renderer-lifecycle-and-frame-api.md:12-14`, `docs/api/02-renderer-lifecycle-and-frame-api.md:114`, `docs/api/02-renderer-lifecycle-and-frame-api.md:119`, `docs/api/02-renderer-lifecycle-and-frame-api.md:133-134` |
| Stale-reference sweep results are reconciled | Pass | Targeted sweep has accepted deferred-feature hits only; broader sweep has accepted context hits only. |
| `artifacts/validation-summary.json` remains parseable and conservative | Pass | `python -m json.tool .../validation-summary.json >/dev/null` passed; summary has `fully_validated: false` and Phase 04 pending. |

## Commands Run

| Command | Result |
| --- | --- |
| `cargo run -- --help` | Pass, exit 0. Help reports `Usage: engine --project <path> [options]` and lists `--headless`, `--capture_target <present\|draw>`, capture sequence flags, and debug timing flags. Existing renderer dead-code warnings were observed before help output. |
| `rg -n 'migration stub\|runtime project launcher.*deferred\|renderer examples.*only runtime\|cargo run\` prints\|dynamic Rust hot reload\|scripting implemented\|physics implemented\|audio implemented\|prints migration guidance' README.md docs apps/dungeon_dogfood` | Pass with accepted hits only. Hits were deferred-feature wording in `docs/api/11-runtime-project-launcher.md`, `docs/api/10-packaging-cli.md`, `docs/api/09-editor-asset-browser-and-wall-chunks.md`, and `docs/api/01-student-quickstart.md`. |
| `python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json >/dev/null` | Pass, exit 0. |
| `git diff --check` | Pass, exit 0. |
| `rg -n -i 'migration\|stub\|only prints\|prints .*guidance\|deferred\|only runtime\|canonical runtime\|primary .*runtime\|root .*cargo run\|cargo run .*root\|runtime project launcher\|project launcher' README.md docs apps/dungeon_dogfood` | Pass with accepted context hits only. The prior lifecycle-doc stale wording no longer appears. Accepted hits include correct root launcher docs, explicit deferred-feature lists, schema/content migration wording, and internal legacy stub/deferred cleanup references. |

Compile checks were not rerun in this validator pass. The main thread and Phase 03 report record prior passes for `cargo fmt --check`, `cargo check -p engine`, `cargo check -p editor`, and `cargo check -p engine_pack --locked`.

## Remediation Outcome

The previous `docs_or_evidence_defect` is resolved. No further Phase 03 repair handoff is required.

## Residual Risk

No browser proof is applicable for this docs-only phase. Phase 04 capture/debug closeout remains pending and must continue to require true headless draw-target proof with `--headless --capture_target draw`; desktop screenshots, compositor screenshots, and present-target captures remain invalid.
