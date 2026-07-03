# Sprint 08 Phase 01 Validation Report

Date: 2026-07-03
Validator: Codex validation agent

## Verdict

Pass after focused revalidation.

Phase 01 satisfies the contract audit scope and correctly identifies the live scripting, event, script-asset, app-template, docs-drift, inherited-residual, and Phase 02 readiness state. The prior inaccurate quickstart citation was repaired: the audit now cites `docs/api/01-student-quickstart.md:101` for deferred scripting/runtime/generated-template/hot Rust claims and records the correction in its repair notes.

Capture validation is not applicable because Phase 01 is non-visual and made no renderer/editor visual behavior claim.

## Findings

### F1 - Resolved: inaccurate quickstart citation in verified docs-deferred claim

Severity: Medium
Classification: docs_or_evidence_defect

Revalidation result: Pass.

The audit previously cited `docs/api/01-student-quickstart.md:86-88` for deferred dynamic Rust hot reload, scripting runtime execution, and generated app templates. The repaired audit now cites `docs/api/01-student-quickstart.md:101`, which is the source-backed deferred-feature statement. The audit also includes a repair note explaining that the original cited lines were only custom app pseudocode.

Evidence:

- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md:21`
- `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md:66`
- `docs/api/01-student-quickstart.md:101`
- `rg --no-ignore -n "01-student-quickstart\\.md:(86|87|88)|docs/api/01-student-quickstart.md:86-88|86-88" .../artifacts/phase-01-current-state-contract-audit.md .../reports/phase-01-email.md`

### R1 - Protected local state is dirty and cannot be attributed from this validation pass

Severity: Low / Residual

`git status --short` currently reports `M .idea/engine.iml` and `?? .reasonix/`. The audit records those as pre-existing unrelated local changes and states they were not touched. This validation pass confirms no tracked product code, docs, tests, or Cargo files are dirty, but it cannot independently prove when the `.idea/engine.iml` and `.reasonix/` changes were created.

Evidence:

- `git status --short` -> `M .idea/engine.iml`, `?? .reasonix/`
- `git diff --name-only` -> `.idea/engine.iml`
- `git ls-files --others --exclude-standard` -> `.reasonix/truncated-results/...`
- `git diff -- .idea/engine.iml` shows only IDE source-folder entries for existing workspace crates/tools.

No remediation is requested for Phase 01 unless the orchestrator requires a clean protected-file baseline before dispatching Phase 02.

## Criteria Results

| Criterion | Result | Evidence |
|---|---:|---|
| Read worker directive and applicable governance docs | Pass | Read root `AGENTS.md`, `.internal-dev/AGENTS.md`, phase directive, specification lock, current-state analysis, target design, and senior-engineer guidance. |
| Audit artifact exists | Pass | `.internal-dev/.../artifacts/phase-01-current-state-contract-audit.md` exists and has scope, verified facts, drift, opportunities, blockers, readiness, and evidence sections. |
| Phase report exists and is email-ready | Pass | `.internal-dev/.../reports/phase-01-email.md` exists and summarizes outcome, readiness, files, and validation commands. |
| Audit names exact files and claims to update | Pass | Prior quickstart deferred-support citation was corrected to `docs/api/01-student-quickstart.md:101`; see F1. |
| Audit confirms whether `script` asset kind exists | Pass | Correctly states no `AssetKind::Script`; verified in `src/renderer/src/data/asset_registry.rs:122-175`. |
| Audit confirms whether app-template tooling exists | Pass | Correctly states no `new-app` or template command; verified in `tools/engine_pack/src/main.rs:39-48` and usage lines `589-597`. |
| Audit identifies current scripting API | Pass | Correctly identifies Rhai wrapper with log functions, `eval`, `eval_with_scope`, `eval_file`, `engine_mut`, and `new_scope`; verified in `src/scripting/src/lib.rs:15-65`. |
| Audit identifies scripting event vocabulary | Pass | Correctly identifies `ScriptId`, `EngineEvent::Scripting`, `ScriptingEvent::ScriptEmitted`, and `ScriptingEvent::ScriptError`; verified in `src/events/src/lib.rs:64-74`, `113-120`, and `223-234`. |
| Audit identifies docs drift | Pass | Correctly flags `src/scripting/src/lib.rs:1-4` overclaim and `docs/api/10-packaging-cli.md:92` script-record wording as misleading. |
| Audit records inherited Sprint 07 residual handling | Pass | Correctly records the conditional `cargo test -p dungeon_dogfood` / renderer test-profile `russimp_sys` residual from current-state analysis. |
| Audit identifies Phase 02 readiness | Pass | Correctly says Phase 02 can proceed if it stays inside `tools/engine_pack`, emits a minimal public-facade app crate, and tests generated output builds. |
| No product code/docs/tests/Cargo files changed | Pass | `git diff --name-only` shows only `.idea/engine.iml`; no tracked product code, docs, tests, or Cargo files are modified. |
| `.idea/engine.iml` and `.reasonix` protected | Residual | Current worktree has dirty `.idea/engine.iml` and untracked `.reasonix/`; audit says they were pre-existing and untouched, but this validation cannot independently attribute them. |
| Capture not applicable | Pass | Phase 01 is non-visual; no renderer/editor visual behavior changed or claimed. |

## Commands And Evidence

Commands run from `/home/hickelpickle/Code/Rust/engine`:

```bash
rg -n "Rust engine|engine-headless|Sprint 08|scripting hot|hot rust|2026-07-03-engine-alpha-roadmap|reasonix|engine.iml" /home/hickelpickle/.codex/memories/MEMORY.md
pwd && rg --files -g 'AGENTS.md' -g '!target'
sed -n '1,240p' AGENTS.md
sed -n '1,260p' .internal-dev/AGENTS.md
sed -n '1,280p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/worker-directives/phase-01-current-state-contract-audit.md
sed -n '1,320p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md
sed -n '1,220p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md
git status --short
git diff --name-only
git ls-files --others --exclude-standard
nl -ba docs/api/00-index.md | sed -n '1,120p'
nl -ba docs/api/01-student-quickstart.md | sed -n '70,100p'
nl -ba docs/api/10-packaging-cli.md | sed -n '1,140p'
nl -ba docs/api/11-runtime-project-launcher.md | sed -n '1,130p'
nl -ba docs/api/12-events-and-lifecycle.md | sed -n '1,90p'
nl -ba src/scripting/src/lib.rs | sed -n '1,140p'
nl -ba src/events/src/lib.rs | sed -n '1,380p'
nl -ba src/renderer/src/data/asset_registry.rs | sed -n '100,190p'
nl -ba tools/engine_pack/src/main.rs | sed -n '1,90p;470,520p'
nl -ba tools/engine_pack/tests/cli_validation.rs | sed -n '1,90p;270,365p'
sed -n '1,220p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/00-specification-lock.md
sed -n '1,220p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/01-current-state-analysis.md
sed -n '1,240p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/02-target-design.md
sed -n '1,220p' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/shared/senior-engineer-guidance.md
rg -n "scripting runtime|generated app templates|hot reload|dynamic Rust|app crates|script" docs src tools apps -g '*.md' -g '*.rs' -g '*.toml'
rg -n "Script|script|scripting|hot reload|dynamic Rust|generated app|template|app crates|apps/<name>" docs/api/01-student-quickstart.md docs/api/00-index.md docs/api/10-packaging-cli.md docs/api/11-runtime-project-launcher.md docs/api/12-events-and-lifecycle.md
rg -n "Script|script|scripting|engine_events|EngineEvent|ScriptingEvent" src/scripting/src/lib.rs src/scripting/Cargo.toml src/events/Cargo.toml src/events/src/lib.rs
rg -n "AssetKind|script|new-app|new app|template|generated|validate-package|new-project|new-package|scan-assets|add-asset|pack" tools/engine_pack/src/main.rs tools/engine_pack/tests/cli_validation.rs src/renderer/src/data/asset_registry.rs
git diff -- .idea/engine.iml | sed -n '1,160p'
find .reasonix -maxdepth 3 -type f -printf '%p\n' | sort
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md | sed -n '1,180p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md | sed -n '1,160p'
```

Focused revalidation commands run after citation repair:

```bash
rg -n "Sprint 08|sprint-08|phase-01|scripting|hot Rust|generated templates|01-student-quickstart|deferred" /home/hickelpickle/.codex/memories/MEMORY.md
pwd && rg --files -g 'AGENTS.md' -g '.internal-dev/AGENTS.md'
git status --short
sed -n '1,240p' AGENTS.md
sed -n '1,240p' .internal-dev/AGENTS.md
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md | sed -n '1,260p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md | sed -n '1,220p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-01-validation-report.md | sed -n '1,260p'
nl -ba docs/api/01-student-quickstart.md | sed -n '80,110p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/worker-directives/phase-01-current-state-contract-audit.md | sed -n '1,260p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/00-specification-lock.md | sed -n '1,220p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/01-current-state-analysis.md | sed -n '1,220p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/02-target-design.md | sed -n '1,240p'
nl -ba .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/shared/senior-engineer-guidance.md | sed -n '1,220p'
rg -n "01-student-quickstart\\.md:(86|87|88)|docs/api/01-student-quickstart.md:86-88|86-88" .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-01-validation-report.md
rg -n 'docs/api/01-student-quickstart\.md:101|Corrected the `docs/api/01-student-quickstart.md` citation|deferred-feature statement|dynamic Rust hot reload|generated app templates' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md
rg --no-ignore -n "scripting runtime|generated app templates|hot reload|dynamic Rust|app crates|script" docs src tools apps -g '*.md' -g '*.rs' -g '*.toml'
git status --short && git diff --name-only && git ls-files --others --exclude-standard | sed -n '1,40p'
rg --no-ignore -n "01-student-quickstart\\.md:(86|87|88)|docs/api/01-student-quickstart.md:86-88|86-88" .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/artifacts/phase-01-current-state-contract-audit.md .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-01-email.md
rg --no-ignore -n "01-student-quickstart\\.md:(86|87|88)|docs/api/01-student-quickstart.md:86-88|86-88" .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-01-validation-report.md
```

Focused revalidation notes:

- One intermediate `rg` command used double quotes around a pattern containing markdown backticks, which caused shell command substitution and a `Permission denied` diagnostic for `docs/api/01-student-quickstart.md`; the same check was rerun with safe single-quote shell quoting and passed.
- `rg --no-ignore` was used for the broad docs/source search because the plain `rg` form omitted ignored docs paths in this workspace.

No compile/test/capture commands were run because the phase is documentation/audit-only and the directive states compile commands are not required unless needed to verify an audit claim.

## Browser Or Capture Checklist

Not applicable. Phase 01 is non-visual and has no browser/UI/renderer capture surface.

## Report Path

`.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-01-validation-report.md`
