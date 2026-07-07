# Phase 02 Worker Directive: Docs And Gap Report Repair

## Objective

Repair current documentation drift and ensure stale gap-report content can no longer drive future alpha planning as current truth.

## User-Visible Outcome

Root docs and indexes describe the live workspace and direct readers to a current alpha readiness baseline instead of stale "no subsystem exists" claims.

## Editable Targets

- `AGENTS.md`
- `README.md`
- `docs/api/00-index.md`
- `docs/internal/00-index.md`
- `docs/gap-report.md` or a replacement current report such as `docs/alpha-readiness-report.md`
- Sprint artifacts as needed:
  - `artifacts/docs-drift-audit.md`
  - `artifacts/validation-summary.json`
  - `validation/phase-02-validation-report.md`

## Read-Only Supporting Inputs

- Root `Cargo.toml`
- package `Cargo.toml` files
- `src/renderer/AGENTS.md`
- `src/input/AGENTS.md`
- `docs/gap-report.md`
- `artifacts/baseline-inventory.md`

## Forbidden Scope

- Do not edit Rust source.
- Do not implement features described by the docs.
- Do not add broad new docs outside the stale-report repair unless required to keep links coherent.
- Do not claim alpha feature completeness.

## Senior-Engineer Guidance

- Preserve the root guide's role as orientation; do not make it a long architecture document.
- Prefer correcting facts over adding caveats everywhere.
- Keep `docs/gap-report.md` path stable if many docs link to it, but rewrite the file as a superseded/current-readiness pointer if that is cleaner than moving it.
- If creating `docs/alpha-readiness-report.md`, update indexes so readers land there first and label old gap material historical.
- Current workspace members must match root `Cargo.toml`, including `audio`, `physics`, `scripting`, `apps/editor`, and `apps/dungeon_dogfood`.

## Ordered Steps

1. Confirm phase 01 is validated, committed, pushed, and emailed.
2. Run targeted stale docs search:
   ```bash
   rg -n "gap-report|known limitations|No audio|No physics|No scripting|No project system|No scene serialization|headless" README.md AGENTS.md docs
   ```
3. Compare docs to root `Cargo.toml` workspace members.
4. Update `AGENTS.md` workspace list and runtime entrypoint guidance to reflect live workspace.
5. Update `README.md` with concise current entrypoints and alpha-doc guidance.
6. Replace/supersede stale `docs/gap-report.md` with a current alpha readiness baseline or historical notice plus current report link.
7. Update `docs/api/00-index.md` and `docs/internal/00-index.md` so "known limitations" points to current readiness/residual tracking, not stale claims.
8. Write `artifacts/docs-drift-audit.md` summarizing repaired drifts and any deferred doc issues.
9. Update `artifacts/validation-summary.json`.
10. Run validation commands.
11. Write `validation/phase-02-validation-report.md`.
12. Commit, push, and send Dwight the post-phase HTML AgentMail report.

## Acceptance Criteria

- Root docs list all live workspace crates/apps accurately.
- Current docs do not cite stale `docs/gap-report.md` as active known limitations unless the file has been rewritten as current.
- Stale claims contradicted by live source are removed, marked historical, or moved into a residual register for phase 03 classification.
- Links in docs indexes still resolve.

## Negative Checks

- No product code changes.
- No stale "no audio/physics/scripting/project/editor" claims remain as current truth.
- No docs claim a subsystem is alpha-ready solely because a crate exists.
- No broad `.internal-dev` reads.

## Validation Commands

```bash
git status --short --branch
rg -n "gap-report|known limitations|No audio|No physics|No scripting|No project system|No scene serialization|headless" README.md AGENTS.md docs
rg -n "src/audio|src/physics|src/scripting|apps/editor|apps/dungeon_dogfood" AGENTS.md README.md docs/api/00-index.md docs/internal/00-index.md docs/gap-report.md
```

If link-check tooling exists locally, run it. Otherwise manually inspect modified Markdown links and record that no project link checker exists.

## Stop Conditions

- Stale docs cannot be safely repaired without deciding a product/design question.
- Modified docs would contradict live code.
- Unexpected product code changes appear.
- Push or AgentMail send fails.

## Evidence Expectations

- Validation report: `validation/phase-02-validation-report.md`
- Docs drift audit: `artifacts/docs-drift-audit.md`
- Evidence index update: `artifacts/validation-summary.json`
- Commit hash, pushed branch/ref, GitHub links, email evidence.

## Do Not Close Unless

- Phase validator passes.
- Commit is pushed.
- HTML email report is sent.
- Stale gap-report current-truth issue is resolved or explicitly blocked with a plan-defect escalation.
