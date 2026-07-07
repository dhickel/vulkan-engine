# Phase 01 Worker Directive: Release Inventory And Docs Lock

## Objective

Audit the current release-facing docs and live contracts, then lock the public documentation shape for the alpha release candidate.

## User-Visible Outcome

The release candidate has a truthful documentation plan: what docs will be public, what commands are supported, what platforms/drivers/toolchain are expected, and what limitations must be listed before any validation claim is made.

## Editable Targets

- `README.md`
- `docs/api/00-index.md`
- `docs/api/01-student-quickstart.md`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- `apps/dungeon_dogfood/README.md`
- New release docs if needed, such as `docs/alpha-release-candidate.md` or `docs/known-issues.md`
- This sprint artifacts:
  - `reports/phase-01-release-inventory.md`
  - `artifacts/validation-summary.json`

## Forbidden Scope

- Do not edit product code.
- Do not edit `SPRINT-TRACKER.md`.
- Do not edit `.idea/engine.iml` or `.reasonix/`.
- Do not edit active Sprint 09 files; read only if needed.
- Do not write changelogs, notes, knowledge, or bug records unless the main thread confirms timing.

## Supporting Docs To Read

- `AGENTS.md`
- `.internal-dev/AGENTS.md`
- `src/renderer/AGENTS.md`
- `.internal-dev/skills/engine-alpha-sprint/SKILL.md`
- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Roadmap README and this sprint spec files.
- Current public docs listed in editable targets.

## Senior-Engineer Guidance

- Release docs should describe validated alpha behavior, not aspirations.
- Preserve existing quickstart structure unless it is misleading.
- Platform/driver expectations should be concrete enough to help users: Rust toolchain, Vulkan runtime/driver, shader tools if required, and host-dependent audio caveats.
- Known issues should separate release blockers from accepted alpha limitations.
- If predecessor Sprint 10-12 artifacts exist by execution time, include their final contracts; if they do not, record a blocker instead of guessing.

## Ordered Steps

1. Run `git status --short` and record protected/dirty state.
2. Read the current public docs and compare command claims to live source entrypoints.
3. Check whether Sprint 10-12 plan/closeout artifacts exist; record status and any release contract dependency.
4. Decide release-doc shape:
   - update existing docs only, or
   - add a focused alpha release candidate doc plus known issues doc.
5. Draft or update docs with:
   - supported workflows;
   - platform/driver/toolchain expectations;
   - package/editor/runtime/dogfood quickstart links;
   - validation evidence policy;
   - known limitations and deferred work.
6. Write `reports/phase-01-release-inventory.md`.
7. Update `artifacts/validation-summary.json` conservatively.

## Acceptance Criteria

- Docs identify alpha as release candidate, not production-ready.
- Docs include true headless draw capture policy.
- Docs do not contradict root launcher/package/editor/dogfood code.
- Predecessor sprint status and any release dependency are recorded.
- No protected paths or tracker files are modified.

## Negative Checks

- Scan for overclaims:
  - `production`
  - `stable`
  - `binary archive`
  - `hot reload`
  - `full physics`
  - `full audio`
  - `full scripting`
  - `dogfood project manifest`
- Expected matches must be limitations or historical context, not support claims.

## Validation Commands

```sh
cargo fmt --check
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json >/dev/null
rg -n "desktop screenshot|present-target proof|production-ready|stable API|binary archive|dynamic Rust hot reload|full physics|full audio|full scripting" README.md docs apps/dungeon_dogfood .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate
```

## Stop Conditions

- Stop if Sprint 10-12 contracts are missing and the main thread has not approved planning Sprint 13 from current proposed contracts.
- Stop if docs require product behavior not present in code.
- Stop if release docs would need protected-path or tracker edits.

## Evidence Expectations

- Worker report: `reports/phase-01-release-inventory.md`
- Validator report: `validation/phase-01-validation-report.md`
- Updated evidence index with `phase_01.status` no stronger than `implementation_checks_passed` before validation.

## Do Not Close Unless

- Docs are internally consistent.
- Release/no-release criteria are visible.
- Known unsupported features are not advertised.
- Protected-path check is recorded.

