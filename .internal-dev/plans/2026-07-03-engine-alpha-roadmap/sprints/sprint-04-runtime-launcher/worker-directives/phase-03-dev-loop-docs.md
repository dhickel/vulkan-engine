# Phase 03 Worker Directive: Dev Loop Docs

## Objective

Update public documentation so the supported alpha runtime and Rust application development loops match the implemented root launcher.

## User-Visible Outcome

A reader can tell:

- data-driven projects run through the root `engine` launcher with `--project`;
- custom Rust behavior belongs in app crates under `apps/<name>`;
- renderer examples are diagnostics/examples;
- dogfood is currently a custom app path;
- hot Rust reload, scripting, event system, physics, audio, and dogfood migration are deferred.

## Direct Editable Targets

Primary docs:

- `README.md`
- `docs/api/00-index.md`
- `docs/api/07-engine-arguments.md` or new `docs/api/11-runtime-project-launcher.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `apps/dungeon_dogfood/README.md`

Possible:

- `docs/api/01-student-quickstart.md` if it still claims root `cargo run` is only a migration stub.

Evidence:

- `validation/phase-03-validation-report.md`
- `artifacts/validation-summary.json`

Forbidden:

- Product code changes except tiny docs-link/test reference repairs if a doc check requires them.
- Dogfood code migration.
- New claims about dynamic Rust hot reload, scripting, event system, physics, audio, or lifecycle APIs.

## Supporting Docs To Read

- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- Phase 01 and Phase 02 validation reports.
- Live root launcher help output.
- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `docs/api/10-packaging-cli.md`
- `apps/dungeon_dogfood/README.md`

## Senior-Engineer Guidance

- Docs should follow live behavior, not planned behavior.
- Keep examples copy-pastable from workspace root.
- Do not erase renderer example commands; reclassify them as renderer diagnostics/examples.
- Be explicit about what is not implemented. This prevents future sprint drift.
- Dogfood should be documented as the example of a custom app crate path, not as migrated to project manifests.
- If adding a new API doc page, update `docs/api/00-index.md`.

## Implementation Steps

1. Capture live root launcher help output and use it as the command contract source.
2. Update `README.md`:
   - root `engine` is runtime project launcher;
   - sample project command;
   - headless draw capture command;
   - renderer examples as diagnostics.
3. Update API docs:
   - add root runtime launcher page or revise engine arguments page;
   - include CLI flags, usage, validation capture, and error behavior;
   - cross-link packaging CLI and editor docs.
4. Update packaging CLI docs:
   - remove/deprecate `runtime project launcher` from deferred work;
   - explain `engine_pack` validates/authors data consumed by root launcher.
5. Update editor placement docs:
   - remove stale claim that runtime project launcher is not included;
   - keep editor-specific launch/capture docs distinct from root launcher.
6. Update dogfood README:
   - state dogfood is a custom Rust app crate path using direct renderer facade and custom content/generation;
   - state migration to project manifests is not part of Sprint 04.
7. Sweep docs for stale claims:
   - migration stub;
   - root `cargo run` only prints migration guidance;
   - runtime project launcher deferred;
   - renderer examples as only runtime path;
   - unsupported hot reload/scripting/physics/audio claims.
8. Update validation summary conservatively.

## Acceptance Criteria

- Docs show exact root launcher sample command.
- Docs show exact headless draw-target capture command.
- Docs explain `apps/<name>` custom Rust app crate loop.
- Docs keep dogfood custom path and deferred migration explicit.
- Docs no longer list runtime project launcher as deferred after implementation.
- Stale-reference sweep results are reconciled.

## Negative Criteria

- Do not claim generated app templates are implemented unless Phase 02 actually added them. The current plan does not require templates.
- Do not claim asset/data hot reload is complete.
- Do not claim scripting or dynamic Rust reload.
- Do not imply dogfood was migrated to project manifests.

## Validation Commands

```bash
cargo fmt --check
cargo check -p engine
cargo check -p editor
cargo check -p engine_pack --locked
cargo run -- --help
git diff --check
rg -n "migration stub|runtime project launcher.*deferred|renderer examples.*only runtime|cargo run\\` prints|dynamic Rust hot reload|scripting implemented|physics implemented|audio implemented" README.md docs apps/dungeon_dogfood
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-04-runtime-launcher/artifacts/validation-summary.json >/dev/null
```

The `rg` command is a sweep, not automatically a failure. The validation report must classify each hit as fixed, acceptable historical/context wording, or requiring remediation.

## Evidence Expectations

Write `validation/phase-03-validation-report.md` with:

- docs changed;
- help output source checked;
- stale-reference sweep results;
- any accepted residual wording;
- validation-summary status.

## Commit/Push/Report Gates

- Commit only after Phase 03 validator passes.
- Commit scope should include docs and phase evidence only.
- Do not push unless the orchestrator opens the push gate.
- Do not send reports/email from this worker.

## Stop Conditions

- Stop if docs cannot be made truthful because Phase 02 behavior is not actually implemented.
- Stop if a required doc claim conflicts with live behavior.
- Stop if adding docs requires new product features.

## Do Not Close Unless

- Root launcher docs are accurate.
- App crate dev loop is documented.
- Dogfood status is documented.
- Deferred systems are explicit.
- Phase validation report exists.
