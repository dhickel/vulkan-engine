# Phase 03 Worker Directive: Authoring And Pack Commands

## Objective

Implement alpha authoring commands and folder-based pack output on top of the Phase 02 CLI.

## User-Visible Outcome

Users can create starter projects/packages, scan asset directories, add asset records, and produce a validated folder package for alpha distribution.

## Editable Targets

- `tools/engine_pack/src/**`
- `tools/engine_pack/tests/**`
- `tools/engine_pack/fixtures/**`
- optional small additions to shared validation if needed by authoring flows
- sprint evidence under this sprint directory

## Forbidden Scope

- No binary archive format.
- No thumbnail generation.
- No editor UI import workflow.
- No runtime launcher.
- Do not stage `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `02-target-design.md`
- `shared/implementation-notes.md`
- Phase 02 validation report and command behavior
- `docs/api/04-assets-sync-deferred-and-handles.md`

## Senior Guidance

- Generated IDs must be stable durable IDs and should default to human-readable values supplied by the user or derived conservatively from package/category/name.
- `scan-assets` can suggest records without mutating unless an explicit output flag exists.
- `add-asset` should preserve existing manifest data and avoid reordering churn where practical.
- `pack` should copy only project-relative, validated paths into the output folder and write a report.

## Implementation Steps

1. Implement `new-project <dir> --id <project_id> --name <name>`.
2. Implement `new-package <path> --id <package_id> --name <display_name>`.
3. Implement `scan-assets <asset-root> [--package-id <id>]` for known extensions:
   - models: `.gltf`, `.glb`, `.obj`
   - textures: `.png`, `.jpg`, `.jpeg`, `.ktx`, `.ktx2`
   - environments: `.hdr`, `.exr`
   - scripts/audio/collision may be reported as unsupported or future if kinds are not added in Phase 01.
4. Implement `add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]`.
5. Implement `pack <engine.project.toml> --out <dir>` as folder copy with `PACK_REPORT.json`.
6. Add tests proving generated projects/packages validate.
7. Add tests proving `pack` rejects unsafe paths and missing files.

## Acceptance Criteria

- Authoring commands exist and have tests.
- Generated files validate through Phase 02 commands.
- `pack` output contains expected manifests/assets/scenes and a report.
- `scan-assets` output is deterministic for fixtures.

## Negative Checks

- `pack` does not create a binary archive.
- No absolute or parent-traversing paths are copied.
- No command writes runtime handle identity.

## Validation Commands

```bash
cargo check -p engine_pack
cargo test -p engine_pack
cargo run -p engine_pack -- new-project .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/engine-pack-smoke/project --id project.engine_pack_smoke --name "Engine Pack Smoke"
cargo run -p engine_pack -- new-package .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/engine-pack-smoke/project/assets/smoke.package.toml --id smoke --name "Smoke Assets"
cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
cargo run -p engine_pack -- pack apps/editor/sample_project/engine.project.toml --out .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-02-packaging-tools/artifacts/engine-pack-smoke/packed
```

Use a better sprint-local evidence directory if implementation creates one; keep it under this sprint's `artifacts/` directory.

## Evidence Expectations

- Validation report path: `validation/phase-03-validation-report.md`
- Include files/line counts/git links matrix.
- Include generated fixture/output paths.
- Include capture decision `not_required_cli_schema_only`.

## Commit/Push/AgentMail Gate

After phase validation passes, orchestrator must commit scoped changes, push `sprint/alpha-02-packaging-tools`, and send an AgentMail HTML progress report using `email-report-template.html`.

## Stop Conditions

- Stop if stable ID generation requirements need user product decisions.
- Stop if adding audio/script/collision kinds would force cross-sprint API commitments.
- Stop if `pack` would need unresolved package dependency semantics.

## Do Not Close Unless

- Generated output revalidates.
- Pack output report exists in evidence.
- Validation report records commit, pushed ref, GitHub links, and AgentMail evidence after orchestration gates.
