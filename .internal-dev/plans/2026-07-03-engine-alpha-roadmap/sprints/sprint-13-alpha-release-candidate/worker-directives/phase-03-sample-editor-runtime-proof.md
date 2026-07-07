# Phase 03 Worker Directive: Sample Project, Editor, Runtime Proof

## Objective

Prove the alpha sample project can be validated, packaged, opened in the editor, edited, saved, and run through the root launcher with true headless draw visual evidence.

## User-Visible Outcome

The release candidate has an evidence-backed sample workflow a new user can follow.

## Editable Targets

- `apps/editor/sample_project/` only if release intentionally updates canonical sample fixtures.
- `apps/editor/` only for release-blocking editor open/edit/save defects.
- `tools/engine_pack/` only for release-blocking validation/pack defects.
- `src/runtime.rs` / `src/launch.rs` only for release-blocking runtime sample launch defects.
- Docs touched in Phase 01 only if command behavior changes.
- Artifacts:
  - `artifacts/sample-pack/`
  - `artifacts/sample-edited-scene.engine.scene.json`
  - `reports/phase-03-sample-editor-runtime-proof.md`
  - `artifacts/validation-summary.json`

## Forbidden Scope

- Do not broaden editor features beyond release proof.
- Do not serialize runtime handles into scene files.
- Do not edit Sprint 09 active renderer files unless main thread confirms Sprint 09 is no longer active and the edit is release-blocking.
- Do not use desktop screenshots.

## Supporting Docs To Read

- `docs/api/09-editor-asset-browser-and-wall-chunks.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`
- Phase 01 and Phase 02 reports.
- Headless capture skill.

## Experience Contract

- Editor proof should demonstrate the operational editor surface, not a marketing screenshot.
- Required visible states:
  - project/sample scene loaded;
  - package-backed asset browser data available;
  - edited/saved scene produces a visible object or transform difference in the draw capture;
  - status/errors are not hiding a failed load.
- Desktop/mobile responsive web criteria do not apply; this is a desktop engine/editor validation.

## Senior-Engineer Guidance

- Prefer writing an edited scene copy under sprint artifacts rather than mutating canonical sample data.
- If editor has no deterministic non-interactive edit command, use existing scene APIs/tests to create the edited scene artifact, then validate editor can open and capture it. Do not fake an editor workflow; report the limitation.
- Capture sidecars matter as much as PNGs.
- The root runtime should be tested with canonical sample and edited scene where possible.

## Ordered Steps

1. Validate sample project and scene:
   ```sh
   cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
   cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
   ```
2. Pack sample project:
   ```sh
   rm -rf .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-pack
   cargo run -p engine_pack -- pack apps/editor/sample_project/engine.project.toml --out .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-pack
   ```
3. Create or obtain edited scene artifact:
   - Use editor save path if deterministic automation exists.
   - Otherwise create a scene copy through existing scene APIs/tests or a focused helper and clearly record that it validates persistence but not interactive input.
4. Validate edited scene:
   ```sh
   cargo run -p engine_pack -- validate-scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-edited-scene.engine.scene.json --project apps/editor/sample_project/engine.project.toml
   ```
5. Capture editor with edited scene:
   ```sh
   RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- \
     --project apps/editor/sample_project/engine.project.toml \
     --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-edited-scene.engine.scene.json \
     --headless \
     --capture_target draw \
     --capture_frames 3 \
     --capture_frame_start 5 \
     --capture_frame_interval 5 \
     --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/editor-sample-draw
   ```
6. Capture root runtime with canonical or edited scene:
   ```sh
   RUST_LOG=info timeout --signal=INT 60s cargo run -- \
     --project apps/editor/sample_project/engine.project.toml \
     --scene .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-edited-scene.engine.scene.json \
     --headless \
     --capture_target draw \
     --capture_frames 3 \
     --capture_frame_start 5 \
     --capture_frame_interval 5 \
     --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/sample-runtime-draw
   ```
7. Run focused compile/tests for touched crates.
8. Write report and update evidence index.

## Acceptance Criteria

- Project/scene validation passes.
- Pack output exists and includes `PACK_REPORT.json`.
- Edited scene validates and does not contain runtime handles.
- Editor and runtime captures are draw-target, successful, and visually inspectable.
- Any inability to prove interactive edit/save is explicitly classified.

## Negative Checks

```sh
rg -n '"slot"|"generation"|MeshHandle|TextureHandle|SceneNodeId|LoadTicket' .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-edited-scene.engine.scene.json
```

This scan should return no matches for runtime-handle identity.

## Validation Commands

```sh
cargo fmt --check
cargo check -p editor
cargo check -p engine
cargo check -p engine_pack --locked
cargo test -p engine_pack --locked
cargo test -p renderer scene
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json >/dev/null
```

## Stop Conditions

- Stop if sample validation fails from canonical fixture corruption.
- Stop if editor or runtime capture cannot be produced through `--headless --capture_target draw`.
- Stop if proof requires mutating canonical sample without release justification.

## Evidence Expectations

- Worker report: `reports/phase-03-sample-editor-runtime-proof.md`
- Validator report: `validation/phase-03-validation-report.md`
- Capture directories under `.internal-dev/captures/sprint-13-alpha-release-candidate/`
- Updated `artifacts/validation-summary.json`

## Do Not Close Unless

- Sample workflow proof is reproducible.
- Capture sidecars pass predicate.
- Visual observations are recorded.
- Residuals are release-classified.

