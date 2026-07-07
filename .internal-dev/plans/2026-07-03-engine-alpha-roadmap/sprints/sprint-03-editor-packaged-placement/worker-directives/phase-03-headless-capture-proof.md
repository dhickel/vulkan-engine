# Phase 03 Worker Directive: Headless Capture Proof

## Objective

Create deterministic engine-owned visual proof that packaged editor assets placed through the Sprint 03 data path render visibly after save/reload.

## User-Visible Outcome

The sprint has PNG and sidecar capture evidence showing a package-backed model and wall chunk placed in a scene, with commands and observations recorded under `.internal-dev`.

## Editable Targets

- `apps/editor/src/launch.rs`
- `apps/editor/src/main.rs`
- Optional small capture harness under `src/renderer/examples/capture_tests/` or an existing renderer example if editor headless automation is insufficient
- Optional sprint-local scene/capture specs under `.internal-dev/headless_capture_tests/`
- Capture output under `.internal-dev/captures/`
- Evidence references in `artifacts/validation-summary.json`

## Forbidden Scope

- Do not build a broad editor automation framework.
- Do not claim desktop screenshot proof.
- Do not change product renderer behavior merely to make a capture easier unless a real product bug is proven.
- Do not mutate canonical sample scene without approval.
- Do not close Sprint 01 or include unrelated `.idea/engine.iml` / `.reasonix/`.

## Supporting Docs To Read

- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- Phase 01 and Phase 02 validation reports.
- `02-target-design.md`
- `shared/validation-matrix.md`

## Senior Engineer Guidance

- Prefer using the Phase 02 saved scene copy as capture input. This keeps visual proof tied to persistence.
- If the editor can load the saved scene headlessly and capture frames, use it.
- If editor headless capture cannot deterministically show the placement, write a minimal capture-focused harness that loads the same sample project/package/saved scene data path and renders the scene.
- A capture harness defect is not automatically a product defect. Fix the harness first unless the evidence shows real runtime/editor failure.

## Ordered Implementation Steps

1. Identify the saved scene copy from Phase 02 containing package-backed model and wall chunk nodes.
2. Read the headless capture skill and define expected visual behavior: asset IDs, transforms, camera, frame numbers, and visible features.
3. Try the editor headless capture path if it can load the saved scene and render deterministically.
4. If editor capture is insufficient, add a small capture-focused renderer example/test scene that loads the same project/package/saved scene data path.
5. Run timeout-bound capture commands.
6. Inspect PNG and sidecar JSON.
7. Record command, capture directory, PNG paths, sidecar paths, expected result, actual observation, and pass/fail/inconclusive status.
8. Update `artifacts/validation-summary.json` capture fields conservatively.

## Preferred Validation Command

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p editor -- --project apps/editor/sample_project/engine.project.toml --scene <saved-scene-copy> --headless --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement
```

## Fallback Validation Command

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example <capture-example> -- --headless --capture_frames 3 --capture_frame_start 5 --capture_frame_interval 5 --capture_dir .internal-dev/captures/sprint-03-editor-packaged-placement
```

Also run:

```bash
cargo fmt --check
cargo check -p editor
cargo check -p renderer --examples
cargo test -p renderer scene
git diff --check
```

## Acceptance Criteria

- Capture command completes or reaches timeout after successful startup without fatal render errors.
- Capture directory contains PNG and sidecar JSON artifacts.
- Inspected PNG shows the package-backed placed model and wall chunk, or the report explains why evidence is inconclusive and stops.
- Evidence path exercises the saved scene/package data path from Phase 02 or documents an equivalent package-backed harness.
- Validation summary records capture artifacts and does not overstate pass status before validator reconciliation.

## Negative Checks

- No visual validation from compile checks alone.
- No generic hardcoded model proof that bypasses package/project/scene references.
- No broad renderer/Vulkan rewrite.
- No stale `/tmp` artifact-only proof; durable paths must be under `.internal-dev`.

## Evidence Expectations

- Write or prepare `validation/phase-03-validation-report.md`.
- Record capture artifact paths.
- Include notes on camera/framing and visual pass/fail.
- Main thread records commit/push/email report evidence after validation passes.

## Stop Conditions

- Stop if headless renderer initialization fails.
- Stop if capture output is missing after a successful run.
- Stop if the scene cannot be deterministic without broader product changes.
- Stop if the PNG cannot support a visual judgment.

## Do Not Close Unless

- Capture artifacts exist, are inspected, and are reconciled against expected visible behavior.
- The proof path is package-backed, not a disconnected renderer demo.
