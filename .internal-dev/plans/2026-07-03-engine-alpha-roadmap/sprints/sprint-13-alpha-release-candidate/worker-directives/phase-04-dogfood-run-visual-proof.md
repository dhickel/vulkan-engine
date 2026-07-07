# Phase 04 Worker Directive: Dogfood Run And Visual Proof

## Objective

Prove the dogfood app runs with documented content settings and produces true engine-owned headless draw visual evidence.

## User-Visible Outcome

The alpha release candidate includes a real custom Rust app demo path with honest setup instructions, known limitations, runtime smoke evidence, and visual baseline evidence.

## Editable Targets

- `apps/dungeon_dogfood/src/`
- `apps/dungeon_dogfood/README.md`
- `docs/api/00-index.md` or release docs if dogfood instructions change
- `reports/phase-04-dogfood-run-visual-proof.md`
- `artifacts/validation-summary.json`

## Forbidden Scope

- Do not migrate dogfood to project manifests unless already required by predecessor sprint contracts.
- Do not rework gameplay, physics, or content generation beyond release-blocking proof.
- Do not use desktop screenshots.
- Do not touch `.idea/engine.iml`, `.reasonix/`, or `SPRINT-TRACKER.md`.

## Supporting Docs To Read

- `apps/dungeon_dogfood/README.md`
- `apps/dungeon_dogfood/.developer-documentation.md` if present and relevant
- `docs/api/11-runtime-project-launcher.md` for capture argument conventions
- Headless capture skill
- Phase 01 and Phase 02 reports

## Experience Contract

- Dogfood proof should show the actual default alpha demo, not a stripped-down placeholder unless full content is proven too slow and the limitation is accepted.
- Required visual expectations:
  - generated or selected dungeon geometry visible;
  - custom environment/lighting applied when full-content settings request it;
  - no fatal asset-load errors hidden behind a blank frame;
  - visual result matches documented known compromises, such as torch prop fallback if still current.

## Senior-Engineer Guidance

- Current planning scan did not find dogfood `--headless` support. First verify current branch state; if absent, add the narrowest app-owned headless mode.
- Reuse root/editor capture argument semantics where practical so release docs stay consistent.
- Dogfood full-content startup can be slow. Keep timeout-bound commands and add debug timing if startup crosses expected budget.
- Host audio device availability is not required for normal dogfood proof; `--audio-smoke` remains opt-in unless release scope changes.

## Ordered Steps

1. Verify current dogfood CLI/env behavior:
   ```sh
   cargo check -p dungeon_dogfood
   cargo test -p dungeon_dogfood
   ```
   If `cargo test -p dungeon_dogfood` fails from inherited renderer test-profile issues, record exact error and classify.
2. Run documented full-content windowed smokes with timeout:
   ```sh
   DUNGEON_DOGFOOD_FAST_STARTUP=0 DUNGEON_DOGFOOD_LOAD_PROPS=1 DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level generated_sprawl
   DUNGEON_DOGFOOD_FAST_STARTUP=0 DUNGEON_DOGFOOD_LOAD_PROPS=1 DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level level_02_ramps
   DUNGEON_DOGFOOD_FAST_STARTUP=0 DUNGEON_DOGFOOD_LOAD_PROPS=1 DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level level_03_lighting
   ```
3. Verify headless capture support. If absent, implement narrow dogfood support for:
   - `--headless`;
   - `--capture_target draw`;
   - capture sequence flags or a clearly documented supported subset;
   - same scene/content path as windowed dogfood.
4. Run dogfood draw capture:
   ```sh
   DUNGEON_DOGFOOD_FAST_STARTUP=0 \
   DUNGEON_DOGFOOD_LOAD_PROPS=1 \
   DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
   RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
     --level generated_sprawl \
     --headless \
     --capture_target draw \
     --capture_frames 3 \
     --capture_frame_start 5 \
     --capture_frame_interval 5 \
     --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/dogfood-draw
   ```
5. Inspect PNGs and sidecars.
6. Update dogfood docs/release docs with exact commands, content settings, and known limitations.
7. Write report and update validation summary.

## Acceptance Criteria

- Dogfood documented run commands work or release blockers are recorded.
- Full-content env settings are documented and tested.
- Dogfood has accepted true headless draw capture proof.
- Capture output is nonblank/inspectable and sidecars report success/draw target.
- Any inherited `cargo test -p dungeon_dogfood` issue is re-verified and accurately classified.

## Negative Checks

- No desktop screenshot evidence.
- No accidental audio device requirement for normal dogfood startup.
- No hardcoded absolute paths.
- No dogfood docs claiming project-manifest migration unless implemented and validated.

## Validation Commands

```sh
cargo fmt --check
cargo check -p dungeon_dogfood
cargo check -p renderer
cargo test -p dungeon_dogfood
git diff --check
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json >/dev/null
```

If product code adds capture argument parsing, add focused tests for accepted/rejected flags.

## Stop Conditions

- Stop if dogfood cannot initialize renderer/content with documented settings.
- Stop if headless draw capture cannot be implemented narrowly without broad renderer/app redesign.
- Stop if visual output is blank/inconclusive after capture succeeds.

## Evidence Expectations

- Worker report: `reports/phase-04-dogfood-run-visual-proof.md`
- Validator report: `validation/phase-04-validation-report.md`
- Capture directory: `.internal-dev/captures/sprint-13-alpha-release-candidate/dogfood-draw/`
- Optional debug timing: `.internal-dev/debug_reports/sprint-13-alpha-release-candidate/dogfood-*.jsonl`

## Do Not Close Unless

- Dogfood visual proof is true headless draw capture.
- Docs match commands.
- Residuals are release-classified.

