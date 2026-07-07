# Phase 04 Worker Directive: True Headless Visual Baseline

## Objective

Implement and validate true engine-owned headless draw-target capture for the dogfood vertical slice.

## User-Visible Outcome

The sprint has repeatable visual proof under `.internal-dev/captures/sprint-11-dogfood-vertical-slice/` showing the dogfood dungeon slice rendered through `--headless --capture_target draw`.

## Editable Files

Likely editable:

- `apps/dungeon_dogfood/src/main.rs`
- dogfood launch/capture helper modules if created.
- shared launch/capture helpers only if Phase 03 established a safe path.
- `.internal-dev/headless_capture_tests/sprint-11-dogfood-vertical-slice/**` for temporary capture specs.
- `validation/phase-04-validation-report.md`
- `artifacts/validation-summary.json`

Possible but high-risk:

- `src/runtime.rs`
- `src/launch.rs`
- renderer capture API modules.

Do not edit high-risk shared files unless the need is narrow, Phase 03 validation passed, and Sprint 09 conflicts are cleared.

## Supporting Docs To Read

- `.internal-dev/skills/engine-headless-capture-validation/SKILL.md`
- `docs/api/11-runtime-project-launcher.md`
- Phase 03 validation report.
- `src/renderer/AGENTS.md`

## Ordered Steps

1. Confirm Phase 03 dogfood run path and full-content flags.
2. Implement or finalize dogfood `--headless`, capture frame, capture target, capture directory, and debug timing support using engine-owned capture APIs.
3. Ensure headless mode does not create a desktop window/event loop.
4. Use deterministic camera/scene setup suitable for capture.
5. Run true draw-target capture with full-content mode enabled.
6. Inspect sidecar JSON and PNG output.
7. If the capture is visually inconclusive, adjust deterministic camera/scene setup in dogfood or a focused capture spec, not broad renderer internals.
8. Update validation summary with capture paths and status.
9. Write validation report.

## Senior-Engineer Guidance

- The evidence is the engine draw target, not the user's desktop.
- Sidecar metadata matters as much as PNG existence.
- A blank/black image is a failed visual proof even if the command exits successfully.
- If headless app capture needs broad root-runtime extraction, stop and report blocker rather than doing architecture work in this phase.

## Acceptance Criteria

- Command includes `--headless --capture_target draw`.
- Capture output lands under `.internal-dev/captures/sprint-11-dogfood-vertical-slice/`.
- Sidecar JSON reports success, draw target, positive extent, and valid PNG path.
- PNG shows expected dogfood dungeon baseline with geometry, light, environment/fallback state, and props in full-content mode.
- Validation report includes actual observed result and uncertainty if any.

## Negative Checks

- No desktop screenshots.
- No present-target-only captures.
- No capture without full-content flags for final baseline.
- No `fully_validated` status update in this phase unless final review has also passed, which it should not do here.

## Validation Command

```sh
mkdir -p .internal-dev/captures/sprint-11-dogfood-vertical-slice
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- \
  --project apps/dungeon_dogfood/engine.project.toml \
  --level generated_sprawl \
  --headless \
  --capture_target draw \
  --capture_frames=3 \
  --capture_frame_start=5 \
  --capture_frame_interval=5 \
  --capture_dir .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl
```

Metadata inspection:

```sh
find .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl -maxdepth 2 -type f | sort
rg -n "\"status\"|\"capture_target\"|\"extent\"|\"png\"|draw|succeeded" .internal-dev/captures/sprint-11-dogfood-vertical-slice/dogfood-generated-sprawl
```

## Stop Conditions

- Stop if headless renderer initialization fails.
- Stop if capture output is missing after a successful command.
- Stop if sidecar says present target or no target.
- Stop if visual judgment cannot be made from generated PNG.
- Stop if fixing capture requires broad renderer refactor.

## Evidence Expectations

- Command and environment variables.
- Capture directory.
- PNG path(s).
- Sidecar JSON path(s).
- Metadata fields inspected.
- Human visual observation.
- Validation report path: `validation/phase-04-validation-report.md`.

## Do Not Close Unless

- Draw-target capture evidence exists and is inspected.
- Validation summary names capture directory and status.
- Any visual compromise is documented in residuals.
