---
name: engine-headless-capture-validation
description: Use in this engine repo when validating renderer, scene, shader, camera, material, asset, or Vulkan rendering changes with headless frame captures, timeout-bound example runs, deterministic scene/camera setups, or screenshot evidence without taking over the user's desktop.
---

# Engine Headless Capture Validation

Use this project skill when a task changes visible rendering behavior or needs visual proof from the engine without relying on desktop screenshots.

## Core Rule

Prefer engine-owned headless frame capture over compositor screenshots. The agent does not fully control the user's windowing environment, so validation should use deterministic scenes, camera transforms, and engine capture output whenever possible.

## Allowed Test Assets

Agents are free to create project-local test material for validation when it reduces ambiguity:

- Source-controlled reusable tests/scenes: `src/renderer/examples/capture_tests/`
- Temporary plans, scene specs, notes, or generated evidence: `.internal-dev/headless_capture_tests/`
- Capture output: `.internal-dev/captures/`

Keep reusable test scenes small and explicit. Use `.internal-dev/headless_capture_tests/` for one-off investigation artifacts that should not become product code.

## Default Commands

Engine startup can take 20-30 seconds. Use a timeout so failed render validation does not hang the session.

Single capture:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frame=5
```

N-frame capture:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5
```

Alternate environment:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_target draw --env src/renderer/src/assets/sky_maps/indoor_4k.exr --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5
```

Captures land by default under:

```text
.internal-dev/captures/<app-name>-<YYYYMMDD-HHMMSS-mmm>-pid<PID>/
```

## Validation Workflow

1. Define the expected visual behavior in concrete terms: object, material, lighting, camera position, frame number, and pass/fail criteria.
2. Reuse an existing example when it covers the behavior.
3. If the existing examples are ambiguous, create a small deterministic test scene or capture-focused example under `src/renderer/examples/capture_tests/`.
4. Run a timeout-bound headless capture command.
5. Inspect the PNG and sidecar JSON in `.internal-dev/captures/`.
   - If the current model supports image input, inspect the PNG directly.
   - If the current model is known not to support image input, load and follow the available `image-viewing` skill, passing it the local PNG path and the concrete visual pass/fail criteria.
   - Do not invoke `image-viewing` merely because image capability is uncertain; follow its mandatory trigger gate.
6. Record evidence paths and note whether the result is a strict pass, a visual regression, or inconclusive.

## Scene And Camera Guidance

- Use fixed camera transforms and deterministic frame numbers.
- Prefer simple geometry/material setups that isolate the changed behavior.
- For shader/material work, include at least one high-contrast case that makes the expected effect obvious.
- For asset/cache work, use a known asset path and record it in the validation notes.
- For lighting/IBL changes, validate at least one default environment and one explicit environment when practical.

## Evidence Standards

Evidence should include:

- command run;
- confirmation that the command used `--headless --capture_target draw`;
- capture directory;
- PNG path(s);
- sidecar JSON path(s);
- expected visual result;
- actual observed result;
- any uncertainty or environment limitation.

Do not claim visual validation from compile checks alone. If only compile checks were run, say that runtime capture validation was not run.

## When To Stop

Stop and report a blocker if:

- headless renderer initialization fails;
- capture output is missing after a successful command;
- the scene cannot be made deterministic without broader product changes;
- a required visual judgment cannot be made from the generated PNG.
