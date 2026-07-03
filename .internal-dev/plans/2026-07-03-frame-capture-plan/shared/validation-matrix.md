# Validation Matrix

## Compile Gates

| Gate | Command | Required |
|---|---|---|
| Workspace check | `cargo check` | Yes |
| Renderer check | `cargo check -p renderer` | Yes |
| Renderer examples check | `cargo check -p renderer --examples` | Yes |
| Input check | `cargo check -p input` | Yes |

## Focused Tests

| Area | Expected Coverage |
|---|---|
| Launch parser | capture frame/count/start/interval/path/dir/headless/manual flags, invalid zero values, missing values, existing timing/env flags |
| Capture scheduler | single fires once, sequence fires exactly N, interval behavior, manual queues next frame, no unbounded captures |
| Output naming | deterministic paths, manual default directory, no overwrite unless explicit behavior is documented |
| Capture errors | invalid path, invalid extent, unsupported format, resize/defer behavior |

## Required Runtime Capture Matrix

All rows require PNG proof and image sanity checks.

| ID | Command Shape | Required Artifact |
|---|---|---|
| pbr-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_pbr -- --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/demo_pbr.png` | `demo_pbr.png` |
| unlit-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_unlit -- --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/demo_unlit.png` | `demo_unlit.png` |
| model-load-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_model_load -- --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/demo_model_load.png` | `demo_model_load.png` |
| async-loading-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example demo_async_loading -- --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/demo_async_loading.png` | `demo_async_loading.png` |
| api-test-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/api_test.png` | `api_test.png` |
| api-test-env-windowed | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --env src/renderer/src/assets/sky_maps/indoor_4k.exr --capture_frame=<n> --capture_frame_path=.internal-dev/debug_reports/frame-capture/api_test_indoor_4k.png` | `api_test_indoor_4k.png` |

Use the final implemented flag names if they differ, and update this matrix in the same phase.

## N-Frame Validation

Required proof:

```sh
RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- \
  --capture_frames=5 \
  --capture_frame_start=<n> \
  --capture_frame_interval=10 \
  --capture_dir=.internal-dev/debug_reports/frame-capture/api_test-sequence
```

Pass criteria:

- exactly 5 PNG files exist;
- filenames encode sequence/frame identity;
- no sixth file is produced;
- all 5 images pass metadata and nonblank/nonuniform checks.

## Headless Validation

Preferred full matrix mirrors the required runtime capture matrix with `--headless` and output under `.internal-dev/debug_reports/frame-capture/headless/`.

Minimum if implementation is blocked:

- phase 03 validation report records the blocker;
- user-decision gate is invoked before accepting a fallback;
- final evidence index records `blocked_tooling_constraint` or an approved fallback status rather than `fully_validated`.

## Manual Capture Validation

Preferred:

- launch one example locally;
- trigger `F12`;
- confirm one PNG appears under `.internal-dev/debug_reports/manual-captures/`.

Fallback when input automation is unavailable:

- invoke the same public queueing path from a focused test or temporary validation harness;
- prove output path defaults to manual-captures;
- record the input automation blocker in the phase report and evidence index.

## Image Sanity Checks

For each PNG:

- `file <path>` reports PNG image data;
- dimensions are greater than zero;
- dimensions match the configured viewport/capture extent unless a documented scale applies;
- pixel sample/histogram check shows more than one unique color or otherwise proves nonuniform rendered content;
- zero-byte, all-black, all-transparent, or fully uniform images fail unless a specific example intentionally renders that way and a secondary proof explains it.

