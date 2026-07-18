# Visual Regression Harness — Phase 01

## Contract

`src/renderer/tests/visual_regression.rs` decodes baseline and retained capture PNGs with
`image`, requires identical decoded dimensions and color models, computes maximum absolute
per-channel error and differing-pixel ratio, and applies per-case thresholds. Missing files,
decode errors, dimension/color drift, threshold failures, and sidecar drift are hard failures;
tests never silently skip.

For both baseline and capture, a JSON sidecar is mandatory. The capture must exactly match the
reviewed baseline’s `target`, `extent`, `frame`, `scene_preset`, and `declared_regions`; its
extent must also equal the decoded PNG. Every declared region must be named, non-empty, and
inside the image.

## Fixtures

`src/renderer/tests/fixtures/visual_regression/` retains baseline and capture PNG/JSON pairs for
`capture_geometry` and `capture_shadows`. Current thresholds are maximum channel error `5` and
maximum differing-pixel ratio `0.02` per case.

## Compare

```bash
cargo test -p renderer visual_regression -- --nocapture
```

## Generate retained captures

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example capture_geometry -- \
  --headless --capture_target draw --capture_frame=5 \
  --capture_frame_path=src/renderer/tests/fixtures/visual_regression/capture_geometry.capture.png

RUST_LOG=info timeout --signal=INT 60s cargo run -p renderer --example capture_shadows -- \
  --headless --capture_target draw --capture_frame=5 \
  --capture_frame_path=src/renderer/tests/fixtures/visual_regression/capture_shadows.capture.png
```

Retain or normalize the generated JSON sidecars at the adjacent `.capture.json` paths and run
the comparator before considering a capture valid.

## Baseline update command

After visually reviewing the decoded capture PNGs and their sidecars, run exactly:

```bash
VISUAL_REGRESSION_UPDATE=1 cargo test -p renderer visual_regression -- --nocapture
```

This copies both capture PNGs and mandatory sidecars over the reviewed baselines. Review the
resulting decoded output and diff again before retaining it. The update path fails if either
capture member is absent.
