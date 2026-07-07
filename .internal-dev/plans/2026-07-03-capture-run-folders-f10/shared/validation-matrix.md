# Validation Matrix

## Local Required Checks

| Area | Command | Evidence |
| --- | --- | --- |
| Formatting | `cargo fmt --check` | No diff required |
| Workspace compile | `cargo check` | Completes successfully |
| Renderer compile | `cargo check -p renderer` | Completes successfully |
| Examples compile | `cargo check -p renderer --examples` | Completes successfully |
| Input compile | `cargo check -p input` | Completes successfully |
| Unit coverage | `cargo test -p renderer capture` | Capture path tests pass |
| Headless single capture | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_frame=5` | PNG lands under `.internal-dev/captures/<run>/` |
| Headless N-frame capture | `RUST_LOG=debug timeout --signal=INT 60s cargo run -p renderer --example api_test -- --headless --capture_frames=3 --capture_frame_start=5 --capture_frame_interval=5` | Three PNGs land under one `.internal-dev/captures/<run>/` |

## Manual Windowed Check

| Area | Command | Evidence |
| --- | --- | --- |
| F10 manual capture | `cargo run -p renderer --example api_test` then press `F10` | One manual PNG lands under that run's `.internal-dev/captures/<run>/` folder |

## Environment Matrix

| Environment | Single Headless | N-Frame Headless | Windowed F10 | Required Evidence |
| --- | --- | --- | --- | --- |
| Linux Wayland | Required | Required | Best effort | Command log, path listing, PNG |
| Linux X11 | Target | Target | Target | Command log, path listing, PNG |
| Windows | Target | Target | Target | Command log, path listing, PNG |
| macOS | Target if supported | Target if supported | Target if supported | Command log, path listing, PNG |

If only local Linux is available, record the local evidence and leave the broader matrix as a follow-up validation target.
