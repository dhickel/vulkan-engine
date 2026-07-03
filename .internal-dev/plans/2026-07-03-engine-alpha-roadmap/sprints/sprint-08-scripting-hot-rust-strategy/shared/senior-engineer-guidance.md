# Senior Engineer Guidance

## Core Direction

- Treat app crates as the product path. This keeps Rust ownership, compile-time checks, and facade boundaries intact.
- Treat scripts as data-driven helpers. They are useful only if they are sandboxed, observable, and explicitly scoped.
- Treat hot Rust as tooling. A runtime reload promise creates ABI, safety, lifetime, and platform obligations the alpha engine is not ready to own.

## Direct Targets

- `tools/engine_pack` for app-template tooling.
- `src/scripting` for Rhai wrapper hardening.
- `src/events` only for small vocabulary/helper additions; keep it dependency-free.
- `src/renderer/src/data/asset_registry.rs` and `src/renderer/src/api/scene.rs` only for durable script asset/reference validation.
- Docs under `docs/api` and `docs/internal` for contract truth.

## Gotchas

- Existing `engine_mut` can bypass the intended safety story. Do not advertise it as the default API.
- Adding `script` as an asset kind without tests can make `engine_pack scan-assets` and validation drift.
- Generated app templates can fail because they are not workspace members. Test with `cargo check --manifest-path <generated>/Cargo.toml` or document the required workspace step.
- Script error handling must preserve script ID/context, not only a generic Rhai string.
- Avoid dependency inversion: `engine_events` must not depend on `scripting`, `renderer`, editor, dogfood, physics, or audio.

## Best Practices

- Prefer copied snapshots into script scope over references to live engine state.
- Keep every new public claim paired with a test, fixture, or command.
- Use existing validation diagnostic style and durable-ID rules.
- Add docs in the same phase as behavior changes so validators can compare code and intended truth.
- Record skipped host-dependent checks honestly; do not turn optional smoke into hidden pass/fail.

## Likely Failure Modes

- Worker expands script API into scene mutation to make demos impressive. Reject as out of scope.
- Template command mutates root workspace unexpectedly. Require explicit behavior and tests.
- Docs say "hot reload" without specifying asset/script/data/Rust. Require precise wording.
- Evidence index says `fully_validated` before final validator reconciles residuals. Treat as evidence defect.
- Capture is requested for non-visual changes. Skip and record "not applicable" unless visible behavior changed.
