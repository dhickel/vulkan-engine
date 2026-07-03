# Implementation Notes

## Repo Governance

- Read top-level `AGENTS.md` and relevant module `AGENTS.md` before editing.
- Preserve `.idea/engine.iml` and `.reasonix/`.
- Do not read `.internal-dev` broadly; use this plan and targeted relevant sprint docs only.
- Ask before logging out-of-scope future considerations under `.internal-dev/notes/`.

## Suggested Local Patterns

- For package validation, follow existing collision/audio metadata diagnostics in `asset_registry.rs`.
- For scene validation, follow existing audio/collision reference validation in `scene.rs`.
- For CLI tests, follow `tools/engine_pack/tests/cli_validation.rs`.
- For events, keep `engine_events` independent and use conversion helpers in the producer crate if needed.
- For scripts, prefer a typed result that can convert into `ScriptingEvent::ScriptError`.

## App Template Testing

The template phase should prove generation with one of these:

```bash
cargo run -p engine_pack -- new-app /tmp/engine-template-app --id app.template --name "Template App"
cargo check --manifest-path /tmp/engine-template-app/Cargo.toml
```

If the generated app must be inside this workspace to resolve path dependencies, the command and docs must say that. A test fixture may create the app under a temp directory inside the workspace root if needed, but generated output must not be committed as product code unless deliberately chosen.

## Script Error/Event Shape

Useful narrow helper shape:

```text
evaluate(script_id, source, scope_snapshot) -> ScriptRunReport {
  result,
  emitted_events: Vec<ScriptingEvent>,
  error: Option<ScriptingEvent::ScriptError>
}
```

This is illustrative, not required. The implementation should match existing Rust style.

## Capture Policy

Only run capture when visible renderer/editor behavior changes. Required form:

```bash
RUST_LOG=info timeout --signal=INT 60s cargo run -- \
  --project apps/editor/sample_project/engine.project.toml \
  --headless \
  --capture_target draw \
  --capture_frames 3 \
  --capture_frame_start 5 \
  --capture_frame_interval 5 \
  --capture_dir .internal-dev/captures/sprint-08-scripting-hot-rust-strategy/<case>
```

Desktop screenshots never satisfy capture requirements.

## Phase Reports

Each phase implementation worker drafts:

- `reports/phase-XX-email.md`
- validation command summary in handoff
- residuals and capture applicability

The main thread sends HTML email after validator pass; the plan only defines the expectation.
