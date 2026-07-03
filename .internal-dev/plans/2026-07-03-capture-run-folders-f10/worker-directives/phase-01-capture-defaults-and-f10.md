# Worker Directive: Capture Defaults and F10

## Scope

Implement the locked capture-default contract in one coherent pass.

## Editable Files

- `src/renderer/src/api/config.rs`
- `src/renderer/src/api/mod.rs`
- `src/renderer/examples/common/mod.rs`
- `src/renderer/examples/api_test.rs`
- `apps/editor/src/main.rs`
- `apps/editor/src/launch.rs`, only if needed for tests/docs alignment

## Requirements

1. Change default capture root to `.internal-dev/captures`.
2. Add a shared run-folder helper using sanitized app name, timestamp, and pid.
3. Ensure empty sanitized app names fall back to `engine`.
4. Ensure a renderer scheduler has one stable default manual run folder.
5. Ensure examples/editor compute one run folder per launch and use it for default single, sequence, and manual captures.
6. Preserve exact CLI override behavior.
7. Add `F10` windowed manual capture in examples and editor.
8. Ignore `F10` key-repeat events.
9. Keep headless single/N-frame capture behavior working.

## Validation Handoff

Run formatting and compile/test checks before handing off:

```bash
cargo fmt
cargo check
cargo check -p renderer
cargo check -p renderer --examples
cargo check -p input
cargo test -p renderer capture
```

If a command fails because of unrelated environment/toolchain issues, capture the exact failure and stop for validator review.
