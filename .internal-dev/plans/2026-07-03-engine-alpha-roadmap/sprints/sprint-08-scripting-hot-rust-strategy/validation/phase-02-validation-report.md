# Sprint 08 Phase 02 Validation Report

Date: 2026-07-03
Validator: Codex validation agent
Phase: Phase 02 - Rust App Template Path

## Result

PASS.

No blocking findings were found. `engine_pack new-app` is implemented as a deterministic support-crate scaffold, rejects existing targets, emits `Cargo.toml`, `src/main.rs`, and `README.md`, builds from `/tmp` with public support crates only, and does not implement or overclaim hot Rust reload, dynamic plugins, or runtime reload.

## Findings

No blocking findings.

Residual worktree state: `git status --short` still shows `M .idea/engine.iml` and `?? .reasonix/`. The Phase 01 audit recorded those exact paths as pre-existing unrelated local changes before Sprint 08 edits, so I did not attribute them to Phase 02. They remain visible in the worktree and should stay out of Phase 02 closeout commits unless the owner deliberately handles them separately.

## Criterion Results

| Criterion | Status | Evidence |
|---|---:|---|
| `engine_pack new-app` is deterministic, has usage text, rejects existing target paths without overwrite, and creates `Cargo.toml`, `src/main.rs`, and `README.md` | PASS | Command dispatch and target protection are in `tools/engine_pack/src/main.rs:43` and `tools/engine_pack/src/main.rs:57-90`; usage text includes `new-app` at `tools/engine_pack/src/main.rs:701-708`; direct `/tmp` generation produced exactly the three expected files. |
| Generated app builds from an off-workspace path using public support crates only and no private renderer internals | PASS | Generated `Cargo.toml` uses only `engine_events`, `input`, and `physics` path dependencies from `tools/engine_pack/src/main.rs:643-654`; `cargo check --manifest-path /tmp/engine-sprint08-template/Cargo.toml` passed. |
| Support-crate scaffold choice is honest in docs and report | PASS | Packaging docs call it a "standalone Rust support-crate scaffold" and distinguish renderer-window app templates as deferred at `docs/api/10-packaging-cli.md:51-58` and `docs/api/10-packaging-cli.md:140-153`; launcher docs do the same at `docs/api/11-runtime-project-launcher.md:99-106`; Phase report states the same. |
| Hot Rust reload, dynamic plugins, and runtime reload are not implemented or overclaimed | PASS | Scoped source scan found only negative/deferred language in implementation surfaces; generated README says it does not implement dynamic Rust reload, plugin ABI loading, or runtime hot reload at `tools/engine_pack/src/main.rs:669-674`. |
| Tests cover usage errors, overwrite protection, generated content, private renderer negative checks, and generated app cargo check | PASS | Tests cover usage error at `tools/engine_pack/tests/cli_validation.rs:284-293`, overwrite protection at `tools/engine_pack/tests/cli_validation.rs:296-332`, generated content and private renderer negative checks at `tools/engine_pack/tests/cli_validation.rs:335-370`, and generated app cargo check at `tools/engine_pack/tests/cli_validation.rs:372-390`. `cargo test -p engine_pack` passed with 20 integration tests. |
| No renderer internals, dogfood, `.idea`, or `.reasonix` touched by Phase 02 | PASS WITH RESIDUAL NOTE | Phase implementation diff is limited to the named CLI/docs/test/report surfaces, except pre-existing local `M .idea/engine.iml` and `?? .reasonix/` remain in the worktree. Phase 01 recorded those as already present at `artifacts/phase-01-current-state-contract-audit.md:57-60`. No `src/renderer` or `apps/dungeon_dogfood` diff was present. |
| Capture is not applicable unless visible renderer/editor behavior changed | PASS | Phase changed CLI/docs/tests only; no renderer/editor visible behavior changed. No capture required. |

## Commands Run

```bash
cargo fmt --check
```

Result: PASS.

```bash
cargo test -p engine_pack
```

Result: PASS. Output included existing renderer dead-code warnings, then `20 passed; 0 failed`.

```bash
cargo check -p engine_pack
```

Result: PASS. Output included existing renderer dead-code warnings.

```bash
rm -rf /tmp/engine-sprint08-template && cargo run -p engine_pack -- new-app /tmp/engine-sprint08-template --id sprint08.template --name "Sprint 08 Template"
```

Result: PASS. Printed `created[app]: /tmp/engine-sprint08-template/Cargo.toml`.

```bash
find /tmp/engine-sprint08-template -maxdepth 3 -type f | sort
```

Result: PASS. Files:

```text
/tmp/engine-sprint08-template/Cargo.toml
/tmp/engine-sprint08-template/README.md
/tmp/engine-sprint08-template/src/main.rs
```

```bash
rg -n "crate::vulkan|renderer::vulkan|renderer::data::|src/renderer/src" /tmp/engine-sprint08-template
```

Result: PASS. No matches. `rg` exited `1` because no matches were found.

```bash
cargo check --manifest-path /tmp/engine-sprint08-template/Cargo.toml
```

Result: PASS. Checked `engine_events`, `input`, `physics`, and `sprint08_template`.

```bash
cargo run -p engine_pack -- new-app
```

Result: PASS for usage behavior. Exited `2` and printed `error[cli.usage]: missing required --id`.

```bash
rg -n "dylib|libloading|notify|watcher|dynamic.*reload|hot.*reload|plugin ABI|runtime reload" tools/engine_pack/src/main.rs tools/engine_pack/tests/cli_validation.rs
```

Result: PASS. Matches were limited to negative assertions/documentation in generated README text and test expectations.

```bash
git diff --check
```

Result: PASS.

```bash
git status --short
```

Result: observed expected Phase 02 changes plus residual pre-existing `M .idea/engine.iml` and `?? .reasonix/`.

## Browser And Capture

Not applicable. This phase did not change visible renderer/editor behavior, and the sprint capture policy only applies to visible renderer/editor changes.

## Missing Tests, Docs, Or Evidence

None blocking. The test suite covers the required Phase 02 behavior, and direct `/tmp` generation plus off-workspace manifest checking validated the user-visible path.

## Required Remediation

None.
