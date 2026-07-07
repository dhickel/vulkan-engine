# Phase 02 Worker Directive: Rust App Template Path

## Objective

Create a minimal, tested app-template path for Rust app crates, preferably through `tools/engine_pack`, or produce a deliberate deferral artifact if implementation would require broad workspace/tooling decisions.

## User-Visible Outcome

An engine user can generate or follow a minimal app crate path that builds against public facade/support crates without modifying renderer internals.

## Editable Targets

- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/cli_validation.rs`
- Optional test fixtures under `tools/engine_pack/tests/fixtures/`
- Docs touched only as needed for this phase, such as `docs/api/10-packaging-cli.md`, `docs/api/01-student-quickstart.md`, `docs/api/11-runtime-project-launcher.md`
- `reports/phase-02-email.md`
- Optional `artifacts/phase-02-app-template-decision.md` if deferring implementation

## Forbidden Scope

- Do not edit renderer internals to make a generated app compile.
- Do not implement dynamic Rust reload or plugin ABI.
- Do not migrate dogfood.
- Do not require hidden root `Cargo.toml` mutation unless explicitly documented and tested.
- Do not touch `.idea/engine.iml` or `.reasonix/`.

## Supporting Docs To Read

- `artifacts/phase-01-current-state-contract-audit.md`
- `00-specification-lock.md`
- `02-target-design.md`
- `shared/implementation-notes.md`
- `tools/AGENTS.md`
- `docs/api/10-packaging-cli.md`
- `tools/engine_pack/src/main.rs`
- `tools/engine_pack/tests/cli_validation.rs`

## Senior-Engineer Guidance

- Generated app code should be boring and compile-first.
- Public facade imports are acceptable; private renderer modules are not.
- Prefer deterministic text generation over a template engine dependency.
- If path dependencies make off-workspace compilation awkward, document exact supported invocation and test it.
- Keep CLI errors stable and consistent with existing usage/validation behavior.

## Ordered Implementation Steps

1. Review Phase 01 audit findings.
2. Decide whether to implement `engine_pack new-app` or document a deferral. Stop for planning/user input if this decision is ambiguous.
3. If implementing, add CLI parsing, usage text, generated file creation, overwrite protection, and deterministic output.
4. Generate `Cargo.toml`, `src/main.rs`, and optional README using public app-crate patterns.
5. Add tests for usage errors, existing-directory/file protection, generated file contents, and generated app compile/check path if practical.
6. Update docs to describe the app-template command and still distinguish app crates from hot Rust reload.
7. Run validation commands.
8. Draft `reports/phase-02-email.md`.

## Acceptance Criteria

- `engine_pack` exposes a documented app-template path or an explicit deferral artifact with rationale.
- If implemented, generated app builds without renderer internals.
- Tests cover generated output and failure behavior.
- Docs no longer say generated templates are deferred if the command exists.
- Hot Rust remains scoped as incremental rebuild/dev-loop, not runtime reload.

## Negative Checks

- `rg -n "crate::vulkan|renderer::vulkan|renderer::data::|src/renderer/src" <generated app>` must not show private renderer internals.
- No dynamic plugin, dylib, watcher, or runtime reload implementation.
- No hidden edits to root workspace config unless explicitly in scope and tested.

## Validation Commands

```bash
cargo fmt --check
cargo test -p engine_pack
cargo check -p engine_pack
```

If `new-app` is implemented:

```bash
rm -rf /tmp/engine-sprint08-template
cargo run -p engine_pack -- new-app /tmp/engine-sprint08-template --id sprint08.template --name "Sprint 08 Template"
cargo check --manifest-path /tmp/engine-sprint08-template/Cargo.toml
```

## Evidence Expectations

- Validator report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/validation/phase-02-validation-report.md`
- Phase report path: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-08-scripting-hot-rust-strategy/reports/phase-02-email.md`
- Record exact generated app command and compile result.

## Stop Conditions

- Stop if generated app compilation requires renderer private modules.
- Stop if template generation needs broad workspace membership policy not covered by the plan.
- Stop if worker is tempted to implement hot Rust reload.

## Do Not Close Unless

- App-template status is implemented/tested or deliberately deferred with rationale.
- Docs match the actual status.
- Validator has enough command evidence to verify the claim.
