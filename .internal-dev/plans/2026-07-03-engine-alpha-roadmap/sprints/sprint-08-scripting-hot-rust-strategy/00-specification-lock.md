# Specification Lock

## Work Classification

Large. Sprint 08 spans docs, tooling, package/scene validation, events, the scripting crate, and evidence/reporting. It is not a full runtime scripting or hot reload implementation.

## Locked Objective

Define and test the alpha extension contract for engine users:

- Rust app crates under `apps/<name>` are the primary custom behavior path.
- Generated/minimal app template tooling may be added through `tools/engine_pack` if it can build without renderer internals.
- Rhai scripting remains experimental and must be event/log/status oriented unless deliberately scoped.
- Hot Rust reload is a dev-loop/tooling research topic, not a runtime guarantee.

## Acceptance Criteria

- Current docs and code are audited for scripting, generated templates, app crates, and hot reload claims.
- A buildable app-template path exists, or a deliberate non-implementation artifact explains why the gate is deferred.
- App-template output does not require touching renderer internals and does not serialize runtime handles.
- Script support, if implemented, has durable script IDs, stable error surfacing, no raw renderer access, and tests.
- Package/scene validation accepts/rejects script assets according to a narrow documented schema if enabled.
- Docs and examples say "experimental" or "deferred" where the code does not prove support.
- Phase reports and validator reports record commands, residuals, and known host-dependent blockers.

## Validation Criteria

- `cargo fmt --check`
- `cargo check`
- `cargo test -p scripting`
- `cargo test -p engine_events`
- `cargo test -p renderer`
- `cargo test -p engine_pack`
- `cargo check -p renderer --examples`
- `cargo check -p editor`
- `cargo check -p dungeon_dogfood`
- `cargo test -p dungeon_dogfood` only if a phase changes dogfood tests/runtime expectations; existing renderer test-profile `russimp_sys` blocker must be recorded if it prevents execution.
- True headless capture only if visible renderer/editor behavior changes:
  `RUST_LOG=info timeout --signal=INT 60s cargo run -- --project apps/editor/sample_project/engine.project.toml --headless --capture_target draw ...`

## Negative Criteria

- No full scripting runtime, runtime scene mutation from scripts, physics/audio mutation from scripts, or broad engine binding surface.
- No `engine_mut`-based docs that encourage arbitrary binding of renderer internals.
- No plugin ABI/dylib reload implementation.
- No "hot Rust reload works" claim unless a bounded command and tests prove exactly that claim.
- No desktop screenshots.
- No unrelated local state changes to `.idea/engine.iml` or `.reasonix/`.

## Non-Goals

- Production script sandbox policy.
- Script-driven editor UI.
- Script package dependency graph.
- Runtime Rust plugin ABI.
- Live code reloading in running game/editor process.
- Dogfood conversion to generated template.

## Constraints

- Code is logical source of truth; docs are intended truth.
- `.internal-dev` is untracked and parent must force-add plan/evidence later.
- `tools/engine_pack` is the likely shipped CLI home for tool changes.
- Python is allowed only for temporary audits, not shipped tooling.
- Preserve existing Sprint 07 residuals honestly.

## Assumptions To Verify In Phase 01

- `src/scripting` remains a thin Rhai wrapper with log bindings and raw `engine_mut`.
- `engine_events` has the required script event vocabulary.
- Existing package/scene validation has no script asset kind yet.
- Existing docs mark templates/scripting/hot reload deferred.

## User Decision Gates

- Stop and ask if the only way to make an app template build is to change workspace membership strategy broadly.
- Stop and ask before implementing plugin ABI, runtime Rust reload, or direct mutable script access.
- Stop and ask before adding out-of-scope future-consideration notes under `.internal-dev/notes/`.

## Stop Rules

- Stop if a phase requires renderer internals to satisfy the app-template gate.
- Stop if script bindings need mutable renderer/scene/physics/audio access to appear useful.
- Stop if validation requires desktop screenshots instead of true engine headless capture.
- Stop if same targeted issue fails validation twice after repair attempts; escalate to fresh high-reasoning repair.
