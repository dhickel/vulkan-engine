# Phase 02 Worker Directive: Fresh-Clone Validation

## Objective

Prove the release candidate can be built and validated from a clean checkout or clean worktree equivalent with documented commands.

## User-Visible Outcome

A contributor can clone or check out the release-candidate branch and run the documented validation commands without relying on hidden local state.

## Editable Targets

- `reports/phase-02-fresh-clone-validation.md`
- `artifacts/validation-summary.json`
- Validation evidence under `.internal-dev/fresh-clone-validation/sprint-13/`
- Debug reports under `.internal-dev/debug_reports/sprint-13-alpha-release-candidate/` if needed

Product/docs edits are forbidden in this phase unless the validator/main thread routes a scoped release-blocker repair.

## Forbidden Scope

- Do not edit product code or docs as part of the initial validation pass.
- Do not edit `SPRINT-TRACKER.md`.
- Do not edit `.idea/engine.iml` or `.reasonix/`.
- Do not depend on uncommitted local files.

## Supporting Docs To Read

- Phase 01 report and validation report.
- `README.md`
- `docs/api/01-student-quickstart.md`
- `docs/api/10-packaging-cli.md`
- `docs/api/11-runtime-project-launcher.md`

## Senior-Engineer Guidance

- Use a true clone when possible. If network/remote branch availability blocks it, use a clean `git worktree` and record why.
- The clean environment may not contain `.internal-dev` because it is untracked. Run validation from tracked files, then write evidence back to the main workspace sprint directory.
- Missing Vulkan/driver/toolchain support is a real environment blocker; record it precisely and do not claim normal validation.
- Do not repair failures in this phase unless the orchestrator dispatches a separate scoped repair.

## Ordered Steps

1. Record current branch, commit, remote URL, and `git status --short`.
2. Create clean validation directory under `.internal-dev/fresh-clone-validation/sprint-13/`.
3. Prefer:
   ```sh
   git clone --branch sprint/alpha-13-alpha-release-candidate "$(git remote get-url origin)" .internal-dev/fresh-clone-validation/sprint-13/engine
   ```
   If unavailable, use:
   ```sh
   git worktree add .internal-dev/fresh-clone-validation/sprint-13/worktree sprint/alpha-13-alpha-release-candidate
   ```
4. From the clean path, run baseline commands:
   ```sh
   cargo fmt --check
   cargo check
   cargo check -p renderer --examples
   cargo check -p input
   cargo check -p engine
   cargo check -p editor
   cargo check -p dungeon_dogfood
   cargo check -p engine_pack --locked
   cargo test -p input
   cargo test -p engine
   cargo test -p engine_pack --locked
   cargo run -- --help
   cargo run -p engine_pack -- validate-project apps/editor/sample_project/engine.project.toml
   cargo run -p engine_pack -- validate-scene apps/editor/sample_project/scenes/start.engine.scene.json --project apps/editor/sample_project/engine.project.toml
   ```
5. Run root launcher headless draw capture if host supports Vulkan:
   ```sh
   RUST_LOG=info timeout --signal=INT 60s cargo run -- \
     --project apps/editor/sample_project/engine.project.toml \
     --headless \
     --capture_target draw \
     --capture_frames 1 \
     --capture_frame_start 5 \
     --capture_dir .internal-dev/captures/sprint-13-alpha-release-candidate/fresh-clone-runtime-draw
   ```
6. Record environment details: OS, Rust version, `cargo --version`, branch/commit, Vulkan/driver notes if available.
7. Write report and update validation summary.

## Acceptance Criteria

- Clean validation path is isolated and recorded.
- Required command results are recorded with pass/fail/blocker.
- Any environment blocker is precise and conservative.
- Root runtime capture either passes with draw sidecar evidence or is recorded as a release/no-release issue for final decision.

## Negative Checks

- `git status --short` in clean path must not show unexpected product-code modifications after commands.
- No command may use files from the dirty original workspace except the recorded evidence destination.

## Validation Commands

Validator should re-check:

```sh
python -m json.tool .internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/validation-summary.json >/dev/null
git status --short
```

For capture sidecars:

```sh
for f in .internal-dev/fresh-clone-validation/sprint-13/*/.internal-dev/captures/sprint-13-alpha-release-candidate/fresh-clone-runtime-draw/*.json; do
  jq -e '.status == "succeeded" and .capture_target == "draw"' "$f"
done
```

## Stop Conditions

- Stop if neither clone nor worktree can be created.
- Stop if validation uses uncommitted local state.
- Stop if required branch does not exist and the main thread has not approved a substitute base.

## Evidence Expectations

- Worker report: `reports/phase-02-fresh-clone-validation.md`
- Validator report: `validation/phase-02-validation-report.md`
- Updated evidence index with fresh validation mode/path/status.

## Do Not Close Unless

- Clean-path command evidence exists.
- Failures are classified as release-blocking, accepted residual, or tooling constraint.
- Validation summary is conservative.

