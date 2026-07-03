# Senior Engineer Guidance

## Decision Principles

- Treat release readiness as a contract audit, not a feature grab bag. Fix only release-blocking defects proven by validation.
- Code is logical truth; docs are intended truth. When they diverge, either fix docs or record a release-blocking issue.
- The release candidate should be honest. A known issue is acceptable alpha debt only when it is visible, bounded, and not fatal to the documented quickstart.
- Clean-checkout validation is not optional. If it cannot run, the release candidate is blocked or `TOOLING_CONSTRAINT` must be accepted by the user.

## Direct Targets

- Public docs should converge around `README.md`, `docs/api/00-index.md`, quickstarts, runtime launcher docs, editor docs, packaging docs, dogfood README, and any selected release notes/known issues file.
- Internal evidence belongs under this sprint directory, `.internal-dev/captures/sprint-13-alpha-release-candidate/`, `.internal-dev/debug_reports/sprint-13-alpha-release-candidate/`, and `.internal-dev/fresh-clone-validation/sprint-13/`.
- Keep `SPRINT-TRACKER.md` unchanged. Main thread reconciles tracker status after review.

## Gotchas

- Existing local Sprint 09 files are dirty. Do not build a release claim on uncommitted local state.
- Dogfood windowed smoke can pass while headless visual proof is absent. That is not sufficient for release visuals.
- Prior sprint validation summaries are useful history, not current proof. Re-run release-critical commands.
- Renderer warning noise may be accepted residual debt, but release docs must not hide it if it affects user commands.
- `cargo test -p dungeon_dogfood` had a historical test-profile blocker. Re-check it before carrying the residual forward.

## Best Practices

- Prefer command output summaries in reports over huge pasted logs.
- Capture commands should use timeout-bound runs and deterministic output directories.
- For edited sample scenes, use a sprint artifact copy first. Update canonical fixtures only if release docs or app defaults truly require it.
- When adding release-blocker code, keep it narrow and add focused tests around argument parsing, output paths, and no-runtime-handle serialization.
- Use `jq` or `python -m json.tool` to validate evidence JSON, but use engine APIs/CLIs for project/package/scene validation.

## Likely Failure Modes

- Fresh clone lacks local `.internal-dev` plans because they are untracked. The clean validation worker must copy only the needed sprint evidence into the clean validation path or run commands from tracked project files and record evidence back in the main workspace.
- Capture output exists but sidecars show `capture_target = "present"`. That fails release visual proof.
- Docs refer to paths under `/tmp` or old sprint capture directories as canonical release evidence. That fails stale-reference review.
- Dogfood full content times out before first meaningful frame. Use debug timing records to determine whether this is startup cost, asset load failure, or a real release blocker.

## Reasoning Cues

- If a problem affects only internal contributor comfort and not documented release workflows, track it as known issue or future debt.
- If a problem breaks a documented command from a clean checkout, it is release-blocking unless the user explicitly narrows the release scope.
- If a command requires local GPU/driver capabilities, record platform/driver details and failure mode instead of pretending it is universal.

