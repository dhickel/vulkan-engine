# Implementation Notes

## Required Branch And Dirty State

- Required branch: `sprint/alpha-02-packaging-tools`.
- Preserve and exclude existing dirty state:
  - `.idea/engine.iml`
  - `.reasonix/`

Before each phase commit, run:

```bash
git status --short --branch
git diff --stat
```

Stage only in-scope files. Use `git status --short` after staging to verify `.idea/engine.iml` and `.reasonix/` are not staged.

## Likely Dependencies

`renderer` already depends on `serde`, `serde_json`, and `toml`. The CLI can depend on `renderer` and may add:

- `clap` for commands if chosen;
- `tempfile` for tests if useful;
- `assert_cmd` and `predicates` for CLI integration tests if useful.

Avoid dependency churn if simple standard-library tests are enough.

## Fixture Guidance

Create small, explicit fixtures. Suggested invalid cases:

- package missing `format_version`;
- unsupported package `format_version`;
- duplicate durable asset ID;
- asset ID equal to or shaped like `models/crate.glb`;
- absolute or escaping asset path;
- package ID mismatch against project reference;
- project missing startup scene file;
- project references missing package manifest;
- scene missing `format_version`;
- scene duplicate node IDs;
- scene parent references missing node;
- scene asset reference missing durable `id`;
- scene contains runtime handle-shaped identity such as `{ "slot": 4, "generation": 2 }`.

## Report Matrix Requirement

Every phase report and AgentMail HTML report must include a table with:

| File | Created/Changed/Deleted | Added Lines | Removed Lines | Commit | GitHub Link |
|---|---:|---:|---:|---|---|

Use `git diff --numstat <base>..<head>` or equivalent after commit. If a GitHub link cannot be formed, write `unavailable` and explain why.

## GitHub Link Format

If remote is GitHub, derive:

- commit: `https://github.com/<owner>/<repo>/commit/<hash>`
- compare: `https://github.com/<owner>/<repo>/compare/<base>...<head>`

If remote is not GitHub or unavailable, record the remote URL and mark link fields unavailable.
