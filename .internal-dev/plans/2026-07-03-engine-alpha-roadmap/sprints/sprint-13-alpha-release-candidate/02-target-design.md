# Target Design

## Release Artifact Shape

Sprint 13 should leave two layers of release material:

- Public docs: quickstarts, supported platform/driver/toolchain expectations, dogfood instructions, known issues, and contributor workflow notes.
- Internal evidence: command reports, capture directories, clean-checkout report, validation reports, and `artifacts/validation-summary.json`.

Public docs should be concise and user-facing. Internal evidence should be exhaustive enough for validators to reproduce decisions.

## Validation Architecture

The release candidate decision is evidence-driven:

1. Inventory docs and live contracts.
2. Validate from clean source state.
3. Prove sample project package/editor/runtime workflows.
4. Prove dogfood full-content runtime and visual baseline.
5. Draft release notes and reconcile all residuals in final quality review.

Each phase writes:

- worker report under `reports/`;
- validator report under `validation/`;
- evidence entries in `artifacts/validation-summary.json`.

## Fresh-Clone Equivalent

Preferred:

```sh
mkdir -p .internal-dev/fresh-clone-validation/sprint-13
git clone --branch sprint/alpha-13-alpha-release-candidate "$(git remote get-url origin)" .internal-dev/fresh-clone-validation/sprint-13/engine
cd .internal-dev/fresh-clone-validation/sprint-13/engine
```

Fallback when network or remote state makes clone impractical:

```sh
mkdir -p .internal-dev/fresh-clone-validation/sprint-13
git worktree add .internal-dev/fresh-clone-validation/sprint-13/worktree sprint/alpha-13-alpha-release-candidate
cd .internal-dev/fresh-clone-validation/sprint-13/worktree
git status --short
```

Either route must record branch, commit, command output summary, path, and any `TOOLING_CONSTRAINT`.

## Sample Project Proof Design

Use the canonical sample project as input, but write release proof artifacts under this sprint:

- generated pack output: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-pack/`
- edited scene copy: `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-13-alpha-release-candidate/artifacts/sample-edited-scene.engine.scene.json`
- root runtime capture: `.internal-dev/captures/sprint-13-alpha-release-candidate/sample-runtime-draw/`
- editor capture: `.internal-dev/captures/sprint-13-alpha-release-candidate/editor-sample-draw/`

If the release intentionally updates the canonical sample scene, the worker must justify it and preserve a pre-change copy in sprint artifacts.

## Dogfood Proof Design

Dogfood release proof must use documented full-content settings:

```sh
DUNGEON_DOGFOOD_FAST_STARTUP=0 \
DUNGEON_DOGFOOD_LOAD_PROPS=1 \
DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV=1 \
RUST_LOG=info timeout --signal=INT 60s cargo run -p dungeon_dogfood -- --level generated_sprawl
```

Visual proof must use true headless draw capture. If dogfood does not yet support this, implement the narrowest app-owned path that:

- constructs `Renderer::new_headless`;
- loads the same content settings and level selector;
- renders with `render_scene_headless`;
- accepts `--headless`, `--capture_target draw`, and sequence capture flags consistent with root/editor conventions where practical;
- writes capture output under `.internal-dev/captures/sprint-13-alpha-release-candidate/dogfood-draw/`.

Do not route dogfood proof through desktop screenshots.

## Release/No-Release Criteria

Release candidate may pass only when:

- clean validation proves clone/worktree instructions;
- sample project and dogfood workflows pass or have user-accepted non-critical residuals;
- all visual proof is draw-target headless capture;
- public docs match validated behavior;
- known issues list every release-relevant residual;
- final quality validator passes.

No-release if:

- clean validation fails on required setup;
- a user cannot run documented quickstarts from a clean checkout;
- sample edit/save/run or dogfood run proof fails;
- required visual proof is missing or inconclusive;
- release notes overclaim unsupported behavior.

