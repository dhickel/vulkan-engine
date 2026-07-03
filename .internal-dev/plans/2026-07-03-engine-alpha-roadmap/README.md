# Engine Alpha Roadmap

Date: 2026-07-03

Status: planning seed, ready for sprint-by-sprint ironing-out and advanced planning.

## Objective

Move the repository from a renderer/editor/dogfood workbench into an alpha engine build that can be used by this project and shown to early community users without hiding the rough edges.

Alpha does not mean complete. Alpha means:

- a user can create or open a project;
- assets can be packaged, validated, browsed, placed, saved, and reloaded;
- at least one real dogfood app can be built with the engine APIs;
- the facade API stays simple for normal users;
- advanced rendering control exists behind explicit opt-in boundaries;
- validation evidence is repeatable enough that regressions are caught before sprint closeout;
- known defects, deferred features, and code smells are tracked honestly.

## Current Shape

Current source has more foundation than the older gap report claims:

- versioned project manifests and package manifests exist in the renderer asset registry;
- scene JSON persistence exists with stable scene and asset identity concepts;
- `apps/editor` exists and loads project packages, scenes, editor UI panels, placement actions, and capture/debug flags;
- `apps/dungeon_dogfood` exists as a dogfood app;
- `audio`, `physics`, and `scripting` crates exist, but they are thin and not yet integrated as engine-level alpha workflows;
- runtime validation has debug-record and frame-capture paths, with a project-local headless capture validation skill.

There is also process and documentation drift:

- `AGENTS.md` references `.internal-dev/AGENTS.md`, but the current file is archived rather than active;
- `AGENTS.md` describes only renderer/input workspace crates, while the live workspace includes audio, physics, scripting, editor, and dungeon dogfood crates;
- `docs/gap-report.md` is stale and should not drive new sprint planning without a current-state audit.

## Roadmap Backbone

The alpha roadmap should run as gated sprints. Each sprint gets ironed out into a concrete brief, then promoted through the advanced planner into an execution plan. Do not start broad implementation directly from this file.

Recommended sprint order:

1. Alpha baseline audit and process repair.
2. Asset package authoring and validation tools.
3. Editor packaged-asset placement hardening.
4. Runtime project launcher and application dev loop.
5. Event system and application lifecycle.
6. Physics and collision foundation.
7. Audio foundation.
8. Scripting and hot Rust development strategy.
9. Facade API alpha contract.
10. Advanced rendering opt-in contract.
11. Dogfood vertical slice.
12. Quality, bug debt, and code smell burn-down.
13. Alpha release candidate.

## Sprint 01: Alpha Baseline Audit And Process Repair

Purpose: make the repo's planning inputs trustworthy before more code is stacked on them.

Deliverables:

- active `.internal-dev/AGENTS.md` restored or recreated from the archived operating guide;
- root `AGENTS.md`, `README.md`, and API/internal docs updated to reflect actual workspace members;
- stale `docs/gap-report.md` replaced or archived with a current alpha readiness report;
- a consolidated bug/code-smell register under `.internal-dev/bugs/` or `.internal-dev/reviews/`;
- validation matrix for alpha sprints.

Gate:

- `cargo check` and package-level checks run or blockers recorded;
- docs/code divergences are listed explicitly;
- no future sprint may cite the stale gap report as current truth.

## Sprint 02: Asset Package Authoring And Validation Tools

Purpose: turn the existing project/package formats into a usable authoring workflow, not just a runtime/editor loader.

Recommendation: build the package tool in Rust, not Python.

Why:

- the canonical schema types already live in Rust (`Project`, `PackageManifest`, `AssetKind`, `DurableAssetRecord`);
- validators should share logic with runtime/editor code instead of duplicating format rules in Python;
- alpha users should get one cross-platform CLI binary;
- future editor import actions can call the same Rust library functions.

Suggested shape:

- create a standalone `tools/engine_pack` crate or an `xtask`-style crate;
- move reusable format validation into a small library module, probably renderer-owned at first, then split only if dependency boundaries get painful;
- commands: `new-project`, `new-package`, `scan-assets`, `add-asset`, `validate-project`, `validate-package`, `validate-scene`, `pack`;
- package output can remain manifest + copied asset directory for alpha; binary archives are not required yet.
- editor import/refresh should call the same validation/resolution code path where possible.

Non-goal for MVP:

- no complex binary bundle format;
- no thumbnail renderer unless the editor sprint needs it;
- no automatic material graph import.

Gate:

- package/project/scene negative fixtures exist;
- CLI validation errors are clear and stable;
- editor sample project passes CLI validation;
- at least one dogfood package is generated or normalized by the tool.

## Sprint 03: Editor Packaged-Asset Placement Hardening

Purpose: prove the editor can use packaged assets as the normal authoring path.

Deliverables:

- project open, package list, asset browser, scene hierarchy, inspector, transform editing, placement, undo/redo, save, reload;
- placement of packaged models, prefabs, wall chunks, materials, environments, audio clips, scripts, and collision descriptors as each kind becomes supported;
- mesh-derived picking bounds or an explicit alpha fallback with visible limitations;
- editor visual capture proof for startup, placement, selection, inspector edit, save/reload;
- project/package/scene error reporting suitable for a user, not only logs.

Gate:

- editor sample project round-trips;
- at least one saved scene renders through an app/runtime path;
- capture evidence is stored under `.internal-dev/captures/` or `.internal-dev/debug_reports/`;
- known editor limitations are tracked.

## Sprint 04: Runtime Project Launcher And Application Dev Loop

Purpose: make projects runnable without telling users to launch renderer examples manually.

Deliverables:

- decide whether the root `engine` binary becomes a project launcher or remains a migration stub;
- support `cargo run -- --project <engine.project.toml>` or a named app runner;
- load startup scene, packages, default environment, and project settings through one runtime path;
- define how applications are authored separately from engine internals;
- define the fast development loop for application code, including whether app crates live under `apps/`, whether a template app exists, and what can reload without rebuilding the whole engine;
- provide an alpha build/run README for project authors.

Gate:

- sample editor project runs outside the editor;
- dogfood project runs through the same or intentionally documented parallel path;
- CLI arguments are documented and tested enough to not regress every sprint.

## Sprint 05: Event System And Application Lifecycle

Purpose: give apps, tools, scripts, physics, audio, editor, and gameplay one coherent way to react to engine/runtime changes.

Likely event lanes:

- window and app lifecycle events;
- input/action events after the input system frame boundary;
- scene events such as node created, node removed, transform changed, asset placed, scene loaded, scene saved;
- asset events such as package loaded, asset load started, asset ready, asset failed, hot-reload invalidated;
- physics events such as collision enter/stay/exit and trigger enter/exit;
- audio events such as clip started, finished, failed;
- scripting/events bridge with explicit safety limits.

Gate:

- event types are versioned or documented as alpha unstable;
- app/game code can subscribe without owning renderer internals;
- editor commands emit or consume events consistently;
- events are testable without Vulkan where possible.

## Sprint 06: Physics And Collision Foundation

Purpose: stop treating collision as only dogfood-local code and define the engine alpha contract.

Recommended split:

- physics crate owns Rapier-backed rigid bodies, colliders, triggers, queries, and stepping;
- renderer scene owns renderable transforms and durable scene identity;
- an integration layer syncs selected scene nodes/components to physics and back;
- dogfood collision either migrates to engine collision or stays as a documented custom controller until migration is planned.

Gate:

- static/dynamic colliders can be authored or generated from packaged assets;
- basic ray/capsule/shape queries exist for editor picking and gameplay;
- collision events feed the event system;
- deterministic focused tests cover stepping and contact/trigger behavior.

## Sprint 07: Audio Foundation

Purpose: define what "sound support" means for alpha rather than shipping a thin crate as a finished feature.

Deliverables:

- package audio assets as durable IDs;
- load/play/stop clips through facade-level or app-level APIs;
- expose volume groups or at least master/effects/music channels;
- decide whether spatial audio is supported in alpha or deferred;
- editor can reference audio assets even if full audio-authoring UI is deferred.

Gate:

- at least one dogfood or sample app uses packaged audio;
- failure paths are clear when no audio device exists;
- tests cover asset metadata and non-device logic; device-dependent tests are optional/manual.

## Sprint 08: Scripting And Hot Rust Development Strategy

Purpose: decide how users extend apps without rebuilding or touching engine internals.

Recommended stance:

- Rust app crates are the primary alpha path for serious application code;
- Rhai scripting can be an experimental automation/gameplay layer only after bindings are deliberately scoped;
- hot Rust recompilation should be treated as a dev-loop/tooling feature, not as a magical runtime guarantee.

Options to iron out:

- app crates under `apps/<name>` with engine crates as dependencies;
- template generator in the package/tool CLI;
- dynamic plugin ABI later, not alpha default, unless a narrow use case justifies it;
- script assets packaged as durable IDs and run in a sandboxed/event-driven way;
- hot reload asset/data/scripts first, Rust code second.

Gate:

- app template builds without modifying renderer internals;
- script support is either useful and documented or explicitly experimental;
- hot reload claims are tested and scoped.

## Sprint 09: Facade API Alpha Contract

Purpose: lock the simple API path before community alpha.

The beginner facade should stay small:

- create renderer;
- load project/package assets;
- load or create scene;
- update input;
- render scene;
- request/poll asset loads;
- save/load scene;
- basic debug/capture controls.

Likely missing or under-hardened API features:

- stable project runtime API, not only editor-specific load code;
- explicit app loop helper or example template;
- better error types around project/package/scene loading;
- first-class material override API that is more than metadata strings;
- camera mode helpers beyond FPS for editor/game use;
- documented input-profile TOML schema;
- a small public alpha prelude with only supported surface area.

Gate:

- docs show the alpha-supported facade surface;
- examples use the same APIs that users are expected to use;
- unsupported internals are not accidentally exported as beginner APIs;
- API examples compile.

## Sprint 10: Advanced Rendering Opt-In Contract

Purpose: give power users a path without corrupting the facade.

Recommended model:

- keep safe facade hooks for app logic and telemetry;
- keep unsafe `advanced-interop` as explicitly unstable;
- add a limited advanced render extension only after rendergraph resource/order contracts are validated;
- prefer named extension points over raw Vulkan handles until synchronization guarantees are clear.

Candidate advanced features:

- read-only frame/depth/debug textures;
- custom debug views;
- rendergraph pass registration behind a feature gate;
- material/shader override registration with manifest validation;
- capture/readback hooks for tools.

Gate:

- advanced APIs are feature-gated and documented as alpha/unstable;
- misuse risks are documented;
- facade examples do not need advanced APIs for normal workflows.

## Sprint 11: Dogfood Vertical Slice

Purpose: build something real enough to expose API pain.

Target:

- one small playable or explorable dungeon slice using packaged assets and saved scenes where practical;
- dogfood content should use package/project contracts instead of its own isolated manifest unless there is a deliberate migration note;
- runtime should exercise input, camera, scene loading, materials, lighting, environment, and at least one gameplay loop.

Gate:

- dogfood app can be run from clean checkout instructions;
- visual baseline is captured;
- API friction found during dogfood is filed into the alpha backlog, not papered over in app code.

## Sprint 12: Quality, Bug Debt, And Code Smell Burn-Down

Purpose: remove risks that make alpha unstable or embarrassing.

Priority classes:

- known Vulkan lifecycle issues, destroy-path `todo!()`, swapchain cleanup risks;
- unwraps/panics on runtime paths;
- stale docs and examples;
- unbounded frame stalls or asset loading stalls;
- public API exports that are not intended support contracts;
- test gaps in scene/package/runtime flows.

Gate:

- tracked critical bugs closed or explicitly accepted for alpha with mitigation;
- validation matrix is green or has named residuals;
- conservative status language is used when residuals remain.

## Sprint 13: Alpha Release Candidate

Purpose: package the first community-facing build and docs.

Deliverables:

- alpha tag/release notes draft;
- supported platform and driver expectations;
- quickstart project;
- package-tool quickstart;
- editor quickstart;
- dogfood demo instructions;
- known issues;
- contributor/agent workflow notes.

Gate:

- fresh clone validation performed or blockers recorded;
- alpha sample project can be opened, edited, saved, and run;
- dogfood app runs with documented content settings;
- release notes list limitations plainly.

## Quality Gates Between Sprints

Every sprint should close with:

- `cargo check` or scoped equivalent, with blockers recorded;
- focused tests for touched crates;
- renderer examples or app smoke when runtime behavior changed;
- headless capture validation when visuals changed;
- docs updated where public behavior changed;
- `.internal-dev` artifact updated: sprint status, validation evidence, known residuals, and follow-up bugs.

Escalation rules:

- if a sprint uncovers source/docs divergence, fix docs if in scope or file a follow-up before closeout;
- if validation is inconclusive, mark the sprint incomplete or `final_quality_pending`;
- if the same class of bug reappears across two sprints, create a dedicated burn-down sprint before adding more feature surface.

## Tracking Files

- Progress tracker: `sprints/SPRINT-TRACKER.md`
- Sprint template: `sprints/SPRINT-TEMPLATE.md`
- Deep tracks: `tracks/DEEP-TRACKS.md`
- Advanced-planner handoff seed: `advanced-planner-handoff.md`
- Project-local sprint skill: `.internal-dev/skills/engine-alpha-sprint/SKILL.md`
