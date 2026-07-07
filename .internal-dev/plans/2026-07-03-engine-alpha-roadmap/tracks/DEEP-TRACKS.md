# Engine Alpha Deep Tracks

Date: 2026-07-03

Purpose: capture the deep planning surface behind the alpha roadmap. Each track below should be ironed out with the user before it becomes an advanced-planner sprint.

## Track A: Asset Packaging Tools

Problem:

The engine has project/package/scene formats, but authors need tools to create, validate, repair, and pack assets without hand-editing TOML/JSON.

Recommended architecture:

- Rust CLI as canonical tool: `tools/engine_pack` or `xtask`;
- shared validation library using the same schema/types as runtime/editor code;
- editor calls the same validation/resolution functions where possible;
- Python only for temporary audits/import migration scripts.

Required commands:

- `new-project`: create `engine.project.toml`, scene folder, package folder, and sample package;
- `new-package`: create package manifest with stable `package_id`;
- `scan-assets`: discover models, textures, environments, audio, scripts, and collision metadata candidates;
- `add-asset`: add one asset record with generated stable ID and optional metadata;
- `validate-package`: check package manifest, duplicate IDs, kind support, paths, metadata, and dependency references;
- `validate-project`: check project manifest, enabled packages, startup scene, default environment, and settings;
- `validate-scene`: check scene stable IDs, graph validity, package asset references, materials, environment, and unsupported handles;
- `pack`: produce an alpha distribution folder with manifests and copied assets.

Format implications:

- package records must grow beyond render assets: audio clips, scripts, collision shapes, prefab metadata, and maybe app modules;
- package dependencies need a decision before community alpha;
- package output can be folder-based for alpha; binary archives can wait.

Quality gates:

- sample editor project validates;
- dogfood content can be represented or has a documented migration gap;
- invalid fixtures produce stable, readable errors;
- CLI and editor disagreeing about validity is a blocking bug.

## Track B: Packaged Asset Placement In Editor

Problem:

The editor can list and place some package records, but alpha authoring needs a reliable package-to-scene workflow.

Supported alpha placement kinds:

- model;
- prefab;
- wall chunk;
- environment;
- material reference or material override;
- collision shape or generated collision descriptor;
- audio emitter/reference;
- script behavior reference, if scripting graduates from experimental.

Required editor behavior:

- package reload/refresh;
- search/filter by kind/tags/package;
- place asset into scene through command history;
- selection and transform editing;
- inspect durable package ID, asset ID, path hint, metadata, and warnings;
- save/reload scene without losing durable identity;
- show validation errors before broken assets become placeable.

Quality gates:

- visual proof for placement/select/save/reload;
- scene round-trip tests for each supported asset kind;
- missing or changed package assets produce clear degraded scene state instead of crashes.

## Track C: Runtime Project Launcher And App Dev Loop

Problem:

Examples are not enough for alpha users. Apps need a stable way to run projects and iterate without touching renderer internals.

Recommended alpha path:

- app crates under `apps/<app_name>`;
- engine/root launcher can open `engine.project.toml` for non-custom projects;
- package tool can generate an app template;
- runtime project loading uses the same project/package/scene contracts as editor.

Hot development stance:

- hot reload assets, scenes, scripts, and data first;
- Rust application code reload is a separate advanced dev-loop feature;
- default alpha Rust loop can be `cargo run -p <app>` with incremental builds;
- dynamic Rust plugins should be deferred unless a narrow proof makes it worth the ABI and safety cost.

Quality gates:

- generated app template builds and runs;
- project launcher can run sample project;
- dogfood app either uses the same path or has a documented reason not to.

## Track D: Event System

Problem:

Input, scene commands, assets, physics, audio, scripts, editor UI, and dogfood gameplay need a common event contract.

Recommended shape:

- `EventBus` or `EngineEvents` crate/module with typed events;
- app-level subscription without raw renderer ownership;
- frame-staged dispatch to avoid mid-render mutation hazards;
- event recording/debug view for diagnosis;
- bridge from input action maps to app events.

Initial event families:

- lifecycle: app start, project loaded, scene loaded, scene saved, shutdown;
- input/action: action pressed/released/axis after input dispatch;
- scene: node created/removed/renamed/transformed, asset placed, material changed;
- asset: package loaded, asset loading, ready, failed, invalidated;
- physics: collision enter/stay/exit, trigger enter/exit, query hit;
- audio: clip started/stopped/finished/failed;
- scripting: script event emitted, script error.

Quality gates:

- events can be tested without Vulkan;
- event ordering is documented;
- editor commands and runtime systems do not emit contradictory events.

## Track E: Physics And Collision

Problem:

The physics crate exists, and dogfood has custom collision, but alpha needs a clear engine-level collision model.

Recommended shape:

- physics crate owns Rapier world, bodies, colliders, triggers, queries, and stepping;
- scene keeps durable authored identity;
- integration layer maps scene components to physics handles;
- package metadata can define collision source: box, sphere, capsule, trimesh/convex hull, or generated from model bounds.

Editor needs:

- collision preview toggle;
- attach/edit simple collider metadata;
- generated collision for wall chunks/prefabs;
- placement validation for collision descriptors.

Dogfood decision:

- migrate dogfood collision to engine physics when it improves reuse;
- keep custom controller temporarily if engine physics would delay alpha, but file migration debt.

Quality gates:

- unit tests for stepping, queries, triggers, and contact events;
- simple dogfood/sample collision proof;
- no renderer dependency in core physics tests.

## Track F: Audio

Problem:

Audio exists as a thin Rodio wrapper, but alpha needs asset identity, runtime control, and failure behavior.

Recommended alpha support:

- packaged audio asset kind;
- load clip by durable ID;
- play/stop/pause handle;
- master/effects/music volume groups;
- device-missing failure path;
- optional positional metadata only if dogfood proves it.

Editor support:

- audio asset records visible in browser;
- attach audio emitter/reference metadata to scene nodes;
- full waveform/timeline editing deferred.

Quality gates:

- non-device metadata tests;
- manual or optional smoke for playback;
- sample/dogfood use before advertising support.

## Track G: Scripting And Hot Rust

Problem:

The scripting crate can eval Rhai, but it does not yet define safe engine bindings or app workflow. Hot Rust is a separate problem.

Recommended alpha stance:

- Rust app crates are the primary extension model;
- scripts are event-driven and sandboxed;
- script bindings start small: log, read input/action, inspect selected scene metadata, emit events;
- mutating scene/physics/audio from scripts should wait until event ordering and borrow safety are clear.

Hot Rust options:

- incremental app crate rebuild as default;
- app template with small dependency surface;
- dynamic library/plugin reload as future research;
- script/data hot reload before Rust code reload.

Quality gates:

- script assets can be packaged if enabled;
- script errors surface in editor/runtime status;
- no unsandboxed direct renderer access from scripts.

## Track H: Facade And Advanced Rendering API

Problem:

The facade must remain simple while advanced users still need escape hatches.

Beginner facade:

- renderer lifecycle;
- project/package/scene loading;
- input update and action snapshots;
- asset request/poll;
- scene mutation commands;
- render scene;
- capture/debug basics.

Advanced opt-in:

- feature-gated raw interop remains unsafe and unstable;
- future advanced rendergraph/pass APIs need explicit resource and ordering contracts;
- custom debug views and readback/capture hooks are safer early wins.

Quality gates:

- examples use supported facade APIs;
- advanced APIs are isolated and documented as unstable;
- facade docs and exports do not accidentally promise internals.
