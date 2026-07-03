# Engine Alpha Sprint Tracker

Date: 2026-07-03

Use this file as the high-level resume point. Individual sprints should get their own subdirectory only after they are selected, ironed out, and sent to advanced planning.

Status values:

- `proposed`: roadmap-level candidate only;
- `ironing-out`: scope and gates are being hardened;
- `planned`: advanced plan exists;
- `executing`: implementation is active;
- `validating`: implementation is complete enough for final checks;
- `closed`: accepted and changelogged;
- `blocked`: cannot proceed without user or external decision.

| Sprint | Status | Target | Primary Gate | Notes |
|---|---|---|---|---|
| 01 | blocked | Alpha baseline audit and process repair | Current docs/process match live repo, stale reports retired | Phase 04 compile/test baseline passed and stale headless docs were remediated; closeout is blocked on changelog timing confirmation and parent-owned commit/push/email gates. Plan suite: `sprints/sprint-01-alpha-baseline-audit/`. Execute on `sprint/alpha-01-baseline-audit`. |
| 02 | closed | Asset package authoring and validation tools | Rust CLI validates/authors package/project/scene fixtures and sample project | Closed with docs, validation reports, final quality review, and changelog. Branch: `sprint/alpha-02-packaging-tools`. Sprint 01 remains blocked on changelog timing confirmation and was not closed by Sprint 02. |
| 03 | closed | Editor packaged-asset placement hardening | Packaged assets place, select, save, reload, and visually prove | Closed with docs, validation reports, final quality review, changelog, pushed closeout/evidence commits, final report email, and accepted `--headless --capture_target draw` evidence under `.internal-dev/captures/sprint-03-editor-packaged-placement-headless-draw/`. Branch: `sprint/alpha-03-editor-packaged-placement`. Sprint 01 remains blocked and was not closed by Sprint 03. |
| 04 | closed | Runtime project launcher and application dev loop | Sample project runs outside editor through documented app path | Closed with root launcher CLI/runtime loop, docs, true `--headless --capture_target draw` proof, debug timing smoke, validation reports, changelog, pushed phase/evidence commits, and final report email. Branch: `sprint/alpha-04-runtime-launcher`. |
| 05 | closed | Event system and application lifecycle | Apps/tools can subscribe to lifecycle/input/scene/asset/physics/audio events | Closed with event crate, renderer/root runtime integration, app consumers, docs, true headless draw capture, final quality review, pushed evidence commits, and final closeout email. Branch: `sprint/alpha-05-event-system-lifecycle`. |
| 06 | closed | Physics and collision foundation | Scene/package authored collision and basic queries/events work | Closed with physics crate alpha contract, package/scene collision metadata validation, physics event bridge, dogfood migration debt artifact, docs, changelog, validation reports, final quality review, pushed phase/evidence commits, and phase closeout email. Branch: `sprint/alpha-06-physics-collision-foundation`. |
| 07 | proposed | Audio foundation | Packaged audio can be referenced and played in a sample/dogfood path | Do not oversell device-dependent support. |
| 08 | proposed | Scripting and hot Rust development strategy | App template and scripting/hot reload boundaries are explicit and tested | Rust app crates first; scripts experimental unless proven. |
| 09 | proposed | Facade API alpha contract | Supported beginner API is documented and examples compile | Keep simple path small. |
| 10 | proposed | Advanced rendering opt-in contract | Advanced APIs feature-gated and misuse risks documented | Avoid raw internals as default surface. |
| 11 | proposed | Dogfood vertical slice | Real dogfood app uses alpha contracts and visual baseline | Prefer package/project contracts over one-off manifests. |
| 12 | proposed | Quality, bug debt, and code smell burn-down | Critical residuals closed or accepted with mitigation | May split into multiple burn-down sprints. |
| 13 | proposed | Alpha release candidate | Fresh-clone style validation and public docs | Produces alpha release notes. |

## Active Sprint Pointer

Current active sprint: none. Next proposed sprint: 07 - Audio foundation.

When starting a sprint:

1. Change the sprint status to `ironing-out`.
2. Create `.internal-dev/plans/2026-07-03-engine-alpha-roadmap/sprints/sprint-XX-<slug>/`.
3. Copy `SPRINT-TEMPLATE.md` into that directory as `README.md`.
4. Fill objective, scope, gates, and validation.
5. Produce an advanced-planner handoff only after the sprint brief is reviewable.

## Cross-Sprint Stop Rules

- Stop feature work if package/project/scene identity regresses into runtime handle serialization.
- Stop release work if validation evidence is missing for changed visual/runtime behavior.
- Stop API expansion if the same need can be solved by documenting an existing facade call.
- Stop sprint closeout if known critical residuals are neither fixed nor tracked.
- Stop using stale docs as truth when code disagrees.
