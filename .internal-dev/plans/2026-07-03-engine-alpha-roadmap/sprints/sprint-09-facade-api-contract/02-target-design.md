# Target Design

## Facade Contract Model

Sprint 09 should produce a tiered public contract:

- Supported alpha beginner facade: small API path for getting a renderer on screen, loading project/package assets, creating/loading/saving scenes, updating input, rendering frames, requesting/polling asset loads, and using debug/capture controls.
- Compatibility public exports: existing root exports that remain available because current tests or likely users depend on them, but are not advertised as beginner alpha.
- Advanced interop: explicitly gated or clearly advanced APIs, preserving the `advanced-interop` feature boundary.
- Internal implementation details: modules/types that should not be newly documented as user-facing even if reachable through legacy exports.
- Deferred: real gaps that require larger work in Sprint 10 or later.

## Beginner API Surface

The beginner path should remain intentionally small:

- `Renderer`, `RendererConfig`, `RendererError`, `FrameRenderOutcome`, `FrameContext`.
- `Scene`, `SceneNodeId`, `PointLight`, `SceneAssetReference`, scene validation/load/save helpers.
- `AssetManager`, `LoadTicket`, `LoadStatus`, durable asset/project/package manifest types, package/project validation helpers.
- `InputSystem`, `InputSnapshot`, `ActionMap`, `ActionId`, layer descriptors, action bindings required by examples.
- `CaptureTarget`, `FrameCaptureRequest`, `FrameCaptureSequence`, `FrameCaptureStatus`, debug timing/runtime mode controls.
- Error enums needed for user-facing diagnostics.

Do not automatically include `SceneWorld`, command history commands, low-level camera/frustum helpers, animation internals, or advanced rendergraph access in the beginner prelude unless the worker proves a beginner example needs them.

## Export Shape

Preferred low-risk design:

- Keep existing root re-exports for compatibility.
- Add or document a small `renderer::prelude` only if it can be curated and compile-checked.
- Keep `renderer::api` as the explicit source namespace for facade modules.
- Add docs that explain root exports may include compatibility items and that the alpha-supported beginner set is the prelude/docs table, not every public symbol.
- Preserve `advanced-interop` as the opt-in path for advanced internals.

## Example Contract

Examples must compile and should demonstrate only APIs that docs call supported beginner facade or explicitly diagnostic. Existing diagnostic examples may stay, but their docs should not present advanced/legacy usage as the beginner template.

Expected example path:

- a minimal app loop helper/template pattern in docs or example code if current examples are too much to copy;
- `api_test` and demos use public facade/root/prelude imports consistently;
- if an example requires compatibility-only exports, label it diagnostic or advanced-adjacent.

## Targeted Hardening Areas

Phase 03 may harden these areas only in narrow, validated ways:

- Error docs/tests around project/package/scene loading, without replacing the full error hierarchy.
- Input-profile TOML schema documentation and compile/test coverage if the schema already exists.
- Camera mode helper docs or small helper wrappers if current camera types are already public but confusing.
- First-class material override API only if a small existing facade hook can be wrapped; otherwise document current metadata/string behavior as a deferred friction item.
- Debug/capture docs that use true engine-owned headless draw capture when visual proof is relevant.

## Documentation Shape

Docs should make the alpha contract explicit without rewriting every chapter:

- Update `docs/api/00-index.md` to define the supported alpha path and classify legacy chapters.
- Add or update a concise facade contract chapter if the worker finds it cleaner than bloating the index.
- Cross-link duplicate docs to the canonical supported chapter.
- Keep old docs available if still useful, but mark them legacy/reference where appropriate.

## Evidence Shape

`artifacts/validation-summary.json` is the canonical status file. It must include:

- top-level conservative status;
- phase report paths and pass/fail state;
- command results;
- capture status and artifact directory if applicable;
- email/report status paths;
- residual risks;
- model/tooling constraints;
- superseded artifacts.
