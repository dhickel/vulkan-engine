---
schema_version: 1
document_type: validation-report
phase_id: phase-01-docs-spec-compatibility-labeling
directive_path: .internal-dev/plans/deprecated-renderer-api-migration/worker-directives/phase-01-docs-spec-compatibility-labeling.md
validator: validator-p1-docs-labeling
validated_at: 2026-07-07
verdict: PASS (with remediation notes)
---

# Phase 01 Validation Report — Docs And Spec Compatibility Labeling

## 1. Summary

Phase 01 edited 11 files (9 docs/api, 2 docs/internal, 2 specs) to add explicit "renderer compatibility path" and "current app-owned path" labels. The core labeling is thorough and correct for the edited files. Three narrow gaps were found in files that were flagged by Phase 00 inventory but not edited: a contradictory debug toggle claim, a missing cross-reference in render hooks docs, and the index page's "Canonical Renderer Example" section lacking an inline label. These are non-blocking but should be remediated. Overall the phase substantially meets its objectives.

## 2. Acceptance Criteria — Pass/Fail

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| AC-01 | Docs clearly distinguish current app-owned APIs from renderer compatibility APIs | **PASS (with notes)** | All edited files now use explicit labels: "renderer compatibility path" and "current app-owned path." 11 files were edited. See Section 6 for three unedited-file gaps. |
| AC-02 | Specs match the accepted compatibility model | **PASS** | `api.md` updated: `set_camera_look_at` status changed to "active (compatibility/capture facade)" with app-owned `CameraView` note. `decisions.md`: DECISION-20260707-06 added documenting the labeling approach. All other spec rows correctly distinguish active from compatibility surfaces. |
| AC-03 | No docs imply renderer examples prove current custom-app guidance | **PASS** | All edited docs correctly present renderer examples as compatibility/demo paths. The index page section is titled "Canonical Renderer Example" making clear it describes example patterns, not custom-app guidance. Quick nav table (line 31) links to `15-app-owned-loop.md`. |
| AC-04 | No code behavior changes | **PASS** | `git diff -- . ':!docs' ':!.internal-dev'` produced **no output**. Only docs and spec files were modified. |
| AC-05 | Changelog exists and explains spec impact | **PASS** | `.internal-dev/changelogs/2026-07-07-phase-01-docs-compatibility-labeling.md` exists, lists all edited files with change descriptions, specifies "docs-only change, no API or behavior modifications." |

## 3. Negative Criteria — Pass/Fail

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| NC-01 | No removal/deprecation claims | **PASS** | Searched edited docs for "deprecat", "remov", "should migrate" — no hits in labeling context. Compatibility APIs are labeled, not deprecated. Changelog explicitly states "No API or behavior changes — docs-only labeling." |
| NC-02 | No "all examples should migrate" implication | **PASS** | No such language found. Compatibility sections are explicitly retained. Edited docs use "good for demos, examples, and smoke testing" framing. |
| NC-03 | No stale quickstart teaching `update_input`/`render_scene` as only path without compatibility label | **PASS** | `01-quickstart.md` now has a prominent compatibility note at section 4: "This section describes the **renderer compatibility path**... For custom apps, prefer the **current app-owned path**" with link to `15-app-owned-loop.md`. `01-student-quickstart.md` has explicit labels for both paths. |
| NC-04 | No contradictory API status between docs and specs | **PASS** | Spec `api.md` correctly labels `set_camera_look_at` as "active (compatibility/capture facade)". `update_input`/`render_scene` documented as compatibility both in specs and docs. `CameraView`/`route_platform_input`/`render_scene_with_view` documented as active both in specs and docs. |

## 4. Cargo Check and Tests

| Command | Result | Notes |
|---------|--------|-------|
| `cargo check` | **PASS** | Compiles with 83 renderer warnings (pre-existing) |
| `cargo test -p engine` | **PASS** | 0 tests (no tests defined yet) |
| `cargo test -p renderer` | **PASS** (pre-existing failure) | 166 passed, 1 failed: `editor_packaged_scene_save_copy_round_trips_model_and_wall_chunk`. Confirmed pre-existing via `git stash` → test → same failure → `git stash pop`. Not caused by Phase 01. |

## 5. Files Edited (Verified)

| file | labels added | quality |
|------|-------------|---------|
| `docs/api/01-quickstart.md` | Compatibility note at section 4; debug toggle note for `route_platform_input`; prelude note | Good — explicit, linked to app-owned loop doc |
| `docs/api/01-student-quickstart.md` | "renderer compatibility path" / "current app-owned path" labels in section 2; snippet labels | Good |
| `docs/api/02-renderer-lifecycle-and-frame-api.md` | Path labels in section 2; `render_scene()` snippet labeled compatibility | Good |
| `docs/api/02-renderer.md` | Compatibility notes on `render_scene()`, Input Integration, Camera Access sections | Good — thorough, all three API groups labeled |
| `docs/api/06-input.md` | "Integration with Renderer — renderer compatibility path" heading + compatibility note | Good |
| `docs/api/06-input-polling-and-listeners.md` | Path labels in section 2; snippet labels; camera controls description | Good |
| `docs/api/12-events-and-lifecycle.md` | Path labels in section 2; `renderer.events_mut()` snippet labeled compatibility | Good |
| `docs/internal/10-event-system-and-lifecycle.md` | Labels on `events()`/`events_mut()`/`set_event_recorder()`; app-owned path in Apps section | Good |
| `docs/internal/12-audio-foundation.md` | Compatibility note on `renderer.events_mut()` dogfood bridge | Good |
| `.internal-dev/specifications/api.md` | `set_camera_look_at` status to "active (compatibility/capture facade)" | Good |
| `.internal-dev/specifications/decisions.md` | DECISION-20260707-06 added | Good |

## 6. Findings — Gaps Requiring Remediation

### 6.1 GAP-01: `docs/api/08-debug.md:17` — Contradictory Debug Toggle Claim

**Severity:** Medium

**Detail:** The debug doc states: "These toggles are handled by `Renderer::update_input()` — if you bypass it for custom input handling, they won't work." This is now **contradicted** by `docs/api/01-quickstart.md:107`, which was correctly updated in Phase 01 to state: "On the **current app-owned path**, `route_platform_input` handles the same debug toggles (F1/F2) through the renderer's platform side effects, so app-owned input routing does not lose debug UI support."

The Phase 00 inventory matrix explicitly flagged this line (row: `docs/api/08-debug.md:17`, classification: `stale_documentation`) with the note: "Phase 01: add note about `route_platform_input` also handling toggles."

**Remediation:** Add to `docs/api/08-debug.md:17` a note that `route_platform_input` (used on the current app-owned path) also handles F1/F2 toggles through the renderer's platform side effects. Reference `15-app-owned-loop.md`.

### 6.2 GAP-02: `docs/api/00-index.md:107-136` — Canonical Renderer Example Section Lacks Inline Label

**Severity:** Low

**Detail:** The "Canonical Renderer Example" section (lines 107-136) presents the renderer-owned pattern (`renderer.update_input()`, `renderer.render_scene()`) as numbered steps without an explicit "renderer compatibility path" label or direct link to `15-app-owned-loop.md` within the section body. The section title "Canonical Renderer Example" does imply it's example-specific, and the quick navigation table (line 31) links to `15-app-owned-loop.md`. However, a reader landing on this section could still interpret these steps as the recommended path for new apps.

The Phase 00 inventory flagged lines 112, 120, 123 with: "Phase 01: Add compatibility label, link to app-owned loop doc."

**Remediation:** Add a brief compatibility note at the top of the section or within the numbered steps, e.g., "These steps describe the renderer compatibility pattern used by examples. For custom apps, see [15-app-owned-loop.md](15-app-owned-loop.md)."

### 6.3 GAP-03: `docs/api/05-render-hooks-and-extension-points.md:8` — Pipeline Diagram Uses `render_scene()` Without Cross-Reference

**Severity:** Low

**Detail:** The pipeline diagram at line 8 shows `Renderer::render_scene(...)` as the entry point. While it parenthetically notes "(or explicit frame API)", it does not mention that `render_scene_with_view` also triggers the same hook pipeline. The Phase 00 inventory flagged this with: "Phase 01: label compatibility in diagram; note that same pipeline serves `render_scene_with_view`."

**Remediation:** Add a note that the same hook pipeline is triggered by `render_scene_with_view` and `render_scene_headless_with_view` with caller-provided `CameraView`.

## 7. Internal Docs — Intentionally Not Labeled (Acceptable)

The following internal docs use `render_scene()` in pipeline diagrams describing renderer internals. Per the directive ("Internal backend docs may mention `render_scene` legitimately; label only where reader guidance or API status is ambiguous"), these are acceptable as-is:

| file | line | reason acceptable |
|------|------|-------------------|
| `docs/internal/01-rendering-pipeline-mental-model.md` | 8 | Internal mental model doc; `render_scene()` describes internal code flow |
| `docs/internal/03-asset-pipeline.md` | 96 | Internal asset doc; mentions polling behavior that applies to both paths |
| `docs/internal/04-api-to-backend-handoff.md` | 8 | Already shows both compatibility and app-owned paths |
| `docs/internal/05-vulkan-sync-and-frame-lifecycle.md` | 8 | Internal Vulkan backend doc; `render_scene()` as internal entry point |
| `docs/internal/07-rendergraph-dependencies-and-aliasing.md` | 8 | Internal rendergraph doc; backend implementation reference |
| `docs/internal/08-scene-flattening-and-culling.md` | 8 | Internal scene internals doc; backend reference |
| `docs/internal/09-input-winit-integration.md` | 8 | Already shows both compatibility and app-owned paths |

## 8. Positive Findings

- **Thorough labeling on edited files:** Every API surface group (`render_scene`, input integration, camera access, events, FPS installation) receives a distinct compatibility label with an explicit app-owned alternative and link to `15-app-owned-loop.md`.
- **Consistent terminology:** All edited files use the directive-prescribed labels "renderer compatibility path" and "current app-owned path" consistently.
- **No removal of content:** All compatibility examples and API documentation are retained, accurately labeled, and not deprecated.
- **Spec alignment:** `api.md` correctly updated; `decisions.md` has a durable decision recording the labeling approach.
- **Changelog quality:** Complete, lists all edited files with change descriptions and validation commands.
- **No product code changes:** Confirmed via `git diff`.
- **Quickstart properly fixed:** The most critical doc (`01-quickstart.md`) now leads with a prominent compatibility note and links to the app-owned path.

## 9. Residual Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Reader of `08-debug.md` is misled about debug toggle support | Medium | Low | GAP-01 remediation; debug toggles still work either way, just docs are wrong |
| Reader of index page assumes compatibility = recommended | Low | Low | Section title says "Canonical Renderer Example"; quick nav links to app-owned loop |
| Reader of render hooks doc misses `render_scene_with_view` connection | Low | Low | "(or explicit frame API)" provides partial context; hooks work for both paths |

## 10. Verdict

**RESULT: PASS (with remediation notes)**

Phase 01 successfully added explicit compatibility/app-owned labeling to all 11 target files. The core acceptance criteria are met: docs clearly distinguish paths, specs match the compatibility model, no docs imply examples prove custom-app guidance, no code was changed, and a changelog exists. Three narrow documentation gaps (GAP-01 through GAP-03) were identified in files flagged by Phase 00 inventory but not edited. These gaps should be remediated but do not block the phase — the critical user-facing docs (quickstart, renderer lifecycle, input) are correctly labeled.

### Remediation Handoff

The following edits are recommended before closing Phase 01:

1. **`docs/api/08-debug.md:17`** — Replace "won't work" with a note that `route_platform_input` also handles debug toggles.
2. **`docs/api/00-index.md:112`** — Add inline compatibility note linking to `15-app-owned-loop.md`.
3. **`docs/api/05-render-hooks-and-extension-points.md:8`** — Add cross-reference to `render_scene_with_view`.

Estimated effort: ~10 minutes. These are purely additive docs changes with no code impact.
