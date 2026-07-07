---
schema_version: 1
document_type: changelog
status: completed
owner: deprecated-renderer-api-migration
created: 2026-07-07
---

# Phase 01 — Docs Compatibility Labeling

## Summary

Added explicit "renderer compatibility path" and "current app-owned path" labels to all docs/api and docs/internal files that presented renderer-owned `update_input`/`render_scene`/`install_default_fps_input` APIs as the primary or default custom-app path. No API or behavior changes — docs-only labeling.

## Files Edited

| file | change |
|------|--------|
| `docs/api/01-quickstart.md` | Added compatibility note at section 4 (The Render Loop) referencing app-owned path. Updated section 6 (Debug UI) to note `route_platform_input` handles same toggles. Updated prelude note. |
| `docs/api/01-student-quickstart.md` | Added explicit "renderer compatibility path" and "current app-owned path" labels in section 2. Updated frame styles list and code snippet labels. |
| `docs/api/02-renderer-lifecycle-and-frame-api.md` | Added explicit path labels in section 2. Marked `render_scene()` snippet as compatibility path. |
| `docs/api/02-renderer.md` | Added compatibility note to `render_scene()` section. Labeled Input Integration as compatibility path with app-owned reference. Labeled Camera Access as compatibility/capture path. |
| `docs/api/06-input.md` | Labeled renderer input integration as compatibility path with app-owned reference. |
| `docs/api/06-input-polling-and-listeners.md` | Added explicit path labels in section 2. Labeled code snippets. Updated camera controls description. |
| `docs/api/12-events-and-lifecycle.md` | Added explicit path labels in section 2. Labeled `renderer.events_mut()` snippet as compatibility path with app-owned note. |
| `docs/internal/10-event-system-and-lifecycle.md` | Labeled `events()`/`events_mut()`/`set_event_recorder()` as renderer compatibility path. Labeled app-owned path in Apps section. |
| `docs/internal/12-audio-foundation.md` | Added compatibility note to `renderer.events_mut()` dogfood bridge description. |
| `.internal-dev/specifications/api.md` | Updated `set_camera_look_at` status to "active (compatibility/capture facade)" with app-owned `CameraView` note. |
| `.internal-dev/specifications/decisions.md` | Added DECISION-20260707-06 recording the compatibility labeling approach. |

## Spec Impact

None — docs-only change, no API or behavior modifications.

## Validation

```bash
cargo check 2>&1 | tail -3
rg -n "update_input\(|render_scene\(|install_default_fps_input" docs/api docs/internal 2>&1 | head -30
```

All remaining matches of these symbols in docs are now properly labeled as compatibility path references.
