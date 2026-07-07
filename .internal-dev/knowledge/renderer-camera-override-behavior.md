# Renderer Camera/View Behavior in Headless Mode

## Topic
The legacy renderer-owned frame APIs use the renderer's internal `Camera` state during frame rendering. The app-owned caller-view APIs use a supplied `CameraView` instead.

## Source References
- `src/renderer/src/api/renderer.rs` — legacy renderer-owned frame paths build scene camera data from internal renderer camera state
- `src/renderer/src/api/renderer.rs` — `render_scene_with_view` and `render_scene_headless_with_view` render from caller-provided `CameraView`
- `src/renderer/src/data/camera.rs` lines 182-241 — `Camera` struct with position, orientation (quaternion), pitch, yaw
- Default camera: position (0,0,0), identity orientation (looks along -Z), FOVY=70°, near=0.1, far=10000

## Key Takeaways
1. **Legacy renderer-owned APIs override scene camera data** from the renderer's internal `Camera` state.
2. **Caller-view APIs bypass renderer-owned camera ownership**: apps can build `CameraView` from their own camera/controller/collision state and pass it to `render_scene_with_view` or `render_scene_headless_with_view`.
3. **Renderer camera facade calls still have effect on legacy paths**: use `renderer.set_camera_position(position)` for translation-only changes and `renderer.set_camera_look_at(eye, target, up)` when orientation matters on those paths.
4. **Headless capture sidecars must distinguish requested and applied camera/view path**. A caller-view capture should identify the supplied view path; a legacy renderer-owned capture should identify renderer camera calls.
5. **For new app-owned headless capture tests**: prefer caller-provided `CameraView` when the app owns camera/gameplay state. For renderer examples that intentionally use legacy camera ownership, prefer `renderer.set_camera_look_at(eye, target, up)` for deterministic inspection views.
6. **The default projection remains 70° FOVY perspective** unless a caller-provided view overrides the projection details available through the public view DTO.

## Engine Relevance
Critical for any headless capture validation work. Scene-level camera settings are not the app ownership boundary. New app-owned flows should submit a `CameraView`; legacy renderer-owned flows should control the renderer camera explicitly.

## Open Questions
- Should the scene camera take precedence when no FPS controller is installed?
