//! Renderer facade primitives re-exported from the `renderer` support crate.

use renderer::Camera;

pub use renderer::api::{
    default_capture_root, default_capture_run_dir, default_manual_capture_dir,
    default_single_capture_path, parse_package_manifest, single_capture_path,
    validate_package_manifest_file, validate_package_manifest_str, validate_project_file,
    validate_project_str, validate_scene_file, validate_scene_file_with_options,
    validate_scene_str, validate_scene_str_with_options, AssetManifestMode, AssetPolicyConfig,
    AssetRegistry, AssetRegistryError, CameraView, CaptureTarget, DebugTimingRow,
    DebugTimingSnapshot, DebugUiFrameContext, DueFrameCapture, DurableAssetRecord,
    EnvironmentHandle, EnvironmentRuntimeStatus, EnvironmentSource, EnvironmentState, FacePattern,
    FilterMode, FrameCaptureConfigError, FrameCaptureRequest, FrameCaptureScheduler,
    FrameCaptureSequence, FrameCaptureSource, FrameCaptureStatus, HookError, LoadStatus,
    LoadTicket, PackageAssetRecord, PackageManifest, PackageValidationOptions, PbrMaterialDesc,
    ProceduralMeshData, ProceduralVertex, Project, ProjectPackage, ProjectSettings,
    ProjectValidationOptions, RendererFrameError, RendererInitError, ResolvedTexturePolicy,
    SamplerOverride, SceneAssetReference, SceneFragment, SceneFragmentMount, SceneFragmentNode,
    SceneFragmentNodeId, SceneNodeSummary, SceneValidationOptions, TextureLoadOptions,
    ValidationArea, ValidationDiagnostic, ValidationError, WrapMode,
};
pub use renderer::{
    boxed_render_hook, install_app_event_logger, AnimationPlayer, AssetError, AssetKind,
    AssetManager, BoxedRenderHook, DebugRuntimeMode, EventBus, FrameContext, FrameRenderOutcome,
    MaterialHandle, MeshHandle, PointLight, PointLightId, RenderHook, RenderHookContext, Renderer,
    RendererConfig, RendererError, RendererInputRouting, RendererInputSuppression, Scene,
    SceneError, SceneNodeId, SceneWorld, TextureHandle, VisualTuning,
};

/// Construct a [`CameraView`] for a framebuffer/window size.
///
/// Zero-sized dimensions are normalized for projection construction so callers
/// can use this helper directly during resize transitions.
pub fn camera_view_for_size(camera: &Camera, width: u32, height: u32) -> CameraView {
    let aspect = if height == 0 {
        1.0
    } else {
        width.max(1) as f32 / height as f32
    };

    CameraView::from_camera(camera, aspect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_view_for_size_uses_normal_aspect_ratio() {
        let camera = Camera::default();

        assert_eq!(
            camera_view_for_size(&camera, 1920, 1080),
            CameraView::from_camera(&camera, 1920.0 / 1080.0)
        );
    }

    #[test]
    fn camera_view_for_size_treats_zero_width_as_one() {
        let camera = Camera::default();

        assert_eq!(
            camera_view_for_size(&camera, 0, 100),
            CameraView::from_camera(&camera, 1.0 / 100.0)
        );
    }

    #[test]
    fn camera_view_for_size_treats_zero_height_as_square_aspect() {
        let camera = Camera::default();

        assert_eq!(
            camera_view_for_size(&camera, 640, 0),
            CameraView::from_camera(&camera, 1.0)
        );
    }
}
