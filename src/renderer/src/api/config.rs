use crate::vulkan::vk_render;
use std::path::{Path, PathBuf};

const DEFAULT_CAPTURE_ROOT: &str = ".internal-dev/debug_reports";
const DEFAULT_MANUAL_CAPTURE_DIR: &str = ".internal-dev/debug_reports/manual-captures";

/// Startup runtime mode used for controlled render-path validation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum DebugRuntimeMode {
    #[default]
    Default,
    TestPbr,
    TestUnlit,
}

impl DebugRuntimeMode {
    pub fn from_label(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "testpbr" => Some(Self::TestPbr),
            "testunlit" => Some(Self::TestUnlit),
            _ => None,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::TestPbr => "testpbr",
            Self::TestUnlit => "testunlit",
        }
    }
}

impl From<DebugRuntimeMode> for vk_render::DebugRuntimeMode {
    fn from(value: DebugRuntimeMode) -> Self {
        match value {
            DebugRuntimeMode::Default => Self::Default,
            DebugRuntimeMode::TestPbr => Self::TestPbr,
            DebugRuntimeMode::TestUnlit => Self::TestUnlit,
        }
    }
}

/// Controls how the engine handles `.meta` sidecar manifest files for assets.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum AssetManifestMode {
    /// Manifests are ignored even if present.
    Disabled,
    /// Manifests are loaded when found; parse errors log a warning and fall back to defaults.
    #[default]
    BestEffort,
    /// Manifests are required for assets that have them; parse errors fail the load.
    Strict,
}

/// Controls texture compression behavior.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum TextureCompressionMode {
    /// Compression is disabled; textures remain uncompressed (e.g. R8G8B8A8).
    #[default]
    Disabled,
    /// Textures are compressed if supported by the format/device, falling back to uncompressed.
    Auto,
    /// Textures must be compressed; failure to compress returns an error.
    Force,
}

/// Configuration for texture compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub mode: TextureCompressionMode,
    pub quality: u8, // 0..=100, interpreted by backend (e.g. BC7 quality)
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            mode: TextureCompressionMode::Disabled,
            quality: 50, // Balanced default
        }
    }
}

/// Policy configuration for asset loading behavior.
#[derive(Debug, Clone)]
pub struct AssetPolicyConfig {
    pub manifest_mode: AssetManifestMode,
    /// When true, filename patterns (e.g. `_n.`, `_normal.`) can influence sRGB/sampler defaults.
    pub allow_filename_heuristics: bool,
    /// Texture compression policy.
    pub compression: CompressionConfig,
}

impl Default for AssetPolicyConfig {
    fn default() -> Self {
        Self {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: CompressionConfig::default(),
        }
    }
}

/// App-controlled visual baseline applied to skybox tonemapping and IBL shading.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VisualTuning {
    pub exposure: f32,
    pub gamma: f32,
    pub ibl_ambient_scale: f32,
}

impl Default for VisualTuning {
    fn default() -> Self {
        Self {
            exposure: 4.5,
            gamma: 2.2,
            ibl_ambient_scale: 1.0,
        }
    }
}

/// Public renderer configuration contract.
#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub app_name: String,
    pub validation_layer: bool,
    pub shader_debug_mode: DebugRuntimeMode,
    pub compile_shaders: bool,
    /// When true, renderer preloads the built-in startup/debug scene during initialization.
    /// Disable this for app-driven scenes to reduce startup latency.
    pub preload_startup_scene: bool,
    /// App-owned visual tuning applied consistently across skybox and IBL lighting.
    pub visual_tuning: VisualTuning,
    /// Reserved in v1. `true` currently returns `RendererError::Unsupported`.
    pub headless: bool,
    pub asset_policy: AssetPolicyConfig,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            window_width: 1920,
            window_height: 1080,
            app_name: "engine".to_string(),
            validation_layer: false,
            shader_debug_mode: DebugRuntimeMode::Default,
            compile_shaders: false,
            preload_startup_scene: true,
            visual_tuning: VisualTuning::default(),
            headless: false,
            asset_policy: AssetPolicyConfig::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum CaptureTarget {
    #[default]
    Present,
    Draw,
}

impl CaptureTarget {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "present" => Some(Self::Present),
            "draw" => Some(Self::Draw),
            _ => None,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Draw => "draw",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FrameCaptureRequest {
    pub target: CaptureTarget,
    pub output_path: PathBuf,
    pub sidecar_path: Option<PathBuf>,
}

impl FrameCaptureRequest {
    pub fn new(target: CaptureTarget, output_path: impl Into<PathBuf>) -> Self {
        Self {
            target,
            output_path: output_path.into(),
            sidecar_path: None,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FrameCaptureSequence {
    pub target: CaptureTarget,
    pub output_dir: PathBuf,
    pub start_frame: u32,
    pub interval: u32,
    pub remaining: u32,
}

impl FrameCaptureSequence {
    pub fn new(
        target: CaptureTarget,
        output_dir: impl Into<PathBuf>,
        start_frame: u32,
        interval: u32,
        count: u32,
    ) -> Result<Self, FrameCaptureConfigError> {
        if interval == 0 {
            return Err(FrameCaptureConfigError::InvalidInterval);
        }
        if count == 0 {
            return Err(FrameCaptureConfigError::InvalidCount);
        }

        Ok(Self {
            target,
            output_dir: output_dir.into(),
            start_frame,
            interval,
            remaining: count,
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DueFrameCapture {
    pub request: FrameCaptureRequest,
    pub frame_number: u32,
    pub sequence_index: Option<u32>,
    pub source: FrameCaptureSource,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FrameCaptureSource {
    Single,
    Sequence,
    Manual,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FrameCaptureStatus {
    Pending {
        frame_number: u32,
        target: CaptureTarget,
        output_path: PathBuf,
        source: FrameCaptureSource,
    },
    Succeeded {
        frame_number: u32,
        target: CaptureTarget,
        output_path: PathBuf,
        sidecar_path: Option<PathBuf>,
        source: FrameCaptureSource,
        width: u32,
        height: u32,
    },
    BackendNotImplemented {
        frame_number: u32,
        target: CaptureTarget,
        output_path: PathBuf,
        source: FrameCaptureSource,
    },
    Failed {
        frame_number: u32,
        target: CaptureTarget,
        output_path: PathBuf,
        source: FrameCaptureSource,
        message: String,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FrameCaptureConfigError {
    InvalidCount,
    InvalidInterval,
    EmptyOutputPath,
    EmptyOutputDir,
}

impl std::fmt::Display for FrameCaptureConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCount => write!(f, "frame capture count must be >= 1"),
            Self::InvalidInterval => write!(f, "frame capture interval must be >= 1"),
            Self::EmptyOutputPath => write!(f, "frame capture output path must not be empty"),
            Self::EmptyOutputDir => write!(f, "frame capture output directory must not be empty"),
        }
    }
}

impl std::error::Error for FrameCaptureConfigError {}

#[derive(Debug, Clone)]
struct ScheduledSingleCapture {
    frame_number: u32,
    request: FrameCaptureRequest,
    source: FrameCaptureSource,
}

#[derive(Debug, Clone)]
struct ActiveCaptureSequence {
    config: FrameCaptureSequence,
    total_count: u32,
    emitted: u32,
}

#[derive(Debug, Clone)]
pub struct FrameCaptureScheduler {
    app_name: String,
    single_captures: Vec<ScheduledSingleCapture>,
    sequences: Vec<ActiveCaptureSequence>,
    manual_output_dir: PathBuf,
    manual_sequence: u32,
    last_status: Option<FrameCaptureStatus>,
}

impl Default for FrameCaptureScheduler {
    fn default() -> Self {
        Self::new("engine")
    }
}

impl FrameCaptureScheduler {
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            single_captures: Vec::new(),
            sequences: Vec::new(),
            manual_output_dir: default_manual_capture_dir(),
            manual_sequence: 0,
            last_status: None,
        }
    }

    pub fn set_app_name(&mut self, app_name: impl Into<String>) {
        self.app_name = app_name.into();
    }

    pub fn configure_manual_output_dir(
        &mut self,
        output_dir: Option<PathBuf>,
    ) -> Result<(), FrameCaptureConfigError> {
        let output_dir = output_dir.unwrap_or_else(default_manual_capture_dir);
        ensure_non_empty_path(&output_dir, FrameCaptureConfigError::EmptyOutputDir)?;
        self.manual_output_dir = output_dir;
        Ok(())
    }

    pub fn schedule_single_capture(
        &mut self,
        frame_number: u32,
        request: FrameCaptureRequest,
    ) -> Result<(), FrameCaptureConfigError> {
        ensure_non_empty_path(
            &request.output_path,
            FrameCaptureConfigError::EmptyOutputPath,
        )?;
        self.single_captures.push(ScheduledSingleCapture {
            frame_number,
            request,
            source: FrameCaptureSource::Single,
        });
        Ok(())
    }

    pub fn configure_sequence(
        &mut self,
        sequence: FrameCaptureSequence,
    ) -> Result<(), FrameCaptureConfigError> {
        if sequence.remaining == 0 {
            return Err(FrameCaptureConfigError::InvalidCount);
        }
        if sequence.interval == 0 {
            return Err(FrameCaptureConfigError::InvalidInterval);
        }
        ensure_non_empty_path(
            &sequence.output_dir,
            FrameCaptureConfigError::EmptyOutputDir,
        )?;

        self.sequences.push(ActiveCaptureSequence {
            total_count: sequence.remaining,
            config: sequence,
            emitted: 0,
        });
        Ok(())
    }

    pub fn queue_manual_capture(
        &mut self,
        current_frame: u32,
        target: CaptureTarget,
    ) -> Result<(), FrameCaptureConfigError> {
        ensure_non_empty_path(
            &self.manual_output_dir,
            FrameCaptureConfigError::EmptyOutputDir,
        )?;
        let manual_index = self.manual_sequence;
        self.manual_sequence = self.manual_sequence.wrapping_add(1);
        let output_path = manual_capture_path(
            &self.manual_output_dir,
            &self.app_name,
            current_frame.wrapping_add(1),
            target,
            manual_index,
        );
        self.single_captures.push(ScheduledSingleCapture {
            frame_number: current_frame.wrapping_add(1),
            request: FrameCaptureRequest::new(target, output_path),
            source: FrameCaptureSource::Manual,
        });
        Ok(())
    }

    pub fn due_captures(&mut self, frame_number: u32) -> Vec<DueFrameCapture> {
        let mut due = Vec::new();
        let mut pending = Vec::with_capacity(self.single_captures.len());
        for capture in self.single_captures.drain(..) {
            if capture.frame_number == frame_number {
                due.push(DueFrameCapture {
                    request: capture.request,
                    frame_number,
                    sequence_index: None,
                    source: capture.source,
                });
            } else {
                pending.push(capture);
            }
        }
        self.single_captures = pending;

        for sequence in self.sequences.iter_mut() {
            if sequence.config.remaining == 0 || frame_number < sequence.config.start_frame {
                continue;
            }
            let offset = frame_number - sequence.config.start_frame;
            if offset % sequence.config.interval != 0 {
                continue;
            }

            let sequence_index = sequence.emitted;
            let output_path = sequence_capture_path(
                &sequence.config.output_dir,
                &self.app_name,
                frame_number,
                sequence.config.target,
                sequence_index,
            );
            sequence.config.remaining -= 1;
            sequence.emitted += 1;

            due.push(DueFrameCapture {
                request: FrameCaptureRequest::new(sequence.config.target, output_path),
                frame_number,
                sequence_index: Some(sequence_index),
                source: FrameCaptureSource::Sequence,
            });
        }

        self.sequences.retain(|sequence| {
            sequence.config.remaining > 0 && sequence.emitted < sequence.total_count
        });
        for capture in due.iter() {
            self.last_status = Some(FrameCaptureStatus::Pending {
                frame_number: capture.frame_number,
                target: capture.request.target,
                output_path: capture.request.output_path.clone(),
                source: capture.source,
            });
        }
        due
    }

    pub fn record_status(&mut self, status: FrameCaptureStatus) {
        self.last_status = Some(status);
    }

    pub fn last_status(&self) -> Option<&FrameCaptureStatus> {
        self.last_status.as_ref()
    }
}

pub fn default_capture_root() -> PathBuf {
    PathBuf::from(DEFAULT_CAPTURE_ROOT)
}

pub fn default_manual_capture_dir() -> PathBuf {
    PathBuf::from(DEFAULT_MANUAL_CAPTURE_DIR)
}

pub fn default_single_capture_path(
    app_name: &str,
    frame_number: u32,
    target: CaptureTarget,
) -> PathBuf {
    default_capture_root().join(format!(
        "{}-frame-{}-{}.png",
        sanitize_capture_name(app_name),
        frame_number,
        target.as_label()
    ))
}

fn sequence_capture_path(
    output_dir: &Path,
    app_name: &str,
    frame_number: u32,
    target: CaptureTarget,
    sequence_index: u32,
) -> PathBuf {
    output_dir.join(format!(
        "{}-frame-{}-{}-seq-{:04}.png",
        sanitize_capture_name(app_name),
        frame_number,
        target.as_label(),
        sequence_index
    ))
}

fn manual_capture_path(
    output_dir: &Path,
    app_name: &str,
    frame_number: u32,
    target: CaptureTarget,
    manual_index: u32,
) -> PathBuf {
    output_dir.join(format!(
        "{}-frame-{}-{}-manual-{:04}.png",
        sanitize_capture_name(app_name),
        frame_number,
        target.as_label(),
        manual_index
    ))
}

fn sanitize_capture_name(app_name: &str) -> String {
    let mut sanitized = String::new();
    for ch in app_name.chars() {
        if ch.is_ascii_alphanumeric() {
            sanitized.push(ch.to_ascii_lowercase());
        } else if !sanitized.ends_with('-') {
            sanitized.push('-');
        }
    }
    sanitized.trim_matches('-').to_string()
}

fn ensure_non_empty_path(
    path: &Path,
    error: FrameCaptureConfigError,
) -> Result<(), FrameCaptureConfigError> {
    if path.as_os_str().is_empty() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn single_capture_fires_once_at_requested_frame() {
        let mut scheduler = FrameCaptureScheduler::new("api_test");
        scheduler
            .schedule_single_capture(
                3,
                FrameCaptureRequest::new(CaptureTarget::Present, "capture.png"),
            )
            .unwrap();

        assert!(scheduler.due_captures(2).is_empty());
        let due = scheduler.due_captures(3);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].request.output_path, PathBuf::from("capture.png"));
        assert!(scheduler.due_captures(3).is_empty());
        assert!(scheduler.due_captures(4).is_empty());
    }

    #[test]
    fn sequence_capture_fires_exact_count_at_interval() {
        let mut scheduler = FrameCaptureScheduler::new("demo pbr");
        scheduler
            .configure_sequence(
                FrameCaptureSequence::new(CaptureTarget::Draw, "captures", 2, 3, 3).unwrap(),
            )
            .unwrap();

        let mut frames = Vec::new();
        for frame in 0..12 {
            if !scheduler.due_captures(frame).is_empty() {
                frames.push(frame);
            }
        }

        assert_eq!(frames, vec![2, 5, 8]);
        assert!(scheduler.due_captures(11).is_empty());
    }

    #[test]
    fn manual_capture_uses_default_dir_and_next_frame() {
        let mut scheduler = FrameCaptureScheduler::new("Engine Editor");
        scheduler
            .queue_manual_capture(9, CaptureTarget::Present)
            .unwrap();

        assert!(scheduler.due_captures(9).is_empty());
        let due = scheduler.due_captures(10);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].source, FrameCaptureSource::Manual);
        assert_eq!(
            due[0].request.output_path,
            PathBuf::from(".internal-dev/debug_reports/manual-captures/engine-editor-frame-10-present-manual-0000.png")
        );
    }

    #[test]
    fn scheduler_rejects_unbounded_sequences() {
        assert_eq!(
            FrameCaptureSequence::new(CaptureTarget::Present, "captures", 0, 1, 0).unwrap_err(),
            FrameCaptureConfigError::InvalidCount
        );
        assert_eq!(
            FrameCaptureSequence::new(CaptureTarget::Present, "captures", 0, 0, 1).unwrap_err(),
            FrameCaptureConfigError::InvalidInterval
        );
    }
}
