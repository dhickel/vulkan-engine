use crate::vulkan::vk_render;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_CAPTURE_ROOT: &str = ".internal-dev/captures";

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
    /// Optional path to a model file to load during startup.
    ///
    /// When `preload_startup_scene` is true and this is `None`, the renderer loads
    /// the built-in default model (`DEFAULT_STARTUP_MODEL_PATH`). When `Some(path)`,
    /// the given model is loaded instead. This is a renderer-owned startup/debug preload
    /// intended for diagnostic and early-visualization scenarios; it is not a root project
    /// scene selection mechanism.
    pub startup_model_path: Option<PathBuf>,
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
            startup_model_path: None,
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
    default_manual_output_dir: PathBuf,
    manual_output_dir: PathBuf,
    manual_sequence: u32,
    next_manual_capture_frame: Option<u32>,
    last_status: Option<FrameCaptureStatus>,
}

impl Default for FrameCaptureScheduler {
    fn default() -> Self {
        Self::new("engine")
    }
}

impl FrameCaptureScheduler {
    pub fn new(app_name: impl Into<String>) -> Self {
        let app_name = app_name.into();
        let default_manual_output_dir = default_capture_run_dir(&app_name);
        Self {
            app_name,
            single_captures: Vec::new(),
            sequences: Vec::new(),
            manual_output_dir: default_manual_output_dir.clone(),
            default_manual_output_dir,
            manual_sequence: 0,
            next_manual_capture_frame: None,
            last_status: None,
        }
    }

    pub fn set_app_name(&mut self, app_name: impl Into<String>) {
        let app_name = app_name.into();
        let was_using_default_manual_dir = self.manual_output_dir == self.default_manual_output_dir;
        self.default_manual_output_dir = default_capture_run_dir(&app_name);
        if was_using_default_manual_dir {
            self.manual_output_dir = self.default_manual_output_dir.clone();
        }
        self.app_name = app_name;
    }

    pub fn configure_manual_output_dir(
        &mut self,
        output_dir: Option<PathBuf>,
    ) -> Result<(), FrameCaptureConfigError> {
        let output_dir = output_dir.unwrap_or_else(|| self.default_manual_output_dir.clone());
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
        let scheduled_frame = self
            .next_manual_capture_frame
            .unwrap_or_else(|| current_frame.wrapping_add(1));
        self.next_manual_capture_frame = Some(scheduled_frame.wrapping_add(1));
        let output_path = manual_capture_path(
            &self.manual_output_dir,
            &self.app_name,
            scheduled_frame,
            target,
            manual_index,
        );
        self.single_captures.push(ScheduledSingleCapture {
            frame_number: scheduled_frame,
            request: FrameCaptureRequest::new(target, output_path),
            source: FrameCaptureSource::Manual,
        });
        Ok(())
    }

    pub fn due_captures(&mut self, frame_number: u32) -> Vec<DueFrameCapture> {
        let mut due = Vec::new();
        let mut pending = Vec::with_capacity(self.single_captures.len());
        for capture in self.single_captures.drain(..) {
            if capture.frame_number <= frame_number {
                due.push(DueFrameCapture {
                    request: capture.request,
                    frame_number: capture.frame_number,
                    sequence_index: None,
                    source: capture.source,
                });
            } else {
                pending.push(capture);
            }
        }
        self.single_captures = pending;
        if !self
            .single_captures
            .iter()
            .any(|capture| capture.source == FrameCaptureSource::Manual)
        {
            self.next_manual_capture_frame = None;
        }

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
    default_capture_run_dir("engine")
}

pub fn default_capture_run_dir(app_name: &str) -> PathBuf {
    default_capture_root().join(format!(
        "{}-{}-pid{}",
        sanitize_capture_name(app_name),
        current_capture_timestamp(),
        std::process::id()
    ))
}

pub fn default_single_capture_path(
    app_name: &str,
    frame_number: u32,
    target: CaptureTarget,
) -> PathBuf {
    single_capture_path(
        default_capture_run_dir(app_name),
        app_name,
        frame_number,
        target,
    )
}

pub fn single_capture_path(
    output_dir: impl AsRef<Path>,
    app_name: &str,
    frame_number: u32,
    target: CaptureTarget,
) -> PathBuf {
    output_dir.as_ref().join(format!(
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
    let sanitized = sanitized.trim_matches('-').to_string();
    if sanitized.is_empty() {
        "engine".to_string()
    } else {
        sanitized
    }
}

fn current_capture_timestamp() -> String {
    let millis_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i128;
    capture_timestamp_from_unix_millis(millis_since_epoch)
}

fn capture_timestamp_from_unix_millis(millis_since_epoch: i128) -> String {
    let seconds = millis_since_epoch.div_euclid(1_000);
    let millis = millis_since_epoch.rem_euclid(1_000);
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days as i64);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}-{millis:03}")
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
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
    fn overdue_single_capture_fires_on_next_valid_frame() {
        let mut scheduler = FrameCaptureScheduler::new("api_test");
        scheduler
            .schedule_single_capture(3, FrameCaptureRequest::new(CaptureTarget::Draw, "late.png"))
            .unwrap();

        let due = scheduler.due_captures(5);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].frame_number, 3);
        assert_eq!(due[0].request.output_path, PathBuf::from("late.png"));
        assert!(scheduler.due_captures(6).is_empty());
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
            .configure_manual_output_dir(Some(PathBuf::from("captures/run")))
            .unwrap();
        scheduler
            .queue_manual_capture(9, CaptureTarget::Present)
            .unwrap();

        assert!(scheduler.due_captures(9).is_empty());
        let due = scheduler.due_captures(10);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].source, FrameCaptureSource::Manual);
        assert_eq!(
            due[0].request.output_path,
            PathBuf::from("captures/run/engine-editor-frame-10-present-manual-0000.png")
        );
    }

    #[test]
    fn default_capture_root_uses_internal_captures() {
        assert_eq!(
            default_capture_root(),
            PathBuf::from(".internal-dev/captures")
        );
    }

    #[test]
    fn run_dir_includes_sanitized_app_timestamp_and_pid() {
        let run_dir = default_capture_run_dir("Renderer Facade API Test!");
        let root = default_capture_root();
        assert!(run_dir.starts_with(&root));

        let folder = run_dir
            .file_name()
            .and_then(|name| name.to_str())
            .expect("run folder should be utf-8");
        assert!(folder.starts_with("renderer-facade-api-test-"));
        assert!(folder.ends_with(&format!("-pid{}", std::process::id())));
    }

    #[test]
    fn empty_sanitized_capture_name_falls_back_to_engine() {
        let path = single_capture_path("captures/run", " --- ", 7, CaptureTarget::Draw);
        assert_eq!(path, PathBuf::from("captures/run/engine-frame-7-draw.png"));
    }

    #[test]
    fn scheduler_reuses_one_default_manual_run_dir() {
        let mut scheduler = FrameCaptureScheduler::new("Editor");
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();

        let first_due = scheduler.due_captures(1);
        let second_due = scheduler.due_captures(2);
        assert_eq!(first_due.len(), 1);
        assert_eq!(second_due.len(), 1);
        let first_parent = first_due[0].request.output_path.parent();
        let second_parent = second_due[0].request.output_path.parent();
        assert_eq!(first_parent, second_parent);
        assert!(first_parent
            .unwrap()
            .starts_with(PathBuf::from(".internal-dev/captures")));
    }

    #[test]
    fn rapid_manual_captures_are_staggered_across_future_frames() {
        let mut scheduler = FrameCaptureScheduler::new("Editor");
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();

        let first_due = scheduler.due_captures(1);
        let second_due = scheduler.due_captures(2);
        let third_due = scheduler.due_captures(3);
        assert_eq!(first_due.len(), 1);
        assert_eq!(second_due.len(), 1);
        assert_eq!(third_due.len(), 1);
        assert_eq!(first_due[0].frame_number, 1);
        assert_eq!(second_due[0].frame_number, 2);
        assert_eq!(third_due[0].frame_number, 3);

        let first_path = &first_due[0].request.output_path;
        let second_path = &second_due[0].request.output_path;
        let third_path = &third_due[0].request.output_path;
        assert_ne!(first_path, second_path);
        assert_ne!(first_path, third_path);
        assert_ne!(second_path, third_path);
        assert_eq!(first_path.parent(), second_path.parent());
        assert_eq!(first_path.parent(), third_path.parent());
        assert!(first_path.ends_with(PathBuf::from("editor-frame-1-present-manual-0000.png")));
        assert!(second_path.ends_with(PathBuf::from("editor-frame-2-present-manual-0001.png")));
        assert!(third_path.ends_with(PathBuf::from("editor-frame-3-present-manual-0002.png")));
    }

    #[test]
    fn set_app_name_refreshes_default_manual_run_dir() {
        let mut scheduler = FrameCaptureScheduler::new("Editor");
        scheduler.set_app_name("API Test");
        scheduler
            .queue_manual_capture(0, CaptureTarget::Present)
            .unwrap();

        let due = scheduler.due_captures(1);
        assert_eq!(due.len(), 1);
        let run_folder = due[0]
            .request
            .output_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .expect("manual capture should have a run folder");
        assert!(run_folder.starts_with("api-test-"));
        assert!(due[0]
            .request
            .output_path
            .ends_with(PathBuf::from("api-test-frame-1-present-manual-0000.png")));
    }

    #[test]
    fn unix_millis_timestamp_uses_locked_shape() {
        assert_eq!(capture_timestamp_from_unix_millis(0), "19700101-000000-000");
        assert_eq!(
            capture_timestamp_from_unix_millis(1_700_000_000_123),
            "20231114-221320-123"
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
