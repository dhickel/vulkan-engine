use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use super::config::FrameCaptureConfigError;
use crate::scene::scene_world::SceneNodeId;

#[derive(Debug)]
pub enum RendererError {
    Init(RendererInitError),
    Frame(RendererFrameError),
    Scene(SceneError),
    Asset(AssetError),
    Hook(HookError),
    CaptureConfig(FrameCaptureConfigError),
    Unsupported(String),
    InvalidState(String),
    /// The Vulkan device has been lost (VK_ERROR_DEVICE_LOST).
    /// The host application should destroy and recreate the Renderer.
    DeviceLost,
    /// A Vulkan operation was attempted after a prior terminal error.
    /// The backend is poisoned; destroy and recreate the Renderer.
    BackendPoisoned(String),
    /// A backend-mutating operation failed during hook execution.
    /// The frame backend may require drain/poison before the next submit.
    HookBackend(String),
}

impl Display for RendererError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(err) => write!(f, "renderer init error: {err}"),
            Self::Frame(err) => write!(f, "renderer frame error: {err}"),
            Self::Scene(err) => write!(f, "scene error: {err}"),
            Self::Asset(err) => write!(f, "asset error: {err}"),
            Self::Hook(err) => write!(f, "hook error: {err}"),
            Self::CaptureConfig(err) => write!(f, "frame capture configuration error: {err}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::DeviceLost => write!(f, "renderer error: Vulkan device lost"),
            Self::BackendPoisoned(msg) => write!(f, "renderer backend poisoned: {msg}"),
            Self::HookBackend(msg) => write!(f, "hook-induced backend failure: {msg}"),
        }
    }
}

impl Error for RendererError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Init(err) => Some(err),
            Self::Frame(err) => Some(err),
            Self::Scene(err) => Some(err),
            Self::Asset(err) => Some(err),
            Self::Hook(err) => Some(err),
            Self::CaptureConfig(err) => Some(err),
            Self::Unsupported(_) => None,
            Self::InvalidState(_) => None,
            Self::DeviceLost => None,
            Self::BackendPoisoned(_) => None,
            Self::HookBackend(_) => None,
        }
    }
}

impl From<RendererInitError> for RendererError {
    fn from(value: RendererInitError) -> Self {
        Self::Init(value)
    }
}

impl From<RendererFrameError> for RendererError {
    fn from(value: RendererFrameError) -> Self {
        Self::Frame(value)
    }
}

impl From<SceneError> for RendererError {
    fn from(value: SceneError) -> Self {
        Self::Scene(value)
    }
}

impl From<AssetError> for RendererError {
    fn from(value: AssetError) -> Self {
        Self::Asset(value)
    }
}

impl From<HookError> for RendererError {
    fn from(value: HookError) -> Self {
        Self::Hook(value)
    }
}

impl From<FrameCaptureConfigError> for RendererError {
    fn from(value: FrameCaptureConfigError) -> Self {
        Self::CaptureConfig(value)
    }
}

impl From<Box<dyn std::error::Error>> for RendererError {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        RendererError::InvalidState(error.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for RendererError {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        RendererError::InvalidState(error.to_string())
    }
}

#[derive(Debug)]
pub enum RendererInitError {
    Vulkan(String),
    Window(String),
    ShaderCompile(String),
    StartupScene(String),
}

impl Display for RendererInitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vulkan(msg) => write!(f, "{msg}"),
            Self::Window(msg) => write!(f, "{msg}"),
            Self::ShaderCompile(msg) => write!(f, "{msg}"),
            Self::StartupScene(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for RendererInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

#[derive(Debug)]
pub enum RendererFrameError {
    Input(String),
    Resize(String),
    Render(String),
    FrameContext(String),
}

impl Display for RendererFrameError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(msg) => write!(f, "{msg}"),
            Self::Resize(msg) => write!(f, "{msg}"),
            Self::Render(msg) => write!(f, "{msg}"),
            Self::FrameContext(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for RendererFrameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

#[derive(Debug)]
pub enum SceneError {
    InvalidNode(SceneNodeId),
    StaleNode(SceneNodeId),
    CycleDetected,
    InvalidParent(SceneNodeId),
    MergeFailed(String),
    InvalidPointLight(String),
    StalePointLight(crate::api::scene::PointLightId),
    InvalidDirectionalLight(String),
    StaleDirectionalLight(crate::api::scene::DirectionalLightId),
    InvalidSpotLight(String),
    StaleSpotLight(crate::api::scene::SpotLightId),
    UnsupportedLightFeature(String),
    UnsupportedSceneVersion { found: u32, expected: u32 },
    MissingAssetId(String),
    BadSerializedParent { node_id: String, parent_id: String },
    DuplicateSerializedNodeId(String),
    DisconnectedGraph(String),
    AnimationError(AnimationError),
    CommandError(CommandError),
    SerializationError(String),
    InvalidMutation(String),
}

impl Display for SceneError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNode(node) => write!(
                f,
                "invalid scene node handle (slot={}, generation={})",
                node.slot, node.generation
            ),
            Self::StaleNode(node) => write!(
                f,
                "stale scene node handle (slot={}, generation={})",
                node.slot, node.generation
            ),
            Self::CycleDetected => write!(f, "cycle detected in scene hierarchy"),
            Self::InvalidParent(node) => write!(
                f,
                "invalid parent node handle (slot={}, generation={})",
                node.slot, node.generation
            ),
            Self::MergeFailed(msg) => write!(f, "{msg}"),
            Self::InvalidPointLight(msg) => write!(f, "invalid point light: {msg}"),
            Self::InvalidDirectionalLight(msg) => write!(f, "invalid directional light: {msg}"),
            Self::StaleDirectionalLight(id) => write!(
                f,
                "stale directional light handle (slot={}, generation={})",
                id.slot, id.generation
            ),
            Self::InvalidSpotLight(msg) => write!(f, "invalid spot light: {msg}"),
            Self::StaleSpotLight(id) => write!(
                f,
                "stale spot light handle (slot={}, generation={})",
                id.slot, id.generation
            ),
            Self::UnsupportedLightFeature(msg) => write!(f, "unsupported light feature: {msg}"),
            Self::StalePointLight(id) => write!(
                f,
                "stale point light handle (slot={}, generation={})",
                id.slot, id.generation
            ),
            Self::UnsupportedSceneVersion { found, expected } => write!(
                f,
                "unsupported scene format version {found}; expected {expected}"
            ),
            Self::MissingAssetId(context) => {
                write!(f, "missing durable asset id for {context}")
            }
            Self::BadSerializedParent { node_id, parent_id } => write!(
                f,
                "scene node '{node_id}' references missing parent '{parent_id}'"
            ),
            Self::DuplicateSerializedNodeId(id) => {
                write!(f, "duplicate serialized scene node id '{id}'")
            }
            Self::DisconnectedGraph(msg) => write!(f, "disconnected scene graph: {msg}"),
            Self::AnimationError(err) => write!(f, "animation error: {err}"),
            Self::CommandError(err) => write!(f, "command error: {err}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
            Self::InvalidMutation(msg) => write!(f, "invalid mutation: {msg}"),
        }
    }
}

impl From<AnimationError> for SceneError {
    fn from(err: AnimationError) -> Self {
        Self::AnimationError(err)
    }
}

impl From<CommandError> for SceneError {
    fn from(err: CommandError) -> Self {
        Self::CommandError(err)
    }
}

impl Error for SceneError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

#[derive(Debug, Clone)]
pub enum AssetError {
    Load {
        path: Option<PathBuf>,
        message: String,
    },
    Io {
        path: PathBuf,
        message: String,
    },
    Decode {
        path: PathBuf,
        message: String,
    },
    InvalidHandle {
        resource: &'static str,
        slot: u32,
        generation: u32,
    },
    StaleHandle {
        resource: &'static str,
        slot: u32,
        generation: u32,
    },
    NotLoaded {
        resource: &'static str,
        slot: u32,
        generation: u32,
    },
    OutOfBounds {
        resource: &'static str,
        slot: u32,
        generation: u32,
    },
    ReservedHandle {
        resource: &'static str,
        slot: u32,
        generation: u32,
    },
    UnknownTicket {
        ticket: u64,
    },
    CancelRejected {
        ticket: u64,
        reason: String,
    },
    ManifestParse {
        path: PathBuf,
        message: String,
    },
    Cache(String),
    Sync(String),
    Unsupported(String),
    Internal(String),
}

impl Display for AssetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load { path, message } => {
                if let Some(path) = path {
                    write!(f, "asset load failed for '{}': {message}", path.display())
                } else {
                    write!(f, "asset load failed: {message}")
                }
            }
            Self::Io { path, message } => {
                write!(f, "asset io failed for '{}': {message}", path.display())
            }
            Self::Decode { path, message } => {
                write!(f, "asset decode failed for '{}': {message}", path.display())
            }
            Self::InvalidHandle {
                resource,
                slot,
                generation,
            } => write!(
                f,
                "invalid {resource} handle (slot={slot}, generation={generation})"
            ),
            Self::StaleHandle {
                resource,
                slot,
                generation,
            } => write!(
                f,
                "stale {resource} handle (slot={slot}, generation={generation})"
            ),
            Self::NotLoaded {
                resource,
                slot,
                generation,
            } => write!(
                f,
                "{resource} handle is not loaded (slot={slot}, generation={generation})"
            ),
            Self::OutOfBounds {
                resource,
                slot,
                generation,
            } => write!(
                f,
                "{resource} handle is out of bounds (slot={slot}, generation={generation})"
            ),
            Self::ReservedHandle {
                resource,
                slot,
                generation,
            } => write!(
                f,
                "cannot unload reserved {resource} handle (slot={slot}, generation={generation})"
            ),
            Self::UnknownTicket { ticket } => write!(f, "unknown load ticket ({ticket})"),
            Self::CancelRejected { ticket, reason } => {
                write!(f, "cannot cancel load ticket ({ticket}): {reason}")
            }
            Self::ManifestParse { path, message } => {
                write!(
                    f,
                    "manifest parse error for '{}': {message}",
                    path.display()
                )
            }
            Self::Cache(msg) => write!(f, "cache operation failed: {msg}"),
            Self::Sync(msg) => write!(f, "asset synchronization failed: {msg}"),
            Self::Unsupported(msg) => write!(f, "{msg}"),
            Self::Internal(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for AssetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<crate::data::assimp_util::AssimpImportError> for AssetError {
    fn from(err: crate::data::assimp_util::AssimpImportError) -> Self {
        use crate::data::assimp_util::AssimpImportError;
        match &err {
            AssimpImportError::InvalidPath(path) => AssetError::Load {
                path: Some(std::path::PathBuf::from(path)),
                message: err.to_string(),
            },
            AssimpImportError::SceneLoadFailed { path, .. } => AssetError::Load {
                path: Some(std::path::PathBuf::from(path)),
                message: err.to_string(),
            },
            AssimpImportError::TextureDecode { texture_ref, .. } => AssetError::Decode {
                path: std::path::PathBuf::from(texture_ref),
                message: err.to_string(),
            },
            AssimpImportError::Internal(msg) if msg.contains("lock poisoned") => {
                AssetError::Sync(err.to_string())
            }
            _ => AssetError::Load {
                path: None,
                message: err.to_string(),
            },
        }
    }
}

#[derive(Debug)]
pub enum HookError {
    Unsupported(String),
    Registration(String),
    Invocation(String),
    /// A pre-frame hook failure observed before rendering began.
    /// Frame number is preserved for diagnostics.
    PreFrameFailure {
        frame: u64,
        message: String,
    },
    /// A post-frame hook failure observed after rendering completed.
    PostFrameFailure {
        frame: u64,
        message: String,
    },
}

impl Display for HookError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "{msg}"),
            Self::Registration(msg) => write!(f, "{msg}"),
            Self::Invocation(msg) => write!(f, "{msg}"),
            Self::PreFrameFailure { frame, message } => {
                write!(f, "pre-frame hook failed at frame {frame}: {message}")
            }
            Self::PostFrameFailure { frame, message } => {
                write!(f, "post-frame hook failed at frame {frame}: {message}")
            }
        }
    }
}

impl Error for HookError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

/// Frame-scoped hook failure entry suitable for structured diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HookFailureEntry {
    pub frame_index: u64,
    pub stage: HookFailureStage,
    pub message: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HookFailureStage {
    PreRender,
    PostRender,
}

/// Per-frame safe-facade hook diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HookReport {
    frame_index: u64,
    failures: Vec<HookFailureEntry>,
}

impl HookReport {
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            failures: Vec::new(),
        }
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn failures(&self) -> &[HookFailureEntry] {
        &self.failures
    }

    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    pub(crate) fn push_failure(&mut self, failure: HookFailureEntry) {
        self.failures.push(failure);
    }
}

impl HookFailureEntry {
    pub fn pre_render(frame_index: u64, message: impl Into<String>) -> Self {
        Self {
            frame_index,
            stage: HookFailureStage::PreRender,
            message: message.into(),
        }
    }

    pub fn post_render(frame_index: u64, message: impl Into<String>) -> Self {
        Self {
            frame_index,
            stage: HookFailureStage::PostRender,
            message: message.into(),
        }
    }
}

pub(crate) fn map_init_err(err: impl Into<String>) -> RendererError {
    RendererInitError::Vulkan(err.into()).into()
}

pub(crate) fn map_frame_input_err(err: impl Into<String>) -> RendererError {
    RendererFrameError::Input(err.into()).into()
}

pub(crate) fn map_frame_render_err(err: impl Into<String>) -> RendererError {
    RendererFrameError::Render(err.into()).into()
}

// ── Animation errors ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AnimationError {
    InvalidClip(String),
    InvalidChannel(String),
    InvalidSampler(String),
    InvalidTarget(String),
    InvalidKeyframe(String),
    StaleTarget(crate::scene::SceneNodeId),
    InvalidDuration(String),
    InvalidTimestamp(String),
    NonFiniteOutput(String),
    CardinalityMismatch(String),
}

impl Display for AnimationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidClip(msg) => write!(f, "invalid animation clip: {msg}"),
            Self::InvalidChannel(msg) => write!(f, "invalid animation channel: {msg}"),
            Self::InvalidSampler(msg) => write!(f, "invalid animation sampler: {msg}"),
            Self::InvalidTarget(msg) => write!(f, "invalid animation target: {msg}"),
            Self::InvalidKeyframe(msg) => write!(f, "invalid keyframe: {msg}"),
            Self::StaleTarget(id) => write!(
                f,
                "stale animation target (slot={}, generation={})",
                id.slot, id.generation
            ),
            Self::InvalidDuration(msg) => write!(f, "invalid animation duration: {msg}"),
            Self::InvalidTimestamp(msg) => write!(f, "invalid animation timestamp: {msg}"),
            Self::NonFiniteOutput(msg) => write!(f, "non-finite animation output: {msg}"),
            Self::CardinalityMismatch(msg) => write!(f, "animation cardinality mismatch: {msg}"),
        }
    }
}

impl Error for AnimationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

// ── Command errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum CommandError {
    NothingToUndo,
    NothingToRedo,
    CommandExecutionFailed(String),
    UndoFailed(String),
    RedoFailed(String),
}

impl Display for CommandError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NothingToUndo => write!(f, "nothing to undo"),
            Self::NothingToRedo => write!(f, "nothing to redo"),
            Self::CommandExecutionFailed(msg) => write!(f, "command execution failed: {msg}"),
            Self::UndoFailed(msg) => write!(f, "undo failed: {msg}"),
            Self::RedoFailed(msg) => write!(f, "redo failed: {msg}"),
        }
    }
}

impl Error for CommandError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
