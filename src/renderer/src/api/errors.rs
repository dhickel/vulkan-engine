use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::scene::scene_world::SceneNodeId;

#[derive(Debug)]
pub enum RendererError {
    Init(RendererInitError),
    Frame(RendererFrameError),
    Scene(SceneError),
    Asset(AssetError),
    Hook(HookError),
    Unsupported(&'static str),
    InvalidState(&'static str),
}

impl Display for RendererError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(err) => write!(f, "renderer init error: {err}"),
            Self::Frame(err) => write!(f, "renderer frame error: {err}"),
            Self::Scene(err) => write!(f, "scene error: {err}"),
            Self::Asset(err) => write!(f, "asset error: {err}"),
            Self::Hook(err) => write!(f, "hook error: {err}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

impl Error for RendererError {}

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

impl Error for RendererInitError {}

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

impl Error for RendererFrameError {}

#[derive(Debug)]
pub enum SceneError {
    InvalidNode(SceneNodeId),
    StaleNode(SceneNodeId),
    CycleDetected,
    InvalidParent(SceneNodeId),
    MergeFailed(String),
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
        }
    }
}

impl Error for SceneError {}

#[derive(Debug)]
pub enum AssetError {
    Load(String),
    Unsupported(String),
    Internal(String),
}

impl Display for AssetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(msg) => write!(f, "{msg}"),
            Self::Unsupported(msg) => write!(f, "{msg}"),
            Self::Internal(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for AssetError {}

#[derive(Debug)]
pub enum HookError {
    Unsupported(String),
    Registration(String),
    Invocation(String),
}

impl Display for HookError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "{msg}"),
            Self::Registration(msg) => write!(f, "{msg}"),
            Self::Invocation(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for HookError {}

pub(crate) fn map_init_err(err: impl Into<String>) -> RendererError {
    RendererInitError::Vulkan(err.into()).into()
}

pub(crate) fn map_frame_input_err(err: impl Into<String>) -> RendererError {
    RendererFrameError::Input(err.into()).into()
}

pub(crate) fn map_frame_resize_err(err: impl Into<String>) -> RendererError {
    RendererFrameError::Resize(err.into()).into()
}

pub(crate) fn map_frame_render_err(err: impl Into<String>) -> RendererError {
    RendererFrameError::Render(err.into()).into()
}

pub(crate) fn map_asset_err(err: impl Into<String>) -> RendererError {
    AssetError::Internal(err.into()).into()
}

pub(crate) fn map_hook_err(err: impl Into<String>) -> RendererError {
    HookError::Invocation(err.into()).into()
}
