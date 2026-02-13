use crate::vulkan::vk_render;

/// Startup runtime mode used for controlled render-path validation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DebugRuntimeMode {
    Default,
    TestPbr,
    TestUnlit,
}

impl Default for DebugRuntimeMode {
    fn default() -> Self {
        Self::Default
    }
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

/// Public renderer configuration contract.
#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub app_name: String,
    pub validation_layer: bool,
    pub shader_debug_mode: DebugRuntimeMode,
    pub compile_shaders: bool,
    /// Reserved in v1. `true` currently returns `RendererError::Unsupported`.
    pub headless: bool,
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
            headless: false,
        }
    }
}
