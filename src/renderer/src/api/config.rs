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

/// Controls how the engine handles `.meta` sidecar manifest files for assets.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AssetManifestMode {
    /// Manifests are ignored even if present.
    Disabled,
    /// Manifests are loaded when found; parse errors log a warning and fall back to defaults.
    BestEffort,
    /// Manifests are required for assets that have them; parse errors fail the load.
    Strict,
}

impl Default for AssetManifestMode {
    fn default() -> Self {
        Self::BestEffort
    }
}

/// Policy configuration for asset loading behavior.
#[derive(Debug, Clone)]
pub struct AssetPolicyConfig {
    pub manifest_mode: AssetManifestMode,
    /// When true, filename patterns (e.g. `_n.`, `_normal.`) can influence sRGB/sampler defaults.
    pub allow_filename_heuristics: bool,
}

impl Default for AssetPolicyConfig {
    fn default() -> Self {
        Self {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
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
            headless: false,
            asset_policy: AssetPolicyConfig::default(),
        }
    }
}
