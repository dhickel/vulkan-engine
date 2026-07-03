use crate::vulkan::vk_render;

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
