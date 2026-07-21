//! Configuration types with generation/presentation separation.
//!
//! `NormalizedConfig` captures every input that affects generation output.
//! `PresentationConfig` captures runtime-only options (headless, capture, env).

use std::path::PathBuf;

/// Generation-affecting configuration.
///
/// All fields in this struct contribute to deterministic output. Two runs with
/// the same `NormalizedConfig` and the same RNG seed must produce byte-identical
/// results. Never add presentation-only fields (e.g. `--headless`, `--capture_dir`)
/// to this struct.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedConfig {
    /// RNG seed for deterministic generation.
    pub seed: u64,
    /// Cubic lattice resolution. Must be one of 64, 96, or 128.
    pub resolution: u32,
    /// Thickness of the solid shell around the cave boundary (voxels).
    pub shell_thickness: u32,
    /// Maximum point lights allowed in the generated scene.
    pub light_budget: u32,
}

impl NormalizedConfig {
    /// Canonical byte representation for deterministic hashing.
    #[allow(dead_code)]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&self.resolution.to_be_bytes());
        bytes.extend_from_slice(&self.shell_thickness.to_be_bytes());
        bytes.extend_from_slice(&self.light_budget.to_be_bytes());
        bytes
    }
}

/// Presentation-only configuration.
///
/// These options affect how the result is displayed or captured, but never
/// the generated content itself.
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationConfig {
    /// Run headless (no window).
    pub headless: bool,
    /// Output directory for frame captures.
    pub capture_dir: Option<PathBuf>,
    /// Environment map path for IBL.
    pub env_path: Option<PathBuf>,
}
