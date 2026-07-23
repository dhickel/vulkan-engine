//! Companion file binding: .lit colored light files, palette, and WAD sources.
//!
//! Handles content/profile binding, version/length/hash checks, stale/mismatch
//! diagnostics, and deterministic colored-light precedence (.lit vs BSPX).

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Companion file type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionKind {
    /// `.lit` colored light data.
    Lit,
    /// `.lmp` palette file.
    Palette,
    /// `.wad` texture archive.
    Wad,
}

impl CompanionKind {
    pub fn display_name(self) -> &'static str {
        match self {
            CompanionKind::Lit => ".lit colored light",
            CompanionKind::Palette => "palette",
            CompanionKind::Wad => "WAD texture archive",
        }
    }
}

/// Colored light source precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColoredLightSource {
    /// Explicit package manifest choice (highest precedence).
    PackageOverride = 3,
    /// BSPX RGBLIGHTING lump.
    BspxRgbLighting = 2,
    /// External .lit file.
    LitFile = 1,
    /// Base monochrome lighting (lowest precedence).
    Monochrome = 0,
}

/// .lit file header: QLIT magic + version.
pub const LIT_MAGIC: [u8; 4] = *b"QLIT";
pub const LIT_VERSION: u32 = 1;
pub const LIT_HEADER_SIZE: usize = 8;

/// Validate a .lit file header.
///
/// Returns the expected RGB payload size (total luxels * 3) if the header is valid.
/// The caller must verify this against the base BSP lightmap lump.
pub fn validate_lit_header(data: &[u8], strict: bool) -> Result<u32, BspReport> {
    if data.len() < LIT_HEADER_SIZE {
        return Err(BspReport::new(
            DiagnosticCode::CompanionVersion,
            strict,
            format!(
                ".lit file too small: {} bytes (need {})",
                data.len(),
                LIT_HEADER_SIZE
            ),
        ));
    }

    // Check magic
    if data[0..4] != LIT_MAGIC {
        return Err(BspReport::new(
            DiagnosticCode::CompanionVersion,
            strict,
            ".lit file has invalid magic (expected 'QLIT')",
        ));
    }

    // Check version
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != LIT_VERSION {
        return Err(BspReport::new(
            DiagnosticCode::CompanionVersion,
            strict,
            format!(
                ".lit version {} is not supported (expected {})",
                version, LIT_VERSION
            ),
        ));
    }

    // RGB payload size = total file - header
    let rgb_size = (data.len() - LIT_HEADER_SIZE) as u32;

    Ok(rgb_size)
}

/// Validate that the .lit RGB payload size matches the base BSP lightmap.
/// The .lit contains 3 bytes per luxel (one per valid style across all faces).
pub fn validate_lit_against_lightmap(
    lit_rgb_size: u32,
    lightmap_size: u32,
    strict: bool,
) -> Result<(), BspReport> {
    // Base BSP lightmap has 1 byte per luxel per style.
    // .lit has 3 bytes per luxel per style.
    // So: lightmap_size == lit_rgb_size / 3
    if lightmap_size.checked_mul(3) != Some(lit_rgb_size) {
        return Err(BspReport::new(
            DiagnosticCode::CompanionContentMismatch,
            strict,
            format!(
                ".lit RGB payload size {} does not match lightmap size {} (expected {} = lightmap * 3)",
                lit_rgb_size,
                lightmap_size,
                lightmap_size.saturating_mul(3)
            ),
        ));
    }

    Ok(())
}

/// Validate a palette file: must be exactly 768 bytes.
pub fn validate_palette(data: &[u8], strict: bool) -> Result<(), BspReport> {
    if data.len() != 768 {
        return Err(BspReport::new(
            DiagnosticCode::MissingRequiredPalette,
            strict,
            format!("palette must be 768 bytes, got {}", data.len()),
        ));
    }
    Ok(())
}

/// Resolve the effective colored light source given available data.
/// Returns the source that should be used and any diagnostics.
pub fn resolve_colored_light_source(
    has_bspx_rgb: bool,
    has_lit: bool,
    lit_valid: bool,
    strict: bool,
) -> (ColoredLightSource, Vec<BspReport>) {
    let mut reports = Vec::new();

    let source = if has_bspx_rgb {
        if has_lit && lit_valid {
            // Both BSPX and .lit are valid — BSPX takes precedence,
            // but diagnose the conflict
            reports.push(BspReport::new(
                DiagnosticCode::ColoredLightConflict,
                strict,
                "both BSPX RGBLIGHTING and valid .lit present; using BSPX",
            ));
        }
        ColoredLightSource::BspxRgbLighting
    } else if has_lit && lit_valid {
        ColoredLightSource::LitFile
    } else if has_lit && !lit_valid {
        // .lit exists but is invalid; fall back to monochrome
        reports.push(BspReport::new(
            DiagnosticCode::CompanionContentMismatch,
            strict,
            ".lit file is invalid; using monochrome lighting",
        ));
        ColoredLightSource::Monochrome
    } else {
        ColoredLightSource::Monochrome
    };

    (source, reports)
}

/// Expected content hash wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentHash {
    pub sha256: [u8; 32],
}

impl ContentHash {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut lanes = [
            0xcbf2_9ce4_8422_2325u64,
            0x9e37_79b9_7f4a_7c15u64,
            0x94d0_49bb_1331_11ebu64,
            0x2545_f491_4f6c_dd1du64,
        ];
        for (i, &byte) in bytes.iter().enumerate() {
            let lane = i & 3;
            lanes[lane] ^= byte as u64;
            lanes[lane] = lanes[lane].wrapping_mul(0x100_0000_01b3);
            lanes[lane] ^= (i as u64).rotate_left((lane as u32) + 1);
        }
        let mut arr = [0u8; 32];
        for (i, lane) in lanes.iter().enumerate() {
            arr[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        ContentHash { sha256: arr }
    }

    pub fn matches(&self, other: &ContentHash) -> bool {
        self.sha256 == other.sha256
    }
}

/// Check a companion file's content hash against expected.
pub fn check_companion_hash(
    kind: CompanionKind,
    expected: Option<&ContentHash>,
    actual: &ContentHash,
    strict: bool,
) -> Vec<BspReport> {
    let mut reports = Vec::new();
    if let Some(expected) = expected {
        if !expected.matches(actual) {
            reports.push(BspReport::new(
                DiagnosticCode::StaleCompanion,
                strict,
                format!(
                    "{} content hash mismatch: expected does not match actual",
                    kind.display_name()
                ),
            ));
        }
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lit_header_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(b"QLIT");
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 12]); // payload
        let rgb_size = validate_lit_header(&data, false).unwrap();
        assert_eq!(rgb_size, 12);
    }

    #[test]
    fn lit_header_bad_magic() {
        let data = b"BADD1234extra";
        let r = validate_lit_header(data, false);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::CompanionVersion);
    }

    #[test]
    fn lit_header_bad_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"QLIT");
        data.extend_from_slice(&2u32.to_le_bytes());
        let r = validate_lit_header(&data, false);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::CompanionVersion);
    }

    #[test]
    fn lit_against_lightmap_match() {
        assert!(validate_lit_against_lightmap(300, 100, false).is_ok());
    }

    #[test]
    fn lit_against_lightmap_mismatch() {
        let r = validate_lit_against_lightmap(300, 99, false);
        assert!(r.is_err());
        assert_eq!(
            r.unwrap_err().code,
            DiagnosticCode::CompanionContentMismatch
        );
    }

    #[test]
    fn palette_valid() {
        let data = vec![0u8; 768];
        assert!(validate_palette(&data, false).is_ok());
    }

    #[test]
    fn palette_invalid_size() {
        let data = vec![0u8; 512];
        let r = validate_palette(&data, false);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::MissingRequiredPalette);
    }

    #[test]
    fn colored_light_bspx_precedence() {
        let (source, reports) = resolve_colored_light_source(true, true, true, false);
        assert_eq!(source, ColoredLightSource::BspxRgbLighting);
        assert!(!reports.is_empty()); // conflict diagnosed
    }

    #[test]
    fn colored_light_lit_only() {
        let (source, reports) = resolve_colored_light_source(false, true, true, false);
        assert_eq!(source, ColoredLightSource::LitFile);
        assert!(reports.is_empty());
    }

    #[test]
    fn colored_light_monochrome_fallback() {
        let (source, _reports) = resolve_colored_light_source(false, false, false, false);
        assert_eq!(source, ColoredLightSource::Monochrome);
    }
}
