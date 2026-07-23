//! Profile detection: BSP29 vs BSP2 magic/version, supported extension
//! combinations, and stable unsupported-dialect diagnostics.

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Recognized BSP profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BspProfile {
    /// Standard Quake 1 BSP29.
    Bsp29,
    /// ericw-tools BSP2 with extended field widths.
    Bsp2,
}

impl BspProfile {
    /// Human-readable profile tag.
    pub fn tag(self) -> &'static str {
        match self {
            BspProfile::Bsp29 => "bsp29",
            BspProfile::Bsp2 => "bsp2",
        }
    }

    /// Canonical profile name for cache identity.
    pub fn canonical_name(self) -> &'static str {
        "q1-portable-ericw"
    }

    /// Whether the profile uses 32-bit indices for vertices/edges/faces/markfaces.
    pub fn uses_32bit_indices(self) -> bool {
        match self {
            BspProfile::Bsp29 => false,
            BspProfile::Bsp2 => true,
        }
    }

    /// Maximum vertices for this profile.
    pub fn max_vertices(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_VERTICES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum edges for this profile.
    pub fn max_edges(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_EDGES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum faces for this profile.
    pub fn max_faces(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_FACES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum nodes for this profile.
    pub fn max_nodes(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_NODES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum leaves for this profile.
    pub fn max_leaves(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_LEAVES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum clipnodes for this profile.
    pub fn max_clipnodes(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_CLIPNODES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum models for this profile.
    pub fn max_models(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_MODELS_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum markfaces for this profile.
    pub fn max_markfaces(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_MARKFACES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }

    /// Maximum surfedges for this profile.
    pub fn max_surfedges(self) -> u32 {
        match self {
            BspProfile::Bsp29 => super::limits::MAX_SURFEDGES_BSP29,
            BspProfile::Bsp2 => super::limits::MAX_ELEMENTS_BSP2,
        }
    }
}

/// BSP29 magic value as a little-endian u32: 29.
pub const BSP29_MAGIC: u32 = 29;
/// BSP2 magic bytes: `BSP2`.
pub const BSP2_MAGIC: [u8; 4] = *b"BSP2";

// Known unsupported magic values for diagnostic purposes.
const HL_BSP30_MAGIC: u32 = 30;
const Q2_BSP38_MAGIC: [u8; 4] = *b"2PSB";
const Q3_IBSP46_MAGIC: [u8; 4] = *b"IBSP";

/// BSP header size: 4 bytes version/magic + 15 lumps * 8 bytes = 124.
pub const BSP_HEADER_SIZE: usize = 124;
/// Number of standard lumps.
pub const STANDARD_LUMP_COUNT: usize = 15;

/// Detect the BSP profile from the first 4 bytes of the file.
///
/// Returns `Ok(BspProfile)` for recognised profiles, or `Err(BspReport)` with
/// `UnsupportedDialect` for unknown magic.
pub fn detect_profile(first_four_bytes: &[u8]) -> Result<BspProfile, BspReport> {
    if first_four_bytes.len() < 4 {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            format!(
                "file too small for magic detection: {} bytes",
                first_four_bytes.len()
            ),
        ));
    }

    // Check BSP29: first 4 bytes as i32 LE == 29
    let magic_u32 = u32::from_le_bytes([
        first_four_bytes[0],
        first_four_bytes[1],
        first_four_bytes[2],
        first_four_bytes[3],
    ]);
    if magic_u32 == BSP29_MAGIC {
        return Ok(BspProfile::Bsp29);
    }

    // Check BSP2: first 4 bytes are "BSP2"
    if first_four_bytes == BSP2_MAGIC {
        return Ok(BspProfile::Bsp2);
    }

    // Known unsupported dialects — provide specific messages
    if magic_u32 == HL_BSP30_MAGIC {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            "Half-Life BSP30 is not supported".to_string(),
        ));
    }
    if first_four_bytes == Q2_BSP38_MAGIC {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            "Quake 2 BSP38 is not supported".to_string(),
        ));
    }
    if first_four_bytes == Q3_IBSP46_MAGIC {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            "Quake 3 IBSP BSP46 is not supported".to_string(),
        ));
    }

    // Check for other version numbers that look like BSP30+
    if magic_u32 > 29 && magic_u32 < 256 {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            format!("unsupported BSP version {}", magic_u32),
        ));
    }

    Err(BspReport::fatal(
        DiagnosticCode::UnsupportedDialect,
        format!(
            "unrecognized magic bytes: {:02x} {:02x} {:02x} {:02x}",
            first_four_bytes[0], first_four_bytes[1], first_four_bytes[2], first_four_bytes[3]
        ),
    ))
}

/// Validate that the file is large enough to contain the BSP header.
pub fn validate_header_size(file_len: usize) -> Result<(), BspReport> {
    if file_len < BSP_HEADER_SIZE {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "file too small for BSP header: {} bytes (need {})",
                file_len, BSP_HEADER_SIZE
            ),
        ));
    }
    Ok(())
}

/// Validate that approved extension combinations are valid.
/// Returns diagnostics for unsupported or ambiguous extensions.
pub fn validate_extensions(has_bspx: bool, has_lit: bool, _strict: bool) -> Vec<BspReport> {
    let reports = Vec::new();
    // Both BSPX and .lit colored light are valid (BSPX takes precedence later)
    // No inherent conflict in having both present; the conflict arises if both
    // provide colored light data for the same face. That's handled in companions.rs.
    let _ = (has_bspx, has_lit);
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bsp29() {
        let magic = 29u32.to_le_bytes();
        assert_eq!(detect_profile(&magic).unwrap(), BspProfile::Bsp29);
    }

    #[test]
    fn detect_bsp2() {
        assert_eq!(detect_profile(b"BSP2").unwrap(), BspProfile::Bsp2);
    }

    #[test]
    fn detect_half_life_rejected() {
        let magic = 30u32.to_le_bytes();
        let r = detect_profile(&magic);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn detect_quake2_rejected() {
        let r = detect_profile(b"2PSB");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn detect_quake3_rejected() {
        let r = detect_profile(b"IBSP");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn detect_unknown_rejected() {
        let r = detect_profile(b"XXXX");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn detect_too_small() {
        let r = detect_profile(&[0x1D]);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn profile_limits_bsp29() {
        assert_eq!(BspProfile::Bsp29.max_vertices(), 65_535);
        assert_eq!(BspProfile::Bsp29.max_models(), 256);
    }

    #[test]
    fn profile_uses_32bit() {
        assert!(!BspProfile::Bsp29.uses_32bit_indices());
        assert!(BspProfile::Bsp2.uses_32bit_indices());
    }
}
