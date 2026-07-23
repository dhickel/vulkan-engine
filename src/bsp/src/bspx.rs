//! BSPX extension lump directory parsing.
//!
//! BSPX lumps are appended after the standard 15 lumps. The BSPX directory
//! sits at the end of the file, identified by a 4-byte magic `BSPX` followed
//! by entries of (name: [u8; 24], offset: u32, size: u32).

use crate::decode;
use crate::diagnostic::{BspReport, DiagnosticCode, SourceSpan};

/// BSPX directory magic.
pub const BSPX_MAGIC: [u8; 4] = *b"BSPX";

/// Maximum number of BSPX entries.
pub const MAX_BSPX_ENTRIES: u32 = 64;

/// A single BSPX extension lump entry.
#[derive(Debug, Clone)]
pub struct BspxEntry {
    /// Normalized lump name (ASCII, trimmed, uppercased for comparison).
    pub name: String,
    /// Byte offset within the file.
    pub offset: u32,
    /// Byte size.
    pub size: u32,
}

/// Parsed BSPX directory.
#[derive(Debug, Clone)]
pub struct BspxDirectory {
    pub entries: Vec<BspxEntry>,
    /// File offset where the BSPX directory itself starts.
    pub directory_offset: usize,
}

/// Known BSPX extension names.
pub mod known_names {
    pub const RGBLIGHTING: &str = "RGBLIGHTING";
}

/// Try to locate a BSPX directory at the end of `data`.
///
/// The BSPX directory is at the end of the file, before the 4-byte `BSPX`
/// magic. The last 4 bytes of the directory are the magic. Before that is
/// the entry count (u32 LE), then the entries.
///
/// Returns `None` if no BSPX directory is present (not an error).
/// Returns `Err` if the BSPX magic is present but the directory is malformed.
pub fn discover_bspx(data: &[u8]) -> Result<Option<BspxDirectory>, BspReport> {
    if data.len() < 8 {
        return Ok(None);
    }

    // Check last 4 bytes for BSPX magic
    let magic_pos = data.len() - 4;
    if data[magic_pos..] != BSPX_MAGIC {
        return Ok(None);
    }

    // Read entry count (u32 LE) before the magic
    if data.len() < 12 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "BSPX directory too small for entry count",
        ));
    }
    let count_offset = data.len() - 8;
    let entry_count = decode::read_u32_le(data, count_offset)?;

    if entry_count > MAX_BSPX_ENTRIES {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "BSPX entry count {} exceeds max {}",
                entry_count, MAX_BSPX_ENTRIES
            ),
        ));
    }

    // Each entry is 32 bytes: name[24] + offset:u32 + size:u32
    let entry_stride: u32 = 32;
    let entries_size = entry_count.checked_mul(entry_stride).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            "BSPX entries size overflow",
        )
    })?;
    let total_dir_size = entries_size
        .checked_add(8) // count (4) + magic (4)
        .ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                "BSPX directory size overflow",
            )
        })?;

    if (total_dir_size as usize) > data.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "BSPX directory size exceeds file length",
        ));
    }

    let entries_start = data.len() - 8 - entries_size as usize;
    let mut entries = Vec::with_capacity(entry_count as usize);
    let mut seen_names: Vec<(String, usize)> = Vec::new();

    for i in 0..entry_count {
        let off = entries_start + (i as usize) * 32;
        let name_bytes = &data[off..off + 24];
        // Find null terminator or take full 24
        let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(24);
        let raw_name = std::str::from_utf8(&name_bytes[..name_len])
            .map_err(|_| {
                BspReport::fatal(
                    DiagnosticCode::StructuralCorruptLump,
                    format!("BSPX entry {} has non-UTF-8 name", i),
                )
            })?
            .trim()
            .to_ascii_uppercase();

        if raw_name.is_empty() {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!("BSPX entry {} has empty name", i),
            ));
        }

        // Check for duplicates without hash-based ordering or output nondeterminism.
        if let Some((_, prev_idx)) = seen_names.iter().find(|(name, _)| name == &raw_name) {
            return Err(BspReport::fatal(
                DiagnosticCode::BspxDuplicateName,
                format!(
                    "duplicate BSPX entry name '{}' at entries {} and {}",
                    raw_name, prev_idx, i
                ),
            )
            .with_span(SourceSpan::BspxLump { name: "" })); // name borrowed
        }
        seen_names.push((raw_name.clone(), i as usize));

        let entry_offset = decode::read_u32_le(data, off + 24)?;
        let entry_size = decode::read_u32_le(data, off + 28)?;

        // Validate that entry data lies within the file and before the BSPX directory
        let data_end = entry_offset.checked_add(entry_size).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("BSPX entry '{}' offset+size overflow", raw_name),
            )
        })?;

        if (data_end as usize) > entries_start {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "BSPX entry '{}' range [{}, {}) overlaps BSPX directory (starts at {})",
                    raw_name, entry_offset, data_end, entries_start
                ),
            ));
        }

        entries.push(BspxEntry {
            name: raw_name,
            offset: entry_offset,
            size: entry_size,
        });
    }

    Ok(Some(BspxDirectory {
        entries,
        directory_offset: count_offset,
    }))
}

/// Read the raw bytes for a named BSPX extension lump.
pub fn read_bspx_lump<'a>(data: &'a [u8], dir: &BspxDirectory, name: &str) -> Option<&'a [u8]> {
    let name_upper = name.to_ascii_uppercase();
    for entry in &dir.entries {
        if entry.name == name_upper {
            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            return data.get(start..end);
        }
    }
    None
}

/// Approved BSPX extensions that the parser accepts.
pub fn is_approved_bspx(name: &str) -> bool {
    matches!(name, "RGBLIGHTING")
}

/// Validate BSPX entries: reject unapproved extensions per policy,
/// and ensure BSPX lumps don't overlap standard lump ranges.
pub fn validate_bspx_entries(
    dir: &BspxDirectory,
    strict: bool,
    standard_lump_end: usize,
) -> Vec<BspReport> {
    let mut reports = Vec::new();

    for entry in &dir.entries {
        // Check if entry overlaps standard lump region
        if (entry.offset as usize) < standard_lump_end {
            reports.push(
                BspReport::fatal(
                    DiagnosticCode::BspxLumpOverlap,
                    format!(
                        "BSPX entry '{}' at offset {} overlaps standard lump region (ends at {})",
                        entry.name, entry.offset, standard_lump_end
                    ),
                )
                .with_span(SourceSpan::BspxLump { name: "" }),
            );
        }

        // Report unapproved extensions
        if !is_approved_bspx(&entry.name) {
            let code = DiagnosticCode::UnsupportedExtension;
            let _severity = code.severity(strict);
            reports.push(BspReport::new(
                code,
                strict,
                format!("unknown BSPX extension: '{}'", entry.name),
            ));
        }
    }

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bspx_dir(entries: &[(&str, u32, u32)]) -> Vec<u8> {
        // Build BSPX directory at end of data
        let mut data = vec![0u8; 200]; // dummy file data
        let _dir_start = data.len();

        // Entries
        for &(name, offset, size) in entries {
            let mut name_bytes = [0u8; 24];
            let name_u = name.to_ascii_uppercase();
            let copy_len = name_u.len().min(24);
            name_bytes[..copy_len].copy_from_slice(&name_u.as_bytes()[..copy_len]);
            data.extend_from_slice(&name_bytes);
            data.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(&size.to_le_bytes());
        }

        // Entry count + magic
        data.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        data.extend_from_slice(&BSPX_MAGIC);

        data
    }

    #[test]
    fn no_bspx_returns_none() {
        let data = vec![0u8; 100];
        assert!(discover_bspx(&data).unwrap().is_none());
    }

    #[test]
    fn valid_bspx_rgblighting() {
        let data = make_bspx_dir(&[("RGBLIGHTING", 124, 64)]);
        let dir = discover_bspx(&data).unwrap().unwrap();
        assert_eq!(dir.entries.len(), 1);
        assert_eq!(dir.entries[0].name, "RGBLIGHTING");
        assert_eq!(dir.entries[0].offset, 124);
        assert_eq!(dir.entries[0].size, 64);
    }

    #[test]
    fn duplicate_bspx_rejected() {
        let data = make_bspx_dir(&[("RGBLIGHTING", 124, 64), ("RGBLIGHTING", 200, 64)]);
        let r = discover_bspx(&data);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::BspxDuplicateName);
    }

    #[test]
    fn unknown_bspx_diagnosed() {
        let data = make_bspx_dir(&[("UNKNOWNSTUFF", 124, 64)]);
        let dir = discover_bspx(&data).unwrap().unwrap();
        let reports = validate_bspx_entries(&dir, false, 124);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].code, DiagnosticCode::UnsupportedExtension);
    }

    #[test]
    fn bspx_overlap_standard_rejected() {
        // Construct a file with a BSPX entry whose data offset lies within the
        // standard lump region. discover_bspx succeeds; validate_bspx_entries catches it.
        let mut data = vec![0u8; 200]; // standard lump region dummy fill

        // BSPX directory: entry + count + magic
        let mut name_bytes = [0u8; 24];
        name_bytes[..11].copy_from_slice(b"RGBLIGHTING");
        data.extend_from_slice(&name_bytes);
        data.extend_from_slice(&100u32.to_le_bytes()); // offset = 100 (inside standard region)
        data.extend_from_slice(&64u32.to_le_bytes()); // size = 64

        // Entry count + BSPX magic
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&BSPX_MAGIC);

        let dir = discover_bspx(&data).unwrap().unwrap();
        // Standard lump region ends at 200; BSPX entry at offset 100 is within it
        let reports = validate_bspx_entries(&dir, false, 200);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].code, DiagnosticCode::BspxLumpOverlap);
    }
}
