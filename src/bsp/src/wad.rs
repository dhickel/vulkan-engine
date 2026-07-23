//! WAD2 texture archive parsing.
//!
//! WAD2 is a simple archive format: 12-byte header (magic + num_entries + dir_offset),
//! followed by lump data and a directory of entries.

use crate::decode;
use crate::diagnostic::{BspReport, DiagnosticCode};
use crate::limits;

/// WAD2 magic bytes.
pub const WAD2_MAGIC: [u8; 4] = *b"WAD2";
/// WAD3 magic bytes (Half-Life variant — explicitly unsupported).
pub const WAD3_MAGIC: [u8; 4] = *b"WAD3";

/// Maximum WAD entry name length.
pub const MAX_WAD_NAME_LEN: usize = 16;
/// Maximum miptex name length.
pub const MAX_MIPTEX_NAME_LEN: usize = 16;

/// A single WAD2 directory entry.
#[derive(Debug, Clone)]
pub struct WadEntry {
    /// Sanitized texture name (ASCII, null-stripped).
    pub name: String,
    /// Offset in the WAD file to the miptex data.
    pub offset: u32,
    /// Disk size (compressed, typically same as size in WAD2).
    pub disk_size: u32,
    /// Uncompressed size.
    pub size: u32,
    /// Type byte (0x44 = miptex in WAD2).
    pub entry_type: u8,
    /// Compression flag (0 = none).
    pub compression: u8,
}

/// Parsed WAD2 file header and directory.
#[derive(Debug, Clone)]
pub struct WadArchive {
    pub entries: Vec<WadEntry>,
    /// Raw bytes of the entire WAD file (for lump extraction).
    pub data: Vec<u8>,
}

/// Parse a WAD2 archive from raw bytes.
pub fn parse_wad(data: Vec<u8>) -> Result<WadArchive, BspReport> {
    if data.len() < 12 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "WAD file too small for header",
        ));
    }

    // Check magic
    let magic = &data[0..4];
    if magic == WAD3_MAGIC {
        return Err(BspReport::fatal(
            DiagnosticCode::UnsupportedDialect,
            "WAD3 (Half-Life) texture archives are not supported",
        ));
    }
    if magic != WAD2_MAGIC {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "invalid WAD magic: {:02x} {:02x} {:02x} {:02x}",
                magic[0], magic[1], magic[2], magic[3]
            ),
        ));
    }

    let num_entries = decode::read_u32_le(&data, 4)?;
    let dir_offset = decode::read_i32_le(&data, 8)?;

    if num_entries > limits::MAX_WAD_ENTRY_COUNT {
        return Err(BspReport::fatal(
            DiagnosticCode::WadEntryCountExceeded,
            format!(
                "WAD entry count {} exceeds max {}",
                num_entries,
                limits::MAX_WAD_ENTRY_COUNT
            ),
        ));
    }

    if dir_offset < 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "WAD directory offset is negative",
        ));
    }
    let dir_offset = dir_offset as u32;

    // Validate directory is within file bounds
    let dir_size = num_entries
        .checked_mul(32) // each entry is 32 bytes
        .ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                "WAD directory size overflow",
            )
        })?;
    let dir_end = dir_offset.checked_add(dir_size).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            "WAD directory end overflow",
        )
    })?;
    if dir_end as usize > data.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "WAD directory [{}, {}) exceeds file length {}",
                dir_offset,
                dir_end,
                data.len()
            ),
        ));
    }

    let mut entries = Vec::with_capacity(num_entries as usize);
    for i in 0..num_entries {
        let off = dir_offset as usize + (i as usize) * 32;
        let entry_offset = decode::read_u32_le(&data, off)?;
        let disk_size = decode::read_u32_le(&data, off + 4)?;
        let size = decode::read_u32_le(&data, off + 8)?;
        let entry_type = decode::read_u8(&data, off + 12)?;
        let compression = decode::read_u8(&data, off + 13)?;

        // Read name: 16 bytes, null-terminated ASCII
        let name_bytes = &data[off + 16..off + 32];
        let name_len = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_WAD_NAME_LEN);
        let raw_name = std::str::from_utf8(&name_bytes[..name_len]).map_err(|_| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!("WAD entry {} has non-UTF-8 name", i),
            )
        })?;
        let name = sanitize_basename(raw_name);
        if name.is_empty() {
            return Err(BspReport::fatal(
                DiagnosticCode::SecurityPathTraversal,
                format!("WAD entry {} has unsafe texture name '{}'", i, raw_name),
            ));
        }

        // Validate entry data within file
        let entry_end = entry_offset.checked_add(disk_size).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("WAD entry '{}' offset+size overflow", name),
            )
        })?;
        if entry_end as usize > data.len() {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "WAD entry '{}' range [{}, {}) exceeds file length {}",
                    name,
                    entry_offset,
                    entry_end,
                    data.len()
                ),
            ));
        }

        entries.push(WadEntry {
            name,
            offset: entry_offset,
            disk_size,
            size,
            entry_type,
            compression,
        });
    }

    Ok(WadArchive { entries, data })
}

/// Sanitize a texture basename: strip paths, normalize, reject unsafe chars.
pub fn sanitize_basename(name: &str) -> String {
    // Reject traversal attempts before any processing
    if name.contains("..") {
        return String::new();
    }

    // Take only the basename part (after last / or \)
    let basename = name
        .rsplit(|c: char| c == '/' || c == '\\')
        .next()
        .unwrap_or(name);

    // Only allow printable ASCII (32-126), strip everything else
    let sanitized: String = basename
        .chars()
        .filter(|&c| c.is_ascii_graphic() || c == ' ')
        .take(MAX_WAD_NAME_LEN)
        .collect();

    // Reject names that try path traversal
    if sanitized.contains("..") || sanitized.starts_with('/') || sanitized.starts_with('\\') {
        return String::new();
    }

    sanitized.trim().to_string()
}

/// Check a path for traversal attempts.
pub fn is_safe_path_component(component: &str) -> bool {
    if component.is_empty() {
        return false;
    }
    if component == "." || component == ".." {
        return false;
    }
    if component.contains('/') || component.contains('\\') || component.contains('\0') {
        return false;
    }
    // Must be valid UTF-8 and only contain printable/safe chars
    component
        .chars()
        .all(|c| c.is_ascii_graphic() || c == ' ' || c == '-' || c == '_' || c == '.')
}

/// Extract the raw miptex lump data for a named entry.
pub fn read_wad_lump<'a>(archive: &'a WadArchive, name: &str) -> Option<&'a [u8]> {
    // Case-sensitive lookup
    for entry in &archive.entries {
        if entry.name == name {
            let start = entry.offset as usize;
            let end = start + entry.disk_size as usize;
            return archive.data.get(start..end);
        }
    }
    None
}

/// Decoded miptex mip-0 pixels: RGBA albedo + separate fullbright mask.
#[derive(Debug, Clone)]
pub struct MiptexPixels {
    /// Raw palette indices for mip level 0 (width × height bytes).
    pub palette_indices: Vec<u8>,
    /// RGBA8 albedo pixels (width × height × 4 bytes).
    pub albedo: Vec<u8>,
    /// Fullbright emissive mask (width × height bytes, 0 = lit, 255 = fullbright).
    pub fullbright_mask: Vec<u8>,
    /// Texture width.
    pub width: u32,
    /// Texture height.
    pub height: u32,
}

impl MiptexPixels {
    /// Number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Decode mip-0 palette indices from a miptex entry into RGBA albedo and fullbright mask.
///
/// `palette` is 256 RGB triples. `fullbright_start..=fullbright_end` defines the
/// emission range. Palette index 255 is NOT globally transparent — alpha is determined
/// by surface classification, not pixel data.
///
/// Returns `MiptexPixels` if the entry is valid, or a `MiptexCorrupt` diagnostic.
pub fn decode_miptex_pixels(
    data: &[u8],
    palette: &[[u8; 3]; 256],
    fullbright_start: u8,
    fullbright_end: u8,
) -> Result<MiptexPixels, BspReport> {
    let info = parse_miptex_header(data)?;
    let w = info.width as usize;
    let h = info.height as usize;
    let pixel_count = w.checked_mul(h).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::MiptexCorrupt,
            format!("miptex '{}' dimensions {}x{} overflow", info.name, w, h),
        )
    })?;

    // Validate texture dimensions
    crate::resources::validate_texture_dimension(info.width, &info.name)?;
    crate::resources::validate_texture_dimension(info.height, &info.name)?;

    // Mip 0 offset
    let mip0_off = info.mip_offsets[0] as usize;
    let mip0_end = mip0_off.checked_add(pixel_count).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::MiptexCorrupt,
            format!("miptex '{}' mip-0 offset+size overflow", info.name),
        )
    })?;
    if mip0_end > data.len() {
        return Err(BspReport::fatal(
            DiagnosticCode::MiptexCorrupt,
            format!(
                "miptex '{}' mip-0 data [{}, {}) exceeds entry length {}",
                info.name, mip0_off, mip0_end, data.len()
            ),
        ));
    }

    let indices = &data[mip0_off..mip0_end];
    let mut albedo = vec![0u8; pixel_count * 4];
    let mut fullbright_mask = vec![0u8; pixel_count];

    for (i, &idx) in indices.iter().enumerate() {
        let rgb = palette[idx as usize];
        let dst = i * 4;
        albedo[dst] = rgb[0];
        albedo[dst + 1] = rgb[1];
        albedo[dst + 2] = rgb[2];
        albedo[dst + 3] = 255; // opaque by default; alpha-mask handled by surface class

        if idx >= fullbright_start && idx <= fullbright_end {
            fullbright_mask[i] = 255;
        }
    }

    Ok(MiptexPixels {
        palette_indices: indices.to_vec(),
        albedo,
        fullbright_mask,
        width: info.width,
        height: info.height,
    })
}

/// Decode mip-0 palette indices from an embedded miptex lump header.
///
/// The embedded miptex lump starts with a count (i32 LE), then an offset table
/// (count × i32 LE), then each miptex entry at its offset.
pub fn read_embedded_miptex_entry(
    lump_data: &[u8],
    entry_index: u32,
) -> Option<&[u8]> {
    if lump_data.len() < 4 {
        return None;
    }
    let count = i32::from_le_bytes([lump_data[0], lump_data[1], lump_data[2], lump_data[3]]);
    if count <= 0 || entry_index >= count as u32 {
        return None;
    }
    let count = count as usize;
    let ei = entry_index as usize;
    let off_table_end = 4 + count * 4;
    if off_table_end > lump_data.len() || (ei + 1) * 4 + 4 > lump_data.len() {
        return None;
    }

    let entry_off = i32::from_le_bytes([
        lump_data[4 + ei * 4],
        lump_data[4 + ei * 4 + 1],
        lump_data[4 + ei * 4 + 2],
        lump_data[4 + ei * 4 + 3],
    ]);
    if entry_off < 0 {
        return None;
    }
    let start = entry_off as usize;
    // Read width/height from header to determine entry size
    if start + 40 > lump_data.len() {
        return None;
    }
    let width = u32::from_le_bytes([
        lump_data[start + 16],
        lump_data[start + 17],
        lump_data[start + 18],
        lump_data[start + 19],
    ]);
    let height = u32::from_le_bytes([
        lump_data[start + 20],
        lump_data[start + 21],
        lump_data[start + 22],
        lump_data[start + 23],
    ]);
    // Mip 0 pixel count + header + optional mips; read enough bytes for at least mip 0
    let pixel_count = (width as usize).checked_mul(height as usize)?;
    let entry_end = start.checked_add(40 + pixel_count)?;
    let entry_end = entry_end.min(lump_data.len());
    Some(&lump_data[start..entry_end])
}

/// Parse a miptex header to get dimensions and mip offsets.
/// Miptex header: name[16] + width:u32 + height:u32 + offsets[4]:u32
pub fn parse_miptex_header(data: &[u8]) -> Result<MiptexInfo, BspReport> {
    if data.len() < 40 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "miptex data too small for header",
        ));
    }

    let name_bytes = &data[0..16];
    let name_len = name_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(MAX_MIPTEX_NAME_LEN);
    let name = std::str::from_utf8(&name_bytes[..name_len]).map_err(|_| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "miptex name is not UTF-8",
        )
    })?;

    let width = decode::read_u32_le(data, 16)?;
    let height = decode::read_u32_le(data, 20)?;
    let mip_offsets: [u32; 4] = [
        decode::read_u32_le(data, 24)?,
        decode::read_u32_le(data, 28)?,
        decode::read_u32_le(data, 32)?,
        decode::read_u32_le(data, 36)?,
    ];

    Ok(MiptexInfo {
        name: name.to_string(),
        width,
        height,
        mip_offsets,
    })
}

#[derive(Debug, Clone)]
pub struct MiptexInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub mip_offsets: [u32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wad2(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut data = vec![0u8; 12]; // header placeholder
        let mut dir = Vec::new();
        let num_entries = entries.len() as u32;

        for (name, payload) in entries {
            let offset = data.len() as u32;
            data.extend_from_slice(payload);

            let mut name_bytes = [0u8; 16];
            let name_ascii = name.as_bytes();
            let copy_len = name_ascii.len().min(16);
            name_bytes[..copy_len].copy_from_slice(&name_ascii[..copy_len]);

            dir.extend_from_slice(&offset.to_le_bytes());
            dir.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // disk_size
            dir.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // size
            dir.push(0x44); // type = miptex
            dir.push(0); // compression
            dir.extend_from_slice(&[0u8; 2]); // padding
            dir.extend_from_slice(&name_bytes);
        }

        let dir_offset = data.len() as u32;
        data.extend_from_slice(&dir);

        // Write header
        data[0..4].copy_from_slice(b"WAD2");
        data[4..8].copy_from_slice(&num_entries.to_le_bytes());
        data[8..12].copy_from_slice(&dir_offset.to_le_bytes());

        data
    }

    #[test]
    fn parse_valid_wad2() {
        let data = make_wad2(&[("TESTTEX", &[0u8; 100])]);
        let archive = parse_wad(data).unwrap();
        assert_eq!(archive.entries.len(), 1);
        assert_eq!(archive.entries[0].name, "TESTTEX");
    }

    #[test]
    fn parse_wad2_empty() {
        let data = make_wad2(&[]);
        let archive = parse_wad(data).unwrap();
        assert!(archive.entries.is_empty());
    }

    #[test]
    fn parse_wad3_rejected() {
        let mut data = make_wad2(&[]);
        data[0..4].copy_from_slice(b"WAD3");
        let r = parse_wad(data);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn sanitize_basename_strips_path() {
        assert_eq!(sanitize_basename("path/to/texture"), "texture");
        assert_eq!(sanitize_basename("texture"), "texture");
    }

    #[test]
    fn sanitize_basename_rejects_traversal() {
        assert_eq!(sanitize_basename("../escape"), "");
        assert_eq!(sanitize_basename("..\\escape"), "");
    }

    #[test]
    fn is_safe_path_component_checks() {
        assert!(is_safe_path_component("texture.wad"));
        assert!(!is_safe_path_component(".."));
        assert!(!is_safe_path_component("../escape"));
        assert!(!is_safe_path_component(""));
    }
}
