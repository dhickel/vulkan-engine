//! Richness V1 theme asset validation and strict package staging.
//!
//! Provides deterministic, fail-closed validation of Richness theme closures:
//! PNG CRC/decode with dimension checks, normal/gloss pixel validation,
//! palette size verification, WAD identity and mip-chain completeness,
//! provenance hash matching, license allowlist enforcement (CC0 only),
//! case-stability checks, and strict package staging with exact identity
//! closure plus required companions.
//!
//! # Design rules
//!
//! - No floating-point arithmetic — all comparisons use integer thresholds.
//! - Fail-closed: any mismatch, corruption, ambiguity, or unexpected file
//!   immediately returns an error.
//! - Symlinks are rejected at every boundary.
//! - Exact-case filename matching is required; case-insensitive fallback is
//!   never enabled for Richness assets.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Frozen package-side view of the nine Richness material roles.  This stays
/// local because `bsp_generator::enhanced_v3::richness` is crate-private until
/// the atomic Richness release phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticRole {
    Wall,
    Floor,
    Ceiling,
    Accent,
    Portal,
    Vertical,
    Cave,
    Prop,
    Emissive,
}

impl SemanticRole {
    const ALL: [Self; 9] = [
        Self::Wall,
        Self::Floor,
        Self::Ceiling,
        Self::Accent,
        Self::Portal,
        Self::Vertical,
        Self::Cave,
        Self::Prop,
        Self::Emissive,
    ];

    const fn identity(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Floor => "floor",
            Self::Ceiling => "ceiling",
            Self::Accent => "accent",
            Self::Portal => "portal",
            Self::Vertical => "vertical",
            Self::Cave => "cave",
            Self::Prop => "prop",
            Self::Emissive => "emissive",
        }
    }
}

/// File-oriented package definition.  The matching `theme.toml` is checked
/// during closure validation so this frozen table cannot drift from assets.
#[derive(Debug, Clone)]
pub struct ThemeDefinition {
    pub dir_name: &'static str,
    pub wad_filename: &'static str,
    pub palette_filename: &'static str,
    pub texture_size: u32,
    pub palette_size: usize,
}

impl ThemeDefinition {
    pub fn all_png_filenames(&self) -> Vec<String> {
        SemanticRole::ALL
            .iter()
            .flat_map(|role| {
                let identity = role.identity();
                [
                    format!("{identity}_basecolor.png"),
                    format!("{identity}_norm.png"),
                    format!("{identity}_gloss.png"),
                ]
            })
            .collect()
    }

    pub fn all_wad_identities(&self) -> Vec<&'static str> {
        SemanticRole::ALL
            .iter()
            .map(|role| role.identity())
            .chain(std::iter::once("skip"))
            .collect()
    }
}

pub const THEME_ANCIENT: ThemeDefinition = ThemeDefinition {
    dir_name: "richness_ancient_v1",
    wad_filename: "richness_ancient_v1.wad",
    palette_filename: "palette.lmp",
    texture_size: 256,
    palette_size: 768,
};
pub const THEME_EGYPTIAN: ThemeDefinition = ThemeDefinition {
    dir_name: "richness_egyptian_v1",
    wad_filename: "richness_egyptian_v1.wad",
    palette_filename: "palette.lmp",
    texture_size: 256,
    palette_size: 768,
};
pub const THEME_BRUTALIST: ThemeDefinition = ThemeDefinition {
    dir_name: "richness_brutalist_v1",
    wad_filename: "richness_brutalist_v1.wad",
    palette_filename: "palette.lmp",
    texture_size: 256,
    palette_size: 768,
};

#[doc(hidden)]
pub fn test_theme_def(name: &'static str) -> ThemeDefinition {
    ThemeDefinition {
        dir_name: name,
        wad_filename: "test.wad",
        palette_filename: "palette.lmp",
        texture_size: 256,
        palette_size: 768,
    }
}

// ── Error type ────────────────────────────────────────────────────────────

/// Errors returned by Richness asset validation and staging.
#[derive(Debug)]
pub enum RichnessAssetError {
    /// I/O failure.
    Io { path: PathBuf, message: String },
    /// PNG structural failure (bad signature, truncated chunk, CRC mismatch).
    PngStructure { path: PathBuf, reason: String },
    /// PNG dimension mismatch.
    PngDimensions {
        path: PathBuf,
        expected: (u32, u32),
        actual: (u32, u32),
    },
    /// Normal map pixel validation failure.
    NormalMapInvalid {
        path: PathBuf,
        reason: String,
        mean_r: u32,
        mean_g: u32,
        mean_b: u32,
    },
    /// Gloss map pixel validation failure.
    GlossMapInvalid {
        path: PathBuf,
        reason: String,
        mean: u32,
    },
    /// Palette size mismatch.
    PaletteSize {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    /// WAD structural failure.
    WadStructure { path: PathBuf, reason: String },
    /// WAD identity is not lowercase ASCII.
    WadIdentityCase { path: PathBuf, identity: String },
    /// WAD missing a required identity.
    WadMissingIdentity { path: PathBuf, identity: String },
    /// WAD has an unexpected (extra) identity.
    WadExtraIdentity { path: PathBuf, identity: String },
    /// WAD miptex has incomplete mip levels.
    WadIncompleteMips {
        path: PathBuf,
        identity: String,
        expected: usize,
        actual: usize,
    },
    /// Provenance hash mismatch.
    ProvenanceHashMismatch {
        path: PathBuf,
        filename: String,
        expected: String,
        actual: String,
    },
    /// License is not CC0.
    LicenseNotCc0 { path: PathBuf, found: String },
    /// Case stability violation — file on disk does not match expected casing.
    CaseMismatch {
        dir: PathBuf,
        expected: String,
        found: Option<String>,
    },
    /// Symlink detected where a regular file was expected.
    SymlinkRejected { path: PathBuf },
    /// Extra file present in closure.
    ExtraFile { dir: PathBuf, filename: String },
    /// Required file missing from closure.
    MissingFile { dir: PathBuf, filename: String },
    /// Ambiguous path (e.g. multiple case variations of the same name).
    AmbiguousPath { dir: PathBuf, filename: String },
    /// Fresh-build comparator failure — output does not match checked-in asset.
    FreshBuildMismatch {
        theme_dir: PathBuf,
        filename: String,
        details: String,
    },
    /// Fresh-build toolchain unavailable.
    FreshBuildUnavailable { theme_dir: PathBuf, reason: String },
    /// Staging destination error.
    Staging { path: PathBuf, message: String },
}

impl std::fmt::Display for RichnessAssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "I/O error at '{}': {message}", path.display()),
            Self::PngStructure { path, reason } => {
                write!(f, "PNG structure error in '{}': {reason}", path.display())
            }
            Self::PngDimensions { path, expected, actual } => write!(
                f,
                "PNG dimension mismatch in '{}': expected {}x{}, got {}x{}",
                path.display(),
                expected.0,
                expected.1,
                actual.0,
                actual.1
            ),
            Self::NormalMapInvalid { path, reason, mean_r, mean_g, mean_b } => write!(
                f,
                "normal map invalid '{}': {reason} (mean R={mean_r} G={mean_g} B={mean_b})",
                path.display()
            ),
            Self::GlossMapInvalid { path, reason, mean } => {
                write!(f, "gloss map invalid '{}': {reason} (mean={mean})", path.display())
            }
            Self::PaletteSize { path, expected, actual } => write!(
                f,
                "palette size mismatch at '{}': expected {expected}, got {actual}",
                path.display()
            ),
            Self::WadStructure { path, reason } => {
                write!(f, "WAD structure error in '{}': {reason}", path.display())
            }
            Self::WadIdentityCase { path, identity } => {
                write!(f, "WAD identity '{identity}' in '{}' is not lowercase ASCII", path.display())
            }
            Self::WadMissingIdentity { path, identity } => {
                write!(f, "WAD '{}' missing required identity '{identity}'", path.display())
            }
            Self::WadExtraIdentity { path, identity } => {
                write!(f, "WAD '{}' has unexpected identity '{identity}'", path.display())
            }
            Self::WadIncompleteMips { path, identity, expected, actual } => write!(
                f,
                "WAD miptex '{identity}' in '{}' has {actual} mip levels, expected {expected}",
                path.display()
            ),
            Self::ProvenanceHashMismatch { path, filename, expected, actual } => write!(
                f,
                "provenance hash mismatch for '{filename}' in '{}': expected {expected}, got {actual}",
                path.display()
            ),
            Self::LicenseNotCc0 { path, found } => {
                write!(f, "license at '{}' is not CC0: found '{found}'", path.display())
            }
            Self::CaseMismatch { dir, expected, found } => {
                write!(
                    f,
                    "case mismatch in '{}': expected '{expected}', found {:?}",
                    dir.display(),
                    found
                )
            }
            Self::SymlinkRejected { path } => {
                write!(f, "symlink rejected at '{}'", path.display())
            }
            Self::ExtraFile { dir, filename } => {
                write!(f, "extra file '{filename}' in '{}'", dir.display())
            }
            Self::MissingFile { dir, filename } => {
                write!(f, "missing file '{filename}' in '{}'", dir.display())
            }
            Self::AmbiguousPath { dir, filename } => {
                write!(f, "ambiguous path '{filename}' in '{}'", dir.display())
            }
            Self::FreshBuildMismatch { theme_dir, filename, details } => write!(
                f,
                "fresh-build mismatch in '{}': file '{}' — {details}",
                theme_dir.display(),
                filename
            ),
            Self::FreshBuildUnavailable { theme_dir, reason } => write!(
                f,
                "fresh-build unavailable for '{}': {reason}",
                theme_dir.display()
            ),
            Self::Staging { path, message } => {
                write!(f, "staging error at '{}': {message}", path.display())
            }
        }
    }
}

impl std::error::Error for RichnessAssetError {}

// ── Utility: read file bytes ──────────────────────────────────────────────

fn read_file(path: &Path) -> Result<Vec<u8>, RichnessAssetError> {
    std::fs::read(path).map_err(|e| RichnessAssetError::Io {
        path: path.to_path_buf(),
        message: format!("read: {e}"),
    })
}

/// Verify a path is a regular file (not a symlink, not a directory).
fn require_regular_file(path: &Path) -> Result<(), RichnessAssetError> {
    let meta = path
        .symlink_metadata()
        .map_err(|e| RichnessAssetError::Io {
            path: path.to_path_buf(),
            message: format!("metadata: {e}"),
        })?;
    if meta.file_type().is_symlink() {
        return Err(RichnessAssetError::SymlinkRejected {
            path: path.to_path_buf(),
        });
    }
    if !meta.is_file() {
        return Err(RichnessAssetError::Io {
            path: path.to_path_buf(),
            message: "expected a regular file".into(),
        });
    }
    Ok(())
}

/// Verify a path is a directory (not a symlink).
fn require_directory(path: &Path) -> Result<(), RichnessAssetError> {
    let meta = path
        .symlink_metadata()
        .map_err(|e| RichnessAssetError::Io {
            path: path.to_path_buf(),
            message: format!("metadata: {e}"),
        })?;
    if meta.file_type().is_symlink() {
        return Err(RichnessAssetError::SymlinkRejected {
            path: path.to_path_buf(),
        });
    }
    if !meta.is_dir() {
        return Err(RichnessAssetError::Io {
            path: path.to_path_buf(),
            message: "expected a directory".into(),
        });
    }
    Ok(())
}

// ── SHA-256 (self-contained) ──────────────────────────────────────────────

/// Compute SHA-256 digest of bytes, returned as lowercase hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut buf = [0u8; 64];
    let mut buf_len = 0usize;
    let mut total_len = 0u64;

    for &byte in data {
        total_len += 1;
        buf[buf_len] = byte;
        buf_len += 1;
        if buf_len == 64 {
            compress_block(&mut state, &buf);
            buf_len = 0;
        }
    }

    // Padding
    let total_bits = total_len.wrapping_mul(8);
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 56 {
        for i in buf_len..64 {
            buf[i] = 0;
        }
        compress_block(&mut state, &buf);
        buf_len = 0;
    }
    for i in buf_len..56 {
        buf[i] = 0;
    }
    buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
    compress_block(&mut state, &buf);

    let mut hex = String::with_capacity(64);
    for word in &state {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        let base = i * 4;
        w[i] = u32::from_be_bytes([
            block[base],
            block[base + 1],
            block[base + 2],
            block[base + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K256[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

// ── PNG validation ────────────────────────────────────────────────────────

/// Validate PNG structure from raw bytes: signature, CRC for every chunk,
/// dimensions in IHDR, IDAT present, IEND present, no trailing data.
/// Returns (width, height).
pub fn validate_png_crc_and_dimensions_bytes(
    bytes: &[u8],
    expected_size: u32,
) -> Result<(u32, u32), RichnessAssetError> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < SIGNATURE.len() || &bytes[..SIGNATURE.len()] != SIGNATURE {
        return Err(RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: "missing PNG signature".into(),
        });
    }
    let mut offset = SIGNATURE.len();
    let mut dimensions: Option<(u32, u32)> = None;
    let mut saw_idat = false;
    let mut saw_iend = false;

    while offset < bytes.len() {
        let header_end = offset
            .checked_add(8)
            .ok_or_else(|| RichnessAssetError::PngStructure {
                path: PathBuf::new(),
                reason: "chunk header overflow".into(),
            })?;
        if header_end > bytes.len() {
            return Err(RichnessAssetError::PngStructure {
                path: PathBuf::new(),
                reason: "truncated chunk header".into(),
            });
        }
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().map_err(|_| {
            RichnessAssetError::PngStructure {
                path: PathBuf::new(),
                reason: "invalid chunk length".into(),
            }
        })?) as usize;
        let kind = &bytes[offset + 4..header_end];
        let data_start = header_end;
        let data_end =
            data_start
                .checked_add(length)
                .ok_or_else(|| RichnessAssetError::PngStructure {
                    path: PathBuf::new(),
                    reason: "chunk length overflow".into(),
                })?;
        let chunk_end =
            data_end
                .checked_add(4)
                .ok_or_else(|| RichnessAssetError::PngStructure {
                    path: PathBuf::new(),
                    reason: "CRC offset overflow".into(),
                })?;
        if chunk_end > bytes.len() {
            return Err(RichnessAssetError::PngStructure {
                path: PathBuf::new(),
                reason: "truncated chunk data".into(),
            });
        }
        let expected_crc =
            u32::from_be_bytes(bytes[data_end..chunk_end].try_into().map_err(|_| {
                RichnessAssetError::PngStructure {
                    path: PathBuf::new(),
                    reason: "invalid CRC bytes".into(),
                }
            })?);
        let actual_crc = png_crc32(&bytes[offset + 4..data_end]);
        if actual_crc != expected_crc {
            return Err(RichnessAssetError::PngStructure {
                path: PathBuf::new(),
                reason: format!(
                    "CRC mismatch in chunk {:?}: expected {expected_crc:08x}, got {actual_crc:08x}",
                    std::str::from_utf8(kind).unwrap_or("<invalid>")
                ),
            });
        }
        match kind {
            b"IHDR" if dimensions.is_none() && offset == SIGNATURE.len() => {
                if length != 13 {
                    return Err(RichnessAssetError::PngStructure {
                        path: PathBuf::new(),
                        reason: "IHDR must be exactly 13 bytes".into(),
                    });
                }
                let w = u32::from_be_bytes(bytes[data_start..data_start + 4].try_into().unwrap());
                let h =
                    u32::from_be_bytes(bytes[data_start + 4..data_start + 8].try_into().unwrap());
                if w == 0 || h == 0 {
                    return Err(RichnessAssetError::PngStructure {
                        path: PathBuf::new(),
                        reason: "dimensions must be nonzero".into(),
                    });
                }
                let bit_depth = bytes[data_start + 8];
                let color_type = bytes[data_start + 9];
                if !matches!(bit_depth, 1 | 2 | 4 | 8 | 16)
                    || !matches!(color_type, 0 | 2 | 3 | 4 | 6)
                    || bytes[data_start + 10] != 0
                    || bytes[data_start + 11] != 0
                    || bytes[data_start + 12] > 1
                {
                    return Err(RichnessAssetError::PngStructure {
                        path: PathBuf::new(),
                        reason: "unsupported PNG parameters".into(),
                    });
                }
                if w != expected_size || h != expected_size {
                    return Err(RichnessAssetError::PngDimensions {
                        path: PathBuf::new(),
                        expected: (expected_size, expected_size),
                        actual: (w, h),
                    });
                }
                dimensions = Some((w, h));
            }
            b"IDAT" if dimensions.is_some() && !saw_iend => saw_idat = true,
            b"IEND" if dimensions.is_some() && !saw_iend => {
                if length != 0 {
                    return Err(RichnessAssetError::PngStructure {
                        path: PathBuf::new(),
                        reason: "IEND must be empty".into(),
                    });
                }
                saw_iend = true;
                if chunk_end != bytes.len() {
                    return Err(RichnessAssetError::PngStructure {
                        path: PathBuf::new(),
                        reason: "trailing bytes after IEND".into(),
                    });
                }
            }
            b"IHDR" => {
                return Err(RichnessAssetError::PngStructure {
                    path: PathBuf::new(),
                    reason: "IHDR must be the first chunk".into(),
                });
            }
            _ if saw_iend => {
                return Err(RichnessAssetError::PngStructure {
                    path: PathBuf::new(),
                    reason: "chunk after IEND".into(),
                });
            }
            _ => {}
        }
        offset = chunk_end;
    }
    if !saw_idat {
        return Err(RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: "no IDAT chunk".into(),
        });
    }
    if !saw_iend {
        return Err(RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: "no IEND chunk".into(),
        });
    }
    dimensions.ok_or_else(|| RichnessAssetError::PngStructure {
        path: PathBuf::new(),
        reason: "no IHDR chunk".into(),
    })
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Decode a PNG into 8-bit RGB pixel buffer using the `png` crate.
fn decode_png_rgb8(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), RichnessAssetError> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: format!("png decode info: {e}"),
        })?;
    let info = reader.info();
    let width = info.width;
    let height = info.height;

    if info.bit_depth != png::BitDepth::Eight || info.color_type != png::ColorType::Rgb {
        return Err(RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: format!(
                "expected 8-bit RGB, got {:?} {:?}",
                info.bit_depth, info.color_type
            ),
        });
    }

    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 3];
    reader
        .next_frame(&mut pixels)
        .map_err(|e| RichnessAssetError::PngStructure {
            path: PathBuf::new(),
            reason: format!("png decode frame: {e}"),
        })?;
    Ok((width, height, pixels))
}

/// Validate that an RGB pixel buffer is a plausible tangent-space normal map.
///
/// Checks:
/// - Red channel mean is approximately 128 (within ±32).
/// - Green channel mean is approximately 128 (within ±32).
/// - Blue channel mean is > 200.
fn validate_normal_map_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<(), RichnessAssetError> {
    let pixel_count = (width as usize) * (height as usize);
    let mut sum_r: u64 = 0;
    let mut sum_g: u64 = 0;
    let mut sum_b: u64 = 0;

    for i in 0..pixel_count {
        sum_r += pixels[i * 3] as u64;
        sum_g += pixels[i * 3 + 1] as u64;
        sum_b += pixels[i * 3 + 2] as u64;
    }

    let n = pixel_count as u64;
    let mean_r = (sum_r / n) as u32;
    let mean_g = (sum_g / n) as u32;
    let mean_b = (sum_b / n) as u32;

    // R and G should be centred near 128 (common for tangent-space normal maps).
    // Allow ±32 tolerance to handle reasonable surface variation.
    if mean_r < 96 || mean_r > 160 {
        return Err(RichnessAssetError::NormalMapInvalid {
            path: PathBuf::new(),
            reason: format!("R channel mean {mean_r} not near 128 (±32)"),
            mean_r,
            mean_g,
            mean_b,
        });
    }
    if mean_g < 96 || mean_g > 160 {
        return Err(RichnessAssetError::NormalMapInvalid {
            path: PathBuf::new(),
            reason: format!("G channel mean {mean_g} not near 128 (±32)"),
            mean_r,
            mean_g,
            mean_b,
        });
    }
    // Blue channel in tangent-space normals is predominantly >200 (pointing toward viewer).
    if mean_b <= 200 {
        return Err(RichnessAssetError::NormalMapInvalid {
            path: PathBuf::new(),
            reason: format!("B channel mean {mean_b} not > 200"),
            mean_r,
            mean_g,
            mean_b,
        });
    }
    Ok(())
}

/// Validate that an RGB pixel buffer is a plausible gloss/smoothness map.
///
/// Checks:
/// - Mean value is in the valid PBR gloss range 76–128 (the expected range
///   for stone materials in this pipeline).
/// - No individual channel outside 64–140.
fn validate_gloss_map_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<(), RichnessAssetError> {
    let pixel_count = (width as usize) * (height as usize);
    let mut sum: u64 = 0;
    let mut min_val: u32 = 255;
    let mut max_val: u32 = 0;

    for i in 0..pixel_count {
        let r = pixels[i * 3] as u32;
        let g = pixels[i * 3 + 1] as u32;
        let b = pixels[i * 3 + 2] as u32;
        if r != g || r != b {
            return Err(RichnessAssetError::GlossMapInvalid {
                path: PathBuf::new(),
                reason: "gloss map must be grayscale (R=G=B)".into(),
                mean: r,
            });
        }
        sum += r as u64;
        if r < min_val {
            min_val = r;
        }
        if r > max_val {
            max_val = r;
        }
    }

    let mean = (sum / pixel_count as u64) as u32;

    // Valid PBR gloss band for stone: 76–128 (see bsp-theme-texture-pipeline knowledge).
    if mean < 76 || mean > 128 {
        return Err(RichnessAssetError::GlossMapInvalid {
            path: PathBuf::new(),
            reason: format!("mean gloss {mean} outside valid range 76–128"),
            mean,
        });
    }

    // No single channel should be wildly outside the expected range.
    if min_val < 64 || max_val > 140 {
        return Err(RichnessAssetError::GlossMapInvalid {
            path: PathBuf::new(),
            reason: format!("gloss range [{min_val}–{max_val}] outside [64–140]"),
            mean,
        });
    }

    Ok(())
}

/// Full PNG validation: CRC, dimensions, decode, and pixel-level checks
/// where applicable (normal/gloss maps).
pub fn validate_png_asset(
    path: &Path,
    expected_size: u32,
    kind: PngCompanionKind,
) -> Result<(), RichnessAssetError> {
    require_regular_file(path)?;
    let bytes = read_file(path)?;

    let (w, h) =
        validate_png_crc_and_dimensions_bytes(&bytes, expected_size).map_err(|mut e| {
            // Attach path to error
            attach_path(&mut e, path);
            e
        })?;

    // Decode and validate pixels
    let (_w, _h, pixels) = decode_png_rgb8(&bytes).map_err(|mut e| {
        attach_path(&mut e, path);
        e
    })?;

    match kind {
        PngCompanionKind::Basecolor => {
            // Basecolor: no pixel-level constraints beyond valid PNG.
            let _ = (w, h, pixels);
        }
        PngCompanionKind::Normal => {
            validate_normal_map_pixels(&pixels, w, h).map_err(|mut e| {
                attach_path(&mut e, path);
                e
            })?;
        }
        PngCompanionKind::Gloss => {
            validate_gloss_map_pixels(&pixels, w, h).map_err(|mut e| {
                attach_path(&mut e, path);
                e
            })?;
        }
    }
    Ok(())
}

/// Category of PNG companion for pixel validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngCompanionKind {
    Basecolor,
    Normal,
    Gloss,
}

impl PngCompanionKind {
    pub fn from_filename_suffix(filename: &str) -> Option<Self> {
        if filename.ends_with("_basecolor.png") {
            Some(Self::Basecolor)
        } else if filename.ends_with("_norm.png") {
            Some(Self::Normal)
        } else if filename.ends_with("_gloss.png") {
            Some(Self::Gloss)
        } else {
            None
        }
    }
}

fn attach_path(err: &mut RichnessAssetError, path: &Path) {
    let path_buf = path.to_path_buf();
    match err {
        RichnessAssetError::PngStructure { path, .. }
        | RichnessAssetError::PngDimensions { path, .. }
        | RichnessAssetError::NormalMapInvalid { path, .. }
        | RichnessAssetError::GlossMapInvalid { path, .. } => {
            *path = path_buf;
        }
        _ => {}
    }
}

// ── Palette validation ────────────────────────────────────────────────────

/// Validate a palette.lmp file: exact size, regular file, not symlink.
pub fn validate_palette(path: &Path, expected_size: usize) -> Result<(), RichnessAssetError> {
    require_regular_file(path)?;
    let bytes = read_file(path)?;
    if bytes.len() != expected_size {
        return Err(RichnessAssetError::PaletteSize {
            path: path.to_path_buf(),
            expected: expected_size,
            actual: bytes.len(),
        });
    }
    Ok(())
}

// ── WAD validation ────────────────────────────────────────────────────────

/// Validate a WAD2 file: correct magic, all entries have lowercase ASCII names,
/// every entry has complete mip levels (4 levels for power-of-two >= 8).
pub fn validate_wad(path: &Path, expected_identities: &[&str]) -> Result<(), RichnessAssetError> {
    require_regular_file(path)?;
    let bytes = read_file(path)?;

    if bytes.len() < 12 {
        return Err(RichnessAssetError::WadStructure {
            path: path.to_path_buf(),
            reason: "too short for WAD2 header".into(),
        });
    }
    let magic = &bytes[0..4];
    if magic != b"WAD2" {
        return Err(RichnessAssetError::WadStructure {
            path: path.to_path_buf(),
            reason: format!(
                "invalid magic: {:?}",
                std::str::from_utf8(magic).unwrap_or("<bad>")
            ),
        });
    }
    let num_entries = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let dir_offset = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

    if dir_offset + num_entries * 32 > bytes.len() {
        return Err(RichnessAssetError::WadStructure {
            path: path.to_path_buf(),
            reason: "directory offset exceeds file size".into(),
        });
    }

    let mut found_identities: BTreeSet<String> = BTreeSet::new();
    let mut identity_dimensions: BTreeMap<String, (u32, u32)> = BTreeMap::new();

    for i in 0..num_entries {
        let entry_offset = dir_offset + i * 32;
        let entry = &bytes[entry_offset..entry_offset + 32];
        let filepos = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        let disksize = u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]) as usize;
        let _size = u32::from_le_bytes([entry[8], entry[9], entry[10], entry[11]]) as usize;
        let typ = entry[12];
        let _compressed = entry[13];
        let _pad = u16::from_le_bytes([entry[14], entry[15]]);
        let name_bytes = &entry[16..32];
        let nul_pos = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_bytes[..nul_pos]).map_err(|_| {
            RichnessAssetError::WadIdentityCase {
                path: path.to_path_buf(),
                identity: String::from_utf8_lossy(&name_bytes[..nul_pos]).into_owned(),
            }
        })?;

        // Identity must be lowercase ASCII
        if !name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            return Err(RichnessAssetError::WadIdentityCase {
                path: path.to_path_buf(),
                identity: name.to_string(),
            });
        }

        if typ != 0x44 {
            return Err(RichnessAssetError::WadStructure {
                path: path.to_path_buf(),
                reason: format!("entry '{name}' has type {typ:#04x}, expected 0x44"),
            });
        }

        found_identities.insert(name.to_string());

        // Validate miptex structure
        if disksize < 40
            || filepos
                .checked_add(disksize)
                .is_none_or(|end| end > bytes.len())
        {
            return Err(RichnessAssetError::WadStructure {
                path: path.to_path_buf(),
                reason: format!("miptex '{name}' payload exceeds file bounds"),
            });
        }
        if filepos + 40 > bytes.len() {
            return Err(RichnessAssetError::WadStructure {
                path: path.to_path_buf(),
                reason: format!("miptex '{name}' header offset out of range"),
            });
        }
        let miptex = &bytes[filepos..];
        let mip_name_bytes = &miptex[0..16];
        let mip_nul = mip_name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let mip_name = std::str::from_utf8(&mip_name_bytes[..mip_nul]).map_err(|_| {
            RichnessAssetError::WadIdentityCase {
                path: path.to_path_buf(),
                identity: String::from_utf8_lossy(&mip_name_bytes[..mip_nul]).into_owned(),
            }
        })?;
        if mip_name != name {
            return Err(RichnessAssetError::WadStructure {
                path: path.to_path_buf(),
                reason: format!(
                    "miptex internal name '{mip_name}' does not match directory name '{name}'"
                ),
            });
        }
        let mip_w = u32::from_le_bytes([miptex[16], miptex[17], miptex[18], miptex[19]]);
        let mip_h = u32::from_le_bytes([miptex[20], miptex[21], miptex[22], miptex[23]]);
        let offsets: [u32; 4] = [
            u32::from_le_bytes([miptex[24], miptex[25], miptex[26], miptex[27]]),
            u32::from_le_bytes([miptex[28], miptex[29], miptex[30], miptex[31]]),
            u32::from_le_bytes([miptex[32], miptex[33], miptex[34], miptex[35]]),
            u32::from_le_bytes([miptex[36], miptex[37], miptex[38], miptex[39]]),
        ];

        identity_dimensions.insert(name.to_string(), (mip_w, mip_h));

        // Verify mip chain: each level should be exactly size/2 on each axis
        let full_mip0_size = (mip_w * mip_h) as usize;
        let expected_mip_sizes = [
            full_mip0_size,
            full_mip0_size / 4,
            full_mip0_size / 16,
            full_mip0_size / 64,
        ];
        let mut mip_count = 0usize;
        for level in 0..4 {
            if offsets[level] == 0 {
                break;
            }
            let mip_start = filepos + offsets[level] as usize;
            let expected_size = expected_mip_sizes[level];
            if mip_start + expected_size > bytes.len() {
                return Err(RichnessAssetError::WadStructure {
                    path: path.to_path_buf(),
                    reason: format!("miptex '{name}' mip level {level} exceeds file bounds"),
                });
            }
            mip_count += 1;
        }
        if mip_count != 4 {
            return Err(RichnessAssetError::WadIncompleteMips {
                path: path.to_path_buf(),
                identity: name.to_string(),
                expected: 4,
                actual: mip_count,
            });
        }
    }

    // Check expected identities are present
    let expected_set: BTreeSet<String> =
        expected_identities.iter().map(|s| s.to_string()).collect();

    for id in &expected_set {
        if !found_identities.contains(id) {
            return Err(RichnessAssetError::WadMissingIdentity {
                path: path.to_path_buf(),
                identity: id.clone(),
            });
        }
    }

    for id in &found_identities {
        if !expected_set.contains(id) {
            return Err(RichnessAssetError::WadExtraIdentity {
                path: path.to_path_buf(),
                identity: id.clone(),
            });
        }
    }
    for identity in &expected_set {
        let expected_dimension = if identity == "skip" { 64 } else { 256 };
        let actual = identity_dimensions[identity];
        if actual != (expected_dimension, expected_dimension) {
            return Err(RichnessAssetError::WadStructure {
                path: path.to_path_buf(),
                reason: format!(
                    "miptex '{identity}' dimensions are {}x{}, expected {expected_dimension}x{expected_dimension}",
                    actual.0, actual.1
                ),
            });
        }
    }

    Ok(())
}

// ── License validation ────────────────────────────────────────────────────

/// Validate the LICENSE file is present and contains a CC0 dedication.
pub fn validate_license_cc0(path: &Path) -> Result<(), RichnessAssetError> {
    require_regular_file(path)?;
    let content = std::fs::read_to_string(path).map_err(|e| RichnessAssetError::Io {
        path: path.to_path_buf(),
        message: format!("read: {e}"),
    })?;
    let lower = content.to_ascii_lowercase();
    if !lower.contains("cc0") && !lower.contains("public domain") {
        return Err(RichnessAssetError::LicenseNotCc0 {
            path: path.to_path_buf(),
            found: content.lines().next().unwrap_or("<empty>").to_string(),
        });
    }
    Ok(())
}

// ── Theme definition cross-check ─────────────────────────────────────────

/// Verify the package-local frozen role table against the checked-in theme
/// document. This deliberately reads only offline asset metadata; the
/// generator itself remains free of runtime filesystem/TOML parsing.
pub fn validate_theme_toml(
    theme_dir: &Path,
    theme_def: &ThemeDefinition,
) -> Result<(), RichnessAssetError> {
    let path = theme_dir.join("theme.toml");
    require_regular_file(&path)?;
    let contents = std::fs::read_to_string(&path).map_err(|e| RichnessAssetError::Io {
        path: path.clone(),
        message: format!("read: {e}"),
    })?;
    let value: toml::Value = toml::from_str(&contents).map_err(|e| RichnessAssetError::Io {
        path: path.clone(),
        message: format!("invalid TOML: {e}"),
    })?;
    let theme = value.get("theme").and_then(toml::Value::as_table);
    let roles = value.get("roles").and_then(toml::Value::as_table);
    let name = theme
        .and_then(|table| table.get("name"))
        .and_then(toml::Value::as_str);
    let wad = theme
        .and_then(|table| table.get("wad"))
        .and_then(toml::Value::as_str);
    if name != Some(theme_def.dir_name) {
        return Err(RichnessAssetError::ProvenanceHashMismatch {
            path,
            filename: "theme.name".into(),
            expected: theme_def.dir_name.into(),
            actual: name.unwrap_or("<missing>").into(),
        });
    }
    if wad != Some(theme_def.wad_filename) {
        return Err(RichnessAssetError::ProvenanceHashMismatch {
            path,
            filename: "theme.wad".into(),
            expected: theme_def.wad_filename.into(),
            actual: wad.unwrap_or("<missing>").into(),
        });
    }
    let roles = roles.ok_or_else(|| RichnessAssetError::ProvenanceHashMismatch {
        path: path.clone(),
        filename: "roles".into(),
        expected: "nine frozen roles".into(),
        actual: "missing".into(),
    })?;
    if roles.len() != SemanticRole::ALL.len() {
        return Err(RichnessAssetError::ProvenanceHashMismatch {
            path,
            filename: "roles".into(),
            expected: SemanticRole::ALL.len().to_string(),
            actual: roles.len().to_string(),
        });
    }
    for role in SemanticRole::ALL {
        let identity = role.identity();
        if roles.get(identity).and_then(toml::Value::as_str) != Some(identity) {
            return Err(RichnessAssetError::ProvenanceHashMismatch {
                path: theme_dir.join("theme.toml"),
                filename: format!("roles.{identity}"),
                expected: identity.into(),
                actual: roles
                    .get(identity)
                    .and_then(toml::Value::as_str)
                    .unwrap_or("<missing>")
                    .into(),
            });
        }
    }
    Ok(())
}

// ── Provenance validation ─────────────────────────────────────────────────

/// Validate that provenance hashes match every generated output.
/// hashes for the associated output files.
pub fn validate_provenance_hashes(
    theme_dir: &Path,
    theme_def: &ThemeDefinition,
) -> Result<(), RichnessAssetError> {
    let prov_path = theme_dir.join("provenance.toml");
    require_regular_file(&prov_path)?;
    let _content = std::fs::read_to_string(&prov_path).map_err(|e| RichnessAssetError::Io {
        path: prov_path.clone(),
        message: format!("read: {e}"),
    })?;

    // Verify provenance.toml is valid TOML
    let _prov: toml::Value = toml::from_str(&_content).map_err(|e| RichnessAssetError::Io {
        path: prov_path.clone(),
        message: format!("invalid TOML: {e}"),
    })?;

    // Compute hashes of all output files and record them. The provenance file
    // itself declares the expected file count and identities; we verify that
    // every declared output actually exists in the closure.
    let wad_path = theme_dir.join(theme_def.wad_filename);
    let palette_path = theme_dir.join(theme_def.palette_filename);
    let textures_dir = theme_dir.join("textures");

    if wad_path.exists() {
        require_regular_file(&wad_path)?;
    }
    if palette_path.exists() {
        require_regular_file(&palette_path)?;
    }
    if textures_dir.exists() {
        require_directory(&textures_dir)?;
    }

    let prov_name = _prov
        .get("theme")
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if prov_name != theme_def.dir_name {
        return Err(RichnessAssetError::ProvenanceHashMismatch {
            path: prov_path,
            filename: "provenance.toml".into(),
            expected: theme_def.dir_name.to_string(),
            actual: prov_name.to_string(),
        });
    }

    let hashes = _prov
        .get("hashes")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| RichnessAssetError::ProvenanceHashMismatch {
            path: prov_path.clone(),
            filename: "[hashes]".into(),
            expected: "hash table for every generated output".into(),
            actual: "missing or invalid hash table".into(),
        })?;
    let expected = provenance_hashed_filenames(theme_def);
    if hashes.len() != expected.len() {
        return Err(RichnessAssetError::ProvenanceHashMismatch {
            path: prov_path.clone(),
            filename: "[hashes]".into(),
            expected: expected.len().to_string(),
            actual: hashes.len().to_string(),
        });
    }
    for filename in expected {
        let expected_hash = hashes
            .get(&filename)
            .and_then(toml::Value::as_str)
            .ok_or_else(|| RichnessAssetError::ProvenanceHashMismatch {
                path: prov_path.clone(),
                filename: filename.clone(),
                expected: "lowercase SHA-256".into(),
                actual: "missing or non-string".into(),
            })?;
        let asset_path = theme_dir.join(&filename);
        let actual_hash = sha256_hex(&read_file(&asset_path)?);
        if expected_hash != actual_hash {
            return Err(RichnessAssetError::ProvenanceHashMismatch {
                path: prov_path.clone(),
                filename,
                expected: expected_hash.to_string(),
                actual: actual_hash,
            });
        }
    }

    Ok(())
}

// ── Case stability ────────────────────────────────────────────────────────

/// Verify that every file in the theme directory matches its expected
/// exact-case filename. Reject case-insensitive ambiguities.
pub fn validate_case_stability(
    theme_dir: &Path,
    expected_filenames: &BTreeSet<String>,
) -> Result<(), RichnessAssetError> {
    require_directory(theme_dir)?;

    let entries = std::fs::read_dir(theme_dir).map_err(|e| RichnessAssetError::Io {
        path: theme_dir.to_path_buf(),
        message: format!("read_dir: {e}"),
    })?;

    let mut on_disk: BTreeMap<String, String> = BTreeMap::new(); // lower → actual
    for entry in entries {
        let entry = entry.map_err(|e| RichnessAssetError::Io {
            path: theme_dir.to_path_buf(),
            message: format!("entry: {e}"),
        })?;
        let actual_name = entry.file_name().to_string_lossy().into_owned();
        let lower = actual_name.to_ascii_lowercase();

        if let std::collections::btree_map::Entry::Vacant(e) = on_disk.entry(lower.clone()) {
            e.insert(actual_name);
        } else {
            return Err(RichnessAssetError::AmbiguousPath {
                dir: theme_dir.to_path_buf(),
                filename: lower,
            });
        }
    }

    for expected in expected_filenames {
        let lower_expected = expected.to_ascii_lowercase();
        match on_disk.get(&lower_expected) {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(RichnessAssetError::CaseMismatch {
                    dir: theme_dir.to_path_buf(),
                    expected: expected.clone(),
                    found: Some(actual.clone()),
                });
            }
            None => {
                return Err(RichnessAssetError::MissingFile {
                    dir: theme_dir.to_path_buf(),
                    filename: expected.clone(),
                });
            }
        }
    }

    Ok(())
}

fn validate_case_stability_dir(
    dir: &Path,
    expected_filenames: &BTreeSet<String>,
) -> Result<(), RichnessAssetError> {
    let entries = std::fs::read_dir(dir).map_err(|e| RichnessAssetError::Io {
        path: dir.to_path_buf(),
        message: format!("read_dir: {e}"),
    })?;

    let mut on_disk: BTreeMap<String, String> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| RichnessAssetError::Io {
            path: dir.to_path_buf(),
            message: format!("entry: {e}"),
        })?;
        let actual_name = entry.file_name().to_string_lossy().into_owned();
        let lower = actual_name.to_ascii_lowercase();

        if let std::collections::btree_map::Entry::Vacant(e) = on_disk.entry(lower.clone()) {
            e.insert(actual_name);
        } else {
            return Err(RichnessAssetError::AmbiguousPath {
                dir: dir.to_path_buf(),
                filename: lower,
            });
        }
    }

    for expected in expected_filenames {
        let lower_expected = expected.to_ascii_lowercase();
        match on_disk.get(&lower_expected) {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(RichnessAssetError::CaseMismatch {
                    dir: dir.to_path_buf(),
                    expected: expected.clone(),
                    found: Some(actual.clone()),
                });
            }
            None => {} // Not every expected file lives in textures/
        }
    }

    Ok(())
}

// ── Full theme closure validation ─────────────────────────────────────────

/// Collect all expected filenames for a theme closure (relative to theme root).
/// Generated outputs covered by the provenance hash table.  The provenance
/// document cannot hash itself; `build.py` is an input, not a generated output.
fn provenance_hashed_filenames(theme_def: &ThemeDefinition) -> BTreeSet<String> {
    let mut expected = BTreeSet::new();
    expected.insert(theme_def.wad_filename.to_string());
    expected.insert(theme_def.palette_filename.to_string());
    expected.insert("LICENSE".to_string());
    expected.insert("theme.toml".to_string());
    for filename in theme_def.all_png_filenames() {
        expected.insert(format!("textures/{filename}"));
    }
    expected
}

fn expected_closure_filenames(theme_def: &ThemeDefinition) -> BTreeSet<String> {
    let mut expected = BTreeSet::new();
    expected.insert(theme_def.wad_filename.to_string());
    expected.insert(theme_def.palette_filename.to_string());
    expected.insert("LICENSE".to_string());
    expected.insert("provenance.toml".to_string());
    expected.insert("theme.toml".to_string());
    expected.insert("build.py".to_string());
    // textures/ files (relative to textures/ dir)
    for filename in theme_def.all_png_filenames() {
        expected.insert(format!("textures/{filename}"));
    }
    expected.insert("textures".to_string()); // the directory itself
    expected
}

/// Validate a complete theme closure against a `ThemeDefinition`.
///
/// Checks:
/// - Every expected file exists and is a regular file (or directory, for textures/).
/// - No extra files beyond the expected closure.
/// - No symlinks.
/// - Exact-case filename match.
/// - PNG CRC + dimensions + pixel validation (normal/gloss).
/// - Palette size 768.
/// - WAD identities + mip chains.
/// - License is CC0.
/// - Provenance is valid TOML and name matches.
pub fn validate_theme_closure(
    theme_dir: &Path,
    theme_def: &ThemeDefinition,
) -> Result<(), RichnessAssetError> {
    require_directory(theme_dir)?;

    // Verify no stray files
    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    let mut on_disk_files: Vec<PathBuf> = Vec::new();
    collect_files(theme_dir, theme_dir, &mut on_disk, &mut on_disk_files)?;

    let expected = expected_closure_filenames(theme_def);

    for path_str in &on_disk {
        if !expected.contains(path_str) {
            return Err(RichnessAssetError::ExtraFile {
                dir: theme_dir.to_path_buf(),
                filename: path_str.clone(),
            });
        }
    }

    for expected_name in &expected {
        if expected_name == "textures" {
            continue; // directory handled separately
        }
        if !on_disk.contains(expected_name) {
            return Err(RichnessAssetError::MissingFile {
                dir: theme_dir.to_path_buf(),
                filename: expected_name.clone(),
            });
        }
    }

    // Validate case stability — split expected into top-level and textures/
    let expected_top: BTreeSet<String> = expected
        .iter()
        .filter(|name| !name.starts_with("textures/") && *name != "textures")
        .cloned()
        .collect();
    let expected_textures: BTreeSet<String> = expected
        .iter()
        .filter(|name| name.starts_with("textures/"))
        .map(|name| name.strip_prefix("textures/").unwrap_or(name).to_string())
        .collect();
    validate_case_stability(theme_dir, &expected_top)?;
    let textures_dir = theme_dir.join("textures");
    if textures_dir.exists() {
        validate_case_stability_dir(&textures_dir, &expected_textures)?;
    }

    // Validate specific assets
    validate_theme_toml(theme_dir, theme_def)?;

    // Palette
    let palette_path = theme_dir.join(theme_def.palette_filename);
    validate_palette(&palette_path, theme_def.palette_size)?;

    // WAD
    let wad_path = theme_dir.join(theme_def.wad_filename);
    validate_wad(&wad_path, &theme_def.all_wad_identities())?;

    // License
    let license_path = theme_dir.join("LICENSE");
    validate_license_cc0(&license_path)?;

    // Provenance
    validate_provenance_hashes(theme_dir, theme_def)?;

    // PNGs in textures/
    let textures_dir = theme_dir.join("textures");
    for filename in theme_def.all_png_filenames() {
        let png_path = textures_dir.join(&filename);
        let kind = PngCompanionKind::from_filename_suffix(&filename)
            .unwrap_or(PngCompanionKind::Basecolor);
        validate_png_asset(&png_path, theme_def.texture_size, kind)?;
    }

    Ok(())
}

fn collect_files(
    root: &Path,
    dir: &Path,
    relative_set: &mut BTreeSet<String>,
    file_paths: &mut Vec<PathBuf>,
) -> Result<(), RichnessAssetError> {
    let entries = std::fs::read_dir(dir).map_err(|e| RichnessAssetError::Io {
        path: dir.to_path_buf(),
        message: format!("read_dir: {e}"),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| RichnessAssetError::Io {
            path: dir.to_path_buf(),
            message: format!("entry: {e}"),
        })?;
        let path = entry.path();
        let meta = entry.metadata().map_err(|e| RichnessAssetError::Io {
            path: path.clone(),
            message: format!("metadata: {e}"),
        })?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        if meta.file_type().is_symlink() {
            return Err(RichnessAssetError::SymlinkRejected { path });
        }
        if meta.is_dir() {
            relative_set.insert(rel.clone());
            collect_files(root, &path, relative_set, file_paths)?;
        } else if meta.is_file() {
            relative_set.insert(rel.clone());
            file_paths.push(path);
        }
    }
    Ok(())
}

// ── Deterministic fresh-build comparator ──────────────────────────────────

/// Run a theme's build.py in a clean temporary directory and byte-compare
/// every declared output against the checked-in closure.
///
/// Returns `Ok(())` if all outputs match. Returns `FreshBuildUnavailable`
/// if Python 3 or Pillow is not available. Returns `FreshBuildMismatch`
/// if any output differs.
pub fn fresh_build_compare(
    theme_dir: &Path,
    theme_def: &ThemeDefinition,
) -> Result<(), RichnessAssetError> {
    // Check that Python 3 is available
    let python_check = std::process::Command::new("python3")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match python_check {
        Ok(status) if status.success() => {}
        _ => {
            return Err(RichnessAssetError::FreshBuildUnavailable {
                theme_dir: theme_dir.to_path_buf(),
                reason: "python3 not available".into(),
            });
        }
    }

    // Check Pillow is importable
    let pillow_check = std::process::Command::new("python3")
        .args(["-c", "from PIL import Image"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match pillow_check {
        Ok(status) if status.success() => {}
        _ => {
            return Err(RichnessAssetError::FreshBuildUnavailable {
                theme_dir: theme_dir.to_path_buf(),
                reason: "Pillow not available (pip install Pillow)".into(),
            });
        }
    }

    // Create a temporary directory
    let build_dir = create_temp_dir()?;
    let second_build_dir = create_temp_dir()?;
    let _cleanup = TempDirGuard(build_dir.clone());
    let _second_cleanup = TempDirGuard(second_build_dir.clone());

    // Copy the deterministic input into two independent clean roots.
    let build_py_src = theme_dir.join("build.py");
    for dir in [&build_dir, &second_build_dir] {
        std::fs::copy(&build_py_src, dir.join("build.py")).map_err(|e| RichnessAssetError::Io {
            path: build_py_src.clone(),
            message: format!("copy build.py: {e}"),
        })?;
    }

    // Run the first clean build.
    let output = std::process::Command::new("python3")
        .arg("build.py")
        .arg(build_dir.to_string_lossy().as_ref())
        .current_dir(&build_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| RichnessAssetError::FreshBuildUnavailable {
            theme_dir: theme_dir.to_path_buf(),
            reason: format!("failed to run build.py: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RichnessAssetError::FreshBuildMismatch {
            theme_dir: theme_dir.to_path_buf(),
            filename: "build.py".into(),
            details: format!("build.py failed: {stderr}"),
        });
    }

    // Run the second clean build before comparing either result.
    let second_output = std::process::Command::new("python3")
        .arg("build.py")
        .arg(second_build_dir.to_string_lossy().as_ref())
        .current_dir(&second_build_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| RichnessAssetError::FreshBuildUnavailable {
            theme_dir: theme_dir.to_path_buf(),
            reason: format!("failed to run second build.py: {e}"),
        })?;
    if !second_output.status.success() {
        return Err(RichnessAssetError::FreshBuildMismatch {
            theme_dir: theme_dir.to_path_buf(),
            filename: "build.py".into(),
            details: format!(
                "second build.py failed: {}",
                String::from_utf8_lossy(&second_output.stderr)
            ),
        });
    }

    // Compare every declared output
    let compare_files = [
        theme_def.wad_filename,
        theme_def.palette_filename,
        "theme.toml",
        "LICENSE",
        "provenance.toml",
    ];

    for filename in &compare_files {
        let checked_in = theme_dir.join(filename);
        let fresh = build_dir.join(filename);
        if !fresh.exists() {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: filename.to_string(),
                details: "fresh build did not produce this file".into(),
            });
        }
        let checked_bytes = std::fs::read(&checked_in).map_err(|e| RichnessAssetError::Io {
            path: checked_in.clone(),
            message: format!("read: {e}"),
        })?;
        let fresh_bytes = std::fs::read(&fresh).map_err(|e| RichnessAssetError::Io {
            path: fresh.clone(),
            message: format!("read: {e}"),
        })?;
        if checked_bytes != fresh_bytes {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: filename.to_string(),
                details: "byte-level mismatch between checked-in and fresh build".into(),
            });
        }
    }

    // Compare textures/
    let fresh_textures = build_dir.join("textures");
    let checked_textures = theme_dir.join("textures");
    for filename in theme_def.all_png_filenames() {
        let checked_png = checked_textures.join(&filename);
        let fresh_png = fresh_textures.join(&filename);
        if !fresh_png.exists() {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: format!("textures/{filename}"),
                details: "fresh build did not produce this texture".into(),
            });
        }
        let checked_bytes = std::fs::read(&checked_png).map_err(|e| RichnessAssetError::Io {
            path: checked_png.clone(),
            message: format!("read: {e}"),
        })?;
        let fresh_bytes = std::fs::read(&fresh_png).map_err(|e| RichnessAssetError::Io {
            path: fresh_png.clone(),
            message: format!("read: {e}"),
        })?;
        if checked_bytes != fresh_bytes {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: format!("textures/{filename}"),
                details: "byte-level mismatch between checked-in and fresh build".into(),
            });
        }
    }

    // The two fresh roots must agree byte-for-byte independently of the
    // checked-in closure. This catches a stale but self-consistent checkout.
    for filename in compare_files {
        let first =
            std::fs::read(build_dir.join(filename)).map_err(|e| RichnessAssetError::Io {
                path: build_dir.join(filename),
                message: format!("read: {e}"),
            })?;
        let second =
            std::fs::read(second_build_dir.join(filename)).map_err(|e| RichnessAssetError::Io {
                path: second_build_dir.join(filename),
                message: format!("read: {e}"),
            })?;
        if first != second {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: filename.to_string(),
                details: "the two clean builds differ".into(),
            });
        }
    }
    for filename in theme_def.all_png_filenames() {
        let first = std::fs::read(build_dir.join("textures").join(&filename)).map_err(|e| {
            RichnessAssetError::Io {
                path: build_dir.join("textures").join(&filename),
                message: format!("read: {e}"),
            }
        })?;
        let second =
            std::fs::read(second_build_dir.join("textures").join(&filename)).map_err(|e| {
                RichnessAssetError::Io {
                    path: second_build_dir.join("textures").join(&filename),
                    message: format!("read: {e}"),
                }
            })?;
        if first != second {
            return Err(RichnessAssetError::FreshBuildMismatch {
                theme_dir: theme_dir.to_path_buf(),
                filename: format!("textures/{filename}"),
                details: "the two clean builds differ".into(),
            });
        }
    }

    Ok(())
}

// ── Strict package staging ────────────────────────────────────────────────

/// Stage exactly the required identities plus companions for a Richness theme.
///
/// Fails on: extra files, missing files, ambiguous paths, symlinked paths.
/// Copies: WAD, palette, LICENSE, theme.toml, provenance.toml, and all
/// PNG companions to `textures/` under the staging root.
pub fn stage_richness_package(
    theme_dir: &Path,
    staging: &Path,
    theme_def: &ThemeDefinition,
) -> Result<BTreeSet<String>, RichnessAssetError> {
    validate_theme_closure(theme_dir, theme_def)?;
    require_directory(staging)?;
    if std::fs::read_dir(staging)
        .map_err(|e| RichnessAssetError::Staging {
            path: staging.to_path_buf(),
            message: format!("read staging directory: {e}"),
        })?
        .next()
        .is_some()
    {
        return Err(RichnessAssetError::Staging {
            path: staging.to_path_buf(),
            message: "staging directory must be empty".into(),
        });
    }

    // Create staging root + textures/
    std::fs::create_dir_all(staging.join("textures")).map_err(|e| RichnessAssetError::Staging {
        path: staging.to_path_buf(),
        message: format!("create textures dir: {e}"),
    })?;

    let mut staged: BTreeSet<String> = BTreeSet::new();

    // Copy WAD
    let wad_src = theme_dir.join(theme_def.wad_filename);
    require_regular_file(&wad_src)?;
    let wad_dst = staging.join(theme_def.wad_filename);
    std::fs::copy(&wad_src, &wad_dst).map_err(|e| RichnessAssetError::Io {
        path: wad_src.clone(),
        message: format!("copy WAD: {e}"),
    })?;
    staged.insert(theme_def.wad_filename.to_string());

    // Copy palette
    let palette_src = theme_dir.join(theme_def.palette_filename);
    require_regular_file(&palette_src)?;
    let palette_dst = staging.join(theme_def.palette_filename);
    std::fs::copy(&palette_src, &palette_dst).map_err(|e| RichnessAssetError::Io {
        path: palette_src.clone(),
        message: format!("copy palette: {e}"),
    })?;
    staged.insert(theme_def.palette_filename.to_string());

    // Copy LICENSE
    let license_src = theme_dir.join("LICENSE");
    require_regular_file(&license_src)?;
    let license_dst = staging.join("LICENSE");
    std::fs::copy(&license_src, &license_dst).map_err(|e| RichnessAssetError::Io {
        path: license_src.clone(),
        message: format!("copy LICENSE: {e}"),
    })?;
    staged.insert("LICENSE".to_string());

    // Copy theme.toml
    let theme_toml_src = theme_dir.join("theme.toml");
    require_regular_file(&theme_toml_src)?;
    let theme_toml_dst = staging.join("theme.toml");
    std::fs::copy(&theme_toml_src, &theme_toml_dst).map_err(|e| RichnessAssetError::Io {
        path: theme_toml_src.clone(),
        message: format!("copy theme.toml: {e}"),
    })?;
    staged.insert("theme.toml".to_string());

    // Copy provenance.toml
    let prov_src = theme_dir.join("provenance.toml");
    require_regular_file(&prov_src)?;
    let prov_dst = staging.join("provenance.toml");
    std::fs::copy(&prov_src, &prov_dst).map_err(|e| RichnessAssetError::Io {
        path: prov_src.clone(),
        message: format!("copy provenance.toml: {e}"),
    })?;
    staged.insert("provenance.toml".to_string());

    // Copy all PNG companions
    let textures_src = theme_dir.join("textures");
    require_directory(&textures_src)?;
    for filename in theme_def.all_png_filenames() {
        let src = textures_src.join(&filename);
        require_regular_file(&src)?;
        let dst = staging.join("textures").join(&filename);
        std::fs::copy(&src, &dst).map_err(|e| RichnessAssetError::Io {
            path: src.clone(),
            message: format!("copy companion: {e}"),
        })?;
        staged.insert(format!("textures/{filename}"));
    }

    Ok(staged)
}

/// Compute SHA-256 hashes for all files in a Richness theme closure.
pub fn compute_richness_hashes(
    theme_dir: &Path,
    theme_def: &ThemeDefinition,
) -> Result<Vec<(String, String)>, RichnessAssetError> {
    let mut hashes: Vec<(String, String)> = Vec::new();

    let files = [
        theme_def.wad_filename,
        theme_def.palette_filename,
        "LICENSE",
        "theme.toml",
        "provenance.toml",
    ];

    for filename in &files {
        let path = theme_dir.join(filename);
        let bytes = read_file(&path)?;
        let hash = sha256_hex(&bytes);
        hashes.push((filename.to_string(), hash));
    }

    let textures_dir = theme_dir.join("textures");
    for filename in theme_def.all_png_filenames() {
        let path = textures_dir.join(&filename);
        let bytes = read_file(&path)?;
        let hash = sha256_hex(&bytes);
        hashes.push((format!("textures/{filename}"), hash));
    }

    hashes.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(hashes)
}

// ── Temporary directory helpers ────────────────────────────────────────────

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn create_temp_dir() -> Result<PathBuf, RichnessAssetError> {
    let mut base = std::env::temp_dir();
    let suffix: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    base.push(format!("richness_fresh_build_{suffix:x}"));
    std::fs::create_dir_all(&base).map_err(|e| RichnessAssetError::Staging {
        path: base.clone(),
        message: format!("create temp dir: {e}"),
    })?;
    Ok(base)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn theme_root() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .expect("engine_pack not under workspace tools/")
            .join("src/bsp_generator/themes")
    }

    fn ancient_dir() -> PathBuf {
        theme_root().join("richness_ancient_v1")
    }

    fn egyptian_dir() -> PathBuf {
        theme_root().join("richness_egyptian_v1")
    }

    fn brutalist_dir() -> PathBuf {
        theme_root().join("richness_brutalist_v1")
    }

    // ── Closure validation ────────────────────────────────────────────

    #[test]
    fn ancient_closure_passes_validation() {
        let dir = ancient_dir();
        validate_theme_closure(&dir, &THEME_ANCIENT).expect("ancient closure must validate");
    }

    #[test]
    fn python_bytecode_is_not_part_of_the_exact_closure() {
        let dir = create_temp_dir().expect("temp dir");
        let _cleanup = TempDirGuard(dir.clone());
        let pycache = dir.join("__pycache__");
        std::fs::create_dir_all(&pycache).expect("create pycache");
        std::fs::write(pycache.join("build.cpython-314.pyc"), b"not allowed").expect("write pyc");

        let error = validate_theme_closure(&dir, &THEME_ANCIENT)
            .expect_err("python bytecode must fail the exact closure");
        assert!(matches!(
            error,
            RichnessAssetError::ExtraFile { ref filename, .. }
                if filename == "__pycache__"
        ));
    }

    #[test]
    fn egyptian_closure_passes_validation() {
        let dir = egyptian_dir();
        validate_theme_closure(&dir, &THEME_EGYPTIAN).expect("egyptian closure must validate");
    }

    #[test]
    fn brutalist_closure_passes_validation() {
        let dir = brutalist_dir();
        validate_theme_closure(&dir, &THEME_BRUTALIST).expect("brutalist closure must validate");
    }

    // ── Palette validation ────────────────────────────────────────────

    #[test]
    fn all_palettes_are_768_bytes() {
        for (dir, def) in &[
            (ancient_dir(), &THEME_ANCIENT),
            (egyptian_dir(), &THEME_EGYPTIAN),
            (brutalist_dir(), &THEME_BRUTALIST),
        ] {
            let path = dir.join(def.palette_filename);
            validate_palette(&path, 768).unwrap_or_else(|e| panic!("{dir:?}: {e}"));
        }
    }

    // ── WAD validation ────────────────────────────────────────────────

    #[test]
    fn all_wads_have_correct_identities() {
        for (dir, def) in &[
            (ancient_dir(), &THEME_ANCIENT),
            (egyptian_dir(), &THEME_EGYPTIAN),
            (brutalist_dir(), &THEME_BRUTALIST),
        ] {
            let path = dir.join(def.wad_filename);
            validate_wad(&path, &def.all_wad_identities())
                .unwrap_or_else(|e| panic!("{dir:?}: {e}"));
        }
    }

    // ── License validation ────────────────────────────────────────────

    #[test]
    fn all_licenses_are_cc0() {
        for dir in &[ancient_dir(), egyptian_dir(), brutalist_dir()] {
            let path = dir.join("LICENSE");
            validate_license_cc0(&path).unwrap_or_else(|e| panic!("{dir:?}: {e}"));
        }
    }

    // ── Case stability ────────────────────────────────────────────────

    #[test]
    fn all_themes_are_case_stable() {
        for (dir, def) in &[
            (ancient_dir(), &THEME_ANCIENT),
            (egyptian_dir(), &THEME_EGYPTIAN),
            (brutalist_dir(), &THEME_BRUTALIST),
        ] {
            let expected = expected_closure_filenames(def);
            let expected_top: BTreeSet<String> = expected
                .iter()
                .filter(|name| !name.starts_with("textures/") && *name != "textures")
                .cloned()
                .collect();
            validate_case_stability(dir, &expected_top).unwrap_or_else(|e| panic!("{dir:?}: {e}"));
            // Also check textures/ subdirectory
            let textures_dir = dir.join("textures");
            if textures_dir.exists() {
                let expected_tex: BTreeSet<String> = expected
                    .iter()
                    .filter(|name| name.starts_with("textures/"))
                    .map(|name| name.strip_prefix("textures/").unwrap_or(name).to_string())
                    .collect();
                if !expected_tex.is_empty() {
                    validate_case_stability_dir(&textures_dir, &expected_tex)
                        .unwrap_or_else(|e| panic!("{dir:?}/textures: {e}"));
                }
            }
        }
    }

    // ── PNG validation ────────────────────────────────────────────────

    #[test]
    fn all_pngs_pass_crc_and_dimensions() {
        for (dir, def) in &[
            (ancient_dir(), &THEME_ANCIENT),
            (egyptian_dir(), &THEME_EGYPTIAN),
            (brutalist_dir(), &THEME_BRUTALIST),
        ] {
            let textures_dir = dir.join("textures");
            for filename in def.all_png_filenames() {
                let path = textures_dir.join(&filename);
                let kind = PngCompanionKind::from_filename_suffix(&filename)
                    .unwrap_or(PngCompanionKind::Basecolor);
                validate_png_asset(&path, 256, kind).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            }
        }
    }

    // ── Hash computation ──────────────────────────────────────────────

    #[test]
    fn compute_all_hashes() {
        for (dir, def) in &[
            (ancient_dir(), &THEME_ANCIENT),
            (egyptian_dir(), &THEME_EGYPTIAN),
            (brutalist_dir(), &THEME_BRUTALIST),
        ] {
            let hashes = compute_richness_hashes(dir, def).expect("hash computation");
            assert_eq!(
                hashes.len(),
                5 + 27,
                "expected 32 files (5 static + 27 PNGs)"
            );
            for (filename, hash) in &hashes {
                assert_eq!(
                    hash.len(),
                    64,
                    "SHA-256 must be 64 hex chars for {filename}"
                );
                assert!(
                    hash.chars().all(|c| c.is_ascii_hexdigit()),
                    "non-hex hash for {filename}"
                );
            }
        }
    }

    // ── Staging ───────────────────────────────────────────────────────

    #[test]
    fn stage_ancient_produces_complete_closure() {
        let staging = tempfile::tempdir().expect("tempdir");
        let dir = ancient_dir();
        let def = &THEME_ANCIENT;
        let staged = stage_richness_package(&dir, staging.path(), def).expect("stage");
        assert_eq!(staged.len(), 5 + 27, "all static + PNG files staged");
        assert!(staging.path().join(def.wad_filename).exists());
        assert!(staging.path().join("textures/wall_basecolor.png").exists());
        assert!(staging.path().join("textures/wall_norm.png").exists());
        assert!(staging.path().join("textures/wall_gloss.png").exists());
    }

    // ── Negative tests ────────────────────────────────────────────────

    #[test]
    fn reject_missing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let theme_toml = tmp.path().join("theme.toml");
        std::fs::write(&theme_toml, "x").unwrap();
        let result = validate_case_stability(tmp.path(), &{
            let mut s = BTreeSet::new();
            s.insert("theme.toml".to_string());
            s.insert("nonexistent.txt".to_string());
            s
        });
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("missing"),
            "expected missing file error, got: {msg}"
        );
    }

    #[test]
    fn reject_palette_wrong_size() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let pal = tmp.path().join("palette.lmp");
        std::fs::write(&pal, &[0u8; 100]).unwrap();
        let result = validate_palette(&pal, 768);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("size mismatch"),
            "expected size error, got: {msg}"
        );
    }

    #[test]
    fn reject_symlink() {
        // This test can only run on platforms with symlink support
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real.png");
        std::fs::write(&real, b"\x89PNG\r\n\x1a\n").unwrap();
        let link = tmp.path().join("link.png");
        let symlink_result = std::os::unix::fs::symlink(&real, &link);
        if symlink_result.is_ok() {
            let result = require_regular_file(&link);
            assert!(result.is_err());
        }
    }

    #[test]
    fn reject_invalid_png() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let png = tmp.path().join("bad.png");
        std::fs::write(&png, b"not a png file").unwrap();
        let result = validate_png_crc_and_dimensions_bytes(b"not a png file", 256);
        assert!(result.is_err());
    }

    #[test]
    fn fresh_build_compare_ancient() {
        let dir = ancient_dir();
        let def = &THEME_ANCIENT;
        match fresh_build_compare(&dir, def) {
            Ok(()) => {} // success
            Err(RichnessAssetError::FreshBuildUnavailable { .. }) => {
                eprintln!("SKIP: fresh-build comparison requires python3 + Pillow");
            }
            Err(e) => panic!("fresh-build comparison failed: {e}"),
        }
    }
}
