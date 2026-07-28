//! Environment source import pipeline.
//!
//! Handles decoding, layout detection, and conversion of environment source files
//! into `PendingSkyboxSource` for GPU upload.

use std::path::{Path, PathBuf};

use ash::vk;
use image::{DynamicImage, GenericImageView};
use log::{info, warn};

/// Decoded RGBA32F image in linear space.
pub struct DecodedImageF32 {
    pub width: u32,
    pub height: u32,
    /// RGBA32F linear, tightly packed (4 floats per pixel).
    pub rgba32f: Vec<f32>,
}

/// Detected source layout after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLayout {
    Equirectangular,
    CubeStrip,
}

/// Face-major cubemap data ready for GPU upload.
#[derive(Debug)]
pub struct CubemapFaces {
    pub face_size: u32,
    /// RGBA32F bytes in face-major order: +X, -X, +Y, -Y, +Z, -Z.
    pub rgba32f: Vec<f32>,
}

/// Pending skybox source data waiting for GPU upload.
#[derive(Clone)]
pub enum PendingSkyboxSource {
    /// Face-major cubemap data: +X, -X, +Y, -Y, +Z, -Z.
    CubemapFaces {
        face_size: u32,
        format: vk::Format,
        bytes: Vec<u8>,
    },
    /// 2D equirectangular image for GPU-side conversion.
    Equirectangular2D {
        width: u32,
        height: u32,
        format: vk::Format,
        bytes: Vec<u8>,
    },
}

/// Face naming pattern for directory-based cubemap loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacePattern {
    /// Try all known aliases (px/posx/right/+x, etc.)
    AutoAliases,
    /// Expect px/nx/py/ny/pz/nz naming.
    PxNxPyNyPzNz,
    /// Expect posx/negx/posy/negy/posz/negz naming.
    PosxNegxPosyNegyPoszNegz,
}

/// Canonical face order: +X, -X, +Y, -Y, +Z, -Z.
const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// Alias sets for each face (canonical order).
const FACE_ALIASES: [[&str; 4]; 6] = [
    ["px", "posx", "+x", "right"],
    ["nx", "negx", "-x", "left"],
    ["py", "posy", "+y", "top"],
    ["ny", "negy", "-y", "bottom"],
    ["pz", "posz", "+z", "front"],
    ["nz", "negz", "-z", "back"],
];

/// Specific aliases for PxNxPyNyPzNz pattern.
const FACE_ALIASES_PX: [&str; 6] = ["px", "nx", "py", "ny", "pz", "nz"];

/// Specific aliases for PosxNegxPosyNegyPoszNegz pattern.
const FACE_ALIASES_POSX: [&str; 6] = ["posx", "negx", "posy", "negy", "posz", "negz"];

// ─── Decoding ───────────────────────────────────────────────────────────────

/// Decode an environment image file to RGBA32F linear data.
pub fn decode_environment_file(path: &Path) -> Result<DecodedImageF32, String> {
    let image = image::open(path).map_err(|err| {
        format!(
            "Failed to decode environment file '{}': {}",
            path.display(),
            err
        )
    })?;

    let (width, height) = image.dimensions();

    // Check for LDR formats and warn
    match &image {
        DynamicImage::ImageRgb8(_)
        | DynamicImage::ImageRgba8(_)
        | DynamicImage::ImageLuma8(_)
        | DynamicImage::ImageLumaA8(_) => {
            warn!(
                "Environment file '{}' is LDR format; IBL quality/dynamic range will be limited",
                path.display()
            );
        }
        _ => {}
    }

    let rgba32f_image = image.into_rgba32f();
    let mut rgba32f = rgba32f_image.into_raw();

    // Sanitize NaN/Inf values
    sanitize_float_data(&mut rgba32f);

    info!(
        "Decoded environment file '{}': {}x{}, {} floats",
        path.display(),
        width,
        height,
        rgba32f.len()
    );

    Ok(DecodedImageF32 {
        width,
        height,
        rgba32f,
    })
}

/// Sanitize float data: NaN -> 0.0, +/-Inf -> clamped.
fn sanitize_float_data(data: &mut [f32]) {
    const MAX_HDR: f32 = 65504.0; // max representable in f16
    for v in data.iter_mut() {
        if v.is_nan() {
            *v = 0.0;
        } else if *v == f32::INFINITY {
            *v = MAX_HDR;
        } else if *v == f32::NEG_INFINITY {
            *v = -MAX_HDR;
        }
    }
}

// ─── Layout Detection ───────────────────────────────────────────────────────

/// Detect source layout from image dimensions.
///
/// - `w == 2 * h` -> Equirectangular
/// - `w == 6 * h` -> CubeStrip (horizontal strip)
/// - otherwise -> error
pub fn detect_source_layout(width: u32, height: u32) -> Result<SourceLayout, String> {
    if width == 2 * height {
        Ok(SourceLayout::Equirectangular)
    } else if width == 6 * height {
        Ok(SourceLayout::CubeStrip)
    } else {
        Err(format!(
            "Cannot auto-detect environment layout for {}x{} image. \
             Expected 2:1 (equirectangular) or 6:1 (cube strip). \
             Use an explicit EnvironmentSource variant instead of Auto.",
            width, height
        ))
    }
}

// ─── Strip Conversion ───────────────────────────────────────────────────────

/// Convert a horizontal 6:1 cube strip to face-major cubemap data.
///
/// Input: RGBA32F row-major image where `w = 6 * face_size`, `h = face_size`.
/// Output: RGBA32F face-major data (+X, -X, +Y, -Y, +Z, -Z).
pub fn convert_strip_to_face_major(decoded: &DecodedImageF32) -> Result<CubemapFaces, String> {
    let face_size = decoded.height;
    if decoded.width != 6 * face_size {
        return Err(format!(
            "Strip image dimensions {}x{} are not 6:1 (expected {}x{})",
            decoded.width,
            decoded.height,
            6 * face_size,
            face_size
        ));
    }
    if face_size == 0 {
        return Err("Strip image has zero height".to_string());
    }

    let pixels_per_face = (face_size * face_size) as usize;
    let floats_per_face = pixels_per_face * 4;
    let mut output = Vec::with_capacity(floats_per_face * 6);

    let stride = decoded.width as usize; // pixels per row in the full strip

    for face_idx in 0..6u32 {
        let x_offset = (face_idx * face_size) as usize;

        for row in 0..face_size as usize {
            let src_pixel_start = row * stride + x_offset;
            let src_float_start = src_pixel_start * 4;
            let src_float_end = src_float_start + (face_size as usize * 4);
            output.extend_from_slice(&decoded.rgba32f[src_float_start..src_float_end]);
        }
    }

    Ok(CubemapFaces {
        face_size,
        rgba32f: output,
    })
}

// ─── Face Directory Loading ─────────────────────────────────────────────────

/// Load a cubemap from a directory of face images.
pub fn load_face_directory(dir: &Path, pattern: FacePattern) -> Result<CubemapFaces, String> {
    if !dir.is_dir() {
        return Err(format!(
            "Face directory path '{}' does not exist or is not a directory",
            dir.display()
        ));
    }

    let face_paths = resolve_face_paths(dir, pattern)?;

    let mut face_images: Vec<DecodedImageF32> = Vec::with_capacity(6);
    let mut face_size: Option<u32> = None;

    for (i, face_path) in face_paths.iter().enumerate() {
        let decoded = decode_environment_file(face_path)?;

        if decoded.width != decoded.height {
            return Err(format!(
                "Face {} file '{}' is not square ({}x{})",
                FACE_NAMES[i],
                face_path.display(),
                decoded.width,
                decoded.height
            ));
        }

        match face_size {
            None => face_size = Some(decoded.width),
            Some(expected) => {
                if decoded.width != expected {
                    return Err(format!(
                        "Face {} file '{}' has size {} but expected {} (mismatch with previous faces)",
                        FACE_NAMES[i],
                        face_path.display(),
                        decoded.width,
                        expected
                    ));
                }
            }
        }

        face_images.push(decoded);
    }

    let face_size = face_size.ok_or("No face images loaded")?;

    // Combine into contiguous face-major buffer
    let floats_per_face = (face_size * face_size * 4) as usize;
    let mut output = Vec::with_capacity(floats_per_face * 6);
    for face in &face_images {
        output.extend_from_slice(&face.rgba32f);
    }

    Ok(CubemapFaces {
        face_size,
        rgba32f: output,
    })
}

/// Resolve paths for all 6 cubemap faces from a directory.
fn resolve_face_paths(dir: &Path, pattern: FacePattern) -> Result<Vec<PathBuf>, String> {
    match pattern {
        FacePattern::PxNxPyNyPzNz => resolve_specific_aliases(dir, &FACE_ALIASES_PX),
        FacePattern::PosxNegxPosyNegyPoszNegz => resolve_specific_aliases(dir, &FACE_ALIASES_POSX),
        FacePattern::AutoAliases => resolve_auto_aliases(dir),
    }
}

/// Resolve faces using specific single-alias stems.
fn resolve_specific_aliases(dir: &Path, stems: &[&str; 6]) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::with_capacity(6);

    for (face_idx, stem) in stems.iter().enumerate() {
        let found = find_file_with_stem(dir, stem)?;
        match found {
            Some(path) => paths.push(path),
            None => {
                return Err(format!(
                    "Missing face {} file: no file matching '{}.*' in '{}'",
                    FACE_NAMES[face_idx],
                    stem,
                    dir.display()
                ));
            }
        }
    }

    Ok(paths)
}

/// Resolve faces using all known aliases; error on ambiguity.
fn resolve_auto_aliases(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::with_capacity(6);

    // List directory entries once
    let entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory '{}': {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .collect();

    for (face_idx, aliases) in FACE_ALIASES.iter().enumerate() {
        let mut candidates: Vec<PathBuf> = Vec::new();

        for entry in &entries {
            let file_name = entry.file_name();
            let name_str = file_name.to_string_lossy();

            // Extract stem (filename without extension), case-insensitive
            let stem = match name_str.rfind('.') {
                Some(dot_pos) => &name_str[..dot_pos],
                None => &name_str,
            };
            let stem_lower = stem.to_lowercase();

            if aliases.iter().any(|alias| stem_lower == *alias) {
                candidates.push(entry.path());
            }
        }

        match candidates.len() {
            0 => {
                return Err(format!(
                    "Missing face {} file: no file matching any of {:?} in '{}'",
                    FACE_NAMES[face_idx],
                    aliases,
                    dir.display()
                ));
            }
            1 => paths.push(candidates.into_iter().next().unwrap()),
            _ => {
                return Err(format!(
                    "Ambiguous face {} files: multiple candidates {:?} in '{}'",
                    FACE_NAMES[face_idx],
                    candidates,
                    dir.display()
                ));
            }
        }
    }

    Ok(paths)
}

/// Find a file in a directory whose stem (case-insensitive) matches the given stem.
fn find_file_with_stem(dir: &Path, target_stem: &str) -> Result<Option<PathBuf>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory '{}': {}", dir.display(), e))?;

    let target_lower = target_stem.to_lowercase();

    for entry in entries.filter_map(|e| e.ok()) {
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }

        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        let stem = match name_str.rfind('.') {
            Some(dot_pos) => &name_str[..dot_pos],
            None => &name_str,
        };

        if stem.to_lowercase() == target_lower {
            return Ok(Some(entry.path()));
        }
    }

    Ok(None)
}

// ─── GPU Format Selection ───────────────────────────────────────────────────

/// Pick preferred float format for environment textures.
pub fn pick_environment_float_format(
    supported: &std::collections::HashSet<vk::Format>,
) -> Result<vk::Format, String> {
    if supported.contains(&vk::Format::R16G16B16A16_SFLOAT) {
        Ok(vk::Format::R16G16B16A16_SFLOAT)
    } else if supported.contains(&vk::Format::R32G32B32A32_SFLOAT) {
        Ok(vk::Format::R32G32B32A32_SFLOAT)
    } else {
        Err("Device supports neither R16G16B16A16_SFLOAT nor R32G32B32A32_SFLOAT for environment maps".to_string())
    }
}

/// Convert RGBA32F float data to byte buffer in the target format.
pub fn rgba32f_to_format_bytes(rgba32f: &[f32], format: vk::Format) -> Result<Vec<u8>, String> {
    match format {
        vk::Format::R32G32B32A32_SFLOAT => Ok(bytemuck::cast_slice(rgba32f).to_vec()),
        vk::Format::R16G16B16A16_SFLOAT => {
            let half_data: Vec<u16> = rgba32f
                .iter()
                .map(|&f| half::f16::from_f32(f).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&half_data).to_vec())
        }
        _ => Err(format!(
            "Unsupported environment target format {:?}",
            format
        )),
    }
}

// ─── Unified Import ─────────────────────────────────────────────────────────

/// Public environment source specification.
#[derive(Debug)]
pub enum EnvironmentSource {
    /// Auto-detect layout from file dimensions.
    Auto(PathBuf),
    /// Equirectangular (2:1) HDR/EXR/image file.
    Equirectangular(PathBuf),
    /// Horizontal 6:1 cube strip file.
    CubeStrip(PathBuf),
    /// Directory of 6 face images.
    FaceDirectory { path: PathBuf, pattern: FacePattern },
}

/// Import an environment source into a `PendingSkyboxSource`.
pub fn import_environment_source(
    source: &EnvironmentSource,
    supported_formats: &std::collections::HashSet<vk::Format>,
) -> Result<PendingSkyboxSource, String> {
    let format = pick_environment_float_format(supported_formats)?;

    match source {
        EnvironmentSource::Auto(path) => {
            let decoded = decode_environment_file(path)?;
            let layout = detect_source_layout(decoded.width, decoded.height)?;

            match layout {
                SourceLayout::Equirectangular => make_equirect_source(decoded, format),
                SourceLayout::CubeStrip => {
                    let faces = convert_strip_to_face_major(&decoded)?;
                    make_cubemap_source(faces, format)
                }
            }
        }
        EnvironmentSource::Equirectangular(path) => {
            let decoded = decode_environment_file(path)?;
            make_equirect_source(decoded, format)
        }
        EnvironmentSource::CubeStrip(path) => {
            let decoded = decode_environment_file(path)?;
            if decoded.width != 6 * decoded.height {
                return Err(format!(
                    "CubeStrip source '{}' has dimensions {}x{} which is not 6:1",
                    path.display(),
                    decoded.width,
                    decoded.height
                ));
            }
            let faces = convert_strip_to_face_major(&decoded)?;
            make_cubemap_source(faces, format)
        }
        EnvironmentSource::FaceDirectory { path, pattern } => {
            let faces = load_face_directory(path, *pattern)?;
            make_cubemap_source(faces, format)
        }
    }
}

fn make_equirect_source(
    decoded: DecodedImageF32,
    format: vk::Format,
) -> Result<PendingSkyboxSource, String> {
    let bytes = rgba32f_to_format_bytes(&decoded.rgba32f, format)?;
    Ok(PendingSkyboxSource::Equirectangular2D {
        width: decoded.width,
        height: decoded.height,
        format,
        bytes,
    })
}

fn make_cubemap_source(
    faces: CubemapFaces,
    format: vk::Format,
) -> Result<PendingSkyboxSource, String> {
    let bytes = rgba32f_to_format_bytes(&faces.rgba32f, format)?;
    Ok(PendingSkyboxSource::CubemapFaces {
        face_size: faces.face_size,
        format,
        bytes,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use std::collections::HashSet;

    // ── Layout Detection ────────────────────────────────────────────────

    #[test]
    fn detect_equirectangular_from_2_to_1() {
        assert_eq!(
            detect_source_layout(4096, 2048).unwrap(),
            SourceLayout::Equirectangular
        );
        assert_eq!(
            detect_source_layout(2, 1).unwrap(),
            SourceLayout::Equirectangular
        );
    }

    #[test]
    fn detect_cube_strip_from_6_to_1() {
        assert_eq!(
            detect_source_layout(6144, 1024).unwrap(),
            SourceLayout::CubeStrip
        );
        assert_eq!(detect_source_layout(6, 1).unwrap(), SourceLayout::CubeStrip);
    }

    #[test]
    fn reject_unknown_aspect_ratio() {
        assert!(detect_source_layout(1024, 1024).is_err());
        assert!(detect_source_layout(1024, 512).is_ok()); // 2:1
        assert!(detect_source_layout(100, 99).is_err());
        assert!(detect_source_layout(3, 1).is_err());
    }

    // ── Strip Conversion ────────────────────────────────────────────────

    #[test]
    fn strip_conversion_face_major_order() {
        // Create a 6x1 strip (face_size=1, one pixel per face)
        let _face_size = 1u32;
        let width = 6;
        let height = 1;

        // Each face is one pixel with distinct color
        let mut rgba32f = Vec::new();
        for face_idx in 0..6 {
            rgba32f.push(face_idx as f32); // R
            rgba32f.push(0.0); // G
            rgba32f.push(0.0); // B
            rgba32f.push(1.0); // A
        }

        let decoded = DecodedImageF32 {
            width,
            height,
            rgba32f,
        };

        let faces = convert_strip_to_face_major(&decoded).unwrap();
        assert_eq!(faces.face_size, 1);
        assert_eq!(faces.rgba32f.len(), 6 * 4); // 6 faces * 4 floats

        // Verify face order: face i should have R = i
        for face_idx in 0..6 {
            assert_eq!(
                faces.rgba32f[face_idx * 4],
                face_idx as f32,
                "Face {} has wrong color",
                face_idx
            );
        }
    }

    #[test]
    fn strip_conversion_multi_pixel_faces() {
        // 2x2 faces in a 12x2 strip
        let face_size = 2u32;
        let width = 12;
        let height = 2;

        // Fill each pixel with (face_idx, row, col, 1.0)
        let mut rgba32f = vec![0.0f32; (width * height * 4) as usize];
        for row in 0..height {
            for col in 0..width {
                let face_idx = col / face_size;
                let local_col = col % face_size;
                let pixel_offset = ((row * width + col) * 4) as usize;
                rgba32f[pixel_offset] = face_idx as f32;
                rgba32f[pixel_offset + 1] = row as f32;
                rgba32f[pixel_offset + 2] = local_col as f32;
                rgba32f[pixel_offset + 3] = 1.0;
            }
        }

        let decoded = DecodedImageF32 {
            width,
            height,
            rgba32f,
        };

        let faces = convert_strip_to_face_major(&decoded).unwrap();
        assert_eq!(faces.face_size, 2);
        assert_eq!(faces.rgba32f.len(), 6 * 4 * 4); // 6 faces * 2*2 pixels * 4 floats

        // Check face 3 (+Y = face index 3), pixel (1, 0) => row=1, col=0
        // Audit (2026-07-03, AGR-009): convert_strip_to_face_major outputs
        // face-major, row-major order. For a 2×2 face, face 3 starts at
        // 3 * (face_size² * 4) = 3 * 16 = 48. Pixel (row=1, col=0) within
        // face 3 is at offset 48 + 1*(2*4) + 0 = 56. Verified correct.
        // The old `0 * 4` term was dead arithmetic (always zero); removed.
        let face3_offset = 3 * (4 * 4); // 3 faces * 4 pixels * 4 floats
        let pixel_offset = face3_offset + 1 * (2 * 4); // row=1 offset (1 * face_width * 4 floats) + col=0
        assert_eq!(faces.rgba32f[pixel_offset], 3.0); // face_idx
        assert_eq!(faces.rgba32f[pixel_offset + 1], 1.0); // row
        assert_eq!(faces.rgba32f[pixel_offset + 2], 0.0); // local_col
    }

    #[test]
    fn strip_conversion_rejects_non_6_to_1() {
        let decoded = DecodedImageF32 {
            width: 4,
            height: 2,
            rgba32f: vec![0.0; 4 * 2 * 4],
        };
        assert!(convert_strip_to_face_major(&decoded).is_err());
    }

    // ── Format Selection ────────────────────────────────────────────────

    #[test]
    fn pick_format_prefers_r16() {
        let mut supported = HashSet::new();
        supported.insert(vk::Format::R16G16B16A16_SFLOAT);
        supported.insert(vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(
            pick_environment_float_format(&supported).unwrap(),
            vk::Format::R16G16B16A16_SFLOAT
        );
    }

    #[test]
    fn pick_format_falls_back_to_r32() {
        let mut supported = HashSet::new();
        supported.insert(vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(
            pick_environment_float_format(&supported).unwrap(),
            vk::Format::R32G32B32A32_SFLOAT
        );
    }

    #[test]
    fn pick_format_fails_when_none_supported() {
        let supported = HashSet::new();
        assert!(pick_environment_float_format(&supported).is_err());
    }

    // ── Format Conversion ───────────────────────────────────────────────

    #[test]
    fn rgba32f_to_r32_bytes_roundtrip() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = rgba32f_to_format_bytes(&input, vk::Format::R32G32B32A32_SFLOAT).unwrap();
        let roundtrip: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(roundtrip, &input[..]);
    }

    #[test]
    fn rgba32f_to_r16_bytes_length() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = rgba32f_to_format_bytes(&input, vk::Format::R16G16B16A16_SFLOAT).unwrap();
        // 4 components * 2 bytes each = 8 bytes
        assert_eq!(bytes.len(), 8);
    }

    // ── NaN/Inf Sanitization ────────────────────────────────────────────

    #[test]
    fn sanitize_nan_and_inf() {
        let mut data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 42.0];
        sanitize_float_data(&mut data);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 65504.0);
        assert_eq!(data[2], -65504.0);
        assert_eq!(data[3], 42.0);
    }

    // ── Face Directory Alias Resolution ─────────────────────────────────

    #[test]
    fn face_directory_resolve_requires_existing_dir() {
        let result = load_face_directory(
            Path::new("/nonexistent/path/to/nowhere"),
            FacePattern::AutoAliases,
        );
        assert!(result.is_err());
    }

    #[test]
    fn face_directory_with_temp_dir() {
        let tmp = std::env::temp_dir().join("env_import_test_faces");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create 6 minimal 1x1 HDR-like PNG files with px/nx/py/ny/pz/nz names
        // We'll use PNG since the image crate can write those easily
        for name in &["px", "nx", "py", "ny", "pz", "nz"] {
            let path = tmp.join(format!("{}.png", name));
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(1, 1, Rgba([128, 128, 128, 255]));
            img.save(&path).unwrap();
        }

        let result = load_face_directory(&tmp, FacePattern::PxNxPyNyPzNz);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let faces = result.unwrap();
        assert_eq!(faces.face_size, 1);
        assert_eq!(faces.rgba32f.len(), 6 * 4); // 6 faces * 1 pixel * 4 floats

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn project_default_cc0_hdr_cubemap_is_spatial_and_high_dynamic_range() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/assets/sky_maps/cc0_dungeon_hdr");
        let faces = load_face_directory(&path, FacePattern::PxNxPyNyPzNz)
            .expect("project default HDR cubemap must decode");

        assert_eq!(faces.face_size, 256);
        assert_eq!(faces.rgba32f.len(), 6 * 256 * 256 * 4);

        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let floats_per_face = 256 * 256 * 4;
        for (face_index, face) in faces.rgba32f.chunks_exact(floats_per_face).enumerate() {
            let mut channel_min = [f32::INFINITY; 3];
            let mut channel_max = [f32::NEG_INFINITY; 3];
            for pixel in face.chunks_exact(4) {
                for channel in 0..3 {
                    channel_min[channel] = channel_min[channel].min(pixel[channel]);
                    channel_max[channel] = channel_max[channel].max(pixel[channel]);
                    minimum = minimum.min(pixel[channel]);
                    maximum = maximum.max(pixel[channel]);
                }
            }
            assert!(
                channel_max
                    .into_iter()
                    .zip(channel_min)
                    .any(|(max, min)| max - min > 0.005),
                "HDR face {face_index} is spatially flat"
            );
        }
        assert!(
            minimum < 0.1,
            "HDR cubemap lacks dark dungeon detail: {minimum}"
        );
        assert!(
            maximum > 8.0,
            "HDR cubemap lacks bright IBL highlights: {maximum}"
        );
    }

    #[test]
    fn face_directory_auto_aliases_detects_right_left_etc() {
        let tmp = std::env::temp_dir().join("env_import_test_auto_aliases");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Use alias names: right, left, top, bottom, front, back
        for name in &["right", "left", "top", "bottom", "front", "back"] {
            let path = tmp.join(format!("{}.png", name));
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(2, 2, Rgba([64, 64, 64, 255]));
            img.save(&path).unwrap();
        }

        let result = load_face_directory(&tmp, FacePattern::AutoAliases);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let faces = result.unwrap();
        assert_eq!(faces.face_size, 2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn face_directory_rejects_missing_face() {
        let tmp = std::env::temp_dir().join("env_import_test_missing_face");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Only create 5 of 6 faces
        for name in &["px", "nx", "py", "ny", "pz"] {
            let path = tmp.join(format!("{}.png", name));
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(1, 1, Rgba([128, 128, 128, 255]));
            img.save(&path).unwrap();
        }

        let result = load_face_directory(&tmp, FacePattern::PxNxPyNyPzNz);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn face_directory_rejects_duplicate_aliases() {
        let tmp = std::env::temp_dir().join("env_import_test_duplicate");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create both "px.png" and "right.png" (both alias to +X)
        for name in &["px", "right", "nx", "py", "ny", "pz", "nz"] {
            let path = tmp.join(format!("{}.png", name));
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(1, 1, Rgba([128, 128, 128, 255]));
            img.save(&path).unwrap();
        }

        let result = load_face_directory(&tmp, FacePattern::AutoAliases);
        assert!(result.is_err());
        assert!(
            result.as_ref().unwrap_err().contains("Ambiguous"),
            "Expected ambiguity error, got: {:?}",
            result.err()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn face_directory_rejects_mismatched_dimensions() {
        let tmp = std::env::temp_dir().join("env_import_test_mismatch");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // 5 faces at 2x2, one at 3x3
        for name in &["px", "nx", "py", "ny", "pz"] {
            let path = tmp.join(format!("{}.png", name));
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(2, 2, Rgba([128, 128, 128, 255]));
            img.save(&path).unwrap();
        }
        {
            let path = tmp.join("nz.png");
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_pixel(3, 3, Rgba([128, 128, 128, 255]));
            img.save(&path).unwrap();
        }

        let result = load_face_directory(&tmp, FacePattern::PxNxPyNyPzNz);
        assert!(result.is_err());
        assert!(
            result.as_ref().unwrap_err().contains("mismatch"),
            "Expected mismatch error, got: {:?}",
            result.err()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
