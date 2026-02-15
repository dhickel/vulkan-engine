use crate::api::config::{AssetPolicyConfig, CompressionConfig, TextureCompressionMode};
use crate::data::gpu_data::{TextureMeta, TexturePayload, TextureSemantic};
use ash::vk;
use intel_tex_2::{bc1, bc3, bc4, bc5, bc7};
use log::debug;
use std::collections::HashSet;

#[derive(Debug)]
pub struct CompressionDecision {
    pub format: vk::Format,
    pub quality: u8,
}

pub fn apply_compression_policy(
    mut meta: TextureMeta,
    semantic: TextureSemantic,
    config: &AssetPolicyConfig,
    supported_formats: &HashSet<vk::Format>,
) -> TextureMeta {
    let (width, height, bytes, format) = match &meta.payload {
        TexturePayload::Raw {
            width,
            height,
            bytes,
            format,
            ..
        } => (*width, *height, bytes, *format),
        _ => return meta,
    };

    match compress_texture(
        width,
        height,
        bytes,
        semantic,
        &config.compression,
        config.compression.mode != TextureCompressionMode::Disabled,
        format,
        supported_formats,
    ) {
        Ok(Some(compressed_payload)) => {
            debug!(
                "Compressed texture {:?} to {:?}",
                semantic,
                compressed_payload.format()
            );
            meta.payload = compressed_payload;
            meta
        }
        Ok(None) => meta,
        Err(e) => {
            log::warn!("Texture compression failed for {:?}: {}", semantic, e);
            meta
        }
    }
}

fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::R8_SRGB
    )
}

pub fn decide_compression(
    semantic: TextureSemantic,
    source_format: vk::Format,
    config: &CompressionConfig,
    supported_formats: &HashSet<vk::Format>,
) -> Result<Option<CompressionDecision>, String> {
    if config.mode == TextureCompressionMode::Disabled {
        return Ok(None);
    }

    // Only compress standard 8-bit unorm/srgb formats
    match source_format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SRGB
        | vk::Format::R8G8_UNORM
        | vk::Format::R8_UNORM
        | vk::Format::R8_SRGB => {}
        _ => {
            return match config.mode {
                TextureCompressionMode::Auto => Ok(None),
                TextureCompressionMode::Force => Err(format!(
                    "Texture compression forced, but source format {:?} is not supported",
                    source_format
                )),
                TextureCompressionMode::Disabled => Ok(None),
            };
        }
    }

    let target_format = match semantic {
        TextureSemantic::BaseColor | TextureSemantic::Emissive | TextureSemantic::Generic => {
            if is_srgb(source_format) {
                vk::Format::BC7_SRGB_BLOCK
            } else {
                vk::Format::BC7_UNORM_BLOCK
            }
        }
        TextureSemantic::Normal => vk::Format::BC5_UNORM_BLOCK,
        TextureSemantic::MetallicRoughness => vk::Format::BC5_UNORM_BLOCK,
        TextureSemantic::Occlusion => vk::Format::BC4_UNORM_BLOCK,
    };

    if !supported_formats.contains(&target_format) {
        return match config.mode {
            TextureCompressionMode::Auto => Ok(None),
            TextureCompressionMode::Force => Err(format!(
                "Texture compression forced for {:?}, but target format {:?} is unsupported by this device",
                semantic, target_format
            )),
            TextureCompressionMode::Disabled => Ok(None),
        };
    }

    Ok(Some(CompressionDecision {
        format: target_format,
        quality: config.quality,
    }))
}

pub fn compress_texture(
    width: u32,
    height: u32,
    bytes: &[u8],
    semantic: TextureSemantic,
    config: &CompressionConfig,
    generate_mips: bool,
    source_format: vk::Format,
    supported_formats: &HashSet<vk::Format>,
) -> Result<Option<TexturePayload>, String> {
    let Some(decision) =
        decide_compression(semantic, source_format, config, supported_formats)?
    else {
        return Ok(None);
    };

    // 1. Generate mips on CPU
    let mut mips = Vec::new();
    
    // Convert input bytes to Rgba8 for processing.
    // We assume input is R8G8B8A8_UNORM/SRGB based on decide_compression check.
    // If it's R8, we need to expand? decide_compression only returns Some for 8-bit formats.
    // But we need to handle the input format correctly to get Rgba8.

    let base_image = match source_format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
            if bytes.len() != (width * height * 4) as usize {
                return Err(format!("Input bytes length {} does not match R8G8B8A8 dimensions {}x{}", bytes.len(), width, height));
            }
            image::RgbaImage::from_raw(width, height, bytes.to_vec()).ok_or("Failed to create image from raw bytes")?
        }
        vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SRGB => {
             // Expand RGB to RGBA
             if bytes.len() != (width * height * 3) as usize {
                 return Err("Input bytes length mismatch for RGB".to_string());
             }
             let img = image::RgbImage::from_raw(width, height, bytes.to_vec()).ok_or("Failed to create RGB image")?;
             image::DynamicImage::ImageRgb8(img).to_rgba8()
        }
        vk::Format::R8_UNORM | vk::Format::R8_SRGB => {
             // Expand R to RGBA (Luma)
             if bytes.len() != (width * height) as usize {
                 return Err("Input bytes length mismatch for R8".to_string());
             }
             let img = image::GrayImage::from_raw(width, height, bytes.to_vec()).ok_or("Failed to create Gray image")?;
             image::DynamicImage::ImageLuma8(img).to_rgba8()
        }
        vk::Format::R8G8_UNORM => {
             // Expand RG to RGBA (LumaAlpha)
             if bytes.len() != (width * height * 2) as usize {
                 return Err("Input bytes length mismatch for R8G8".to_string());
             }
             let img = image::GrayAlphaImage::from_raw(width, height, bytes.to_vec()).ok_or("Failed to create GrayAlpha image")?;
             image::DynamicImage::ImageLumaA8(img).to_rgba8()
        }
        _ => return Err(format!("Unsupported source format for compression input: {:?}", source_format)),
    };
    
    mips.push(base_image.clone());

    if generate_mips {
        let mut w = width;
        let mut h = height;
        let mut current_mip = image::DynamicImage::ImageRgba8(base_image);

        while w > 1 || h > 1 {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            current_mip = current_mip.resize(w, h, image::imageops::FilterType::Triangle);
            mips.push(current_mip.to_rgba8());
        }
    }

    let mips_count = mips.len() as u32;
    let mut all_compressed_bytes = Vec::new();
    let mut mip_offsets = Vec::new();

    for mip in mips {
        let mip_width = mip.width();
        let mip_height = mip.height();
        let mut rgba_bytes = mip.into_raw();

        // Metallic-roughness textures are normalized to R=roughness, G=metalness upstream.
        // BC5 encoding simply consumes the RG channels in that canonical layout.

        let compressed_data = match decision.format {
            vk::Format::BC1_RGB_UNORM_BLOCK | vk::Format::BC1_RGB_SRGB_BLOCK => {
                let surface = intel_tex_2::RgbaSurface {
                    width: mip_width,
                    height: mip_height,
                    stride: mip_width * 4,
                    data: &rgba_bytes,
                };
                bc1::compress_blocks(&surface)
            }
            vk::Format::BC3_UNORM_BLOCK | vk::Format::BC3_SRGB_BLOCK => {
                let surface = intel_tex_2::RgbaSurface {
                    width: mip_width,
                    height: mip_height,
                    stride: mip_width * 4,
                    data: &rgba_bytes,
                };
                bc3::compress_blocks(&surface)
            }
            vk::Format::BC4_UNORM_BLOCK | vk::Format::BC4_SNORM_BLOCK => {
                // Extract Red channel for BC4
                let r_bytes: Vec<u8> = rgba_bytes.chunks_exact(4).map(|p| p[0]).collect();
                let surface = intel_tex_2::RSurface {
                    width: mip_width,
                    height: mip_height,
                    stride: mip_width,
                    data: &r_bytes,
                };
                bc4::compress_blocks(&surface)
            }
            vk::Format::BC5_UNORM_BLOCK | vk::Format::BC5_SNORM_BLOCK => {
                // Extract RG channels for BC5
                // Metallic-roughness data is pre-normalized to RG layout before compression.
                let rg_bytes: Vec<u8> = rgba_bytes.chunks_exact(4).flat_map(|p| [p[0], p[1]]).collect();
                let surface = intel_tex_2::RgSurface {
                    width: mip_width,
                    height: mip_height,
                    stride: mip_width * 2,
                    data: &rg_bytes,
                };
                bc5::compress_blocks(&surface)
            }
            vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => {
                let surface = intel_tex_2::RgbaSurface {
                    width: mip_width,
                    height: mip_height,
                    stride: mip_width * 4,
                    data: &rgba_bytes,
                };
                let settings = if decision.quality < 30 {
                     intel_tex_2::bc7::opaque_ultra_fast_settings()
                } else if decision.quality < 70 {
                     intel_tex_2::bc7::opaque_fast_settings()
                } else {
                     intel_tex_2::bc7::opaque_basic_settings()
                };
                bc7::compress_blocks(&settings, &surface)
            }
            _ => return Err(format!("Unsupported compression format: {:?}", decision.format)),
        };

        mip_offsets.push(all_compressed_bytes.len() as u32);
        all_compressed_bytes.extend_from_slice(&compressed_data);
    }

    Ok(Some(TexturePayload::Compressed {
        bytes: all_compressed_bytes,
        width,
        height,
        format: decision.format,
        mips_levels: mips_count,
        mip_offsets,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mode: TextureCompressionMode) -> CompressionConfig {
        CompressionConfig { mode, quality: 50 }
    }

    #[test]
    fn choose_bc4_for_occlusion_when_supported() {
        let mut supported = HashSet::new();
        supported.insert(vk::Format::BC4_UNORM_BLOCK);

        let decision = decide_compression(
            TextureSemantic::Occlusion,
            vk::Format::R8_UNORM,
            &cfg(TextureCompressionMode::Auto),
            &supported,
        )
        .expect("decision should not error in auto mode")
        .expect("occlusion should choose compression when supported");

        assert_eq!(decision.format, vk::Format::BC4_UNORM_BLOCK);
    }

    #[test]
    fn auto_mode_falls_back_when_target_unsupported() {
        let supported = HashSet::new();
        let decision = decide_compression(
            TextureSemantic::Normal,
            vk::Format::R8G8B8A8_UNORM,
            &cfg(TextureCompressionMode::Auto),
            &supported,
        )
        .expect("auto mode should not hard fail");
        assert!(decision.is_none());
    }

    #[test]
    fn force_mode_errors_when_target_unsupported() {
        let supported = HashSet::new();
        let err = decide_compression(
            TextureSemantic::Normal,
            vk::Format::R8G8B8A8_UNORM,
            &cfg(TextureCompressionMode::Force),
            &supported,
        )
        .expect_err("force mode must fail when target is unsupported");

        assert!(err.contains("unsupported"));
    }
}
