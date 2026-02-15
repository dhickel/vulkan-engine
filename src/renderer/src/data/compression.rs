use crate::api::config::{CompressionConfig, TextureCompressionMode};
use crate::data::gpu_data::{TexturePayload, TextureSemantic};
use ash::vk;
use image::DynamicImage;
use intel_tex_2::{bc1, bc3, bc4, bc5, bc7};

pub struct CompressionDecision {
    pub format: vk::Format,
    pub quality: u8,
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
) -> Option<CompressionDecision> {
    if config.mode == TextureCompressionMode::Disabled {
        return None;
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
        _ => return None,
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
        TextureSemantic::MetallicRoughness | TextureSemantic::Occlusion => {
            vk::Format::BC5_UNORM_BLOCK
        }
    };

    Some(CompressionDecision {
        format: target_format,
        quality: config.quality,
    })
}

pub fn compress_texture(
    width: u32,
    height: u32,
    bytes: &[u8],
    semantic: TextureSemantic,
    config: &CompressionConfig,
    generate_mips: bool,
    source_format: vk::Format,
) -> Result<Option<TexturePayload>, String> {
    let Some(decision) = decide_compression(semantic, source_format, config) else {
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

        // 2. Swizzle for BC5 if needed
        if decision.format == vk::Format::BC5_UNORM_BLOCK
            || decision.format == vk::Format::BC5_SNORM_BLOCK
        {
            if semantic == TextureSemantic::MetallicRoughness {
                // glTF: G=Roughness, B=Metallic.
                // BC5 encodes R and G.
                // We want R=Roughness, G=Metallic (or vice versa, match shader expectation).
                // Let's assume shader will read .r for Roughness, .g for Metallic.
                // So input R <- G, Input G <- B.
                for pixel in rgba_bytes.chunks_exact_mut(4) {
                    let r_val = pixel[1]; // G (Roughness)
                    let g_val = pixel[2]; // B (Metallic)
                    pixel[0] = r_val;
                    pixel[1] = g_val;
                }
            }
        }

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
                // Note: We already swizzled RGBA in place if needed (MetallicRoughness).
                // Now we just pack them into RG buffer.
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
