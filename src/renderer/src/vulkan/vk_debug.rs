//! # Vulkan Debug Capture Helpers
//!
//! Utilities for copying GPU images to CPU and writing debug snapshots to disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::api::config::{CaptureTarget, FrameCaptureSource};
use crate::vulkan::vk_types::{VkBuffer, VkDestroyable};
use crate::vulkan::vk_util;
use ash::vk;
use half::f16;
use image::{ImageBuffer, Rgba};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct FrameCaptureTargetDesc {
    pub target: CaptureTarget,
    pub image: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub current_layout: vk::ImageLayout,
    pub restored_layout: vk::ImageLayout,
}

#[derive(Debug)]
pub struct PendingFrameCapture {
    pub frame_number: u32,
    pub sequence_index: Option<u32>,
    pub source: FrameCaptureSource,
    pub target: CaptureTarget,
    pub output_path: PathBuf,
    pub sidecar_path: Option<PathBuf>,
    pub source_format: vk::Format,
    pub color_conversion: &'static str,
    pub extent: vk::Extent2D,
    readback: VkBuffer,
}

#[derive(Debug, Clone)]
pub struct FrameCaptureReport {
    pub frame_number: u32,
    pub target: CaptureTarget,
    pub output_path: PathBuf,
    pub sidecar_path: Option<PathBuf>,
    pub source: FrameCaptureSource,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct FrameCaptureError {
    pub message: String,
}

impl FrameCaptureError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FrameCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for FrameCaptureError {}

pub fn record_frame_capture(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    cmd_buffer: vk::CommandBuffer,
    frame_number: u32,
    sequence_index: Option<u32>,
    source: FrameCaptureSource,
    output_path: &Path,
    sidecar_path: Option<&Path>,
    target_desc: FrameCaptureTargetDesc,
) -> Result<PendingFrameCapture, FrameCaptureError> {
    if target_desc.extent.width == 0 || target_desc.extent.height == 0 {
        return Err(FrameCaptureError::new("capture source extent is zero"));
    }

    let format_info = CaptureFormatInfo::from_vk_format(target_desc.format)?;
    let buffer_size = format_info.buffer_size(target_desc.extent)?;
    let readback = vk_util::allocate_buffer(
        allocator,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk_mem::MemoryUsage::AutoPreferHost,
    )
    .map_err(FrameCaptureError::new)?;

    vk_util::transition_image(
        device,
        cmd_buffer,
        target_desc.image,
        target_desc.current_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );

    let copy_region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: target_desc.extent.width,
            height: target_desc.extent.height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd_buffer,
            target_desc.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.buffer,
            &[copy_region],
        );
    }

    vk_util::transition_image(
        device,
        cmd_buffer,
        target_desc.image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        target_desc.restored_layout,
    );

    Ok(PendingFrameCapture {
        frame_number,
        sequence_index,
        source,
        target: target_desc.target,
        output_path: output_path.to_path_buf(),
        sidecar_path: sidecar_path.map(Path::to_path_buf),
        source_format: target_desc.format,
        color_conversion: format_info.color_conversion,
        extent: target_desc.extent,
        readback,
    })
}

pub fn finalize_frame_capture(
    device: &ash::Device,
    allocator: &vk_mem::Allocator,
    mut pending: PendingFrameCapture,
) -> Result<FrameCaptureReport, FrameCaptureError> {
    let result = finalize_frame_capture_inner(allocator, &pending);
    pending.readback.destroy(device, allocator);
    result
}

fn finalize_frame_capture_inner(
    allocator: &vk_mem::Allocator,
    pending: &PendingFrameCapture,
) -> Result<FrameCaptureReport, FrameCaptureError> {
    allocator
        .invalidate_allocation(&pending.readback.allocation, 0, vk::WHOLE_SIZE)
        .map_err(|err| {
            FrameCaptureError::new(format!("failed to invalidate readback memory: {err:?}"))
        })?;

    let mapped_data = pending.readback.alloc_info.mapped_data;
    if mapped_data.is_null() {
        return Err(FrameCaptureError::new("readback allocation was not mapped"));
    }

    let format_info = CaptureFormatInfo::from_vk_format(pending.source_format)?;
    let byte_len = format_info.buffer_size(pending.extent)? as usize;
    let raw = unsafe { std::slice::from_raw_parts(mapped_data as *const u8, byte_len) };
    let rgba = format_info.convert_to_rgba(raw, pending.extent)?;

    write_png(&pending.output_path, pending.extent, rgba)?;
    if let Some(sidecar_path) = pending.sidecar_path.as_ref() {
        write_sidecar(sidecar_path, pending)?;
    }

    let report = FrameCaptureReport {
        frame_number: pending.frame_number,
        target: pending.target,
        output_path: pending.output_path.clone(),
        sidecar_path: pending.sidecar_path.clone(),
        source: pending.source,
        width: pending.extent.width,
        height: pending.extent.height,
    };
    Ok(report)
}

fn write_png(
    output_path: &Path,
    extent: vk::Extent2D,
    rgba: Vec<u8>,
) -> Result<(), FrameCaptureError> {
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            FrameCaptureError::new(format!(
                "failed to create capture directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(extent.width, extent.height, rgba)
        .ok_or_else(|| {
            FrameCaptureError::new("converted RGBA buffer did not match capture dimensions")
        })?;
    image.save(output_path).map_err(|err| {
        FrameCaptureError::new(format!(
            "failed to save PNG {}: {err}",
            output_path.display()
        ))
    })
}

fn write_sidecar(path: &Path, pending: &PendingFrameCapture) -> Result<(), FrameCaptureError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            FrameCaptureError::new(format!(
                "failed to create sidecar directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let captured_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let sidecar = json!({
        "status": "succeeded",
        "frame_number": pending.frame_number,
        "sequence_index": pending.sequence_index,
        "capture_target": pending.target.as_label(),
        "source": format!("{:?}", pending.source),
        "png_path": pending.output_path,
        "extent": {
            "width": pending.extent.width,
            "height": pending.extent.height,
        },
        "format": format!("{:?}", pending.source_format),
        "color_conversion": pending.color_conversion,
        "row_layout": "vkCmdCopyImageToBuffer tightly packed (buffer_row_length=0)",
        "captured_at_unix_ms": captured_at_unix_ms,
    });

    let bytes = serde_json::to_vec_pretty(&sidecar)
        .map_err(|err| FrameCaptureError::new(format!("failed to serialize sidecar: {err}")))?;
    fs::write(path, bytes).map_err(|err| {
        FrameCaptureError::new(format!("failed to write sidecar {}: {err}", path.display()))
    })
}

#[derive(Debug, Copy, Clone)]
struct CaptureFormatInfo {
    bytes_per_pixel: u64,
    color_conversion: &'static str,
    convert: fn(&[u8], vk::Extent2D) -> Result<Vec<u8>, FrameCaptureError>,
}

impl CaptureFormatInfo {
    fn from_vk_format(format: vk::Format) -> Result<Self, FrameCaptureError> {
        match format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Ok(Self {
                bytes_per_pixel: 4,
                color_conversion: "bgra8-to-rgba8",
                convert: convert_bgra8_to_rgba8,
            }),
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Ok(Self {
                bytes_per_pixel: 4,
                color_conversion: "rgba8",
                convert: convert_rgba8_to_rgba8,
            }),
            vk::Format::R16G16B16A16_SFLOAT => Ok(Self {
                bytes_per_pixel: 8,
                color_conversion: "rgba16f-linear-clamped-to-rgba8",
                convert: convert_rgba16f_to_rgba8,
            }),
            other => Err(FrameCaptureError::new(format!(
                "unsupported capture source format {other:?}"
            ))),
        }
    }

    fn buffer_size(self, extent: vk::Extent2D) -> Result<u64, FrameCaptureError> {
        let pixels = (extent.width as u64)
            .checked_mul(extent.height as u64)
            .ok_or_else(|| FrameCaptureError::new("capture dimensions overflowed"))?;
        pixels
            .checked_mul(self.bytes_per_pixel)
            .ok_or_else(|| FrameCaptureError::new("capture buffer size overflowed"))
    }

    fn convert_to_rgba(
        self,
        raw: &[u8],
        extent: vk::Extent2D,
    ) -> Result<Vec<u8>, FrameCaptureError> {
        (self.convert)(raw, extent)
    }
}

fn expected_len(extent: vk::Extent2D, bytes_per_pixel: usize) -> Result<usize, FrameCaptureError> {
    (extent.width as usize)
        .checked_mul(extent.height as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| FrameCaptureError::new("capture buffer size overflowed"))
}

fn convert_rgba8_to_rgba8(raw: &[u8], extent: vk::Extent2D) -> Result<Vec<u8>, FrameCaptureError> {
    let expected = expected_len(extent, 4)?;
    if raw.len() != expected {
        return Err(FrameCaptureError::new(
            "readback RGBA8 byte length mismatch",
        ));
    }
    Ok(raw.to_vec())
}

fn convert_bgra8_to_rgba8(raw: &[u8], extent: vk::Extent2D) -> Result<Vec<u8>, FrameCaptureError> {
    let expected = expected_len(extent, 4)?;
    if raw.len() != expected {
        return Err(FrameCaptureError::new(
            "readback BGRA8 byte length mismatch",
        ));
    }

    let mut rgba = Vec::with_capacity(raw.len());
    for px in raw.chunks_exact(4) {
        rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    Ok(rgba)
}

fn convert_rgba16f_to_rgba8(
    raw: &[u8],
    extent: vk::Extent2D,
) -> Result<Vec<u8>, FrameCaptureError> {
    let expected = expected_len(extent, 8)?;
    if raw.len() != expected {
        return Err(FrameCaptureError::new(
            "readback RGBA16F byte length mismatch",
        ));
    }

    let mut rgba = Vec::with_capacity(expected / 2);
    for px in raw.chunks_exact(8) {
        for channel in 0..4 {
            let offset = channel * 2;
            let bits = u16::from_le_bytes([px[offset], px[offset + 1]]);
            let value = f16::from_bits(bits).to_f32();
            rgba.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    Ok(rgba)
}
