use std::cmp::max;
use crate::data::gpu_data;
use crate::data::gpu_data::{MaterialMeta, MeshMeta, MetRoughUniform, Sampler, SurfaceMeta, TextureMeta, Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer};
use crate::vulkan::vk_types::{VkBuffer, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use glam::{vec4, Vec4};
use std::collections::HashMap;
use half::f16;
use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use vk_mem::Alloc;

pub const EXTENT3D_ONE: vk::Extent3D = vk::Extent3D {
    width: 1,
    height: 1,
    depth: 1,
};

pub trait PackUnorm {
    fn pack_unorm_4x8(&self) -> u32;
}

impl PackUnorm for Vec4 {
    fn pack_unorm_4x8(&self) -> u32 {
        let x = (self.x.clamp(0.0, 1.0) * 255.0).round() as u32;
        let y = (self.y.clamp(0.0, 1.0) * 255.0).round() as u32;
        let z = (self.z.clamp(0.0, 1.0) * 255.0).round() as u32;
        let w = (self.w.clamp(0.0, 1.0) * 255.0).round() as u32;

        (x << 0) | (y << 8) | (z << 16) | (w << 24)
    }
}

pub fn convert_rgb32f_to_rgba32f(img: ImageBuffer<Rgb<f32>, Vec<f32>>) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();

    ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = img.get_pixel(x, y);
        Rgba([pixel[0], pixel[1], pixel[2], 1.0])
    })
}

pub fn calc_mips_count(width: u32, height: u32) -> u32 {
    let max_dimension = max(width, height) as f64;
    (max_dimension.log2().floor() as u32) + 1
}

pub fn bytes_per_pixel(format : vk::Format) -> u32 {
    match format {
        vk::Format::R8_UNORM => 1,
        vk::Format::R8G8_UNORM => 2,
        vk::Format::R8G8B8_UNORM => 3,
        vk::Format::R8G8B8A8_UNORM => 4,
        vk::Format::R16_SFLOAT => 2,
        vk::Format::R16G16_SFLOAT => 4,
        vk::Format::R16G16B16_SFLOAT => 6,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R32_SFLOAT => 4,
        vk::Format::R32G32_SFLOAT => 8,
        vk::Format::R32G32B32_SFLOAT => 12,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => panic!("Cannot calculate bytes per pixel: Unsupported format")
    }
}
