use crate::data::gpu_data;
use crate::data::gpu_data::{MaterialMeta, MeshMeta, MetRoughUniform, Sampler, SurfaceMeta, TextureMeta, Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer};
use crate::vulkan::vk_types::{VkBuffer, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use glam::{vec4, Vec4};
use std::collections::HashMap;
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
