//! # Directional Shadow Map Resources
//!
//! Manages per-frame-in-flight shadow map images, image views, and samplers
//! for directional light shadow mapping. The shadow map is a single 2048² D32
//! depth image rendered from the light's perspective each frame.

use ash::vk;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

use crate::vulkan::vk_types::{VkDestroyable, VkImageAlloc};
use crate::vulkan::vk_util;

/// Per-frame shadow map resources.
pub struct VkShadowFrame {
    pub shadow_map: VkImageAlloc,
    pub shadow_map_view: vk::ImageView,
    pub shadow_sampler: vk::Sampler,
}

impl VkShadowFrame {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_image_view(self.shadow_map_view, None);
            device.destroy_sampler(self.shadow_sampler, None);
        }
        VkDestroyable::destroy(&mut self.shadow_map, device, allocator);
    }
}

/// Collection of per-frame shadow resources for all frames in flight.
pub struct VkShadowResources {
    pub frames: Vec<VkShadowFrame>,
    pub shadow_map_extent: vk::Extent2D,
}

impl VkShadowResources {
    /// Shadow map resolution.
    pub const SHADOW_MAP_DIM: u32 = 2048;

    /// Create per-frame shadow resources.
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        frame_count: u32,
    ) -> Result<Self, String> {
        let allocator = allocator.lock().map_err(|e| format!("allocator lock: {e}"))?;
        let shadow_map_extent = vk::Extent2D {
            width: Self::SHADOW_MAP_DIM,
            height: Self::SHADOW_MAP_DIM,
        };

        let mut frames = Vec::with_capacity(frame_count as usize);
        for _ in 0..frame_count {
            let shadow_map = vk_util::create_image(
                device,
                &allocator,
                vk::Extent3D {
                    width: Self::SHADOW_MAP_DIM,
                    height: Self::SHADOW_MAP_DIM,
                    depth: 1,
                },
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                1,
            );

            let view_info = vk::ImageViewCreateInfo::default()
                .image(shadow_map.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let shadow_map_view = unsafe { device.create_image_view(&view_info, None) }
                .map_err(|e| format!("failed to create shadow map view: {e:?}"))?;

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .max_lod(1.0)
                .min_lod(0.0);

            let shadow_sampler = unsafe { device.create_sampler(&sampler_info, None) }
                .map_err(|e| format!("failed to create shadow sampler: {e:?}"))?;

            frames.push(VkShadowFrame {
                shadow_map,
                shadow_map_view,
                shadow_sampler,
            });
        }

        Ok(Self {
            frames,
            shadow_map_extent,
        })
    }

    /// Get shadow map view and sampler for a given frame index.
    pub fn get_frame(&self, frame_index: u32) -> &VkShadowFrame {
        &self.frames[frame_index as usize]
    }

    /// Destroy all shadow resources.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        for frame in &mut self.frames {
            frame.destroy(device, allocator);
        }
        self.frames.clear();
    }
}

/// Compute a light view-projection matrix from a directional light vector and
/// a conservative scene bounding box.
///
/// Returns `(view, projection, view_projection)` where:
/// - `view` looks along the light direction from a distance far enough to
///   encompass the whole AABB.
/// - `projection` is an orthographic frustum covering the AABB extent.
/// - `view_projection = projection * view`.
pub fn compute_light_view_projection(
    light_dir: glam::Vec3,
    scene_aabb_min: glam::Vec3,
    scene_aabb_max: glam::Vec3,
) -> (glam::Mat4, glam::Mat4, glam::Mat4) {
    let center = (scene_aabb_min + scene_aabb_max) * 0.5;
    let extent = (scene_aabb_max - scene_aabb_min).abs();
    let radius = extent.length() * 0.5;

    // Place the light far enough back so the whole scene fits in its frustum.
    let light_pos = center + light_dir.normalize() * radius;
    let up = if light_dir.x.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let view = glam::Mat4::look_at_rh(light_pos, center, up);

    // Orthographic projection covering the AABB extent with a margin.
    let margin = radius * 0.2;
    let half = radius + margin;
    let projection = glam::Mat4::orthographic_rh(-half, half, -half, half, 0.01, radius * 3.0);

    let view_projection = projection * view;

    (view, projection, view_projection)
}
