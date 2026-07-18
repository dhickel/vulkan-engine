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

        let mut frames: Vec<VkShadowFrame> = Vec::with_capacity(frame_count as usize);
        for _ in 0..frame_count {
            let shadow_map = match vk_util::create_image(
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
            ) {
                Ok(image) => image,
                Err(err) => {
                    for frame in &mut frames {
                        frame.destroy(device, &allocator);
                    }
                    return Err(err);
                }
            };

            let shadow_map_view = shadow_map.image_view;

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .max_lod(0.0)
                .min_lod(0.0);

            let shadow_sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
                Ok(sampler) => sampler,
                Err(err) => {
                    let mut shadow_map = shadow_map;
                    VkDestroyable::destroy(&mut shadow_map, device, &allocator);
                    for frame in &mut frames {
                        frame.destroy(device, &allocator);
                    }
                    return Err(format!("failed to create shadow sampler: {err:?}"));
                }
            };

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
    let radius = (extent.length() * 0.5).max(0.5);
    let direction_to_light = light_dir.normalize();

    // DirectionalLight::direction points from the scene toward the light.
    let light_pos = center + direction_to_light * (radius * 2.0 + 1.0);
    let up = if direction_to_light.dot(glam::Vec3::Y).abs() < 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let view = glam::Mat4::look_at_rh(light_pos, center, up);

    let corners = aabb_corners(scene_aabb_min, scene_aabb_max);
    let (view_min, view_max) = corners.iter().fold(
        (
            glam::Vec3::splat(f32::INFINITY),
            glam::Vec3::splat(f32::NEG_INFINITY),
        ),
        |(min, max), corner| {
            let view_corner = view.transform_point3(*corner);
            (min.min(view_corner), max.max(view_corner))
        },
    );

    let xy_margin = (view_max - view_min).truncate().length().max(1.0) * 0.05;
    let depth_margin = radius * 0.1 + 0.1;
    let near = (-view_max.z - depth_margin).max(0.01);
    let far = (-view_min.z + depth_margin).max(near + 0.01);
    let projection = glam::Mat4::orthographic_rh(
        view_min.x - xy_margin,
        view_max.x + xy_margin,
        view_min.y - xy_margin,
        view_max.y + xy_margin,
        near,
        far,
    );

    let view_projection = projection * view;
    (view, projection, view_projection)
}

pub fn compute_draw_light_view_projection<'a>(
    light_dir: glam::Vec3,
    draws: impl IntoIterator<Item = &'a crate::data::gpu_data::RenderObject>,
) -> Option<glam::Mat4> {
    let (scene_min, scene_max, count) = draws.into_iter().fold(
        (
            glam::Vec3::splat(f32::INFINITY),
            glam::Vec3::splat(f32::NEG_INFINITY),
            0usize,
        ),
        |(min, max, count), draw| {
            aabb_corners(draw.bounds_min, draw.bounds_max)
                .into_iter()
                .map(|corner| draw.transform.transform_point3(corner))
                .fold((min, max, count + 1), |(min, max, count), point| {
                    (min.min(point), max.max(point), count)
                })
        },
    );

    (count > 0).then(|| compute_light_view_projection(light_dir, scene_min, scene_max).2)
}

fn aabb_corners(min: glam::Vec3, max: glam::Vec3) -> [glam::Vec3; 8] {
    [
        glam::Vec3::new(min.x, min.y, min.z),
        glam::Vec3::new(max.x, min.y, min.z),
        glam::Vec3::new(min.x, max.y, min.z),
        glam::Vec3::new(max.x, max.y, min.z),
        glam::Vec3::new(min.x, min.y, max.z),
        glam::Vec3::new(max.x, min.y, max.z),
        glam::Vec3::new(min.x, max.y, max.z),
        glam::Vec3::new(max.x, max.y, max.z),
    ]
}

#[cfg(test)]
mod tests {
    use super::{aabb_corners, compute_light_view_projection};
    use glam::Vec3;

    #[test]
    fn vertical_direction_produces_finite_matrix_covering_scene_bounds() {
        let min = Vec3::new(-4.0, -1.0, -3.0);
        let max = Vec3::new(7.0, 5.0, 2.0);
        let (_, _, view_projection) =
            compute_light_view_projection(Vec3::Y, min, max);

        assert!(view_projection.is_finite());
        for corner in aabb_corners(min, max) {
            let clip = view_projection * corner.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!((-1.001..=1.001).contains(&ndc.x), "x={}", ndc.x);
            assert!((-1.001..=1.001).contains(&ndc.y), "y={}", ndc.y);
            assert!((-0.001..=1.001).contains(&ndc.z), "z={}", ndc.z);
        }
    }
}
