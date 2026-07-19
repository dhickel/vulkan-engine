//! # Directional Shadow Map Resources (CSM + Legacy)
//!
//! Manages per-frame-in-flight shadow map images, image views, and samplers
//! for directional light shadow mapping.
//!
//! ## Legacy path (CSM disabled)
//! Single 2048² D32 depth image per frame slot, sampled as `sampler2DShadow`.
//!
//! ## CSM path (`csm` feature, runtime-enabled)
//! One D32 image per frame slot with array layer count 3 (1024² each),
//! one whole-array sampling view, three single-layer depth-attachment views,
//! and one comparison sampler.

use crate::data::camera::Aabb;
use crate::data::gpu_data::{CSM_CASCADE_COUNT, CSM_CASCADE_DIM, RenderObject};
use crate::vulkan::vk_types::{VkDestroyable, VkImageAlloc};
use crate::vulkan::vk_util;
use ash::vk;
use glam::{Mat4, Vec3, Vec4Swizzles};
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;

// ── Shared constants ──────────────────────────────────────────────────────

/// Legacy single-map shadow resolution.
pub const LEGACY_SHADOW_MAP_DIM: u32 = 2048;

/// CSM blend band fraction (0.0–1.0, clamped in shader).
pub const CSM_BLEND_FRACTION: f32 = 0.1;

/// Guard band multiplier for depth bounds in light space.
const DEPTH_GUARD_BAND: f32 = 0.1;

/// Lambda for practical cascade split mix (0 = uniform, 1 = logarithmic).
const CSM_LAMBDA: f32 = 0.75;

/// Minimum near plane for cascade splits (meters).
const MIN_CASCADE_NEAR: f32 = 0.01;

/// ── Per-frame shadow resource types ──────────────────────────────────────

/// Per-frame shadow map resources for the legacy single-map path.
pub struct VkShadowFrame {
    pub shadow_map: VkImageAlloc,
    pub shadow_map_view: vk::ImageView,
    pub shadow_sampler: vk::Sampler,
}

impl VkShadowFrame {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_image_view(self.shadow_map_view, None);
        }
        VkDestroyable::destroy(&mut self.shadow_map, device, allocator);
    }
}

/// Per-frame CSM shadow map resources.
pub struct VkCsmShadowFrame {
    /// D32 2D-array image (1024² × 3 layers).
    pub csm_image: VkImageAlloc,
    /// Whole-array view for sampling (layer_count = 3).
    pub csm_array_view: vk::ImageView,
    /// Three single-layer depth-attachment views for rendering each cascade.
    pub csm_layer_views: [vk::ImageView; CSM_CASCADE_COUNT as usize],
    /// Comparison sampler with linear filtering and clamp-to-border.
    pub csm_sampler: vk::Sampler,
}

impl VkCsmShadowFrame {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        unsafe {
            device.destroy_sampler(self.csm_sampler, None);
            device.destroy_image_view(self.csm_array_view, None);
            for view in &self.csm_layer_views {
                device.destroy_image_view(*view, None);
            }
        }
        VkDestroyable::destroy(&mut self.csm_image, device, allocator);
    }
}

// ── Resource collections ──────────────────────────────────────────────────

/// Collection of per-frame legacy shadow resources for all frames in flight.
pub struct VkShadowResources {
    pub frames: Vec<VkShadowFrame>,
    pub shadow_map_extent: vk::Extent2D,
}

impl VkShadowResources {
    pub const SHADOW_MAP_DIM: u32 = LEGACY_SHADOW_MAP_DIM;

    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        frame_count: u32,
    ) -> Result<Self, String> {
        let allocator = allocator
            .lock()
            .map_err(|e| format!("allocator lock: {e}"))?;
        let shadow_map_extent = vk::Extent2D {
            width: Self::SHADOW_MAP_DIM,
            height: Self::SHADOW_MAP_DIM,
        };

        let mut frames: Vec<VkShadowFrame> = Vec::with_capacity(frame_count as usize);
        for _ in 0..frame_count {
            let (mut shadow_map, shadow_map_view) = match vk_util::create_image(
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
                Ok(image) => {
                    let view = image.image_view;
                    (image, view)
                }
                Err(err) => {
                    for frame in &mut frames {
                        frame.destroy(device, &allocator);
                    }
                    return Err(err);
                }
            };

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
                    VkDestroyable::destroy(&mut shadow_map, device, &allocator);
                    unsafe { device.destroy_image_view(shadow_map_view, None); }
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

    pub fn get_frame(&self, frame_index: u32) -> &VkShadowFrame {
        &self.frames[frame_index as usize]
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        for frame in &mut self.frames {
            frame.destroy(device, allocator);
        }
        self.frames.clear();
    }
}

/// Collection of per-frame CSM shadow resources.
pub struct VkCsmShadowResources {
    pub frames: Vec<VkCsmShadowFrame>,
    pub extent: vk::Extent2D,
}

impl VkCsmShadowResources {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        frame_count: u32,
    ) -> Result<Self, String> {
        let allocator = allocator
            .lock()
            .map_err(|e| format!("allocator lock: {e}"))?;
        let extent = vk::Extent2D {
            width: CSM_CASCADE_DIM,
            height: CSM_CASCADE_DIM,
        };

        let mut frames: Vec<VkCsmShadowFrame> = Vec::with_capacity(frame_count as usize);

        for _ in 0..frame_count {
            // Create the 2D-array D32 image with 3 layers.
            let csm_image = match vk_util::create_array_image(
                device,
                &allocator,
                vk::Extent3D {
                    width: CSM_CASCADE_DIM,
                    height: CSM_CASCADE_DIM,
                    depth: 1,
                },
                CSM_CASCADE_COUNT,
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

            // Whole-array view for sampling (all 3 layers).
            let array_view_info = vk::ImageViewCreateInfo::default()
                .image(csm_image.image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: CSM_CASCADE_COUNT,
                });

            let csm_array_view = unsafe {
                device
                    .create_image_view(&array_view_info, None)
                    .map_err(|e| format!("failed to create CSM array view: {e:?}"))?
            };

            // Single-layer depth-attachment views for each cascade.
            let mut csm_layer_views = [vk::ImageView::null(); CSM_CASCADE_COUNT as usize];
            for layer in 0..CSM_CASCADE_COUNT {
                let layer_view_info = vk::ImageViewCreateInfo::default()
                    .image(csm_image.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    });

                csm_layer_views[layer as usize] = unsafe {
                    device
                        .create_image_view(&layer_view_info, None)
                        .map_err(|e| format!("failed to create CSM layer {layer} view: {e:?}"))?
                };
            }

            // Comparison sampler.
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

            let csm_sampler = unsafe {
                device
                    .create_sampler(&sampler_info, None)
                    .map_err(|e| format!("failed to create CSM sampler: {e:?}"))?
            };

            frames.push(VkCsmShadowFrame {
                csm_image,
                csm_array_view,
                csm_layer_views,
                csm_sampler,
            });
        }

        Ok(Self { frames, extent })
    }

    pub fn get_frame(&self, frame_index: u32) -> &VkCsmShadowFrame {
        &self.frames[frame_index as usize]
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &Allocator) {
        for frame in &mut self.frames {
            frame.destroy(device, allocator);
        }
        self.frames.clear();
    }
}

// ── Cascade fitting (Steps 7–9) ────────────────────────────────────────────

/// Parameters computed per CSM cascade slice.
#[derive(Debug, Clone)]
pub struct CsmCascadeParams {
    /// Light view-projection matrix for this cascade.
    pub light_view_proj: Mat4,
    /// View-space near/far split distances.
    pub split_near: f32,
    pub split_far: f32,
    /// World-space AABB of the cascade frustum slice.
    pub frustum_slice_aabb: Aabb,
    /// Number of candidate casters for this cascade (pre-cull).
    pub candidate_casters: usize,
    /// Number of casters emitted after culling.
    pub emitted_casters: usize,
}

/// Compute eight frustum corners in world space from the Vulkan [0,1]
/// projection-view matrix. Returns `None` if the matrix is non-invertible.
pub fn frustum_corners_from_vp(view_projection: &Mat4) -> Option<[Vec3; 8]> {
    let inv_vp = view_projection.inverse();

    // Vulkan NDC cube corners (x∈[-1,1], y∈[-1,1], z∈[0,1]).
    let ndc_corners: [(f32, f32, f32); 8] = [
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ];

    let mut corners = [Vec3::ZERO; 8];
    for (i, (x, y, z)) in ndc_corners.iter().enumerate() {
        let world = inv_vp * glam::Vec4::new(*x, *y, *z, 1.0);
        if world.w.abs() < 1e-10 {
            return None;
        }
        corners[i] = world.xyz() / world.w;
    }

    if corners.iter().any(|c| !c.is_finite()) {
        return None;
    }

    Some(corners)
}

/// Compute the eight corners of a sub-frustum slice defined by near/far depth.
/// (Unused stub — slice corners are computed via `compute_slice_corners_from_splits`.)
#[allow(dead_code)]
fn frustum_slice_corners(
    _inv_vp: &Mat4,
    _near: f32,
    _far: f32,
) -> Option<[Vec3; 8]> {
    None
}

/// Compute three practical cascade split distances (lambda-weighted mix).
/// Returns `[near0, far0, near1, far1, near2, far2]` as view-space depths.
fn compute_cascade_splits(
    camera_near: f32,
    camera_far: f32,
    lambda: f32,
) -> [f32; 6] {
    let count = CSM_CASCADE_COUNT as f32;

    let mut splits = [0.0_f32; 6];
    for i in 0..CSM_CASCADE_COUNT as usize {
        let p = (i as f32 + 1.0) / count;
        let log_split = camera_near * (camera_far / camera_near).powf(p);
        let uniform_split = camera_near + (camera_far - camera_near) * p;
        let split = lambda * log_split + (1.0 - lambda) * uniform_split;

        let near = if i == 0 {
            camera_near
        } else {
            splits[(i - 1) * 2 + 1]
        };
        let far = split;

        splits[i * 2] = near;
        splits[i * 2 + 1] = far;
    }

    splits
}

/// Stable light view-projection for one cascade.
///
/// 1. Computes a bounding sphere from the frustum slice corners.
/// 2. Quantizes the radius conservatively.
/// 3. Derives world-units-per-texel at CSM resolution.
/// 4. Snaps the light-space center to texel grid.
/// 5. Extends depth bounds by projecting caster AABBs and adding guard band.
pub fn compute_cascade_light_view_proj(
    light_dir: Vec3,
    frustum_corners: &[Vec3; 8],
    caster_aabbs: &[Aabb],
    cascade_dim: u32,
) -> Option<(Mat4, Mat4)> {
    if frustum_corners.iter().any(|c| !c.is_finite()) {
        return None;
    }

    // Compute bounding sphere of frustum corners.
    let center: Vec3 = frustum_corners.iter().sum::<Vec3>() / 8.0;
    let radius = frustum_corners
        .iter()
        .map(|c| (*c - center).length())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0)
        .max(0.5);

    let direction_to_light = light_dir.normalize();

    // Light position: move far enough to encompass the sphere.
    let light_pos = center + direction_to_light * (radius * 2.0 + 1.0);
    let up = if direction_to_light.dot(Vec3::Y).abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let view = Mat4::look_at_rh(light_pos, center, up);

    // Project frustum corners into light space.
    let (light_min, light_max) = frustum_corners.iter().fold(
        (
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        ),
        |(min, max), corner| {
            let view_corner = view.transform_point3(*corner);
            (min.min(view_corner), max.max(view_corner))
        },
    );

    // Extend Z range to include casters.
    let (z_min, z_max) = caster_aabbs.iter().fold(
        (light_min.z, light_max.z),
        |(zmin, zmax), aabb| {
            let corners = aabb_corners_vec(aabb.min, aabb.max);
            let (cmin, cmax) = corners.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(mn, mx), c| {
                    let vc = view.transform_point3(*c);
                    (mn.min(vc.z), mx.max(vc.z))
                },
            );
            (zmin.min(cmin), zmax.max(cmax))
        },
    );

    // Quantize and snap.
    let world_units_per_texel = (light_max.x - light_min.x) / cascade_dim as f32;
    let quantized_radius = (radius / world_units_per_texel).ceil() * world_units_per_texel;
    let snapped_center_x = (center.x / world_units_per_texel).floor() * world_units_per_texel;
    let snapped_center_y = (center.y / world_units_per_texel).floor() * world_units_per_texel;
    let snapped_center_z = (center.z / world_units_per_texel).floor() * world_units_per_texel;
    let snapped_center = Vec3::new(snapped_center_x, snapped_center_y, snapped_center_z);

    let snapped_light_pos = snapped_center + direction_to_light * (quantized_radius * 2.0 + 1.0);
    let snapped_view = Mat4::look_at_rh(snapped_light_pos, snapped_center, up);

    let (snapped_min, snapped_max) = frustum_corners.iter().fold(
        (
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        ),
        |(min, max), corner| {
            let view_corner = snapped_view.transform_point3(*corner);
            (min.min(view_corner), max.max(view_corner))
        },
    );

    let xy_margin = (snapped_max - snapped_min).truncate().length().max(1.0) * 0.05;
    let depth_margin = (z_max - z_min).max(1.0) * DEPTH_GUARD_BAND;

    // Add caster depth extension.
    let caster_z_min = z_min.min(snapped_min.z);
    let caster_z_max = z_max.max(snapped_max.z);

    let near = (caster_z_max + depth_margin).max(MIN_CASCADE_NEAR);
    let far = (caster_z_min - depth_margin).max(near + 0.01);

    let projection = Mat4::orthographic_rh(
        snapped_min.x - xy_margin,
        snapped_max.x + xy_margin,
        snapped_min.y - xy_margin,
        snapped_max.y + xy_margin,
        near,
        far,
    );

    let view_projection = projection * snapped_view;
    Some((snapped_view, view_projection))
}

/// Check whether a caster AABB (in world space) overlaps with a cascade's light-space
/// XY receiver footprint. This implements the off-camera caster invariant:
/// a caster is included when its world AABB overlaps the cascade's footprint
/// regardless of camera-frustum membership.
pub fn caster_overlaps_cascade_light_footprint(
    caster_aabb: &Aabb,
    light_view_proj: &Mat4,
) -> bool {
    let corners = aabb_corners_vec(caster_aabb.min, caster_aabb.max);

    // Project all corners into light clip space.
    let (clip_min, clip_max) = corners.iter().fold(
        (
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        ),
        |(min, max), corner| {
            let clip = *light_view_proj * corner.extend(1.0);
            if clip.w.abs() > 1e-10 {
                let ndc = clip.xyz() / clip.w;
                (min.min(ndc), max.max(ndc))
            } else {
                (min, max)
            }
        },
    );

    // Test overlap with light-space NDC [-1, 1] in XY and [-1, 1] in Z (depth range).
    clip_max.x >= -1.0
        && clip_min.x <= 1.0
        && clip_max.y >= -1.0
        && clip_min.y <= 1.0
        && clip_max.z >= -1.0
        && clip_min.z <= 1.0
}

/// Cull known rigid casters independently per cascade using conservative light-space AABB overlap.
/// Unknown/skinned/deformed casters are included in every active cascade.
pub fn cull_casters_for_cascade<'a>(
    casters: impl IntoIterator<Item = &'a RenderObject>,
    light_view_proj: &Mat4,
) -> Vec<&'a RenderObject> {
    casters
        .into_iter()
        .filter(|draw| {
            let aabb = Aabb::from_min_max(draw.bounds_min, draw.bounds_max);
            // If bounds are degenerate/invalid (unknown caster), include.
            if !aabb.is_finite() || !aabb.is_ordered() || aabb.min == aabb.max {
                return true;
            }
            // Transform world AABB corners.
            let world_aabb = aabb_transformed(&aabb, &draw.transform);
            caster_overlaps_cascade_light_footprint(&world_aabb, light_view_proj)
        })
        .collect()
}

// ── Legacy fitting functions (unchanged from prior version) ────────────────

pub fn compute_light_view_projection(
    light_dir: glam::Vec3,
    scene_aabb_min: glam::Vec3,
    scene_aabb_max: glam::Vec3,
) -> (glam::Mat4, glam::Mat4, glam::Mat4) {
    let center = (scene_aabb_min + scene_aabb_max) * 0.5;
    let extent = (scene_aabb_max - scene_aabb_min).abs();
    let radius = (extent.length() * 0.5).max(0.5);
    let direction_to_light = light_dir.normalize();

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
    draws: impl IntoIterator<Item = &'a RenderObject>,
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

fn aabb_corners_vec(min: Vec3, max: Vec3) -> [Vec3; 8] {
    aabb_corners(min, max)
}

fn aabb_transformed(aabb: &Aabb, transform: &Mat4) -> Aabb {
    let corners = aabb_corners_vec(aabb.min, aabb.max);
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for corner in &corners {
        let p = transform.transform_point3(*corner);
        world_min = world_min.min(p);
        world_max = world_max.max(p);
    }
    Aabb::from_min_max(world_min, world_max)
}

// ── CSM cascade computation entry point ────────────────────────────────────

/// Compute CSM cascade parameters from camera matrices and caster bounds.
///
/// Returns `None` if the view/projection is non-invertible.
pub fn compute_csm_cascades(
    view: &Mat4,
    projection: &Mat4,
    light_dir: Vec3,
    camera_near: f32,
    camera_far: f32,
    casters: &[RenderObject],
) -> Option<Vec<CsmCascadeParams>> {
    let view_proj = *projection * *view;
    let inv_vp = view_proj.inverse();

    if !inv_vp.is_finite() {
        return None;
    }

    let splits = compute_cascade_splits(camera_near, camera_far, CSM_LAMBDA);
    let full_corners = frustum_corners_from_vp(&view_proj)?;

    let mut cascades = Vec::with_capacity(CSM_CASCADE_COUNT as usize);

    for i in 0..CSM_CASCADE_COUNT as usize {
        let split_near = splits[i * 2];
        let split_far = splits[i * 2 + 1];

        // Compute the frustum slice corners by interpolating between
        // the full near/far plane corners in world space.
        // We use the inverse VP with adjusted NDC z.
        // For standard perspective: we find the NDC z values corresponding
        // to split_near/split_far and transform through inv_vp.
        let slice_corners = compute_slice_corners_from_splits(
            &inv_vp,
            &full_corners,
            camera_near,
            camera_far,
            split_near,
            split_far,
        )?;

        // Compute slice AABB.
        let (slice_min, slice_max) = slice_corners.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), c| (min.min(*c), max.max(*c)),
        );
        let frustum_slice_aabb = Aabb::from_min_max(slice_min, slice_max);

        // Collect conservative caster AABBs (in world space).
        let caster_aabbs: Vec<Aabb> = casters
            .iter()
            .map(|draw| {
                let local = Aabb::from_min_max(draw.bounds_min, draw.bounds_max);
                if !local.is_finite() || !local.is_ordered() {
                    Aabb::from_min_max(Vec3::splat(-1000.0), Vec3::splat(1000.0))
                } else {
                    aabb_transformed(&local, &draw.transform)
                }
            })
            .collect();

        let candidate_casters = casters.len();

        // Compute light view-proj for this cascade.
        let (_, light_view_proj) = compute_cascade_light_view_proj(
            light_dir,
            &slice_corners,
            &caster_aabbs,
            CSM_CASCADE_DIM,
        )?;

        // Cull casters for this cascade.
        let visible_casters = cull_casters_for_cascade(casters.iter(), &light_view_proj);
        let emitted_casters = visible_casters.len();

        cascades.push(CsmCascadeParams {
            light_view_proj,
            split_near,
            split_far,
            frustum_slice_aabb,
            candidate_casters,
            emitted_casters,
        });
    }

    Some(cascades)
}

/// Compute frustum slice corners from split distances by interpolating
/// between the full near and far plane corners.
fn compute_slice_corners_from_splits(
    _inv_vp: &Mat4,
    full_corners: &[Vec3; 8],
    camera_near: f32,
    camera_far: f32,
    split_near: f32,
    split_far: f32,
) -> Option<[Vec3; 8]> {
    let full_near_corners: [Vec3; 4] = [
        full_corners[0],
        full_corners[1],
        full_corners[2],
        full_corners[3],
    ];
    let full_far_corners: [Vec3; 4] = [
        full_corners[4],
        full_corners[5],
        full_corners[6],
        full_corners[7],
    ];

    let depth_range = camera_far - camera_near;
    if depth_range < 1e-6 {
        return None;
    }

    let near_ratio = (split_near - camera_near) / depth_range;
    let far_ratio = (split_far - camera_near) / depth_range;

    let mut slice_corners = [Vec3::ZERO; 8];
    for i in 0..4 {
        slice_corners[i] = full_near_corners[i] + (full_far_corners[i] - full_near_corners[i]) * near_ratio;
        slice_corners[4 + i] =
            full_near_corners[i] + (full_far_corners[i] - full_near_corners[i]) * far_ratio;
    }

    if slice_corners.iter().any(|c| !c.is_finite()) {
        return None;
    }

    Some(slice_corners)
}

/// Derive camera near/far from the frustum corners (conservative).
pub fn derive_camera_near_far_from_corners(corners: &[Vec3; 8]) -> (f32, f32) {
    let center = corners.iter().sum::<Vec3>() / 8.0;
    let max_dist = corners
        .iter()
        .map(|c| (*c - center).length())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(100.0);

    (MIN_CASCADE_NEAR, max_dist.max(10.0))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn vertical_direction_produces_finite_matrix_covering_scene_bounds() {
        let min = Vec3::new(-4.0, -1.0, -3.0);
        let max = Vec3::new(7.0, 5.0, 2.0);
        let (_, _, view_projection) = compute_light_view_projection(Vec3::Y, min, max);

        assert!(view_projection.is_finite());
        for corner in aabb_corners(min, max) {
            let clip = view_projection * corner.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!((-1.001..=1.001).contains(&ndc.x), "x={}", ndc.x);
            assert!((-1.001..=1.001).contains(&ndc.y), "y={}", ndc.y);
            assert!((-0.001..=1.001).contains(&ndc.z), "z={}", ndc.z);
        }
    }

    #[test]
    fn frustum_corners_produces_finite_results_from_standard_perspective() {
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = proj * view;

        let corners = frustum_corners_from_vp(&vp);
        assert!(corners.is_some());
        let corners = corners.unwrap();
        for c in &corners {
            assert!(c.is_finite(), "corner not finite: {:?}", c);
        }
    }

    #[test]
    fn cascade_splits_are_monotonic_and_bounded() {
        let splits = compute_cascade_splits(0.1, 100.0, CSM_LAMBDA);
        // 3 cascades => 6 values: [near0, far0, near1, far1, near2, far2]
        assert_eq!(splits.len(), 6);
        assert!(splits[0] >= 0.1);
        assert!(splits[5] <= 100.0);
        for i in 0..5 {
            assert!(splits[i] <= splits[i + 1], "non-monotonic at {}", i);
        }
    }

    #[test]
    fn cascade_split_lambda_zero_is_uniform() {
        let splits = compute_cascade_splits(0.1, 10.0, 0.0);
        // Uniform: each cascade gets equal share of depth range.
        let expected_step = (10.0 - 0.1) / 3.0;
        for i in 0..3 {
            let near = 0.1 + i as f32 * expected_step;
            let far = near + expected_step;
            assert!((splits[i * 2] - near).abs() < 0.01, "near mismatch cascade {i}");
            assert!((splits[i * 2 + 1] - far).abs() < 0.01, "far mismatch cascade {i}");
        }
    }

    #[test]
    fn cascade_split_lambda_one_is_logarithmic() {
        let splits = compute_cascade_splits(0.1, 100.0, 1.0);
        // With lambda=1, splits grow exponentially.
        let ratio1 = splits[1] / splits[0];
        let ratio2 = splits[3] / splits[2];
        assert!(ratio1 > 1.5, "expected log growth for first split");
        assert!(ratio2 > 1.5, "expected log growth for second split");
    }

    #[test]
    fn caster_overlaps_detects_intersection() {
        let light_dir = Vec3::Y;
        let center = Vec3::ZERO;
        let radius = 5.0;
        let light_pos = center + light_dir.normalize() * (radius * 2.0 + 1.0);
        let view = Mat4::look_at_rh(light_pos, center, Vec3::X);
        let proj = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        let vp = proj * view;

        // AABB inside the footprint.
        let inside = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(caster_overlaps_cascade_light_footprint(&inside, &vp));

        // AABB far outside the footprint.
        let outside = Aabb::from_min_max(Vec3::splat(100.0), Vec3::splat(101.0));
        assert!(!caster_overlaps_cascade_light_footprint(&outside, &vp));
    }

    #[test]
    fn off_camera_caster_still_overlaps_if_in_light_footprint() {
        // Light looking straight down, camera looking forward.
        // A caster behind the camera should still cast shadow if it's
        // within the light's XY footprint.
        let light_dir = Vec3::NEG_Y;
        let center = Vec3::new(0.0, 0.0, 5.0);
        let radius = 10.0;
        let light_pos = center + light_dir.normalize() * (radius * 2.0 + 1.0);
        let view = Mat4::look_at_rh(light_pos, center, Vec3::Z);
        let proj = Mat4::orthographic_rh(-20.0, 20.0, -20.0, 20.0, 0.1, 200.0);
        let vp = proj * view;

        // Caster behind camera but under the light.
        let behind_camera = Aabb::from_min_max(
            Vec3::new(-1.0, -0.1, 15.0),
            Vec3::new(1.0, 0.1, 16.0),
        );
        assert!(caster_overlaps_cascade_light_footprint(
            &behind_camera,
            &vp
        ));
    }

    #[test]
    fn frustum_corners_rejects_non_invertible_matrix() {
        let zero = Mat4::ZERO;
        assert!(frustum_corners_from_vp(&zero).is_none());
    }
}
