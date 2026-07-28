//! Sprite batch Vulkan backend: per-frame host-visible vertex upload, pipeline
//! binding, and draw. Gated behind the `sprites-2d` Cargo feature.
//!
//! Each sprite is expanded into a quad (2 triangles = 6 vertices) in a
//! host-visible ring buffer. A single pipeline is created at init time
//! (alpha-blended, depth-tested, triangle-list topology, no culling).

use crate::data::data_cache::VkPipelineType;
use crate::vulkan::vk_types::{VkBuffer, VkDestroyable, VkWindowState};
use crate::vulkan::vk_util;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};

// ---------------------------------------------------------------------------
// GPU vertex format — 32 bytes, matches shader SpriteVertex
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct SpriteGpuVertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
    pub color: [f32; 4],
}

impl SpriteGpuVertex {
    pub fn size_bytes() -> u32 {
        std::mem::size_of::<Self>() as u32
    }
}

// ---------------------------------------------------------------------------
// Push constants — matches shader layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct SpritePushConsts {
    pub view_projection: [[f32; 4]; 4],
    pub vertex_buffer_addr: u64,
}

impl SpritePushConsts {
    pub fn new(view_projection: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            view_projection: view_projection.to_cols_array_2d(),
            vertex_buffer_addr,
        }
    }
}

// ---------------------------------------------------------------------------
// Sprite instance (CPU-side, public)
// ---------------------------------------------------------------------------

/// A single sprite instance submitted for rendering.
#[derive(Copy, Clone, Debug)]
pub struct SpriteInstance {
    /// World-space center position.
    pub position: Vec2,
    /// Width × height in world units.
    pub size: Vec2,
    /// Rotation around the sprite center in radians.
    pub rotation: f32,
    /// Per-vertex color (modulated per-sprite).
    pub color: [f32; 4],
    /// Depth-sort layer (higher = drawn later, i.e. on top).
    pub layer: i32,
}

impl Default for SpriteInstance {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            size: Vec2::new(1.0, 1.0),
            rotation: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
            layer: 0,
        }
    }
}

impl SpriteInstance {
    /// Create a new sprite at `position` with given `size`.
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            position,
            size,
            ..Default::default()
        }
    }

    /// Set the color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set the rotation in radians.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Set the depth layer.
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    /// Expand this sprite into 6 GPU vertices (two triangles forming a quad).
    fn to_vertices(&self) -> [SpriteGpuVertex; 6] {
        let half = self.size * 0.5;
        // Corner positions relative to center (counter-clockwise):
        //   v0 = (-hw, -hh)   bottom-left
        //   v1 = ( hw, -hh)   bottom-right
        //   v2 = (-hw,  hh)   top-left
        //   v3 = ( hw,  hh)   top-right
        let corners: [Vec2; 4] = [
            Vec2::new(-half.x, -half.y),
            Vec2::new(half.x, -half.y),
            Vec2::new(-half.x, half.y),
            Vec2::new(half.x, half.y),
        ];

        let (sin_r, cos_r) = self.rotation.sin_cos();
        let rotate = |c: Vec2| -> [f32; 2] {
            let rx = cos_r * c.x - sin_r * c.y;
            let ry = sin_r * c.x + cos_r * c.y;
            [self.position.x + rx, self.position.y + ry]
        };

        let p0 = rotate(corners[0]);
        let p1 = rotate(corners[1]);
        let p2 = rotate(corners[2]);
        let p3 = rotate(corners[3]);

        // Two triangles: (p0,p1,p2) and (p2,p1,p3)
        // Winding is counter-clockwise in default Vulkan coordinate system
        // (no Y-flip in this vertex shader; projection handles it).
        let v0 = SpriteGpuVertex {
            position: p0,
            texcoord: [0.0, 0.0],
            color: self.color,
        };
        let v1 = SpriteGpuVertex {
            position: p1,
            texcoord: [1.0, 0.0],
            color: self.color,
        };
        let v2 = SpriteGpuVertex {
            position: p2,
            texcoord: [0.0, 1.0],
            color: self.color,
        };
        let v3 = SpriteGpuVertex {
            position: p3,
            texcoord: [1.0, 1.0],
            color: self.color,
        };

        [v0, v1, v2, v2, v1, v3]
    }
}

// ---------------------------------------------------------------------------
// Sprite batch backend
// ---------------------------------------------------------------------------

/// Default maximum sprites per frame (each sprite = 6 vertices).
pub(crate) const DEFAULT_MAX_SPRITES: usize = 16384;

pub(crate) struct VkSprites {
    /// One host-visible buffer per frame slot.
    pub vertex_buffers: Vec<Option<VkBuffer>>,
    /// Cached device address for the current frame slot's buffer.
    pub vertex_buffer_address: vk::DeviceAddress,
    /// Maximum vertex capacity for the current buffer.
    pub max_vertices: u32,
    /// Current write cursor (vertices uploaded this frame).
    pub written_vertices: u32,
}

impl VkSprites {
    pub fn new(max_sprites: usize) -> Self {
        let max_vertices = (max_sprites * 6).max(6) as u32;
        Self {
            vertex_buffers: Vec::new(),
            vertex_buffer_address: vk::DeviceAddress::default(),
            max_vertices,
            written_vertices: 0,
        }
    }

    /// Clear per-frame vertex count.
    pub fn clear_frame(&mut self) {
        self.written_vertices = 0;
    }

    /// Returns `true` when there are no sprites to draw this frame.
    pub fn is_empty(&self) -> bool {
        self.written_vertices == 0
    }

    /// Upload sprite vertices into the GPU buffer. Returns the vertex count
    /// uploaded (0 when sprites is empty).
    ///
    /// Sprites are sorted by layer before upload.
    pub fn upload_sprites(
        &mut self,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        frame_index: u32,
        sprites: &[SpriteInstance],
    ) -> Result<u32, String> {
        if sprites.is_empty() {
            return Ok(0);
        }

        // Sort by layer (ascending — lower layers drawn first).
        let mut sorted: Vec<&SpriteInstance> = sprites.iter().collect();
        sorted.sort_by_key(|s| s.layer);

        let sprite_count = sorted.len();
        let vertex_count = (sprite_count * 6) as u32;
        let clamped = vertex_count.min(self.max_vertices - (self.max_vertices % 6));
        if clamped == 0 {
            return Ok(0);
        }

        let size_bytes = clamped as u64 * SpriteGpuVertex::size_bytes() as u64;
        let slot = frame_index as usize;
        if self.vertex_buffers.len() <= slot {
            self.vertex_buffers.resize_with(slot + 1, || None);
        }
        let needs_realloc = self.vertex_buffers[slot]
            .as_ref()
            .is_none_or(|buffer| buffer.size < size_bytes);

        if needs_realloc {
            if let Some(mut old) = self.vertex_buffers[slot].take() {
                old.destroy(device, allocator);
            }

            let new_buffer = vk_util::allocate_buffer(
                allocator,
                size_bytes,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk_mem::MemoryUsage::AutoPreferHost,
            )?;
            self.vertex_buffers[slot] = Some(new_buffer);
        }

        let buffer = self.vertex_buffers[slot].as_mut().expect("buffer allocated above");
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        self.vertex_buffer_address = unsafe { device.get_buffer_device_address(&addr_info) };

        // Map, write, unmap.
        unsafe {
            let data_ptr = allocator
                .map_memory(&mut buffer.allocation)
                .map_err(|err| format!("sprite buffer map: {err:?}"))?;

            let dst: *mut SpriteGpuVertex = data_ptr.cast();
            let clipped_count = (clamped / 6) as usize;
            for (i, sprite) in sorted.iter().take(clipped_count).enumerate() {
                let verts = sprite.to_vertices();
                let base = i * 6;
                *dst.add(base) = verts[0];
                *dst.add(base + 1) = verts[1];
                *dst.add(base + 2) = verts[2];
                *dst.add(base + 3) = verts[3];
                *dst.add(base + 4) = verts[4];
                *dst.add(base + 5) = verts[5];
            }

            allocator.unmap_memory(&mut buffer.allocation);
        }

        self.written_vertices = clamped;
        Ok(clamped)
    }

    /// Record the draw call into the active command buffer.
    pub fn record_draw(
        &self,
        device: &ash::Device,
        cmd_buffer: vk::CommandBuffer,
        pipeline_cache: &crate::data::data_cache::VkPipelineCache,
        window_state: &VkWindowState,
        view_projection: Mat4,
    ) -> Result<(), String> {
        if self.is_empty() {
            return Ok(());
        }

        let pipeline_obj = pipeline_cache.get_pipeline(VkPipelineType::Sprites);
        let push_consts = SpritePushConsts::new(view_projection, self.vertex_buffer_address);

        unsafe {
            device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_obj.pipeline,
            );

            let viewport = window_state.get_viewport();
            let scissor = window_state.get_scissor();
            device.cmd_set_viewport(cmd_buffer, 0, viewport);
            device.cmd_set_scissor(cmd_buffer, 0, scissor);

            device.cmd_push_constants(
                cmd_buffer,
                pipeline_obj.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push_consts),
            );

            device.cmd_draw(cmd_buffer, self.written_vertices, 1, 0, 0);
        }

        Ok(())
    }
}

impl VkDestroyable for VkSprites {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        for buffer in self.vertex_buffers.iter_mut().flatten() {
            buffer.destroy(device, allocator);
        }
        self.vertex_buffers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{SpriteGpuVertex, SpriteInstance};
    use glam::Vec2;

    #[test]
    fn vertex_layout_matches_std430_offsets() {
        let sprite = SpriteInstance::new(Vec2::new(5.0, 3.0), Vec2::new(2.0, 2.0));
        let verts = sprite.to_vertices();
        let v = &verts[0];
        let base = (v as *const SpriteGpuVertex) as usize;
        assert_eq!((&v.position as *const [f32; 2]) as usize - base, 0);
        assert_eq!((&v.texcoord as *const [f32; 2]) as usize - base, 8);
        assert_eq!((&v.color as *const [f32; 4]) as usize - base, 16);
        assert_eq!(SpriteGpuVertex::size_bytes(), 32);
    }

    #[test]
    fn quad_produces_six_vertices() {
        let sprite = SpriteInstance::new(Vec2::ZERO, Vec2::new(4.0, 2.0));
        let verts = sprite.to_vertices();
        assert_eq!(verts.len(), 6);
    }

    #[test]
    fn zero_rotation_is_axis_aligned() {
        let sprite = SpriteInstance::new(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let verts = sprite.to_vertices();
        // Bottom-left should be (-1, -1)
        assert!((verts[0].position[0] + 1.0).abs() < 1e-6);
        assert!((verts[0].position[1] + 1.0).abs() < 1e-6);
        // Top-right should be (1, 1)
        assert!((verts[3].position[0] - 1.0).abs() < 1e-6);
        assert!((verts[3].position[1] - 1.0).abs() < 1e-6);
    }
}
