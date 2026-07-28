//! Debug line Vulkan backend: per-frame host-visible vertex upload, pipeline
//! binding, and draw. Gated behind the `debug-draw` Cargo feature.
//!
//! The vertex buffer uses a ring-buffer overwrite pattern; a single pipeline
//! is created at init time (unlit, depth-tested, line-list topology, no culling).

use crate::data::data_cache::VkPipelineType;
use crate::vulkan::vk_types::{VkBuffer, VkDestroyable, VkWindowState};
use crate::vulkan::vk_util;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

// ---------------------------------------------------------------------------
// GPU vertex format
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct DebugLineGpuVertex {
    pub position: [f32; 3],
    pub _position_pad: f32,
    pub color: [f32; 3],
    pub _color_pad: f32,
}

impl DebugLineGpuVertex {
    pub fn new(from: Vec3, to: Vec3, color: Vec3) -> [Self; 2] {
        [
            Self {
                position: from.to_array(),
                _position_pad: 0.0,
                color: color.to_array(),
                _color_pad: 0.0,
            },
            Self {
                position: to.to_array(),
                _position_pad: 0.0,
                color: color.to_array(),
                _color_pad: 0.0,
            },
        ]
    }

    pub fn size_bytes() -> u32 {
        std::mem::size_of::<Self>() as u32
    }
}

// ---------------------------------------------------------------------------
// Push constants — matches shader layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct DebugLinePushConsts {
    pub view_projection: [[f32; 4]; 4],
    pub vertex_buffer_addr: u64,
}

impl DebugLinePushConsts {
    pub fn new(view_projection: Mat4, vertex_buffer_addr: vk::DeviceAddress) -> Self {
        Self {
            view_projection: view_projection.to_cols_array_2d(),
            vertex_buffer_addr,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug line backend
// ---------------------------------------------------------------------------

/// Default maximum lines per frame (each line is 2 vertices).
pub(crate) const DEFAULT_MAX_DEBUG_LINES: usize = 65536;

pub(crate) struct VkDebugLines {
    /// One host-visible buffer per frame slot. A slot is reused only after its
    /// fence signals, so uploading never overwrites vertices an in-flight frame reads.
    pub vertex_buffers: Vec<Option<VkBuffer>>,
    /// Cached address for the currently recording frame slot's buffer.
    pub vertex_buffer_address: vk::DeviceAddress,
    /// Maximum vertex capacity for the current buffer.
    pub max_vertices: u32,
    /// Current write cursor (vertex count uploaded this frame).
    pub written_vertices: u32,
}

impl VkDebugLines {
    pub fn new(max_lines: usize) -> Self {
        let max_vertices = (max_lines * 2).max(2) as u32;
        Self {
            vertex_buffers: Vec::new(),
            vertex_buffer_address: vk::DeviceAddress::default(),
            max_vertices,
            written_vertices: 0,
        }
    }

    /// Clear per-frame vertex count. Called at the start of each frame.
    #[allow(dead_code)]
    pub fn clear_frame(&mut self) {
        self.written_vertices = 0;
    }

    /// Returns `true` when there are no lines to draw this frame.
    pub fn is_empty(&self) -> bool {
        self.written_vertices == 0
    }

    /// Upload debug line vertices into the GPU buffer. Returns the vertex count
    /// uploaded (0 when the buffer is empty or lines is empty).
    pub fn upload_lines(
        &mut self,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        frame_index: u32,
        lines: &[(Vec3, Vec3, Vec3)],
    ) -> Result<u32, String> {
        let line_count = lines.len();
        if line_count == 0 {
            return Ok(0);
        }

        let vertex_count = (line_count * 2) as u32;
        let clamped = vertex_count.min(self.max_vertices - (self.max_vertices % 2));
        if clamped == 0 {
            return Ok(0);
        }

        let size_bytes = clamped as u64 * DebugLineGpuVertex::size_bytes() as u64;
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
                .map_err(|err| format!("debug line buffer map: {err:?}"))?;

            let dst: *mut DebugLineGpuVertex = data_ptr.cast();
            for (i, &(from, to, color)) in lines.iter().take(clamped as usize / 2).enumerate() {
                let verts = DebugLineGpuVertex::new(from, to, color);
                *dst.add(i * 2) = verts[0];
                *dst.add(i * 2 + 1) = verts[1];
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

        let pipeline_obj = pipeline_cache.get_pipeline(VkPipelineType::DebugLines);

        let push_consts =
            DebugLinePushConsts::new(view_projection, self.vertex_buffer_address);

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

#[cfg(test)]
mod tests {
    use super::DebugLineGpuVertex;

    #[test]
    fn vertex_layout_matches_std430_vec3_offsets() {
        let vertex = DebugLineGpuVertex::new(glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z)[0];
        let base = (&vertex as *const DebugLineGpuVertex) as usize;
        assert_eq!((&vertex.position as *const [f32; 3]) as usize - base, 0);
        assert_eq!((&vertex.color as *const [f32; 3]) as usize - base, 16);
        assert_eq!(DebugLineGpuVertex::size_bytes(), 32);
    }
}

impl VkDestroyable for VkDebugLines {
    fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        for buffer in self.vertex_buffers.iter_mut().flatten() {
            buffer.destroy(device, allocator);
        }
        self.vertex_buffers.clear();
    }
}
