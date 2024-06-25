#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub uv_x: f32,
    pub normal: glam::Vec3,
    pub uv_y: f32,
    pub color: glam::Vec4,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Default::default(),
            uv_x: 0.0,
            normal: Default::default(),
            uv_y: 0.0,
            color: Default::default(),
        }
    }
}
