use ash::vk;

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

pub struct GPUScene {
    pub data: GPUSceneData,
    pub descriptor: [vk::DescriptorSetLayout; 1],
}

impl GPUScene {
    pub fn new(descriptor: vk::DescriptorSetLayout) -> Self {
        Self {
            data: Default::default(),
            descriptor: [descriptor],
        }
    }
}

pub struct GPUSceneData {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub view_projection: glam::Mat4,
    pub ambient_color: glam::Vec4,
    pub sunlight_direction: glam::Vec4,
    pub sunlight_color: glam::Vec4,
}

impl Default for GPUSceneData {
    fn default() -> Self {
        Self {
            view: Default::default(),
            projection: Default::default(),
            view_projection: Default::default(),
            ambient_color: Default::default(),
            sunlight_direction: Default::default(),
            sunlight_color: Default::default(),
        }
    }
}
