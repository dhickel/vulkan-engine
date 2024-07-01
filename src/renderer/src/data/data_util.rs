use ash::vk;
use glam::Vec4;


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