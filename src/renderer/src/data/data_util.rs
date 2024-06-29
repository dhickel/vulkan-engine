use ash::vk;

pub const EXTENT3D_ONE: vk::Extent3D = vk::Extent3D {
    width: 1,
    height: 1,
    depth: 1,
};

pub fn pack_unorm4x8(color: [f32; 4]) -> u32 {
    let r = (color[0] * 255.0).round() as u32;
    let g = (color[1] * 255.0).round() as u32;
    let b = (color[2] * 255.0).round() as u32;
    let a = (color[3] * 255.0).round() as u32;
    (r << 24) | (g << 16) | (b << 8) | a
}
