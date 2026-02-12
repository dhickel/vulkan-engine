#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshHandle {
    pub slot: u32,
    pub generation: u32,
}

impl MeshHandle {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialHandle {
    pub slot: u32,
    pub generation: u32,
}

impl MaterialHandle {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureHandle {
    pub slot: u32,
    pub generation: u32,
}

impl TextureHandle {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EnvironmentHandle {
    pub slot: u32,
    pub generation: u32,
}

impl EnvironmentHandle {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheError {
    InvalidHandle,
    StaleHandle,
    NotLoaded,
    OutOfBounds,
}
