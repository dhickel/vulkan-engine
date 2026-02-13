//! # Stable Handles & Resource Referencing
//!
//! ## Purpose
//! This module defines the "Handle" pattern used throughout the engine to reference GPU resources
//! (meshes, textures, materials) without using raw pointers or direct indices.
//!
//! ## Key Concepts
//! - **Slot**: The index into a cache's internal storage (usually a `Vec`).
//! - **Generation**: A counter that increments every time a slot is reused.
//! - **Safety**: If you try to use a handle with `generation: 1` but the cache is now at
//!   `generation: 2` for that slot, the engine can detect that the handle is **stale**
//!   and avoid rendering garbage or crashing.
//!
//! ## Why Handles?
//! 1. **Memory Safety**: Avoids the "Dangling Pointer" problem.
//! 2. **Performance**: Much faster than string-based lookups or smart pointers.
//! 3. **Indirection**: Allows the cache to move or reallocate internal memory without
//!    invalidating the handles held by game logic.

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
