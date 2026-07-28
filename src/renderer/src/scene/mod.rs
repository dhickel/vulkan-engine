//! # Scene Runtime Boundary
//!
//! Holds scene-graph structures and the renderer-facing submission payload.
//! This module is the CPU scene layer and intentionally does not own Vulkan objects.

#[cfg(feature = "bsp")]
pub mod bsp_visibility;
pub mod command;
pub mod debug_scenarios;
pub mod object_store;
pub mod render_submission;
pub mod scene_world;

#[cfg(feature = "scene-bvh")]
pub mod bvh;
#[cfg(feature = "scene-bvh")]
pub mod lod;

pub use scene_world::SceneNodeId;
