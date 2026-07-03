//! # Scene Runtime Boundary
//!
//! Holds scene-graph structures and the renderer-facing submission payload.
//! This module is the CPU scene layer and intentionally does not own Vulkan objects.

pub mod command;
pub mod debug_scenarios;
pub mod render_submission;
pub mod scene_world;

pub use scene_world::SceneNodeId;
