//! # Scene Runtime Boundary
//!
//! Holds scene-graph structures and the renderer-facing submission payload.
//! This module is the CPU scene layer and intentionally does not own Vulkan objects.

pub mod debug_scenarios;
pub mod render_submission;
pub mod scene_world;

pub use render_submission::{RenderSubmission, SubmissionFlags};
pub use scene_world::{SceneNode, SceneNodeId, SceneWorld};
