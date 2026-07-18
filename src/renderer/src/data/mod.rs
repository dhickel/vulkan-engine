//! # Data Layer
//!
//! CPU-side asset, scene, and cache systems that feed render submissions and Vulkan upload paths.

pub mod asset_manifest;
pub mod asset_registry;
pub mod assimp_util;
pub mod camera;
pub mod compression;
pub mod data_cache;
pub mod data_util;
pub mod environment_import;
pub mod gpu_data;
pub mod handles;
pub mod mesh_geometry;
pub(crate) mod thread_pool;
pub mod validation;
