//! # Vulkan Backend Module
//!
//! Entry point for Vulkan subsystems: init, descriptors, pipelines, frame execution, memory,
//! and debug helpers.

pub(crate) mod vk_commands;
pub mod vk_debug;
pub mod vk_descriptor;
#[allow(dead_code)]
mod vk_device_budget;
pub(crate) mod vk_frame;
pub mod vk_init;
pub mod vk_pipeline;
pub mod vk_render;
pub mod vk_shadow;
pub mod vk_storage;
pub mod vk_swapchain;
pub mod vk_types;
pub mod vk_util;
