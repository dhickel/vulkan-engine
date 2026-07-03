//! Advanced rendering interop — feature-gated, alpha/unstable.
//!
//! This module is only available when the `advanced-interop` Cargo feature is
//! enabled. It provides unsafe raw backend access for engine-internal
//! experiments and expert diagnostics.
//!
//! # Stability
//!
//! **Alpha unstable.** The types and functions in this module may change or be
//! removed across alpha sprints without prior notice. There is no API
//! compatibility guarantee. Beginner applications should not depend on this
//! module.
//!
//! # Safety
//!
//! Functions in this module bypass the facade API's synchronization, descriptor
//! lifecycle, and swapchain safety invariants. Misuse can produce Vulkan
//! validation errors, GPU hangs, or undefined behavior. Prefer safe extension
//! points (`RenderHook`, debug views, frame capture) before resorting to raw
//! backend access.

use crate::vulkan::vk_render::VkRenderCore;

use super::renderer::Renderer;

/// Unsafe advanced interop for engine-internal Vulkan access.
///
/// Returns a mutable reference to the internal [`VkRenderCore`], bypassing all
/// facade invariants.
///
/// # Availability
///
/// This function is feature-gated behind `advanced-interop` and excluded from
/// the default safe API and `renderer::prelude`.
///
/// # Safety
///
/// The returned core handle bypasses facade invariants and may break frame
/// synchronization, descriptor lifecycle, or swapchain safety if misused.
/// Callers are responsible for maintaining all Vulkan synchronization and
/// resource ownership contracts.
///
/// # Stability
///
/// **Alpha unstable.** This function's signature and behavior may change across
/// alpha sprints.
pub unsafe fn renderer_core_mut(renderer: &mut Renderer) -> &mut VkRenderCore {
    renderer.raw_core_mut()
}
