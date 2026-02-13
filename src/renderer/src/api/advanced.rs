use crate::vulkan::vk_render::VkRenderCore;

use super::renderer::Renderer;

/// Unsafe advanced interop for engine-internal Vulkan access.
///
/// This is intentionally feature-gated (`advanced-interop`) and excluded from the safe default API.
///
/// # Safety
/// The returned core handle bypasses facade invariants and may break frame synchronization,
/// descriptor lifecycle, or swapchain safety if misused.
pub unsafe fn renderer_core_mut(renderer: &mut Renderer) -> &mut VkRenderCore {
    renderer.raw_core_mut()
}
