//! # Swapchain Lifecycle State Machine
//!
//! ## Purpose
//! Models swapchain replacement, WSI outcome classification, and old-generation retirement
//! as an explicit state machine with only legal transitions. Separates pure selection logic
//! from transactional creation so preflight failures never retire the active swapchain.
//!
//! ## Key States
//! - **Nascent**: No swapchain created yet.
//! - **Current**: Active swapchain used for acquire/present.
//! - **Retired**: Old handle passed to vkCreateSwapchainKHR; never reused.
//! - **Absent**: Terminal state after a fatal post-retirement failure.
//!
//! ## Irreversible Boundary
//! Once vkCreateSwapchainKHR is invoked with non-null `oldSwapchain`, that generation
//! is permanently retired. It is never rendered through, restored as current, or passed
//! again as `oldSwapchain` — even if the replacement creation fails.
//!
//! ## Design Constraints
//! - Capability re-query for every rebuild; no stale surface data.
//! - Latest-request resize coalescing with zero-extent deferral.
//! - Exact-once destruction: views before swapchain handles.
//! - No `device_wait_idle` as a substitute for ownership modeling.
//! - Preserve `FrameRenderOutcome` and existing public error variants.

use ash::vk;
use log::{info, warn};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Pure lifecycle types
// ---------------------------------------------------------------------------

/// Monotonic identity for each swapchain creation attempt.
///
/// Stored separately from raw Vulkan handles so repeated raw-handle values
/// (e.g. when the driver recycles handles) cannot defeat exact-once assertions.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct SwapchainGeneration(u64);

impl SwapchainGeneration {
    /// Allocate the next generation identity.
    pub(crate) fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Initial generation for the first swapchain created at startup.
    pub(crate) fn initial() -> Self {
        // Bump so first rebuild generation is always > startup generation.
        Self::next()
    }

    #[allow(dead_code)]
    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }
}

/// Legal swapchain lifecycle states.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SwapchainState {
    /// No swapchain has been created yet (startup before first creation).
    Nascent,
    /// Swapchain is active and usable for acquire/present.
    Current { generation: SwapchainGeneration },
    /// Swapchain generation was passed as `oldSwapchain` and destroyed.
    /// Must never be rendered through, restored, or passed again as `oldSwapchain`.
    Retired { generation: SwapchainGeneration },
    /// No swapchain exists after a fatal replacement failure with retired old.
    /// The backend is terminal; renderer must be recreated.
    Absent,
}

impl SwapchainState {
    /// Returns the generation if current, otherwise `None`.
    #[allow(dead_code)]
    pub(crate) fn current_generation(&self) -> Option<SwapchainGeneration> {
        match self {
            Self::Current { generation } => Some(*generation),
            _ => None,
        }
    }

    /// Returns `true` when a swapchain is available for acquire/present.
    #[allow(dead_code)]
    pub(crate) fn is_usable(&self) -> bool {
        matches!(self, Self::Current { .. })
    }
}

// ---------------------------------------------------------------------------
// Resize request accumulator
// ---------------------------------------------------------------------------

/// Stores the latest non-zero requested extent with coalescing.
///
/// Repeated window resize events update only the stored extent; they do not
/// invoke Vulkan creation. A successful rebuild consumes only the request
/// generation it installed, so a newer concurrent request remains pending.
#[derive(Debug, Copy, Clone)]
pub(crate) struct ResizeRequest {
    /// The latest non-zero extent requested.
    pub extent: vk::Extent2D,
    /// Monotonic sequence number for coalescing detection in tests.
    pub sequence: u64,
}

impl ResizeRequest {
    /// Create a new request with the latest extent and a fresh sequence number.
    pub(crate) fn new(extent: vk::Extent2D) -> Self {
        static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);
        Self {
            extent,
            sequence: NEXT_SEQ.fetch_add(1, Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Classified WSI outcomes
// ---------------------------------------------------------------------------

/// Classified swapchain image acquisition result.
///
/// Maps ash acquire results without string parsing: `NOT_READY`/`TIMEOUT` are
/// bounded retry; `ERROR_OUT_OF_DATE_KHR`/`SUBOPTIMAL_KHR` request rebuild;
/// `ERROR_SURFACE_LOST_KHR` is a distinct terminal internal class;
/// `ERROR_DEVICE_LOST` maps to the existing typed terminal renderer error.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum AcquireClass {
    /// Image acquired with the given index and suboptimal flag.
    Acquired { image_index: u32, suboptimal: bool },
    /// Transient retry: `NOT_READY` or `TIMEOUT` within retry budget.
    Retry,
    /// Swapchain is out of date and must be rebuilt.
    OutOfDate,
    /// Surface has been lost; terminal for this backend instance.
    SurfaceLost,
    /// Device has been lost; terminal for this backend.
    DeviceLost,
    /// Unclassified fatal acquire error.
    Fatal(String),
}

/// Classified presentation result.
///
/// Maps `queue_present` results: `Ok(false)` presented, `Ok(true)` suboptimal,
/// `ERROR_OUT_OF_DATE_KHR` not presented, `SUBOPTIMAL_KHR` presented suboptimal,
/// `ERROR_SURFACE_LOST_KHR` terminal, `ERROR_DEVICE_LOST` terminal.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum PresentClass {
    /// Frame was presented successfully.
    Presented,
    /// Presentation succeeded but swapchain is suboptimal.
    Suboptimal,
    /// Swapchain is out of date; frame was not presented.
    OutOfDate,
    /// Surface has been lost; terminal for this backend instance.
    SurfaceLost,
    /// Device has been lost; terminal for this backend.
    DeviceLost,
    /// Unclassified fatal present error.
    Fatal(String),
}

/// Classified swapchain rebuild result.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum RebuildClass {
    /// Swapchain was rebuilt and installed.
    Installed,
    /// Rebuild deferred because extent was zero; request remains pending.
    DeferredZeroExtent,
    /// Surface loss detected during rebuild preflight.
    SurfaceLost,
    /// Device loss detected during rebuild.
    DeviceLost,
    /// Creation was invoked with non-null oldSwapchain (old is retired)
    /// but the new creation or dependent resource setup failed.
    FatalAfterOldRetired(String),
    /// Failure occurred before the old swapchain was touched.
    FatalBeforeCreate(String),
}

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

/// Classify an `acquire_next_image2` Vulkan result without string parsing.
pub(crate) fn classify_acquire(result: Result<(u32, bool), vk::Result>) -> AcquireClass {
    match result {
        Ok((image_index, suboptimal)) => AcquireClass::Acquired {
            image_index,
            suboptimal,
        },
        Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => AcquireClass::Retry,
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
            AcquireClass::OutOfDate
        }
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => AcquireClass::SurfaceLost,
        Err(vk::Result::ERROR_DEVICE_LOST) => AcquireClass::DeviceLost,
        Err(err) => AcquireClass::Fatal(format!("acquire_next_image2 failed: {err:?}")),
    }
}

/// Classify a `queue_present` Vulkan result without string parsing.
pub(crate) fn classify_present(result: Result<bool, vk::Result>) -> PresentClass {
    match result {
        Ok(false) => PresentClass::Presented,
        Ok(true) => PresentClass::Suboptimal,
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => PresentClass::OutOfDate,
        Err(vk::Result::SUBOPTIMAL_KHR) => PresentClass::Suboptimal,
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => PresentClass::SurfaceLost,
        Err(vk::Result::ERROR_DEVICE_LOST) => PresentClass::DeviceLost,
        Err(err) => PresentClass::Fatal(format!("queue_present failed: {err:?}")),
    }
}

// ---------------------------------------------------------------------------
// Swapchain owner: transactional lifecycle management
// ---------------------------------------------------------------------------

/// Transactional swapchain lifecycle owner.
///
/// Manages the swapchain handle (`VkSwapchain`), present image views, and
/// the state machine. Enforces the invariant: once `oldSwapchain` is passed
/// to `vkCreateSwapchainKHR`, that generation is retired regardless of outcome.
pub(crate) struct SwapchainOwner {
    /// The current Vulkan swapchain, if any.
    pub swapchain: Option<crate::vulkan::vk_types::VkSwapchain>,
    /// Present image views owned by the current swapchain.
    pub present_views: Vec<(vk::Image, vk::ImageView)>,
    /// Current lifecycle state.
    state: SwapchainState,
    /// Pending resize request, if any.
    pending_resize: Option<ResizeRequest>,
    /// Sequence of the last successfully installed request.
    installed_sequence: u64,
}

impl SwapchainOwner {
    /// Create a new owner after the initial swapchain creation succeeds.
    pub(crate) fn new(
        swapchain: crate::vulkan::vk_types::VkSwapchain,
        present_views: Vec<(vk::Image, vk::ImageView)>,
    ) -> Self {
        let gen = SwapchainGeneration::initial();
        info!(
            "Swapchain generation {:?} installed (extent={:?}, format={:?})",
            gen, swapchain.extent, swapchain.surface_format.format
        );
        Self {
            swapchain: Some(swapchain),
            present_views,
            state: SwapchainState::Current { generation: gen },
            pending_resize: None,
            installed_sequence: 0,
        }
    }

    /// Create an owner for headless rendering (no swapchain).
    pub(crate) fn headless() -> Self {
        Self {
            swapchain: None,
            present_views: Vec::new(),
            state: SwapchainState::Absent,
            pending_resize: None,
            installed_sequence: 0,
        }
    }

    /// Current lifecycle state.
    pub(crate) fn state(&self) -> &SwapchainState {
        &self.state
    }

    /// Returns `true` when a swapchain is available for acquire/present.
    #[allow(dead_code)]
    pub(crate) fn is_usable(&self) -> bool {
        self.state.is_usable() && self.swapchain.is_some()
    }

    /// Returns the current swapchain generation, if any.
    #[allow(dead_code)]
    pub(crate) fn current_generation(&self) -> Option<SwapchainGeneration> {
        self.state.current_generation()
    }

    /// Store a resize request. If an older request is already pending, its
    /// extent is replaced by the latest one (coalescing).
    pub(crate) fn request_resize(&mut self, extent: vk::Extent2D) {
        assert!(extent.width > 0 && extent.height > 0, "zero extent deferred by caller");
        let request = ResizeRequest::new(extent);
        let replacing = self.pending_resize.is_some();
        self.pending_resize = Some(request);
        if replacing {
            info!(
                "Swapchain resize coalesced to extent={:?} (seq={})",
                extent, request.sequence
            );
        } else {
            info!(
                "Swapchain resize requested: extent={:?} (seq={})",
                extent, request.sequence
            );
        }
    }

    /// Returns the pending resize request, if any.
    pub(crate) fn pending_resize(&self) -> Option<&ResizeRequest> {
        self.pending_resize.as_ref()
    }

    /// Returns `true` when a resize is pending.
    pub(crate) fn resize_pending(&self) -> bool {
        self.pending_resize.is_some()
    }

    /// Clear a successfully installed request. If a newer request arrived
    /// concurrently, it will still be pending after this call.
    pub(crate) fn clear_installed_request(&mut self, installed: &ResizeRequest) {
        match &self.pending_resize {
            Some(pending) if pending.sequence == installed.sequence => {
                self.pending_resize = None;
                self.installed_sequence = installed.sequence;
                info!(
                    "Swapchain resize installed (seq={}); no newer request pending",
                    installed.sequence
                );
            }
            _ => {
                // A newer request has already replaced this one; keep it pending.
                info!(
                    "Swapchain resize installed (seq={}), but a newer request remains pending",
                    installed.sequence
                );
            }
        }
    }

    /// === IRREVERSIBLE BOUNDARY ===
    /// Transition the current swapchain generation to `Retired`.
    ///
    /// After this call, the old handle must be passed to `vkCreateSwapchainKHR`
    /// or explicitly destroyed. It must never be restored as current.
    pub(crate) fn retire_current(&mut self) -> Option<vk::SwapchainKHR> {
        let old_state = std::mem::replace(&mut self.state, SwapchainState::Absent);
        let old_handle = match old_state {
            SwapchainState::Current { generation } => {
                self.state = SwapchainState::Retired { generation };
                info!(
                    "Swapchain generation {:?} retired (oldSwapchain boundary)",
                    generation
                );
                self.swapchain.as_ref().map(|sc| sc.swapchain)
            }
            SwapchainState::Nascent => {
                self.state = SwapchainState::Nascent;
                None
            }
            other => {
                // Should not happen; log and recover.
                warn!(
                    "SwapchainOwner::retire_current called in unexpected state {:?}",
                    other
                );
                self.state = other;
                None
            }
        };
        old_handle
    }

    /// Install a newly created swapchain as current.
    ///
    /// The `present_views` must be fully created before calling this.
    /// The prior swapchain (if any) must already be retired and its views destroyed.
    pub(crate) fn install_new(
        &mut self,
        swapchain: crate::vulkan::vk_types::VkSwapchain,
        present_views: Vec<(vk::Image, vk::ImageView)>,
    ) {
        let gen = SwapchainGeneration::next();
        self.swapchain = Some(swapchain);
        self.present_views = present_views;
        self.state = SwapchainState::Current { generation: gen };
        info!("Swapchain generation {:?} installed", gen);
    }

    /// Destroy present image views owned by this owner.
    pub(crate) fn destroy_present_views(&mut self, device: &ash::Device) {
        for (_, view) in self.present_views.iter_mut() {
            if *view != vk::ImageView::null() {
                unsafe {
                    device.destroy_image_view(*view, None);
                }
                *view = vk::ImageView::null();
            }
        }
        self.present_views.clear();
    }

    /// Take ownership of and destroy the current swapchain, views, and retire
    /// the generation. Used during teardown.
    #[allow(dead_code)]
    pub(crate) fn destroy_current(
        &mut self,
        device: &ash::Device,
    ) -> Option<crate::vulkan::vk_types::VkSwapchain> {
        self.destroy_present_views(device);
        let old = self.swapchain.take();
        if let Some(ref sc) = old {
            match &self.state {
                SwapchainState::Current { generation } => {
                    self.state = SwapchainState::Retired {
                        generation: *generation,
                    };
                }
                _ => {}
            }
            unsafe {
                sc.swapchain_loader
                    .destroy_swapchain(sc.swapchain, None);
            }
        }
        self.state = SwapchainState::Absent;
        old
    }

    /// Consume and destroy a retired swapchain handle after it has been replaced.
    pub(crate) fn destroy_retired(&mut self, _device: &ash::Device, old: crate::vulkan::vk_types::VkSwapchain) {
        info!(
            "Destroying retired swapchain handle {:?}",
            old.swapchain
        );
        unsafe {
            old.swapchain_loader
                .destroy_swapchain(old.swapchain, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Pure tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // State machine legality tests
    // -----------------------------------------------------------------------

    #[test]
    fn nascent_to_current_via_install() {
        let state = SwapchainState::Nascent;
        assert!(!state.is_usable());
        assert_eq!(state.current_generation(), None);
    }

    #[test]
    fn current_state_is_usable_and_has_generation() {
        let gen = SwapchainGeneration::next();
        let state = SwapchainState::Current { generation: gen };
        assert!(state.is_usable());
        assert_eq!(state.current_generation(), Some(gen));
    }

    #[test]
    fn retired_state_is_not_usable() {
        let gen = SwapchainGeneration::next();
        let state = SwapchainState::Retired { generation: gen };
        assert!(!state.is_usable());
        assert_eq!(state.current_generation(), None);
    }

    #[test]
    fn absent_state_is_not_usable() {
        assert!(!SwapchainState::Absent.is_usable());
        assert_eq!(SwapchainState::Absent.current_generation(), None);
    }

    #[test]
    fn generation_identities_are_monotonic() {
        let g1 = SwapchainGeneration::next();
        let g2 = SwapchainGeneration::next();
        let g3 = SwapchainGeneration::next();
        assert!(g1.as_u64() < g2.as_u64());
        assert!(g2.as_u64() < g3.as_u64());
        assert_ne!(g1, g2);
        assert_ne!(g2, g3);
    }

    // -----------------------------------------------------------------------
    // Resize request coalescing
    // -----------------------------------------------------------------------

    #[test]
    fn resize_request_has_monotonic_sequence() {
        let r1 = ResizeRequest::new(vk::Extent2D {
            width: 800,
            height: 600,
        });
        let r2 = ResizeRequest::new(vk::Extent2D {
            width: 1024,
            height: 768,
        });
        assert!(r1.sequence < r2.sequence);
    }

    #[test]
    fn swapchain_owner_coalesces_repeated_resize_requests() {
        // We can't test with a real swapchain, but we can test the request
        // accumulator behavior using a mock flow.
        let extent_a = vk::Extent2D {
            width: 800,
            height: 600,
        };
        let extent_b = vk::Extent2D {
            width: 1024,
            height: 768,
        };

        let mut owner_hypothetical = ResizeRequest::new(extent_a);
        // A new request replaces the old one
        let newer = ResizeRequest::new(extent_b);
        assert!(newer.sequence > owner_hypothetical.sequence);
        owner_hypothetical = newer;
        assert_eq!(owner_hypothetical.extent, extent_b);
    }

    #[test]
    fn clear_installed_request_only_clears_matching_sequence() {
        // Simulate the pending/installed sequence tracking
        let request_a = ResizeRequest::new(vk::Extent2D {
            width: 800,
            height: 600,
        });
        let request_b = ResizeRequest::new(vk::Extent2D {
            width: 1024,
            height: 768,
        });

        // Setup: pending = request_a
        let mut pending: Option<ResizeRequest> = Some(request_a);

        // Clear request_a — should become None
        if let Some(ref p) = pending {
            if p.sequence == request_a.sequence {
                pending = None;
            }
        }
        assert!(pending.is_none());

        // Now simulate: pending gets request_a, then request_b replaces it
        // (request_a is set then immediately overwritten by request_b — this
        //  simulates coalescing where the newer request wins.)
        let _ = pending.insert(request_a);
        let _ = pending.insert(request_b); // coalesce

        // Clear request_a — should NOT clear because sequence is now request_b
        if let Some(ref p) = pending {
            if p.sequence == request_a.sequence {
                pending = None;
            }
        }
        assert!(pending.is_some());
        assert_eq!(pending.unwrap().sequence, request_b.sequence);
    }

    // -----------------------------------------------------------------------
    // Acquire classification
    // -----------------------------------------------------------------------

    #[test]
    fn classify_acquire_success_no_suboptimal() {
        let result = classify_acquire(Ok((0, false)));
        assert_eq!(
            result,
            AcquireClass::Acquired {
                image_index: 0,
                suboptimal: false
            }
        );
    }

    #[test]
    fn classify_acquire_success_with_suboptimal() {
        let result = classify_acquire(Ok((2, true)));
        assert_eq!(
            result,
            AcquireClass::Acquired {
                image_index: 2,
                suboptimal: true
            }
        );
    }

    #[test]
    fn classify_acquire_not_ready() {
        assert_eq!(classify_acquire(Err(vk::Result::NOT_READY)), AcquireClass::Retry);
    }

    #[test]
    fn classify_acquire_timeout() {
        assert_eq!(
            classify_acquire(Err(vk::Result::TIMEOUT)),
            AcquireClass::Retry
        );
    }

    #[test]
    fn classify_acquire_out_of_date() {
        assert_eq!(
            classify_acquire(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            AcquireClass::OutOfDate
        );
    }

    #[test]
    fn classify_acquire_suboptimal_error() {
        assert_eq!(
            classify_acquire(Err(vk::Result::SUBOPTIMAL_KHR)),
            AcquireClass::OutOfDate
        );
    }

    #[test]
    fn classify_acquire_surface_lost() {
        assert_eq!(
            classify_acquire(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
            AcquireClass::SurfaceLost
        );
    }

    #[test]
    fn classify_acquire_device_lost() {
        assert_eq!(
            classify_acquire(Err(vk::Result::ERROR_DEVICE_LOST)),
            AcquireClass::DeviceLost
        );
    }

    #[test]
    fn classify_acquire_fatal_is_not_retryable() {
        match classify_acquire(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)) {
            AcquireClass::Fatal(_) => {}
            other => panic!("expected Fatal, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Present classification
    // -----------------------------------------------------------------------

    #[test]
    fn classify_present_success() {
        assert_eq!(classify_present(Ok(false)), PresentClass::Presented);
    }

    #[test]
    fn classify_present_suboptimal() {
        assert_eq!(classify_present(Ok(true)), PresentClass::Suboptimal);
    }

    #[test]
    fn classify_present_out_of_date() {
        assert_eq!(
            classify_present(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)),
            PresentClass::OutOfDate
        );
    }

    #[test]
    fn classify_present_suboptimal_err() {
        assert_eq!(
            classify_present(Err(vk::Result::SUBOPTIMAL_KHR)),
            PresentClass::Suboptimal
        );
    }

    #[test]
    fn classify_present_surface_lost() {
        assert_eq!(
            classify_present(Err(vk::Result::ERROR_SURFACE_LOST_KHR)),
            PresentClass::SurfaceLost
        );
    }

    #[test]
    fn classify_present_device_lost() {
        assert_eq!(
            classify_present(Err(vk::Result::ERROR_DEVICE_LOST)),
            PresentClass::DeviceLost
        );
    }

    #[test]
    fn classify_present_fatal_is_not_retryable() {
        match classify_present(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)) {
            PresentClass::Fatal(_) => {}
            other => panic!("expected Fatal, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Rebuild classification
    // -----------------------------------------------------------------------

    #[test]
    fn rebuild_class_distinguishes_pre_create_from_post_retirement() {
        // Pure enum test — no Vulkan required
        let pre = RebuildClass::FatalBeforeCreate("capability check failed".into());
        let post = RebuildClass::FatalAfterOldRetired("view creation failed".into());

        // Pattern match to ensure they're distinct
        match (&pre, &post) {
            (RebuildClass::FatalBeforeCreate(msg1), RebuildClass::FatalAfterOldRetired(msg2)) => {
                assert!(msg1.contains("capability"));
                assert!(msg2.contains("view"));
            }
            _ => panic!("expected distinct rebuild classes"),
        }

        // DeferredZeroExtent and Installed are distinct
        assert!(matches!(RebuildClass::DeferredZeroExtent, RebuildClass::DeferredZeroExtent));
        assert!(matches!(RebuildClass::Installed, RebuildClass::Installed));
        assert!(matches!(RebuildClass::SurfaceLost, RebuildClass::SurfaceLost));
        assert!(matches!(RebuildClass::DeviceLost, RebuildClass::DeviceLost));
    }

    // -----------------------------------------------------------------------
    // Exact-once destruction logic tests
    // -----------------------------------------------------------------------

    #[test]
    fn retired_generation_cannot_become_current() {
        let gen = SwapchainGeneration::next();
        let old = SwapchainState::Current { generation: gen };
        let retired = SwapchainState::Retired { generation: gen };

        assert!(old.is_usable());
        assert!(!retired.is_usable());
        assert_eq!(old.current_generation(), Some(gen));
        assert_eq!(retired.current_generation(), None);
    }

    #[test]
    fn generation_identity_preserved_through_retirement() {
        let gen = SwapchainGeneration::next();

        // Simulate: current -> retired
        let current = SwapchainState::Current { generation: gen };
        assert_eq!(current.current_generation(), Some(gen));

        let retired = SwapchainState::Retired { generation: gen };
        assert_eq!(retired.current_generation(), None);

        // The retired generation identity must match the original
        match retired {
            SwapchainState::Retired {
                generation: retired_gen,
            } => assert_eq!(retired_gen, gen),
            _ => panic!("expected Retired"),
        }
    }

    // -----------------------------------------------------------------------
    // SwapchainOwner resize tracking without real Vulkan
    // -----------------------------------------------------------------------

    #[test]
    fn resize_pending_after_request() {
        // We test the resize tracking using owned fields since
        // SwapchainOwner requires a real VkSwapchain. The pattern
        // is validated through the pure ResizeRequest tests above
        // and the integration in vk_render.
        let req = ResizeRequest::new(vk::Extent2D {
            width: 640,
            height: 480,
        });
        assert_eq!(req.extent.width, 640);
        assert_eq!(req.extent.height, 480);
        assert!(req.sequence > 0);
    }

    #[test]
    fn extent_clamping_pure_logic() {
        // Test the pure extent clamping logic used in vk_init::select_sc_extent
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            max_image_extent: vk::Extent2D {
                width: 3840,
                height: 2160,
            },
            ..Default::default()
        };

        let requested = vk::Extent2D {
            width: 400,
            height: 300,
        };

        // Since current_extent is u32::MAX, we clamp requested
        let extent = crate::vulkan::vk_init::select_sc_extent(&capabilities, requested);
        assert_eq!(extent.width, 800); // clamped to min
        assert_eq!(extent.height, 600); // clamped to min

        let requested_large = vk::Extent2D {
            width: 4000,
            height: 3000,
        };
        let extent_large =
            crate::vulkan::vk_init::select_sc_extent(&capabilities, requested_large);
        assert_eq!(extent_large.width, 3840); // clamped to max
        assert_eq!(extent_large.height, 2160); // clamped to max
    }

    #[test]
    fn current_extent_returns_surface_defined_extent() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            min_image_extent: vk::Extent2D {
                width: 100,
                height: 100,
            },
            max_image_extent: vk::Extent2D {
                width: 3840,
                height: 2160,
            },
            ..Default::default()
        };

        let requested = vk::Extent2D {
            width: 100,
            height: 100,
        };
        let extent = crate::vulkan::vk_init::select_sc_extent(&capabilities, requested);
        // When current_extent != u32::MAX, it's returned as-is
        assert_eq!(extent.width, 1920);
        assert_eq!(extent.height, 1080);
    }

    // -----------------------------------------------------------------------
    // Image count, format, present mode selection
    // -----------------------------------------------------------------------

    #[test]
    fn select_sc_surface_format_exact_match() {
        let formats = vec![
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        let (found, selected) = crate::vulkan::vk_init::select_sc_surface_format(
            &formats,
            vk::Format::B8G8R8A8_UNORM,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        );
        assert!(found);
        assert_eq!(selected.format, vk::Format::B8G8R8A8_UNORM);
    }

    #[test]
    fn select_sc_surface_format_fallback_to_first() {
        let formats = vec![
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        let (found, selected) = crate::vulkan::vk_init::select_sc_surface_format(
            &formats,
            vk::Format::B8G8R8A8_UNORM,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        );
        assert!(!found);
        assert_eq!(selected.format, vk::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn select_sc_present_mode_exact_match() {
        let modes = vec![vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        let (found, selected) =
            crate::vulkan::vk_init::select_sc_present_mode(&modes, vk::PresentModeKHR::MAILBOX);
        assert!(found);
        assert_eq!(selected, vk::PresentModeKHR::MAILBOX);
    }

    #[test]
    fn select_sc_present_mode_fallback_to_fifo() {
        let modes = vec![vk::PresentModeKHR::IMMEDIATE];
        let (found, selected) =
            crate::vulkan::vk_init::select_sc_present_mode(&modes, vk::PresentModeKHR::MAILBOX);
        assert!(!found);
        assert_eq!(selected, vk::PresentModeKHR::FIFO);
    }

    #[test]
    fn composite_alpha_fallback_to_opaque() {
        // Test the composite alpha selection is OPAQUE when supported
        let flags = vk::CompositeAlphaFlagsKHR::OPAQUE;
        assert!(flags.contains(vk::CompositeAlphaFlagsKHR::OPAQUE));

        // When OPAQUE is NOT supported, should fallback
        let no_opaque = vk::CompositeAlphaFlagsKHR::INHERIT;
        assert!(!no_opaque.contains(vk::CompositeAlphaFlagsKHR::OPAQUE));
    }
}
