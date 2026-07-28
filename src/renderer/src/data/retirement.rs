//! Fence-Aware GPU Handle Retirement
//!
//! ## Purpose
//! Provides a generic, frame-serial retirement mechanism that invalidates slot+generation
//! handles immediately while delaying payload destruction and slot reuse until the last
//! referencing GPU frame completes.
//!
//! ## Key Concepts
//! - **FrameSerial**: Monotonically increasing submission identifier. Assigned at submit,
//!   advanced only by successful fence observations.
//! - **RetirementClass**: Semantic lifecycle category for assertions and auditing.
//! - **RetirementRecord<T>**: A payload held for GPU retirement, keyed by `retire_after`.
//! - **GpuRetirementQueue<T>**: Ordered queue that reaps records once `completed_serial`
//!   reaches or passes each record's `retire_after`.
//!
//! ## Safety Contract
//! - A stale handle must fail immediately after invalidation even while its payload is
//!   retained for GPU safety.
//! - A slot cannot be returned to its free list before all frames that referenced the
//!   old generation have completed.
//! - Completion advances only from successful fence observations; submit failure cannot
//!   fabricate a completed serial.
//! - Reserved default slots are non-retirable and non-reusable.
//! - Generation wrap must reject reuse before collision.

use crate::data::gpu_data::VkMeshBuffers;
use crate::vulkan::vk_types::{VkImageAlloc, VkSubAlloc};
use ash::vk;

/// Descriptor allocations that must be released with a retired payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescriptorReleaseData {
    /// Image/sampler descriptor sets to return to the descriptor allocator.
    pub image_descriptor_sets: Vec<vk::DescriptorSet>,
}

impl DescriptorReleaseData {
    /// Construct release data for one image descriptor set.
    pub fn image_descriptor(set: vk::DescriptorSet) -> Self {
        if set == vk::DescriptorSet::null() {
            Self::default()
        } else {
            Self {
                image_descriptor_sets: vec![set],
            }
        }
    }

    /// True when no descriptor allocator release is required.
    pub fn is_empty(&self) -> bool {
        self.image_descriptor_sets.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FrameSerial
// ---------------------------------------------------------------------------

/// Monotonically increasing frame submission identifier.
///
/// Assigned exactly once per successful GPU submission. Advanced only by successful
/// fence observations. Used as the retirement boundary: a record with `retire_after = S`
/// may be reaped once `completed_serial >= S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameSerial(u64);

impl FrameSerial {
    /// The zero serial — used when no submission has ever completed.
    pub const ZERO: Self = Self(0);

    /// Construct a `FrameSerial` from a raw value.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Return the next serial, or `None` if the counter would overflow.
    /// Overflow is treated as a terminal backend error; the engine must not wrap.
    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }

    /// Raw integer value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for FrameSerial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FrameSerial({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// RetirementClass
// ---------------------------------------------------------------------------

/// Semantic lifecycle category for a retired resource.
///
/// Every retirement record must declare its class so lifecycle assertions and
/// auditing can verify that each class is reaped before dependent phases proceed.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "remaining classes are contracts for dependent sprint phases"
    )
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetirementClass {
    /// Scene-node bounds metadata entry.
    BoundsEntry,
    /// Per-mesh collider recipe (convex hull or trimesh).
    ColliderRecipe,
    /// BVH leaf node referencing a mesh.
    BvhLeaf,
    /// LOD chain reference for a mesh group.
    LodChainReference,
    /// Frame-local instance record for instanced draws.
    InstanceRecord,
    /// Mesh geometry payload (VkMeshBuffers + suballocations).
    MeshGeometry,
    /// Material payload (SSBO suballocation + descriptor set).
    MaterialPayload,
    /// Texture geometry, view, and sampler.
    TextureGeometry,
    /// BSP arena retirement closure (all arena-owned resources).
    BspArenaRetirement,
}

impl std::fmt::Display for RetirementClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundsEntry => write!(f, "BoundsEntry"),
            Self::ColliderRecipe => write!(f, "ColliderRecipe"),
            Self::BvhLeaf => write!(f, "BvhLeaf"),
            Self::LodChainReference => write!(f, "LodChainReference"),
            Self::InstanceRecord => write!(f, "InstanceRecord"),
            Self::MeshGeometry => write!(f, "MeshGeometry"),
            Self::MaterialPayload => write!(f, "MaterialPayload"),
            Self::TextureGeometry => write!(f, "TextureGeometry"),
            Self::BspArenaRetirement => write!(f, "BspArenaRetirement"),
        }
    }
}

// ---------------------------------------------------------------------------
// RetirementRecord
// ---------------------------------------------------------------------------

/// A payload held until the GPU frame identified by `retire_after` completes.
#[derive(Debug, Clone)]
pub struct RetirementRecord<T> {
    /// Lifecycle class for assertions.
    pub class: RetirementClass,
    /// The greatest frame serial that references this payload.
    /// Reaping is permitted when `completed_serial >= retire_after`.
    pub retire_after: FrameSerial,
    /// The payload to destroy after reaping.
    pub payload: T,
}

// ---------------------------------------------------------------------------
// GpuRetirementQueue
// ---------------------------------------------------------------------------

/// Ordered queue of [`RetirementRecord`]s awaiting GPU completion.
///
/// Records are sorted by `retire_after` then by stable insertion order.
/// `reap_through` removes and returns every record whose `retire_after` is
/// less than or equal to the supplied completed serial.
pub struct GpuRetirementQueue<T> {
    records: Vec<RetirementRecord<T>>,
    last_reaped_through: FrameSerial,
}

/// Invalid completion observations rejected by [`GpuRetirementQueue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetirementError {
    CompletionRegressed {
        previous: FrameSerial,
        observed: FrameSerial,
    },
}

/// Error returned by [`GpuRetirementQueue::try_reap`].
///
/// Either a completion regression was detected or the caller-supplied
/// closure failed. On closure failure the queue is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TryReapError<E> {
    CompletionRegressed {
        previous: FrameSerial,
        observed: FrameSerial,
    },
    ClosureFailed(E),
}

impl<T> GpuRetirementQueue<T> {
    /// Create an empty retirement queue.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            last_reaped_through: FrameSerial::ZERO,
        }
    }

    /// Enqueue a payload for retirement.
    ///
    /// Insertion maintains sort by `retire_after` ascending; ties preserve
    /// insertion order so tests are deterministic.
    pub fn enqueue(&mut self, class: RetirementClass, retire_after: FrameSerial, payload: T) {
        let record = RetirementRecord {
            class,
            retire_after,
            payload,
        };
        // Insert after all records with an equal serial to preserve insertion order.
        let pos = self
            .records
            .partition_point(|r| r.retire_after <= retire_after);
        self.records.insert(pos, record);
    }

    /// Reap all records whose `retire_after` is less than or equal to `completed_serial`.
    ///
    /// Returns the reaped records in retirement order. Records whose `retire_after`
    /// exceeds `completed_serial` remain in the queue.
    ///
    /// Duplicate completion is idempotent. A regressive completion is rejected so callers
    /// cannot silently move the retirement boundary backwards.
    pub fn reap_through(
        &mut self,
        completed_serial: FrameSerial,
    ) -> Result<Vec<RetirementRecord<T>>, RetirementError> {
        if completed_serial < self.last_reaped_through {
            return Err(RetirementError::CompletionRegressed {
                previous: self.last_reaped_through,
                observed: completed_serial,
            });
        }
        self.last_reaped_through = completed_serial;

        let split_idx = self
            .records
            .partition_point(|r| r.retire_after <= completed_serial);
        Ok(self.records.drain(..split_idx).collect())
    }

    /// Try to reap eligible records within a caller-supplied closure.
    ///
    /// Validates completion, calls `f` with a shared reference to every
    /// record whose `retire_after` ≤ `completed_serial`, and commits
    /// removal only when `f` returns `Ok`. On closure failure the queue
    /// retains every record unchanged.
    ///
    /// Completion regression is rejected before the closure runs.
    /// Duplicate completion with no eligible records advances the
    /// high-water mark idempotently without calling `f`.
    pub(crate) fn try_reap<F, E>(
        &mut self,
        completed_serial: FrameSerial,
        f: F,
    ) -> Result<Vec<RetirementRecord<T>>, TryReapError<E>>
    where
        F: FnOnce(&[RetirementRecord<T>]) -> Result<(), E>,
    {
        if completed_serial < self.last_reaped_through {
            return Err(TryReapError::CompletionRegressed {
                previous: self.last_reaped_through,
                observed: completed_serial,
            });
        }

        let split_idx = self
            .records
            .partition_point(|r| r.retire_after <= completed_serial);

        if split_idx == 0 {
            self.last_reaped_through = completed_serial;
            return Ok(Vec::new());
        }

        // Hand references to the caller. The queue still owns every record.
        f(&self.records[..split_idx]).map_err(TryReapError::ClosureFailed)?;

        // Commit: advance high-water mark and remove exactly the records
        // that were validated.
        self.last_reaped_through = completed_serial;
        Ok(self.records.drain(..split_idx).collect())
    }

    /// Number of pending records for a specific retirement class.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "lifecycle assertion API is consumed by dependent stores"
        )
    )]
    pub fn pending_by_class(&self, class: RetirementClass) -> usize {
        self.records.iter().filter(|r| r.class == class).count()
    }

    /// Total number of pending records.
    #[cfg_attr(not(test), allow(dead_code, reason = "retirement diagnostics API"))]
    pub fn pending_count(&self) -> usize {
        self.records.len()
    }

    /// True when no records are pending.
    #[cfg_attr(not(test), allow(dead_code, reason = "retirement diagnostics API"))]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl<T> Default for GpuRetirementQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MeshRetiredPayload — mesh-specific retirement payload
// ---------------------------------------------------------------------------

/// Payload held in the retirement queue for a retired mesh.
///
/// Contains the suballocation handles that must be deallocated after GPU
/// completion and the cache slot to release for reuse.
#[derive(Debug, Clone)]
pub struct MeshRetiredPayload {
    /// The cache slot to release after reaping.
    pub slot: u32,
    /// Complete copied GPU draw metadata, including both suballocations.
    pub buffers: VkMeshBuffers,
}

/// Payload held in the retirement queue for a retired material.
///
/// Contains the SSBO suballocation and descriptor set that must be
/// released after GPU completion, plus the cache slot to release for reuse.
/// Texture ownership is NOT cascaded — shared/default textures survive
/// unrelated material unload.
#[derive(Debug, Clone)]
pub struct MaterialRetiredPayload {
    /// The cache slot to release after reaping.
    pub slot: u32,
    /// Generation invalidated at retirement time.
    pub generation: u32,
    /// Suballocation in the material metadata SSBO.
    pub meta_alloc: VkSubAlloc,
    /// Descriptor allocator release data owned by this material payload.
    pub descriptor_release: DescriptorReleaseData,
}

/// Payload held in the retirement queue for a retired texture.
///
/// Contains the Vulkan image, image view, and sampler that must be
/// destroyed after GPU completion, plus the cache slot to release for reuse.
#[derive(Debug)]
pub struct TextureRetiredPayload {
    /// The cache slot to release after reaping.
    pub slot: u32,
    /// Generation invalidated at retirement time.
    pub generation: u32,
    /// Image allocation and view to destroy on reap.
    pub alloc: VkImageAlloc,
    /// Sampler to destroy on reap.
    pub sampler: vk::Sampler,
    /// Descriptor allocator release data owned by this texture payload.
    /// Textures currently own no descriptor sets; material payloads own sampler descriptors.
    pub descriptor_release: DescriptorReleaseData,
}

// ---------------------------------------------------------------------------
// BSP Retirement Closure
// ---------------------------------------------------------------------------

/// Complete retirement closure for one BSP mount arena.
///
/// Carries every arena-owned GPU resource for fence-aware destruction.
/// Borrowed defaults are never included. The closure is enqueued once and
/// reaped in dependency order: descriptor pools → surfaces/frames → atlas
/// → mesh/texture/material cache slots.
#[cfg(feature = "bsp")]
#[derive(Debug)]
pub struct BspRetirementClosure {
    /// Arena identity that owns this closure.
    pub arena_id: u64,
    /// Lightmap atlas payload (image, view, sampler) — destroyed after pool.
    pub lightmap_atlas: Option<crate::data::data_cache::BspLightmapAtlasGpu>,
    /// Surface UBO buffer/allocation.
    pub surface_ubo: Option<crate::data::data_cache::BspSurfaceUboGpu>,
    /// Frame-values UBO buffer/allocation.
    pub frame_values_ubo: Option<crate::data::data_cache::BspSurfaceUboGpu>,
    /// Material descriptor pool — freed first (releases all its sets).
    pub material_desc_pool: Option<vk::DescriptorPool>,
    /// Frame-values descriptor pool — freed after material pool.
    pub frame_values_desc_pool: Option<vk::DescriptorPool>,
    /// Material slots to invalidate in the surface cache.
    pub material_slots: Vec<u32>,
    /// Mesh handles to deallocate from the mesh cache.
    pub mesh_handles: Vec<crate::data::handles::MeshHandle>,
    /// Texture handles to deallocate from the texture cache.
    pub texture_handles: Vec<crate::data::handles::TextureHandle>,
}

// ---------------------------------------------------------------------------
// Test-only fault hooks
// ---------------------------------------------------------------------------

#[cfg(test)]
impl<T> GpuRetirementQueue<T> {
    /// Access records for state inspection after fault injection.
    fn records_snapshot(&self) -> &[RetirementRecord<T>] {
        &self.records
    }

    /// Return the high-water serial for completion-regression assertions.
    fn last_reaped_through(&self) -> FrameSerial {
        self.last_reaped_through
    }

    /// Replace the last-reaped-through serial (fault injection).
    fn set_last_reaped_through(&mut self, serial: FrameSerial) {
        self.last_reaped_through = serial;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FrameSerial ---

    #[test]
    fn frame_serial_next_succeeds() {
        let s = FrameSerial::ZERO;
        assert_eq!(s.next().unwrap(), FrameSerial(1));
        assert_eq!(FrameSerial(42).next().unwrap(), FrameSerial(43));
    }

    #[test]
    fn frame_serial_next_overflow_returns_none() {
        assert!(FrameSerial(u64::MAX).next().is_none());
    }

    #[test]
    fn frame_serial_ordering() {
        assert!(FrameSerial(0) < FrameSerial(1));
        assert!(FrameSerial(10) <= FrameSerial(10));
        assert!(FrameSerial(5) > FrameSerial(3));
    }

    // --- GpuRetirementQueue ---

    #[test]
    fn queue_empty_by_default() {
        let q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn enqueue_and_reap_single() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(5), 42);

        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.pending_by_class(RetirementClass::MeshGeometry), 1);
        assert_eq!(q.pending_by_class(RetirementClass::BoundsEntry), 0);

        // Not yet eligible
        let reaped = q.reap_through(FrameSerial(4)).unwrap();
        assert!(reaped.is_empty());
        assert_eq!(q.pending_count(), 1);

        // Exactly at boundary
        let reaped = q.reap_through(FrameSerial(5)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 42);
        assert!(q.is_empty());
    }

    #[test]
    fn reaps_in_order() {
        let mut q: GpuRetirementQueue<&str> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(10), "third");
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(3), "first");
        q.enqueue(RetirementClass::BvhLeaf, FrameSerial(7), "second");

        let reaped = q.reap_through(FrameSerial(10)).unwrap();
        assert_eq!(reaped.len(), 3);
        assert_eq!(reaped[0].payload, "first");
        assert_eq!(reaped[1].payload, "second");
        assert_eq!(reaped[2].payload, "third");
    }

    #[test]
    fn partial_reap_leaves_future_records() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(2), 100);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 200);
        q.enqueue(RetirementClass::BvhLeaf, FrameSerial(8), 300);

        let reaped = q.reap_through(FrameSerial(4)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 100);

        assert_eq!(q.pending_count(), 2);

        let reaped = q.reap_through(FrameSerial(7)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 200);
        assert_eq!(q.pending_count(), 1);

        let reaped = q.reap_through(FrameSerial(10)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 300);
        assert!(q.is_empty());
    }

    #[test]
    fn reap_is_idempotent() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(2), 10);

        assert_eq!(q.reap_through(FrameSerial(5)).unwrap().len(), 1);
        assert!(q.reap_through(FrameSerial(5)).unwrap().is_empty());
        assert!(q.reap_through(FrameSerial(10)).unwrap().is_empty());
        assert!(matches!(
            q.reap_through(FrameSerial(2)),
            Err(RetirementError::CompletionRegressed {
                previous: FrameSerial(10),
                observed: FrameSerial(2),
            })
        ));
    }

    #[test]
    fn duplicate_completion_idempotent() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), 1);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), 2);

        let reaped = q.reap_through(FrameSerial(1)).unwrap();
        assert_eq!(reaped.len(), 2);
        assert!(q.reap_through(FrameSerial(1)).unwrap().is_empty());
    }

    #[test]
    fn out_of_order_enqueue_maintains_sort() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(9), 9);
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(1), 1);
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(5), 5);

        let reaped = q.reap_through(FrameSerial(10)).unwrap();
        assert_eq!(reaped.len(), 3);
        assert_eq!(reaped[0].payload, 1);
        assert_eq!(reaped[1].payload, 5);
        assert_eq!(reaped[2].payload, 9);
    }

    #[test]
    fn same_serial_stable_insertion_order() {
        let mut q: GpuRetirementQueue<&str> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), "A");
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(3), "B");
        q.enqueue(RetirementClass::BvhLeaf, FrameSerial(3), "C");

        let reaped = q.reap_through(FrameSerial(3)).unwrap();
        assert_eq!(reaped.len(), 3);
        assert_eq!(reaped[0].payload, "A");
        assert_eq!(reaped[1].payload, "B");
        assert_eq!(reaped[2].payload, "C");
    }

    #[test]
    fn pending_by_class_counts_correctly() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(1), 1);
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(2), 2);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(3), 3);
        q.enqueue(RetirementClass::ColliderRecipe, FrameSerial(4), 4);
        q.enqueue(RetirementClass::BvhLeaf, FrameSerial(5), 5);
        q.enqueue(RetirementClass::LodChainReference, FrameSerial(6), 6);
        q.enqueue(RetirementClass::InstanceRecord, FrameSerial(7), 7);

        assert_eq!(q.pending_by_class(RetirementClass::MeshGeometry), 2);
        assert_eq!(q.pending_by_class(RetirementClass::BoundsEntry), 1);
        assert_eq!(q.pending_by_class(RetirementClass::ColliderRecipe), 1);
        assert_eq!(q.pending_by_class(RetirementClass::BvhLeaf), 1);
        assert_eq!(q.pending_by_class(RetirementClass::LodChainReference), 1);
        assert_eq!(q.pending_by_class(RetirementClass::InstanceRecord), 1);
    }

    // --- fake-fence multi-slot test ---

    #[test]
    fn fake_fence_multi_slot_no_premature_reap() {
        // Simulate three frame slots submitting interleaved work.
        // Slot A submits at serial 1 and 4. Slot B at 2 and 5. Slot C at 3.
        // Completion advances slot-by-slot: first A completes (serial 1),
        // then B (2), then C (3). Records for serial 4 and 5 stay pending
        // until those slots complete too.
        let mut q: GpuRetirementQueue<&str> = GpuRetirementQueue::new();

        // Enqueue records that must not be reaped before their referencing serials.
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(1), "mesh-A1");
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(2), "bounds-B2");
        q.enqueue(
            RetirementClass::ColliderRecipe,
            FrameSerial(3),
            "collider-C3",
        );
        q.enqueue(RetirementClass::BvhLeaf, FrameSerial(4), "bvh-A4");
        q.enqueue(RetirementClass::LodChainReference, FrameSerial(5), "lod-B5");

        // completed_serial = 1: only "mesh-A1" eligible
        let reaped = q.reap_through(FrameSerial(1)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, "mesh-A1");
        assert_eq!(q.pending_count(), 4);

        // completed_serial = 2: "bounds-B2" eligible
        let reaped = q.reap_through(FrameSerial(2)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, "bounds-B2");
        assert_eq!(q.pending_count(), 3);

        // completed_serial = 3: "collider-C3" eligible
        let reaped = q.reap_through(FrameSerial(3)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, "collider-C3");
        assert_eq!(q.pending_count(), 2);

        // Records at serial 4 and 5 still pending
        assert_eq!(q.pending_by_class(RetirementClass::BvhLeaf), 1);
        assert_eq!(q.pending_by_class(RetirementClass::LodChainReference), 1);

        // completed_serial = 5: both remaining eligible
        let reaped = q.reap_through(FrameSerial(5)).unwrap();
        assert_eq!(reaped.len(), 2);
        assert!(q.is_empty());
    }

    #[test]
    fn fence_boundary_delays_drop_and_slot_reuse_exactly_once() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct TrackedPayload {
            slot: u32,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for TrackedPayload {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let old_generation = 7;
        let live_generation = old_generation + 1; // immediate invalidation
        let mut free_slots = Vec::new();
        let mut q = GpuRetirementQueue::new();
        q.enqueue(
            RetirementClass::MeshGeometry,
            FrameSerial(4),
            TrackedPayload {
                slot: 3,
                drops: Arc::clone(&drops),
            },
        );

        assert_ne!(
            old_generation, live_generation,
            "old handle must be stale immediately"
        );
        assert!(free_slots.is_empty(), "slot reused before its fence");
        assert!(q.reap_through(FrameSerial(2)).unwrap().is_empty());
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "payload freed before its fence"
        );
        assert!(free_slots.is_empty(), "slot reused before its fence");

        let retired = q.reap_through(FrameSerial(4)).unwrap();
        assert_eq!(retired.len(), 1);
        let released_slot = retired[0].payload.slot;
        drop(retired); // physical destruction precedes slot release
        free_slots.push(released_slot);
        assert_eq!(free_slots, vec![3]);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(q.reap_through(FrameSerial(4)).unwrap().is_empty());
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "duplicate completion double-dropped"
        );
    }

    #[test]
    fn out_of_order_slot_observation_uses_monotonic_completion_high_water() {
        let mut q = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), "slot-0-old");
        q.enqueue(
            RetirementClass::InstanceRecord,
            FrameSerial(2),
            "slot-1-new",
        );
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), "slot-0-next");

        // Slot 1 is observed first. Because submissions share one ordered graphics queue,
        // its serial-2 fence proves serials 1 and 2 complete even though slot 0 was not polled.
        let mut completion_high_water = FrameSerial::ZERO;
        completion_high_water = completion_high_water.max(FrameSerial(2));
        let reaped = q.reap_through(completion_high_water).unwrap();
        assert_eq!(
            reaped
                .iter()
                .map(|record| record.payload)
                .collect::<Vec<_>>(),
            vec!["slot-0-old", "slot-1-new"]
        );
        assert_eq!(q.pending_count(), 1);

        // A later stale/duplicate slot observation cannot regress the high-water mark.
        completion_high_water = completion_high_water.max(FrameSerial(1));
        assert!(q.reap_through(completion_high_water).unwrap().is_empty());
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn stale_handle_never_revived_by_reap() {
        // A retirement queue does not resurrect handles — it only releases
        // slots. The generation bump at invalidation ensures the old handle
        // remains stale regardless of when reaping occurs.
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(10), 42);

        // Before reaping, the payload is still held.
        assert_eq!(q.pending_count(), 1);

        // After reaping, the record is returned but the old generation is
        // already incremented elsewhere; the queue does not manage generations.
        let reaped = q.reap_through(FrameSerial(10)).unwrap();
        assert_eq!(reaped[0].payload, 42);
        assert!(q.is_empty());
    }

    #[test]
    fn all_classes_exercised() {
        let classes = [
            RetirementClass::BoundsEntry,
            RetirementClass::ColliderRecipe,
            RetirementClass::BvhLeaf,
            RetirementClass::LodChainReference,
            RetirementClass::InstanceRecord,
            RetirementClass::MeshGeometry,
            RetirementClass::MaterialPayload,
            RetirementClass::TextureGeometry,
        ];

        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        for (i, &cls) in classes.iter().enumerate() {
            q.enqueue(cls, FrameSerial(i as u64 + 1), i as u32);
        }

        assert_eq!(q.pending_count(), 8);
        let reaped = q.reap_through(FrameSerial(8)).unwrap();
        assert_eq!(reaped.len(), 8);
        for (i, rec) in reaped.iter().enumerate() {
            assert_eq!(rec.class, classes[i]);
            assert_eq!(rec.payload, i as u32);
        }
    }

    // ── try_reap regression tests ───────────────────────────────────────

    #[test]
    fn try_reap_success_removes_eligible() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), 100);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 200);

        let mut seen = Vec::new();
        let reaped = q
            .try_reap(FrameSerial(4), |eligible| {
                seen.extend(eligible.iter().map(|r| r.payload));
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(seen, vec![100]);
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 100);
        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.pending_by_class(RetirementClass::BoundsEntry), 1);
    }

    #[test]
    fn try_reap_closure_failure_leaves_queue_unchanged() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), 100);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 200);

        let snapshot_before: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| (r.class, r.retire_after, r.payload))
            .collect();

        let result = q.try_reap(FrameSerial(4), |_eligible| Err("simulated lock failure"));
        assert!(matches!(
            result,
            Err(TryReapError::ClosureFailed("simulated lock failure"))
        ));

        // Queue unchanged: same count, same records
        assert_eq!(q.pending_count(), 2);
        let snapshot_after: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| (r.class, r.retire_after, r.payload))
            .collect();
        assert_eq!(snapshot_before, snapshot_after);
        assert_eq!(q.last_reaped_through(), FrameSerial::ZERO);
    }

    #[test]
    fn try_reap_retry_after_failure_succeeds() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(2), 42);

        // First attempt: simulated failure
        let result = q.try_reap(FrameSerial(5), |_| Err("fail"));
        assert!(result.is_err());
        assert_eq!(q.pending_count(), 1);

        // Retry succeeds
        let reaped = q
            .try_reap(FrameSerial(5), |eligible| {
                assert_eq!(eligible.len(), 1);
                assert_eq!(eligible[0].payload, 42);
                Ok::<(), &str>(())
            })
            .unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 42);
        assert!(q.is_empty());
    }

    #[test]
    fn try_reap_mixed_eligible_future() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), 10);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(3), 20);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 30);

        let reaped = q
            .try_reap(FrameSerial(2), |eligible| {
                assert_eq!(eligible.len(), 1);
                assert_eq!(eligible[0].payload, 10);
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 10);
        assert_eq!(q.pending_count(), 2);

        // Closure failure on future-only reap leaves queue unchanged
        let err = q.try_reap(FrameSerial(4), |eligible| {
            assert_eq!(eligible.len(), 1);
            assert_eq!(eligible[0].payload, 20);
            Err("fail")
        });
        assert!(err.is_err());
        assert_eq!(q.pending_count(), 2);
        assert_eq!(q.records_snapshot()[0].payload, 20);
        assert_eq!(q.records_snapshot()[1].payload, 30);
    }

    #[test]
    fn try_reap_equal_serial_ordering() {
        let mut q: GpuRetirementQueue<&str> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(3), "A");
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), "B");
        q.enqueue(RetirementClass::ColliderRecipe, FrameSerial(3), "C");

        let reaped = q
            .try_reap(FrameSerial(3), |eligible| {
                assert_eq!(eligible.len(), 3);
                assert_eq!(eligible[0].payload, "A");
                assert_eq!(eligible[1].payload, "B");
                assert_eq!(eligible[2].payload, "C");
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(reaped.len(), 3);
        assert!(q.is_empty());
    }

    #[test]
    fn try_reap_completion_regression_rejected() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 10);

        // Advance to serial 5 and reap
        let reaped = q.reap_through(FrameSerial(5)).unwrap();
        assert_eq!(reaped.len(), 1);

        // Regression attempt via try_reap
        let result = q.try_reap(FrameSerial(3), |_| Ok::<(), ()>(()));
        assert!(matches!(
            result,
            Err(TryReapError::CompletionRegressed {
                previous: FrameSerial(5),
                observed: FrameSerial(3),
            })
        ));
    }

    #[test]
    fn try_reap_idempotent_no_eligible() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 10);

        // No eligible records, closure not called
        let reaped = q
            .try_reap(FrameSerial(2), |_| -> Result<(), ()> {
                panic!("closure must not be called")
            })
            .unwrap();
        assert!(reaped.is_empty());
        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.last_reaped_through(), FrameSerial(2));
    }

    #[test]
    fn try_reap_closure_receives_exact_eligible_slice() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), 1);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(2), 2);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(4), 4);
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 5);

        q.try_reap(FrameSerial(2), |eligible| {
            assert_eq!(eligible.len(), 2);
            assert_eq!(eligible[0].retire_after, FrameSerial(1));
            assert_eq!(eligible[1].retire_after, FrameSerial(2));
            Ok::<(), ()>(())
        })
        .unwrap();
    }

    #[test]
    fn try_reap_preserves_exact_class_and_payload_identity_on_failure() {
        // Simulates lock-poison: eligible records must retain every
        // observable field (class, payload, slot for relevant payloads).
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MaterialPayload, FrameSerial(2), 77);
        q.enqueue(RetirementClass::TextureGeometry, FrameSerial(3), 88);

        let snapshot: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| (r.class, r.retire_after.raw(), r.payload))
            .collect();

        let _ = q.try_reap(FrameSerial(3), |_| Err("poison"));
        assert_eq!(q.pending_count(), 2);

        let after: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| (r.class, r.retire_after.raw(), r.payload))
            .collect();
        assert_eq!(snapshot, after);
    }

    #[test]
    fn try_reap_no_duplicate_slot_release() {
        // After a successful reap, a second reap must return empty.
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(1), 10);

        let first = q.try_reap(FrameSerial(1), |_| Ok::<(), ()>(())).unwrap();
        assert_eq!(first.len(), 1);

        let second = q.try_reap(FrameSerial(1), |_| Ok::<(), ()>(())).unwrap();
        assert!(second.is_empty());
    }

    /// Simulated descriptor-release retention: a custom payload that
    /// carries a descriptor set vector must survive closure failure.
    #[test]
    fn try_reap_descriptor_release_retained_on_failure() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct FakeMaterialPayload {
            slot: u32,
            generation: u32,
            descriptor_sets: Vec<u32>,
        }

        let mut q: GpuRetirementQueue<FakeMaterialPayload> = GpuRetirementQueue::new();
        q.enqueue(
            RetirementClass::MaterialPayload,
            FrameSerial(1),
            FakeMaterialPayload {
                slot: 5,
                generation: 3,
                descriptor_sets: vec![100, 101],
            },
        );
        q.enqueue(
            RetirementClass::MaterialPayload,
            FrameSerial(2),
            FakeMaterialPayload {
                slot: 7,
                generation: 1,
                descriptor_sets: vec![200],
            },
        );

        let snapshot: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| {
                (
                    r.class,
                    r.retire_after.raw(),
                    r.payload.slot,
                    r.payload.generation,
                    r.payload.descriptor_sets.clone(),
                )
            })
            .collect();

        let _ = q.try_reap(FrameSerial(2), |_| Err("lock poison"));
        assert_eq!(q.pending_count(), 2);

        let after: Vec<_> = q
            .records_snapshot()
            .iter()
            .map(|r| {
                (
                    r.class,
                    r.retire_after.raw(),
                    r.payload.slot,
                    r.payload.generation,
                    r.payload.descriptor_sets.clone(),
                )
            })
            .collect();
        assert_eq!(snapshot, after);
    }

    /// Terminal cleanup: after a failed reap is retried and succeeds,
    /// the queue is empty and reusable.
    #[test]
    fn try_reap_terminal_cleanup_after_retry() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(1), 42);
        q.enqueue(RetirementClass::TextureGeometry, FrameSerial(2), 99);

        // Fail first
        let _ = q.try_reap(FrameSerial(2), |_| Err("fail"));

        // Succeed
        let reaped = q
            .try_reap(FrameSerial(2), |eligible| {
                assert_eq!(eligible.len(), 2);
                Ok::<(), &str>(())
            })
            .unwrap();
        assert_eq!(reaped.len(), 2);
        assert!(q.is_empty());
        assert_eq!(q.last_reaped_through(), FrameSerial(2));

        // Queue is still valid for new enqueues
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(5), 55);
        assert_eq!(q.pending_count(), 1);
        let reaped = q.reap_through(FrameSerial(5)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload, 55);
    }

    /// Completion regression: a completed serial must never regress.
    #[test]
    fn try_reap_rejects_completion_regression() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.reap_through(FrameSerial(10)).unwrap();

        let result = q.try_reap(FrameSerial(9), |_| Ok::<(), ()>(()));
        assert!(matches!(
            result,
            Err(TryReapError::CompletionRegressed {
                previous: FrameSerial(10),
                observed: FrameSerial(9),
            })
        ));
    }

    /// Verify that the fault-injection hook for last_reaped_through
    /// correctly simulates a regressed state.
    #[test]
    fn fault_hook_set_last_reaped_through() {
        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        q.set_last_reaped_through(FrameSerial(42));
        assert_eq!(q.last_reaped_through(), FrameSerial(42));

        // Regression from the injected state
        let result = q.try_reap(FrameSerial(10), |_| Ok::<(), ()>(()));
        assert!(matches!(
            result,
            Err(TryReapError::CompletionRegressed {
                previous: FrameSerial(42),
                observed: FrameSerial(10),
            })
        ));
    }

    // ── BSP retirement closure tests ──────────────────────────────────

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_closure_is_debug() {
        let closure = crate::data::retirement::BspRetirementClosure {
            arena_id: 1,
            lightmap_atlas: None,
            surface_ubo: None,
            frame_values_ubo: None,
            material_desc_pool: None,
            frame_values_desc_pool: None,
            material_slots: vec![1, 2],
            mesh_handles: vec![],
            texture_handles: vec![],
        };
        let debug_str = format!("{:?}", closure);
        assert!(debug_str.contains("BspRetirementClosure"));
        assert!(debug_str.contains("arena_id: 1"));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_closure_enqueue_reap_cycle() {
        let mut q: GpuRetirementQueue<crate::data::retirement::BspRetirementClosure> =
            GpuRetirementQueue::new();

        let closure = crate::data::retirement::BspRetirementClosure {
            arena_id: 7,
            lightmap_atlas: None,
            surface_ubo: None,
            frame_values_ubo: None,
            material_desc_pool: None,
            frame_values_desc_pool: None,
            material_slots: vec![0, 3],
            mesh_handles: vec![],
            texture_handles: vec![],
        };

        q.enqueue(RetirementClass::BspArenaRetirement, FrameSerial(5), closure);
        assert_eq!(q.pending_count(), 1);
        assert_eq!(
            q.pending_by_class(RetirementClass::BspArenaRetirement),
            1
        );

        // Not yet eligible
        let reaped = q.reap_through(FrameSerial(4)).unwrap();
        assert!(reaped.is_empty());
        assert_eq!(q.pending_count(), 1);

        // Eligible at serial 5
        let reaped = q.reap_through(FrameSerial(5)).unwrap();
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].payload.arena_id, 7);
        assert!(q.is_empty());
    }
}
