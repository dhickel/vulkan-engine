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
    allow(dead_code, reason = "remaining classes are contracts for dependent sprint phases")
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

    /// Number of pending records for a specific retirement class.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "lifecycle assertion API is consumed by dependent stores")
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
        q.enqueue(RetirementClass::ColliderRecipe, FrameSerial(3), "collider-C3");
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

        assert_ne!(old_generation, live_generation, "old handle must be stale immediately");
        assert!(free_slots.is_empty(), "slot reused before its fence");
        assert!(q.reap_through(FrameSerial(2)).unwrap().is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 0, "payload freed before its fence");
        assert!(free_slots.is_empty(), "slot reused before its fence");

        let retired = q.reap_through(FrameSerial(4)).unwrap();
        assert_eq!(retired.len(), 1);
        let released_slot = retired[0].payload.slot;
        drop(retired); // physical destruction precedes slot release
        free_slots.push(released_slot);
        assert_eq!(free_slots, vec![3]);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(q.reap_through(FrameSerial(4)).unwrap().is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 1, "duplicate completion double-dropped");
    }

    #[test]
    fn out_of_order_slot_observation_uses_monotonic_completion_high_water() {
        let mut q = GpuRetirementQueue::new();
        q.enqueue(RetirementClass::BoundsEntry, FrameSerial(1), "slot-0-old");
        q.enqueue(RetirementClass::InstanceRecord, FrameSerial(2), "slot-1-new");
        q.enqueue(RetirementClass::MeshGeometry, FrameSerial(3), "slot-0-next");

        // Slot 1 is observed first. Because submissions share one ordered graphics queue,
        // its serial-2 fence proves serials 1 and 2 complete even though slot 0 was not polled.
        let mut completion_high_water = FrameSerial::ZERO;
        completion_high_water = completion_high_water.max(FrameSerial(2));
        let reaped = q.reap_through(completion_high_water).unwrap();
        assert_eq!(
            reaped.iter().map(|record| record.payload).collect::<Vec<_>>(),
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
        ];

        let mut q: GpuRetirementQueue<u32> = GpuRetirementQueue::new();
        for (i, &cls) in classes.iter().enumerate() {
            q.enqueue(cls, FrameSerial(i as u64 + 1), i as u32);
        }

        assert_eq!(q.pending_count(), 6);
        let reaped = q.reap_through(FrameSerial(6)).unwrap();
        assert_eq!(reaped.len(), 6);
        for (i, rec) in reaped.iter().enumerate() {
            assert_eq!(rec.class, classes[i]);
            assert_eq!(rec.payload, i as u32);
        }
    }
}
