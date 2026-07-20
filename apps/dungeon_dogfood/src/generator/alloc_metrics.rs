//! Optional allocation counter behind `generator-bench-alloc` feature.
//!
//! Reset before generate(), sample after generate().
//! Keep recursion-safe.
//! The non-instrumented release latency binary must NOT contain or enable
//! this allocator wrapper.

/// Snapshot of allocation counters at a point in time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocSnapshot {
    pub allocations: u64,
    pub deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub peak_bytes: u64,
}

/// A thread-local allocation counter.
///
/// Incremented via the global allocator hook when the
/// `generator-bench-alloc` feature is enabled.
#[cfg(feature = "generator-bench-alloc")]
mod inner {
    use std::sync::atomic::{AtomicU64, Ordering};

    static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
    static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
    static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
    static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
    static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

    pub(super) fn reset() {
        ALLOC_COUNT.store(0, Ordering::SeqCst);
        DEALLOC_COUNT.store(0, Ordering::SeqCst);
        ALLOC_BYTES.store(0, Ordering::SeqCst);
        DEALLOC_BYTES.store(0, Ordering::SeqCst);
        PEAK_BYTES.store(0, Ordering::SeqCst);
    }

    pub(super) fn snapshot() -> super::AllocSnapshot {
        let alloc = ALLOC_COUNT.load(Ordering::SeqCst);
        let dealloc = DEALLOC_COUNT.load(Ordering::SeqCst);
        let alloc_bytes = ALLOC_BYTES.load(Ordering::SeqCst);
        let dealloc_bytes = DEALLOC_BYTES.load(Ordering::SeqCst);
        let peak = PEAK_BYTES.load(Ordering::SeqCst);
        super::AllocSnapshot {
            allocations: alloc,
            deallocations: dealloc,
            bytes_allocated: alloc_bytes,
            bytes_deallocated: dealloc_bytes,
            peak_bytes: peak,
        }
    }

    /// Called by the allocator hook on allocation.
    pub(super) fn record_alloc(size: usize) {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        let prev = ALLOC_BYTES.fetch_add(size as u64, Ordering::SeqCst);
        let current = prev + size as u64;
        // Update peak (best-effort; not truly atomic with fetch_add).
        let mut peak = PEAK_BYTES.load(Ordering::SeqCst);
        while current > peak {
            match PEAK_BYTES.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    /// Called by the allocator hook on deallocation.
    pub(super) fn record_dealloc(size: usize) {
        DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        DEALLOC_BYTES.fetch_add(size as u64, Ordering::SeqCst);
        // Subtract from current bytes
        ALLOC_BYTES.fetch_sub(size as u64, Ordering::SeqCst);
    }
}

/// Global allocator wrapper instrumented behind `generator-bench-alloc`.
/// The non-instrumented release latency binary must not contain this path.
#[cfg(feature = "generator-bench-alloc")]
mod alloc_hook {
    use std::alloc::{GlobalAlloc, Layout, System};

    struct InstrumentedAlloc;

    unsafe impl GlobalAlloc for InstrumentedAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                super::inner::record_alloc(layout.size());
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            super::inner::record_dealloc(layout.size());
            System.dealloc(ptr, layout);
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc_zeroed(layout);
            if !ptr.is_null() {
                super::inner::record_alloc(layout.size());
            }
            ptr
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = System.realloc(ptr, layout, new_size);
            if !new_ptr.is_null() {
                super::inner::record_dealloc(layout.size());
                super::inner::record_alloc(new_size);
            }
            new_ptr
        }
    }

    #[global_allocator]
    static GLOBAL: InstrumentedAlloc = InstrumentedAlloc;
}

#[cfg(not(feature = "generator-bench-alloc"))]
mod inner {
    pub(super) fn reset() {}
    pub(super) fn snapshot() -> super::AllocSnapshot {
        super::AllocSnapshot::default()
    }
}

/// Reset allocation counters. Call immediately before the measured generate() call.
/// Recursion-safe: may be called from within an instrumented allocator hook.
pub fn reset() {
    inner::reset();
}

/// Snapshot allocation counters. Call immediately after the measured generate() call.
/// Recursion-safe: does not allocate.
pub fn snapshot() -> AllocSnapshot {
    inner::snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_and_snapshot_are_recursion_safe() {
        // Even without the alloc feature, reset + snapshot must work.
        reset();
        let snap = snapshot();
        // In non-instrumented mode, all counters are zero.
        assert_eq!(snap.allocations, 0);
        assert_eq!(snap.bytes_allocated, 0);
    }

    #[test]
    fn snapshot_is_copy_and_default() {
        let s1 = AllocSnapshot::default();
        let s2 = s1;
        assert_eq!(s1, s2);
        let s3 = snapshot();
        // In non-instrumented mode, snapshot is zero.
        assert_eq!(s3.allocations, 0);
    }

    #[cfg(feature = "generator-bench-alloc")]
    #[test]
    fn instrumented_records_allocations() {
        reset();
        // Simulate allocation recording
        inner::record_alloc(1024);
        inner::record_alloc(2048);
        inner::record_dealloc(1024);
        let snap = snapshot();
        assert_eq!(snap.allocations, 2);
        assert_eq!(snap.deallocations, 1);
        assert_eq!(snap.bytes_allocated, 1024); // 3072 - 1024 - 1024 = 1024
        assert_eq!(snap.bytes_deallocated, 1024);
        assert_eq!(snap.peak_bytes, 3072);
    }
}
