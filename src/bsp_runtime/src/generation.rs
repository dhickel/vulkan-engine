//! BSP generation token: slot + monotonic generation counter with cancellation.
//!
//! Every `prepare()` increments the generation. Operations that carry a stale
//! generation token are rejected, and cancellation is implicit: incrementing
//! the generation discards any in-flight prepare for the previous generation.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum generation value before exhaustion.
pub const MAX_GENERATION: u64 = u64::MAX - 1;

/// A generation token issued during `prepare()`.
///
/// The token must be presented at `validate()` and `commit()`. If the
/// coordinator's current generation no longer matches, the operation is
/// rejected as stale.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BspGenerationToken {
    /// The generation value this token was issued for.
    pub generation: u64,
}

/// A cancellation guard that, when dropped, signals cancellation of any
/// in-flight work associated with a previous generation.
///
/// The coordinator's current generation is atomically readable; workers
/// should poll `is_cancelled()` periodically during long-running extraction
/// or upload work.
#[derive(Debug)]
pub struct CancellationToken {
    /// The generation this token was created for.
    issued_generation: u64,
    /// Shared reference to the coordinator's current generation.
    current: *const AtomicU64,
}

// SAFETY: CancellationToken is Send + Sync because the AtomicU64 it points
// to lives in the coordinator, which outlives any token.
unsafe impl Send for CancellationToken {}
unsafe impl Sync for CancellationToken {}

impl CancellationToken {
    /// Create a new cancellation token bound to a coordinator's generation counter.
    ///
    /// # Safety
    /// `current` must point to a valid, stable `AtomicU64` that outlives this token.
    pub(crate) unsafe fn new(issued_generation: u64, current: *const AtomicU64) -> Self {
        Self {
            issued_generation,
            current,
        }
    }

    /// Returns `true` if this token's generation has been superseded.
    pub fn is_cancelled(&self) -> bool {
        // SAFETY: `current` points to a valid AtomicU64 that lives longer than this token.
        let current_gen = unsafe { (*self.current).load(Ordering::Acquire) };
        current_gen != self.issued_generation
    }

    /// Returns the generation this token was issued for.
    pub fn generation(&self) -> u64 {
        self.issued_generation
    }
}

/// Monotonically increasing generation counter.
///
/// Used by [`super::coordinator::BspCoordinator`] to serialize prepare
/// operations and detect stale completion.
#[derive(Debug)]
pub struct BspGenerationCounter {
    current: AtomicU64,
}

impl BspGenerationCounter {
    /// Create a new counter starting at generation 0.
    pub fn new() -> Self {
        Self {
            current: AtomicU64::new(0),
        }
    }

    /// Increment the generation and return the new value.
    ///
    /// Returns `None` if the counter has exhausted its range.
    pub fn increment(&self) -> Option<u64> {
        loop {
            let prev = self.current.load(Ordering::Relaxed);
            if prev >= MAX_GENERATION {
                return None;
            }
            let next = prev + 1;
            if self
                .current
                .compare_exchange_weak(prev, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return Some(next);
            }
        }
    }

    /// Return the current generation value.
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }

    /// Create a token representing the current generation.
    pub fn token(&self) -> BspGenerationToken {
        BspGenerationToken {
            generation: self.current(),
        }
    }

    /// Create a cancellation token for in-flight work.
    ///
    /// Workers should poll `is_cancelled()` periodically.
    pub fn cancellation_token(&self) -> CancellationToken {
        let gen = self.current();
        // SAFETY: `self.current` lives for the duration of the counter, which
        // outlives any token.
        unsafe { CancellationToken::new(gen, &self.current as *const AtomicU64) }
    }

    /// Validate that a token matches the current generation.
    pub fn validate(&self, token: BspGenerationToken) -> Result<(), crate::error::BspRuntimeError> {
        let cur = self.current();
        if token.generation != cur {
            return Err(crate::error::BspRuntimeError::StaleGeneration {
                expected: token.generation,
                current: cur,
            });
        }
        Ok(())
    }
}

impl Default for BspGenerationCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increment_produces_monotonic_values() {
        let counter = BspGenerationCounter::new();
        assert_eq!(counter.current(), 0);

        let g1 = counter.increment().unwrap();
        assert_eq!(g1, 1);
        assert_eq!(counter.current(), 1);

        let g2 = counter.increment().unwrap();
        assert_eq!(g2, 2);
    }

    #[test]
    fn token_validation() {
        let counter = BspGenerationCounter::new();
        let token = counter.token();
        assert_eq!(token.generation, 0);

        // Token valid before increment
        assert!(counter.validate(token).is_ok());

        // After increment, token is stale
        counter.increment().unwrap();
        assert!(counter.validate(token).is_err());
    }

    #[test]
    fn cancellation_token_detects_cancellation() {
        let counter = BspGenerationCounter::new();
        let ct = counter.cancellation_token();
        assert!(!ct.is_cancelled());
        assert_eq!(ct.generation(), 0);

        counter.increment().unwrap();
        assert!(ct.is_cancelled());
    }

    #[test]
    fn generation_exhaustion() {
        let counter = BspGenerationCounter {
            current: AtomicU64::new(MAX_GENERATION),
        };
        assert!(counter.increment().is_none());
    }
}
