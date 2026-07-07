//! Small frame timing helpers for app-owned runtime loops.

use std::time::{Duration, Instant};

/// Frame timing snapshot returned by [`FrameClock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameInfo {
    pub index: u64,
    pub delta: Duration,
    pub delta_seconds: f32,
}

/// Monotonic frame clock for app-owned loops.
#[derive(Clone, Debug)]
pub struct FrameClock {
    next_index: u64,
    last_tick: Instant,
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameClock {
    pub fn new() -> Self {
        Self::from_instant(Instant::now())
    }

    pub fn from_instant(start: Instant) -> Self {
        Self {
            next_index: 0,
            last_tick: start,
        }
    }

    pub fn tick(&mut self) -> FrameInfo {
        self.tick_at(Instant::now())
    }

    pub fn tick_at(&mut self, now: Instant) -> FrameInfo {
        let delta = now
            .checked_duration_since(self.last_tick)
            .unwrap_or(Duration::ZERO);
        self.last_tick = now;

        let info = FrameInfo {
            index: self.next_index,
            delta,
            delta_seconds: delta.as_secs_f32(),
        };
        self.next_index = self.next_index.saturating_add(1);
        info
    }

    pub fn next_index(&self) -> u64 {
        self.next_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_clock_advances_indices_and_deltas() {
        let start = Instant::now();
        let mut clock = FrameClock::from_instant(start);

        let first = clock.tick_at(start + Duration::from_millis(16));
        let second = clock.tick_at(start + Duration::from_millis(33));

        assert_eq!(first.index, 0);
        assert_eq!(first.delta, Duration::from_millis(16));
        assert_eq!(second.index, 1);
        assert_eq!(second.delta, Duration::from_millis(17));
        assert_eq!(clock.next_index(), 2);
    }

    #[test]
    fn frame_clock_clamps_backwards_ticks_to_zero_delta() {
        let start = Instant::now();
        let mut clock = FrameClock::from_instant(start);

        let info = clock.tick_at(start - Duration::from_millis(1));

        assert_eq!(info.index, 0);
        assert_eq!(info.delta, Duration::ZERO);
    }
}
