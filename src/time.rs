//! Caller-owned scaled time service.
//!
//! Composes a [`FrameClock`] and [`FixedStepClock`] internally, exposes
//! scaled/unscaled deltas, fixed-step count, alpha, and accumulated state.

use std::time::{Duration, Instant};

use crate::frame::{FixedStepClock, FixedStepConfig, FrameClock, FrameInfo};

/// Error returned when setting an invalid time scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeError {
    /// Scale was NaN, infinite, or negative.
    InvalidScale,
}

impl std::fmt::Display for TimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidScale => f.write_str("time scale must be finite and non-negative"),
        }
    }
}

impl std::error::Error for TimeError {}

/// Configuration for a [`Time`] instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeConfig {
    /// Fixed-step duration.
    pub step: Duration,
    /// Maximum fixed steps per frame.
    pub max_steps_per_frame: u32,
    /// Initial time scale (must be finite and non-negative).
    pub time_scale: f32,
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            step: Duration::from_secs_f32(1.0 / 60.0),
            max_steps_per_frame: 10,
            time_scale: 1.0,
        }
    }
}

/// Snapshot returned after advancing [`Time`] by one delta.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeUpdate {
    /// Unscaled real-time frame delta.
    pub unscaled_delta: Duration,
    /// Delta scaled by the current time scale.
    pub scaled_delta: Duration,
    /// Fixed-step duration (quantum).
    pub fixed_delta: Duration,
    /// Number of fixed steps consumed this frame.
    pub fixed_step_count: u32,
    /// Interpolation alpha from remainder / step (0.0..1.0).
    pub alpha: f32,
    /// Accumulated remainder after consuming fixed steps.
    pub remainder: Duration,
    /// Time dropped because the accumulator exceeded the catch-up cap.
    pub dropped_time: Duration,
    /// Monotonic frame index.
    pub frame_index: u64,
    /// Current time scale value.
    pub scale: f32,
}

/// Caller-owned scaled time service.
///
/// Composes an internal [`FrameClock`] and [`FixedStepClock`]. The caller
/// ticks the frame clock and feeds deltas to advance the fixed-step
/// accumulator. A time scale can be adjusted at runtime; the fixed quantum
/// is never altered by scale changes.
#[derive(Clone, Debug)]
pub struct Time {
    frame_clock: FrameClock,
    fixed_clock: FixedStepClock,
    fixed_delta: Duration,
    time_scale: f32,
    last_update: TimeUpdate,
}

impl Time {
    /// Create a new `Time` with the given configuration.
    ///
    /// Returns `Err(TimeError::InvalidScale)` if `config.time_scale` is NaN,
    /// infinite, or negative.
    pub fn new(config: TimeConfig) -> Result<Self, TimeError> {
        Self::validate_scale(config.time_scale)?;

        let fixed_delta = config.step;
        Ok(Self {
            frame_clock: FrameClock::new(),
            fixed_clock: FixedStepClock::new(FixedStepConfig {
                step: config.step,
                max_steps_per_frame: config.max_steps_per_frame,
            }),
            fixed_delta,
            time_scale: config.time_scale,
            last_update: TimeUpdate {
                unscaled_delta: Duration::ZERO,
                scaled_delta: Duration::ZERO,
                fixed_delta,
                fixed_step_count: 0,
                alpha: 0.0,
                remainder: Duration::ZERO,
                dropped_time: Duration::ZERO,
                frame_index: 0,
                scale: config.time_scale,
            },
        })
    }

    /// Create a new `Time` with a specific starting instant (for tests).
    pub fn from_instant(config: TimeConfig, start: Instant) -> Result<Self, TimeError> {
        Self::validate_scale(config.time_scale)?;

        let fixed_delta = config.step;
        Ok(Self {
            frame_clock: FrameClock::from_instant(start),
            fixed_clock: FixedStepClock::new(FixedStepConfig {
                step: config.step,
                max_steps_per_frame: config.max_steps_per_frame,
            }),
            fixed_delta,
            time_scale: config.time_scale,
            last_update: TimeUpdate {
                unscaled_delta: Duration::ZERO,
                scaled_delta: Duration::ZERO,
                fixed_delta,
                fixed_step_count: 0,
                alpha: 0.0,
                remainder: Duration::ZERO,
                dropped_time: Duration::ZERO,
                frame_index: 0,
                scale: config.time_scale,
            },
        })
    }

    /// Tick the internal frame clock and return frame info.
    ///
    /// This is typically called by `begin_app_frame_with_time` rather than
    /// directly by the app loop.
    pub fn tick(&mut self) -> FrameInfo {
        self.frame_clock.tick()
    }

    /// Tick the frame clock at a specific instant (for tests).
    pub fn tick_at(&mut self, now: Instant) -> FrameInfo {
        self.frame_clock.tick_at(now)
    }

    /// Advance the fixed-step accumulator by a delta and produce a time update.
    ///
    /// This is the primary caller-facing method. After calling `tick()` (or
    /// `begin_app_frame_with_time`), the app receives a `TimeUpdate` by
    /// calling this method with the frame delta.
    ///
    /// The fixed-step clock is always fed the *scaled* delta, so a scale of
    /// zero pauses simulation while a scale of 2.0 doubles simulation speed.
    pub fn advance(&mut self, unscaled_delta: Duration) -> TimeUpdate {
        let scaled_delta = duration_mul_f32(unscaled_delta, self.time_scale);
        let fixed_update = self.fixed_clock.update(scaled_delta);

        self.last_update = TimeUpdate {
            unscaled_delta,
            scaled_delta,
            fixed_delta: self.fixed_delta,
            fixed_step_count: fixed_update.steps,
            alpha: fixed_update.alpha,
            remainder: fixed_update.accumulated,
            dropped_time: fixed_update.dropped_time,
            frame_index: self.frame_clock.next_index().saturating_sub(1),
            scale: self.time_scale,
        };

        self.last_update
    }

    /// Convenience: tick the frame clock at `now` and advance by the computed delta.
    ///
    /// For tests that control time deterministically.
    pub fn advance_at(&mut self, now: Instant) -> TimeUpdate {
        let frame = self.tick_at(now);
        self.advance(frame.delta)
    }

    /// Return the last computed time update.
    pub fn update(&self) -> &TimeUpdate {
        &self.last_update
    }

    /// Set the time scale.
    ///
    /// Returns the previous scale on success.
    /// Rejects NaN, infinite, and negative values with [`TimeError::InvalidScale`].
    /// A scale of zero pauses scaled simulation without affecting the fixed
    /// quantum or unscaled frame delta.
    pub fn set_time_scale(&mut self, scale: f32) -> Result<f32, TimeError> {
        Self::validate_scale(scale)?;
        let old = self.time_scale;
        self.time_scale = scale;
        Ok(old)
    }

    /// Current time scale.
    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    /// Reset the fixed-step accumulator.
    pub fn reset(&mut self) {
        self.fixed_clock.reset();
    }

    fn validate_scale(scale: f32) -> Result<(), TimeError> {
        if scale.is_finite() && scale >= 0.0 {
            Ok(())
        } else {
            Err(TimeError::InvalidScale)
        }
    }
}

fn duration_mul_f32(duration: Duration, scale: f32) -> Duration {
    if scale <= 0.0 {
        return Duration::ZERO;
    }
    let secs = duration.as_secs_f64() * scale as f64;
    Duration::from_secs_f64(secs.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time_config() -> TimeConfig {
        TimeConfig {
            step: Duration::from_millis(16),
            max_steps_per_frame: 4,
            time_scale: 1.0,
        }
    }

    #[test]
    fn rejects_invalid_scales() {
        assert!(Time::validate_scale(f32::NAN).is_err());
        assert!(Time::validate_scale(f32::INFINITY).is_err());
        assert!(Time::validate_scale(f32::NEG_INFINITY).is_err());
        assert!(Time::validate_scale(-0.1).is_err());
        assert!(Time::validate_scale(-1.0).is_err());
        assert!(Time::validate_scale(0.0).is_ok());
        assert!(Time::validate_scale(1.0).is_ok());
        assert!(Time::validate_scale(2.5).is_ok());
    }

    #[test]
    fn time_scale_zero_pauses_simulation() {
        let mut time = Time::new(TimeConfig {
            time_scale: 0.0,
            ..time_config()
        })
        .unwrap();

        let update = time.advance(Duration::from_millis(100));

        assert_eq!(update.unscaled_delta, Duration::from_millis(100));
        assert_eq!(update.scaled_delta, Duration::ZERO);
        assert_eq!(update.fixed_step_count, 0);
        assert_eq!(update.alpha, 0.0);
        assert_eq!(update.scale, 0.0);
    }

    #[test]
    fn time_scale_doubles_simulation() {
        let mut time = Time::new(TimeConfig {
            time_scale: 2.0,
            ..time_config()
        })
        .unwrap();

        let update = time.advance(Duration::from_millis(16));

        assert_eq!(update.unscaled_delta, Duration::from_millis(16));
        assert_eq!(update.scaled_delta, Duration::from_millis(32));
        assert_eq!(update.fixed_step_count, 2);
        assert_eq!(update.scale, 2.0);
    }

    #[test]
    fn set_time_scale_validates_and_returns_old() {
        let mut time = Time::new(time_config()).unwrap();

        let old = time.set_time_scale(0.5).unwrap();
        assert_eq!(old, 1.0);
        assert_eq!(time.time_scale(), 0.5);

        assert_eq!(
            time.set_time_scale(f32::NAN).unwrap_err(),
            TimeError::InvalidScale
        );
        assert_eq!(time.time_scale(), 0.5); // unchanged
    }

    #[test]
    fn advance_at_produces_time_update() {
        let now = Instant::now();
        let mut time = Time::from_instant(time_config(), now).unwrap();

        let update = time.advance_at(now + Duration::from_millis(32));

        assert_eq!(update.frame_index, 0);
        assert_eq!(update.unscaled_delta, Duration::from_millis(32));
        assert_eq!(update.scaled_delta, Duration::from_millis(32));
        assert_eq!(update.fixed_step_count, 2);
        assert_eq!(update.remainder, Duration::ZERO);
        assert_eq!(update.dropped_time, Duration::ZERO);
    }

    #[test]
    fn reset_clears_accumulator() {
        let mut time = Time::new(time_config()).unwrap();

        let _ = time.advance(Duration::from_millis(10));
        time.reset();
        let update = time.advance(Duration::from_millis(6));

        assert_eq!(update.fixed_step_count, 0);
        assert_eq!(update.remainder, Duration::from_millis(6));
    }

    #[test]
    fn dropped_time_when_exceeding_catch_up() {
        let mut time = Time::new(TimeConfig {
            max_steps_per_frame: 2,
            ..time_config()
        })
        .unwrap();

        let update = time.advance(Duration::from_millis(100));

        assert_eq!(update.fixed_step_count, 2);
        assert!(update.dropped_time > Duration::ZERO);
    }
}
