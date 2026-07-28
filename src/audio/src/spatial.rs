//! Conditional spatial audio: stereo panning + distance attenuation.
//!
//! Always available (no feature gate). Provides:
//!
//! - 3D math types for listener/source positions and directions
//! - Equal-power stereo panning from listener right vector
//! - Inverse-square distance attenuation with min/max clamping
//! - Thread-safe atomic gain pair for real-time spatial updates
//! - A [`rodio::Source`] adapter that wraps a mono source and produces
//!   interleaved stereo with independently updateable left/right gains
//!
//! No HRTF, Doppler, occlusion, or multichannel spatialization.

use rodio::source::SeekError;
use rodio::{Sample, Source};
use std::f32::consts::PI;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ── 3D math ────────────────────────────────────────────────────────────────

/// A point or direction in 3D space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Euclidean length.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to a unit vector, returning `None` for zero or non-finite.
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len > 0.0 && len.is_finite() {
            Some(Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            })
        } else {
            None
        }
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Vector from `self` to `other`.
    pub fn to(&self, other: &Self) -> Self {
        Self {
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

// ── Pose types ─────────────────────────────────────────────────────────────

/// Listener pose: position and right direction (world-space).
///
/// The `right` vector must be non-zero and finite. The listener is assumed to
/// face forward; only the right axis is used for panning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ListenerPose {
    pub position: Vec3,
    pub right: Vec3,
}

impl ListenerPose {
    pub fn new(position: Vec3, right: Vec3) -> Self {
        Self { position, right }
    }
}

/// Source pose: position only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourcePose {
    pub position: Vec3,
}

impl SourcePose {
    pub fn new(position: Vec3) -> Self {
        Self { position }
    }
}

// ── Attenuation settings ───────────────────────────────────────────────────

/// Distance-based attenuation parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attenuation {
    /// Distance at which gain is 1.0 (no attenuation). Must be >= 0.
    pub min_distance: f32,
    /// Distance at and beyond which gain is 0.0. Must be >= min_distance.
    pub max_distance: f32,
}

impl Default for Attenuation {
    fn default() -> Self {
        Self {
            min_distance: 1.0,
            max_distance: 100.0,
        }
    }
}

// ── Gain types ─────────────────────────────────────────────────────────────

const CENTER_GAIN: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.7071

/// Computed stereo gain pair from spatialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialGain {
    pub left: f32,
    pub right: f32,
}

impl SpatialGain {
    /// Equal-power center pan (both channels at 1/√2).
    pub const CENTER: Self = Self {
        left: CENTER_GAIN,
        right: CENTER_GAIN,
    };

    /// Create a gain pair. Values are clamped to `[0, 1]` and NaN → 0.
    pub fn new(left: f32, right: f32) -> Self {
        fn clamp_finite(v: f32) -> f32 {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        Self {
            left: clamp_finite(left),
            right: clamp_finite(right),
        }
    }
}

/// Thread-safe stereo gain pair for real-time updates during playback.
///
/// Stores gains as `f32` bit patterns in [`AtomicU32`] so that the audio
/// thread can read them without locking.
pub struct AtomicSpatialGains {
    left: AtomicU32,
    right: AtomicU32,
}

impl fmt::Debug for AtomicSpatialGains {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicSpatialGains")
            .field("left", &self.left())
            .field("right", &self.right())
            .finish()
    }
}

impl AtomicSpatialGains {
    pub fn new(gains: SpatialGain) -> Self {
        Self {
            left: AtomicU32::new(gains.left.to_bits()),
            right: AtomicU32::new(gains.right.to_bits()),
        }
    }

    /// Overwrite both gains atomically (best-effort: two independent stores).
    pub fn set(&self, gains: SpatialGain) {
        self.left.store(gains.left.to_bits(), Ordering::Relaxed);
        self.right.store(gains.right.to_bits(), Ordering::Relaxed);
    }

    /// Read the current left gain.
    pub fn left(&self) -> f32 {
        f32::from_bits(self.left.load(Ordering::Relaxed))
    }

    /// Read the current right gain.
    pub fn right(&self) -> f32 {
        f32::from_bits(self.right.load(Ordering::Relaxed))
    }

    /// Read both gains as a [`SpatialGain`].
    pub fn get(&self) -> SpatialGain {
        SpatialGain {
            left: self.left(),
            right: self.right(),
        }
    }
}

// ── Spatialization math ────────────────────────────────────────────────────

/// Compute a pan amount in `[-1, 1]` from the listener right vector and
/// the direction to the source.
///
/// - `-1.0` = hard left (source is opposite to right)
/// - ` 0.0` = center (source is perpendicular to right or coincident)
/// - ` 1.0` = hard right (source is in the right direction)
///
/// Degenerate inputs (zero-length right or zero-distance source) return 0.0.
pub fn compute_pan(listener_right: &Vec3, to_source: &Vec3) -> f32 {
    let right = match listener_right.normalize() {
        Some(r) => r,
        None => return 0.0,
    };
    let dir = match to_source.normalize() {
        Some(d) => d,
        None => return 0.0,
    };
    right.dot(&dir).clamp(-1.0, 1.0)
}

/// Convert a pan amount in `[-1, 1]` to equal-power stereo gains.
///
/// Uses the constant-power law:
/// - `pan = -1` → left = 1.0, right = 0.0
/// - `pan =  0` → left = right = 1/√2 ≈ 0.707
/// - `pan =  1` → left = 0.0, right = 1.0
pub fn pan_to_gains(pan: f32) -> SpatialGain {
    let pan = pan.clamp(-1.0, 1.0);
    let angle = (pan + 1.0) * PI / 4.0;
    SpatialGain {
        left: angle.cos(),
        right: angle.sin(),
    }
}

/// Compute inverse-square distance attenuation clamped to `[min_distance, max_distance]`.
///
/// - `distance <= min_distance` → 1.0
/// - `distance >= max_distance` → 0.0
/// - Otherwise → `(min_distance / distance)²`
/// - Degenerate inputs (NaN, negative) → 1.0
pub fn compute_attenuation(distance: f32, min_distance: f32, max_distance: f32) -> f32 {
    if !distance.is_finite() || distance < 0.0 {
        return 1.0;
    }
    if distance <= min_distance {
        return 1.0;
    }
    if distance >= max_distance {
        return 0.0;
    }
    let att = min_distance / distance;
    att * att
}

/// Compute spatial stereo gains from listener, source, and attenuation.
///
/// Combines equal-power panning and distance attenuation. The result can be
/// blended toward [`SpatialGain::CENTER`] by the caller using `spatial_blend`.
pub fn spatialize(
    listener: &ListenerPose,
    source: &SourcePose,
    attenuation: &Attenuation,
) -> SpatialGain {
    let to_source = listener.position.to(&source.position);
    let distance = to_source.length();
    let pan = compute_pan(&listener.right, &to_source);
    let gains = pan_to_gains(pan);
    let att = compute_attenuation(distance, attenuation.min_distance, attenuation.max_distance);
    SpatialGain {
        left: gains.left * att,
        right: gains.right * att,
    }
}

/// Linearly interpolate between two gain pairs by `t` (clamped to `[0, 1]`).
pub fn blend_gains(a: SpatialGain, b: SpatialGain, t: f32) -> SpatialGain {
    let t = t.clamp(0.0, 1.0);
    SpatialGain {
        left: a.left + (b.left - a.left) * t,
        right: a.right + (b.right - a.right) * t,
    }
}

// ── Spatial source adapter ─────────────────────────────────────────────────

/// Wraps a mono [`Source`] to produce interleaved stereo with independently
/// updateable left/right gains.
///
/// The input source must have exactly 1 channel. Each input sample is
/// multiplied by the current left gain (output first) and right gain
/// (output second). The gains are read from shared [`AtomicSpatialGains`]
/// so the playback handle can update them in real time.
///
/// # Channel layout
///
/// Input:  `[s₀, s₁, s₂, ...]`  (mono)
/// Output: `[s₀·L, s₀·R, s₁·L, s₁·R, ...]`  (stereo interleaved)
pub struct SpatialSource<I>
where
    I: Source,
    I::Item: Sample,
{
    input: I,
    gains: Arc<AtomicSpatialGains>,
    pending_right: Option<I::Item>,
}

impl<I> SpatialSource<I>
where
    I: Source,
    I::Item: Sample,
{
    /// Wrap a mono source with the given shared gain handle.
    ///
    /// The caller must ensure the input has exactly 1 channel.
    pub fn new(input: I, gains: Arc<AtomicSpatialGains>) -> Self {
        Self {
            input,
            gains,
            pending_right: None,
        }
    }

    /// Shared gain handle for external position updates.
    pub fn gains(&self) -> &Arc<AtomicSpatialGains> {
        &self.gains
    }
}

impl<I> Iterator for SpatialSource<I>
where
    I: Source,
    I::Item: Sample,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        // Return buffered right-channel sample first.
        if let Some(right) = self.pending_right.take() {
            return Some(right);
        }

        // Fetch next mono sample and produce L, buffer R.
        if let Some(sample) = self.input.next() {
            let left_gain = self.gains.left();
            let right_gain = self.gains.right();
            self.pending_right = Some(sample.amplify(right_gain));
            Some(sample.amplify(left_gain))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.input.size_hint();
        let bonus = if self.pending_right.is_some() { 1 } else { 0 };
        (
            lo.checked_mul(2)
                .and_then(|v| v.checked_add(bonus))
                .unwrap_or(usize::MAX),
            hi.and_then(|h| {
                h.checked_mul(2)
                    .and_then(|v| v.checked_add(bonus))
            }),
        )
    }
}

impl<I> Source for SpatialSource<I>
where
    I: Source,
    I::Item: Sample,
{
    fn current_frame_len(&self) -> Option<usize> {
        self.input.current_frame_len()
    }

    fn channels(&self) -> u16 {
        2
    }

    fn sample_rate(&self) -> u32 {
        self.input.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration()
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.pending_right = None;
        self.input.try_seek(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::buffer::SamplesBuffer;

    /// A mono source that yields the given f32 samples.
    fn mono_source(samples: Vec<f32>, sample_rate: u32) -> SamplesBuffer<f32> {
        SamplesBuffer::new(1, sample_rate, samples)
    }

    fn collect_samples<I: Source<Item = f32>>(source: I, limit: usize) -> Vec<f32> {
        source.take(limit).collect()
    }

    // ── SpatialSource iteration tests ───────────────────────────────────

    #[test]
    fn spatial_source_produces_interleaved_stereo() {
        let mono = mono_source(vec![1.0, 0.5, 0.0], 8000);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::new(1.0, 0.5)));
        let spatial = SpatialSource::new(mono, gains);

        let out: Vec<f32> = collect_samples(spatial, 6);
        // s₀·L, s₀·R, s₁·L, s₁·R, s₂·L, s₂·R
        assert_eq!(out, vec![1.0 * 1.0, 1.0 * 0.5, 0.5 * 1.0, 0.5 * 0.5, 0.0 * 1.0, 0.0 * 0.5]);
    }

    #[test]
    fn spatial_source_reports_two_channels() {
        let mono = mono_source(vec![1.0, 2.0], 8000);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::CENTER));
        let spatial = SpatialSource::new(mono, gains);
        assert_eq!(spatial.channels(), 2);
    }

    #[test]
    fn spatial_source_preserves_sample_rate() {
        let mono = mono_source(vec![1.0], 44100);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::CENTER));
        let spatial = SpatialSource::new(mono, gains);
        assert_eq!(spatial.sample_rate(), 44100);
    }

    #[test]
    fn hard_left_gain_produces_silent_right_channel() {
        let mono = mono_source(vec![1.0, 2.0, 3.0], 8000);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::new(1.0, 0.0)));
        let spatial = SpatialSource::new(mono, gains);
        let out: Vec<f32> = collect_samples(spatial, 6);
        assert_eq!(out, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn hard_right_gain_produces_silent_left_channel() {
        let mono = mono_source(vec![1.0, 2.0], 8000);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::new(0.0, 1.0)));
        let spatial = SpatialSource::new(mono, gains);
        let out: Vec<f32> = collect_samples(spatial, 4);
        assert_eq!(out, vec![0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn independent_gain_update_via_atomics() {
        let mono = mono_source(vec![1.0, 1.0, 1.0, 1.0], 8000);
        let gains = Arc::new(AtomicSpatialGains::new(SpatialGain::new(1.0, 1.0)));
        let spatial = SpatialSource::new(mono, gains.clone());

        let mut spatial = spatial;
        // First stereo frame: both gains = 1.0
        assert_eq!(spatial.next(), Some(1.0)); // L
        assert_eq!(spatial.next(), Some(1.0)); // R

        // Update gains to hard-left
        gains.set(SpatialGain::new(1.0, 0.0));

        // Second stereo frame
        assert_eq!(spatial.next(), Some(1.0)); // L (still 1.0)
        assert_eq!(spatial.next(), Some(0.0)); // R (now 0.0)

        // Update gains to hard-right
        gains.set(SpatialGain::new(0.0, 1.0));

        // Third stereo frame
        assert_eq!(spatial.next(), Some(0.0)); // L (now 0.0)
        assert_eq!(spatial.next(), Some(1.0)); // R (now 1.0)
    }

    // ── Panning math tests ──────────────────────────────────────────────

    #[test]
    fn pan_hard_right() {
        // Listener right = (1, 0, 0), source to the right
        let right = Vec3::new(1.0, 0.0, 0.0);
        let to_source = Vec3::new(1.0, 0.0, 0.0);
        let pan = compute_pan(&right, &to_source);
        assert!((pan - 1.0).abs() < 1e-6, "expected 1.0, got {pan}");
    }

    #[test]
    fn pan_hard_left() {
        let right = Vec3::new(1.0, 0.0, 0.0);
        let to_source = Vec3::new(-1.0, 0.0, 0.0);
        let pan = compute_pan(&right, &to_source);
        assert!((pan - (-1.0)).abs() < 1e-6, "expected -1.0, got {pan}");
    }

    #[test]
    fn pan_center_perpendicular() {
        let right = Vec3::new(1.0, 0.0, 0.0);
        let to_source = Vec3::new(0.0, 1.0, 0.0); // front
        let pan = compute_pan(&right, &to_source);
        assert!((pan - 0.0).abs() < 1e-6, "expected 0.0, got {pan}");
    }

    #[test]
    fn pan_center_degenerate_zero_right() {
        let right = Vec3::ZERO;
        let to_source = Vec3::new(1.0, 0.0, 0.0);
        let pan = compute_pan(&right, &to_source);
        assert_eq!(pan, 0.0);
    }

    #[test]
    fn pan_center_zero_distance() {
        let right = Vec3::new(1.0, 0.0, 0.0);
        let to_source = Vec3::ZERO;
        let pan = compute_pan(&right, &to_source);
        assert_eq!(pan, 0.0);
    }

    #[test]
    fn pan_to_gains_center() {
        let g = pan_to_gains(0.0);
        let expected = CENTER_GAIN;
        assert!((g.left - expected).abs() < 1e-5, "left: {}", g.left);
        assert!((g.right - expected).abs() < 1e-5, "right: {}", g.right);
    }

    #[test]
    fn pan_to_gains_hard_right() {
        let g = pan_to_gains(1.0);
        assert!((g.left - 0.0).abs() < 1e-5, "left: {}", g.left);
        assert!((g.right - 1.0).abs() < 1e-5, "right: {}", g.right);
    }

    #[test]
    fn pan_to_gains_hard_left() {
        let g = pan_to_gains(-1.0);
        assert!((g.left - 1.0).abs() < 1e-5, "left: {}", g.left);
        assert!((g.right - 0.0).abs() < 1e-5, "right: {}", g.right);
    }

    #[test]
    fn pan_to_gains_clamps_out_of_range() {
        let g = pan_to_gains(5.0);
        assert!((g.left - 0.0).abs() < 1e-5);
        assert!((g.right - 1.0).abs() < 1e-5);

        let g = pan_to_gains(-5.0);
        assert!((g.left - 1.0).abs() < 1e-5);
        assert!((g.right - 0.0).abs() < 1e-5);
    }

    // ── Attenuation tests ───────────────────────────────────────────────

    #[test]
    fn attenuation_within_min_distance_is_full() {
        assert_eq!(compute_attenuation(0.5, 1.0, 10.0), 1.0);
        assert_eq!(compute_attenuation(0.0, 1.0, 10.0), 1.0);
        assert_eq!(compute_attenuation(1.0, 1.0, 10.0), 1.0);
    }

    #[test]
    fn attenuation_at_max_distance_is_zero() {
        assert_eq!(compute_attenuation(10.0, 1.0, 10.0), 0.0);
    }

    #[test]
    fn attenuation_beyond_max_distance_is_zero() {
        assert_eq!(compute_attenuation(100.0, 1.0, 10.0), 0.0);
    }

    #[test]
    fn attenuation_in_between_is_inverse_square() {
        let att = compute_attenuation(2.0, 1.0, 10.0);
        let expected = (1.0_f32 / 2.0).powi(2); // 0.25
        assert!((att - expected).abs() < 1e-6, "got {att}, expected {expected}");
    }

    #[test]
    fn attenuation_degenerate_returns_full() {
        assert_eq!(compute_attenuation(f32::NAN, 1.0, 10.0), 1.0);
        assert_eq!(compute_attenuation(f32::INFINITY, 1.0, 10.0), 1.0);
        assert_eq!(compute_attenuation(-1.0, 1.0, 10.0), 1.0);
    }

    // ── Full spatialize pipeline ────────────────────────────────────────

    #[test]
    fn spatialize_source_at_listener_produces_center_with_full_gain() {
        let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let source = SourcePose::new(Vec3::ZERO);
        let att = Attenuation::default();
        let g = spatialize(&listener, &source, &att);
        let expected = CENTER_GAIN;
        assert!((g.left - expected).abs() < 1e-5);
        assert!((g.right - expected).abs() < 1e-5);
    }

    #[test]
    fn spatialize_source_to_right_at_min_distance_is_full_right() {
        let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let source = SourcePose::new(Vec3::new(1.0, 0.0, 0.0)); // 1 unit right
        let att = Attenuation {
            min_distance: 1.0,
            max_distance: 100.0,
        };
        let g = spatialize(&listener, &source, &att);
        assert!((g.left - 0.0).abs() < 1e-5, "left should be 0, got {}", g.left);
        assert!((g.right - 1.0).abs() < 1e-5, "right should be 1, got {}", g.right);
    }

    #[test]
    fn spatialize_attenuates_over_distance() {
        let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let source = SourcePose::new(Vec3::new(10.0, 0.0, 0.0)); // 10 units right
        let att = Attenuation {
            min_distance: 1.0,
            max_distance: 100.0,
        };
        let g = spatialize(&listener, &source, &att);
        // attenuation = (1/10)² = 0.01, right gain = 1.0 * 0.01 = 0.01
        assert!((g.left - 0.0).abs() < 1e-5);
        assert!((g.right - 0.01).abs() < 1e-5, "got {}", g.right);
    }

    #[test]
    fn spatialize_beyond_max_distance_is_silent() {
        let listener = ListenerPose::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let source = SourcePose::new(Vec3::new(200.0, 0.0, 0.0));
        let att = Attenuation {
            min_distance: 1.0,
            max_distance: 100.0,
        };
        let g = spatialize(&listener, &source, &att);
        assert_eq!(g.left, 0.0);
        assert_eq!(g.right, 0.0);
    }

    // ── blend_gains tests ───────────────────────────────────────────────

    #[test]
    fn blend_t_zero_is_a() {
        let a = SpatialGain::new(0.2, 0.8);
        let b = SpatialGain::new(0.9, 0.1);
        let g = blend_gains(a, b, 0.0);
        assert!((g.left - 0.2).abs() < 1e-5);
        assert!((g.right - 0.8).abs() < 1e-5);
    }

    #[test]
    fn blend_t_one_is_b() {
        let a = SpatialGain::new(0.2, 0.8);
        let b = SpatialGain::new(0.9, 0.1);
        let g = blend_gains(a, b, 1.0);
        assert!((g.left - 0.9).abs() < 1e-5);
        assert!((g.right - 0.1).abs() < 1e-5);
    }

    #[test]
    fn blend_t_half_interpolates() {
        let a = SpatialGain::new(0.0, 1.0);
        let b = SpatialGain::new(1.0, 0.0);
        let g = blend_gains(a, b, 0.5);
        assert!((g.left - 0.5).abs() < 1e-5);
        assert!((g.right - 0.5).abs() < 1e-5);
    }

    // ── SpatialGain validation ──────────────────────────────────────────

    #[test]
    fn spatial_gain_clamps_and_rejects_nan() {
        let g = SpatialGain::new(-0.5, 2.0);
        assert_eq!(g.left, 0.0);
        assert_eq!(g.right, 1.0);

        let g = SpatialGain::new(f32::NAN, f32::NAN);
        assert_eq!(g.left, 0.0);
        assert_eq!(g.right, 0.0);
    }

    // ── AtomicSpatialGains round-trip ───────────────────────────────────

    #[test]
    fn atomic_gains_round_trip() {
        let ag = AtomicSpatialGains::new(SpatialGain::new(0.3, 0.7));
        let g = ag.get();
        assert!((g.left - 0.3).abs() < 1e-6);
        assert!((g.right - 0.7).abs() < 1e-6);
    }

    #[test]
    fn atomic_gains_update_and_read() {
        let ag = AtomicSpatialGains::new(SpatialGain::CENTER);
        ag.set(SpatialGain::new(0.1, 0.9));
        let g = ag.get();
        assert!((g.left - 0.1).abs() < 1e-6);
        assert!((g.right - 0.9).abs() < 1e-6);
    }
}
