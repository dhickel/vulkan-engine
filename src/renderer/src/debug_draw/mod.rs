//! Debug-draw ring buffer for editor gizmos, physics wireframes, and
//! development overlays. Gated behind the `debug-draw` Cargo feature.
//!
//! # Ring-buffer semantics
//!
//! Lines are accumulated into a pre-allocated ring buffer each frame.
//! `clear()` resets the write cursor; `push_line()` appends a world-space
//! segment. The buffer is converted to an immutable slice for GPU upload
//! during render submission.
//!
//! # Capacity
//!
//! Default capacity is 64K lines (128K vertices). The buffer silently
//! discards lines beyond capacity rather than reallocating.

use glam::Vec3;
use std::collections::VecDeque;

/// Default maximum lines stored per frame.
const DEFAULT_CAPACITY_LINES: usize = 65536;

/// Ring buffer of debug-line segments.
///
/// Owns a fixed-capacity ring. Once full, newly pushed lines replace the
/// oldest line so the current frame always contains the most recent diagnostics.
pub struct DebugDrawState {
    lines: VecDeque<(Vec3, Vec3, Vec3)>,
    capacity: usize,
    overflow_count: u64,
}

impl DebugDrawState {
    /// Create a ring buffer with the default capacity (64K lines).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY_LINES)
    }

    /// Create a ring buffer with `capacity` lines (2× vertices).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            lines: VecDeque::with_capacity(capacity),
            capacity,
            overflow_count: 0,
        }
    }

    /// Append a world-space line segment.
    ///
    /// When the ring is full, the oldest segment is replaced.
    pub fn push_line(&mut self, from: Vec3, to: Vec3, color: Vec3) {
        if self.capacity == 0 {
            self.overflow_count += 1;
            return;
        }
        if self.lines.len() == self.capacity {
            self.lines.pop_front();
            self.overflow_count += 1;
        }
        self.lines.push_back((from, to, color));
    }

    /// Append a world-space axis-aligned bounding box as 12 line segments.
    pub fn push_aabb(&mut self, min: Vec3, max: Vec3, color: Vec3) {
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, max.y, max.z),
        ];

        // Bottom face
        self.push_line(corners[0], corners[1], color);
        self.push_line(corners[1], corners[3], color);
        self.push_line(corners[3], corners[2], color);
        self.push_line(corners[2], corners[0], color);

        // Top face
        self.push_line(corners[4], corners[5], color);
        self.push_line(corners[5], corners[7], color);
        self.push_line(corners[7], corners[6], color);
        self.push_line(corners[6], corners[4], color);

        // Vertical edges
        self.push_line(corners[0], corners[4], color);
        self.push_line(corners[1], corners[5], color);
        self.push_line(corners[2], corners[6], color);
        self.push_line(corners[3], corners[7], color);
    }

    /// Append a world-space sphere as 3 orthogonal rings.
    pub fn push_sphere(&mut self, center: Vec3, radius: f32, color: Vec3) {
        let segments = 32;
        for i in 0..segments {
            let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let (s0, c0) = (angle0.sin(), angle0.cos());
            let (s1, c1) = (angle1.sin(), angle1.cos());

            // XY plane
            self.push_line(
                center + Vec3::new(c0 * radius, s0 * radius, 0.0),
                center + Vec3::new(c1 * radius, s1 * radius, 0.0),
                color,
            );
            // XZ plane
            self.push_line(
                center + Vec3::new(c0 * radius, 0.0, s0 * radius),
                center + Vec3::new(c1 * radius, 0.0, s1 * radius),
                color,
            );
            // YZ plane
            self.push_line(
                center + Vec3::new(0.0, c0 * radius, s0 * radius),
                center + Vec3::new(0.0, c1 * radius, s1 * radius),
                color,
            );
        }
    }

    /// Append a cross/gizmo at `position` with `size` per axis.
    pub fn push_cross(
        &mut self,
        position: Vec3,
        size: f32,
        color_x: Vec3,
        color_y: Vec3,
        color_z: Vec3,
    ) {
        let half = size * 0.5;
        self.push_line(
            position - Vec3::X * half,
            position + Vec3::X * half,
            color_x,
        );
        self.push_line(
            position - Vec3::Y * half,
            position + Vec3::Y * half,
            color_y,
        );
        self.push_line(
            position - Vec3::Z * half,
            position + Vec3::Z * half,
            color_z,
        );
    }

    /// Clear accumulated lines for a new frame.
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Number of lines currently stored.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Number of lines displaced because the fixed-capacity ring was full.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Returns `true` when no lines are stored.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Maximum line capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Consume the state and return all lines in insertion order.
    pub fn take_lines(&mut self) -> Vec<(Vec3, Vec3, Vec3)> {
        self.lines.drain(..).collect()
    }
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_newest_lines_and_clear_is_frame_local() {
        let mut debug = DebugDrawState::with_capacity(2);
        debug.push_line(Vec3::ZERO, Vec3::X, Vec3::X);
        debug.push_line(Vec3::ZERO, Vec3::Y, Vec3::Y);
        debug.push_line(Vec3::ZERO, Vec3::Z, Vec3::Z);

        assert_eq!(debug.len(), 2);
        assert_eq!(debug.overflow_count(), 1);
        let lines = debug.take_lines();
        assert_eq!(lines[0].1, Vec3::Y);
        assert_eq!(lines[1].1, Vec3::Z);
        assert!(debug.is_empty());

        debug.push_cross(Vec3::ZERO, 1.0, Vec3::X, Vec3::Y, Vec3::Z);
        debug.clear();
        assert!(debug.is_empty());
    }
}
