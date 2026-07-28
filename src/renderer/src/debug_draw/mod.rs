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

use glam::{Vec3, Vec4};

/// Default maximum lines stored per frame.
const DEFAULT_CAPACITY_LINES: usize = 65536;

/// Ring buffer of debug-line segments.
///
/// Owns a fixed-capacity `Vec` used as a ring buffer. `push_line` appends
/// until the buffer is full; overflowing lines are silently dropped.
pub struct DebugDrawState {
    lines: Vec<(Vec3, Vec3, Vec3)>,
    capacity: usize,
    len: usize,
}

impl DebugDrawState {
    /// Create a ring buffer with the default capacity (64K lines).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY_LINES)
    }

    /// Create a ring buffer with `capacity` lines (2× vertices).
    pub fn with_capacity(capacity: usize) -> Self {
        let lines = Vec::with_capacity(capacity);
        Self {
            lines,
            capacity,
            len: 0,
        }
    }

    /// Append a world-space line segment.
    ///
    /// Lines beyond capacity are silently discarded.
    pub fn push_line(&mut self, from: Vec3, to: Vec3, color: Vec3) {
        if self.len < self.capacity {
            self.lines.push((from, to, color));
            self.len += 1;
        }
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
    pub fn push_cross(&mut self, position: Vec3, size: f32, color_x: Vec3, color_y: Vec3, color_z: Vec3) {
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
        self.len = 0;
    }

    /// Number of lines currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no lines are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum line capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Borrow the stored lines as an immutable slice.
    pub fn lines_slice(&self) -> &[(Vec3, Vec3, Vec3)] {
        &self.lines[..self.len]
    }

    /// Consume the state and return all lines.
    pub fn take_lines(&mut self) -> Vec<(Vec3, Vec3, Vec3)> {
        let taken = std::mem::take(&mut self.lines);
        self.len = 0;
        taken
    }
}

impl Default for DebugDrawState {
    fn default() -> Self {
        Self::new()
    }
}
