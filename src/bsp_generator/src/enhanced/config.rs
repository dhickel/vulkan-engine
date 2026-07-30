//! Enhanced v2 configuration — M2-only, two-layer, validated at construction.

use crate::config::{
    CONSTRUCTION_QUANTUM, M2_LOOP_COUNT_MAX, M2_LOOP_COUNT_MIN, M2_ROOM_COUNT_MAX,
    M2_ROOM_COUNT_MIN, M2_XY_MAX, M2_Z_MAX,
};
use crate::MapClass;

use super::error::EnhancedError;
use super::profile::{STAIR_RISER, STAIR_TREAD};

// ── Selected vertical contract (Phase 01 evidence) ─────────────────────────

/// Lower floor Z (entry layer). Evidence-selected constant.
pub const ENHANCED_LOWER_FLOOR_Z: i32 = 0;

/// Upper floor Z. Evidence-selected constant.
pub const ENHANCED_UPPER_FLOOR_Z: i32 = 192;

/// Standard room height for both layers. Evidence-selected constant.
pub const ENHANCED_ROOM_HEIGHT: i32 = 176;

/// Stair riser height. Evidence-selected constant.
pub const ENHANCED_RISER: i32 = STAIR_RISER;

/// Stair tread depth — fixed at 16 for the Enhanced v2 stair repair.
/// Both Type A and Type B exclusively use this value.
pub const ENHANCED_TREAD: i32 = STAIR_TREAD;

/// Backward-compatible name for the fixed Enhanced v2 tread contract.
pub const ENHANCED_TREAD_DEFAULT: i32 = ENHANCED_TREAD;

/// Exact Enhanced layer count (frozen).
pub const ENHANCED_LAYER_COUNT: u32 = 2;

/// Minimum outer room span in Quake units (7 quanta = 112).
pub const ENHANCED_MIN_ROOM_SPAN: i32 = 112;

/// Maximum outer room span in Quake units (16 quanta = 256).
pub const ENHANCED_MAX_ROOM_SPAN: i32 = 256;

/// Socket aperture width in Quake units.
pub const SOCKET_APERTURE: i32 = 64;

/// Corner margin in Quake units — sockets inset from room corners.
pub const SOCKET_CORNER_MARGIN: i32 = 32;

/// Minimum wall length required to host a socket.
pub const MIN_WALL_FOR_SOCKET: i32 = SOCKET_APERTURE + 2 * SOCKET_CORNER_MARGIN;

// ── Enhanced config ────────────────────────────────────────────────────────

/// Validated Enhanced v2 configuration.
///
/// Construction via `EnhancedConfig::new()` validates all fields against
/// the frozen M2 bounds and the Phase 01 selected vertical contract.
/// An `EnhancedConfig` is immutable after construction.
///
/// Tread depth is fixed at 16 after the Enhanced v2 stair repair; the
/// constructor validates this and rejects any other value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnhancedConfig {
    /// Number of rooms (M2 range: 17..=40).
    room_count: u32,
    /// Number of horizontal loops (M2 range: 1..=6).
    loop_count: u32,
    /// Number of vertical edges between layers (1..=3).
    vertical_edges: u32,
    /// Stair tread depth; public input retained for contract validation.
    tread_depth: i32,
    /// XY extent per axis in Quake units (≤ M2_XY_MAX).
    xy_extent: u32,
    /// Number of placement candidates generated per room attempt.
    placement_candidates: u32,
    /// Maximum placement attempts per room.
    max_placement_attempts: u32,
    /// Maximum pillars allowed per room (0 = no pillars).
    max_pillars_per_room: u32,
}

impl EnhancedConfig {
    /// Create and validate an Enhanced v2 configuration.
    ///
    /// Tread depth remains an explicit public argument so incompatible caller
    /// input is rejected rather than silently changed; only 16 is accepted.
    pub fn new(
        room_count: u32,
        loop_count: u32,
        vertical_edges: u32,
        tread_depth: i32,
        xy_extent: u32,
    ) -> Result<Self, EnhancedError> {
        Self::with_placement_params(
            room_count,
            loop_count,
            vertical_edges,
            tread_depth,
            xy_extent,
            MapClass::M2.max_placement_candidates(),
            MapClass::M2.max_placement_attempts(),
        )
    }

    /// Create and validate an Enhanced v2 configuration with explicit
    /// placement parameters.
    pub fn with_placement_params(
        room_count: u32,
        loop_count: u32,
        vertical_edges: u32,
        tread_depth: i32,
        xy_extent: u32,
        placement_candidates: u32,
        max_placement_attempts: u32,
    ) -> Result<Self, EnhancedError> {
        Self::with_full_params(
            room_count,
            loop_count,
            vertical_edges,
            tread_depth,
            xy_extent,
            placement_candidates,
            max_placement_attempts,
            2, // default max_pillars_per_room
        )
    }

    /// Create and validate an Enhanced v2 configuration with all parameters
    /// including feature variance.
    pub fn with_full_params(
        room_count: u32,
        loop_count: u32,
        vertical_edges: u32,
        tread_depth: i32,
        xy_extent: u32,
        placement_candidates: u32,
        max_placement_attempts: u32,
        max_pillars_per_room: u32,
    ) -> Result<Self, EnhancedError> {
        if room_count < M2_ROOM_COUNT_MIN || room_count > M2_ROOM_COUNT_MAX {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "room_count",
                value: room_count as u64,
                min: M2_ROOM_COUNT_MIN as u64,
                max: M2_ROOM_COUNT_MAX as u64,
            });
        }
        if loop_count < M2_LOOP_COUNT_MIN || loop_count > M2_LOOP_COUNT_MAX {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "loop_count",
                value: loop_count as u64,
                min: M2_LOOP_COUNT_MIN as u64,
                max: M2_LOOP_COUNT_MAX as u64,
            });
        }
        if vertical_edges < 1 || vertical_edges > 3 {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "vertical_edges",
                value: vertical_edges as u64,
                min: 1,
                max: 3,
            });
        }
        if tread_depth != ENHANCED_TREAD {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "tread_depth",
                value: tread_depth as u64,
                min: ENHANCED_TREAD as u64,
                max: ENHANCED_TREAD as u64,
            });
        }
        if xy_extent == 0 || xy_extent > M2_XY_MAX || xy_extent % CONSTRUCTION_QUANTUM != 0 {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "xy_extent",
                value: xy_extent as u64,
                min: CONSTRUCTION_QUANTUM as u64,
                max: M2_XY_MAX as u64,
            });
        }

        // Verify the vertical contract fits within M2 Z bounds
        let total_z = ENHANCED_UPPER_FLOOR_Z as u32 + ENHANCED_ROOM_HEIGHT as u32;
        if total_z > M2_Z_MAX {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "total_z",
                value: total_z as u64,
                min: 0,
                max: M2_Z_MAX as u64,
            });
        }

        // Validate placement parameters
        let max_candidates = MapClass::M2.max_placement_candidates();
        if placement_candidates == 0 || placement_candidates > max_candidates {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "placement_candidates",
                value: placement_candidates as u64,
                min: 1,
                max: max_candidates as u64,
            });
        }
        let max_attempts = MapClass::M2.max_placement_attempts();
        if max_placement_attempts == 0 || max_placement_attempts > max_attempts {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "max_placement_attempts",
                value: max_placement_attempts as u64,
                min: 1,
                max: max_attempts as u64,
            });
        }

        // Ensure min room span fits within xy_extent
        if (ENHANCED_MIN_ROOM_SPAN as u32) > xy_extent {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "xy_extent",
                value: xy_extent as u64,
                min: ENHANCED_MIN_ROOM_SPAN as u64,
                max: M2_XY_MAX as u64,
            });
        }

        // Validate max_pillars_per_room (0 = no pillars, bounded at 8)
        if max_pillars_per_room > 8 {
            return Err(EnhancedError::ConfigOutOfRange {
                field: "max_pillars_per_room",
                value: max_pillars_per_room as u64,
                min: 0,
                max: 8,
            });
        }

        Ok(Self {
            room_count,
            loop_count,
            vertical_edges,
            tread_depth,
            xy_extent,
            placement_candidates,
            max_placement_attempts,
            max_pillars_per_room,
        })
    }

    /// Nominal M2 Enhanced configuration.
    pub fn nominal() -> Self {
        Self::with_full_params(28, 3, 1, ENHANCED_TREAD, 2048, 32, 96, 2).expect("nominal config")
    }

    /// Minimal M2 Enhanced configuration (17 rooms, 1 loop, 1024 extent).
    pub fn minimal() -> Self {
        Self::with_full_params(17, 1, 1, ENHANCED_TREAD, 1024, 32, 96, 1).expect("minimal config")
    }

    /// Maximal M2 Enhanced configuration (40 rooms, 6 loops, 3072 extent).
    pub fn maximal() -> Self {
        Self::with_full_params(40, 6, 3, ENHANCED_TREAD, 3072, 32, 96, 4).expect("maximal config")
    }

    // Accessors
    pub fn room_count(&self) -> u32 {
        self.room_count
    }
    pub fn loop_count(&self) -> u32 {
        self.loop_count
    }
    pub fn vertical_edges(&self) -> u32 {
        self.vertical_edges
    }
    /// Stair tread depth — always 16 after the Enhanced v2 stair repair.
    pub fn tread_depth(&self) -> i32 {
        self.tread_depth
    }
    pub fn xy_extent(&self) -> u32 {
        self.xy_extent
    }
    pub fn placement_candidates(&self) -> u32 {
        self.placement_candidates
    }
    pub fn max_placement_attempts(&self) -> u32 {
        self.max_placement_attempts
    }
    pub fn max_pillars_per_room(&self) -> u32 {
        self.max_pillars_per_room
    }
    pub fn layer_count(&self) -> u32 {
        ENHANCED_LAYER_COUNT
    }
    pub fn lower_floor_z(&self) -> i32 {
        ENHANCED_LOWER_FLOOR_Z
    }
    pub fn upper_floor_z(&self) -> i32 {
        ENHANCED_UPPER_FLOOR_Z
    }
    pub fn room_height(&self) -> i32 {
        ENHANCED_ROOM_HEIGHT
    }
    pub fn riser(&self) -> i32 {
        ENHANCED_RISER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_config_is_valid() {
        let cfg = EnhancedConfig::nominal();
        assert_eq!(cfg.room_count(), 28);
        assert_eq!(cfg.loop_count(), 3);
        assert_eq!(cfg.vertical_edges(), 1);
    }

    #[test]
    fn rejects_m1_room_count() {
        assert!(EnhancedConfig::new(8, 2, 1, 16, 2048).is_err());
    }

    #[test]
    fn rejects_one_layer() {
        // Layer count is frozen at 2, verified via layer_count() accessor
        let cfg = EnhancedConfig::nominal();
        assert_eq!(cfg.layer_count(), 2);
    }

    #[test]
    fn tread_always_sixteen() {
        let cfg = EnhancedConfig::nominal();
        assert_eq!(cfg.tread_depth(), 16);
        let cfg2 = EnhancedConfig::minimal();
        assert_eq!(cfg2.tread_depth(), 16);
        let cfg3 = EnhancedConfig::maximal();
        assert_eq!(cfg3.tread_depth(), 16);
    }

    #[test]
    fn rejects_vertical_edges_zero() {
        assert!(EnhancedConfig::new(28, 3, 0, 16, 2048).is_err());
    }

    #[test]
    fn rejects_vertical_edges_too_many() {
        assert!(EnhancedConfig::new(28, 3, 4, 16, 2048).is_err());
    }

    #[test]
    fn rejects_xy_not_quantum_aligned() {
        assert!(EnhancedConfig::new(28, 3, 1, 16, 2047).is_err());
    }
}
