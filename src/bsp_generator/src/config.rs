use crate::error::GeneratorError;

/// Construction quantum: all geometry snaps to multiples of this value.
pub const CONSTRUCTION_QUANTUM: u32 = 16;

/// Minimum constructible Z span: `2 * CONSTRUCTION_QUANTUM + 80 = 112`.
///
/// A vertical shell must accommodate a 16-unit floor slab, an 80-unit clear
/// throat, and a 16-unit ceiling slab. Any `z_span` below this cannot produce
/// walkable headroom.
pub const MIN_Z_SPAN: u32 = 2 * CONSTRUCTION_QUANTUM + 80; // 112

// ── M1 frozen bounds ──────────────────────────────────────────────────────

/// M1 room count range (inclusive).
pub const M1_ROOM_COUNT_MIN: u32 = 8;
pub const M1_ROOM_COUNT_MAX: u32 = 16;

/// M1 loop count range (inclusive).
pub const M1_LOOP_COUNT_MIN: u32 = 0;
pub const M1_LOOP_COUNT_MAX: u32 = 2;

/// M1 maximum XY extent per axis (Quake units).
pub const M1_XY_MAX: u32 = 1536;

/// M1 maximum Z span (Quake units).
pub const M1_Z_MAX: u32 = 256;

// ── M2 frozen bounds ──────────────────────────────────────────────────────

/// M2 room count range (inclusive).
pub const M2_ROOM_COUNT_MIN: u32 = 17;
pub const M2_ROOM_COUNT_MAX: u32 = 40;

/// M2 loop count range (inclusive).
pub const M2_LOOP_COUNT_MIN: u32 = 1;
pub const M2_LOOP_COUNT_MAX: u32 = 6;

/// M2 maximum XY extent per axis (Quake units).
pub const M2_XY_MAX: u32 = 3072;

/// M2 maximum Z span (Quake units).
pub const M2_Z_MAX: u32 = 384;

// ── Map class ─────────────────────────────────────────────────────────────

/// The two output tiers defined by the frozen generation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MapClass {
    /// Small test/arena maps: 8–16 rooms, 0–2 loops, ≤1536².
    M1,
    /// Medium single-player dungeons: 17–40 rooms, 1–6 loops, ≤3072².
    M2,
}

impl MapClass {
    /// Maximum placement candidates per room attempt for this class.
    pub fn max_placement_candidates(self) -> u32 {
        match self {
            MapClass::M1 => 16,
            MapClass::M2 => 32,
        }
    }

    /// Maximum placement attempts per room per candidate for this class.
    pub fn max_placement_attempts(self) -> u32 {
        match self {
            MapClass::M1 => 64,
            MapClass::M2 => 96,
        }
    }

    /// Maximum A* expansions per candidate for this class.
    pub fn max_astar_expansions(self) -> u32 {
        match self {
            MapClass::M1 => 131_072,
            MapClass::M2 => 524_288,
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────

/// Raw dungeon generator configuration before validation.
///
/// All fields are caller-supplied; pass through [`DungeonConfig::validate`] to
/// obtain a [`ValidatedConfig`] that is guaranteed to satisfy all frozen
/// contract bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DungeonConfig {
    /// Map output tier.
    pub class: MapClass,
    /// Number of rooms to place.
    pub room_count: u32,
    /// Number of explicit spatial loops to introduce.
    pub loop_count: u32,
    /// Outer XY extent in Quake units: `(x, y)`.
    pub xy_bounds: (u32, u32),
    /// Total Z span in Quake units (common floor-to-ceiling height).
    pub z_span: u32,
    /// Placement candidates generated per room attempt.
    pub placement_candidates: u32,
    /// Maximum placement attempts per room per candidate.
    pub max_placement_attempts: u32,
    /// Maximum A* node expansions per corridor candidate.
    pub max_astar_expansions: u32,
}

/// A fully validated dungeon configuration.
///
/// Obtainable only via [`DungeonConfig::validate`]; the constructor guarantees
/// all fields are within the frozen contract bounds for the declared
/// [`MapClass`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidatedConfig {
    pub class: MapClass,
    pub room_count: u32,
    pub loop_count: u32,
    pub xy_bounds: (u32, u32),
    pub z_span: u32,
    pub placement_candidates: u32,
    pub max_placement_attempts: u32,
    pub max_astar_expansions: u32,
}

impl DungeonConfig {
    /// Validate all fields against the frozen contract for the declared
    /// [`MapClass`].
    ///
    /// # Errors
    ///
    /// Returns [`GeneratorError::InvalidConfig`] if any field is out of range,
    /// unsnapped from the construction quantum, zero, or would overflow a
    /// bounds computation.
    pub fn validate(&self) -> Result<ValidatedConfig, GeneratorError> {
        // ── MapClass existence checked implicitly by enum ─────────────────

        // ── Per-class range checks ────────────────────────────────────────
        let (room_min, room_max, loop_min, loop_max, xy_max, z_max) = match self.class {
            MapClass::M1 => (
                M1_ROOM_COUNT_MIN,
                M1_ROOM_COUNT_MAX,
                M1_LOOP_COUNT_MIN,
                M1_LOOP_COUNT_MAX,
                M1_XY_MAX,
                M1_Z_MAX,
            ),
            MapClass::M2 => (
                M2_ROOM_COUNT_MIN,
                M2_ROOM_COUNT_MAX,
                M2_LOOP_COUNT_MIN,
                M2_LOOP_COUNT_MAX,
                M2_XY_MAX,
                M2_Z_MAX,
            ),
        };

        if self.room_count < room_min || self.room_count > room_max {
            return Err(GeneratorError::InvalidConfig(format!(
                "room_count {} outside {:?} range [{}, {}]",
                self.room_count, self.class, room_min, room_max,
            )));
        }

        if self.loop_count < loop_min || self.loop_count > loop_max {
            return Err(GeneratorError::InvalidConfig(format!(
                "loop_count {} outside {:?} range [{}, {}]",
                self.loop_count, self.class, loop_min, loop_max,
            )));
        }

        // ── XY bounds checks ──────────────────────────────────────────────
        let (bx, by) = self.xy_bounds;

        // Zero dimensions
        if bx == 0 || by == 0 {
            return Err(GeneratorError::InvalidConfig(format!(
                "xy_bounds ({}, {}) must be non-zero",
                bx, by,
            )));
        }

        // Unsnapped to quantum
        if bx % CONSTRUCTION_QUANTUM != 0 || by % CONSTRUCTION_QUANTUM != 0 {
            return Err(GeneratorError::InvalidConfig(format!(
                "xy_bounds ({}, {}) must be multiples of construction quantum {}",
                bx, by, CONSTRUCTION_QUANTUM,
            )));
        }

        // Per-class maximum
        if bx > xy_max || by > xy_max {
            return Err(GeneratorError::InvalidConfig(format!(
                "xy_bounds ({}, {}) exceeds {:?} maximum {} per axis",
                bx, by, self.class, xy_max,
            )));
        }

        // ── Z span checks ─────────────────────────────────────────────────
        if self.z_span == 0 {
            return Err(GeneratorError::InvalidConfig(
                "z_span must be non-zero".to_string(),
            ));
        }

        if self.z_span < MIN_Z_SPAN {
            return Err(GeneratorError::InvalidConfig(format!(
                "z_span {} is below the minimum constructible span {} (16 floor + 80 clear + 16 ceiling)",
                self.z_span, MIN_Z_SPAN,
            )));
        }

        if self.z_span % CONSTRUCTION_QUANTUM != 0 {
            return Err(GeneratorError::InvalidConfig(format!(
                "z_span {} must be a multiple of construction quantum {}",
                self.z_span, CONSTRUCTION_QUANTUM,
            )));
        }

        if self.z_span > z_max {
            return Err(GeneratorError::InvalidConfig(format!(
                "z_span {} exceeds {:?} maximum {}",
                self.z_span, self.class, z_max,
            )));
        }

        // ── Overflow guards ───────────────────────────────────────────────
        // xy_bounds area must not overflow u32
        bx.checked_mul(by)
            .ok_or(GeneratorError::ArithmeticOverflow)
            .map_err(|_| {
                GeneratorError::InvalidConfig(format!(
                    "xy_bounds ({}, {}) area overflows u32",
                    bx, by,
                ))
            })?;

        // z_span * z_span (used as rough volume bound) must not overflow
        self.z_span
            .checked_mul(self.z_span)
            .ok_or(GeneratorError::ArithmeticOverflow)
            .map_err(|_| {
                GeneratorError::InvalidConfig(format!(
                    "z_span {} squared overflows u32",
                    self.z_span,
                ))
            })?;

        // ── Placement parameter checks ────────────────────────────────────
        if self.placement_candidates == 0 {
            return Err(GeneratorError::InvalidConfig(
                "placement_candidates must be non-zero".to_string(),
            ));
        }

        if self.placement_candidates > self.class.max_placement_candidates() {
            return Err(GeneratorError::InvalidConfig(format!(
                "placement_candidates {} exceeds {:?} maximum {}",
                self.placement_candidates,
                self.class,
                self.class.max_placement_candidates(),
            )));
        }

        if self.max_placement_attempts == 0 {
            return Err(GeneratorError::InvalidConfig(
                "max_placement_attempts must be non-zero".to_string(),
            ));
        }

        if self.max_placement_attempts > self.class.max_placement_attempts() {
            return Err(GeneratorError::InvalidConfig(format!(
                "max_placement_attempts {} exceeds {:?} maximum {}",
                self.max_placement_attempts,
                self.class,
                self.class.max_placement_attempts(),
            )));
        }

        if self.max_astar_expansions == 0 {
            return Err(GeneratorError::InvalidConfig(
                "max_astar_expansions must be non-zero".to_string(),
            ));
        }

        if self.max_astar_expansions > self.class.max_astar_expansions() {
            return Err(GeneratorError::InvalidConfig(format!(
                "max_astar_expansions {} exceeds {:?} maximum {}",
                self.max_astar_expansions,
                self.class,
                self.class.max_astar_expansions(),
            )));
        }

        Ok(ValidatedConfig {
            class: self.class,
            room_count: self.room_count,
            loop_count: self.loop_count,
            xy_bounds: self.xy_bounds,
            z_span: self.z_span,
            placement_candidates: self.placement_candidates,
            max_placement_attempts: self.max_placement_attempts,
            max_astar_expansions: self.max_astar_expansions,
        })
    }

    /// Create the nominal M1 configuration (12 rooms, 1 loop, 1024², Z 192).
    pub fn nominal_m1() -> Self {
        DungeonConfig {
            class: MapClass::M1,
            room_count: 12,
            loop_count: 1,
            xy_bounds: (1024, 1024),
            z_span: 192,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }
    }

    /// Create the nominal M2 configuration (28 rooms, 3 loops, 2048², Z 256).
    pub fn nominal_m2() -> Self {
        DungeonConfig {
            class: MapClass::M2,
            room_count: 28,
            loop_count: 3,
            xy_bounds: (2048, 2048),
            z_span: 256,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_m1_validates() {
        let cfg = DungeonConfig::nominal_m1();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn nominal_m2_validates() {
        let cfg = DungeonConfig::nominal_m2();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validated_config_equals_input() {
        let cfg = DungeonConfig::nominal_m1();
        let v = cfg.validate().unwrap();
        assert_eq!(v.class, cfg.class);
        assert_eq!(v.room_count, cfg.room_count);
        assert_eq!(v.loop_count, cfg.loop_count);
        assert_eq!(v.xy_bounds, cfg.xy_bounds);
        assert_eq!(v.z_span, cfg.z_span);
    }
}
