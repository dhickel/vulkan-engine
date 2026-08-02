//! Supporting types for the machine-generated Richness content constants.
//!
//! These types mirror the closed schema enums from `tools/richness_content_codegen`.
//! The generated `generated_content.rs` imports them via `use super::content_types::*`.
//!
//! Everything in this module is crate-private — no public re-exports until the
//! atomic release phase.

// ── Shape rule ─────────────────────────────────────────────────────────────

/// The geometric footprint family for an archetype room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShapeRule {
    /// Axis-aligned rectangular room.
    Rectangle,
    /// Cardinal + 45° chamfered octagonal footprint.
    Octagon,
    /// Single chamfer cut on one or more corners.
    Chamfer,
    /// Composite partitioned footprint (multiple sub-volumes).
    CompositePartition,
}

// ── Layer occupancy ────────────────────────────────────────────────────────

/// Which Z layer(s) this archetype may occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LayerOccupancy {
    /// Lower layer only (Z = 0..176).
    Lower,
    /// Upper layer only (Z = 192..368).
    Upper,
    /// Both layers (composite multi-storey reservation).
    Both,
}

// ── Rarity tier ────────────────────────────────────────────────────────────

/// Normalized rarity with approximate selection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RarityTier {
    /// ~70% of selections, repeatable after exhaustion.
    Common,
    /// ~25% of selections.
    Uncommon,
    /// ~5% of selections, no-repeat, one-per-map cap.
    Rare,
    /// ~1% of selections, no-repeat, one-per-map cap.
    Legendary,
}

// ── Vertical recipe ────────────────────────────────────────────────────────

/// Vertical feature attached to (or integral with) an archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum VerticalRecipe {
    /// No vertical feature.
    None,
    /// Type A or B 12-tread stairwell.
    Stairwell,
    /// Ladder shaft for climb traversal.
    LadderShaft,
    /// One-way drop hole.
    DropHole,
    /// Open (un-laddered) stairwell opening.
    OpenStairwell,
    /// 12-step spiral stair.
    SpiralStair,
}

// ── Collision behavior ─────────────────────────────────────────────────────

/// Whether a prop is collidable or detail-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CollisionBehavior {
    /// Collidable: emits solid brushes / clip hull.
    Collidable,
    /// Detail-only: visual only, no collision.
    DetailOnly,
}

// ── Placement class ────────────────────────────────────────────────────────

/// Where a light entity is placed relative to room geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PlacementClass {
    /// Wall-mounted sconce/torch light.
    Wall,
    /// Ceiling-mounted light.
    Ceiling,
    /// Floor-standing light.
    Floor,
    /// Hanging pendant light.
    Pendant,
    /// Free-floating ambient volume light.
    Ambient,
}

// ── Falloff style ──────────────────────────────────────────────────────────

/// Light attenuation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FalloffStyle {
    /// Standard Quake linear falloff.
    Linear,
    /// Inverse-square falloff.
    InverseSquare,
}

// ── Lookup functions ───────────────────────────────────────────────────────

/// Look up a prop index by its stable ID.
///
/// Panics (in const context: compile error) on unknown IDs.
/// The IDs must be in the same lexical order as `PROP_IDS` in generated_content.
#[inline]
pub const fn prop_index(id: &str) -> u32 {
    let ids = super::generated_content::PROP_IDS;
    let mut lo = 0;
    let mut hi = ids.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let cmp = const_str_cmp(id, ids[mid]);
        if cmp < 0 {
            hi = mid;
        } else if cmp > 0 {
            lo = mid + 1;
        } else {
            return mid as u32;
        }
    }
    panic!("unknown prop ID");
}

/// Look up a light recipe index by its stable ID.
///
/// Panics (in const context: compile error) on unknown IDs.
/// The IDs must be in the same lexical order as `LIGHT_RECIPE_IDS` in generated_content.
#[inline]
pub const fn light_index(id: &str) -> u32 {
    let ids = super::generated_content::LIGHT_RECIPE_IDS;
    let mut lo = 0;
    let mut hi = ids.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let cmp = const_str_cmp(id, ids[mid]);
        if cmp < 0 {
            hi = mid;
        } else if cmp > 0 {
            lo = mid + 1;
        } else {
            return mid as u32;
        }
    }
    panic!("unknown light recipe ID");
}

/// Const-compatible string comparison. Returns negative/zero/positive.
const fn const_str_cmp(a: &str, b: &str) -> i32 {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut i = 0;
    while i < a.len() && i < b.len() {
        if a[i] < b[i] {
            return -1;
        } else if a[i] > b[i] {
            return 1;
        }
        i += 1;
    }
    if a.len() < b.len() {
        -1
    } else if a.len() > b.len() {
        1
    } else {
        0
    }
}
