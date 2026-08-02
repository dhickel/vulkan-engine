//! Zone construction for the semantic pacing blueprint.
//!
//! Assigns 1-3 correlated zones to the pacing blueprint. Adjacent semantic
//! slots prefer a shared zone; zone boundaries receive explicit transition
//! records. Zone records select theme-scoped variants, material roles,
//! roughness bands, prop families, and lighting families WITHOUT selecting
//! concrete theme assets.
//!
//! # Invariants
//!
//! - One grand-volume landmark
//! - One quiet negative-space room
//! - One dense set-piece with concrete invariant flags downstream stages
//!   must realize.
//!
//! # Contract
//!
//! - Theme NEVER participates in zone selection or candidate keys.
//! - Zone records are theme-independent; theme scoping is deferred to
//!   the composition stage.
//! - All zone records carry integer indices into the generated content
//!   arrays, not resolved theme assets.

use std::collections::BTreeMap;

use super::generated_content;
use super::ids::*;
use super::request::RichnessPreset;

// ── Zone family constants ──────────────────────────────────────────────────

/// Zone semantic family names (frozen).
pub(crate) const ZONE_FAMILIES: &[&str] =
    &["conflict", "entrance", "landmark", "quiet", "transition"];

/// Look up a zone family index by name. Returns `None` if unknown.
pub(crate) fn zone_family_index(name: &str) -> Option<u32> {
    generated_content::ZONE_NAMES
        .iter()
        .position(|&n| n == name)
        .map(|i| i as u32)
}

/// Number of material roles.
const MATERIAL_ROLE_COUNT: usize = 6;

/// Number of roughness bands.
const ROUGHNESS_BAND_COUNT: u32 = 4;

/// Number of prop families.
const PROP_FAMILY_COUNT: u32 = 5;

/// Number of light families.
const LIGHT_FAMILY_COUNT: u32 = 4;

// ── Zone invariant flags ───────────────────────────────────────────────────

/// Concrete invariant flags that downstream stages must realize.
///
/// These flags travel with the zone record and are committed before any
/// geometry or asset work. They are NOT theme-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ZoneInvariantFlags {
    /// This zone contains a grand-volume landmark (must be realized as a
    /// room with a ceiling ≥ 288 units and floor area ≥ 32×32 quantum cells).
    pub is_grand_volume: bool,
    /// This zone contains a quiet negative-space room (must be realized as
    /// a room with ≤ 2 props, ambient-only lighting, and no combat staging).
    pub is_quiet_negative_space: bool,
    /// This zone contains a dense set-piece (must be realized with ≥ 6 props,
    /// ≥ 3 light sources, and a bounded encounter staging area).
    pub is_dense_setpiece: bool,
}

impl ZoneInvariantFlags {
    /// Create empty flags (no invariants set).
    pub const fn empty() -> Self {
        Self {
            is_grand_volume: false,
            is_quiet_negative_space: false,
            is_dense_setpiece: false,
        }
    }

    /// Returns `true` if at least one invariant flag is set.
    pub fn any(self) -> bool {
        self.is_grand_volume || self.is_quiet_negative_space || self.is_dense_setpiece
    }
}

// ── Zone blueprint ─────────────────────────────────────────────────────────

/// The complete zone assignment portion of the pacing blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ZoneBlueprint {
    /// Zone records in canonical order.
    pub zones: BTreeMap<ZoneId, ZoneRecord>,
    /// Explicit zone transition records.
    pub transitions: Vec<ZoneTransition>,
    /// Invariant flags per zone.
    pub invariants: BTreeMap<ZoneId, ZoneInvariantFlags>,
    /// Beat-to-zone assignment (for reverse lookup).
    pub beat_zone_map: BTreeMap<BeatId, ZoneId>,
}

impl ZoneBlueprint {
    /// Validate that all invariants are satisfied.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check grand-volume invariant
        let has_grand_volume = self.invariants.values().any(|f| f.is_grand_volume);
        if !has_grand_volume {
            errors.push("no zone has the grand-volume invariant flag".to_string());
        }

        // Check quiet negative-space invariant
        let has_quiet = self.invariants.values().any(|f| f.is_quiet_negative_space);
        if !has_quiet {
            errors.push("no zone has the quiet-negative-space invariant flag".to_string());
        }

        // Check dense set-piece invariant
        let has_dense = self.invariants.values().any(|f| f.is_dense_setpiece);
        if !has_dense {
            errors.push("no zone has the dense-setpiece invariant flag".to_string());
        }

        // Zone count must be 1-3
        let zone_count = self.zones.len();
        if zone_count < 1 || zone_count > 3 {
            errors.push(format!("zone count {} outside [1, 3]", zone_count));
        }

        // Every zone must have at least one semantic slot
        for zone in self.zones.values() {
            if zone.semantic_slots.is_empty() {
                errors.push(format!("zone {:?} has no semantic slots", zone.id));
            }
        }

        // Transitions must reference valid zones
        for t in &self.transitions {
            if !self.zones.contains_key(&t.from_zone) {
                errors.push(format!("transition from unknown zone {:?}", t.from_zone));
            }
            if !self.zones.contains_key(&t.to_zone) {
                errors.push(format!("transition to unknown zone {:?}", t.to_zone));
            }
        }

        // Adjacent zones in transition must be different
        for t in &self.transitions {
            if t.from_zone == t.to_zone {
                errors.push(format!(
                    "transition {:?} -> {:?} has same source and destination",
                    t.from_zone, t.to_zone
                ));
            }
        }

        errors
    }

    /// Returns `true` if the zone blueprint passes all invariant checks.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

// ── Zone builder ───────────────────────────────────────────────────────────

/// Builder for constructing zone blueprints deterministically.
pub(crate) struct ZoneBuilder {
    zones: BTreeMap<ZoneId, ZoneRecord>,
    transitions: Vec<ZoneTransition>,
    invariants: BTreeMap<ZoneId, ZoneInvariantFlags>,
    beat_zone_map: BTreeMap<BeatId, ZoneId>,
    next_zone_id: u32,
}

impl ZoneBuilder {
    /// Create a new zone builder.
    pub fn new() -> Self {
        Self {
            zones: BTreeMap::new(),
            transitions: Vec::new(),
            invariants: BTreeMap::new(),
            beat_zone_map: BTreeMap::new(),
            next_zone_id: 0,
        }
    }

    /// Add a zone with the given semantic slot beats.
    ///
    /// Theme-scoped family indices are selected here; composition resolves
    /// them to concrete assets only after semantic planning.
    pub fn add_zone(
        &mut self,
        slots: Vec<BeatId>,
        theme_variant_index: u32,
        roughness_band: u32,
        prop_family_index: u32,
        light_family_index: u32,
        invariants: ZoneInvariantFlags,
    ) -> ZoneId {
        let id = ZoneId::new(self.next_zone_id);
        self.next_zone_id += 1;

        // Map beats to this zone
        for &bid in &slots {
            self.beat_zone_map.insert(bid, id);
        }

        self.invariants.insert(id, invariants);

        self.zones.insert(
            id,
            ZoneRecord {
                id,
                semantic_slots: slots,
                theme_variant_index,
                material_role_indices: (0..MATERIAL_ROLE_COUNT as u32).collect(),
                roughness_band,
                prop_family_index,
                light_family_index,
            },
        );

        id
    }

    /// Add a transition between two zones at a beat boundary.
    pub fn add_transition(
        &mut self,
        from_zone: ZoneId,
        to_zone: ZoneId,
        straddle_beats: (BeatId, BeatId),
    ) {
        self.transitions.push(ZoneTransition {
            from_zone,
            to_zone,
            straddle_beats,
        });
    }

    /// Build the zone blueprint.
    pub fn build(self) -> ZoneBlueprint {
        ZoneBlueprint {
            zones: self.zones,
            transitions: self.transitions,
            invariants: self.invariants,
            beat_zone_map: self.beat_zone_map,
        }
    }
}

impl Default for ZoneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Zone assignment from pacing blueprint ──────────────────────────────────

/// Assign zones to an already-constructed pacing blueprint.
///
/// This is called from `build_pacing_blueprint` during the zone assignment step.
/// It takes the beats and beat order and produces zone records.
pub(crate) fn build_zone_blueprint(
    _preset: RichnessPreset,
    zone_count: u32,
    beat_order: &[BeatId],
    beats: &BTreeMap<BeatId, Beat>,
) -> ZoneBlueprint {
    let mut builder = ZoneBuilder::new();

    if beat_order.is_empty() || zone_count == 0 {
        return builder.build();
    }

    // Determine zone assignments: distribute beats across zones
    let beats_per_zone = (beat_order.len() as u32).div_ceil(zone_count);

    // Track which invariants have been assigned
    let mut grand_volume_assigned = false;
    let mut quiet_assigned = false;
    let mut dense_assigned = false;

    for z in 0..zone_count {
        let start = (z * beats_per_zone) as usize;
        let end = (((z + 1) * beats_per_zone) as usize).min(beat_order.len());
        let slots: Vec<BeatId> = beat_order[start..end].to_vec();

        // Determine invariant flags for this zone
        let mut invariants = ZoneInvariantFlags::empty();

        // Check if any beat in this zone has invariant markers
        for bid in &slots {
            if let Some(beat) = beats.get(bid) {
                if beat.is_grand_volume && !grand_volume_assigned {
                    invariants.is_grand_volume = true;
                    grand_volume_assigned = true;
                }
                if beat.is_quiet_negative_space && !quiet_assigned {
                    invariants.is_quiet_negative_space = true;
                    quiet_assigned = true;
                }
                if beat.is_dense_setpiece && !dense_assigned {
                    invariants.is_dense_setpiece = true;
                    dense_assigned = true;
                }
            }
        }

        // Theme-scoped family indices; concrete theme assets remain deferred.
        let theme_variant = z;
        let roughness_band = z % ROUGHNESS_BAND_COUNT;
        let prop_family = z % PROP_FAMILY_COUNT;
        let light_family = z % LIGHT_FAMILY_COUNT;

        builder.add_zone(
            slots,
            theme_variant,
            roughness_band,
            prop_family,
            light_family,
            invariants,
        );
    }

    // Ensure all invariants are assigned
    let all_zones: Vec<ZoneId> = builder.zones.keys().copied().collect();

    if !grand_volume_assigned && !all_zones.is_empty() {
        if let Some(inv) = builder.invariants.get_mut(&all_zones[0]) {
            inv.is_grand_volume = true;
        }
    }
    if !quiet_assigned && all_zones.len() >= 2 {
        if let Some(inv) = builder.invariants.get_mut(&all_zones[1]) {
            inv.is_quiet_negative_space = true;
        }
    }
    if !dense_assigned && !all_zones.is_empty() {
        let idx = if all_zones.len() >= 3 {
            2
        } else {
            all_zones.len() - 1
        };
        if let Some(inv) = builder.invariants.get_mut(&all_zones[idx]) {
            inv.is_dense_setpiece = true;
        }
    }

    // Create transitions between adjacent zones
    for z in 0..zone_count.saturating_sub(1) {
        let from_zone = ZoneId::new(z);
        let to_zone = ZoneId::new(z + 1);
        let boundary_start = ((z + 1) * beats_per_zone) as usize;

        if boundary_start > 0 && boundary_start < beat_order.len() {
            builder.add_transition(
                from_zone,
                to_zone,
                (beat_order[boundary_start - 1], beat_order[boundary_start]),
            );
        }
    }

    builder.build()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn make_test_beat(id: u32, grand: bool, quiet: bool, dense: bool) -> Beat {
        Beat {
            id: BeatId::new(id),
            beat_type: BeatType::LandmarkPeak,
            requests: Vec::new(),
            density: if dense {
                DensityClass::Dense
            } else if quiet {
                DensityClass::Quiet
            } else {
                DensityClass::Transition
            },
            degree: DegreeIntent::new(1, 3),
            progression: ProgressionOrder::new(id),
            on_critical_path: true,
            is_grand_volume: grand,
            is_quiet_negative_space: quiet,
            is_dense_setpiece: dense,
        }
    }

    #[test]
    fn zone_blueprint_empty() {
        let bp = build_zone_blueprint(RichnessPreset::Sparse, 0, &[], &BTreeMap::new());
        assert!(bp.zones.is_empty());
    }

    #[test]
    fn zone_blueprint_single_zone() {
        let beats: BTreeMap<BeatId, Beat> = [
            (BeatId::new(0), make_test_beat(0, true, false, false)),
            (BeatId::new(1), make_test_beat(1, false, true, false)),
            (BeatId::new(2), make_test_beat(2, false, false, true)),
        ]
        .into_iter()
        .collect();
        let beat_order: Vec<BeatId> = (0..3).map(BeatId::new).collect();

        let bp = build_zone_blueprint(RichnessPreset::Sparse, 1, &beat_order, &beats);
        assert_eq!(bp.zones.len(), 1);
        let zone = bp.zones.get(&ZoneId::new(0)).unwrap();
        assert_eq!(zone.semantic_slots.len(), 3);

        // All three invariants should be in the single zone
        let inv = bp.invariants.get(&ZoneId::new(0)).unwrap();
        assert!(inv.is_grand_volume);
        assert!(inv.is_quiet_negative_space);
        assert!(inv.is_dense_setpiece);
    }

    #[test]
    fn zone_blueprint_three_zones() {
        let mut beats = BTreeMap::new();
        let mut beat_order = Vec::new();
        for i in 0..6 {
            let grand = i == 0;
            let quiet = i == 2;
            let dense = i == 4;
            beats.insert(BeatId::new(i), make_test_beat(i, grand, quiet, dense));
            beat_order.push(BeatId::new(i));
        }

        let bp = build_zone_blueprint(RichnessPreset::Rich, 3, &beat_order, &beats);
        assert_eq!(bp.zones.len(), 3);

        // Each zone should have 2 beats
        for z in 0..3 {
            let zone = bp.zones.get(&ZoneId::new(z)).unwrap();
            assert_eq!(
                zone.semantic_slots.len(),
                2,
                "zone {} has wrong slot count",
                z
            );
        }

        // Check invariants
        let inv0 = bp.invariants.get(&ZoneId::new(0)).unwrap();
        let inv1 = bp.invariants.get(&ZoneId::new(1)).unwrap();
        let inv2 = bp.invariants.get(&ZoneId::new(2)).unwrap();
        assert!(inv0.is_grand_volume);
        assert!(inv1.is_quiet_negative_space);
        assert!(inv2.is_dense_setpiece);

        // Should have 2 transitions
        assert_eq!(bp.transitions.len(), 2);
    }

    #[test]
    fn zone_blueprint_validates_invariants() {
        let bp = build_zone_blueprint(RichnessPreset::Sparse, 1, &[], &BTreeMap::new());
        let errors = bp.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("grand-volume")));
    }

    #[test]
    fn zone_blueprint_rejects_empty_zones() {
        let mut builder = ZoneBuilder::new();
        builder.add_zone(
            Vec::new(),
            0,
            0,
            0,
            0,
            ZoneInvariantFlags {
                is_grand_volume: true,
                is_quiet_negative_space: true,
                is_dense_setpiece: true,
            },
        );
        let bp = builder.build();
        let errors = bp.validate();
        assert!(errors.iter().any(|e| e.contains("no semantic slots")));
    }

    #[test]
    fn zone_blueprint_rejects_same_zone_transition() {
        let mut builder = ZoneBuilder::new();
        let zid = builder.add_zone(
            vec![BeatId::new(0)],
            0,
            0,
            0,
            0,
            ZoneInvariantFlags {
                is_grand_volume: true,
                is_quiet_negative_space: true,
                is_dense_setpiece: true,
            },
        );
        builder.add_transition(zid, zid, (BeatId::new(0), BeatId::new(0)));
        let bp = builder.build();
        let errors = bp.validate();
        assert!(errors
            .iter()
            .any(|e| e.contains("same source and destination")));
    }

    #[test]
    fn zone_family_index_lookup() {
        for (i, name) in ZONE_FAMILIES.iter().enumerate() {
            assert_eq!(zone_family_index(name), Some(i as u32));
        }
        assert_eq!(zone_family_index("nonexistent"), None);
        assert_eq!(zone_family_index(""), None);
    }

    #[test]
    fn zone_families_match_generated() {
        assert_eq!(ZONE_FAMILIES.len(), generated_content::ZONE_NAMES.len());
        for (i, name) in ZONE_FAMILIES.iter().enumerate() {
            assert_eq!(generated_content::ZONE_NAMES[i], *name);
        }
    }

    #[test]
    fn invariant_flags_empty_has_none() {
        let f = ZoneInvariantFlags::empty();
        assert!(!f.any());
        assert!(!f.is_grand_volume);
        assert!(!f.is_quiet_negative_space);
        assert!(!f.is_dense_setpiece);
    }

    #[test]
    fn invariant_flags_any_detects_set_flags() {
        let f = ZoneInvariantFlags {
            is_grand_volume: true,
            is_quiet_negative_space: false,
            is_dense_setpiece: false,
        };
        assert!(f.any());
    }

    #[test]
    fn beat_zone_map_correct() {
        let mut builder = ZoneBuilder::new();
        let z0 = builder.add_zone(
            vec![BeatId::new(0), BeatId::new(1)],
            0,
            0,
            0,
            0,
            ZoneInvariantFlags {
                is_grand_volume: true,
                is_quiet_negative_space: false,
                is_dense_setpiece: false,
            },
        );
        let z1 = builder.add_zone(
            vec![BeatId::new(2)],
            1,
            1,
            1,
            1,
            ZoneInvariantFlags {
                is_grand_volume: false,
                is_quiet_negative_space: true,
                is_dense_setpiece: true,
            },
        );

        let bp = builder.build();
        assert_eq!(bp.beat_zone_map.get(&BeatId::new(0)), Some(&z0));
        assert_eq!(bp.beat_zone_map.get(&BeatId::new(1)), Some(&z0));
        assert_eq!(bp.beat_zone_map.get(&BeatId::new(2)), Some(&z1));
        assert_eq!(bp.beat_zone_map.get(&BeatId::new(99)), None);
    }
}
