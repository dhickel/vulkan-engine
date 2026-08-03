//! Typed stable IDs and immutable records for the semantic pacing blueprint
//! and the 3D reservation / occupancy system.
//!
//! Every record in the blueprint carries a typed newtype ID. Records are
//! immutable and ordered canonically (BTreeMap / sorted vectors). Theme never
//! participates in candidate keys or record identity.
//!
//! # Contract
//!
//! - All IDs are `Copy + Eq + Ord + Hash` so they can serve as map keys.
//! - Beat IDs, zone IDs, landmark IDs, reservation IDs, and branch IDs are
//!   distinct namespaces.
//! - Records use `BTreeMap` or explicitly sorted vectors for canonical order.
//! - No floats; only integer, enum, and fixed-point primitives.

use std::collections::BTreeMap;

use super::content_types::RarityTier;
use super::generated_content;
use super::request::RichnessPreset;
use super::zones::ZoneBlueprint;

// ── Beat type ──────────────────────────────────────────────────────────────

/// The semantic role of a pacing beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum BeatType {
    /// The player enters the dungeon.
    Entrance,
    /// The player descends into the dungeon (may be on the critical path or
    /// a branch).
    Descent,
    /// A major visual/structural landmark on the critical path.
    LandmarkPeak,
    /// The dungeon concludes / exit / climax.
    Release,
    /// Terminal semantic node of a side branch.
    SideBranchLeaf,
}

impl BeatType {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Entrance => "entrance",
            Self::Descent => "descent",
            Self::LandmarkPeak => "landmark_peak",
            Self::Release => "release",
            Self::SideBranchLeaf => "side_branch_leaf",
        }
    }
}

// ── Density class ──────────────────────────────────────────────────────────

/// Whether a beat or room is dense or quiet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum DensityClass {
    /// High density: many features, props, lights.
    Dense,
    /// Low density: quiet negative space.
    Quiet,
    /// Transitional density between beats.
    Transition,
}

impl DensityClass {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Quiet => "quiet",
            Self::Transition => "transition",
        }
    }
}

// ── Rarity class ───────────────────────────────────────────────────────────

/// Semantic rarity class for blueprint records.
///
/// Mirrors `RarityTier` from `content_types` but is semantically anchored to
/// the blueprint selection context (natural vs forced, repeat policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum RarityClass {
    /// ~70% of natural selections, repeatable after exhaustion.
    Common,
    /// ~25% of natural selections.
    Uncommon,
    /// ~5% of selections, no-repeat, one-per-map cap.
    Rare,
    /// ~1% of selections, no-repeat, one-per-map cap.
    Legendary,
}

impl RarityClass {
    /// Convert from `RarityTier`.
    pub fn from_tier(tier: RarityTier) -> Self {
        match tier {
            RarityTier::Common => Self::Common,
            RarityTier::Uncommon => Self::Uncommon,
            RarityTier::Rare => Self::Rare,
            RarityTier::Legendary => Self::Legendary,
        }
    }

    /// Normalized integer weight for this rarity class.
    pub fn weight(self) -> u32 {
        match self {
            Self::Common => 70,
            Self::Uncommon => 25,
            Self::Rare => 5,
            Self::Legendary => 1,
        }
    }

    /// Whether this rarity class forbids repeats within a single map.
    pub fn no_repeat(self) -> bool {
        matches!(self, Self::Rare | Self::Legendary)
    }

    /// Whether this rarity class has a one-per-map cap.
    pub fn one_per_map(self) -> bool {
        matches!(self, Self::Rare | Self::Legendary)
    }
}

// ── Typed IDs ──────────────────────────────────────────────────────────────

macro_rules! typed_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub(crate) struct $name(pub(crate) u32);

        impl $name {
            /// Create a new ID from a raw u32.
            pub const fn new(id: u32) -> Self {
                Self(id)
            }

            /// The raw u32 value.
            pub const fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

typed_id!(BeatId, "Typed identifier for a pacing beat.");
typed_id!(ZoneId, "Typed identifier for a semantic zone.");
typed_id!(
    ArchetypeRequestId,
    "Typed identifier for an archetype request."
);
typed_id!(BranchId, "Typed identifier for a side-branch.");
typed_id!(LandmarkId, "Typed identifier for a critical-path landmark.");
typed_id!(
    ShortcutId,
    "Typed identifier for a backward shortcut intent."
);
typed_id!(
    ReservationId,
    "Typed identifier for an immutable occupancy reservation."
);
typed_id!(
    EdgeId,
    "Typed identifier for a candidate edge in the multigraph topology."
);
typed_id!(
    RouteId,
    "Typed identifier for a committed route reservation."
);
typed_id!(
    PortalId,
    "Typed identifier for a committed portal throat reservation."
);
typed_id!(
    TurnId,
    "Typed identifier for a committed turn / junction reservation."
);
typed_id!(
    WallChainId,
    "Typed identifier for a shared structural wall chain."
);

// ── Semantic ID ────────────────────────────────────────────────────────────

/// A unified key across all committed semantic identity namespaces.
///
/// Used to key variation and complexity decisions so that reservations,
/// routes, portals, turns, beats, zones, and archetype requests share one
/// key domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SemanticId {
    Reservation(ReservationId),
    Route(RouteId),
    Portal(PortalId),
    Turn(TurnId),
    Beat(BeatId),
    Zone(ZoneId),
    ArchetypeRequest(ArchetypeRequestId),
}

impl SemanticId {
    /// Stable tag prefix for diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Reservation(_) => "reservation",
            Self::Route(_) => "route",
            Self::Portal(_) => "portal",
            Self::Turn(_) => "turn",
            Self::Beat(_) => "beat",
            Self::Zone(_) => "zone",
            Self::ArchetypeRequest(_) => "archetype_request",
        }
    }

    /// Raw u32 for hashing and ordering.
    pub fn raw(self) -> u32 {
        match self {
            Self::Reservation(id) => id.raw(),
            Self::Route(id) => id.raw(),
            Self::Portal(id) => id.raw(),
            Self::Turn(id) => id.raw(),
            Self::Beat(id) => id.raw(),
            Self::Zone(id) => id.raw(),
            Self::ArchetypeRequest(id) => id.raw(),
        }
    }
}

impl From<ReservationId> for SemanticId {
    fn from(id: ReservationId) -> Self {
        Self::Reservation(id)
    }
}
impl From<RouteId> for SemanticId {
    fn from(id: RouteId) -> Self {
        Self::Route(id)
    }
}
impl From<PortalId> for SemanticId {
    fn from(id: PortalId) -> Self {
        Self::Portal(id)
    }
}
impl From<TurnId> for SemanticId {
    fn from(id: TurnId) -> Self {
        Self::Turn(id)
    }
}
impl From<BeatId> for SemanticId {
    fn from(id: BeatId) -> Self {
        Self::Beat(id)
    }
}
impl From<ZoneId> for SemanticId {
    fn from(id: ZoneId) -> Self {
        Self::Zone(id)
    }
}
impl From<ArchetypeRequestId> for SemanticId {
    fn from(id: ArchetypeRequestId) -> Self {
        Self::ArchetypeRequest(id)
    }
}

// ── Archetype index ────────────────────────────────────────────────────────

/// Index into the generated `ARCHETYPE_IDS` array.
///
/// Valid range: `0..30` (the 30 production archetypes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ArchetypeIndex(pub(crate) u32);

impl ArchetypeIndex {
    /// Create from an index. Panics (debug only) if out of range.
    pub const fn new(idx: u32) -> Self {
        debug_assert!((idx as usize) < generated_content::ARCHETYPE_COUNT);
        Self(idx)
    }

    /// The stable string ID for this archetype.
    pub fn id_str(self) -> &'static str {
        generated_content::ARCHETYPE_IDS[self.0 as usize]
    }

    /// The raw u32 index.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Look up an archetype index by its stable string ID.
    /// Returns `None` if unknown.
    pub fn from_id_str(id: &str) -> Option<Self> {
        generated_content::ARCHETYPE_IDS
            .iter()
            .position(|&s| s == id)
            .map(|i| Self(i as u32))
    }

    /// All 30 archetype indices in canonical order.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..generated_content::ARCHETYPE_COUNT as u32).map(Self)
    }
}

// ── Landmark force reason ──────────────────────────────────────────────────

/// Why a landmark was forced into the blueprint, rather than arising from
/// natural rarity-driven selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum LandmarkForceReason {
    /// Required by the preset (Sparse=1, Moderate=2, Rich=3).
    PresetRequirement,
    /// Required to satisfy the grand-volume invariant.
    GrandVolume,
    /// Required to satisfy the dense-setpiece invariant.
    DenseSetpiece,
    /// Required to satisfy the quiet-negative-space invariant.
    QuietNegativeSpace,
}

// ── Payoff type ─────────────────────────────────────────────────────────────

/// The observable payoff at the termination of a side-branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PayoffType {
    /// A backward shortcut connecting a later point to an earlier point on the
    /// progression path without bypassing any unvisited mandatory beat.
    Shortcut,
    /// A discovery landmark — a minor but intentional visual reward.
    DiscoveryLandmark,
    /// A lore marker — an inscribed message, story fragment, or environmental
    /// storytelling element.
    LoreMarker,
    /// An authored treasure tableau — a deliberate arrangement of valuable
    /// props and lighting.
    AuthoredTreasureTableau,
}

impl PayoffType {
    /// Lowercase exact tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Shortcut => "shortcut",
            Self::DiscoveryLandmark => "discovery_landmark",
            Self::LoreMarker => "lore_marker",
            Self::AuthoredTreasureTableau => "authored_treasure_tableau",
        }
    }

    /// Returns `true` if this payoff is a label-only (non-observable) variant.
    /// All variants are observable by construction; this always returns `false`.
    pub fn is_label_only(self) -> bool {
        false
    }
}

// ── Degree intent ──────────────────────────────────────────────────────────

/// Bounded exit-degree intent for a beat or archetype request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DegreeIntent {
    /// Minimum number of exits (portals) from this beat.
    pub min_exits: u32,
    /// Maximum number of exits (portals) from this beat.
    pub max_exits: u32,
}

impl DegreeIntent {
    /// Create a new `DegreeIntent`. `min_exits` must be ≤ `max_exits`.
    pub fn new(min_exits: u32, max_exits: u32) -> Self {
        debug_assert!(min_exits <= max_exits);
        Self {
            min_exits,
            max_exits,
        }
    }

    /// Returns `true` if this intent is valid (min ≤ max).
    pub fn is_valid(self) -> bool {
        self.min_exits <= self.max_exits
    }
}

// ── Progression order ──────────────────────────────────────────────────────

/// Strict ordinal position in the critical-path progression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ProgressionOrder(pub(crate) u32);

impl ProgressionOrder {
    /// The first beat on the critical path.
    pub const FIRST: Self = Self(0);

    /// Create from an ordinal.
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// The raw ordinal.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// The next ordinal in sequence.
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

// ── Archetype request ──────────────────────────────────────────────────────

/// A request to materialize a specific archetype at a beat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArchetypeRequest {
    /// Unique request ID.
    pub id: ArchetypeRequestId,
    /// The archetype to materialize.
    pub archetype: ArchetypeIndex,
    /// The beat this request belongs to.
    pub beat_id: BeatId,
    /// The zone assigned to this request.
    pub zone_id: ZoneId,
    /// Exit degree bounds.
    pub degree: DegreeIntent,
    /// Whether this is a forced landmark selection (not from natural rarity).
    pub forced: bool,
    /// The rarity class at which this archetype was selected.
    pub rarity_class: RarityClass,
    /// Progression order on the critical path.
    pub progression: ProgressionOrder,
    /// Density class for this beat.
    pub density: DensityClass,
}

// ── Beat ───────────────────────────────────────────────────────────────────

/// An immutable pacing beat on the critical path or a side-branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Beat {
    /// Unique beat ID.
    pub id: BeatId,
    /// Semantic beat type.
    pub beat_type: BeatType,
    /// Archetype requests fulfilled by this beat.
    pub requests: Vec<ArchetypeRequestId>,
    /// Density class.
    pub density: DensityClass,
    /// Exit-degree intent.
    pub degree: DegreeIntent,
    /// Progression order (critical-path beats have strictly increasing ordinals).
    pub progression: ProgressionOrder,
    /// Whether this beat is on the critical path.
    pub on_critical_path: bool,
    /// Whether this beat is a grand-volume landmark (invariant flag).
    pub is_grand_volume: bool,
    /// Whether this beat is a quiet negative-space room (invariant flag).
    pub is_quiet_negative_space: bool,
    /// Whether this beat is a dense set-piece (invariant flag).
    pub is_dense_setpiece: bool,
}

// ── Mandatory edge ─────────────────────────────────────────────────────────

/// A mandatory edge on the critical path connecting two beats.
///
/// These edges must be realized by the topology solver and cannot be
/// reordered, replaced, or omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MandatoryEdge {
    /// The earlier beat in progression order.
    pub from_beat: BeatId,
    /// The later beat in progression order.
    pub to_beat: BeatId,
}

// ── Branch payoff ──────────────────────────────────────────────────────────

/// The observable payoff at the end of a side-branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BranchPayoff {
    /// Unique branch ID.
    pub branch_id: BranchId,
    /// The type of payoff.
    pub payoff_type: PayoffType,
    /// The beat at which the branch originates.
    pub from_beat: BeatId,
    /// The beat at which the branch terminates (if applicable).
    pub to_beat: Option<BeatId>,
    /// For shortcuts: the two progression-separated endpoints.
    /// `(later_beat, earlier_beat)` — shortcut connects later back to earlier.
    pub shortcut_endpoints: Option<(BeatId, BeatId)>,
    /// Whether this payoff is observable (not label-only).
    pub observable: bool,
}

impl BranchPayoff {
    /// Returns `true` if this payoff is a valid observable payoff.
    ///
    /// Label-only payoffs are rejected. Shortcuts must have two
    /// progression-separated endpoints that do not bypass any unvisited
    /// mandatory beat.
    pub fn is_valid(&self) -> bool {
        if !self.observable || self.to_beat.is_none() {
            return false;
        }
        if self.payoff_type == PayoffType::Shortcut {
            matches!(self.shortcut_endpoints, Some((later, earlier)) if later != earlier)
        } else {
            // DiscoveryLandmark, LoreMarker, AuthoredTreasureTableau are
            // observable by construction.
            true
        }
    }
}

// ── Shortcut intent ────────────────────────────────────────────────────────

/// A backward shortcut intent that connects a later progression point to an
/// earlier one without bypassing any unvisited mandatory beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ShortcutIntent {
    /// Unique shortcut ID.
    pub id: ShortcutId,
    /// The later beat in progression order (source of the shortcut).
    pub from_beat: BeatId,
    /// The earlier beat in progression order (destination of the shortcut).
    pub to_beat: BeatId,
}

impl ShortcutIntent {
    /// Returns `true` if this shortcut intent is valid.
    ///
    /// A valid shortcut must:
    /// - Connect a later beat to an earlier beat (from > to in progression).
    /// - Not bypass any unvisited mandatory beat. Since the shortcut connects
    ///   later→earlier, all beats between `to` and `from` were already visited
    ///   during forward progression, so a backward shortcut never bypasses
    ///   unvisited beats by construction.
    pub fn is_valid(
        &self,
        beats: &BTreeMap<BeatId, Beat>,
        _beat_order: &[BeatId],
        _mandatory_edges: &[MandatoryEdge],
    ) -> bool {
        let from_beat = match beats.get(&self.from_beat) {
            Some(b) => b,
            None => return false,
        };
        let to_beat = match beats.get(&self.to_beat) {
            Some(b) => b,
            None => return false,
        };

        // Must connect later to earlier
        if from_beat.progression <= to_beat.progression {
            return false;
        }

        // Both beats must be on the critical path
        if !from_beat.on_critical_path || !to_beat.on_critical_path {
            return false;
        }

        // A backward shortcut from later→earlier never bypasses unvisited
        // mandatory beats because the player has already traversed forward
        // through all intervening beats before reaching the shortcut source.
        true
    }
}

// ── Forced landmark ────────────────────────────────────────────────────────

/// A landmark that was forced into the blueprint, tracked separately from
/// natural rarity evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ForcedLandmark {
    /// The archetype selected for this forced landmark.
    pub archetype: ArchetypeIndex,
    /// Why this landmark was forced.
    pub reason: LandmarkForceReason,
    /// The beat this forced landmark occupies.
    pub beat_id: BeatId,
}

// ── Rarity evidence ────────────────────────────────────────────────────────

/// Natural rarity evidence stream — not distorted by forced landmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RarityEvidence {
    /// Count of common selections (natural only).
    pub common_count: u32,
    /// Count of uncommon selections (natural only).
    pub uncommon_count: u32,
    /// Count of rare selections (natural only).
    pub rare_count: u32,
    /// Count of legendary selections (natural only).
    pub legendary_count: u32,
    /// Total natural selections.
    pub total_natural: u32,
}

impl RarityEvidence {
    /// Create empty evidence.
    pub fn new() -> Self {
        Self {
            common_count: 0,
            uncommon_count: 0,
            rare_count: 0,
            legendary_count: 0,
            total_natural: 0,
        }
    }

    /// Record a natural selection of the given rarity class.
    pub fn record(&mut self, rarity: RarityClass) {
        match rarity {
            RarityClass::Common => self.common_count += 1,
            RarityClass::Uncommon => self.uncommon_count += 1,
            RarityClass::Rare => self.rare_count += 1,
            RarityClass::Legendary => self.legendary_count += 1,
        }
        self.total_natural += 1;
    }

    /// Verify the normalized 70:25:5:1 distribution using integer math only.
    /// `tolerance_pct` is measured in whole percentage points.
    pub fn verify_distribution(&self, tolerance_pct: u32) -> bool {
        if self.total_natural == 0 {
            return true;
        }

        let total = u64::from(self.total_natural);
        let tolerance = total * 101 * u64::from(tolerance_pct);
        [
            (self.common_count, RarityClass::Common.weight()),
            (self.uncommon_count, RarityClass::Uncommon.weight()),
            (self.rare_count, RarityClass::Rare.weight()),
            (self.legendary_count, RarityClass::Legendary.weight()),
        ]
        .into_iter()
        .all(|(count, weight)| {
            let observed = u64::from(count) * 101 * 100;
            let expected = total * u64::from(weight) * 100;
            observed.abs_diff(expected) <= tolerance
        })
    }
}

impl Default for RarityEvidence {
    fn default() -> Self {
        Self::new()
    }
}

// ── Zone record ────────────────────────────────────────────────────────────

/// Immutable zone record for the pacing blueprint.
///
/// Zone records select theme-scoped variants, material roles, roughness bands,
/// prop families, and lighting families WITHOUT selecting concrete theme assets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ZoneRecord {
    /// Unique zone ID.
    pub id: ZoneId,
    /// Semantic slots assigned to this zone (beat IDs).
    pub semantic_slots: Vec<BeatId>,
    /// Theme-scoped variant selection (index into theme variants, deferring
    /// asset resolution).
    pub theme_variant_index: u32,
    /// Indices into `MATERIAL_ROLE_NAMES` for this zone.
    pub material_role_indices: Vec<u32>,
    /// Roughness band index for this zone (0..N).
    pub roughness_band: u32,
    /// Prop family index for this zone (0..N).
    pub prop_family_index: u32,
    /// Lighting family index for this zone (0..N).
    pub light_family_index: u32,
}

// ── Zone transition ────────────────────────────────────────────────────────

/// An explicit transition record between two adjacent zones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ZoneTransition {
    /// The source zone.
    pub from_zone: ZoneId,
    /// The destination zone.
    pub to_zone: ZoneId,
    /// The beats that straddle this transition.
    pub straddle_beats: (BeatId, BeatId),
}

// ── Pacing blueprint ───────────────────────────────────────────────────────

/// The complete deterministic semantic pacing blueprint.
///
/// Constructed from a resolved request and seed. Theme-independent by
/// construction — all candidate keys exclude theme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PacingBlueprint {
    /// The preset this blueprint was built for.
    pub preset: RichnessPreset,
    /// The seed used.
    pub seed: u64,
    /// All beats in canonical order (BTreeMap by BeatId).
    pub beats: BTreeMap<BeatId, Beat>,
    /// Strict beat order on the critical path.
    pub beat_order: Vec<BeatId>,
    /// Complete zone assignment, transitions, reverse lookup, and downstream
    /// realization flags. This is the single canonical zone path.
    pub zone_blueprint: ZoneBlueprint,
    /// All archetype requests.
    pub archetype_requests: BTreeMap<ArchetypeRequestId, ArchetypeRequest>,
    /// Critical-path landmarks.
    pub critical_path_landmarks: Vec<LandmarkId>,
    /// Forced landmark selections (tracked separately from natural rarity).
    pub forced_landmarks: Vec<ForcedLandmark>,
    /// Natural rarity evidence (not distorted by forced selections).
    pub natural_rarity_evidence: RarityEvidence,
    /// Mandatory critical-path edges.
    pub mandatory_edges: Vec<MandatoryEdge>,
    /// Side-branch payoffs.
    pub branch_payoffs: BTreeMap<BranchId, BranchPayoff>,
    /// Backward shortcut intents (Moderate and Rich only).
    pub shortcut_intents: Vec<ShortcutIntent>,
    /// Invariant: at least one grand-volume landmark present.
    pub grand_volume_landmark_present: bool,
    /// Invariant: at least one quiet negative-space room present.
    pub quiet_negative_space_present: bool,
    /// Invariant: at least one dense set-piece present.
    pub dense_setpiece_present: bool,
}

impl PacingBlueprint {
    /// Validate that the blueprint satisfies all invariants.
    ///
    /// Returns a list of validation error strings. An empty list means PASS.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // 1. Landmark count matches preset
        let expected_landmarks = match self.preset {
            RichnessPreset::Sparse => 1,
            RichnessPreset::Moderate => 2,
            RichnessPreset::Rich => 3,
        };
        if self.critical_path_landmarks.len() != expected_landmarks {
            errors.push(format!(
                "landmark count {} != expected {} for preset {:?}",
                self.critical_path_landmarks.len(),
                expected_landmarks,
                self.preset
            ));
        }

        // 2. All critical-path landmark IDs are distinct
        let mut seen = std::collections::BTreeSet::new();
        for lid in &self.critical_path_landmarks {
            if !seen.insert(*lid) {
                errors.push(format!("duplicate landmark ID {:?}", lid));
            }
        }

        // 3. Beat reachability and canonical order: every critical beat must
        // exist exactly once and every mandatory edge must connect adjacent
        // ordered beats.
        let mut ordered = std::collections::BTreeSet::new();
        for bid in &self.beat_order {
            if !self.beats.contains_key(bid) {
                errors.push(format!("beat {:?} in beat_order not in beats map", bid));
            }
            if !ordered.insert(*bid) {
                errors.push(format!("duplicate beat {:?} in beat_order", bid));
            }
        }
        let expected_edges: Vec<_> = self
            .beat_order
            .windows(2)
            .map(|pair| MandatoryEdge {
                from_beat: pair[0],
                to_beat: pair[1],
            })
            .collect();
        if self.mandatory_edges != expected_edges {
            errors.push("mandatory edges do not canonically cover beat order".to_string());
        }

        // 4. Beat order is strict (progression ordinals strictly increasing)
        let mut prev_prog: Option<ProgressionOrder> = None;
        for bid in &self.beat_order {
            if let Some(beat) = self.beats.get(bid) {
                if let Some(prev) = prev_prog {
                    if beat.progression <= prev {
                        errors.push(format!(
                            "non-strict progression at beat {:?}: {:?} <= {:?}",
                            bid, beat.progression, prev
                        ));
                    }
                }
                prev_prog = Some(beat.progression);
            }
        }

        // 5. Dense/quiet alternation on critical path
        let mut last_density: Option<DensityClass> = None;
        for bid in &self.beat_order {
            if let Some(beat) = self.beats.get(bid) {
                if beat.on_critical_path {
                    if let Some(last) = last_density {
                        if last == beat.density && beat.density != DensityClass::Transition {
                            errors.push(format!(
                                "dense/quiet non-alternation at beat {:?}: {:?} follows {:?}",
                                bid, beat.density, last
                            ));
                        }
                    }
                    last_density = Some(beat.density);
                }
            }
        }

        // 6. Every requested critical beat has a materializable archetype
        // request, and every branch has one observable payoff record.
        for bid in &self.beat_order {
            if self
                .beats
                .get(bid)
                .is_some_and(|beat| beat.requests.is_empty())
            {
                errors.push(format!("critical beat {:?} has no archetype request", bid));
            }
        }
        for (bid, payoff) in &self.branch_payoffs {
            if !payoff.is_valid() {
                errors.push(format!("invalid branch payoff {:?}", bid));
                continue;
            }
            if payoff
                .to_beat
                .is_some_and(|leaf| !self.beats.contains_key(&leaf))
            {
                errors.push(format!("branch payoff {:?} targets an unknown leaf", bid));
            }
            if let Some((later, earlier)) = payoff.shortcut_endpoints {
                let valid_endpoints = self
                    .beats
                    .get(&later)
                    .zip(self.beats.get(&earlier))
                    .is_some_and(|(later, earlier)| {
                        later.on_critical_path
                            && earlier.on_critical_path
                            && later.progression > earlier.progression
                    });
                if !valid_endpoints {
                    errors.push(format!("invalid branch shortcut payoff {:?}", bid));
                }
            }
        }

        // 7. Zone assignment is complete, canonical, and carries the
        // downstream-realizable invariant flags.
        errors.extend(self.zone_blueprint.validate());
        for bid in self.beats.keys() {
            if !self.zone_blueprint.beat_zone_map.contains_key(bid) {
                errors.push(format!("semantic beat {:?} has no zone assignment", bid));
            }
        }

        // 8. Degree intents are bounded and valid.
        for beat in self.beats.values() {
            if !beat.degree.is_valid() || beat.degree.max_exits > 5 {
                errors.push(format!("invalid degree intent at beat {:?}", beat.id));
            }
        }
        for request in self.archetype_requests.values() {
            if !request.degree.is_valid() || request.degree.max_exits > 5 {
                errors.push(format!("invalid degree intent at request {:?}", request.id));
            }
        }

        // 9. Shortcut validation for Moderate/Rich
        for shortcut in &self.shortcut_intents {
            if !shortcut.is_valid(&self.beats, &self.beat_order, &self.mandatory_edges) {
                errors.push(format!(
                    "invalid shortcut intent {:?}: from {:?} to {:?}",
                    shortcut.id, shortcut.from_beat, shortcut.to_beat
                ));
            }
        }

        if matches!(self.preset, RichnessPreset::Moderate | RichnessPreset::Rich)
            && self.shortcut_intents.is_empty()
        {
            errors.push("moderate/rich blueprint has no backward shortcut intent".to_string());
        }

        // 10. Grand-volume landmark present
        if !self.grand_volume_landmark_present {
            errors.push("no grand-volume landmark present".to_string());
        }

        // 11. Quiet negative-space room present
        if !self.quiet_negative_space_present {
            errors.push("no quiet negative-space room present".to_string());
        }

        // 12. Dense set-piece present
        if !self.dense_setpiece_present {
            errors.push("no dense set-piece present".to_string());
        }

        // 13. Rarity caps: rare and legendary no-repeat + one-per-map
        let mut rare_count = 0u32;
        let mut legendary_count = 0u32;
        let mut seen_rare: std::collections::BTreeSet<ArchetypeIndex> =
            std::collections::BTreeSet::new();
        let mut seen_legendary: std::collections::BTreeSet<ArchetypeIndex> =
            std::collections::BTreeSet::new();

        for req in self.archetype_requests.values() {
            if req.forced {
                continue; // forced landmarks don't count toward rarity caps
            }
            match req.rarity_class {
                RarityClass::Rare => {
                    rare_count += 1;
                    if !seen_rare.insert(req.archetype) {
                        errors.push(format!(
                            "rare archetype {:?} repeated in request {:?}",
                            req.archetype, req.id
                        ));
                    }
                }
                RarityClass::Legendary => {
                    legendary_count += 1;
                    if !seen_legendary.insert(req.archetype) {
                        errors.push(format!(
                            "legendary archetype {:?} repeated in request {:?}",
                            req.archetype, req.id
                        ));
                    }
                }
                _ => {}
            }
        }

        if rare_count > 1 {
            errors.push(format!(
                "rare cap exceeded: {} rare archetypes (max 1)",
                rare_count
            ));
        }
        if legendary_count > 1 {
            errors.push(format!(
                "legendary cap exceeded: {} legendary archetypes (max 1)",
                legendary_count
            ));
        }

        errors
    }

    /// Returns `true` if validation passes (no errors).
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_type_tags_are_unique() {
        let tags = [
            BeatType::Entrance.tag(),
            BeatType::Descent.tag(),
            BeatType::LandmarkPeak.tag(),
            BeatType::Release.tag(),
        ];
        let set: std::collections::BTreeSet<_> = tags.iter().collect();
        assert_eq!(set.len(), tags.len());
    }

    #[test]
    fn density_class_tags_are_unique() {
        let tags = [
            DensityClass::Dense.tag(),
            DensityClass::Quiet.tag(),
            DensityClass::Transition.tag(),
        ];
        let set: std::collections::BTreeSet<_> = tags.iter().collect();
        assert_eq!(set.len(), tags.len());
    }

    #[test]
    fn rarity_class_weights_sum_to_101() {
        let total = RarityClass::Common.weight()
            + RarityClass::Uncommon.weight()
            + RarityClass::Rare.weight()
            + RarityClass::Legendary.weight();
        assert_eq!(total, 101);
    }

    #[test]
    fn rarity_class_no_repeat_only_rare_and_legendary() {
        assert!(!RarityClass::Common.no_repeat());
        assert!(!RarityClass::Uncommon.no_repeat());
        assert!(RarityClass::Rare.no_repeat());
        assert!(RarityClass::Legendary.no_repeat());
    }

    #[test]
    fn rarity_class_one_per_map_only_rare_and_legendary() {
        assert!(!RarityClass::Common.one_per_map());
        assert!(!RarityClass::Uncommon.one_per_map());
        assert!(RarityClass::Rare.one_per_map());
        assert!(RarityClass::Legendary.one_per_map());
    }

    #[test]
    fn rarity_class_from_tier_converts_all() {
        assert_eq!(
            RarityClass::from_tier(RarityTier::Common),
            RarityClass::Common
        );
        assert_eq!(
            RarityClass::from_tier(RarityTier::Uncommon),
            RarityClass::Uncommon
        );
        assert_eq!(RarityClass::from_tier(RarityTier::Rare), RarityClass::Rare);
        assert_eq!(
            RarityClass::from_tier(RarityTier::Legendary),
            RarityClass::Legendary
        );
    }

    #[test]
    fn typed_ids_are_distinct_types() {
        let b = BeatId::new(1);
        let z = ZoneId::new(1);
        let a = ArchetypeRequestId::new(1);
        let br = BranchId::new(1);
        let l = LandmarkId::new(1);
        let s = ShortcutId::new(1);

        // All raw values are 1 but they're different types
        assert_eq!(b.raw(), 1);
        assert_eq!(z.raw(), 1);
        assert_eq!(a.raw(), 1);
        assert_eq!(br.raw(), 1);
        assert_eq!(l.raw(), 1);
        assert_eq!(s.raw(), 1);
    }

    #[test]
    fn archetype_index_valid_range() {
        for idx in ArchetypeIndex::all() {
            assert!(idx.raw() < 30);
            assert!(!idx.id_str().is_empty());
        }
    }

    #[test]
    fn archetype_index_from_id_str_roundtrips() {
        for idx in ArchetypeIndex::all() {
            let s = idx.id_str();
            let back = ArchetypeIndex::from_id_str(s).unwrap();
            assert_eq!(idx, back);
        }
    }

    #[test]
    fn archetype_index_unknown_id_returns_none() {
        assert!(ArchetypeIndex::from_id_str("nonexistent").is_none());
        assert!(ArchetypeIndex::from_id_str("").is_none());
    }

    #[test]
    fn degree_intent_valid() {
        let d = DegreeIntent::new(1, 3);
        assert!(d.is_valid());
        assert_eq!(d.min_exits, 1);
        assert_eq!(d.max_exits, 3);
    }

    #[test]
    fn degree_intent_min_equals_max() {
        let d = DegreeIntent::new(2, 2);
        assert!(d.is_valid());
    }

    #[test]
    fn progression_order_strict() {
        let p0 = ProgressionOrder::FIRST;
        let p1 = p0.next();
        let p2 = p1.next();
        assert_eq!(p0.raw(), 0);
        assert_eq!(p1.raw(), 1);
        assert_eq!(p2.raw(), 2);
        assert!(p0 < p1);
        assert!(p1 < p2);
    }

    #[test]
    fn payoff_type_no_label_only() {
        assert!(!PayoffType::Shortcut.is_label_only());
        assert!(!PayoffType::DiscoveryLandmark.is_label_only());
        assert!(!PayoffType::LoreMarker.is_label_only());
        assert!(!PayoffType::AuthoredTreasureTableau.is_label_only());
    }

    #[test]
    fn branch_payoff_valid_shortcut_requires_endpoints() {
        let valid = BranchPayoff {
            branch_id: BranchId::new(0),
            payoff_type: PayoffType::Shortcut,
            from_beat: BeatId::new(0),
            to_beat: Some(BeatId::new(1)),
            shortcut_endpoints: Some((BeatId::new(2), BeatId::new(0))),
            observable: true,
        };
        assert!(valid.is_valid());

        let invalid = BranchPayoff {
            branch_id: BranchId::new(0),
            payoff_type: PayoffType::Shortcut,
            from_beat: BeatId::new(0),
            to_beat: Some(BeatId::new(1)),
            shortcut_endpoints: None,
            observable: true,
        };
        assert!(!invalid.is_valid());

        let label_only = BranchPayoff {
            branch_id: BranchId::new(0),
            payoff_type: PayoffType::LoreMarker,
            from_beat: BeatId::new(0),
            to_beat: None,
            shortcut_endpoints: None,
            observable: false,
        };
        assert!(!label_only.is_valid());
    }

    #[test]
    fn every_observable_payoff_variant_is_accepted() {
        for payoff_type in [
            PayoffType::Shortcut,
            PayoffType::DiscoveryLandmark,
            PayoffType::LoreMarker,
            PayoffType::AuthoredTreasureTableau,
        ] {
            let payoff = BranchPayoff {
                branch_id: BranchId::new(0),
                payoff_type,
                from_beat: BeatId::new(0),
                to_beat: Some(BeatId::new(1)),
                shortcut_endpoints: (payoff_type == PayoffType::Shortcut)
                    .then_some((BeatId::new(2), BeatId::new(0))),
                observable: true,
            };
            assert!(payoff.is_valid(), "{} payoff rejected", payoff_type.tag());
        }
    }

    #[test]
    fn rarity_evidence_records_correctly() {
        let mut ev = RarityEvidence::new();
        ev.record(RarityClass::Common);
        ev.record(RarityClass::Common);
        ev.record(RarityClass::Uncommon);
        assert_eq!(ev.common_count, 2);
        assert_eq!(ev.uncommon_count, 1);
        assert_eq!(ev.rare_count, 0);
        assert_eq!(ev.legendary_count, 0);
        assert_eq!(ev.total_natural, 3);
    }

    #[test]
    fn pacing_blueprint_validate_empty_errors_on_landmarks() {
        let bp = PacingBlueprint {
            preset: RichnessPreset::Sparse,
            seed: 0,
            beats: BTreeMap::new(),
            beat_order: Vec::new(),
            zone_blueprint: ZoneBlueprint {
                zones: BTreeMap::new(),
                transitions: Vec::new(),
                invariants: BTreeMap::new(),
                beat_zone_map: BTreeMap::new(),
            },
            archetype_requests: BTreeMap::new(),
            critical_path_landmarks: Vec::new(),
            forced_landmarks: Vec::new(),
            natural_rarity_evidence: RarityEvidence::new(),
            mandatory_edges: Vec::new(),
            branch_payoffs: BTreeMap::new(),
            shortcut_intents: Vec::new(),
            grand_volume_landmark_present: false,
            quiet_negative_space_present: false,
            dense_setpiece_present: false,
        };
        let errors = bp.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("landmark count")));
    }

    #[test]
    fn pacing_blueprint_validate_missing_invariants() {
        let bp = PacingBlueprint {
            preset: RichnessPreset::Sparse,
            seed: 0,
            beats: BTreeMap::new(),
            beat_order: Vec::new(),
            zone_blueprint: ZoneBlueprint {
                zones: BTreeMap::new(),
                transitions: Vec::new(),
                invariants: BTreeMap::new(),
                beat_zone_map: BTreeMap::new(),
            },
            archetype_requests: BTreeMap::new(),
            critical_path_landmarks: Vec::new(),
            forced_landmarks: Vec::new(),
            natural_rarity_evidence: RarityEvidence::new(),
            mandatory_edges: Vec::new(),
            branch_payoffs: BTreeMap::new(),
            shortcut_intents: Vec::new(),
            grand_volume_landmark_present: false,
            quiet_negative_space_present: false,
            dense_setpiece_present: false,
        };
        let errors = bp.validate();
        assert!(errors.iter().any(|e| e.contains("no grand-volume")));
        assert!(errors.iter().any(|e| e.contains("no quiet negative-space")));
        assert!(errors.iter().any(|e| e.contains("no dense set-piece")));
    }
}
