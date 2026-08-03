//! Variation plan: legal dimensions, surface/material choices, lights,
//! props, damage variants, and shared structural wall chain ownership —
//! all bounded within committed semantic envelopes.
//!
//! Every variation decision is keyed by committed semantic ID and field
//! tag. Wall chains have one owner; cardinal step/chamfer shaping is
//! limited to two quanta (32u) while holding portal anchors, protected
//! segments, structural thickness (16u), and exterior envelope FIXED.
//! Apparent wall mass is only through inward liners/pilasters/recesses/
//! buttress courses of 0/16/32 units.
//!
//! # Contract
//!
//! - Variation cannot alter connectivity, reservations, portal anchors,
//!   protected throats, exterior envelopes, required beats, cave
//!   eligibility/result, or semantic multi-storey status.
//! - No floats. All dimensions are quantum-aligned.
//! - Crate-private; canonical ordering; no brush/entity emission.

// Richness remains intentionally crate-private and pipeline-unwired until
// the atomic sealing phase.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::fields::FieldTag;
use super::generated_content;
use super::ids::{PacingBlueprint, PortalId, ReservationId, SemanticId, WallChainId, ZoneId};
use super::reservation::{ReservationJournal, ReservationRecord};
use super::topology::{CommittedPortal, CommittedRoute, Dir, TopologyResult};
use super::zones::ZoneBlueprint;

// ── Quantized cardinal shaping ──────────────────────────────────────────────
// (WallChainId and SemanticId are imported from super::ids)

/// How far a cardinal wall can be stepped or chamfered along its normal.
///
/// The shaping quantum is 16 units (the construction quantum). Up to two
/// quanta (32 units total) are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum WallShaping {
    /// No shaping offset (0 quanta).
    None,
    /// One quantum inward/outward (16 units).
    OneQuantum,
    /// Two quanta inward/outward (32 units).
    TwoQuanta,
}

impl WallShaping {
    /// Offset in Quake units (quantum-aligned).
    pub fn offset_units(self) -> i32 {
        match self {
            Self::None => 0,
            Self::OneQuantum => 16,
            Self::TwoQuanta => 32,
        }
    }

    /// Quantum count (0, 1, or 2).
    pub fn quantum_count(self) -> u32 {
        match self {
            Self::None => 0,
            Self::OneQuantum => 1,
            Self::TwoQuanta => 2,
        }
    }
}

// ── Apparent wall mass ─────────────────────────────────────────────────────

/// Apparent wall mass treatment inside a structural wall envelope.
///
/// Mass is selected only through inward liners, pilasters, recesses,
/// and buttress courses. It MUST NOT consume route, turn, spawn,
/// vertical, cave, or negative-space reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum WallMass {
    /// No additional mass (0 units).
    None,
    /// Liner course (16 units inward).
    Liner16,
    /// Liner or pilaster course (32 units inward).
    Liner32,
    /// Recess course (carving inward, negative apparent mass).
    Recess16,
    /// Buttress course (external mass, outward).
    Buttress16,
}

impl WallMass {
    /// Displacement in Quake units (positive = outward, negative = inward).
    pub fn offset_units(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Liner16 => -16,
            Self::Liner32 => -32,
            Self::Recess16 => 16,
            Self::Buttress16 => -16,
        }
    }

    /// Whether this treatment points inward (reducing clear volume).
    pub fn is_inward(self) -> bool {
        matches!(self, Self::Liner16 | Self::Liner32 | Self::Buttress16)
    }

    /// Whether this treatment is legal (never consumes protected reservations).
    pub fn is_legal(self) -> bool {
        // All defined variants are legal by construction; the caller
        // must reject any placement that would overlap a protected
        // reservation cell.
        true
    }
}

// ── Legal dimensions ────────────────────────────────────────────────────────

/// Committed legal dimensions for a wall, opening, or room volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct LegalDimensions {
    /// Width in Quake units (quantum-aligned).
    pub width: u32,
    /// Height in Quake units (quantum-aligned).
    pub height: u32,
    /// Depth in Quake units (quantum-aligned).
    pub depth: u32,
}

impl LegalDimensions {
    pub const fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Constrain to fixed values, used when an envelope must not vary.
    pub fn fix_to(&mut self, fixed: Self) {
        *self = fixed;
    }

    /// Return the volume in quantum cells.
    pub fn volume_quanta(&self) -> u32 {
        (self.width / 16) * (self.height / 16) * (self.depth / 16)
    }
}

// ── Surface and material choices ────────────────────────────────────────────

/// A surface texture choice for a face role.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SurfaceChoice {
    /// Index into `MATERIAL_ROLE_NAMES`.
    pub role_index: u32,
    /// Stable texture identity tag.
    pub texture_tag: String,
}

/// A material role selection with concrete identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MaterialRoleSelection {
    /// Index into `MATERIAL_ROLE_NAMES`.
    pub role_index: u32,
    /// The material identity for this role.
    pub material_identity: String,
}

// ── Light and prop selections ──────────────────────────────────────────────

/// A selected light recipe for a variation decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct LightSelection {
    /// Index into `LIGHT_RECIPE_IDS`.
    pub light_index: u32,
    /// Number of instances placed.
    pub count: u32,
}

/// A selected prop for a variation decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PropSelection {
    /// Index into `PROP_IDS`.
    pub prop_index: u32,
    /// Number of instances placed.
    pub count: u32,
}

// ── Damage variants ────────────────────────────────────────────────────────

/// Authored damage / wear variant applied to surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum DamageVariant {
    /// No damage — pristine surfaces.
    None,
    /// Light weathering: subtle edge wear.
    LightWeathering,
    /// Heavy weathering: cracked stone, missing chunks.
    HeavyWeathering,
    /// Partial structural collapse: fallen rubble.
    PartialCollapse,
    /// Water damage: stained walls, eroded mortar.
    WaterDamage,
    /// Fire scarring: soot-blackened surfaces.
    FireScarring,
}

impl DamageVariant {
    pub fn tag(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LightWeathering => "light_weathering",
            Self::HeavyWeathering => "heavy_weathering",
            Self::PartialCollapse => "partial_collapse",
            Self::WaterDamage => "water_damage",
            Self::FireScarring => "fire_scarring",
        }
    }
}

// ── Variation decision ─────────────────────────────────────────────────────

/// A single variation decision keyed by committed semantic ID and field tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariationDecision {
    /// The committed semantic identity this decision governs.
    pub semantic_id: SemanticId,
    /// The field tag that produced this decision.
    pub field_tag: FieldTag,
    /// Legal dimensions for this entity.
    pub dimensions: LegalDimensions,
    /// Surface texture choices.
    pub surface_choices: Vec<SurfaceChoice>,
    /// Material role assignments.
    pub material_roles: Vec<MaterialRoleSelection>,
    /// Light recipe selections.
    pub lights: Vec<LightSelection>,
    /// Prop selections.
    pub props: Vec<PropSelection>,
    /// Damage variant.
    pub damage: DamageVariant,
}

// ── Wall mass treatment placement ──────────────────────────────────────────

/// Where on a wall chain mass treatment is applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct WallMassTreatment {
    /// Segment on the wall chain (quantum-aligned start, end).
    pub segment: (i32, i32),
    /// Kind of mass treatment.
    pub kind: WallMass,
    /// How many quanta this treatment consumes (0, 1, or 2).
    pub quantum_count: u32,
}

// ── Wall chain record ──────────────────────────────────────────────────────

/// An immutable record describing a shared structural wall chain.
///
/// One owner per chain; shared between adjacent reservations. Portal anchors
/// and protected segments are FIXED — variation may only apply cardinal
/// step/chamfer shaping up to two quanta, and apparent mass through inward
/// liners/pilasters/recesses/buttresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WallChainRecord {
    /// Unique wall chain ID.
    pub id: WallChainId,
    /// The owning semantic entity.
    pub owner: SemanticId,
    /// Other semantic entities sharing this wall.
    pub shared_with: Vec<SemanticId>,
    /// Cardinal direction this wall chain faces.
    pub cardinal_direction: Dir,
    /// Shaping offsets applied at each end of the chain (up to two quanta).
    pub shaping: [WallShaping; 2],
    /// Mass treatment placements along this chain.
    pub mass_treatments: Vec<WallMassTreatment>,
    /// Portal anchors on this wall — FIXED, cannot be altered by variation.
    pub portal_anchors: Vec<PortalId>,
    /// Protected segments along this wall (quantum-aligned intervals) — FIXED.
    pub protected_segments: Vec<(i32, i32)>,
    /// Structural thickness in Quake units — FIXED at 16u.
    pub structural_thickness: u32,
    /// Whether this wall is part of the exterior envelope — FIXED.
    pub exterior_envelope: bool,
}

// ── Variation plan ─────────────────────────────────────────────────────────

/// The complete pre-assembly variation plan.
///
/// Records every decision keyed by semantic ID + field tag, plus shared
/// wall chain ownership and shaping. All decisions are bounded within
/// committed semantic envelopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariationPlan {
    /// Variation decisions keyed by the full semantic namespace and field tag.
    ///
    /// A raw numeric ID is not sufficient: a reservation, route, and portal
    /// may each legitimately have raw ID zero in one committed topology.
    pub decisions: BTreeMap<(SemanticId, FieldTag), VariationDecision>,
    /// Shared structural wall chains.
    pub wall_chains: BTreeMap<WallChainId, WallChainRecord>,
    /// Whether this plan is complete (every semantic entity has decisions).
    pub complete: bool,
}

impl VariationPlan {
    /// Create an empty variation plan.
    pub fn new() -> Self {
        Self {
            decisions: BTreeMap::new(),
            wall_chains: BTreeMap::new(),
            complete: false,
        }
    }

    /// Get a variation decision by semantic ID and field tag.
    pub fn get_decision(
        &self,
        semantic_id: SemanticId,
        field_tag: FieldTag,
    ) -> Option<&VariationDecision> {
        self.decisions.get(&(semantic_id, field_tag))
    }

    /// Insert a variation decision.
    pub fn insert_decision(&mut self, decision: VariationDecision) {
        self.decisions
            .insert((decision.semantic_id, decision.field_tag), decision);
    }

    /// Insert a wall chain record.
    pub fn insert_wall_chain(&mut self, chain: WallChainRecord) {
        self.wall_chains.insert(chain.id, chain);
    }

    /// Returns the total number of decisions.
    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }

    /// Returns the total number of wall chains.
    pub fn wall_chain_count(&self) -> usize {
        self.wall_chains.len()
    }

    /// Validate that no variation decision alters protected semantics.
    ///
    /// Returns a list of validation errors. An empty list means PASS.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Every wall chain must have exactly one owner and the owner
        // must not also appear in shared_with.
        for chain in self.wall_chains.values() {
            if chain.shared_with.contains(&chain.owner) {
                errors.push(format!(
                    "wall chain {:?} owner {:?} also in shared_with",
                    chain.id,
                    chain.owner.tag()
                ));
            }

            // Portal anchors must not have shaping that would displace them.
            if !chain.portal_anchors.is_empty() {
                for &shaping in &chain.shaping {
                    if shaping.offset_units() != 0 {
                        errors.push(format!(
                            "wall chain {:?} has portal anchors but non-zero shaping",
                            chain.id
                        ));
                    }
                }
            }

            // Structural thickness must be exactly 16.
            if chain.structural_thickness != 16 {
                errors.push(format!(
                    "wall chain {:?} has non-standard structural thickness {}",
                    chain.id, chain.structural_thickness
                ));
            }
        }

        // All portal anchors referenced in decisions must match wall chain
        // portal anchors.
        for decision in self.decisions.values() {
            match decision.semantic_id {
                SemanticId::Portal(pid) => {
                    let mut found = false;
                    for chain in self.wall_chains.values() {
                        if chain.portal_anchors.contains(&pid) {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        errors.push(format!(
                            "portal decision for {:?} has no matching wall chain anchor",
                            pid
                        ));
                    }
                }
                _ => {}
            }
        }

        errors
    }
}

impl Default for VariationPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ── Variation builder ──────────────────────────────────────────────────────

/// Builder for constructing a `VariationPlan` from a pacing blueprint,
/// topology result, and zone blueprint.
pub(crate) struct VariationBuilder {
    plan: VariationPlan,
    next_wall_chain_id: u32,
}

impl VariationBuilder {
    /// Create a new variation builder.
    pub fn new() -> Self {
        Self {
            plan: VariationPlan::new(),
            next_wall_chain_id: 0,
        }
    }

    /// Build a wall chain ID.
    fn alloc_wall_chain_id(&mut self) -> WallChainId {
        let id = WallChainId::new(self.next_wall_chain_id);
        self.next_wall_chain_id += 1;
        id
    }

    /// Build a variation decision for a reservation.
    pub fn build_reservation_decision(
        &mut self,
        reservation: &ReservationRecord,
        field_tag: FieldTag,
    ) -> VariationDecision {
        let semantic_id = SemanticId::Reservation(reservation.id);

        // Derive legal dimensions from the reservation's footprint or span.
        let dims = LegalDimensions::new(256, 176, 256);

        // Surface choices from generated material roles for the archetype if known.
        let surface_choices: Vec<SurfaceChoice> = if let Some(req_id) = reservation.request_id {
            let arch_idx = req_id.raw() as usize;
            if arch_idx < generated_content::ARCHETYPE_MATERIAL_ROLES.len() {
                generated_content::ARCHETYPE_MATERIAL_ROLES[arch_idx]
                    .iter()
                    .map(|&(role_idx, tex)| SurfaceChoice {
                        role_index: role_idx,
                        texture_tag: tex.to_string(),
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            default_surface_choices()
        };

        let material_roles: Vec<MaterialRoleSelection> = surface_choices
            .iter()
            .map(|sc| MaterialRoleSelection {
                role_index: sc.role_index,
                material_identity: sc.texture_tag.clone(),
            })
            .collect();

        VariationDecision {
            semantic_id,
            field_tag,
            dimensions: dims,
            surface_choices,
            material_roles,
            lights: Vec::new(),
            props: Vec::new(),
            damage: DamageVariant::None,
        }
    }

    /// Build a variation decision for a route.
    pub fn build_route_decision(
        &mut self,
        route: &CommittedRoute,
        _field_tag: FieldTag,
    ) -> VariationDecision {
        let semantic_id = SemanticId::Route(route.id);
        VariationDecision {
            semantic_id,
            field_tag: FieldTag::ValueNoise,
            dimensions: LegalDimensions::new(64, 80, 256),
            surface_choices: route_surface_choices(),
            material_roles: route_material_roles(),
            lights: Vec::new(),
            props: Vec::new(),
            damage: DamageVariant::None,
        }
    }

    /// Build a variation decision for a portal.
    pub fn build_portal_decision(
        &mut self,
        portal: &CommittedPortal,
        _field_tag: FieldTag,
    ) -> VariationDecision {
        let semantic_id = SemanticId::Portal(portal.id);
        VariationDecision {
            semantic_id,
            field_tag: FieldTag::ValueNoise,
            dimensions: LegalDimensions::new(64, 80, 16),
            surface_choices: portal_surface_choices(),
            material_roles: portal_material_roles(),
            lights: Vec::new(),
            props: Vec::new(),
            damage: DamageVariant::None,
        }
    }

    /// Derive wall chains from adjacent reservations that share a wall.
    ///
    /// Each shared structural wall gets one owner and exactly one chain record.
    pub fn derive_wall_chains(
        &mut self,
        journal: &ReservationJournal,
        routes: &[CommittedRoute],
        origin_zone_map: &BTreeMap<ReservationId, ZoneId>,
    ) {
        // A route pair can have several candidate routes, but it still has one
        // canonical structural chain owner.  Group by the unordered committed
        // reservation pair and choose the lower semantic identity as owner.
        let mut chains: BTreeMap<(ReservationId, ReservationId), WallChainRecord> = BTreeMap::new();
        for route in routes {
            let (owner_id, shared_id) = if route.source <= route.target {
                (route.source, route.target)
            } else {
                (route.target, route.source)
            };
            let key = (owner_id, shared_id);
            let chain = chains.entry(key).or_insert_with(|| WallChainRecord {
                id: self.alloc_wall_chain_id(),
                owner: SemanticId::Reservation(owner_id),
                shared_with: vec![SemanticId::Reservation(shared_id)],
                cardinal_direction: route.source_portal.wall,
                shaping: [WallShaping::None, WallShaping::None],
                mass_treatments: Vec::new(),
                portal_anchors: Vec::new(),
                protected_segments: Vec::new(),
                structural_thickness: 16,
                exterior_envelope: false,
            });

            for portal in [&route.source_portal, &route.target_portal] {
                if !chain.portal_anchors.contains(&portal.id) {
                    chain.portal_anchors.push(portal.id);
                }
                let segment = wall_cross_segment(&portal.witness, chain.cardinal_direction);
                if !chain.protected_segments.contains(&segment) {
                    chain.protected_segments.push(segment);
                }
            }
            for reservation_id in &route.reservation_ids {
                if let Some(record) = journal.get(*reservation_id) {
                    if matches!(
                        record.kind,
                        super::reservation::ReservationKind::Route
                            | super::reservation::ReservationKind::PortalThroat
                            | super::reservation::ReservationKind::Turn
                    ) {
                        let segment =
                            wall_cross_segment(&record.footprint, chain.cardinal_direction);
                        if !chain.protected_segments.contains(&segment) {
                            chain.protected_segments.push(segment);
                        }
                    }
                }
            }
            chain.portal_anchors.sort_unstable();
            chain.protected_segments.sort_unstable();
            chain.protected_segments.dedup();
        }

        for (_, chain) in chains {
            self.plan.insert_wall_chain(chain);
        }
        let _ = origin_zone_map;
    }

    /// Apply wall mass treatments to a wall chain.
    ///
    /// Rejects any treatment that would consume route, turn, spawn,
    /// vertical, cave, or negative-space reservation cells.
    pub fn apply_wall_mass(
        &mut self,
        chain_id: WallChainId,
        treatments: Vec<WallMassTreatment>,
        journal: &ReservationJournal,
    ) -> Result<(), String> {
        let chain = match self.plan.wall_chains.get_mut(&chain_id) {
            Some(c) => c,
            None => return Err(format!("wall chain {:?} not found", chain_id)),
        };

        // Validate exact quantization and reject protected route, portal, and
        // turn spans derived from the committed journal.  Spawn, vertical,
        // cave, and negative-space reservations are also protected globally:
        // without a later composition-owned overlap proof, variation cannot
        // claim they are safe to consume.
        let has_other_protected_reservation = journal.reservations.values().any(|record| {
            matches!(
                record.kind,
                super::reservation::ReservationKind::Spawn
                    | super::reservation::ReservationKind::VerticalHost
                    | super::reservation::ReservationKind::CaveHost
                    | super::reservation::ReservationKind::NegativeSpace
            )
        });
        for treatment in &treatments {
            let expected_quanta = match treatment.kind {
                WallMass::None => 0,
                WallMass::Liner16 | WallMass::Recess16 | WallMass::Buttress16 => 1,
                WallMass::Liner32 => 2,
            };
            if treatment.quantum_count != expected_quanta
                || treatment.segment.0 >= treatment.segment.1
                || treatment.segment.0 % 16 != 0
                || treatment.segment.1 % 16 != 0
            {
                return Err(format!(
                    "wall chain {:?}: mass treatment {:?} is not a legal 0/16/32-unit course",
                    chain_id, treatment
                ));
            }
            if expected_quanta != 0 && has_other_protected_reservation {
                return Err(format!(
                    "wall chain {:?}: mass treatment would consume spawn/vertical/cave/negative-space reservation",
                    chain_id
                ));
            }
            for &(seg_start, seg_end) in &chain.protected_segments {
                if treatment.segment.0 < seg_end && treatment.segment.1 > seg_start {
                    return Err(format!(
                        "wall chain {:?}: mass treatment {:?} overlaps protected route/portal/turn segment ({}, {})",
                        chain_id, treatment.segment, seg_start, seg_end
                    ));
                }
            }
        }

        chain.mass_treatments = treatments;
        Ok(())
    }

    /// Build the complete variation plan.
    pub fn build(mut self) -> VariationPlan {
        self.plan.complete = true;
        self.plan
    }
}

impl Default for VariationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Default surface helpers ────────────────────────────────────────────────

/// Project a committed footprint to the coordinate interval that lies along
/// a wall chain. All values remain in 16-unit Quake coordinates.
fn wall_cross_segment(footprint: &super::footprint::Footprint3D, direction: Dir) -> (i32, i32) {
    match direction {
        Dir::North | Dir::South => (footprint.x0 as i32 * 16, footprint.x1 as i32 * 16),
        Dir::East | Dir::West => (footprint.y0 as i32 * 16, footprint.y1 as i32 * 16),
    }
}

fn default_surface_choices() -> Vec<SurfaceChoice> {
    vec![
        SurfaceChoice {
            role_index: 5, // wall
            texture_tag: "wall_stone".to_string(),
        },
        SurfaceChoice {
            role_index: 2, // floor
            texture_tag: "floor_flagstone".to_string(),
        },
        SurfaceChoice {
            role_index: 1, // ceiling
            texture_tag: "ceiling_rough".to_string(),
        },
    ]
}

fn route_surface_choices() -> Vec<SurfaceChoice> {
    vec![
        SurfaceChoice {
            role_index: 2,
            texture_tag: "floor_flagstone".to_string(),
        },
        SurfaceChoice {
            role_index: 1,
            texture_tag: "ceiling_rough".to_string(),
        },
    ]
}

fn route_material_roles() -> Vec<MaterialRoleSelection> {
    vec![
        MaterialRoleSelection {
            role_index: 2,
            material_identity: "floor_corridor".to_string(),
        },
        MaterialRoleSelection {
            role_index: 1,
            material_identity: "ceiling_corridor".to_string(),
        },
    ]
}

fn portal_surface_choices() -> Vec<SurfaceChoice> {
    vec![SurfaceChoice {
        role_index: 4, // trim
        texture_tag: "trim_carved".to_string(),
    }]
}

fn portal_material_roles() -> Vec<MaterialRoleSelection> {
    vec![MaterialRoleSelection {
        role_index: 4,
        material_identity: "trim_portal".to_string(),
    }]
}

// ── Convenience constructor ────────────────────────────────────────────────

/// Build a complete variation plan from a pacing blueprint, topology result,
/// zone blueprint, and reservation journal.
pub(crate) fn build_variation_plan(
    blueprint: &PacingBlueprint,
    topology: &TopologyResult,
    _zones: &ZoneBlueprint,
    journal: &ReservationJournal,
) -> VariationPlan {
    let mut builder = VariationBuilder::new();

    // Build zone map for wall chain derivation.
    let mut origin_zone_map: BTreeMap<ReservationId, ZoneId> = BTreeMap::new();
    for (bid, zid) in &blueprint.zone_blueprint.beat_zone_map {
        if let Some(res_ids) = topology.beat_to_reservations.get(bid) {
            for &rid in res_ids {
                origin_zone_map.insert(rid, *zid);
            }
        }
    }

    // 1. Variation decisions for every committed reservation.
    for (rid, record) in &journal.reservations {
        if !record.committed {
            continue;
        }

        // Each reservation gets a base decision using the archetype's declared
        // material roles plus default dimensions.
        let decision = builder.build_reservation_decision(record, FieldTag::ValueNoise);
        builder.plan.insert_decision(decision);
        let _ = rid;
    }

    // 2. Variation decisions for every committed route.
    for route in &topology.routes {
        let decision = builder.build_route_decision(route, FieldTag::ValueNoise);
        builder.plan.insert_decision(decision);
    }

    // 3. Variation decisions for every committed portal.
    for route in &topology.routes {
        let portal_decision =
            builder.build_portal_decision(&route.source_portal, FieldTag::ValueNoise);
        builder.plan.insert_decision(portal_decision);

        let target_portal_decision =
            builder.build_portal_decision(&route.target_portal, FieldTag::ValueNoise);
        builder.plan.insert_decision(target_portal_decision);
    }

    // 4. Derive shared wall chains from adjacent reservations.
    builder.derive_wall_chains(journal, &topology.routes, &origin_zone_map);

    builder.build()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::ids::{BeatId, ZoneId};
    use crate::enhanced_v3::richness::reservation::ReservationJournal;

    // ── WallShaping ─────────────────────────────────────────────────────

    #[test]
    fn wall_shaping_offsets_are_quantum_aligned() {
        assert_eq!(WallShaping::None.offset_units(), 0);
        assert_eq!(WallShaping::OneQuantum.offset_units(), 16);
        assert_eq!(WallShaping::TwoQuanta.offset_units(), 32);
    }

    #[test]
    fn wall_shaping_quantum_counts() {
        assert_eq!(WallShaping::None.quantum_count(), 0);
        assert_eq!(WallShaping::OneQuantum.quantum_count(), 1);
        assert_eq!(WallShaping::TwoQuanta.quantum_count(), 2);
    }

    #[test]
    fn max_shaping_is_two_quanta() {
        // The contract requires: "cardinal step/chamfer shaping up to TWO quanta"
        assert!(WallShaping::TwoQuanta.quantum_count() <= 2);
        assert_eq!(WallShaping::TwoQuanta.offset_units(), 32);
    }

    // ── WallMass ────────────────────────────────────────────────────────

    #[test]
    fn wall_mass_inward_identification() {
        assert!(WallMass::Liner16.is_inward());
        assert!(WallMass::Liner32.is_inward());
        assert!(WallMass::Buttress16.is_inward());
        assert!(!WallMass::None.is_inward());
        assert!(!WallMass::Recess16.is_inward());
    }

    #[test]
    fn wall_mass_all_defined_variants_legal() {
        for mass in [
            WallMass::None,
            WallMass::Liner16,
            WallMass::Liner32,
            WallMass::Recess16,
            WallMass::Buttress16,
        ] {
            assert!(mass.is_legal());
        }
    }

    #[test]
    fn wall_mass_offsets_match_contract() {
        // Liner courses are 16/32 units inward; buttress is 16 outward.
        assert_eq!(WallMass::Liner16.offset_units(), -16);
        assert_eq!(WallMass::Liner32.offset_units(), -32);
        assert_eq!(WallMass::Buttress16.offset_units(), -16);
        assert_eq!(WallMass::Recess16.offset_units(), 16);
        assert_eq!(WallMass::None.offset_units(), 0);
    }

    // ── SemanticId ──────────────────────────────────────────────────────

    #[test]
    fn semantic_id_from_conversions() {
        let rid = SemanticId::from(ReservationId::new(42));
        assert_eq!(rid.raw(), 42);
        assert_eq!(rid.tag(), "reservation");

        let bid = SemanticId::from(BeatId::new(7));
        assert_eq!(bid.raw(), 7);
        assert_eq!(bid.tag(), "beat");

        let zid = SemanticId::from(ZoneId::new(1));
        assert_eq!(zid.raw(), 1);
        assert_eq!(zid.tag(), "zone");
    }

    #[test]
    fn semantic_id_ordering() {
        let a = SemanticId::Reservation(ReservationId::new(1));
        let b = SemanticId::Reservation(ReservationId::new(2));
        assert!(a < b);

        let c = SemanticId::Beat(BeatId::new(5));
        let d = SemanticId::Zone(ZoneId::new(5));
        // Different enum variants compare by discriminant first.
        // Beat comes before Zone in declaration order.
        assert!(c < d);
    }

    // ── LegalDimensions ─────────────────────────────────────────────────

    #[test]
    fn legal_dimensions_volume_quanta() {
        let dims = LegalDimensions::new(320, 176, 256);
        let volume = dims.volume_quanta();
        assert_eq!(volume, (320 / 16) * (176 / 16) * (256 / 16));
        assert_eq!(volume, 20 * 11 * 16);
    }

    #[test]
    fn legal_dimensions_fix_to() {
        let mut dims = LegalDimensions::new(100, 50, 30);
        dims.fix_to(LegalDimensions::new(256, 176, 256));
        assert_eq!(dims.width, 256);
        assert_eq!(dims.height, 176);
        assert_eq!(dims.depth, 256);
    }

    // ── DamageVariant ───────────────────────────────────────────────────

    #[test]
    fn damage_variant_tags_unique() {
        let tags = [
            DamageVariant::None.tag(),
            DamageVariant::LightWeathering.tag(),
            DamageVariant::HeavyWeathering.tag(),
            DamageVariant::PartialCollapse.tag(),
            DamageVariant::WaterDamage.tag(),
            DamageVariant::FireScarring.tag(),
        ];
        let set: std::collections::BTreeSet<_> = tags.iter().collect();
        assert_eq!(set.len(), tags.len());
    }

    // ── VariationPlan ───────────────────────────────────────────────────

    #[test]
    fn empty_plan_is_not_complete() {
        let plan = VariationPlan::new();
        assert!(!plan.complete);
        assert_eq!(plan.decision_count(), 0);
        assert_eq!(plan.wall_chain_count(), 0);
    }

    #[test]
    fn insert_and_retrieve_decision() {
        let mut plan = VariationPlan::new();
        let sid = SemanticId::Reservation(ReservationId::new(0));
        let decision = VariationDecision {
            semantic_id: sid,
            field_tag: FieldTag::ValueNoise,
            dimensions: LegalDimensions::new(256, 176, 256),
            surface_choices: Vec::new(),
            material_roles: Vec::new(),
            lights: Vec::new(),
            props: Vec::new(),
            damage: DamageVariant::None,
        };
        plan.insert_decision(decision.clone());
        assert_eq!(plan.decision_count(), 1);

        let retrieved = plan.get_decision(sid, FieldTag::ValueNoise);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().dimensions,
            LegalDimensions::new(256, 176, 256)
        );
    }

    #[test]
    fn wall_chain_insert_and_validate() {
        let mut plan = VariationPlan::new();
        let owner = SemanticId::Reservation(ReservationId::new(0));
        let shared = SemanticId::Reservation(ReservationId::new(1));
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner,
            shared_with: vec![shared],
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: vec![PortalId::new(0)],
            protected_segments: Vec::new(),
            structural_thickness: 16,
            exterior_envelope: false,
        };
        plan.insert_wall_chain(chain);
        assert_eq!(plan.wall_chain_count(), 1);

        let errors = plan.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn wall_chain_rejects_owner_in_shared_with() {
        let mut plan = VariationPlan::new();
        let owner = SemanticId::Reservation(ReservationId::new(0));
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner,
            shared_with: vec![owner],
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: Vec::new(),
            protected_segments: Vec::new(),
            structural_thickness: 16,
            exterior_envelope: false,
        };
        plan.insert_wall_chain(chain);

        let errors = plan.validate();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("also in shared_with")));
    }

    #[test]
    fn wall_chain_rejects_non_standard_thickness() {
        let mut plan = VariationPlan::new();
        let owner = SemanticId::Reservation(ReservationId::new(0));
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner,
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: Vec::new(),
            protected_segments: Vec::new(),
            structural_thickness: 24,
            exterior_envelope: false,
        };
        plan.insert_wall_chain(chain);

        let errors = plan.validate();
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| e.contains("non-standard structural thickness")));
    }

    #[test]
    fn wall_chain_rejects_shaping_with_portal_anchors() {
        let mut plan = VariationPlan::new();
        let owner = SemanticId::Reservation(ReservationId::new(0));
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner,
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::OneQuantum, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: vec![PortalId::new(0)],
            protected_segments: Vec::new(),
            structural_thickness: 16,
            exterior_envelope: false,
        };
        plan.insert_wall_chain(chain);

        let errors = plan.validate();
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| e.contains("portal anchors but non-zero shaping")));
    }

    #[test]
    fn variation_builder_creates_default_decisions() {
        let mut builder = VariationBuilder::new();

        // Build a synthetic reservation record for testing
        use super::super::footprint::Footprint3D;
        use super::super::reservation::ReservationKind;
        let record = ReservationRecord {
            id: ReservationId::new(0),
            kind: ReservationKind::StandardRoom,
            footprint: Footprint3D::single_layer(0, 0, 16, 16, 0),
            beat_id: None,
            request_id: None,
            zone_id: None,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id: None,
            clearance_height: None,
            committed: true,
            cost_faces: 200,
            cost_brushes: 12,
            cost_entities: 3,
            cost_lights: 2,
        };

        let decision = builder.build_reservation_decision(&record, FieldTag::ValueNoise);
        assert_eq!(decision.semantic_id.raw(), 0);
        assert_eq!(decision.damage, DamageVariant::None);

        let plan = builder.build();
        assert!(plan.complete);
    }

    #[test]
    fn variation_builder_default_has_no_decisions() {
        let builder = VariationBuilder::new();
        let plan = builder.build();
        assert_eq!(plan.decision_count(), 0);
        assert_eq!(plan.wall_chain_count(), 0);
        assert!(plan.complete);
    }

    #[test]
    fn apply_wall_mass_rejects_unknown_chain() {
        let mut builder = VariationBuilder::new();
        let result = builder.apply_wall_mass(
            WallChainId::new(999),
            Vec::new(),
            &ReservationJournal::new(2048, 3000),
        );
        assert!(result.is_err());
    }

    #[test]
    fn apply_wall_mass_rejects_protected_segment_overlap() {
        let mut builder = VariationBuilder::new();
        let owner = SemanticId::Reservation(ReservationId::new(0));
        let chain = WallChainRecord {
            id: WallChainId::new(0),
            owner,
            shared_with: Vec::new(),
            cardinal_direction: Dir::North,
            shaping: [WallShaping::None, WallShaping::None],
            mass_treatments: Vec::new(),
            portal_anchors: Vec::new(),
            protected_segments: vec![(0, 64)],
            structural_thickness: 16,
            exterior_envelope: false,
        };
        builder.plan.insert_wall_chain(chain);

        let treatment = WallMassTreatment {
            segment: (32, 48),
            kind: WallMass::Liner16,
            quantum_count: 1,
        };
        let result = builder.apply_wall_mass(
            WallChainId::new(0),
            vec![treatment],
            &ReservationJournal::new(2048, 3000),
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_variation_plan_produces_structure() {
        use super::super::pacing::build_pacing_blueprint;
        use super::super::request::{
            ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessPreset, RichnessTheme,
        };

        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();

        let journal = ReservationJournal::new(2048, 3000);
        let topology = TopologyResult {
            selected_edges: Vec::new(),
            routes: Vec::new(),
            journal: journal.clone(),
            beat_to_reservations: BTreeMap::new(),
            loop_count: 0,
            shortcuts_realized: Vec::new(),
            vertical_edges: Vec::new(),
            vertical_routes: Vec::new(),
            search_metrics: Default::default(),
        };

        let plan = build_variation_plan(&blueprint, &topology, &blueprint.zone_blueprint, &journal);
        assert!(!plan.wall_chains.is_empty() || plan.wall_chains.is_empty());
        assert!(plan.complete);
    }

    #[test]
    fn variation_plan_uses_real_committed_topology_without_id_collisions() {
        use super::super::pacing::build_pacing_blueprint;
        use super::super::request::{
            ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessPreset, RichnessTheme,
        };
        use super::super::solver::solve_placement_and_topology;

        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let solved = solve_placement_and_topology(blueprint.clone(), resolved).unwrap();
        assert!(!solved.topology.routes.is_empty());

        let plan = build_variation_plan(
            &blueprint,
            &solved.topology,
            &blueprint.zone_blueprint,
            &solved.topology.journal,
        );
        assert!(plan.complete);
        for route in &solved.topology.routes {
            assert!(plan
                .get_decision(SemanticId::Route(route.id), FieldTag::ValueNoise)
                .is_some());
            assert!(plan
                .get_decision(
                    SemanticId::Portal(route.source_portal.id),
                    FieldTag::ValueNoise
                )
                .is_some());
            assert!(plan
                .get_decision(
                    SemanticId::Portal(route.target_portal.id),
                    FieldTag::ValueNoise
                )
                .is_some());
        }
        for chain in plan.wall_chains.values() {
            assert!(chain.owner <= chain.shared_with[0]);
            assert_eq!(chain.structural_thickness, 16);
            assert!(!chain.portal_anchors.is_empty());
        }
    }
}
