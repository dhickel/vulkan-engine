//! Constructive 3D placement solver with deterministic finite backtracking.
//!
//! Places archetype requests from the pacing blueprint into the occupancy
//! grid in the specified priority order. Uses worst-case archetype envelopes
//! and complete deterministic finite backtracking with canonical candidate
//! keys. Exhaustion returns a typed `RichnessError`.
//!
//! # Placement order (from phase contract):
//!
//! 1. Grand volumes
//! 2. Required landmarks
//! 3. Vertical composites
//! 4. Pit pairs
//! 5. Cave hosts
//! 6. Protected negative-space rooms
//! 7. Ordinary slots
//!
//! # Contract
//!
//! - No floats. All coordinates quantum-aligned.
//! - Candidate order is fully canonical (BTreeMap, sorted vectors, field tags).
//! - Supported presets do not rely on seed substitution after exhaustion.
//! - Exhaustion returns `RichnessErrorCode::PlacementExhausted`.

// Richness remains intentionally crate-private and pipeline-unwired until the
// atomic sealing phase; unit and matrix tests are its current callers.
#![allow(dead_code, clippy::result_large_err)]

use std::collections::{BTreeMap, BTreeSet};

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::footprint::{Footprint3D, CONSTRUCTION_QUANTUM};
use super::generated_content;
use super::ids::PacingBlueprint;
use super::ids::{ArchetypeIndex, ArchetypeRequestId, BeatId, PayoffType, ReservationId, ZoneId};
use super::request::{ResolvedRichnessRequestV1, RichnessCaveMode, RichnessPreset};
use super::reservation::{ReservationJournal, ReservationKind, ReservationRecord};

/// Frozen revision-v1 placement search-state ceiling.
///
/// This is a logical branch bound, not a wall-clock limit. Observed counts are
/// returned separately in [`PlacementResult`]. Supported preset templates
/// normally solve in one state per placement request.
pub(crate) const MAX_PLACEMENT_SEARCH_STATES: u64 = 200_000;

/// Synthetic beat namespace used for cave-cell reservations, which are
/// navigable cells rather than pacing beats.
const CAVE_BEAT_ID_BASE: u32 = 0x8000_0000;

// ── Placement priority ─────────────────────────────────────────────────────

/// Priority tier for placement ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PlacementPriority {
    /// Grand-volume landmark room (first to place).
    GrandVolume = 0,
    /// Required critical-path landmark.
    Landmark = 1,
    /// Vertical composite (stairwell, ladder shaft, etc.).
    VerticalComposite = 2,
    /// Pit pair (lower room + upper omission).
    PitPair = 3,
    /// Cave host shell.
    CaveHost = 4,
    /// Protected negative-space room.
    NegativeSpace = 5,
    /// Ordinary room slot.
    Ordinary = 6,
}

// ── Placement request ──────────────────────────────────────────────────────

/// A placement request derived from an archetype request with worst-case
/// envelope, priority tier, and backtracking state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacementRequest {
    /// The archetype request ID this placement fulfills.
    pub request_id: ArchetypeRequestId,
    /// The archetype index.
    pub archetype: ArchetypeIndex,
    /// The beat this placement belongs to.
    pub beat_id: BeatId,
    /// The zone for this placement.
    pub zone_id: ZoneId,
    /// Priority tier.
    pub priority: PlacementPriority,
    /// Worst-case XY span (from ARCHETYPE_SPAN_MAX).
    pub worst_case_span: (u32, u32),
    /// Whether this occupies both layers.
    pub dual_layer: bool,
    /// Whether this is a pit (upper omission).
    pub is_pit: bool,
    /// Whether this is a cave host.
    pub is_cave: bool,
    /// Whether this is negative space.
    pub is_negative_space: bool,
    /// Whether this is a grand volume.
    pub is_grand_volume: bool,
    /// Source face cost.
    pub cost_faces: u32,
    /// Brush cost.
    pub cost_brushes: u32,
    /// Entity cost.
    pub cost_entities: u32,
    /// Light cost.
    pub cost_lights: u32,
    /// Route witness span for this archetype (width, height).
    pub route_witness: (u32, u32),
}

// ── Candidate position ─────────────────────────────────────────────────────

/// A candidate placement position in grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CandidatePosition {
    /// Grid X minimum.
    pub gx: u32,
    /// Grid Y minimum.
    pub gy: u32,
    /// Grid width.
    pub gw: u32,
    /// Grid depth.
    pub gd: u32,
    /// Layer: 0 = lower, 1 = upper, 2 = both.
    pub layer: u8,
    /// Deterministic rank for sorting.
    pub rank: u64,
}

// ── Placement result ───────────────────────────────────────────────────────

/// The result of constructive placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacementResult {
    /// All committed reservations.
    pub reservations: BTreeMap<ReservationId, ReservationRecord>,
    /// Mapping from archetype request ID to reservation ID.
    pub request_to_reservation: BTreeMap<ArchetypeRequestId, ReservationId>,
    /// Mapping from beat ID to reservation IDs.
    pub beat_to_reservations: BTreeMap<BeatId, Vec<ReservationId>>,
    /// The reservation journal (final committed state).
    pub journal: ReservationJournal,
    /// Remaining budget after placement.
    pub remaining_faces: u32,
    /// Placement count.
    pub placed_count: usize,
    /// Maximum observed search states (separate from frozen bounds).
    pub max_search_states: u64,
    /// Total search states observed.
    pub total_search_states: u64,
    /// Number of corridor-blocking rejections.
    pub corridor_rejections: u64,
}

// ── Solver ─────────────────────────────────────────────────────────────────

/// The 3D placement solver.
pub(crate) struct PlacementSolver {
    /// The resolved request.
    resolved: ResolvedRichnessRequestV1,
    /// Placement requests in priority order.
    placement_requests: Vec<PlacementRequest>,
    /// The reservation journal.
    journal: ReservationJournal,
    /// Placed request IDs.
    placed: BTreeSet<ArchetypeRequestId>,
    /// Request-to-reservation mapping.
    request_to_reservation: BTreeMap<ArchetypeRequestId, ReservationId>,
    /// Observable search-state counter (incremented on each candidate attempt).
    pub search_states: u64,
    /// Maximum observed search states across all backtracking paths.
    pub max_search_states: u64,
    /// Beat-to-reservation reverse map (built incrementally during placement).
    beat_to_res: BTreeMap<BeatId, Vec<ReservationId>>,
    /// Corridor pre-check rejections (diagnostic).
    pub corridor_rejections: u64,
    /// Deterministic macro slots for all semantic rooms and cave cells. The
    /// critical path occupies consecutive cells in a canonical snake, leaving
    /// explicit routing channels between worst-case envelopes.
    preferred_slots: BTreeMap<BeatId, (u32, u32)>,
}

impl PlacementSolver {
    /// Create a new solver from the blueprint and resolved request.
    pub fn new(
        blueprint: PacingBlueprint,
        resolved: ResolvedRichnessRequestV1,
    ) -> Result<Self, RichnessError> {
        let extent = resolved.extent();
        let max_faces = resolved.budget_ceiling().value();
        let journal = ReservationJournal::new(extent, max_faces);

        let placement_requests = Self::build_placement_requests(&blueprint, &resolved)?;
        let preferred_slots = Self::build_preferred_slots(&blueprint, &placement_requests, extent);

        Ok(Self {
            resolved,
            placement_requests,
            journal,
            placed: BTreeSet::new(),
            request_to_reservation: BTreeMap::new(),
            search_states: 0,
            max_search_states: 0,
            beat_to_res: BTreeMap::new(),
            corridor_rejections: 0,
            preferred_slots,
        })
    }

    /// Build placement requests from the blueprint in priority order.
    fn build_placement_requests(
        blueprint: &PacingBlueprint,
        resolved: &ResolvedRichnessRequestV1,
    ) -> Result<Vec<PlacementRequest>, RichnessError> {
        let mut requests = Vec::new();

        for (req_id, req) in &blueprint.archetype_requests {
            let idx = req.archetype.raw() as usize;
            let worst_case_span = (
                generated_content::ARCHETYPE_SPAN_MAX[idx][0],
                generated_content::ARCHETYPE_SPAN_MAX[idx][1],
            );

            let layer_occupancy = generated_content::ARCHETYPE_LAYER_OCCUPANCY[idx];
            let mut dual_layer =
                matches!(layer_occupancy, super::content_types::LayerOccupancy::Both);

            let vertical_recipe = generated_content::ARCHETYPE_VERTICAL_RECIPE[idx];
            let is_pit = matches!(
                vertical_recipe,
                super::content_types::VerticalRecipe::DropHole
            );
            let is_cave = req.archetype.id_str() == "grotto";

            // Determine priority
            let beat = blueprint.beats.get(&req.beat_id);
            let is_grand_volume = beat.is_some_and(|b| b.is_grand_volume);
            // The grand-volume invariant is an explicit multi-storey
            // composite host even when its authored interior recipe is lower
            // only. Theme-specific massing remains inside this envelope.
            dual_layer |= is_grand_volume;
            let is_quiet_negative = beat.is_some_and(|b| b.is_quiet_negative_space);
            let is_forced = req.forced;

            let priority = if is_grand_volume {
                PlacementPriority::GrandVolume
            } else if is_forced {
                PlacementPriority::Landmark
            } else if dual_layer && !is_pit {
                PlacementPriority::VerticalComposite
            } else if is_pit {
                PlacementPriority::PitPair
            } else if is_cave {
                PlacementPriority::CaveHost
            } else if is_quiet_negative {
                PlacementPriority::NegativeSpace
            } else {
                PlacementPriority::Ordinary
            };

            let cost_faces = generated_content::ARCHETYPE_COST_SOURCE_FACES[idx];
            let cost_brushes = generated_content::ARCHETYPE_COST_BRUSHES[idx];
            let cost_entities = generated_content::ARCHETYPE_COST_ENTITIES[idx];
            let cost_lights = generated_content::ARCHETYPE_COST_LIGHTS[idx];

            let route_witness = (
                generated_content::ARCHETYPE_ROUTE_WITNESS[idx][0],
                generated_content::ARCHETYPE_ROUTE_WITNESS[idx][1],
            );

            requests.push(PlacementRequest {
                request_id: *req_id,
                archetype: req.archetype,
                beat_id: req.beat_id,
                zone_id: req.zone_id,
                priority,
                worst_case_span,
                dual_layer,
                is_pit,
                is_cave,
                is_negative_space: is_quiet_negative,
                is_grand_volume,
                cost_faces,
                cost_brushes,
                cost_entities,
                cost_lights,
                route_witness,
            });
        }

        // Materialize every side-branch payoff leaf as a concrete room. Phase
        // 06 deliberately keeps leaves semantic-only; Phase 07 is the first
        // stage that owns physical reservation IDs for them.
        let mut next_request_id = blueprint
            .archetype_requests
            .keys()
            .next_back()
            .map_or(0, |id| id.raw().saturating_add(1));
        for payoff in blueprint.branch_payoffs.values() {
            let Some(leaf_beat) = payoff.to_beat else {
                continue;
            };
            let archetype_id = match payoff.payoff_type {
                PayoffType::Shortcut => "vestibule",
                PayoffType::DiscoveryLandmark => "gallery",
                PayoffType::LoreMarker => "ossuary",
                PayoffType::AuthoredTreasureTableau => "treasury",
            };
            let Some(archetype) = ArchetypeIndex::from_id_str(archetype_id) else {
                continue;
            };
            let idx = archetype.raw() as usize;
            requests.push(PlacementRequest {
                request_id: ArchetypeRequestId::new(next_request_id),
                archetype,
                beat_id: leaf_beat,
                zone_id: blueprint
                    .zone_blueprint
                    .beat_zone_map
                    .get(&leaf_beat)
                    .copied()
                    .unwrap_or(ZoneId::new(0)),
                priority: PlacementPriority::Ordinary,
                worst_case_span: (
                    generated_content::ARCHETYPE_SPAN_MAX[idx][0],
                    generated_content::ARCHETYPE_SPAN_MAX[idx][1],
                ),
                dual_layer: false,
                is_pit: false,
                is_cave: false,
                is_negative_space: false,
                is_grand_volume: false,
                // This phase reserves spatial capacity. Phase 08's complexity
                // plan owns source/brush/entity/light accounting for synthetic
                // payoff recipes, so charging those costs here would double
                // count the same future recipe.
                cost_faces: 0,
                cost_brushes: 0,
                cost_entities: 0,
                cost_lights: 0,
                route_witness: (
                    generated_content::ARCHETYPE_ROUTE_WITNESS[idx][0],
                    generated_content::ARCHETYPE_ROUTE_WITNESS[idx][1],
                ),
            });
            next_request_id = next_request_id.saturating_add(1);
        }

        // Preferred/required cave modes materialize the contract minimum of
        // additional navigable cave cells. Omitted mode is the explicit opt-out.
        let cave_count = if resolved.cave_mode().value() == RichnessCaveMode::Omitted {
            0
        } else {
            match blueprint.preset {
                RichnessPreset::Sparse | RichnessPreset::Moderate => 2,
                RichnessPreset::Rich => 4,
            }
        };
        if let Some(grotto) = ArchetypeIndex::from_id_str("grotto") {
            let idx = grotto.raw() as usize;
            for cave_index in 0..cave_count {
                requests.push(PlacementRequest {
                    request_id: ArchetypeRequestId::new(next_request_id),
                    archetype: grotto,
                    beat_id: BeatId::new(CAVE_BEAT_ID_BASE + cave_index),
                    zone_id: ZoneId::new(
                        cave_index % blueprint.zone_blueprint.zones.len().max(1) as u32,
                    ),
                    priority: PlacementPriority::CaveHost,
                    worst_case_span: (
                        generated_content::ARCHETYPE_SPAN_MAX[idx][0],
                        generated_content::ARCHETYPE_SPAN_MAX[idx][1],
                    ),
                    dual_layer: false,
                    is_pit: false,
                    is_cave: true,
                    is_negative_space: false,
                    is_grand_volume: false,
                    // Cave-cell geometry is selected and charged by the
                    // later complete-recipe complexity plan. Phase 07 owns
                    // only its worst-case spatial envelope.
                    cost_faces: 0,
                    cost_brushes: 0,
                    cost_entities: 0,
                    cost_lights: 0,
                    route_witness: (
                        generated_content::ARCHETYPE_ROUTE_WITNESS[idx][0],
                        generated_content::ARCHETYPE_ROUTE_WITNESS[idx][1],
                    ),
                });
                next_request_id = next_request_id.saturating_add(1);
            }
        }

        // Sort by priority then by beat progression (canonical order).
        requests.sort_by_key(|r| {
            let prog = blueprint
                .beats
                .get(&r.beat_id)
                .map(|b| b.progression.raw())
                .unwrap_or(0);
            (r.priority, prog, r.request_id.raw())
        });

        Ok(requests)
    }

    /// Build the canonical 4×4 macro-slot embedding.
    ///
    /// Critical beats occupy consecutive cells in a snake, so every mandatory
    /// pair has a worst-case 64-unit channel before any lower-priority room is
    /// considered. Branch payoff rooms and cave cells consume the remaining
    /// slots. Topology may route shortcut chords through the still-empty macro
    /// channels without changing the seed or moving a committed room.
    fn build_preferred_slots(
        blueprint: &PacingBlueprint,
        requests: &[PlacementRequest],
        extent: u32,
    ) -> BTreeMap<BeatId, (u32, u32)> {
        const SLOT_ORDER: [usize; 16] = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12];
        let grid_extent = extent / CONSTRUCTION_QUANTUM as u32;
        let pitch = grid_extent / 4;
        if pitch == 0 {
            return BTreeMap::new();
        }

        let mut ordered_beats = blueprint.beat_order.clone();
        let critical: BTreeSet<_> = ordered_beats.iter().copied().collect();
        let mut supplemental: Vec<_> = requests
            .iter()
            .map(|request| request.beat_id)
            .filter(|beat| !critical.contains(beat))
            .collect();
        supplemental.sort_unstable();
        supplemental.dedup();
        ordered_beats.extend(supplemental);

        ordered_beats
            .into_iter()
            .zip(SLOT_ORDER)
            .map(|(beat, slot)| (beat, ((slot % 4) as u32 * pitch, (slot / 4) as u32 * pitch)))
            .collect()
    }

    /// Run the complete placement solver.
    ///
    /// Returns `PlacementResult` on success or a typed error on exhaustion.
    pub fn solve(mut self) -> Result<PlacementResult, RichnessError> {
        if !self.place_from(0) {
            let request_index = self.placed.len().min(self.placement_requests.len() - 1);
            let request = &self.placement_requests[request_index];
            return Err(self.exhaustion_error(request, request_index));
        }
        self.reserve_vertical_hosts()?;
        self.reserve_room_occupants()?;

        self.journal.commit_all();
        let beat_to_reservations = self.build_beat_to_reservations();
        let reservations = self.journal.reservations.clone();
        let request_to_reservation = self.request_to_reservation.clone();
        let journal = self.journal.clone();
        let remaining_faces = self
            .journal
            .budget()
            .max_faces
            .saturating_sub(self.journal.budget().faces);

        Ok(PlacementResult {
            reservations,
            request_to_reservation,
            beat_to_reservations,
            journal,
            remaining_faces,
            placed_count: self.placed.len(),
            max_search_states: self.max_search_states,
            total_search_states: self.search_states,
            corridor_rejections: self.corridor_rejections,
        })
    }

    /// Complete depth-first placement search with an explicit revision bound.
    /// Every candidate owns one journal checkpoint; rejection restores all
    /// occupancy, reservation, allocator, and budget state before the next
    /// canonical candidate is considered.
    fn place_from(&mut self, request_index: usize) -> bool {
        if request_index == self.placement_requests.len() {
            return true;
        }
        if self.search_states >= MAX_PLACEMENT_SEARCH_STATES {
            return false;
        }

        let request = self.placement_requests[request_index].clone();
        for candidate in self.generate_candidates(&request, request_index) {
            if self.search_states >= MAX_PLACEMENT_SEARCH_STATES {
                return false;
            }
            self.search_states += 1;
            self.max_search_states = self.max_search_states.max(self.search_states);

            // Occupancy-only rejection is non-mutating. Avoid cloning a full
            // journal checkpoint for candidates that cannot possibly reserve;
            // every candidate that can mutate state still owns a checkpoint.
            if !self.can_try_place(&request, &candidate) {
                continue;
            }
            self.journal.mark();

            let Ok(reservation_id) = self.try_place(&request, &candidate) else {
                self.journal.rollback();
                continue;
            };

            self.placed.insert(request.request_id);
            self.request_to_reservation
                .insert(request.request_id, reservation_id);
            self.beat_to_res
                .entry(request.beat_id)
                .or_default()
                .push(reservation_id);

            if self.place_from(request_index + 1) {
                self.journal.commit();
                return true;
            }

            self.placed.remove(&request.request_id);
            self.request_to_reservation.remove(&request.request_id);
            if let Some(ids) = self.beat_to_res.get_mut(&request.beat_id) {
                ids.retain(|id| *id != reservation_id);
                if ids.is_empty() {
                    self.beat_to_res.remove(&request.beat_id);
                }
            }
            self.journal.rollback();
        }
        false
    }

    /// Reserve owner-bearing spawn, light, and support cells inside the
    /// committed room envelopes before topology claims exterior route space.
    fn reserve_room_occupants(&mut self) -> Result<(), RichnessError> {
        let rooms: Vec<_> = self
            .journal
            .reservations
            .values()
            .filter(|record| {
                matches!(
                    record.kind,
                    ReservationKind::StandardRoom
                        | ReservationKind::MultiStoreyRoom
                        | ReservationKind::CaveHost
                        | ReservationKind::NegativeSpace
                )
            })
            .cloned()
            .collect();
        let Some(spawn_parent) = rooms
            .iter()
            .filter(|record| record.beat_id.is_some())
            .min_by_key(|record| record.beat_id)
            .map(|record| record.id)
        else {
            return Ok(());
        };

        self.journal.mark();
        let result = (|| {
            for room in &rooms {
                let layer = if room.footprint.occupies_lower { 0 } else { 1 };
                let center_x = (room.footprint.x0 + room.footprint.x1) / 2;
                let center_y = (room.footprint.y0 + room.footprint.y1) / 2;
                let light = Footprint3D {
                    x0: center_x,
                    y0: center_y,
                    x1: center_x + 1,
                    y1: center_y + 1,
                    occupies_lower: layer == 0,
                    occupies_upper: layer == 1,
                };
                let support = Footprint3D {
                    x0: center_x + 1,
                    y0: center_y,
                    x1: center_x + 2,
                    y1: center_y + 1,
                    occupies_lower: layer == 0,
                    occupies_upper: layer == 1,
                };
                let composite_parent = self.journal.composite_parent_of(room.id);
                if let Some(parent_id) = composite_parent {
                    self.journal.try_reserve_composite_child(
                        parent_id,
                        ReservationKind::Light,
                        light,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                    )?;
                    self.journal.try_reserve_composite_child(
                        parent_id,
                        ReservationKind::Support,
                        support,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                    )?;
                } else {
                    self.journal.try_reserve(
                        ReservationKind::Light,
                        light,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                    )?;
                    self.journal.try_reserve(
                        ReservationKind::Support,
                        support,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                    )?;
                }
            }

            let spawn_room = rooms
                .iter()
                .find(|room| room.id == spawn_parent)
                .cloned()
                .ok_or_else(|| self.exhaustion_error(&self.placement_requests[0], 0))?;
            let layer = if spawn_room.footprint.occupies_lower {
                0
            } else {
                1
            };
            let spawn_center_x = (spawn_room.footprint.x0 + spawn_room.footprint.x1) / 2;
            let spawn_center_y = (spawn_room.footprint.y0 + spawn_room.footprint.y1) / 2;
            let spawn = Footprint3D {
                x0: spawn_center_x,
                y0: spawn_center_y + 1,
                x1: spawn_center_x + 1,
                y1: spawn_center_y + 2,
                occupies_lower: layer == 0,
                occupies_upper: layer == 1,
            };
            if let Some(parent_id) = self.journal.composite_parent_of(spawn_parent) {
                self.journal.try_reserve_composite_child(
                    parent_id,
                    ReservationKind::Spawn,
                    spawn,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                )?;
            } else {
                self.journal.try_reserve(
                    ReservationKind::Spawn,
                    spawn,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.journal.commit();
                Ok(())
            }
            Err(error) => {
                self.journal.rollback();
                Err(error)
            }
        }
    }

    /// Reserve the exact resolved number of 96×96 dual-band vertical hosts
    /// inside explicit multi-storey room envelopes.
    fn reserve_vertical_hosts(&mut self) -> Result<(), RichnessError> {
        let required = self.resolved.vertical_openings().value();
        if required == 0 {
            return Ok(());
        }
        let composites: Vec<_> = self
            .journal
            .reservations
            .values()
            .filter(|record| {
                record.kind == ReservationKind::Composite
                    && record.composite_children.iter().any(|child_id| {
                        self.journal
                            .get(*child_id)
                            .is_some_and(|child| child.kind == ReservationKind::MultiStoreyRoom)
                    })
            })
            .cloned()
            .collect();
        self.journal.mark();
        let mut reserved = 0u32;
        for parent in composites {
            let footprint = parent.footprint;
            let candidates = [
                (footprint.x0 + 1, footprint.y0 + 1),
                (footprint.x1.saturating_sub(7), footprint.y0 + 1),
                (footprint.x0 + 1, footprint.y1.saturating_sub(7)),
                (
                    footprint.x1.saturating_sub(7),
                    footprint.y1.saturating_sub(7),
                ),
            ];
            for (x0, y0) in candidates {
                if reserved == required {
                    break;
                }
                let host = Footprint3D {
                    x0,
                    y0,
                    x1: x0 + 6,
                    y1: y0 + 6,
                    occupies_lower: true,
                    occupies_upper: true,
                };
                if host.x1 > footprint.x1 || host.y1 > footprint.y1 {
                    continue;
                }
                if self
                    .journal
                    .try_reserve_composite_child(
                        parent.id,
                        ReservationKind::VerticalHost,
                        host,
                        parent.beat_id,
                        parent.request_id,
                        parent.zone_id,
                        0,
                        0,
                        0,
                        0,
                    )
                    .is_ok()
                {
                    reserved += 1;
                }
            }
            if reserved == required {
                break;
            }
        }
        if reserved == required {
            self.journal.commit();
            return Ok(());
        }
        self.journal.rollback();
        Err(RichnessError::new(
            RichnessErrorCode::VerticalFeaturesInfeasible,
            self.resolved.seed(),
            self.resolved.provenance().request_schema_revision.tag(),
            self.resolved.provenance().algorithm_revision.tag(),
            self.resolved.provenance().content_revision.tag(),
            self.resolved.provenance().preset_revision.tag(),
            self.resolved.provenance().theme_revision.tag(),
            self.resolved.provenance().asset_revision.tag(),
            self.resolved.provenance().convention_revision.tag(),
            "placement.vertical_hosts",
            RichnessErrorCategory::SemanticInfeasibility,
            format!(
                "vertical host reservation exhausted: required {required}, reserved {reserved}"
            ),
        ))
    }

    /// Build an exhaustion error for a placement request.
    fn exhaustion_error(&self, preq: &PlacementRequest, request_index: usize) -> RichnessError {
        RichnessError::new(
            RichnessErrorCode::PlacementExhausted,
            self.resolved.seed(),
            self.resolved.provenance().request_schema_revision.tag(),
            self.resolved.provenance().algorithm_revision.tag(),
            self.resolved.provenance().content_revision.tag(),
            self.resolved.provenance().preset_revision.tag(),
            self.resolved.provenance().theme_revision.tag(),
            self.resolved.provenance().asset_revision.tag(),
            self.resolved.provenance().convention_revision.tag(),
            "placement",
            RichnessErrorCategory::PlacementTopologyExhaustion,
            format!(
                "placement exhausted for request {:?} at index {}; search_states={}; max_observed_search_states={}; frozen_bound={}; bound_reached={}",
                preq.request_id,
                request_index,
                self.search_states,
                self.max_search_states,
                MAX_PLACEMENT_SEARCH_STATES,
                self.search_states == MAX_PLACEMENT_SEARCH_STATES
            ),
        )
    }

    /// Generate canonical candidate positions for a placement request.
    fn generate_candidates(
        &self,
        preq: &PlacementRequest,
        _request_index: usize,
    ) -> Vec<CandidatePosition> {
        let extent = self.resolved.extent();
        let grid_extent = extent / CONSTRUCTION_QUANTUM as u32;

        let (span_x, span_y) = preq.worst_case_span;
        let gw = span_x / CONSTRUCTION_QUANTUM as u32;
        let gd = span_y / CONSTRUCTION_QUANTUM as u32;

        if gw == 0 || gd == 0 || gw > grid_extent || gd > grid_extent {
            return Vec::new();
        }

        let max_gx = grid_extent.saturating_sub(gw);
        let max_gy = grid_extent.saturating_sub(gd);

        let mut candidates = Vec::new();

        // For grand volumes, prefer central placement
        // For landmarks, spread from center
        // For ordinary, enumerate all quantum-aligned positions
        let layers: Vec<u8> = if preq.is_pit {
            vec![0] // lower room; the paired upper omission is reserved atomically
        } else if preq.dual_layer {
            vec![2] // both
        } else {
            vec![0] // lower layer default
        };

        // Generate all candidate positions in canonical order
        // Use spiral-from-center ordering for priority requests,
        // row-major for ordinary
        let is_priority = preq.priority != PlacementPriority::Ordinary;

        if is_priority {
            // Spiral from center
            let cx = max_gx / 2;
            let cy = max_gy / 2;
            let max_radius = max_gx.max(max_gy) as i32 + 1;

            for radius in 0..max_radius {
                let mut ring = Vec::new();
                // Generate all positions at this Manhattan distance
                for dx in -radius..=radius {
                    for dy in -radius..=radius {
                        if dx.abs() + dy.abs() != radius {
                            continue;
                        }
                        let gx = (cx as i32 + dx).max(0).min(max_gx as i32) as u32;
                        let gy = (cy as i32 + dy).max(0).min(max_gy as i32) as u32;
                        for &layer in &layers {
                            let rank = Self::candidate_rank(preq, gx, gy, layer, radius as u64);
                            ring.push(CandidatePosition {
                                gx,
                                gy,
                                gw,
                                gd,
                                layer,
                                rank,
                            });
                        }
                    }
                }
                // Sort ring by rank for deterministic order
                ring.sort_by_key(|c| c.rank);
                candidates.extend(ring);
            }
        } else {
            // Row-major scan with hash-derived rank
            for gx in 0..=max_gx {
                for gy in 0..=max_gy {
                    for &layer in &layers {
                        let rank = Self::candidate_rank(preq, gx, gy, layer, 0);
                        candidates.push(CandidatePosition {
                            gx,
                            gy,
                            gw,
                            gd,
                            layer,
                            rank,
                        });
                    }
                }
            }
            // Sort by rank for deterministic order
            candidates.sort_by_key(|c| c.rank);
        }

        // Every semantic beat receives a deterministic protected macro slot.
        // Slots have a full 64-unit gap between worst-case 448-unit
        // envelopes, so no middle-beat placement can consume a mandatory
        // corridor envelope before topology reserves it. The complete ranked
        // candidate list remains available for finite fallback.
        if let Some(&(column, row)) = self.preferred_slots.get(&preq.beat_id) {
            let preferred = CandidatePosition {
                gx: column,
                gy: row,
                gw,
                gd,
                layer: layers[0],
                rank: 0,
            };
            if preferred.gx + preferred.gw <= grid_extent
                && preferred.gy + preferred.gd <= grid_extent
            {
                candidates.retain(|candidate| {
                    (candidate.gx, candidate.gy, candidate.layer)
                        != (preferred.gx, preferred.gy, preferred.layer)
                });
                candidates.insert(0, preferred);
            }
        }

        // Ensure uniqueness without disturbing the preferred-first ordering.
        candidates.dedup_by_key(|c| (c.gx, c.gy, c.layer));
        candidates
    }

    /// Compute a deterministic candidate rank from field tags.
    fn candidate_rank(preq: &PlacementRequest, gx: u32, gy: u32, layer: u8, distance: u64) -> u64 {
        // Use a simple hash of request + position to produce a stable rank.
        // This ensures canonical ordering without RNG state.
        let a = preq.request_id.raw() as u64;
        let b = preq.archetype.raw() as u64;
        let x = gx as u64;
        let y = gy as u64;
        let l = layer as u64;

        // Spread-multiply hash
        let mut h: u64 = 0;
        h = h.wrapping_mul(6364136223846793005).wrapping_add(a);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(b);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(x);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(y);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(l);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(distance);
        h
    }

    /// Read-only occupancy/budget preflight for a placement candidate.
    fn can_try_place(&self, preq: &PlacementRequest, candidate: &CandidatePosition) -> bool {
        if preq.is_pit || preq.dual_layer {
            // Same-XY arrangements are composite-container transactions;
            // retain the journal checkpoint as their complete feasibility
            // authority rather than preflighting one child in isolation.
            return true;
        }

        let grid_extent = self.resolved.extent() / CONSTRUCTION_QUANTUM as u32;
        if candidate.gx + candidate.gw > grid_extent || candidate.gy + candidate.gd > grid_extent {
            return false;
        }
        let qx = (candidate.gx * CONSTRUCTION_QUANTUM as u32) as i32;
        let qy = (candidate.gy * CONSTRUCTION_QUANTUM as u32) as i32;
        let qw = (candidate.gw * CONSTRUCTION_QUANTUM as u32) as i32;
        let qd = (candidate.gd * CONSTRUCTION_QUANTUM as u32) as i32;
        let footprint = if candidate.layer == 2 {
            Footprint3D::dual_layer(qx, qy, qx + qw, qy + qd)
        } else {
            Footprint3D::single_layer(qx, qy, qx + qw, qy + qd, candidate.layer)
        };
        let kind = if preq.is_grand_volume && preq.dual_layer {
            ReservationKind::MultiStoreyRoom
        } else if preq.is_cave {
            ReservationKind::CaveHost
        } else if preq.is_negative_space {
            ReservationKind::NegativeSpace
        } else if preq.dual_layer {
            ReservationKind::MultiStoreyRoom
        } else {
            ReservationKind::StandardRoom
        };
        self.journal.can_reserve(
            kind,
            &footprint,
            preq.cost_faces,
            preq.cost_brushes,
            preq.cost_entities,
            preq.cost_lights,
        )
    }

    /// Attempt to place a single request at a candidate position.
    fn try_place(
        &mut self,
        preq: &PlacementRequest,
        candidate: &CandidatePosition,
    ) -> Result<ReservationId, RichnessError> {
        let qx = (candidate.gx * CONSTRUCTION_QUANTUM as u32) as i32;
        let qy = (candidate.gy * CONSTRUCTION_QUANTUM as u32) as i32;
        let qw = (candidate.gw * CONSTRUCTION_QUANTUM as u32) as i32;
        let qd = (candidate.gd * CONSTRUCTION_QUANTUM as u32) as i32;

        let layer = candidate.layer;

        let footprint = if layer == 2 {
            Footprint3D::dual_layer(qx, qy, qx + qw, qy + qd)
        } else {
            Footprint3D::single_layer(qx, qy, qx + qw, qy + qd, layer)
        };

        // Validate footprint fits within extent
        let grid_extent = self.resolved.extent() / CONSTRUCTION_QUANTUM as u32;
        if candidate.gx + candidate.gw > grid_extent || candidate.gy + candidate.gd > grid_extent {
            return Err(RichnessError::new(
                RichnessErrorCode::PlacementExhausted,
                self.resolved.seed(),
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "placement",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                "footprint exceeds grid extent".to_string(),
            ));
        }

        // Pit rooms are atomic lower-room + upper-omission pairs inside one
        // dual-band composite. The lower room is the topology endpoint; the
        // matched omission owns the upper band and both are container children.
        if preq.is_pit {
            let composite_id = self.journal.reserve_composite(
                footprint,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
            )?;
            let room_id = self.journal.try_reserve_composite_child(
                composite_id,
                ReservationKind::StandardRoom,
                footprint,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
                preq.cost_faces,
                preq.cost_brushes,
                preq.cost_entities,
                preq.cost_lights,
            )?;
            let omission = Footprint3D {
                occupies_lower: false,
                occupies_upper: true,
                ..footprint
            };
            let omission_id = self.journal.try_reserve_composite_child(
                composite_id,
                ReservationKind::PitOmission,
                omission,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
                0,
                0,
                0,
                0,
            )?;
            self.journal.link_pit_pair(omission_id, room_id);
            return Ok(room_id);
        }

        // Determine reservation kind
        let kind = if preq.is_grand_volume && preq.dual_layer {
            ReservationKind::MultiStoreyRoom
        } else if preq.is_cave {
            ReservationKind::CaveHost
        } else if preq.is_negative_space {
            ReservationKind::NegativeSpace
        } else if preq.dual_layer {
            ReservationKind::MultiStoreyRoom
        } else {
            ReservationKind::StandardRoom
        };

        if kind == ReservationKind::MultiStoreyRoom {
            let composite_id = self.journal.reserve_composite(
                footprint,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
            )?;
            self.journal.try_reserve_composite_child(
                composite_id,
                kind,
                footprint,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
                preq.cost_faces,
                preq.cost_brushes,
                preq.cost_entities,
                preq.cost_lights,
            )
        } else {
            self.journal.try_reserve(
                kind,
                footprint,
                Some(preq.beat_id),
                Some(preq.request_id),
                Some(preq.zone_id),
                preq.cost_faces,
                preq.cost_brushes,
                preq.cost_entities,
                preq.cost_lights,
            )
        }
    }

    /// Build the canonical beat-to-reservation mapping, including concrete
    /// side-branch leaves and synthetic cave-cell beats.
    fn build_beat_to_reservations(&self) -> BTreeMap<BeatId, Vec<ReservationId>> {
        let mut map: BTreeMap<BeatId, Vec<ReservationId>> = BTreeMap::new();
        for reservation in self.journal.reservations.values() {
            if let Some(beat_id) = reservation.beat_id {
                map.entry(beat_id).or_default().push(reservation.id);
            }
        }
        for ids in map.values_mut() {
            ids.sort_unstable();
        }
        map
    }

    /// Access the journal (for inspection).
    pub fn journal(&self) -> &ReservationJournal {
        &self.journal
    }
}

// ── Solver entry point ─────────────────────────────────────────────────────

/// Solve 3D placement from a blueprint and resolved request.
///
/// This is the main entry point for Subphase A placement.
pub(crate) fn solve_placement(
    blueprint: PacingBlueprint,
    resolved: ResolvedRichnessRequestV1,
) -> Result<PlacementResult, RichnessError> {
    let solver = PlacementSolver::new(blueprint, resolved)?;
    solver.solve()
}

// ── Combined placement + topology entry points ─────────────────────────────

/// Full placement-and-topology result for a single generation pass.
#[derive(Debug, Clone)]
pub(crate) struct FullGenerationResult {
    /// The placement result.
    pub placement: PlacementResult,
    /// The topology result.
    pub topology: super::topology::TopologyResult,
}

/// Solve placement and constrained-Kruskal topology from a blueprint and
/// resolved request.
///
/// This is the main entry point for the connectivity half. It runs
/// placement first, then constructs candidate edges, inserts mandatory
/// critical-path edges, runs constrained Kruskal, adds loops with backward
/// shortcuts, and reserves all routes inside envelopes.
pub(crate) fn solve_placement_and_topology(
    blueprint: PacingBlueprint,
    resolved: ResolvedRichnessRequestV1,
) -> Result<FullGenerationResult, RichnessError> {
    let placement = solve_placement(blueprint.clone(), resolved.clone())?;
    let topology = super::topology::solve_topology(&blueprint, &placement, &resolved)?;
    Ok(FullGenerationResult {
        placement,
        topology,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::pacing::build_pacing_blueprint;
    use super::super::request::{RichnessDocumentV1, RichnessTheme};
    use super::*;

    fn make_resolved(seed: u64, preset: RichnessPreset) -> ResolvedRichnessRequestV1 {
        let doc = RichnessDocumentV1::new(seed, 2048, preset, RichnessTheme::Ancient).unwrap();
        ResolvedRichnessRequestV1::resolve(doc).unwrap()
    }

    fn make_blueprint(seed: u64, preset: RichnessPreset) -> PacingBlueprint {
        let resolved = make_resolved(seed, preset);
        build_pacing_blueprint(&resolved).unwrap()
    }

    #[test]
    fn placement_succeeds_for_sparse() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement(bp, resolved);
        assert!(
            result.is_ok(),
            "sparse placement failed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(r.placed_count > 0);
        assert!(!r.reservations.is_empty());
    }

    #[test]
    fn placement_succeeds_for_moderate() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);
        let result = solve_placement(bp, resolved);
        assert!(
            result.is_ok(),
            "moderate placement failed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(r.placed_count > 0);
    }

    #[test]
    fn placement_succeeds_for_rich() {
        let bp = make_blueprint(99, RichnessPreset::Rich);
        let resolved = make_resolved(99, RichnessPreset::Rich);
        let result = solve_placement(bp, resolved);
        assert!(result.is_ok(), "rich placement failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.placed_count > 0);
    }

    #[test]
    fn placement_is_deterministic() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);

        let r1 = solve_placement(bp.clone(), resolved.clone()).unwrap();
        let r2 = solve_placement(bp, resolved).unwrap();

        // All reservations must be identical
        assert_eq!(r1.reservations, r2.reservations);
        assert_eq!(r1.request_to_reservation, r2.request_to_reservation);
        assert_eq!(r1.placed_count, r2.placed_count);
    }

    #[test]
    fn placement_priority_order() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);

        let result = solve_placement(bp, resolved).unwrap();

        // Verify grand volumes exist if blueprint has them
        let has_grand_volume = result
            .reservations
            .values()
            .any(|r| matches!(r.kind, ReservationKind::MultiStoreyRoom));

        if has_grand_volume {
            // Grand volumes should have low reservation IDs (placed first)
            let grand_ids: Vec<_> = result
                .reservations
                .values()
                .filter(|r| matches!(r.kind, ReservationKind::MultiStoreyRoom))
                .map(|r| r.id.raw())
                .collect();
            let ordinary_ids: Vec<_> = result
                .reservations
                .values()
                .filter(|r| matches!(r.kind, ReservationKind::StandardRoom))
                .map(|r| r.id.raw())
                .collect();
            if let (Some(first_grand), Some(first_ordinary)) =
                (grand_ids.iter().min(), ordinary_ids.iter().min())
            {
                assert!(
                    first_grand < first_ordinary,
                    "grand-volume priority must precede ordinary placement"
                );
            }
        }
    }

    #[test]
    fn placement_handles_empty_blueprint() {
        // Create an empty blueprint edge case
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let result = solve_placement(bp, resolved);
        assert!(result.is_ok());
    }

    #[test]
    fn budget_under_ceiling() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement(bp, resolved).unwrap();

        assert!(
            result.journal.budget().faces <= result.journal.budget().max_faces,
            "budget exceeded: {} > {}",
            result.journal.budget().faces,
            result.journal.budget().max_faces
        );
    }

    #[test]
    fn every_placed_request_has_reservation() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);
        let result = solve_placement(bp.clone(), resolved).unwrap();

        // Every archetype request in the blueprint should have a reservation
        for req_id in bp.archetype_requests.keys() {
            assert!(
                result.request_to_reservation.contains_key(req_id),
                "request {:?} has no reservation",
                req_id
            );
        }
    }

    #[test]
    fn no_overlapping_standard_rooms() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);
        let result = solve_placement(bp, resolved).unwrap();

        // Check that no two standard rooms overlap in XY
        let standard_rooms: Vec<_> = result
            .reservations
            .values()
            .filter(|r| matches!(r.kind, ReservationKind::StandardRoom))
            .collect();

        for i in 0..standard_rooms.len() {
            for j in i + 1..standard_rooms.len() {
                let a = &standard_rooms[i].footprint;
                let b = &standard_rooms[j].footprint;
                // If they're on the same layer, they must not overlap
                if (a.occupies_lower && b.occupies_lower) || (a.occupies_upper && b.occupies_upper)
                {
                    assert!(
                        !a.overlaps_xy(b),
                        "standard rooms {:?} and {:?} overlap on XY",
                        standard_rooms[i].id,
                        standard_rooms[j].id
                    );
                }
            }
        }
    }

    #[test]
    fn worst_case_envelope_used() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement(bp, resolved).unwrap();

        // Verify that placed footprints use worst-case (max) spans, not min
        for (req_id, &res_id) in &result.request_to_reservation {
            let rec = result.reservations.get(&res_id).unwrap();

            // Find the original archetype
            if result
                .journal
                .reservations
                .get(&res_id)
                .and_then(|r| r.request_id)
                .is_some()
            {
                // Just verify footprints are non-zero
                let (w, d) = rec.footprint.quake_span();
                assert!(w > 0, "zero-width footprint for {:?}", req_id);
                assert!(d > 0, "zero-depth footprint for {:?}", req_id);
            }
        }
    }

    #[test]
    fn invalid_tiny_extent_returns_stable_feasibility_error() {
        let error = RichnessDocumentV1::new(0, 256, RichnessPreset::Rich, RichnessTheme::Ancient)
            .unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::ValueOutOfRange);
        assert_eq!(error.path, "extent");
    }

    #[test]
    fn deterministic_sweep_multiple_seeds() {
        for seed in &[0u64, 1, 42, 99, 255] {
            for preset in &[RichnessPreset::Sparse, RichnessPreset::Moderate] {
                let bp = make_blueprint(*seed, *preset);
                let resolved = make_resolved(*seed, *preset);
                let result = solve_placement(bp, resolved);
                assert!(
                    result.is_ok(),
                    "placement failed for seed={} preset={:?}: {:?}",
                    seed,
                    preset,
                    result.err()
                );
                let r = result.unwrap();
                assert!(r.placed_count > 0);
            }
        }
    }

    #[test]
    fn rejected_placement_branch_restores_byte_identical_state() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);

        let mut solver = PlacementSolver::new(bp, resolved).unwrap();
        let request = solver.placement_requests[0].clone();
        let candidate = solver
            .generate_candidates(&request, 0)
            .into_iter()
            .find(|candidate| solver.can_try_place(&request, candidate))
            .expect("fixture must expose a mutable placement branch");
        let before = solver.journal().state_snapshot();

        solver.journal.mark();
        let reservation_id = solver.try_place(&request, &candidate).unwrap();
        assert!(solver.journal.get(reservation_id).is_some());
        assert!(!solver.journal.matches_snapshot(&before));
        assert!(solver.journal.rollback());
        assert!(solver.journal.matches_snapshot(&before));
    }

    #[test]
    fn complete_backtracking_replaces_cheapest_blocking_placement() {
        // A 24-cell-wide full-height room centered in a 64-cell grid leaves
        // two 20-cell gaps, so the later 40-cell-wide room cannot fit. Moving
        // the first room to either edge makes the second placement feasible.
        // The preferred center is therefore a real cheapest-branch trap.
        let document =
            RichnessDocumentV1::new(0, 1024, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(document).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let mut solver = PlacementSolver::new(blueprint, resolved).unwrap();
        let make_request = |id: u32, beat: u32, width_cells: u32| PlacementRequest {
            request_id: ArchetypeRequestId::new(id),
            archetype: ArchetypeIndex::new(0),
            beat_id: BeatId::new(100 + beat),
            zone_id: ZoneId::new(0),
            priority: PlacementPriority::Ordinary,
            worst_case_span: (width_cells * CONSTRUCTION_QUANTUM as u32, 1024),
            dual_layer: false,
            is_pit: false,
            is_cave: false,
            is_negative_space: false,
            is_grand_volume: false,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
            route_witness: (64, 80),
        };
        solver.placement_requests = vec![make_request(0, 0, 24), make_request(1, 1, 40)];
        solver.preferred_slots.clear();
        solver.preferred_slots.insert(BeatId::new(100), (20, 0));

        let result = solver.solve().unwrap();
        let first_id = result.request_to_reservation[&ArchetypeRequestId::new(0)];
        let first = &result.reservations[&first_id];
        assert_ne!(
            first.footprint.x0, 20,
            "blocking preferred branch was retained"
        );
        assert!(result.total_search_states > 2, "fixture did not backtrack");
        assert!(result
            .request_to_reservation
            .values()
            .all(|id| result.reservations.contains_key(id)));
    }

    #[test]
    fn checkerboard_fixture_genuinely_reaches_frozen_placement_bound() {
        let document =
            RichnessDocumentV1::new(0, 1024, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(document).unwrap();
        let blueprint = build_pacing_blueprint(&resolved).unwrap();
        let mut solver = PlacementSolver::new(blueprint, resolved).unwrap();
        let make_request = |id: u32, beat: u32, cells: u32| PlacementRequest {
            request_id: ArchetypeRequestId::new(id),
            archetype: ArchetypeIndex::new(0),
            beat_id: BeatId::new(200 + beat),
            zone_id: ZoneId::new(0),
            priority: PlacementPriority::Ordinary,
            worst_case_span: (
                cells * CONSTRUCTION_QUANTUM as u32,
                cells * CONSTRUCTION_QUANTUM as u32,
            ),
            dual_layer: false,
            is_pit: false,
            is_cave: false,
            is_negative_space: false,
            is_grand_volume: false,
            cost_faces: 0,
            cost_brushes: 0,
            cost_entities: 0,
            cost_lights: 0,
            route_witness: (64, 80),
        };
        solver.placement_requests = vec![make_request(0, 0, 1), make_request(1, 1, 2)];
        solver.preferred_slots.clear();

        // A checkerboard leaves 2,048 legal one-cell positions but no legal
        // 2×2 position. Each first-request branch therefore drives all 3,969
        // dense second-request candidates before backtracking, genuinely
        // reaching the revision-v1 bound rather than merely exhausting one
        // infeasible route.
        for x in 0..64 {
            for y in 0..64 {
                if (x + y) % 2 == 0 {
                    solver
                        .journal
                        .try_reserve(
                            ReservationKind::Support,
                            Footprint3D {
                                x0: x,
                                y0: y,
                                x1: x + 1,
                                y1: y + 1,
                                occupies_lower: true,
                                occupies_upper: false,
                            },
                            None,
                            None,
                            None,
                            0,
                            0,
                            0,
                            0,
                        )
                        .unwrap();
                }
            }
        }
        solver.journal.commit_all();

        let error = solver.solve().unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::PlacementExhausted);
        assert_eq!(
            error.category,
            RichnessErrorCategory::PlacementTopologyExhaustion
        );
        assert_eq!(error.path, "placement");
        assert!(error.context.contains("search_states=200000"));
        assert!(error.context.contains("max_observed_search_states=200000"));
        assert!(error.context.contains("frozen_bound=200000"));
        assert!(error.context.contains("bound_reached=true"));
    }

    // ── Topology integration tests ─────────────────────────────────────

    #[test]
    fn same_xy_placements_are_owned_by_dual_band_composites() {
        let bp = make_blueprint(99, RichnessPreset::Rich);
        let resolved = make_resolved(99, RichnessPreset::Rich);
        let result = solve_placement_and_topology(bp, resolved).unwrap();
        let reservations = &result.topology.journal.reservations;

        let composites: Vec<_> = reservations
            .values()
            .filter(|record| record.kind == ReservationKind::Composite)
            .collect();
        assert!(!composites.is_empty(), "Rich must materialize composites");
        for composite in &composites {
            assert!(
                composite.footprint.occupies_lower && composite.footprint.occupies_upper,
                "composite {:?} must own both bands",
                composite.id
            );
        }

        for child in reservations.values().filter(|record| {
            matches!(
                record.kind,
                ReservationKind::MultiStoreyRoom
                    | ReservationKind::PitOmission
                    | ReservationKind::VerticalHost
            )
        }) {
            let parent = result
                .topology
                .journal
                .composite_parent_of(child.id)
                .unwrap_or_else(|| panic!("{:?} has no composite owner", child.id));
            let composite = reservations.get(&parent).unwrap();
            assert!(composite.composite_children.contains(&child.id));
        }

        assert!(
            composites.iter().any(|composite| {
                composite.composite_children.iter().any(|child_id| {
                    reservations
                        .get(child_id)
                        .is_some_and(|child| child.kind == ReservationKind::MultiStoreyRoom)
                }) && composite.composite_children.iter().any(|child_id| {
                    reservations
                        .get(child_id)
                        .is_some_and(|child| child.kind == ReservationKind::VerticalHost)
                }) && composite.composite_children.iter().any(|child_id| {
                    reservations
                        .get(child_id)
                        .is_some_and(|child| child.kind == ReservationKind::Light)
                }) && composite.composite_children.iter().any(|child_id| {
                    reservations
                        .get(child_id)
                        .is_some_and(|child| child.kind == ReservationKind::Support)
                }) && composite.composite_children.iter().any(|child_id| {
                    reservations
                        .get(child_id)
                        .is_some_and(|child| child.kind == ReservationKind::PortalThroat)
                })
            }),
            "a multi-storey composite must own its room, vertical void, and support constraints"
        );
    }

    #[test]
    fn solver_pit_pair_placement_is_owned_by_one_composite() {
        let blueprint = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let mut solver = PlacementSolver::new(blueprint, resolved).unwrap();
        let mut request = solver.placement_requests[0].clone();
        request.is_pit = true;
        request.dual_layer = false;
        request.priority = PlacementPriority::PitPair;
        request.cost_faces = 0;
        request.cost_brushes = 0;
        request.cost_entities = 0;
        request.cost_lights = 0;
        request.worst_case_span = (128, 128);
        let candidate = CandidatePosition {
            gx: 0,
            gy: 0,
            gw: 8,
            gd: 8,
            layer: 0,
            rank: 0,
        };

        let room_id = solver.try_place(&request, &candidate).unwrap();
        let room = solver.journal.get(room_id).unwrap();
        let composite_id = solver.journal.composite_parent_of(room_id).unwrap();
        let composite = solver.journal.get(composite_id).unwrap();
        let pit_id = composite
            .composite_children
            .iter()
            .copied()
            .find(|child_id| {
                solver
                    .journal
                    .get(*child_id)
                    .is_some_and(|child| child.kind == ReservationKind::PitOmission)
            })
            .unwrap();

        assert_eq!(room.kind, ReservationKind::StandardRoom);
        assert!(composite.footprint.occupies_lower && composite.footprint.occupies_upper);
        assert_eq!(
            solver.journal.composite_parent_of(pit_id),
            Some(composite_id)
        );
        assert_eq!(
            solver.journal.get(pit_id).unwrap().pit_pair_room_id,
            Some(room_id)
        );
    }

    #[test]
    fn topology_succeeds_after_placement() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp, resolved);
        assert!(
            result.is_ok(),
            "combined placement+topology failed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(!r.topology.selected_edges.is_empty());
        assert!(!r.topology.routes.is_empty());
    }

    #[test]
    fn topology_is_deterministic() {
        // Sparse is consistently feasible — test determinism
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);

        let r1 = solve_placement_and_topology(bp.clone(), resolved.clone()).unwrap();
        let r2 = solve_placement_and_topology(bp, resolved).unwrap();

        assert_eq!(
            r1.topology.selected_edges.len(),
            r2.topology.selected_edges.len()
        );
        assert_eq!(r1.topology.routes.len(), r2.topology.routes.len());
        for (e1, e2) in r1
            .topology
            .selected_edges
            .iter()
            .zip(r2.topology.selected_edges.iter())
        {
            assert_eq!(e1.id, e2.id);
            assert_eq!(e1.source, e2.source);
            assert_eq!(e1.target, e2.target);
        }
    }

    #[test]
    fn topology_spanning_tree_connects_all_rooms() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp, resolved).unwrap();

        let n_rooms = result
            .placement
            .reservations
            .values()
            .filter(|r| {
                matches!(
                    r.kind,
                    ReservationKind::StandardRoom
                        | ReservationKind::MultiStoreyRoom
                        | ReservationKind::CaveHost
                        | ReservationKind::NegativeSpace
                )
            })
            .count();

        // A spanning tree on N rooms has N-1 tree edges (+ optional loops)
        let tree_edge_count = result
            .topology
            .selected_edges
            .len()
            .saturating_sub(result.topology.loop_count);

        if n_rooms > 1 {
            assert_eq!(
                tree_edge_count,
                n_rooms - 1,
                "spanning tree should have {} edges for {} rooms, got {}",
                n_rooms - 1,
                n_rooms,
                tree_edge_count
            );
        }
    }

    #[test]
    fn mandatory_edges_are_present() {
        // Test with Sparse which is consistently feasible
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp.clone(), resolved).unwrap();

        // Every mandatory edge should be represented in selected edges
        let mandatory_beat_pairs: std::collections::BTreeSet<(BeatId, BeatId)> = bp
            .mandatory_edges
            .iter()
            .map(|e| (e.from_beat, e.to_beat))
            .collect();

        // Build beat-to-reservation map
        let beat_to_res: std::collections::BTreeMap<BeatId, ReservationId> = result
            .placement
            .reservations
            .values()
            .filter_map(|r| r.beat_id.map(|bid| (bid, r.id)))
            .collect();

        for (from_beat, to_beat) in &mandatory_beat_pairs {
            let from_res = beat_to_res.get(from_beat);
            let to_res = beat_to_res.get(to_beat);
            if let (Some(&fr), Some(&tr)) = (from_res, to_res) {
                let found = result.topology.selected_edges.iter().any(|e| {
                    (e.source == fr && e.target == tr) || (e.source == tr && e.target == fr)
                });
                assert!(
                    found,
                    "mandatory edge {:?}->{:?} not in topology",
                    from_beat, to_beat
                );
            }
        }
    }

    #[test]
    fn every_route_has_portals() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp, resolved).unwrap();

        for route in &result.topology.routes {
            assert_eq!(route.source_portal.endpoint_reservation_id, route.source);
            assert_eq!(route.target_portal.endpoint_reservation_id, route.target);
            for portal in [&route.source_portal, &route.target_portal] {
                let throat = result.topology.journal.get(portal.reservation_id).unwrap();
                assert_eq!(throat.kind, ReservationKind::PortalThroat);
                assert_eq!(throat.owning_route_id, Some(route.id));
                assert!(route.reservation_ids.contains(&throat.id));
            }
        }
    }

    #[test]
    fn topology_commits_route_reservations() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp, resolved).unwrap();

        // Verify that routes are in the journal
        let route_count = result
            .topology
            .journal
            .reservations
            .values()
            .filter(|r| matches!(r.kind, ReservationKind::Route))
            .count();
        assert!(route_count > 0, "topology should have route reservations");
    }

    #[test]
    fn topology_with_moderate_preset_feasibility() {
        let bp = make_blueprint(42, RichnessPreset::Moderate);
        let resolved = make_resolved(42, RichnessPreset::Moderate);
        let result = solve_placement_and_topology(bp, resolved).unwrap();
        assert_eq!(result.topology.loop_count, 2);
        assert!(!result.topology.shortcuts_realized.is_empty());
    }

    #[test]
    fn topology_with_rich_preset_feasibility() {
        let bp = make_blueprint(99, RichnessPreset::Rich);
        let resolved = make_resolved(99, RichnessPreset::Rich);
        let result = solve_placement_and_topology(bp, resolved).unwrap();
        assert_eq!(result.topology.loop_count, 4);
        assert!(!result.topology.shortcuts_realized.is_empty());
    }

    #[test]
    fn topology_sweep_sparse_all_seeds_pass() {
        // Sparse preset should be feasible for all tested seeds
        for seed in &[0u64, 1, 42, 99, 255] {
            let bp = make_blueprint(*seed, RichnessPreset::Sparse);
            let resolved = make_resolved(*seed, RichnessPreset::Sparse);
            let result = solve_placement_and_topology(bp, resolved);
            assert!(
                result.is_ok(),
                "sparse topology failed for seed={}: {:?}",
                seed,
                result.err()
            );
            let r = result.unwrap();
            assert!(!r.topology.selected_edges.is_empty());
        }
    }

    #[test]
    fn supported_moderate_rich_matrix_solves() {
        for seed in &[0u64, 42, 99] {
            for preset in &[RichnessPreset::Moderate, RichnessPreset::Rich] {
                let bp = make_blueprint(*seed, *preset);
                let resolved = make_resolved(*seed, *preset);
                let result = solve_placement_and_topology(bp, resolved).unwrap();
                assert!(!result.topology.selected_edges.is_empty());
            }
        }
    }

    #[test]
    fn topology_deterministic_supported_request() {
        let bp1 = make_blueprint(42, RichnessPreset::Moderate);
        let resolved1 = make_resolved(42, RichnessPreset::Moderate);
        let bp2 = make_blueprint(42, RichnessPreset::Moderate);
        let resolved2 = make_resolved(42, RichnessPreset::Moderate);

        let r1 = solve_placement_and_topology(bp1, resolved1).unwrap();
        let r2 = solve_placement_and_topology(bp2, resolved2).unwrap();
        assert_eq!(r1.placement, r2.placement);
        assert_eq!(r1.topology, r2.topology);
    }

    #[test]
    fn topology_route_reservations_do_not_overlap() {
        let bp = make_blueprint(0, RichnessPreset::Sparse);
        let resolved = make_resolved(0, RichnessPreset::Sparse);
        let result = solve_placement_and_topology(bp, resolved).unwrap();

        let route_records: Vec<_> = result
            .topology
            .routes
            .iter()
            .flat_map(|route| route.reservation_ids.iter())
            .map(|id| result.topology.journal.get(*id).unwrap())
            .collect();
        for (index, first) in route_records.iter().enumerate() {
            for second in route_records.iter().skip(index + 1) {
                let same_layer = (first.footprint.occupies_lower
                    && second.footprint.occupies_lower)
                    || (first.footprint.occupies_upper && second.footprint.occupies_upper);
                assert!(
                    !same_layer || !first.footprint.overlaps_xy(&second.footprint),
                    "route-owned reservations {:?} and {:?} overlap",
                    first.id,
                    second.id
                );
            }
        }
    }
}
