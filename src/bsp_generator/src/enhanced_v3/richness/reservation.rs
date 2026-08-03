//! Immutable reservation records and transactional reservation journal.
//!
//! Every cell in the occupancy grid is owned by exactly one reservation.
//! Reservations are immutable once committed; the journal supports
//! mark/rollback/commit over the full reservation, ID allocation, budget,
//! and candidate-state structures.
//!
//! # Contract
//!
//! - Late rejection restores byte-identical state.
//! - Budget tracking is integer-only (source faces, brushes, entities, lights).
//! - Composite reservations own same-XY bands explicitly.
//! - Pit pairs carry matched upper-floor omission + lower-floor room.

// Richness remains intentionally crate-private and pipeline-unwired until the
// atomic sealing phase; unit and matrix tests are its current callers.
#![allow(dead_code, clippy::result_large_err, clippy::too_many_arguments)]

use std::collections::BTreeMap;

use super::error::RichnessError;
use super::footprint::{Footprint3D, OccupancyGrid, OccupancyOwnerKind, HEADROOM};
use super::ids::{ArchetypeRequestId, BeatId, ReservationId, RouteId, ZoneId};

// ── Reservation kind ───────────────────────────────────────────────────────

/// The type of a reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ReservationKind {
    /// Standard single-storey room.
    StandardRoom,
    /// Multi-storey room spanning both layers.
    MultiStoreyRoom,
    /// Vertical host (stairwell, ladder shaft, spiral stair, drop hole).
    VerticalHost,
    /// Pit omission: empty volume on upper floor paired with committed lower
    /// room floor as pit bottom.
    PitOmission,
    /// Cave host (protected shell around a cave volume).
    CaveHost,
    /// Route corridor between rooms.
    Route,
    /// Portal throat at room boundary.
    PortalThroat,
    /// Turn / junction.
    Turn,
    /// Spawn point.
    Spawn,
    /// Light entity reservation.
    Light,
    /// Structural support.
    Support,
    /// Protected negative space.
    NegativeSpace,
    /// Composite owning both layers' bands.
    Composite,
}

impl ReservationKind {
    /// Convert to OccupancyOwnerKind.
    pub fn to_owner_kind(self) -> OccupancyOwnerKind {
        match self {
            Self::StandardRoom => OccupancyOwnerKind::StandardRoom,
            Self::MultiStoreyRoom => OccupancyOwnerKind::MultiStoreyRoom,
            Self::VerticalHost => OccupancyOwnerKind::VerticalHost,
            Self::PitOmission => OccupancyOwnerKind::PitOmission,
            Self::CaveHost => OccupancyOwnerKind::CaveHost,
            Self::Route => OccupancyOwnerKind::Route,
            Self::PortalThroat => OccupancyOwnerKind::PortalThroat,
            Self::Turn => OccupancyOwnerKind::Turn,
            Self::Spawn => OccupancyOwnerKind::Spawn,
            Self::Light => OccupancyOwnerKind::Light,
            Self::Support => OccupancyOwnerKind::Support,
            Self::NegativeSpace => OccupancyOwnerKind::NegativeSpace,
            Self::Composite => OccupancyOwnerKind::Composite,
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Self::StandardRoom => "standard_room",
            Self::MultiStoreyRoom => "multi_storey_room",
            Self::VerticalHost => "vertical_host",
            Self::PitOmission => "pit_omission",
            Self::CaveHost => "cave_host",
            Self::Route => "route",
            Self::PortalThroat => "portal_throat",
            Self::Turn => "turn",
            Self::Spawn => "spawn",
            Self::Light => "light",
            Self::Support => "support",
            Self::NegativeSpace => "negative_space",
            Self::Composite => "composite",
        }
    }
}

// ── Immutable reservation record ───────────────────────────────────────────

/// An immutable reservation record that owns cells in the occupancy grid.
///
/// Once committed, this record is immutable. It carries the owner ID,
/// the occupied footprint, links to the semantic blueprint (beat, request,
/// zone), and budget costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReservationRecord {
    /// Unique reservation ID.
    pub id: ReservationId,
    /// The kind of reservation.
    pub kind: ReservationKind,
    /// The footprint occupied by this reservation.
    pub footprint: Footprint3D,
    /// The beat this reservation belongs to (if any).
    pub beat_id: Option<BeatId>,
    /// The archetype request this fulfills (if any).
    pub request_id: Option<ArchetypeRequestId>,
    /// The zone this reservation is assigned to.
    pub zone_id: Option<ZoneId>,
    /// For pit omissions: the paired lower room reservation ID.
    pub pit_pair_room_id: Option<ReservationId>,
    /// For composite reservations: the child reservation IDs.
    pub composite_children: Vec<ReservationId>,
    /// Route that owns this witness reservation. Room/composite placement
    /// records are not route-owned.
    pub owning_route_id: Option<RouteId>,
    /// Protected vertical clearance carried by route-owned witnesses.
    pub clearance_height: Option<i32>,
    /// Whether this reservation is committed (immutable).
    pub committed: bool,
    /// Source face cost (estimate).
    pub cost_faces: u32,
    /// Brush cost (estimate).
    pub cost_brushes: u32,
    /// Entity cost (estimate).
    pub cost_entities: u32,
    /// Light cost (estimate).
    pub cost_lights: u32,
}

// ── Reservation journal ────────────────────────────────────────────────────

/// A transactional journal for reservation operations.
///
/// Tracks reservations, ID allocation, budget consumption, and occupancy
/// grid state. Supports mark/rollback/commit with byte-identical restoration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReservationJournal {
    /// All reservations (committed + pending).
    pub(crate) reservations: BTreeMap<ReservationId, ReservationRecord>,
    /// Next available reservation ID.
    next_reservation_id: u32,
    /// The occupancy grid.
    pub(crate) grid: OccupancyGrid,
    /// Budget tracker.
    budget: BudgetTracker,
    /// Checkpoint stack for transactional rollback.
    checkpoints: Vec<JournalCheckpoint>,
}

/// Budget tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BudgetTracker {
    /// Total source faces consumed.
    pub faces: u32,
    /// Total brushes consumed.
    pub brushes: u32,
    /// Total entities consumed.
    pub entities: u32,
    /// Total lights consumed.
    pub lights: u32,
    /// Maximum allowed face budget.
    pub max_faces: u32,
}

impl BudgetTracker {
    /// Create a new budget tracker with the given ceiling.
    pub fn new(max_faces: u32) -> Self {
        Self {
            faces: 0,
            brushes: 0,
            entities: 0,
            lights: 0,
            max_faces,
        }
    }

    /// Check if adding costs would exceed budget.
    pub fn can_afford(&self, faces: u32, _brushes: u32, _entities: u32, _lights: u32) -> bool {
        self.faces.saturating_add(faces) <= self.max_faces
    }

    /// Add costs to the budget.
    pub fn spend(&mut self, faces: u32, brushes: u32, entities: u32, lights: u32) {
        self.faces = self.faces.saturating_add(faces);
        self.brushes = self.brushes.saturating_add(brushes);
        self.entities = self.entities.saturating_add(entities);
        self.lights = self.lights.saturating_add(lights);
    }

    /// Whether the face budget is exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.faces > self.max_faces
    }
}

/// A stored checkpoint for rollback.
#[derive(Debug, Clone, PartialEq, Eq)]
struct JournalCheckpoint {
    /// Snapshot of all reservations.
    reservations: BTreeMap<ReservationId, ReservationRecord>,
    /// Next reservation ID.
    next_reservation_id: u32,
    /// Occupancy grid snapshot.
    grid_snapshot: super::footprint::OccupancyGridSnapshot,
    /// Budget snapshot.
    budget: BudgetTracker,
}

impl ReservationJournal {
    /// Create a new journal with the given extent and budget ceiling.
    pub fn new(extent: u32, max_faces: u32) -> Self {
        Self {
            reservations: BTreeMap::new(),
            next_reservation_id: 0,
            grid: OccupancyGrid::new(extent),
            budget: BudgetTracker::new(max_faces),
            checkpoints: Vec::new(),
        }
    }

    /// Clone the committed current state for an isolated feasibility probe.
    ///
    /// Search checkpoints describe how to return the live journal to ancestor
    /// states; a detached probe can never use them. Excluding that history
    /// keeps probe cost proportional to the current state instead of recursive
    /// search depth while preserving reservations, allocation, occupancy, and
    /// budget bytes exactly.
    pub fn detached_probe(&self) -> Self {
        let mut grid = self.grid.clone();
        grid.clear_checkpoints();
        Self {
            reservations: self.reservations.clone(),
            next_reservation_id: self.next_reservation_id,
            grid,
            budget: self.budget,
            checkpoints: Vec::new(),
        }
    }

    /// Allocate a new reservation ID.
    pub fn next_id(&mut self) -> ReservationId {
        let id = ReservationId::new(self.next_reservation_id);
        self.next_reservation_id += 1;
        id
    }

    /// Current reservation count.
    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    /// Get a reservation by ID.
    pub fn get(&self, id: ReservationId) -> Option<&ReservationRecord> {
        self.reservations.get(&id)
    }

    /// Read-only preflight for a footprint reservation. This does not consume
    /// an ID, mutate occupancy, or spend budget.
    pub fn can_reserve(
        &self,
        kind: ReservationKind,
        footprint: &Footprint3D,
        cost_faces: u32,
        cost_brushes: u32,
        cost_entities: u32,
        cost_lights: u32,
    ) -> bool {
        if matches!(
            kind,
            ReservationKind::Route
                | ReservationKind::PortalThroat
                | ReservationKind::Turn
                | ReservationKind::MultiStoreyRoom
                | ReservationKind::VerticalHost
                | ReservationKind::PitOmission
        ) {
            return false;
        }
        self.budget
            .can_afford(cost_faces, cost_brushes, cost_entities, cost_lights)
            && self
                .grid
                .can_reserve(
                    footprint,
                    ReservationId::new(self.next_reservation_id),
                    kind.to_owner_kind(),
                )
                .is_ok()
    }

    /// Attempt to reserve a footprint.
    ///
    /// Returns `Ok(id)` on success (the reservation is recorded but not yet
    /// committed). Returns `Err` if occupancy conflicts or budget exceeded.
    pub fn try_reserve(
        &mut self,
        kind: ReservationKind,
        footprint: Footprint3D,
        beat_id: Option<BeatId>,
        request_id: Option<ArchetypeRequestId>,
        zone_id: Option<ZoneId>,
        cost_faces: u32,
        cost_brushes: u32,
        cost_entities: u32,
        cost_lights: u32,
    ) -> Result<ReservationId, RichnessError> {
        self.try_reserve_with_owner(
            kind,
            footprint,
            beat_id,
            request_id,
            zone_id,
            None,
            None,
            cost_faces,
            cost_brushes,
            cost_entities,
            cost_lights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_reserve_with_owner(
        &mut self,
        kind: ReservationKind,
        footprint: Footprint3D,
        beat_id: Option<BeatId>,
        request_id: Option<ArchetypeRequestId>,
        zone_id: Option<ZoneId>,
        composite_parent: Option<ReservationId>,
        owning_route_id: Option<RouteId>,
        cost_faces: u32,
        cost_brushes: u32,
        cost_entities: u32,
        cost_lights: u32,
    ) -> Result<ReservationId, RichnessError> {
        let is_route_owned_kind = matches!(
            kind,
            ReservationKind::Route | ReservationKind::PortalThroat | ReservationKind::Turn
        );
        if is_route_owned_kind != owning_route_id.is_some() {
            return Err(RichnessError::new(
                super::error::RichnessErrorCode::TopologyExhausted,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "reservation.route_owner",
                super::error::RichnessErrorCategory::PlacementTopologyExhaustion,
                format!(
                    "reservation kind {} requires exactly one owning route binding",
                    kind.tag()
                ),
            ));
        }

        let requires_composite_parent = matches!(
            kind,
            ReservationKind::MultiStoreyRoom
                | ReservationKind::VerticalHost
                | ReservationKind::PitOmission
        );
        let intersecting_composite = (!is_route_owned_kind)
            .then(|| {
                self.reservations
                    .values()
                    .find(|record| {
                        record.kind == ReservationKind::Composite
                            && record.footprint.overlaps_xy(&footprint)
                    })
                    .map(|record| record.id)
            })
            .flatten();
        if (requires_composite_parent && composite_parent.is_none())
            || (kind == ReservationKind::Composite && composite_parent.is_some())
            || (!is_route_owned_kind
                && composite_parent.is_none()
                && intersecting_composite.is_some())
        {
            return Err(Self::composite_owner_error(kind));
        }
        if let Some(parent_id) = composite_parent {
            let Some(parent) = self.reservations.get(&parent_id) else {
                return Err(Self::composite_owner_error(kind));
            };
            if parent.kind != ReservationKind::Composite
                || footprint.x0 < parent.footprint.x0
                || footprint.y0 < parent.footprint.y0
                || footprint.x1 > parent.footprint.x1
                || footprint.y1 > parent.footprint.y1
            {
                return Err(Self::composite_owner_error(kind));
            }
        }

        // Reserve the allocator slot only after every non-mutating preflight
        // succeeds. A rejected operation is therefore failure-atomic even
        // when the caller has not opened a surrounding transaction.
        let id = ReservationId::new(self.next_reservation_id);

        // Check budget
        if !self
            .budget
            .can_afford(cost_faces, cost_brushes, cost_entities, cost_lights)
        {
            return Err(RichnessError::new(
                super::error::RichnessErrorCode::BudgetOverrun,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "reservation",
                super::error::RichnessErrorCategory::BudgetOverrun,
                format!(
                    "budget overrun: faces={}+{} > {}",
                    self.budget.faces, cost_faces, self.budget.max_faces
                ),
            ));
        }

        // Check occupancy
        let owner_kind = kind.to_owner_kind();
        self.grid
            .can_reserve(&footprint, id, owner_kind)
            .map_err(|e| {
                RichnessError::new(
                    super::error::RichnessErrorCode::PlacementExhausted,
                    0,
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "reservation",
                    super::error::RichnessErrorCategory::PlacementTopologyExhaustion,
                    format!("occupancy conflict: {:?}", e),
                )
            })?;

        self.next_reservation_id = self.next_reservation_id.saturating_add(1);

        // Reserve cells in grid
        self.grid.reserve(&footprint, id, owner_kind);

        // Spend budget
        self.budget
            .spend(cost_faces, cost_brushes, cost_entities, cost_lights);

        // Create reservation record
        let record = ReservationRecord {
            id,
            kind,
            footprint,
            beat_id,
            request_id,
            zone_id,
            pit_pair_room_id: None,
            composite_children: Vec::new(),
            owning_route_id,
            clearance_height: owning_route_id.map(|_| HEADROOM),
            committed: false,
            cost_faces,
            cost_brushes,
            cost_entities,
            cost_lights,
        };
        self.reservations.insert(id, record);
        if let Some(parent_id) = composite_parent {
            self.add_composite_child(parent_id, id);
        }

        Ok(id)
    }

    fn composite_owner_error(kind: ReservationKind) -> RichnessError {
        RichnessError::new(
            super::error::RichnessErrorCode::PlacementExhausted,
            0,
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            "reservation.composite_owner",
            super::error::RichnessErrorCategory::PlacementTopologyExhaustion,
            format!("{} requires an explicit composite owner", kind.tag()),
        )
    }

    /// Reserve a route witness and bind it to its owning committed-route ID.
    /// The binding is part of the journal checkpoint and therefore rolls back
    /// atomically with occupancy, reservation IDs, and budget state.
    pub fn try_reserve_for_route(
        &mut self,
        route_id: RouteId,
        kind: ReservationKind,
        footprint: Footprint3D,
        cost_faces: u32,
        cost_brushes: u32,
        cost_entities: u32,
        cost_lights: u32,
    ) -> Result<ReservationId, RichnessError> {
        self.try_reserve_with_owner(
            kind,
            footprint,
            None,
            None,
            None,
            None,
            Some(route_id),
            cost_faces,
            cost_brushes,
            cost_entities,
            cost_lights,
        )
    }

    /// Create a composite reservation owning both layers at the given footprint.
    pub fn reserve_composite(
        &mut self,
        footprint: Footprint3D,
        beat_id: Option<BeatId>,
        request_id: Option<ArchetypeRequestId>,
        zone_id: Option<ZoneId>,
    ) -> Result<ReservationId, RichnessError> {
        // Force dual-layer
        let dual = Footprint3D {
            occupies_lower: true,
            occupies_upper: true,
            ..footprint
        };
        self.try_reserve(
            ReservationKind::Composite,
            dual,
            beat_id,
            request_id,
            zone_id,
            0,
            0,
            0,
            0, // composites don't consume budget directly
        )
    }

    /// Reserve a child within an explicit dual-band composite container.
    ///
    /// This is the only placement path for multi-storey rooms, pit omissions,
    /// and vertical hosts. It atomically records the parent-child ownership
    /// after the footprint reservation succeeds.
    #[allow(clippy::too_many_arguments)]
    pub fn try_reserve_composite_child(
        &mut self,
        composite_id: ReservationId,
        kind: ReservationKind,
        footprint: Footprint3D,
        beat_id: Option<BeatId>,
        request_id: Option<ArchetypeRequestId>,
        zone_id: Option<ZoneId>,
        cost_faces: u32,
        cost_brushes: u32,
        cost_entities: u32,
        cost_lights: u32,
    ) -> Result<ReservationId, RichnessError> {
        self.try_reserve_with_owner(
            kind,
            footprint,
            beat_id,
            request_id,
            zone_id,
            Some(composite_id),
            None,
            cost_faces,
            cost_brushes,
            cost_entities,
            cost_lights,
        )
    }

    /// Add a child to a composite reservation.
    pub fn add_composite_child(&mut self, composite_id: ReservationId, child_id: ReservationId) {
        if let Some(record) = self.reservations.get_mut(&composite_id) {
            if record.kind == ReservationKind::Composite
                && !record.composite_children.contains(&child_id)
            {
                record.composite_children.push(child_id);
                record.composite_children.sort_unstable();
            }
        }
    }

    /// Return the explicit composite container for a child reservation.
    pub fn composite_parent_of(&self, child_id: ReservationId) -> Option<ReservationId> {
        self.reservations
            .values()
            .find(|record| {
                record.kind == ReservationKind::Composite
                    && record.composite_children.contains(&child_id)
            })
            .map(|record| record.id)
    }

    /// Link a pit omission to its paired lower room.
    pub fn link_pit_pair(&mut self, pit_id: ReservationId, room_id: ReservationId) {
        if let Some(record) = self.reservations.get_mut(&pit_id) {
            record.pit_pair_room_id = Some(room_id);
        }
    }

    /// Commit all pending reservations (they become immutable).
    pub fn commit_all(&mut self) {
        for record in self.reservations.values_mut() {
            record.committed = true;
        }
    }

    /// Release a pending reservation and its cells.
    ///
    /// Committed records are immutable. Transactional callers restore a
    /// checkpoint instead of deleting committed reservations.
    pub fn release(&mut self, id: ReservationId) {
        if self
            .reservations
            .get(&id)
            .is_some_and(|record| record.committed)
        {
            return;
        }
        if let Some(record) = self.reservations.remove(&id) {
            self.grid.release(id);
            // Reverse budget
            self.budget.faces = self.budget.faces.saturating_sub(record.cost_faces);
            self.budget.brushes = self.budget.brushes.saturating_sub(record.cost_brushes);
            self.budget.entities = self.budget.entities.saturating_sub(record.cost_entities);
            self.budget.lights = self.budget.lights.saturating_sub(record.cost_lights);
        }
    }

    /// Mark a checkpoint for transactional rollback.
    pub fn mark(&mut self) {
        // The journal owns the complete checkpoint.  Do not maintain a
        // second grid checkpoint: restoring the captured grid snapshot must
        // also discard the speculative grid history, otherwise nested
        // commits can make a later rollback target the wrong state.
        self.checkpoints.push(JournalCheckpoint {
            reservations: self.reservations.clone(),
            next_reservation_id: self.next_reservation_id,
            grid_snapshot: self.grid.state_snapshot(),
            budget: self.budget,
        });
    }

    /// Rollback to the most recent checkpoint.
    ///
    /// Restores byte-identical state of reservations, IDs, budget, and
    /// occupancy grid.
    pub fn rollback(&mut self) -> bool {
        let Some(cp) = self.checkpoints.pop() else {
            return false;
        };
        self.reservations = cp.reservations;
        self.next_reservation_id = cp.next_reservation_id;
        self.grid.restore_snapshot(&cp.grid_snapshot);
        self.budget = cp.budget;
        true
    }

    /// Commit the most recent checkpoint by discarding its rollback state.
    pub fn commit(&mut self) -> bool {
        self.checkpoints.pop().is_some()
    }

    /// Discard all checkpoints.
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
        self.grid.clear_checkpoints();
    }

    /// Current budget tracker.
    pub fn budget(&self) -> &BudgetTracker {
        &self.budget
    }

    /// Total budget ceiling.
    pub fn max_faces(&self) -> u32 {
        self.budget.max_faces
    }

    /// Full state snapshot for byte-identity comparisons.
    pub fn state_snapshot(&self) -> JournalSnapshot {
        JournalSnapshot {
            reservations: self.reservations.clone(),
            next_reservation_id: self.next_reservation_id,
            grid_snapshot: self.grid.state_snapshot(),
            budget: self.budget,
            checkpoints: self.checkpoints.clone(),
        }
    }

    /// Verify that the journal state matches a snapshot.
    pub fn matches_snapshot(&self, snapshot: &JournalSnapshot) -> bool {
        self.reservations == snapshot.reservations
            && self.next_reservation_id == snapshot.next_reservation_id
            && self.grid.state_snapshot() == snapshot.grid_snapshot
            && self.budget == snapshot.budget
            && self.checkpoints == snapshot.checkpoints
    }

    /// Whether the budget is currently exceeded.
    pub fn budget_exceeded(&self) -> bool {
        self.budget.is_exceeded()
    }
}

/// Full journal snapshot for byte-identity testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JournalSnapshot {
    pub reservations: BTreeMap<ReservationId, ReservationRecord>,
    pub next_reservation_id: u32,
    pub grid_snapshot: super::footprint::OccupancyGridSnapshot,
    pub budget: BudgetTracker,
    checkpoints: Vec<JournalCheckpoint>,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_create_and_reserve() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let result = journal.try_reserve(
            ReservationKind::StandardRoom,
            fp,
            None,
            None,
            None,
            200,
            12,
            4,
            3,
        );
        assert!(result.is_ok());
        let id = result.unwrap();
        assert_eq!(journal.reservation_count(), 1);
        let rec = journal.get(id).unwrap();
        assert_eq!(rec.kind, ReservationKind::StandardRoom);
        assert!(!rec.committed);
    }

    #[test]
    fn route_witness_kinds_require_an_owner_binding() {
        for kind in [
            ReservationKind::Route,
            ReservationKind::PortalThroat,
            ReservationKind::Turn,
        ] {
            let mut journal = ReservationJournal::new(1024, 3000);
            let before = journal.state_snapshot();
            let result = journal.try_reserve(
                kind,
                Footprint3D::single_layer(0, 0, 64, 64, 0),
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            );
            assert!(result.is_err());
            assert!(journal.matches_snapshot(&before));
        }
    }

    #[test]
    fn journal_budget_enforcement() {
        let mut journal = ReservationJournal::new(1024, 200);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let before = journal.state_snapshot();
        let result = journal.try_reserve(
            ReservationKind::StandardRoom,
            fp,
            None,
            None,
            None,
            201,
            10,
            5,
            2,
        );
        assert!(result.is_err());
        assert!(journal.matches_snapshot(&before));
        assert_eq!(journal.next_id(), ReservationId::new(0));
    }

    #[test]
    fn journal_transactional_rollback_full_state() {
        let mut journal = ReservationJournal::new(1024, 3000);

        // Snapshot before
        let before = journal.state_snapshot();

        // Mark and make changes
        journal.mark();
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        journal
            .try_reserve(
                ReservationKind::StandardRoom,
                fp,
                None,
                None,
                None,
                200,
                12,
                4,
                3,
            )
            .unwrap();

        assert_eq!(journal.reservation_count(), 1);

        // Rollback
        let rolled = journal.rollback();
        assert!(rolled, "rollback should succeed");

        // Verify byte-identical state
        let after = journal.state_snapshot();
        assert_eq!(before, after, "rollback must restore byte-identical state");
        assert_eq!(journal.reservation_count(), 0);
    }

    #[test]
    fn late_rejection_rolls_back_all_allocations_byte_identically() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let before = journal.state_snapshot();
        journal.mark();
        journal
            .try_reserve_for_route(
                RouteId::new(0),
                ReservationKind::Route,
                Footprint3D::single_layer(0, 0, 64, 64, 0),
                100,
                8,
                0,
                0,
            )
            .unwrap();
        journal
            .try_reserve_for_route(
                RouteId::new(0),
                ReservationKind::Turn,
                Footprint3D::single_layer(64, 0, 96, 32, 0),
                50,
                4,
                0,
                0,
            )
            .unwrap();
        assert!(journal
            .try_reserve(
                ReservationKind::StandardRoom,
                Footprint3D::single_layer(32, 16, 80, 48, 0),
                None,
                None,
                None,
                200,
                12,
                2,
                1,
            )
            .is_err());
        assert!(journal.rollback());
        assert_eq!(journal.state_snapshot(), before);
    }

    #[test]
    fn journal_commit_all() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let id = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                fp,
                None,
                None,
                None,
                200,
                12,
                4,
                3,
            )
            .unwrap();

        journal.commit_all();
        let rec = journal.get(id).unwrap();
        assert!(rec.committed);
    }

    #[test]
    fn journal_release_frees_budget() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let id = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                fp,
                None,
                None,
                None,
                200,
                12,
                4,
                3,
            )
            .unwrap();

        let budget_before = journal.budget().faces;
        assert!(budget_before > 0);

        journal.release(id);
        assert_eq!(journal.budget().faces, 0);
        assert_eq!(journal.reservation_count(), 0);
    }

    #[test]
    fn committed_reservations_are_immutable() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let id = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                Footprint3D::single_layer(0, 0, 64, 64, 0),
                None,
                None,
                None,
                100,
                6,
                0,
                0,
            )
            .unwrap();
        journal.commit_all();
        let snapshot = journal.state_snapshot();

        journal.release(id);

        assert!(journal.matches_snapshot(&snapshot));
        assert!(journal.get(id).is_some());
    }

    #[test]
    fn composite_reservation_owns_both_layers() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let fp = Footprint3D::single_layer(0, 0, 128, 128, 0); // only lower specified
        let result = journal.reserve_composite(fp, None, None, None);
        assert!(result.is_ok());
        let id = result.unwrap();
        let rec = journal.get(id).unwrap();
        assert_eq!(rec.kind, ReservationKind::Composite);
        assert!(rec.footprint.occupies_lower);
        assert!(rec.footprint.occupies_upper);
    }

    #[test]
    fn pit_pair_linking_requires_and_records_a_composite_owner() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let room_fp = Footprint3D::single_layer(0, 0, 64, 64, 0);
        let pit_fp = Footprint3D::single_layer(0, 0, 64, 64, 1);
        let composite_id = journal
            .reserve_composite(room_fp, None, None, None)
            .unwrap();

        let room_id = journal
            .try_reserve_composite_child(
                composite_id,
                ReservationKind::StandardRoom,
                room_fp,
                None,
                None,
                None,
                200,
                12,
                4,
                3,
            )
            .unwrap();
        let pit_id = journal
            .try_reserve_composite_child(
                composite_id,
                ReservationKind::PitOmission,
                pit_fp,
                None,
                None,
                None,
                50,
                4,
                1,
                0,
            )
            .unwrap();

        journal.link_pit_pair(pit_id, room_id);
        let composite = journal.get(composite_id).unwrap();
        assert!(composite.footprint.occupies_lower && composite.footprint.occupies_upper);
        assert_eq!(composite.composite_children, vec![room_id, pit_id]);
        assert_eq!(journal.get(pit_id).unwrap().pit_pair_room_id, Some(room_id));
    }

    #[test]
    fn same_xy_specializations_require_an_explicit_composite_owner() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let dual = Footprint3D::dual_layer(0, 0, 64, 64);
        let upper = Footprint3D::single_layer(0, 0, 64, 64, 1);

        for (kind, footprint) in [
            (ReservationKind::MultiStoreyRoom, dual),
            (ReservationKind::VerticalHost, dual),
            (ReservationKind::PitOmission, upper),
        ] {
            let before = journal.state_snapshot();
            let error = journal
                .try_reserve(kind, footprint, None, None, None, 0, 0, 0, 0)
                .unwrap_err();
            assert_eq!(error.path, "reservation.composite_owner");
            assert!(journal.matches_snapshot(&before));
        }
    }

    #[test]
    fn reservation_kind_to_owner_kind_roundtrip() {
        // Every ReservationKind must map to a corresponding OccupancyOwnerKind
        let kinds = [
            ReservationKind::StandardRoom,
            ReservationKind::MultiStoreyRoom,
            ReservationKind::VerticalHost,
            ReservationKind::PitOmission,
            ReservationKind::CaveHost,
            ReservationKind::Route,
            ReservationKind::PortalThroat,
            ReservationKind::Turn,
            ReservationKind::Spawn,
            ReservationKind::Light,
            ReservationKind::Support,
            ReservationKind::NegativeSpace,
            ReservationKind::Composite,
        ];
        for kind in &kinds {
            let _ = kind.to_owner_kind();
            let _ = kind.tag();
        }
    }

    #[test]
    fn nested_checkpoints_rollback_correctly() {
        let mut journal = ReservationJournal::new(1024, 3000);

        journal.mark(); // cp0
        let _r0 = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                Footprint3D::single_layer(0, 0, 32, 32, 0),
                None,
                None,
                None,
                100,
                6,
                2,
                1,
            )
            .unwrap();

        journal.mark(); // cp1
        let _r1 = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                Footprint3D::single_layer(64, 64, 96, 96, 0),
                None,
                None,
                None,
                100,
                6,
                2,
                1,
            )
            .unwrap();
        assert_eq!(journal.reservation_count(), 2);

        // Rollback cp1
        assert!(journal.rollback());
        assert_eq!(journal.reservation_count(), 1);

        // Rollback cp0
        assert!(journal.rollback());
        assert_eq!(journal.reservation_count(), 0);
    }

    #[test]
    fn snapshot_matches_detects_differences() {
        let mut journal = ReservationJournal::new(1024, 3000);
        let snap = journal.state_snapshot();
        assert!(journal.matches_snapshot(&snap));

        let _r = journal
            .try_reserve(
                ReservationKind::StandardRoom,
                Footprint3D::single_layer(0, 0, 32, 32, 0),
                None,
                None,
                None,
                100,
                6,
                2,
                1,
            )
            .unwrap();
        assert!(!journal.matches_snapshot(&snap));
    }
}
