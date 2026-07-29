//! Enhanced v2 reservation system — transactional ownership of cells, sockets,
//! routes, and transitions with full mark/rollback/commit semantics.
//!
//! The [`Transaction`] wraps an [`OccupancyGrid`] and extends it with:
//! - Socket claims (exclusive ownership per socket)
//! - Route and transition records
//! - ID allocation
//! - Loop budget tracking
//!
//! Every mutation goes through the transaction. A [`TransactionMark`] captures
//! the complete observable state; `rollback()` restores it atomically.

use std::collections::BTreeMap;

use super::error::EnhancedError;
use super::intent::{IdAllocator, RouteId, RouteIntent, SocketId, TransitionId, TransitionIntent};
use super::occupancy::{OccupancyGrid, Owner};

const Q_U: u32 = crate::config::CONSTRUCTION_QUANTUM;

// ── Ownership kinds ────────────────────────────────────────────────────────

/// Which kind of entity owns a socket claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OwnerKind {
    /// Claimed by a horizontal route.
    Route(RouteId),
    /// Claimed by a vertical stair transition.
    Transition(TransitionId),
}

// ── Transaction mark ───────────────────────────────────────────────────────

/// A complete snapshot of transaction state for rollback.
#[derive(Debug, Clone)]
pub struct TransactionMark {
    grid_cp: super::occupancy::GridCheckpoint,
    socket_claims: BTreeMap<SocketId, OwnerKind>,
    routes: Vec<RouteIntent>,
    transitions: Vec<TransitionIntent>,
    alloc: IdAllocator,
    loop_budget_remaining: u32,
}

// ── Transaction ────────────────────────────────────────────────────────────

/// Mutable transaction wrapping all Phase 04 state.
///
/// All route/transition materialization goes through this transaction.
/// Marks support bounded canonical backtracking: when a mandatory commitment
/// fails, roll back to the previous mark and try the next alternative.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub grid: OccupancyGrid,
    socket_claims: BTreeMap<SocketId, OwnerKind>,
    routes: Vec<RouteIntent>,
    transitions: Vec<TransitionIntent>,
    pub alloc: IdAllocator,
    loop_budget_remaining: u32,
}

impl Transaction {
    /// Create a new transaction from a post-placement occupancy grid.
    ///
    /// The grid already contains room reservations; the transaction adds
    /// route and transition reservations on top.
    pub fn new(grid: OccupancyGrid, alloc: IdAllocator, loop_budget: u32) -> Self {
        Self {
            grid,
            socket_claims: BTreeMap::new(),
            routes: Vec::new(),
            transitions: Vec::new(),
            alloc,
            loop_budget_remaining: loop_budget,
        }
    }

    // ── Mark / rollback / commit ────────────────────────────────────────

    /// Capture a mark representing the current state.
    pub fn mark(&self) -> TransactionMark {
        TransactionMark {
            grid_cp: self.grid.checkpoint(),
            socket_claims: self.socket_claims.clone(),
            routes: self.routes.clone(),
            transitions: self.transitions.clone(),
            alloc: self.alloc.clone(),
            loop_budget_remaining: self.loop_budget_remaining,
        }
    }

    /// Restore all state to a previously captured mark.
    pub fn rollback(&mut self, mark: TransactionMark) {
        self.grid.restore(mark.grid_cp);
        self.socket_claims = mark.socket_claims;
        self.routes = mark.routes;
        self.transitions = mark.transitions;
        self.alloc = mark.alloc;
        self.loop_budget_remaining = mark.loop_budget_remaining;
    }

    /// Discard rollback state and return the committed records.
    pub fn commit(self) -> CommittedState {
        CommittedState {
            routes: self.routes,
            transitions: self.transitions,
            socket_claims: self.socket_claims,
        }
    }

    // ── Socket claims ───────────────────────────────────────────────────

    /// Check whether a socket is already claimed.
    pub fn socket_is_claimed(&self, socket: SocketId) -> bool {
        self.socket_claims.contains_key(&socket)
    }

    /// Try to claim a socket. Returns an error if already claimed.
    pub fn claim_socket(
        &mut self,
        socket: SocketId,
        owner: OwnerKind,
    ) -> Result<(), EnhancedError> {
        if let Some(existing) = self.socket_claims.get(&socket) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "socket {:?} already claimed by {:?}, cannot claim for {:?}",
                    socket, existing, owner,
                ),
            });
        }
        self.socket_claims.insert(socket, owner);
        Ok(())
    }

    /// Claim both sockets for a route (atomic: both or neither).
    pub fn claim_route_sockets(
        &mut self,
        source: SocketId,
        target: SocketId,
        route: RouteId,
    ) -> Result<(), EnhancedError> {
        // Check both first
        if self.socket_is_claimed(source) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "source socket {:?} already claimed for route {:?}",
                    source, route,
                ),
            });
        }
        if self.socket_is_claimed(target) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "target socket {:?} already claimed for route {:?}",
                    target, route,
                ),
            });
        }
        self.socket_claims.insert(source, OwnerKind::Route(route));
        self.socket_claims.insert(target, OwnerKind::Route(route));
        Ok(())
    }

    /// Claim both sockets for a transition (atomic: both or neither).
    pub fn claim_transition_sockets(
        &mut self,
        lower: SocketId,
        upper: SocketId,
        transition: TransitionId,
    ) -> Result<(), EnhancedError> {
        if self.socket_is_claimed(lower) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "lower socket {:?} already claimed for transition {:?}",
                    lower, transition,
                ),
            });
        }
        if self.socket_is_claimed(upper) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "upper socket {:?} already claimed for transition {:?}",
                    upper, transition,
                ),
            });
        }
        self.socket_claims
            .insert(lower, OwnerKind::Transition(transition));
        self.socket_claims
            .insert(upper, OwnerKind::Transition(transition));
        Ok(())
    }

    // ── Grid reservations ───────────────────────────────────────────────

    /// Reserve a rectangular region in the occupancy grid for a route.
    pub fn reserve_route_rect(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        route: RouteId,
    ) -> Result<(), EnhancedError> {
        self.grid
            .reserve_rect_owner(x0, y0, w, h, Owner::Route(route))
    }

    /// Reserve a rectangular region for a route, allowing overlap only with
    /// its two endpoint rooms or an explicit horizontal junction. A corridor
    /// may meet another route only as a shared junction; transitions and
    /// unrelated rooms always remain exclusive.
    pub fn reserve_route_rect_allow_rooms(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        route: RouteId,
        _allowed_rooms: &[super::intent::RoomId],
    ) -> Result<(), EnhancedError> {
        // Use a two-phase approach: first check, then reserve.
        use super::occupancy::Owner;
        let q = crate::config::CONSTRUCTION_QUANTUM as i32;
        let (qx0, qy0, qw, qh) = {
            if x0 < 0 || y0 < 0 || w <= 0 || h <= 0 {
                return Err(EnhancedError::ContractViolation {
                    detail: format!("invalid rect: ({}, {}) {}×{}", x0, y0, w, h),
                });
            }
            if x0 % q != 0 || y0 % q != 0 || w % q != 0 || h % q != 0 {
                return Err(EnhancedError::ContractViolation {
                    detail: format!(
                        "rect must be quantum-aligned: ({}, {}) {}×{} (quantum {})",
                        x0, y0, w, h, q,
                    ),
                });
            }
            (
                (x0 as u32) / (q as u32),
                (y0 as u32) / (q as u32),
                (w as u32) / (q as u32),
                (h as u32) / (q as u32),
            )
        };

        let qx1 = qx0 + qw;
        let qy1 = qy0 + qh;

        let cells_x = self.grid.cells_x();
        let cells_y = self.grid.cells_y();

        if qx1 > cells_x || qy1 > cells_y {
            return Err(EnhancedError::ContractViolation {
                detail: format!("rect exceeds grid bounds"),
            });
        }

        // We need to check cells manually since we have special allow rules.
        // First pass: check
        for py in qy0..qy1 {
            for px in qx0..qx1 {
                let idx = (cells_x as usize) * (py as usize) + (px as usize);
                match self.grid.cells()[idx] {
                    Owner::Empty => {}
                    // Explicit junction sharing remains a route-owned
                    // reservation for both records; transitions never share.
                    Owner::Route(_) => {}
                    Owner::Room(room) if _allowed_rooms.contains(&room) => {}
                    owner => {
                        return Err(EnhancedError::ContractViolation {
                            detail: format!(
                                "route {:?} conflicts with {:?} at ({}, {})",
                                route,
                                owner,
                                px * Q_U,
                                py * Q_U
                            ),
                        })
                    }
                }
            }
        }

        // Keep endpoint room cells owned by their room. Socket claims and the
        // route record own the aperture/throat; replacing the room ownership
        // would make a later legal socket on that same room appear blocked.
        for py in qy0..qy1 {
            for px in qx0..qx1 {
                let idx = (cells_x as usize) * (py as usize) + (px as usize);
                if matches!(self.grid.cells()[idx], Owner::Room(room) if _allowed_rooms.contains(&room))
                {
                    continue;
                }
                self.grid.cells_mut()[idx] = Owner::Route(route);
            }
        }

        Ok(())
    }

    /// Reserve a rectangular region for a transition. This strict variant
    /// accepts only empty cells (or cells already owned by that transition).
    pub fn reserve_transition_rect(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        transition: TransitionId,
    ) -> Result<(), EnhancedError> {
        self.reserve_transition_rect_allow_rooms(x0, y0, w, h, transition, &[])
    }

    /// Reserve a transition footprint while permitting only its direct host
    /// rooms at its apertures. All unrelated projected ownership conflicts.
    pub fn reserve_transition_rect_allow_rooms(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        transition: TransitionId,
        allowed_rooms: &[super::intent::RoomId],
    ) -> Result<(), EnhancedError> {
        let q = crate::config::CONSTRUCTION_QUANTUM as i32;
        if x0 < 0
            || y0 < 0
            || w <= 0
            || h <= 0
            || x0 % q != 0
            || y0 % q != 0
            || w % q != 0
            || h % q != 0
        {
            return Err(EnhancedError::ContractViolation {
                detail: "invalid transition reservation rectangle".into(),
            });
        }
        let qx0 = x0 as u32 / Q_U;
        let qy0 = y0 as u32 / Q_U;
        let qx1 = qx0 + w as u32 / Q_U;
        let qy1 = qy0 + h as u32 / Q_U;
        if qx1 > self.grid.cells_x() || qy1 > self.grid.cells_y() {
            return Err(EnhancedError::ContractViolation {
                detail: "transition footprint exceeds grid bounds".into(),
            });
        }
        for py in qy0..qy1 {
            for px in qx0..qx1 {
                let idx = self.grid.cells_x() as usize * py as usize + px as usize;
                match self.grid.cells()[idx] {
                    Owner::Empty => {}
                    Owner::Transition(existing) if existing == transition => {}
                    Owner::Room(room) if allowed_rooms.contains(&room) => {}
                    owner => {
                        return Err(EnhancedError::ContractViolation {
                            detail: format!(
                                "transition {:?} conflicts with {:?} at ({}, {})",
                                transition,
                                owner,
                                px * Q_U,
                                py * Q_U
                            ),
                        })
                    }
                }
            }
        }
        for py in qy0..qy1 {
            for px in qx0..qx1 {
                let idx = self.grid.cells_x() as usize * py as usize + px as usize;
                if matches!(self.grid.cells()[idx], Owner::Room(room) if allowed_rooms.contains(&room))
                {
                    continue;
                }
                self.grid.cells_mut()[idx] = Owner::Transition(transition);
            }
        }
        Ok(())
    }

    /// Check if a rect is empty (no rooms, routes, or transitions).
    pub fn is_rect_empty(&self, x0: i32, y0: i32, w: i32, h: i32) -> Result<bool, EnhancedError> {
        self.grid.is_rect_empty(x0, y0, w, h)
    }

    // ── Route / transition records ──────────────────────────────────────

    /// Add a committed route intent.
    pub fn add_route(&mut self, route: RouteIntent) {
        self.routes.push(route);
    }

    /// Add a committed transition intent.
    pub fn add_transition(&mut self, transition: TransitionIntent) {
        self.transitions.push(transition);
    }

    // ── Loop budget ─────────────────────────────────────────────────────

    /// Remaining loop budget.
    pub fn loop_budget_remaining(&self) -> u32 {
        self.loop_budget_remaining
    }

    /// Consume one unit of loop budget.
    pub fn consume_loop_budget(&mut self) -> bool {
        if self.loop_budget_remaining > 0 {
            self.loop_budget_remaining -= 1;
            true
        } else {
            false
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────

    pub fn routes(&self) -> &[RouteIntent] {
        &self.routes
    }
    pub fn transitions(&self) -> &[TransitionIntent] {
        &self.transitions
    }

    /// Canonical socket-ownership snapshot used by post-commit validation.
    pub fn socket_claims(&self) -> &BTreeMap<SocketId, OwnerKind> {
        &self.socket_claims
    }
}

// ── Committed state ────────────────────────────────────────────────────────

/// The fully committed state after a successful topology build.
#[derive(Debug, Clone)]
pub struct CommittedState {
    pub routes: Vec<RouteIntent>,
    pub transitions: Vec<TransitionIntent>,
    pub socket_claims: BTreeMap<SocketId, OwnerKind>,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::intent::{IdAllocator, RoomId};
    use super::super::occupancy::OccupancyGrid;
    use super::*;

    fn make_grid() -> OccupancyGrid {
        OccupancyGrid::new(1024, 1024).unwrap()
    }

    #[test]
    fn mark_rollback_restores_grid() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        let mark = tx.mark();
        tx.reserve_route_rect(0, 0, 64, 64, RouteId(0)).unwrap();
        tx.rollback(mark);

        assert!(tx.is_rect_empty(0, 0, 64, 64).unwrap());
    }

    #[test]
    fn mark_rollback_restores_socket_claims() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        let mark = tx.mark();
        tx.claim_socket(SocketId(0), OwnerKind::Route(RouteId(0)))
            .unwrap();
        assert!(tx.socket_is_claimed(SocketId(0)));

        tx.rollback(mark);
        assert!(!tx.socket_is_claimed(SocketId(0)));
    }

    #[test]
    fn mark_rollback_restores_loop_budget() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        let mark = tx.mark();
        assert!(tx.consume_loop_budget());
        assert_eq!(tx.loop_budget_remaining(), 2);

        tx.rollback(mark);
        assert_eq!(tx.loop_budget_remaining(), 3);
    }

    #[test]
    fn mark_rollback_restores_routes_and_transitions() {
        let grid = make_grid();
        let mut alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc.clone(), 3);

        let mark = tx.mark();
        let route = RouteIntent {
            id: alloc.next_route().unwrap(),
            source_socket: SocketId(0),
            target_socket: SocketId(1),
            source_room: RoomId(0),
            target_room: RoomId(1),
            path: Vec::new(),
            envelopes: Vec::new(),
            headroom: (16, 96),
        };
        tx.add_route(route.clone());

        assert_eq!(tx.routes().len(), 1);
        tx.rollback(mark);
        assert_eq!(tx.routes().len(), 0);
    }

    #[test]
    fn double_claim_rejected() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        tx.claim_socket(SocketId(0), OwnerKind::Route(RouteId(0)))
            .unwrap();
        let err = tx
            .claim_socket(SocketId(0), OwnerKind::Route(RouteId(1)))
            .unwrap_err();
        assert!(err.to_string().contains("already claimed"));
    }

    #[test]
    fn claim_route_sockets_atomic() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 3);

        // Pre-claim one socket
        tx.claim_socket(SocketId(1), OwnerKind::Route(RouteId(99)))
            .unwrap();

        let err = tx
            .claim_route_sockets(SocketId(0), SocketId(1), RouteId(0))
            .unwrap_err();
        assert!(err.to_string().contains("already claimed"));

        // Socket 0 should NOT have been claimed (atomicity)
        assert!(!tx.socket_is_claimed(SocketId(0)));
    }

    #[test]
    fn commit_returns_records() {
        let grid = make_grid();
        let mut alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc.clone(), 3);

        let route = RouteIntent {
            id: alloc.next_route().unwrap(),
            source_socket: SocketId(0),
            target_socket: SocketId(1),
            source_room: RoomId(0),
            target_room: RoomId(1),
            path: Vec::new(),
            envelopes: Vec::new(),
            headroom: (16, 96),
        };
        tx.add_route(route.clone());
        tx.claim_socket(SocketId(0), OwnerKind::Route(route.id))
            .unwrap();
        tx.claim_socket(SocketId(1), OwnerKind::Route(route.id))
            .unwrap();

        let committed = tx.commit();
        assert_eq!(committed.routes.len(), 1);
        assert_eq!(committed.socket_claims.len(), 2);
    }

    #[test]
    fn mark_rollback_restores_id_allocator() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc.clone(), 3);

        let mark = tx.mark();
        let _r0 = tx.alloc.next_route().unwrap();
        let _r1 = tx.alloc.next_route().unwrap();

        tx.rollback(mark);
        // Allocator restored; next ID should match original
        let r0_again = tx.alloc.next_route().unwrap();
        assert_eq!(r0_again.raw(), 0);
    }

    #[test]
    fn loop_budget_consumed_correctly() {
        let grid = make_grid();
        let alloc = IdAllocator::new();
        let mut tx = Transaction::new(grid, alloc, 2);

        assert!(tx.consume_loop_budget());
        assert_eq!(tx.loop_budget_remaining(), 1);
        assert!(tx.consume_loop_budget());
        assert_eq!(tx.loop_budget_remaining(), 0);
        assert!(!tx.consume_loop_budget());
        assert_eq!(tx.loop_budget_remaining(), 0);
    }
}
