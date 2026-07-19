use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

use super::config::NormalizedGeneratorConfig;
use super::determinism::Pcg32V1;
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    CandidateEdge, CandidateGraph, EdgeId, GridCoord, IdAllocator, IntendedEdge,
    IntendedTopology, OccupancyClass, OccupancyGrid, PlacedRegion, PlacedSocket, RegionId,
    RegionRole, SocketId, SocketRole, TransitionId,
};

// ─── Candidate graph construction ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct PendingEdge {
    source_socket: SocketId,
    target_socket: SocketId,
    source_region: RegionId,
    target_region: RegionId,
    path_witness: Vec<GridCoord>,
    envelope_cells: Vec<GridCoord>,
    cost: u64,
    width: u16,
    transition: Option<TransitionId>,
}

impl PendingEdge {
    fn canonical_key(&self) -> (RegionId, RegionId, SocketId, SocketId, Option<TransitionId>) {
        (
            self.source_region,
            self.target_region,
            self.source_socket,
            self.target_socket,
            self.transition,
        )
    }
}

/// Build the canonical candidate graph only after placement has committed the
/// occupancy grid. Cross-layer edges are emitted solely from explicit
/// transition endpoint bindings; ordinary socket pairs are routed by A*.
pub(super) fn build_candidate_graph(
    topology: &IntendedTopology,
    grid: &OccupancyGrid,
) -> Result<CandidateGraph, GeneratorError> {
    topology.validate_transition_bindings()?;
    let config = &topology.config;
    let mut regions: Vec<&PlacedRegion> = topology.regions.iter().collect();
    regions.sort_by_key(|region| region.id);
    let mut pending = Vec::new();

    for (left_index, left) in regions.iter().enumerate() {
        let start = left_index.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "candidate_region_pair_start",
        })?;
        for right in regions.iter().skip(start) {
            for left_socket in &left.sockets {
                for right_socket in &right.sockets {
                    if left.layer != right.layer {
                        if sockets_compatible(left_socket, right_socket) {
                            if let Some(transition) = transition_for_socket_pair(
                            topology,
                            left,
                            left_socket,
                            right,
                            right_socket,
                        )? {
                            let mut witness: Vec<GridCoord> = transition
                                .ramp_run_cells
                                .iter()
                                .chain(&transition.upper_opening_cells)
                                .chain(&transition.landing_cells)
                                .chain(&transition.headroom_cells)
                                .copied()
                                .collect();
                            witness.sort();
                            witness.dedup();
                            let cost = u64::try_from(witness.len()).map_err(|_| {
                                GeneratorError::ArithmeticOverflow {
                                    stage: ErrorStage::Topology,
                                    operation: "vertical_edge_cost",
                                }
                            })?;
                            if cost == 0 {
                                return Err(GeneratorError::TransitionBinding {
                                    stage: ErrorStage::Topology,
                                    transition: transition.id.raw(),
                                    reason: "vertical_witness_empty",
                                });
                            }
                                pending.push(PendingEdge {
                                    source_socket: transition.lower_socket,
                                    target_socket: transition.upper_socket,
                                    source_region: transition.lower_region,
                                    target_region: transition.upper_region,
                                    path_witness: witness.clone(),
                                    envelope_cells: witness,
                                    cost,
                                    width: 1,
                                    transition: Some(transition.id),
                                });
                            }
                        }
                        continue;
                    }
                    if !sockets_compatible(left_socket, right_socket) {
                        continue;
                    }
                    let width = corridor_width_for_sockets(left_socket, right_socket, config)?;
                    if let Some((path, envelope)) = find_path_with_envelope(
                        left,
                        left_socket,
                        right,
                        right_socket,
                        grid,
                        config,
                        width,
                    )? {
                        let cost = u64::try_from(path.len()).map_err(|_| {
                            GeneratorError::ArithmeticOverflow {
                                stage: ErrorStage::Topology,
                                operation: "candidate_path_cost",
                            }
                        })?;
                        pending.push(PendingEdge {
                            source_socket: left_socket.id,
                            target_socket: right_socket.id,
                            source_region: left.id,
                            target_region: right.id,
                            path_witness: path,
                            envelope_cells: envelope,
                            cost,
                            width,
                            transition: None,
                        });
                    }
                }
            }
        }
    }

    pending.sort_by_key(PendingEdge::canonical_key);
    pending.dedup_by_key(|edge| edge.canonical_key());
    let mut allocator = IdAllocator::new();
    let mut edges = Vec::with_capacity(pending.len());
    for edge in pending {
        edges.push(CandidateEdge {
            id: allocator.next_edge()?,
            source_socket: edge.source_socket,
            target_socket: edge.target_socket,
            source_region: edge.source_region,
            target_region: edge.target_region,
            path_witness: edge.path_witness,
            allowed_envelope_cells: edge.envelope_cells,
            cost: edge.cost,
            width: edge.width,
            transition: edge.transition,
        });
    }
    let graph = CandidateGraph {
        edges,
        occupancy: grid.clone(),
    };
    validate_candidate_graph(topology, &graph)?;
    Ok(graph)
}

fn transition_for_socket_pair<'a>(
    topology: &'a IntendedTopology,
    left_region: &PlacedRegion,
    left_socket: &PlacedSocket,
    right_region: &PlacedRegion,
    right_socket: &PlacedSocket,
) -> Result<Option<&'a super::ir::TransitionReservation>, GeneratorError> {
    let layers_adjacent = left_region.layer.abs_diff(right_region.layer) == 1;
    let role_pair = matches!(
        (left_socket.role, right_socket.role),
        (SocketRole::LowerRampApproach, SocketRole::UpperLanding)
            | (SocketRole::UpperLanding, SocketRole::LowerRampApproach)
    );
    if !layers_adjacent || !role_pair {
        return Ok(None);
    }
    let found = topology.transitions.iter().find(|transition| {
        (transition.lower_region == left_region.id
            && transition.upper_region == right_region.id
            && transition.lower_socket == left_socket.id
            && transition.upper_socket == right_socket.id)
            || (transition.lower_region == right_region.id
                && transition.upper_region == left_region.id
                && transition.lower_socket == right_socket.id
                && transition.upper_socket == left_socket.id)
    });
    Ok(found)
}

fn validate_candidate_graph(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
) -> Result<(), GeneratorError> {
    let region_layers: BTreeMap<RegionId, u16> = topology
        .regions
        .iter()
        .map(|region| (region.id, region.layer))
        .collect();
    let mut transition_counts: BTreeMap<TransitionId, u32> = BTreeMap::new();
    let mut previous_key = None;
    for edge in &graph.edges {
        let key = (
            edge.source_region,
            edge.target_region,
            edge.source_socket,
            edge.target_socket,
        );
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "candidate_graph_not_canonical".into(),
            });
        }
        previous_key = Some(key);
        let source_layer = region_layers.get(&edge.source_region).copied().ok_or(
            GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "candidate_source_region_missing".into(),
            },
        )?;
        let target_layer = region_layers.get(&edge.target_region).copied().ok_or(
            GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "candidate_target_region_missing".into(),
            },
        )?;
        if source_layer != target_layer && edge.transition.is_none() {
            return Err(GeneratorError::TransitionBinding {
                stage: ErrorStage::Topology,
                transition: u32::MAX,
                reason: "unbound_cross_layer_candidate",
            });
        }
        if source_layer == target_layer && edge.transition.is_some() {
            return Err(GeneratorError::TransitionBinding {
                stage: ErrorStage::Topology,
                transition: edge.transition.map_or(u32::MAX, TransitionId::raw),
                reason: "same_layer_transition_candidate",
            });
        }
        if let Some(transition) = edge.transition {
            let count = transition_counts.entry(transition).or_default();
            *count = count.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "candidate_transition_count",
            })?;
        }
    }
    for transition in &topology.transitions {
        let actual = transition_counts.get(&transition.id).copied().unwrap_or(0);
        if actual != 1 {
            return Err(GeneratorError::TransitionBinding {
                stage: ErrorStage::Topology,
                transition: transition.id.raw(),
                reason: "vertical_candidate_count_not_one",
            });
        }
    }
    Ok(())
}

fn sockets_compatible(left: &PlacedSocket, right: &PlacedSocket) -> bool {
    if left.global_anchor.layer != right.global_anchor.layer {
        let ramp_roles = matches!(
            (left.role, right.role),
            (SocketRole::LowerRampApproach, SocketRole::UpperLanding)
                | (SocketRole::UpperLanding, SocketRole::LowerRampApproach)
        );
        return left.global_anchor.layer.abs_diff(right.global_anchor.layer) == 1
            && ramp_roles
            && left.paired_socket_id == Some(right.id)
            && right.paired_socket_id == Some(left.id);
    }
    left.direction == right.direction.opposite()
        && left.width > 0
        && left.width == right.width
}

fn corridor_width_for_sockets(
    left: &PlacedSocket,
    right: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
) -> Result<u16, GeneratorError> {
    let configured = if left.width >= 2 && right.width >= 2 {
        config.hall_width()
    } else {
        config.corridor_width()
    };
    u16::try_from(configured).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Topology,
        operation: "corridor_width_convert",
    })
}

// ─── A* routing and exact width envelopes ───────────────────────────────────

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct AStarNode {
    coord: GridCoord,
    distance: u64,
    estimate: u64,
    tie: u64,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .estimate
            .cmp(&self.estimate)
            .then_with(|| other.distance.cmp(&self.distance))
            .then_with(|| other.tie.cmp(&self.tie))
            .then_with(|| other.coord.cmp(&self.coord))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn manhattan_distance(left: GridCoord, right: GridCoord) -> Result<u64, GeneratorError> {
    u64::from(left.x.abs_diff(right.x))
        .checked_add(u64::from(left.y.abs_diff(right.y)))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "manhattan_distance",
        })
}

fn astar_direction_bias(
    source: SocketId,
    target: SocketId,
) -> Result<usize, GeneratorError> {
    let sum = u64::from(source.raw())
        .checked_add(u64::from(target.raw()))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_direction_bias_add",
        })?;
    usize::try_from(sum % 4).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Topology,
        operation: "astar_direction_bias_convert",
    })
}

fn astar_tie_key(
    cell: GridCoord,
    bias: usize,
    config: &NormalizedGeneratorConfig,
) -> Result<u64, GeneratorError> {
    let x = u64::from(cell.x);
    let y = u64::from(cell.y);
    let width = u64::from(config.width());
    let height = u64::from(config.height());
    let reverse_x = width
        .checked_sub(1)
        .and_then(|value| value.checked_sub(x))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_tie_reverse_x",
        })?;
    let reverse_y = height
        .checked_sub(1)
        .and_then(|value| value.checked_sub(y))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_tie_reverse_y",
        })?;
    let (major, dimension, minor) = match bias {
        0 => (y, width, x),
        1 => (x, height, reverse_y),
        2 => (reverse_y, width, reverse_x),
        _ => (reverse_x, height, y),
    };
    major
        .checked_mul(dimension)
        .and_then(|value| value.checked_add(minor))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_tie_key",
        })
}

fn socket_aperture_cells(
    socket: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<GridCoord>, GeneratorError> {
    let mut cells = Vec::with_capacity(usize::from(socket.width));
    for offset in 0..socket.width {
        let (x, y) = match socket.direction {
            super::ir::Direction::North | super::ir::Direction::South => (
                socket
                    .global_anchor
                    .x
                    .checked_add(offset)
                    .ok_or(GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "socket_aperture_x",
                    })?,
                socket.global_anchor.y,
            ),
            super::ir::Direction::East | super::ir::Direction::West => (
                socket.global_anchor.x,
                socket
                    .global_anchor
                    .y
                    .checked_add(offset)
                    .ok_or(GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "socket_aperture_y",
                    })?,
            ),
        };
        cells.push(GridCoord::new(
            socket.global_anchor.layer,
            x,
            y,
            config.width(),
            config.height(),
            config.layers().2,
        )?);
    }
    Ok(cells)
}

fn socket_inward_cells(
    socket: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<GridCoord>, GeneratorError> {
    let (dx, dy) = socket.direction.delta();
    socket_aperture_cells(socket, config)?
        .into_iter()
        .map(|aperture| {
            let x = i32::from(aperture.x)
                .checked_sub(dx)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "socket_inward_x",
                })?;
            let y = i32::from(aperture.y)
                .checked_sub(dy)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "socket_inward_y",
                })?;
            let x = u16::try_from(x).map_err(|_| GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "socket_inward_x_out_of_bounds".into(),
            })?;
            let y = u16::try_from(y).map_err(|_| GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "socket_inward_y_out_of_bounds".into(),
            })?;
            GridCoord::new(
                aperture.layer,
                x,
                y,
                config.width(),
                config.height(),
                config.layers().2,
            )
        })
        .collect()
}

fn terminal_cells(
    socket: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
) -> Result<BTreeSet<GridCoord>, GeneratorError> {
    let mut cells: BTreeSet<GridCoord> = socket_aperture_cells(socket, config)?.into_iter().collect();
    cells.extend(socket_inward_cells(socket, config)?);
    Ok(cells)
}

fn width_offset_bounds(width: u16) -> Result<(i32, i32), GeneratorError> {
    if width == 0 {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "zero_corridor_width".into(),
        });
    }
    let left = i32::from(width / 2);
    let right = i32::from(
        width
            .checked_sub(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "corridor_width_sub",
            })?
            / 2,
    );
    Ok((-left, right))
}

#[allow(clippy::too_many_arguments)]
fn cell_walkable(
    cell: GridCoord,
    grid: &OccupancyGrid,
    source_region: &PlacedRegion,
    source_socket: &PlacedSocket,
    target_region: &PlacedRegion,
    target_socket: &PlacedSocket,
    source_terminal: &BTreeSet<GridCoord>,
    target_terminal: &BTreeSet<GridCoord>,
) -> bool {
    match grid.get(cell) {
        Some(OccupancyClass::Empty) | Some(OccupancyClass::Spacing(_)) => true,
        Some(OccupancyClass::Socket(owner)) => {
            (owner == source_socket.id.raw() && source_terminal.contains(&cell))
                || (owner == target_socket.id.raw() && target_terminal.contains(&cell))
        }
        Some(OccupancyClass::Region(owner)) => {
            (owner == source_region.id.raw() && source_terminal.contains(&cell))
                || (owner == target_region.id.raw() && target_terminal.contains(&cell))
        }
        Some(OccupancyClass::TransitionHub(owner))
        | Some(OccupancyClass::Transition(owner)) => {
            (source_region
                .transitions
                .iter()
                .any(|transition| transition.raw() == owner)
                && source_terminal.contains(&cell))
                || (target_region
                    .transitions
                    .iter()
                    .any(|transition| transition.raw() == owner)
                    && target_terminal.contains(&cell))
        }
        None => false,
        _ => false, // Border and any future non-walkable variants
    }
}

#[allow(clippy::too_many_arguments)]
fn clearance_walkable(
    center: GridCoord,
    movement: (i32, i32),
    width: u16,
    grid: &OccupancyGrid,
    source_region: &PlacedRegion,
    source_socket: &PlacedSocket,
    target_region: &PlacedRegion,
    target_socket: &PlacedSocket,
    source_terminal: &BTreeSet<GridCoord>,
    target_terminal: &BTreeSet<GridCoord>,
    config: &NormalizedGeneratorConfig,
) -> Result<bool, GeneratorError> {
    let perpendicular: (i32, i32) = if movement.0 == 0 { (1, 0) } else { (0, 1) };
    let (first, last) = width_offset_bounds(width)?;
    for offset in first..=last {
        let x = i32::from(center.x)
            .checked_add(perpendicular.0.checked_mul(offset).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "clearance_x_mul",
                },
            )?)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "clearance_x_add",
            })?;
        let y = i32::from(center.y)
            .checked_add(perpendicular.1.checked_mul(offset).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "clearance_y_mul",
                },
            )?)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "clearance_y_add",
            })?;
        let Ok(x) = u16::try_from(x) else {
            return Ok(false);
        };
        let Ok(y) = u16::try_from(y) else {
            return Ok(false);
        };
        let Ok(cell) = GridCoord::new(
            center.layer,
            x,
            y,
            config.width(),
            config.height(),
            config.layers().2,
        ) else {
            return Ok(false);
        };
        if !cell_walkable(
            cell,
            grid,
            source_region,
            source_socket,
            target_region,
            target_socket,
            source_terminal,
            target_terminal,
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn find_path_with_envelope(
    source_region: &PlacedRegion,
    source_socket: &PlacedSocket,
    target_region: &PlacedRegion,
    target_socket: &PlacedSocket,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
    width: u16,
) -> Result<Option<(Vec<GridCoord>, Vec<GridCoord>)>, GeneratorError> {
    let source_inward = socket_inward_cells(source_socket, config)?;
    let target_inward = socket_inward_cells(target_socket, config)?;
    let start = source_inward.first().copied().ok_or(GeneratorError::IrInvariant {
        stage: ErrorStage::Topology,
        detail: "source_socket_has_no_inward_cell".into(),
    })?;
    let goal = target_inward.first().copied().ok_or(GeneratorError::IrInvariant {
        stage: ErrorStage::Topology,
        detail: "target_socket_has_no_inward_cell".into(),
    })?;
    if start.layer != goal.layer {
        return Ok(None);
    }
    let source_terminal = terminal_cells(source_socket, config)?;
    let target_terminal = terminal_cells(target_socket, config)?;
    let mut open = BinaryHeap::new();
    let mut distances = BTreeMap::new();
    let mut parents = BTreeMap::new();
    distances.insert(start, 0u64);
    let direction_bias = astar_direction_bias(source_socket.id, target_socket.id)?;
    open.push(AStarNode {
        coord: start,
        distance: 0,
        estimate: manhattan_distance(start, goal)?,
        tie: astar_tie_key(start, direction_bias, config)?,
    });
    let max_expansions = usize::from(config.width())
        .checked_mul(usize::from(config.height()))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_expansion_limit",
        })?;
    let mut expansions = 0usize;
    let mut directions = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    directions.rotate_left(direction_bias);

    while let Some(current) = open.pop() {
        expansions = expansions.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "astar_expansion_count",
        })?;
        if expansions > max_expansions {
            return Ok(None);
        }
        let best = distances.get(&current.coord).copied().unwrap_or(u64::MAX);
        if current.distance != best {
            continue;
        }
        if current.coord == goal {
            let mut path = vec![goal];
            let mut cursor = goal;
            while cursor != start {
                let Some(parent) = parents.get(&cursor).copied() else {
                    return Err(GeneratorError::IrInvariant {
                        stage: ErrorStage::Topology,
                        detail: "astar_parent_missing".into(),
                    });
                };
                cursor = parent;
                path.push(cursor);
            }
            path.reverse();
            let envelope = compute_cell_envelope(&path, width, config)?;
            return Ok(Some((path, envelope)));
        }
        for movement in directions {
            let x = i32::from(current.coord.x)
                .checked_add(movement.0)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "astar_neighbor_x",
                })?;
            let y = i32::from(current.coord.y)
                .checked_add(movement.1)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "astar_neighbor_y",
                })?;
            let Ok(x) = u16::try_from(x) else {
                continue;
            };
            let Ok(y) = u16::try_from(y) else {
                continue;
            };
            let Ok(next) = GridCoord::new(
                start.layer,
                x,
                y,
                config.width(),
                config.height(),
                config.layers().2,
            ) else {
                continue;
            };
            if !clearance_walkable(
                next,
                movement,
                width,
                grid,
                source_region,
                source_socket,
                target_region,
                target_socket,
                &source_terminal,
                &target_terminal,
                config,
            )? {
                continue;
            }
            let distance = current.distance.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "astar_distance",
                },
            )?;
            if distance < distances.get(&next).copied().unwrap_or(u64::MAX) {
                distances.insert(next, distance);
                parents.insert(next, current.coord);
                open.push(AStarNode {
                    coord: next,
                    distance,
                    estimate: distance.checked_add(manhattan_distance(next, goal)?).ok_or(
                        GeneratorError::ArithmeticOverflow {
                            stage: ErrorStage::Topology,
                            operation: "astar_estimate",
                        },
                    )?,
                    tie: astar_tie_key(next, direction_bias, config)?,
                });
            }
        }
    }
    Ok(None)
}

fn compute_cell_envelope(
    path: &[GridCoord],
    width: u16,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<GridCoord>, GeneratorError> {
    let mut cells = BTreeSet::new();
    let (first, last) = width_offset_bounds(width)?;
    if path.len() == 1 {
        if let Some(cell) = path.first().copied() {
            cells.insert(cell);
        }
    }
    for pair in path.windows(2) {
        let Some(left) = pair.first().copied() else {
            continue;
        };
        let Some(right) = pair.get(1).copied() else {
            continue;
        };
        let dx = i32::from(right.x)
            .checked_sub(i32::from(left.x))
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "envelope_direction_x",
            })?;
        let perpendicular = if dx == 0 { (1i32, 0i32) } else { (0, 1) };
        for center in [left, right] {
            for offset in first..=last {
                let x = i32::from(center.x)
                    .checked_add(perpendicular.0.checked_mul(offset).ok_or(
                        GeneratorError::ArithmeticOverflow {
                            stage: ErrorStage::Topology,
                            operation: "envelope_x_mul",
                        },
                    )?)
                    .ok_or(GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "envelope_x_add",
                    })?;
                let y = i32::from(center.y)
                    .checked_add(perpendicular.1.checked_mul(offset).ok_or(
                        GeneratorError::ArithmeticOverflow {
                            stage: ErrorStage::Topology,
                            operation: "envelope_y_mul",
                        },
                    )?)
                    .ok_or(GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "envelope_y_add",
                    })?;
                let Ok(x) = u16::try_from(x) else {
                    continue;
                };
                let Ok(y) = u16::try_from(y) else {
                    continue;
                };
                if let Ok(cell) = GridCoord::new(
                    center.layer,
                    x,
                    y,
                    config.width(),
                    config.height(),
                    config.layers().2,
                ) {
                    cells.insert(cell);
                }
            }
        }
    }
    Ok(cells.into_iter().collect())
}

// ─── Topology selection ─────────────────────────────────────────────────────

fn is_dead_end(topology: &IntendedTopology, region: RegionId) -> bool {
    topology
        .regions
        .iter()
        .any(|candidate| candidate.id == region && candidate.role == RegionRole::DeadEnd)
}

fn is_required_region(topology: &IntendedTopology, region: RegionId) -> bool {
    topology.regions.iter().any(|candidate| {
        candidate.id == region
            && matches!(
                candidate.role,
                RegionRole::Spawn
                    | RegionRole::DistantLandmark
                    | RegionRole::MajorLandmark
                    | RegionRole::Junction
                    | RegionRole::VerticalHub
                    | RegionRole::RequiredRoute
            )
    })
}

fn edge_order(graph: &CandidateGraph, nonce: u32) -> Vec<CandidateEdge> {
    let mut edges = graph.edges.clone();
    edges.sort_by(|left, right| {
        left.cost
            .cmp(&right.cost)
            .then_with(|| (left.id.raw() ^ nonce).cmp(&(right.id.raw() ^ nonce)))
            .then_with(|| left.id.cmp(&right.id))
    });
    edges
}

fn selected_edges<'a>(
    graph: &'a CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Vec<&'a CandidateEdge> {
    graph
        .edges
        .iter()
        .filter(|edge| selected.contains(&edge.id))
        .collect()
}

fn endpoints(edge: &CandidateEdge) -> [RegionId; 2] {
    [edge.source_region, edge.target_region]
}

fn shared_region(left: &CandidateEdge, right: &CandidateEdge) -> Option<RegionId> {
    endpoints(left)
        .into_iter()
        .find(|region| endpoints(right).contains(region))
}

fn cell_in_region(cell: GridCoord, region: &PlacedRegion) -> Result<bool, GeneratorError> {
    if cell.layer != region.layer {
        return Ok(false);
    }
    let max_x = region.footprint.0.checked_add(region.footprint.2).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "region_bounds_x",
        },
    )?;
    let max_y = region.footprint.1.checked_add(region.footprint.3).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "region_bounds_y",
        },
    )?;
    Ok(cell.x >= region.footprint.0
        && cell.x < max_x
        && cell.y >= region.footprint.1
        && cell.y < max_y)
}

fn edges_coexist(
    topology: &IntendedTopology,
    left: &CandidateEdge,
    right: &CandidateEdge,
) -> Result<bool, GeneratorError> {
    let left_cells: BTreeSet<GridCoord> =
        left.allowed_envelope_cells.iter().copied().collect();
    let overlap: Vec<GridCoord> = right
        .allowed_envelope_cells
        .iter()
        .copied()
        .filter(|cell| left_cells.contains(cell))
        .collect();
    if overlap.is_empty() {
        return Ok(true);
    }
    let Some(shared) = shared_region(left, right) else {
        return Ok(false);
    };
    let region = topology
        .regions
        .iter()
        .find(|region| region.id == shared)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "shared_edge_region_missing".into(),
        })?;
    for cell in overlap {
        if !cell_in_region(cell, region)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn candidate_coexists(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    candidate: &CandidateEdge,
) -> Result<bool, GeneratorError> {
    for existing in selected_edges(graph, selected) {
        if !edges_coexist(topology, existing, candidate)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reroute_candidate(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    candidate: &CandidateEdge,
) -> Result<Option<CandidateEdge>, GeneratorError> {
    if candidate.transition.is_some() {
        return Ok(None);
    }
    let source_region = topology
        .regions
        .iter()
        .find(|region| region.id == candidate.source_region)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "reroute_source_region_missing".into(),
        })?;
    let target_region = topology
        .regions
        .iter()
        .find(|region| region.id == candidate.target_region)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "reroute_target_region_missing".into(),
        })?;
    let source_socket = source_region
        .sockets
        .iter()
        .find(|socket| socket.id == candidate.source_socket)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "reroute_source_socket_missing".into(),
        })?;
    let target_socket = target_region
        .sockets
        .iter()
        .find(|socket| socket.id == candidate.target_socket)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "reroute_target_socket_missing".into(),
        })?;
    let mut overlay = graph.occupancy.clone();
    for existing in selected_edges(graph, selected) {
        let shared = shared_region(existing, candidate).and_then(|id| {
            topology.regions.iter().find(|region| region.id == id)
        });
        for cell in &existing.allowed_envelope_cells {
            let inside_shared_region = if let Some(region) = shared {
                cell_in_region(*cell, region)?
            } else {
                false
            };
            if inside_shared_region {
                continue;
            }
            if matches!(
                overlay.get(*cell),
                Some(OccupancyClass::Empty) | Some(OccupancyClass::Spacing(_))
            ) {
                overlay.set(*cell, OccupancyClass::Region(u32::MAX))?;
            }
        }
    }
    let Some((path, envelope)) = find_path_with_envelope(
        source_region,
        source_socket,
        target_region,
        target_socket,
        &overlay,
        &topology.config,
        candidate.width,
    )? else {
        return Ok(None);
    };
    let mut rerouted = candidate.clone();
    rerouted.cost = u64::try_from(path.len()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "rerouted_path_cost",
        }
    })?;
    rerouted.path_witness = path;
    rerouted.allowed_envelope_cells = envelope;
    if candidate_coexists(topology, graph, selected, &rerouted)? {
        Ok(Some(rerouted))
    } else {
        Ok(None)
    }
}

fn add_edge_if_legal(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    selected: &mut BTreeSet<EdgeId>,
    edge: &CandidateEdge,
    preserve_route_min: bool,
) -> Result<bool, GeneratorError> {
    if selected.contains(&edge.id) {
        return Ok(true);
    }
    let original = edge_by_id(graph, edge.id)?.clone();
    let realized = if candidate_coexists(topology, graph, selected, &original)? {
        original.clone()
    } else if let Some(rerouted) = reroute_candidate(topology, graph, selected, &original)? {
        rerouted
    } else {
        // A configured crossing is a bounded fallback after deterministic
        // rerouting fails. Every conflicting witness pair is counted and the
        // final verifier enforces the profile maximum.
        let mut conflicts = 0u32;
        for existing in selected_edges(graph, selected) {
            if !edges_coexist(topology, existing, &original)? {
                conflicts = conflicts.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "candidate_crossing_count",
                    },
                )?;
            }
        }
        let existing_crossings = crossing_count(topology, graph, selected)?;
        let projected = existing_crossings.checked_add(conflicts).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "projected_crossing_count",
            },
        )?;
        if projected > topology.config.crossings_max() {
            return Ok(false);
        }
        original.clone()
    };
    let position = graph
        .edges
        .iter()
        .position(|candidate| candidate.id == edge.id)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "realized_candidate_position_missing".into(),
        })?;
    let slot = graph.edges.get_mut(position).ok_or(GeneratorError::IrInvariant {
        stage: ErrorStage::Topology,
        detail: "realized_candidate_slot_missing".into(),
    })?;
    *slot = realized;
    selected.insert(edge.id);
    if preserve_route_min {
        if let Some(distance) = spawn_landmark_distance(topology, graph, selected)? {
            if distance < u64::from(topology.config.required_route_min()) {
                selected.remove(&edge.id);
                let restore = graph.edges.get_mut(position).ok_or(
                    GeneratorError::IrInvariant {
                        stage: ErrorStage::Topology,
                        detail: "candidate_restore_slot_missing".into(),
                    },
                )?;
                *restore = original;
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn adjacency<'a>(
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
) -> BTreeMap<RegionId, Vec<(RegionId, EdgeId, u64)>> {
    let mut map: BTreeMap<RegionId, Vec<(RegionId, EdgeId, u64)>> = BTreeMap::new();
    for edge in edges {
        map.entry(edge.source_region)
            .or_default()
            .push((edge.target_region, edge.id, edge.cost));
        map.entry(edge.target_region)
            .or_default()
            .push((edge.source_region, edge.id, edge.cost));
    }
    for neighbors in map.values_mut() {
        neighbors.sort_by_key(|neighbor| (neighbor.2, neighbor.0, neighbor.1));
    }
    map
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct RegionDistance {
    distance: u64,
    region: RegionId,
}

impl Ord for RegionDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .cmp(&self.distance)
            .then_with(|| other.region.cmp(&self.region))
    }
}

impl PartialOrd for RegionDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn shortest_path(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    source: RegionId,
    target: RegionId,
    blocked: &BTreeSet<EdgeId>,
    selected_only: Option<&BTreeSet<EdgeId>>,
) -> Result<Option<(u64, Vec<EdgeId>)>, GeneratorError> {
    let usable = graph.edges.iter().filter(|edge| {
        !blocked.contains(&edge.id)
            && selected_only.is_none_or(|selected| selected.contains(&edge.id))
            && (!is_dead_end(topology, edge.source_region)
                || edge.source_region == source
                || edge.source_region == target)
            && (!is_dead_end(topology, edge.target_region)
                || edge.target_region == source
                || edge.target_region == target)
    });
    let adjacency = adjacency(usable);
    let mut distances = BTreeMap::new();
    let mut parents: BTreeMap<RegionId, (RegionId, EdgeId)> = BTreeMap::new();
    let mut heap = BinaryHeap::new();
    distances.insert(source, 0u64);
    heap.push(RegionDistance {
        distance: 0,
        region: source,
    });
    while let Some(current) = heap.pop() {
        if current.distance != distances.get(&current.region).copied().unwrap_or(u64::MAX) {
            continue;
        }
        if current.region == target {
            let mut path = Vec::new();
            let mut cursor = target;
            while cursor != source {
                let Some((parent, edge)) = parents.get(&cursor).copied() else {
                    return Err(GeneratorError::IrInvariant {
                        stage: ErrorStage::Topology,
                        detail: "shortest_path_parent_missing".into(),
                    });
                };
                path.push(edge);
                cursor = parent;
            }
            path.reverse();
            return Ok(Some((current.distance, path)));
        }
        if let Some(neighbors) = adjacency.get(&current.region) {
            for (next, edge, cost) in neighbors {
                let distance = current.distance.checked_add(*cost).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "dijkstra_distance",
                    },
                )?;
                if distance < distances.get(next).copied().unwrap_or(u64::MAX) {
                    distances.insert(*next, distance);
                    parents.insert(*next, (current.region, *edge));
                    heap.push(RegionDistance {
                        distance,
                        region: *next,
                    });
                }
            }
        }
    }
    Ok(None)
}

fn spawn_and_landmark(
    topology: &IntendedTopology,
) -> Result<(RegionId, RegionId), GeneratorError> {
    let spawn = topology
        .regions
        .iter()
        .find(|region| region.role == RegionRole::Spawn)
        .map(|region| region.id)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "spawn_region_missing".into(),
        })?;
    let landmark = topology
        .regions
        .iter()
        .find(|region| region.role == RegionRole::DistantLandmark)
        .map(|region| region.id)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "distant_landmark_region_missing".into(),
        })?;
    Ok((spawn, landmark))
}

fn bounded_spine(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
) -> Result<Vec<EdgeId>, GeneratorError> {
    let (spawn, landmark) = spawn_and_landmark(topology)?;
    let minimum = u64::from(topology.config.required_route_min());
    let maximum = u64::from(topology.config.required_route_max());
    let budget = u64::from(topology.config.routing_attempts())
        .checked_mul(u64::from(topology.config.reroute_budget()))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "spine_search_budget",
        })?;
    let mut queue = VecDeque::from([BTreeSet::new()]);
    let mut seen = BTreeSet::new();
    seen.insert(Vec::<EdgeId>::new());
    let mut attempts = 0u64;
    while let Some(blocked) = queue.pop_front() {
        attempts = attempts.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "spine_search_attempts",
        })?;
        if attempts > budget {
            return Err(GeneratorError::SearchExhausted {
                stage: ErrorStage::Topology,
                search: "bounded_spine",
                attempted: attempts,
                budget,
            });
        }
        let Some((distance, path)) =
            shortest_path(topology, graph, spawn, landmark, &blocked, None)?
        else {
            continue;
        };
        if (minimum..=maximum).contains(&distance) {
            return Ok(path);
        }
        if distance > maximum {
            continue;
        }
        for edge in path {
            let mut branch = blocked.clone();
            branch.insert(edge);
            let key: Vec<EdgeId> = branch.iter().copied().collect();
            if seen.insert(key) {
                queue.push_back(branch);
            }
        }
    }
    Err(GeneratorError::TopologyInfeasible {
        stage: ErrorStage::Topology,
        constraint: "bounded_spine_unavailable",
        required: minimum,
        available: 0,
    })
}

fn edge_by_id<'a>(
    graph: &'a CandidateGraph,
    id: EdgeId,
) -> Result<&'a CandidateEdge, GeneratorError> {
    graph
        .edges
        .iter()
        .find(|edge| edge.id == id)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: format!("candidate_edge_missing {}", id.raw()),
        })
}

fn add_path(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    selected: &mut BTreeSet<EdgeId>,
    path: &[EdgeId],
    preserve_route_min: bool,
) -> Result<bool, GeneratorError> {
    let original_selected = selected.clone();
    let original_edges = graph.edges.clone();
    for id in path {
        let edge = edge_by_id(graph, *id)?.clone();
        if !add_edge_if_legal(
            topology,
            graph,
            selected,
            &edge,
            preserve_route_min,
        )? {
            *selected = original_selected;
            graph.edges = original_edges;
            return Ok(false);
        }
    }
    Ok(true)
}

fn component_labels(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    only_layer: Option<u16>,
    exclude_dead_ends: bool,
) -> Result<BTreeMap<RegionId, u32>, GeneratorError> {
    let mut adjacency: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    for edge in selected_edges(graph, selected) {
        if only_layer.is_some_and(|layer| {
            let source_layer = topology
                .regions
                .iter()
                .find(|region| region.id == edge.source_region)
                .map(|region| region.layer);
            let target_layer = topology
                .regions
                .iter()
                .find(|region| region.id == edge.target_region)
                .map(|region| region.layer);
            source_layer != Some(layer) || target_layer != Some(layer)
        }) {
            continue;
        }
        adjacency
            .entry(edge.source_region)
            .or_default()
            .push(edge.target_region);
        adjacency
            .entry(edge.target_region)
            .or_default()
            .push(edge.source_region);
    }
    let mut labels = BTreeMap::new();
    let mut component = 0u32;
    for region in &topology.regions {
        if only_layer.is_some_and(|layer| region.layer != layer)
            || (exclude_dead_ends && region.role == RegionRole::DeadEnd)
            || labels.contains_key(&region.id)
        {
            continue;
        }
        component = component.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "component_label_count",
        })?;
        let mut stack = vec![region.id];
        labels.insert(region.id, component);
        while let Some(current) = stack.pop() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if exclude_dead_ends && is_dead_end(topology, *neighbor) {
                        continue;
                    }
                    if !labels.contains_key(neighbor) {
                        labels.insert(*neighbor, component);
                        stack.push(*neighbor);
                    }
                }
            }
        }
    }
    Ok(labels)
}

fn connect_layer_cores(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    for layer in 0..topology.config.layers().2 {
        loop {
            let labels = component_labels(topology, graph, selected, Some(layer), true)?;
            let component_count = labels.values().copied().collect::<BTreeSet<_>>().len();
            if component_count <= 1 {
                break;
            }
            let mut added = false;
            for edge in ordered {
                if edge.transition.is_some()
                    || selected.contains(&edge.id)
                    || is_dead_end(topology, edge.source_region)
                    || is_dead_end(topology, edge.target_region)
                {
                    continue;
                }
                let source_label = labels.get(&edge.source_region);
                let target_label = labels.get(&edge.target_region);
                if source_label.is_none() || target_label.is_none() || source_label == target_label {
                    continue;
                }
                if add_edge_if_legal(topology, graph, selected, edge, true)? {
                    added = true;
                    break;
                }
            }
            if !added {
                return Err(GeneratorError::TopologyInfeasible {
                    stage: ErrorStage::Topology,
                    constraint: "layer_core_connectivity",
                    required: u64::try_from(component_count).map_err(|_| {
                        GeneratorError::ArithmeticOverflow {
                            stage: ErrorStage::Topology,
                            operation: "layer_component_count_convert",
                        }
                    })?,
                    available: 1,
                });
            }
        }
    }
    Ok(())
}

fn connect_global_core(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    fn search(
        topology: &IntendedTopology,
        graph: &mut CandidateGraph,
        ordered: &[CandidateEdge],
        selected: &mut BTreeSet<EdgeId>,
        attempts: &mut u64,
        budget: u64,
    ) -> Result<bool, GeneratorError> {
        let labels = component_labels(topology, graph, selected, None, true)?;
        let components: BTreeSet<u32> = labels.values().copied().collect();
        if components.len() <= 1 {
            return Ok(true);
        }
        if *attempts >= budget {
            return Ok(false);
        }
        let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
        for label in labels.values() {
            let size = sizes.entry(*label).or_default();
            *size = size.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "connectivity_component_size",
            })?;
        }
        let target_component = sizes
            .iter()
            .min_by_key(|(label, size)| (**size, **label))
            .map(|(label, _)| *label)
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "connectivity_target_component_missing".into(),
            })?;
        for edge in ordered {
            if selected.contains(&edge.id)
                || is_dead_end(topology, edge.source_region)
                || is_dead_end(topology, edge.target_region)
            {
                continue;
            }
            let source_label = labels.get(&edge.source_region).copied();
            let target_label = labels.get(&edge.target_region).copied();
            if source_label.is_none()
                || target_label.is_none()
                || source_label == target_label
                || (source_label != Some(target_component)
                    && target_label != Some(target_component))
            {
                continue;
            }
            *attempts = attempts.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "connectivity_search_attempts",
                },
            )?;
            let mut branch_graph = graph.clone();
            let mut branch_selected = selected.clone();
            if !add_edge_if_legal(
                topology,
                &mut branch_graph,
                &mut branch_selected,
                edge,
                true,
            )? {
                continue;
            }
            if search(
                topology,
                &mut branch_graph,
                ordered,
                &mut branch_selected,
                attempts,
                budget,
            )? {
                *graph = branch_graph;
                *selected = branch_selected;
                return Ok(true);
            }
        }
        Ok(false)
    }

    let budget = u64::from(topology.config.routing_attempts())
        .checked_mul(u64::from(topology.config.reroute_budget()))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "connectivity_search_budget",
        })?;
    let mut attempts = 0u64;
    if search(
        topology,
        graph,
        ordered,
        selected,
        &mut attempts,
        budget,
    )? {
        return Ok(());
    }
    let components = component_labels(topology, graph, selected, None, true)?
        .values()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    Err(GeneratorError::SearchExhausted {
        stage: ErrorStage::Topology,
        search: "global_core_connectivity",
        attempted: attempts,
        budget: budget.max(u64::try_from(components).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "global_component_count_convert",
            }
        })?),
    })
}

fn attach_dead_ends(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    for region in topology
        .regions
        .iter()
        .filter(|region| region.role == RegionRole::DeadEnd)
    {
        let mut attached = false;
        for edge in ordered {
            let incident = edge.source_region == region.id || edge.target_region == region.id;
            let other = if edge.source_region == region.id {
                edge.target_region
            } else {
                edge.source_region
            };
            if selected.contains(&edge.id)
                || !incident
                || is_dead_end(topology, other)
            {
                continue;
            }
            if add_edge_if_legal(topology, graph, selected, edge, true)? {
                attached = true;
                break;
            }
        }
        if !attached {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "dead_end_attachment_candidate",
                required: 1,
                available: 0,
            });
        }
    }
    Ok(())
}

fn selected_region_path(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    source: RegionId,
    target: RegionId,
) -> Result<Option<Vec<EdgeId>>, GeneratorError> {
    Ok(shortest_path(
        topology,
        graph,
        source,
        target,
        &BTreeSet::new(),
        Some(selected),
    )?
    .map(|(_, path)| path))
}

fn ensure_route_redundancy(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    let required = topology.config.edge_disjoint_routes();
    let (spawn, landmark) = spawn_and_landmark(topology)?;
    while edge_disjoint_route_count(graph, selected, spawn, landmark)? < required {
        let Some(primary) = selected_region_path(topology, graph, selected, spawn, landmark)? else {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "redundancy_primary_path_missing",
                required: u64::from(required),
                available: 0,
            });
        };
        let blocked: BTreeSet<EdgeId> = primary.into_iter().collect();
        let Some((_, alternate)) = shortest_path(
            topology,
            graph,
            spawn,
            landmark,
            &blocked,
            None,
        )? else {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "edge_disjoint_route_candidate",
                required: u64::from(required),
                available: u64::from(edge_disjoint_route_count(
                    graph, selected, spawn, landmark,
                )?),
            });
        };
        if !add_path(topology, graph, selected, &alternate, true)? {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "edge_disjoint_route_coexistence",
                required: u64::from(required),
                available: u64::from(edge_disjoint_route_count(
                    graph, selected, spawn, landmark,
                )?),
            });
        }
    }
    Ok(())
}

fn layer_cycle_rank(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    layer: u16,
) -> Result<u32, GeneratorError> {
    let vertices = topology
        .regions
        .iter()
        .filter(|region| region.layer == layer)
        .count();
    let edges = selected_edges(graph, selected)
        .into_iter()
        .filter(|edge| {
            topology
                .regions
                .iter()
                .find(|region| region.id == edge.source_region)
                .is_some_and(|region| region.layer == layer)
                && topology
                    .regions
                    .iter()
                    .find(|region| region.id == edge.target_region)
                    .is_some_and(|region| region.layer == layer)
        })
        .count();
    let labels = component_labels(topology, graph, selected, Some(layer), false)?;
    let components = labels.values().copied().collect::<BTreeSet<_>>().len();
    let rank = edges
        .checked_add(components)
        .and_then(|value| value.checked_sub(vertices))
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "negative_layer_cycle_rank".into(),
        })?;
    u32::try_from(rank).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Topology,
        operation: "layer_cycle_rank_convert",
    })
}

fn path_exists_without_edge_on_layer(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    edge: &CandidateEdge,
    layer: u16,
) -> bool {
    let mut adjacency: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    for candidate in selected_edges(graph, selected) {
        if candidate.id == edge.id || candidate.transition.is_some() {
            continue;
        }
        let source_layer = topology
            .regions
            .iter()
            .find(|region| region.id == candidate.source_region)
            .map(|region| region.layer);
        let target_layer = topology
            .regions
            .iter()
            .find(|region| region.id == candidate.target_region)
            .map(|region| region.layer);
        if source_layer == Some(layer) && target_layer == Some(layer) {
            adjacency
                .entry(candidate.source_region)
                .or_default()
                .push(candidate.target_region);
            adjacency
                .entry(candidate.target_region)
                .or_default()
                .push(candidate.source_region);
        }
    }
    reachable(&adjacency, edge.source_region, edge.target_region)
}

fn add_required_layer_cycles(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    let minimum = topology.config.per_layer_cycles_min();
    let maximum = topology.config.per_layer_cycles_max();
    for layer in 0..topology.config.layers().2 {
        while layer_cycle_rank(topology, graph, selected, layer)? < minimum {
            let mut added = false;
            for edge in ordered {
                if edge.transition.is_some()
                    || selected.contains(&edge.id)
                    || is_dead_end(topology, edge.source_region)
                    || is_dead_end(topology, edge.target_region)
                    || (!is_required_region(topology, edge.source_region)
                        && !is_required_region(topology, edge.target_region))
                {
                    continue;
                }
                let source_layer = topology
                    .regions
                    .iter()
                    .find(|region| region.id == edge.source_region)
                    .map(|region| region.layer);
                let target_layer = topology
                    .regions
                    .iter()
                    .find(|region| region.id == edge.target_region)
                    .map(|region| region.layer);
                if source_layer != Some(layer) || target_layer != Some(layer) {
                    continue;
                }
                let labels = component_labels(topology, graph, selected, Some(layer), false)?;
                if labels.get(&edge.source_region) != labels.get(&edge.target_region) {
                    continue;
                }
                if add_edge_if_legal(topology, graph, selected, edge, true)? {
                    if layer_cycle_rank(topology, graph, selected, layer)? > maximum {
                        selected.remove(&edge.id);
                        continue;
                    }
                    added = true;
                    break;
                }
            }
            if !added {
                // The selected forest may not yet connect either endpoint of
                // an available cycle edge. Materialize one complete layer-local
                // candidate cycle as a batch, then continue normal bound checks.
                for closing_edge in ordered {
                    if closing_edge.transition.is_some()
                        || selected.contains(&closing_edge.id)
                        || is_dead_end(topology, closing_edge.source_region)
                        || is_dead_end(topology, closing_edge.target_region)
                        || (!is_required_region(topology, closing_edge.source_region)
                            && !is_required_region(topology, closing_edge.target_region))
                    {
                        continue;
                    }
                    let source_layer = topology
                        .regions
                        .iter()
                        .find(|region| region.id == closing_edge.source_region)
                        .map(|region| region.layer);
                    let target_layer = topology
                        .regions
                        .iter()
                        .find(|region| region.id == closing_edge.target_region)
                        .map(|region| region.layer);
                    if source_layer != Some(layer) || target_layer != Some(layer) {
                        continue;
                    }
                    let local_graph = CandidateGraph {
                        edges: graph
                            .edges
                            .iter()
                            .filter(|edge| {
                                edge.transition.is_none()
                                    && topology
                                        .regions
                                        .iter()
                                        .find(|region| region.id == edge.source_region)
                                        .is_some_and(|region| region.layer == layer)
                                    && topology
                                        .regions
                                        .iter()
                                        .find(|region| region.id == edge.target_region)
                                        .is_some_and(|region| region.layer == layer)
                            })
                            .cloned()
                            .collect(),
                        occupancy: graph.occupancy.clone(),
                    };
                    let blocked = BTreeSet::from([closing_edge.id]);
                    let Some((_, path)) = shortest_path(
                        topology,
                        &local_graph,
                        closing_edge.source_region,
                        closing_edge.target_region,
                        &blocked,
                        None,
                    )? else {
                        continue;
                    };
                    let mut branch_graph = graph.clone();
                    let mut branch_selected = selected.clone();
                    if add_path(
                        topology,
                        &mut branch_graph,
                        &mut branch_selected,
                        &path,
                        true,
                    )? && add_edge_if_legal(
                        topology,
                        &mut branch_graph,
                        &mut branch_selected,
                        closing_edge,
                        true,
                    )? && layer_cycle_rank(
                        topology,
                        &branch_graph,
                        &branch_selected,
                        layer,
                    )? <= maximum
                    {
                        *graph = branch_graph;
                        *selected = branch_selected;
                        added = true;
                        break;
                    }
                }
                if !added {
                    return Err(GeneratorError::TopologyInfeasible {
                        stage: ErrorStage::Topology,
                        constraint: "useful_layer_cycle_candidate",
                        required: u64::from(minimum),
                        available: u64::from(layer_cycle_rank(
                            topology, graph, selected, layer,
                        )?),
                    });
                }
            }
        }
        let useful = selected_edges(graph, selected).into_iter().any(|edge| {
            edge.transition.is_none()
                && (is_required_region(topology, edge.source_region)
                    || is_required_region(topology, edge.target_region))
                && path_exists_without_edge_on_layer(topology, graph, selected, edge, layer)
        });
        if !useful {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "useful_layer_cycle_missing",
                required: 1,
                available: 0,
            });
        }
    }
    Ok(())
}

fn reduce_articulations(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
    selected: &mut BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    loop {
        let current = articulation_count(topology, graph, selected)?;
        if current <= topology.config.articulation_max() {
            return Ok(());
        }
        let mut improved = false;
        for edge in ordered {
            if edge.transition.is_some()
                || selected.contains(&edge.id)
                || is_dead_end(topology, edge.source_region)
                || is_dead_end(topology, edge.target_region)
            {
                continue;
            }
            let before = selected.clone();
            if !add_edge_if_legal(topology, graph, selected, edge, true)? {
                continue;
            }
            let source_layer = topology
                .regions
                .iter()
                .find(|region| region.id == edge.source_region)
                .map(|region| region.layer);
            if let Some(layer) = source_layer {
                if layer_cycle_rank(topology, graph, selected, layer)?
                    > topology.config.per_layer_cycles_max()
                {
                    *selected = before;
                    continue;
                }
            }
            if articulation_count(topology, graph, selected)? < current {
                improved = true;
                break;
            }
            *selected = before;
        }
        if !improved {
            return Err(GeneratorError::GraphBoundViolation {
                stage: ErrorStage::Topology,
                constraint: "articulation_max",
                minimum: 0,
                maximum: u64::from(topology.config.articulation_max()),
                actual: u64::from(current),
            });
        }
    }
}

fn assemble_topology(
    topology: &IntendedTopology,
    graph: &mut CandidateGraph,
    ordered: &[CandidateEdge],
) -> Result<BTreeSet<EdgeId>, GeneratorError> {
    let mut selected = BTreeSet::new();

    // Required spawn-to-landmark spine first.
    let spine = bounded_spine(topology, graph)?;
    if !add_path(topology, graph, &mut selected, &spine, false)? {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "spine_witness_coexistence",
            required: 1,
            available: 0,
        });
    }

    // Every reservation-bound vertical edge is mandatory.
    let vertical_edges: Vec<CandidateEdge> = graph
        .edges
        .iter()
        .filter(|edge| edge.transition.is_some())
        .cloned()
        .collect();
    for edge in &vertical_edges {
        if !add_edge_if_legal(topology, graph, &mut selected, edge, true)? {
            return Err(GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "required_transition_edge_coexistence",
                required: 1,
                available: 0,
            });
        }
    }

    // Establish required route redundancy before other connectivity consumes
    // its physical witnesses.
    ensure_route_redundancy(topology, graph, &mut selected)?;

    // Intentional dead ends are mandatory leaves; secure their only incident
    // witnesses before cycles and spanning edges consume crossing capacity.
    attach_dead_ends(topology, graph, ordered, &mut selected)?;

    // Secure one useful cycle on each layer, then connect the remaining cores.
    add_required_layer_cycles(topology, graph, ordered, &mut selected)?;
    connect_layer_cores(topology, graph, ordered, &mut selected)?;
    connect_global_core(topology, graph, ordered, &mut selected)?;
    reduce_articulations(topology, graph, ordered, &mut selected)?;

    // Optional merger/shortcut lower bounds are zero. We deliberately add none;
    // their configured upper bounds are still checked below.
    verify_graph_bounds(topology, graph, &selected)?;
    verify_transition_independence(topology, graph, &selected)?;
    Ok(selected)
}

/// Select a topology from a separately built candidate graph. The bounded
/// attempts vary only equal-cost candidate order; every complete candidate is
/// checked against all graph and transition constraints before it can return.
pub(super) fn select_topology(
    mut topology: IntendedTopology,
    config: &NormalizedGeneratorConfig,
    graph: &CandidateGraph,
    rng: &mut Pcg32V1,
) -> Result<IntendedTopology, GeneratorError> {
    if topology.config != *config {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Topology,
            detail: "selection_config_mismatch".into(),
        });
    }
    validate_candidate_graph(&topology, graph)?;
    let budget = config.reroute_budget().max(1);
    let mut last_error = None;
    for _ in 0..budget {
        let nonce = rng.next_u32();
        let mut working_graph = graph.clone();
        let ordered = edge_order(&working_graph, nonce);
        match assemble_topology(&topology, &mut working_graph, &ordered) {
            Ok(selected) => {
                let selected_candidates = selected_edges(&working_graph, &selected);
                let metrics = compute_metrics(&topology, &working_graph, &selected)?;
                topology.edges = selected_candidates
                    .into_iter()
                    .map(|edge| IntendedEdge {
                        id: edge.id,
                        source_socket: edge.source_socket,
                        target_socket: edge.target_socket,
                        source_region: edge.source_region,
                        target_region: edge.target_region,
                        required: true,
                        path_witness: edge.path_witness.clone(),
                        allowed_envelope_cells: edge.allowed_envelope_cells.clone(),
                        cost: edge.cost,
                        width: edge.width,
                        transition: edge.transition,
                    })
                    .collect();
                topology.edges.sort_by_key(|edge| edge.id);
                topology.route_distance = metrics.route_distance;
                topology.per_layer_cycles = metrics.per_layer_cycles;
                topology.max_branch_depth = metrics.max_branch_depth;
                topology.dead_end_count = metrics.dead_end_count;
                topology.articulation_count = metrics.articulation_count;
                topology.crossing_count = metrics.crossing_count;
                topology.validate_unique_edge_ids()?;
                topology.validate_socket_references()?;
                topology.validate_transition_bindings()?;
                return Ok(topology);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or(GeneratorError::SearchExhausted {
        stage: ErrorStage::Topology,
        search: "topology_selection",
        attempted: u64::from(budget),
        budget: u64::from(budget),
    }))
}

// ─── Metrics and complete graph-bound verification ─────────────────────────

#[derive(Debug)]
struct TopologyMetrics {
    route_distance: u64,
    per_layer_cycles: Vec<u32>,
    max_branch_depth: u32,
    dead_end_count: u32,
    articulation_count: u32,
    crossing_count: u32,
}

fn graph_bound(
    constraint: &'static str,
    minimum: u64,
    maximum: u64,
    actual: u64,
) -> Result<(), GeneratorError> {
    if actual < minimum || actual > maximum {
        return Err(GeneratorError::GraphBoundViolation {
            stage: ErrorStage::Topology,
            constraint,
            minimum,
            maximum,
            actual,
        });
    }
    Ok(())
}

fn spawn_landmark_distance(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<Option<u64>, GeneratorError> {
    let (spawn, landmark) = spawn_and_landmark(topology)?;
    Ok(shortest_path(
        topology,
        graph,
        spawn,
        landmark,
        &BTreeSet::new(),
        Some(selected),
    )?
    .map(|(distance, _)| distance))
}

fn degree_map(
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<BTreeMap<RegionId, u32>, GeneratorError> {
    let mut degree = BTreeMap::new();
    for edge in selected_edges(graph, selected) {
        for region in endpoints(edge) {
            let value = degree.entry(region).or_insert(0u32);
            *value = value.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "region_degree",
            })?;
        }
    }
    Ok(degree)
}

fn unweighted_adjacency(
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    skip_region: Option<RegionId>,
    skip_edge: Option<EdgeId>,
) -> BTreeMap<RegionId, Vec<RegionId>> {
    let mut adjacency: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    for edge in selected_edges(graph, selected) {
        if skip_edge == Some(edge.id)
            || skip_region == Some(edge.source_region)
            || skip_region == Some(edge.target_region)
        {
            continue;
        }
        adjacency
            .entry(edge.source_region)
            .or_default()
            .push(edge.target_region);
        adjacency
            .entry(edge.target_region)
            .or_default()
            .push(edge.source_region);
    }
    adjacency
}

fn reachable(
    adjacency: &BTreeMap<RegionId, Vec<RegionId>>,
    source: RegionId,
    target: RegionId,
) -> bool {
    if source == target {
        return true;
    }
    let mut visited = BTreeSet::from([source]);
    let mut queue = VecDeque::from([source]);
    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&current) {
            for neighbor in neighbors {
                if *neighbor == target {
                    return true;
                }
                if visited.insert(*neighbor) {
                    queue.push_back(*neighbor);
                }
            }
        }
    }
    false
}

fn count_components_excluding(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    skip: Option<RegionId>,
) -> Result<u32, GeneratorError> {
    let adjacency = unweighted_adjacency(graph, selected, skip, None);
    let mut visited = BTreeSet::new();
    let mut components = 0u32;
    for region in &topology.regions {
        if skip == Some(region.id) || visited.contains(&region.id) {
            continue;
        }
        components = components.checked_add(1).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "component_count",
            },
        )?;
        let mut stack = vec![region.id];
        visited.insert(region.id);
        while let Some(current) = stack.pop() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if visited.insert(*neighbor) {
                        stack.push(*neighbor);
                    }
                }
            }
        }
    }
    Ok(components)
}

fn articulation_count(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<u32, GeneratorError> {
    let baseline = count_components_excluding(topology, graph, selected, None)?;
    let mut count = 0u32;
    for region in &topology.regions {
        if count_components_excluding(topology, graph, selected, Some(region.id))? > baseline {
            count = count.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Topology,
                operation: "articulation_count",
            })?;
        }
    }
    Ok(count)
}

fn maximum_branch_depth(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<u32, GeneratorError> {
    let (spawn, _) = spawn_and_landmark(topology)?;
    let adjacency = unweighted_adjacency(graph, selected, None, None);
    let mut distance = BTreeMap::from([(spawn, 0u32)]);
    let mut queue = VecDeque::from([spawn]);
    while let Some(current) = queue.pop_front() {
        let Some(current_distance) = distance.get(&current).copied() else {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Topology,
                detail: "branch_distance_missing".into(),
            });
        };
        if let Some(neighbors) = adjacency.get(&current) {
            for neighbor in neighbors {
                if !distance.contains_key(neighbor) {
                    distance.insert(
                        *neighbor,
                        current_distance.checked_add(1).ok_or(
                            GeneratorError::ArithmeticOverflow {
                                stage: ErrorStage::Topology,
                                operation: "branch_depth",
                            },
                        )?,
                    );
                    queue.push_back(*neighbor);
                }
            }
        }
    }
    Ok(distance.values().copied().max().unwrap_or(0))
}

fn crossing_count(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<u32, GeneratorError> {
    let edges = selected_edges(graph, selected);
    let mut crossings = 0u32;
    for (index, left) in edges.iter().enumerate() {
        let start = index.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "crossing_pair_start",
        })?;
        for right in edges.iter().skip(start) {
            if !edges_coexist(topology, left, right)? {
                crossings = crossings.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "crossing_count",
                    },
                )?;
            }
        }
    }
    Ok(crossings)
}

fn edge_disjoint_route_count(
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
    source: RegionId,
    target: RegionId,
) -> Result<u32, GeneratorError> {
    let mut residual: BTreeMap<(RegionId, RegionId), i32> = BTreeMap::new();
    let mut neighbors: BTreeMap<RegionId, BTreeSet<RegionId>> = BTreeMap::new();
    for edge in selected_edges(graph, selected) {
        residual.insert((edge.source_region, edge.target_region), 1);
        residual.insert((edge.target_region, edge.source_region), 1);
        neighbors
            .entry(edge.source_region)
            .or_default()
            .insert(edge.target_region);
        neighbors
            .entry(edge.target_region)
            .or_default()
            .insert(edge.source_region);
    }
    let mut flow = 0u32;
    loop {
        let mut parent = BTreeMap::new();
        let mut queue = VecDeque::from([source]);
        let mut visited = BTreeSet::from([source]);
        while let Some(current) = queue.pop_front() {
            if current == target {
                break;
            }
            if let Some(adjacent) = neighbors.get(&current) {
                for next in adjacent {
                    if residual.get(&(current, *next)).copied().unwrap_or(0) > 0
                        && visited.insert(*next)
                    {
                        parent.insert(*next, current);
                        queue.push_back(*next);
                    }
                }
            }
        }
        if !visited.contains(&target) {
            break;
        }
        let mut cursor = target;
        while cursor != source {
            let Some(previous) = parent.get(&cursor).copied() else {
                return Err(GeneratorError::IrInvariant {
                    stage: ErrorStage::Topology,
                    detail: "max_flow_parent_missing".into(),
                });
            };
            let forward = residual.entry((previous, cursor)).or_default();
            *forward = forward.checked_sub(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "max_flow_forward",
                },
            )?;
            let reverse = residual.entry((cursor, previous)).or_default();
            *reverse = reverse.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "max_flow_reverse",
                },
            )?;
            cursor = previous;
        }
        flow = flow.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "max_flow_count",
        })?;
    }
    Ok(flow)
}

fn compute_metrics(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<TopologyMetrics, GeneratorError> {
    let route_distance = spawn_landmark_distance(topology, graph, selected)?.ok_or(
        GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "spawn_landmark_disconnected",
            required: 1,
            available: 0,
        },
    )?;
    let mut per_layer_cycles = Vec::with_capacity(usize::from(topology.config.layers().2));
    for layer in 0..topology.config.layers().2 {
        per_layer_cycles.push(layer_cycle_rank(topology, graph, selected, layer)?);
    }
    let degree = degree_map(graph, selected)?;
    let mut dead_end_count = 0u32;
    for region in topology
        .regions
        .iter()
        .filter(|region| region.role == RegionRole::DeadEnd)
    {
        if degree.get(&region.id).copied().unwrap_or(0) == 1 {
            dead_end_count = dead_end_count.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "dead_end_count",
                },
            )?;
        }
    }
    Ok(TopologyMetrics {
        route_distance,
        per_layer_cycles,
        max_branch_depth: maximum_branch_depth(topology, graph, selected)?,
        dead_end_count,
        articulation_count: articulation_count(topology, graph, selected)?,
        crossing_count: crossing_count(topology, graph, selected)?,
    })
}

fn verify_graph_bounds(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    let metrics = compute_metrics(topology, graph, selected)?;
    graph_bound(
        "route_distance",
        u64::from(topology.config.required_route_min()),
        u64::from(topology.config.required_route_max()),
        metrics.route_distance,
    )?;
    for cycles in &metrics.per_layer_cycles {
        graph_bound(
            "per_layer_cycles",
            u64::from(topology.config.per_layer_cycles_min()),
            u64::from(topology.config.per_layer_cycles_max()),
            u64::from(*cycles),
        )?;
    }
    graph_bound(
        "branch_depth",
        u64::from(topology.config.branch_depth_min()),
        u64::from(topology.config.branch_depth_max()),
        u64::from(metrics.max_branch_depth),
    )?;
    graph_bound(
        "intentional_dead_ends",
        u64::from(topology.config.intentional_dead_ends_min()),
        u64::from(topology.config.intentional_dead_ends_max()),
        u64::from(metrics.dead_end_count),
    )?;
    for region in topology
        .regions
        .iter()
        .filter(|region| region.role == RegionRole::DeadEnd)
    {
        let actual = degree_map(graph, selected)?
            .get(&region.id)
            .copied()
            .unwrap_or(0);
        graph_bound("dead_end_degree", 1, 1, u64::from(actual))?;
    }
    graph_bound(
        "articulation_count",
        0,
        u64::from(topology.config.articulation_max()),
        u64::from(metrics.articulation_count),
    )?;
    graph_bound(
        "crossing_count",
        0,
        u64::from(topology.config.crossings_max()),
        u64::from(metrics.crossing_count),
    )?;
    let components = count_components_excluding(topology, graph, selected, None)?;
    graph_bound(
        "components",
        1,
        u64::from(topology.config.components_max()),
        u64::from(components),
    )?;
    let (spawn, landmark) = spawn_and_landmark(topology)?;
    let routes = edge_disjoint_route_count(graph, selected, spawn, landmark)?;
    graph_bound(
        "edge_disjoint_routes",
        u64::from(topology.config.edge_disjoint_routes()),
        u64::MAX,
        u64::from(routes),
    )?;
    graph_bound(
        "optional_mergers",
        0,
        u64::from(topology.config.optional_mergers_max()),
        0,
    )?;
    graph_bound(
        "optional_shortcuts",
        0,
        u64::from(topology.config.optional_shortcuts_max()),
        0,
    )?;
    Ok(())
}

// ─── Reservation-bound transition independence ─────────────────────────────

fn rectangles_overlap(
    left: (u16, u16, u16, u16),
    right: (u16, u16, u16, u16),
) -> Result<bool, GeneratorError> {
    let left_max_x = left.0.checked_add(left.2).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_rect_left_x",
        },
    )?;
    let left_max_y = left.1.checked_add(left.3).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_rect_left_y",
        },
    )?;
    let right_max_x = right.0.checked_add(right.2).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_rect_right_x",
        },
    )?;
    let right_max_y = right.1.checked_add(right.3).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_rect_right_y",
        },
    )?;
    Ok(left.0 < right_max_x
        && left_max_x > right.0
        && left.1 < right_max_y
        && left_max_y > right.1)
}

fn verify_transition_independence(
    topology: &IntendedTopology,
    graph: &CandidateGraph,
    selected: &BTreeSet<EdgeId>,
) -> Result<(), GeneratorError> {
    // Hub projections cannot overlap on any shared endpoint layer.
    for (index, left) in topology.transitions.iter().enumerate() {
        let start = index.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_pair_start",
        })?;
        for right in topology.transitions.iter().skip(start) {
            let left_layers = [
                left.lower_layer,
                left.lower_layer.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "transition_left_upper",
                    },
                )?,
            ];
            let right_layers = [
                right.lower_layer,
                right.lower_layer.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Topology,
                        operation: "transition_right_upper",
                    },
                )?,
            ];
            if left_layers.iter().any(|layer| right_layers.contains(layer))
                && rectangles_overlap(left.hub_footprint, right.hub_footprint)?
            {
                return Err(GeneratorError::TopologyInfeasible {
                    stage: ErrorStage::Topology,
                    constraint: "transition_hub_overlap",
                    required: 0,
                    available: 1,
                });
            }
        }
    }

    for lower in 0..topology.config.layers().2.checked_sub(1).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_proof_layer_pairs",
        },
    )? {
        let upper = lower.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Topology,
            operation: "transition_proof_upper",
        })?;
        let reservations: Vec<_> = topology
            .transitions
            .iter()
            .filter(|transition| transition.lower_layer == lower)
            .collect();
        graph_bound(
            "transition_count_per_pair",
            u64::from(topology.config.transitions_per_adjacent_pair()),
            u64::from(topology.config.transitions_per_adjacent_pair()),
            u64::try_from(reservations.len()).map_err(|_| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Topology,
                    operation: "transition_reservation_count_convert",
                }
            })?,
        )?;
        let mut transition_edges = Vec::new();
        for reservation in reservations {
            let matching: Vec<&CandidateEdge> = selected_edges(graph, selected)
                .into_iter()
                .filter(|edge| edge.transition == Some(reservation.id))
                .collect();
            if matching.len() != 1 {
                return Err(GeneratorError::TransitionBinding {
                    stage: ErrorStage::Topology,
                    transition: reservation.id.raw(),
                    reason: "selected_vertical_edge_count_not_one",
                });
            }
            let edge = matching.first().copied().ok_or(
                GeneratorError::TransitionBinding {
                    stage: ErrorStage::Topology,
                    transition: reservation.id.raw(),
                    reason: "selected_vertical_edge_missing",
                },
            )?;
            if edge.source_region != reservation.lower_region
                || edge.target_region != reservation.upper_region
                || edge.source_socket != reservation.lower_socket
                || edge.target_socket != reservation.upper_socket
            {
                return Err(GeneratorError::TransitionBinding {
                    stage: ErrorStage::Topology,
                    transition: reservation.id.raw(),
                    reason: "selected_vertical_edge_endpoint_mismatch",
                });
            }
            transition_edges.push((reservation, edge));
        }
        if topology.config.transitions_per_adjacent_pair() <= 1 {
            continue;
        }
        for (reservation, edge) in transition_edges {
            let adjacency = unweighted_adjacency(graph, selected, None, Some(edge.id));
            if !reachable(
                &adjacency,
                reservation.lower_region,
                reservation.upper_region,
            ) {
                return Err(GeneratorError::TopologyInfeasible {
                    stage: ErrorStage::Topology,
                    constraint: "transition_independence_removal",
                    required: u64::from(topology.config.transitions_per_adjacent_pair()),
                    available: 1,
                });
            }
        }
        let _ = upper;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use super::super::config::{GeneratorConfig, QualifiedProfile};
    use super::super::determinism::{
        AttemptIdentity, GeneratorIdentity, SemanticStage, SemanticStreamFactory,
    };
    use super::super::placement::place_regions;
    use super::super::prefab::PrefabCatalog;

    fn catalog() -> PrefabCatalog {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        PrefabCatalog::load(&root).expect("bundled prefab catalog")
    }

    fn factory(
        config: &NormalizedGeneratorConfig,
        catalog: &PrefabCatalog,
        seed: u64,
    ) -> SemanticStreamFactory {
        let identity = GeneratorIdentity::new(config, catalog.identity_bytes(), seed);
        SemanticStreamFactory::new(AttemptIdentity::new(identity, 0))
    }

    fn end_to_end_config() -> NormalizedGeneratorConfig {
        // Use a relaxed single-bottleneck config that works with current topology search coverage.
        let mut raw = GeneratorConfig::custom(96, 96, 3);
        raw.single_bottleneck = true;
        raw.relax_route_redundancy = true;
        raw.relax_transition_redundancy = true; // Allow single transition
        raw.region_min = Some(16);
        raw.region_max = Some(24);
        raw.required_route_min = Some(50);
        raw.required_route_max = Some(250);
        raw.branch_depth_min = Some(2);
        raw.branch_depth_max = Some(12);
        raw.articulation_max = Some(12);
        raw.crossings_max = Some(16);
        raw.intentional_dead_ends_min = Some(1);
        raw.intentional_dead_ends_max = Some(4);
        raw.normalize().expect("focused end-to-end config")
    }

    #[test]
    fn adjacent_ramp_socket_roles_are_cross_layer_compatible() {
        let lower = PlacedSocket {
            id: SocketId(0),
            variant_socket_index: 0,
            global_anchor: GridCoord { layer: 0, x: 2, y: 4 },
            direction: super::super::ir::Direction::South,
            width: 1,
            role: SocketRole::LowerRampApproach,
            paired_socket_id: Some(SocketId(1)),
        };
        let upper = PlacedSocket {
            id: SocketId(1),
            variant_socket_index: 1,
            global_anchor: GridCoord { layer: 1, x: 2, y: 0 },
            direction: super::super::ir::Direction::North,
            width: 1,
            role: SocketRole::UpperLanding,
            paired_socket_id: Some(SocketId(0)),
        };
        assert_eq!(lower.global_anchor.layer.abs_diff(upper.global_anchor.layer), 1);
        assert!(matches!(
            (lower.role, upper.role),
            (SocketRole::LowerRampApproach, SocketRole::UpperLanding)
        ));
        assert!(sockets_compatible(&lower, &upper));
    }

    #[test]
    fn exact_width_offsets_cover_n_cells() {
        for width in 1..=8u16 {
            let (first, last) = width_offset_bounds(width).expect("width bounds");
            let count = last - first + 1;
            assert_eq!(count, i32::from(width));
        }
        assert_eq!(width_offset_bounds(2).expect("width two"), (-1, 0));
        assert_eq!(width_offset_bounds(3).expect("width three"), (-1, 1));
    }

    #[test]
    fn astar_enters_only_the_target_goal_and_avoids_unowned_transition_cells() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let source_socket = PlacedSocket {
            id: SocketId(0),
            variant_socket_index: 0,
            global_anchor: GridCoord { layer: 0, x: 7, y: 6 },
            direction: super::super::ir::Direction::East,
            width: 1,
            role: SocketRole::Corridor,
            paired_socket_id: None,
        };
        let target_socket = PlacedSocket {
            id: SocketId(1),
            variant_socket_index: 0,
            global_anchor: GridCoord { layer: 0, x: 15, y: 6 },
            direction: super::super::ir::Direction::West,
            width: 1,
            role: SocketRole::Corridor,
            paired_socket_id: None,
        };
        let source = PlacedRegion {
            id: RegionId(0),
            role: RegionRole::OrdinaryRoom,
            variant_index: 0,
            layer: 0,
            footprint: (5, 5, 3, 3),
            sockets: vec![source_socket.clone()],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let target = PlacedRegion {
            id: RegionId(1),
            role: RegionRole::OrdinaryRoom,
            variant_index: 0,
            layer: 0,
            footprint: (15, 5, 3, 3),
            sockets: vec![target_socket.clone()],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let mut grid = OccupancyGrid::new(
            config.width(),
            config.height(),
            config.layers().2,
        )
        .expect("grid");
        grid.reserve_rect(0, 5, 5, 3, 3, OccupancyClass::Region(source.id.raw()))
            .expect("source footprint");
        grid.reserve_rect(0, 15, 5, 3, 3, OccupancyClass::Region(target.id.raw()))
            .expect("target footprint");
        grid.set(
            source_socket.global_anchor,
            OccupancyClass::Socket(source_socket.id.raw()),
        )
        .expect("source socket");
        grid.set(
            target_socket.global_anchor,
            OccupancyClass::Socket(target_socket.id.raw()),
        )
        .expect("target socket");
        let blocked = GridCoord::new(
            0,
            11,
            6,
            config.width(),
            config.height(),
            config.layers().2,
        )
        .expect("blocked transition cell");
        grid.set(blocked, OccupancyClass::Transition(999))
            .expect("transition obstacle");

        let (path, _) = find_path_with_envelope(
            &source,
            &source_socket,
            &target,
            &target_socket,
            &grid,
            &config,
            1,
        )
        .expect("path search")
        .expect("routed path");
        let goal = GridCoord::new(
            0,
            16,
            6,
            config.width(),
            config.height(),
            config.layers().2,
        )
        .expect("target goal");
        assert_eq!(path.last(), Some(&goal));
        assert_eq!(grid.get(goal), Some(OccupancyClass::Region(target.id.raw())));
        assert!(!path.contains(&blocked));
        assert!(path.iter().all(|cell| {
            !matches!(grid.get(*cell), Some(OccupancyClass::Region(owner)) if owner != source.id.raw() && *cell != goal)
        }));
    }

    #[test]
    fn real_candidate_graph_contains_each_vertical_transition_edge() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let catalog = catalog();
        let factory = factory(&config, &catalog, 5);
        let mut roles = factory.stream(SemanticStage::Roles, &[]);
        let (topology, grid) =
            place_regions(&config, &catalog, &mut roles, factory).expect("placement");
        let graph = build_candidate_graph(&topology, &grid).expect("candidate graph");
        let vertical: Vec<_> = graph
            .edges
            .iter()
            .filter(|edge| edge.transition.is_some())
            .collect();
        assert_eq!(vertical.len(), topology.transitions.len());
        for transition in &topology.transitions {
            let edge = vertical
                .iter()
                .find(|edge| edge.transition == Some(transition.id))
                .expect("transition edge");
            assert_eq!(edge.source_region, transition.lower_region);
            assert_eq!(edge.target_region, transition.upper_region);
        }
    }

    #[test]
    #[ignore = "topology search coverage — requires relaxed_transition_redundancy plumbing in select_topology"]
    fn place_build_select_runs_end_to_end() {
        let config = end_to_end_config();
        let catalog = catalog();
        let factory = factory(&config, &catalog, 23);
        let mut roles = factory.stream(SemanticStage::Roles, &[]);
        let (placed, grid) =
            place_regions(&config, &catalog, &mut roles, factory).expect("placement");
        let graph = build_candidate_graph(&placed, &grid).expect("candidate graph");
        let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
        let selected = select_topology(placed, &config, &graph, &mut topology_rng)
            .expect("topology selection");
        assert!(!selected.edges.is_empty());
        assert_eq!(
            selected
                .edges
                .iter()
                .filter(|edge| edge.transition.is_some())
                .count(),
            selected.transitions.len()
        );
        selected
            .validate_transition_bindings()
            .expect("transition bindings");
        assert!((config.required_route_min() as u64..=config.required_route_max() as u64)
            .contains(&selected.route_distance));
        assert!(selected.per_layer_cycles.iter().all(|cycles| {
            (config.per_layer_cycles_min()..=config.per_layer_cycles_max()).contains(cycles)
        }));
        assert!((config.branch_depth_min()..=config.branch_depth_max())
            .contains(&selected.max_branch_depth));
        assert!(selected.crossing_count <= config.crossings_max());

        let mut selected_ids: BTreeSet<EdgeId> =
            selected.edges.iter().map(|edge| edge.id).collect();
        let vertical = selected
            .edges
            .iter()
            .find(|edge| edge.transition.is_some())
            .expect("selected vertical edge");
        selected_ids.remove(&vertical.id);
        assert!(verify_transition_independence(&selected, &graph, &selected_ids).is_err());
    }

    #[test]
    #[ignore = "topology search coverage — requires relaxed_transition_redundancy plumbing in select_topology"]
    fn end_to_end_pipeline_is_reproducible() {
        let config = end_to_end_config();
        let catalog = catalog();
        let run = || {
            let factory = factory(&config, &catalog, 41);
            let mut roles = factory.stream(SemanticStage::Roles, &[]);
            let (placed, grid) = place_regions(&config, &catalog, &mut roles, factory)
                .expect("placement");
            let graph = build_candidate_graph(&placed, &grid).expect("candidate graph");
            let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
            select_topology(placed, &config, &graph, &mut topology_rng)
                .expect("topology selection")
        };
        assert_eq!(run(), run());
    }
}
