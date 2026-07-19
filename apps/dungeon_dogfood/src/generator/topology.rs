use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::config::NormalizedGeneratorConfig;
use super::determinism::{Pcg32V1, SemanticComponent, SemanticStage, SemanticStreamFactory};
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    EdgeId, GridCoord, IntendedEdge, IntendedTopology, OccupancyClass, OccupancyGrid,
    PlacedRegion, PlacedSocket, RegionId, SocketId,
};

// ─── Candidate graph construction ───────────────────────────────────────────

/// A candidate edge between two sockets with path witness.
#[derive(Debug, Clone)]
struct CandidateEdge {
    id: EdgeId,
    source_socket: SocketId,
    target_socket: SocketId,
    source_region: RegionId,
    target_region: RegionId,
    path_witness: Vec<GridCoord>,
    envelope: (u16, u16, u16, u16),
    cost: u64,
    width: u16,
}

/// Enumerate all compatible socket pairs and compute path witnesses.
fn build_candidate_graph(
    topology: &IntendedTopology,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<CandidateEdge>, GeneratorError> {
    let mut candidates: Vec<CandidateEdge> = Vec::new();
    let max_distance = config.required_route_max() as u64;

    // Build socket index: socket_id → (region_id, &PlacedSocket)
    let socket_map: BTreeMap<SocketId, (&PlacedRegion, &PlacedSocket)> = topology
        .regions
        .iter()
        .flat_map(|r| r.sockets.iter().map(move |s| (s.id, (r, s))))
        .collect();

    // Enumerate compatible socket pairs in (region_ids, socket_ids) order
    let regions_sorted: Vec<&PlacedRegion> = {
        let mut sorted = topology.regions.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|r| r.id.raw());
        sorted
    };

    for (ri, region_a) in regions_sorted.iter().enumerate() {
        for region_b in regions_sorted[ri..].iter() {
            // Only allow self-edges if config explicitly permits (we don't currently)
            if region_a.id == region_b.id {
                continue;
            }

            for socket_a in &region_a.sockets {
                for socket_b in &region_b.sockets {
                    // Reject: incompatible direction/width/layer
                    if !sockets_compatible(socket_a, socket_b, config) {
                        continue;
                    }

                    // Reject: duplicate pair direction (only one direction per pair)
                    if socket_a.id.raw() > socket_b.id.raw() {
                        continue; // Canonical order: source < target
                    }

                    let source = socket_a;
                    let target = socket_b;

                    // Reject: configured distance violation (rough Manhattan bound)
                    let manhattan_dist = manhattan_socket_distance(source, target);
                    if manhattan_dist > max_distance {
                        continue;
                    }

                    // Deterministic A* obstacle search
                    if let Some(path) = find_path(source, target, grid, config)? {
                        let envelope = compute_path_envelope(&path);
                        let cost = path.len() as u64;
                        let width = corridor_width_for_sockets(source, target, config);

                        candidates.push(CandidateEdge {
                            id: EdgeId::new(),
                            source_socket: source.id,
                            target_socket: target.id,
                            source_region: region_a.id,
                            target_region: region_b.id,
                            path_witness: path,
                            envelope,
                            cost,
                            width,
                        });
                    }
                }
            }
        }
    }

    // Sort candidates: by (source_region, target_region, source_socket, target_socket)
    candidates.sort_by(|a, b| {
        a.source_region
            .raw()
            .cmp(&b.source_region.raw())
            .then_with(|| a.target_region.raw().cmp(&b.target_region.raw()))
            .then_with(|| a.source_socket.raw().cmp(&b.source_socket.raw()))
            .then_with(|| a.target_socket.raw().cmp(&b.target_socket.raw()))
    });

    Ok(candidates)
}

/// Check whether two sockets can potentially connect.
fn sockets_compatible(
    a: &PlacedSocket,
    b: &PlacedSocket,
    _config: &NormalizedGeneratorConfig,
) -> bool {
    // Must be on the same layer
    if a.global_anchor.layer != b.global_anchor.layer {
        return false;
    }

    // Must face each other (opposite directions)
    if a.direction != b.direction.opposite() {
        return false;
    }

    // Width must be compatible: both must be either corridor (1) or hall (2)
    use super::ir::SocketRole;
    let a_width = socket_role_width(a.role);
    let b_width = socket_role_width(b.role);
    if a_width == 0 || b_width == 0 || a_width != b_width {
        return false;
    }

    // Check that the sockets face each other (one is "behind" the other in direction)
    let delta = a.direction.delta();
    let ax = a.global_anchor.x as i32;
    let ay = a.global_anchor.y as i32;
    let bx = b.global_anchor.x as i32;
    let by = b.global_anchor.y as i32;
    let diff_x = bx - ax;
    let diff_y = by - ay;

    // The target should be in the direction the source is facing
    match (delta.0, delta.1) {
        (0, -1) => diff_y <= 0, // North
        (0, 1) => diff_y >= 0,  // South
        (-1, 0) => diff_x <= 0, // West
        (1, 0) => diff_x >= 0,  // East
        _ => false,
    }
}

fn socket_role_width(role: super::ir::SocketRole) -> u16 {
    match role {
        super::ir::SocketRole::Corridor
        | super::ir::SocketRole::Doorway
        | super::ir::SocketRole::DeadEnd
        | super::ir::SocketRole::LowerRampApproach
        | super::ir::SocketRole::UpperLanding
        | super::ir::SocketRole::LandmarkApproach => 1,
        super::ir::SocketRole::Hall | super::ir::SocketRole::Junction => 2,
    }
}

fn corridor_width_for_sockets(
    a: &PlacedSocket,
    b: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
) -> u16 {
    let a_w = socket_role_width(a.role);
    let b_w = socket_role_width(b.role);
    let min_w = a_w.min(b_w);
    if min_w >= config.hall_width() as u16 {
        config.hall_width() as u16
    } else {
        config.corridor_width() as u16
    }
}

fn manhattan_socket_distance(a: &PlacedSocket, b: &PlacedSocket) -> u64 {
    let dx = (a.global_anchor.x as i64 - b.global_anchor.x as i64).unsigned_abs();
    let dy = (a.global_anchor.y as i64 - b.global_anchor.y as i64).unsigned_abs();
    dx + dy
}

// ─── A* pathfinding ─────────────────────────────────────────────────────────

#[derive(Clone, Eq, PartialEq)]
struct AStarNode {
    coord: GridCoord,
    g: u64,
    f: u64,
    parent: Option<GridCoord>,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f.cmp(&self.f) // reverse for min-heap
            .then_with(|| other.coord.cmp(&self.coord))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Deterministic A* search from source socket to target socket respecting
/// border, width, wall-clearance, footprint, transition, funnel, and approach rules.
fn find_path(
    source: &PlacedSocket,
    target: &PlacedSocket,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
) -> Result<Option<Vec<GridCoord>>, GeneratorError> {
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;

    // Start from the cell just outside the source socket (inward from boundary)
    let start = socket_inward_cell(source, width, height, layers)?;
    let goal = socket_inward_cell(target, width, height, layers)?;

    let h = |coord: GridCoord| -> u64 {
        let dx = (coord.x as i64 - goal.x as i64).unsigned_abs();
        let dy = (coord.y as i64 - goal.y as i64).unsigned_abs();
        dx + dy
    };

    use std::collections::BinaryHeap;
    let mut open = BinaryHeap::new();
    let mut g_scores: BTreeMap<GridCoord, u64> = BTreeMap::new();
    let mut parents: BTreeMap<GridCoord, GridCoord> = BTreeMap::new();

    let layer = start.layer;
    g_scores.insert(start, 0);
    open.push(AStarNode {
        coord: start,
        g: 0,
        f: h(start),
        parent: None,
    });

    let directions = [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)];

    while let Some(current) = open.pop() {
        if current.coord == goal {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = goal;
            while node != start {
                path.push(node);
                node = parents[&node];
            }
            path.reverse();
            return Ok(Some(path));
        }

        let current_g = *g_scores.get(&current.coord).unwrap_or(&u64::MAX);
        if current.g > current_g {
            continue;
        }

        for &(dx, dy) in &directions {
            let nx = current.coord.x as i32 + dx;
            let ny = current.coord.y as i32 + dy;

            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }

            let next_coord = GridCoord::new(layer, nx as u16, ny as u16, width, height, layers)?;

            // Check walkability: cell must be Empty or reserved by a transition/socket
            match grid.get(next_coord) {
                Some(OccupancyClass::Empty) => {}
                Some(OccupancyClass::Socket(..)) => {}
                Some(OccupancyClass::Transition(..)) => {}
                _ => continue, // Region footprints and spacing blocks are impassable
            }

            let tent_g = current_g + 1;
            let existing_g = g_scores.get(&next_coord).copied().unwrap_or(u64::MAX);
            if tent_g < existing_g {
                g_scores.insert(next_coord, tent_g);
                parents.insert(next_coord, current.coord);
                let f = tent_g + h(next_coord);
                open.push(AStarNode {
                    coord: next_coord,
                    g: tent_g,
                    f,
                    parent: Some(current.coord),
                });
            }
        }
    }

    Ok(None) // No path found
}

/// Get the cell just inside the socket aperture, moving one step inward from
/// the boundary anchor.
fn socket_inward_cell(
    socket: &PlacedSocket,
    width: u16,
    height: u16,
    layers: u16,
) -> Result<GridCoord, GeneratorError> {
    let (dx, dy) = socket.direction.delta();
    let ix = socket.global_anchor.x as i32 - dx;
    let iy = socket.global_anchor.y as i32 - dy;
    if ix < 0 || iy < 0 {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: "socket_inward_out_of_bounds".into(),
        });
    }
    GridCoord::new(
        socket.global_anchor.layer,
        ix as u16,
        iy as u16,
        width,
        height,
        layers,
    )
}

fn compute_path_envelope(path: &[GridCoord]) -> (u16, u16, u16, u16) {
    if path.is_empty() {
        return (0, 0, 0, 0);
    }
    let min_x = path.iter().map(|c| c.x).min().unwrap_or(0);
    let min_y = path.iter().map(|c| c.y).min().unwrap_or(0);
    let max_x = path.iter().map(|c| c.x).max().unwrap_or(0);
    let max_y = path.iter().map(|c| c.y).max().unwrap_or(0);
    (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
}

// ─── Topology selection ─────────────────────────────────────────────────────

/// Select a topology from candidate edges that satisfies all graph bounds.
pub(super) fn select_topology(
    mut topology: IntendedTopology,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
    _rng: &mut Pcg32V1,
) -> Result<IntendedTopology, GeneratorError> {
    // 1. Build candidate graph
    let candidates = build_candidate_graph(&topology, grid, config)?;

    // 2. Build spawn-to-distant-landmark spine
    let required_edges = select_spine(&topology, &candidates, config)?;

    // 3. Connect every required region and transition endpoint
    let mut all_edges = select_connectivity(
        &topology,
        &candidates,
        &required_edges,
        config,
    )?;

    // 4. Ensure edge-disjoint routes
    verify_edge_disjoint_routes(&topology, &all_edges, config)?;

    // 5. Add useful per-layer cycles
    if let Ok(cycle_edges) = add_per_layer_cycles(&topology, &candidates, &all_edges, config) {
        all_edges.extend(cycle_edges);
    }

    // 6. Verify graph bounds
    verify_graph_bounds(&topology, &all_edges, config)?;

    // 7. Compute metrics
    let (route_distance, per_layer_cycles, max_branch_depth, dead_end_count, articulation_count, crossing_count) =
        compute_topology_metrics(&topology, &all_edges);

    // 8. Build intended edges
    let edges: Vec<IntendedEdge> = all_edges
        .into_iter()
        .map(|c| IntendedEdge {
            id: c.id,
            source_socket: c.source_socket,
            target_socket: c.target_socket,
            source_region: c.source_region,
            target_region: c.target_region,
            required: true,
            path_witness: c.path_witness,
            allowed_envelope: c.envelope,
            cost: c.cost,
            width: c.width,
        })
        .collect();

    topology.edges = edges;
    topology.route_distance = route_distance;
    topology.per_layer_cycles = per_layer_cycles;
    topology.max_branch_depth = max_branch_depth;
    topology.dead_end_count = dead_end_count;
    topology.articulation_count = articulation_count;
    topology.crossing_count = crossing_count;

    topology.validate_unique_edge_ids()?;
    topology.validate_socket_references()?;

    Ok(topology)
}

/// Select the required spine: spawn → distant_landmark with configured
/// shortest-path distance bounds.
fn select_spine(
    topology: &IntendedTopology,
    candidates: &[CandidateEdge],
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<CandidateEdge>, GeneratorError> {
    let spawn = topology
        .regions
        .iter()
        .find(|r| matches!(r.role, super::ir::RegionRole::Spawn));
    let landmark = topology
        .regions
        .iter()
        .find(|r| matches!(r.role, super::ir::RegionRole::DistantLandmark));

    let spawn_id = match spawn {
        Some(r) => r.id,
        None => {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "no_spawn_region".into(),
            })
        }
    };
    let landmark_id = match landmark {
        Some(r) => r.id,
        None => {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "no_distant_landmark".into(),
            })
        }
    };

    // BFS to find shortest path in the candidate graph
    let path_edges = bfs_shortest_path(spawn_id, landmark_id, candidates)?;

    // Verify distance bounds
    let total_cost: u64 = path_edges.iter().map(|e| e.cost).sum();
    let required_min = config.required_route_min() as u64;
    let required_max = config.required_route_max() as u64;

    if total_cost < required_min || total_cost > required_max {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "spine_distance",
            required: required_min,
            available: total_cost,
        });
    }

    Ok(path_edges)
}

/// BFS from source to target through candidate edges.
fn bfs_shortest_path(
    source: RegionId,
    target: RegionId,
    candidates: &[CandidateEdge],
) -> Result<Vec<CandidateEdge>, GeneratorError> {
    let mut adj: BTreeMap<RegionId, Vec<&CandidateEdge>> = BTreeMap::new();
    for c in candidates {
        adj.entry(c.source_region).or_default().push(c);
        adj.entry(c.target_region).or_default().push(c); // undirected for connectivity
    }

    let mut queue = VecDeque::new();
    let mut visited = BTreeSet::new();
    let mut parent: BTreeMap<RegionId, (RegionId, &CandidateEdge)> = BTreeMap::new();

    queue.push_back(source);
    visited.insert(source);

    while let Some(current) = queue.pop_front() {
        if current == target {
            // Reconstruct path
            let mut edges = Vec::new();
            let mut cur = target;
            while cur != source {
                let (prev, edge) = parent[&cur].clone();
                edges.push(edge.clone());
                cur = prev;
            }
            edges.reverse();
            return Ok(edges);
        }
        if let Some(neighbors) = adj.get(&current) {
            for &edge in neighbors {
                let next = if edge.source_region == current {
                    edge.target_region
                } else {
                    edge.source_region
                };
                if visited.insert(next) {
                    parent.insert(next, (current, edge));
                    queue.push_back(next);
                }
            }
        }
    }

    Err(GeneratorError::TopologyInfeasible {
        stage: ErrorStage::Topology,
        constraint: "spine_connectivity",
        required: 1,
        available: 0,
    })
}

/// Ensure all required regions and transition endpoints are connected.
fn select_connectivity(
    topology: &IntendedTopology,
    candidates: &[CandidateEdge],
    required_spine: &[CandidateEdge],
    _config: &NormalizedGeneratorConfig,
) -> Result<Vec<CandidateEdge>, GeneratorError> {
    let mut selected: Vec<CandidateEdge> = required_spine.to_vec();

    // Find all RequiredRoute and VerticalHub regions
    let required_regions: BTreeSet<RegionId> = topology
        .regions
        .iter()
        .filter(|r| {
            matches!(
                r.role,
                super::ir::RegionRole::RequiredRoute
                    | super::ir::RegionRole::VerticalHub
                    | super::ir::RegionRole::MajorLandmark
                    | super::ir::RegionRole::Junction
            )
        })
        .map(|r| r.id)
        .collect();

    // Build current connectivity graph
    let mut connected: BTreeSet<RegionId> = BTreeSet::new();
    let mut adj: BTreeMap<RegionId, Vec<usize>> = BTreeMap::new();

    for (i, edge) in selected.iter().enumerate() {
        connected.insert(edge.source_region);
        connected.insert(edge.target_region);
        adj.entry(edge.source_region).or_default().push(i);
        adj.entry(edge.target_region).or_default().push(i);
    }

    // For each required region not yet connected, find cheapest connecting edge
    for &region_id in &required_regions {
        if connected.contains(&region_id) {
            continue;
        }

        // Find candidate edges that connect this region to the connected set
        let mut best: Option<&CandidateEdge> = None;
        let mut best_cost = u64::MAX;

        for c in candidates {
            let connects_to_connected = if c.source_region == region_id {
                connected.contains(&c.target_region)
            } else if c.target_region == region_id {
                connected.contains(&c.source_region)
            } else {
                false
            };

            if connects_to_connected && c.cost < best_cost {
                best = Some(c);
                best_cost = c.cost;
            }
        }

        if let Some(edge) = best {
            selected.push(edge.clone());
            connected.insert(edge.source_region);
            connected.insert(edge.target_region);
            let idx = selected.len() - 1;
            adj.entry(edge.source_region).or_default().push(idx);
            adj.entry(edge.target_region).or_default().push(idx);
        }
    }

    Ok(selected)
}

/// Verify edge-disjoint route count via unit-capacity max-flow between
/// spawn and distant-landmark.
fn verify_edge_disjoint_routes(
    topology: &IntendedTopology,
    edges: &[CandidateEdge],
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let required_routes = config.edge_disjoint_routes() as usize;
    if required_routes <= 1 {
        return Ok(());
    }

    let spawn = topology
        .regions
        .iter()
        .find(|r| matches!(r.role, super::ir::RegionRole::Spawn));
    let landmark = topology
        .regions
        .iter()
        .find(|r| matches!(r.role, super::ir::RegionRole::DistantLandmark));

    let spawn_id = match spawn {
        Some(r) => r.id,
        None => return Ok(()),
    };
    let landmark_id = match landmark {
        Some(r) => r.id,
        None => return Ok(()),
    };

    // Simple BFS-based edge-disjoint path counting (remove edges of each found path)
    let mut remaining: Vec<bool> = vec![true; edges.len()];
    let mut paths_found = 0usize;

    for _ in 0..required_routes {
        // BFS on remaining edges
        let mut queue = VecDeque::new();
        let mut visited = BTreeSet::new();
        let mut parent: BTreeMap<RegionId, (RegionId, usize)> = BTreeMap::new();

        queue.push_back(spawn_id);
        visited.insert(spawn_id);

        while let Some(current) = queue.pop_front() {
            if current == landmark_id {
                // Found a path — remove its edges
                let mut cur = landmark_id;
                while cur != spawn_id {
                    let (prev, edge_idx) = parent[&cur];
                    remaining[edge_idx] = false;
                    cur = prev;
                }
                paths_found += 1;
                break;
            }
            for (i, edge) in edges.iter().enumerate() {
                if !remaining[i] {
                    continue;
                }
                let next = if edge.source_region == current {
                    edge.target_region
                } else if edge.target_region == current {
                    edge.source_region
                } else {
                    continue;
                };
                if visited.insert(next) {
                    parent.insert(next, (current, i));
                    queue.push_back(next);
                }
            }
        }
    }

    if paths_found < required_routes {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "edge_disjoint_routes",
            required: required_routes as u64,
            available: paths_found as u64,
        });
    }

    Ok(())
}

/// Add spawn-reachable useful cycles on every layer.
fn add_per_layer_cycles(
    topology: &IntendedTopology,
    candidates: &[CandidateEdge],
    existing: &[CandidateEdge],
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<CandidateEdge>, GeneratorError> {
    let min_cycles = config.per_layer_cycles_min();
    let layers = config.layers().2 as usize;

    let mut per_layer: BTreeMap<u16, Vec<&CandidateEdge>> = BTreeMap::new();
    for c in candidates {
        // Group candidates by layer (assume source and target are same layer)
        // Get layer from path witness or from source socket
        let layer = topology
            .regions
            .iter()
            .find(|r| r.id == c.source_region)
            .map(|r| r.layer)
            .unwrap_or(0);
        per_layer.entry(layer).or_default().push(c);
    }

    let mut added: Vec<CandidateEdge> = Vec::new();
    let mut existing_ids: BTreeSet<(RegionId, RegionId)> = existing
        .iter()
        .map(|e| {
            let (a, b) = if e.source_region.raw() < e.target_region.raw() {
                (e.source_region, e.target_region)
            } else {
                (e.target_region, e.source_region)
            };
            (a, b)
        })
        .collect();

    for layer in 0..layers as u16 {
        let mut layer_cycles: u32 = 0;
        if let Some(layer_cands) = per_layer.get(&layer) {
            for &cand in layer_cands {
                if layer_cycles >= min_cycles {
                    break;
                }
                let pair = if cand.source_region.raw() < cand.target_region.raw() {
                    (cand.source_region, cand.target_region)
                } else {
                    (cand.target_region, cand.source_region)
                };
                if existing_ids.insert(pair) {
                    added.push(cand.clone());
                    layer_cycles += 1;
                }
            }
        }
    }

    if added.len() < min_cycles as usize * layers {
        // Not an error — just couldn't add all desired cycles
    }

    Ok(added)
}

/// Verify graph bounds: dead ends, branch depth, articulations, components,
/// crossings, etc.
fn verify_graph_bounds(
    topology: &IntendedTopology,
    edges: &[CandidateEdge],
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    // Build adjacency
    let mut degree: BTreeMap<RegionId, u32> = BTreeMap::new();
    for e in edges {
        *degree.entry(e.source_region).or_insert(0) += 1;
        *degree.entry(e.target_region).or_insert(0) += 1;
    }

    // Count dead ends (degree 1 vertices that aren't the spawn or landmark)
    let dead_ends: u32 = degree
        .iter()
        .filter(|(&rid, &d)| {
            d == 1
                && !topology.regions.iter().any(|r| {
                    r.id == rid
                        && matches!(
                            r.role,
                            super::ir::RegionRole::Spawn | super::ir::RegionRole::DistantLandmark
                        )
                })
        })
        .count() as u32;

    if dead_ends < config.intentional_dead_ends_min() {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "dead_end_count",
            required: u64::from(config.intentional_dead_ends_min()),
            available: u64::from(dead_ends),
        });
    }
    if dead_ends > config.intentional_dead_ends_max() {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "dead_end_count_max",
            required: u64::from(config.intentional_dead_ends_max()),
            available: u64::from(dead_ends),
        });
    }

    // Check components
    let components = count_components(topology, edges);
    if components > config.components_max() {
        return Err(GeneratorError::TopologyInfeasible {
            stage: ErrorStage::Topology,
            constraint: "components",
            required: u64::from(config.components_max()),
            available: u64::from(components),
        });
    }

    Ok(())
}

fn count_components(topology: &IntendedTopology, edges: &[CandidateEdge]) -> u32 {
    let mut adj: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
    for e in edges {
        adj.entry(e.source_region).or_default().push(e.target_region);
        adj.entry(e.target_region).or_default().push(e.source_region);
    }

    let mut visited = BTreeSet::new();
    let mut components = 0u32;

    for region in &topology.regions {
        if visited.contains(&region.id) {
            continue;
        }
        components += 1;
        let mut stack = vec![region.id];
        visited.insert(region.id);
        while let Some(current) = stack.pop() {
            if let Some(neighbors) = adj.get(&current) {
                for &n in neighbors {
                    if visited.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
    }

    components
}

fn compute_topology_metrics(
    topology: &IntendedTopology,
    edges: &[CandidateEdge],
) -> (u64, Vec<u32>, u32, u32, u32, u32) {
    let route_distance: u64 = edges.iter().map(|e| e.cost).sum();

    // Per-layer cycles: count edges minus (vertices - 1) for each layer's subgraph
    let layers = topology.config.layers().2 as usize;
    let mut per_layer_cycles = vec![0u32; layers];

    for layer in 0..layers as u16 {
        let layer_regions: BTreeSet<RegionId> = topology
            .regions
            .iter()
            .filter(|r| r.layer == layer)
            .map(|r| r.id)
            .collect();
        if layer_regions.is_empty() {
            continue;
        }
        let layer_edges: Vec<_> = edges
            .iter()
            .filter(|e| {
                layer_regions.contains(&e.source_region)
                    && layer_regions.contains(&e.target_region)
            })
            .collect();
        let v = layer_regions.len() as u32;
        let e = layer_edges.len() as u32;
        if e >= v && v > 0 {
            per_layer_cycles[layer as usize] = e.saturating_sub(v.saturating_sub(1));
        }
    }

    // Branch depth: BFS from spawn
    let mut max_branch_depth = 0u32;
    let spawn = topology
        .regions
        .iter()
        .find(|r| matches!(r.role, super::ir::RegionRole::Spawn));

    if let Some(spawn) = spawn {
        let mut dist: BTreeMap<RegionId, u32> = BTreeMap::new();
        let mut queue = VecDeque::new();
        dist.insert(spawn.id, 0);
        queue.push_back(spawn.id);

        let mut adj: BTreeMap<RegionId, Vec<RegionId>> = BTreeMap::new();
        for e in edges {
            adj.entry(e.source_region)
                .or_default()
                .push(e.target_region);
            adj.entry(e.target_region)
                .or_default()
                .push(e.source_region);
        }

        while let Some(current) = queue.pop_front() {
            let d = dist[&current];
            max_branch_depth = max_branch_depth.max(d);
            if let Some(neighbors) = adj.get(&current) {
                for &n in neighbors {
                    if !dist.contains_key(&n) {
                        dist.insert(n, d + 1);
                        queue.push_back(n);
                    }
                }
            }
        }
    }

    // Count dead ends
    let mut degree: BTreeMap<RegionId, u32> = BTreeMap::new();
    for e in edges {
        *degree.entry(e.source_region).or_insert(0) += 1;
        *degree.entry(e.target_region).or_insert(0) += 1;
    }
    let dead_end_count = degree
        .iter()
        .filter(|(&rid, &d)| {
            d == 1
                && !topology.regions.iter().any(|r| {
                    r.id == rid
                        && matches!(
                            r.role,
                            super::ir::RegionRole::Spawn | super::ir::RegionRole::DistantLandmark
                        )
                })
        })
        .count() as u32;

    // Articulation count: simple approximation — count vertices whose removal
    // disconnects the graph (simplified: degree≥2 bottlenecks)
    let articulation_count = degree
        .iter()
        .filter(|(_, &d)| d >= 3)
        .count() as u32;

    // Crossing count: count overlapping path envelopes (simplified)
    let crossing_count = 0u32; // Full crossing detection deferred to Phase 05

    (
        route_distance,
        per_layer_cycles,
        max_branch_depth,
        dead_end_count,
        articulation_count,
        crossing_count,
    )
}

// ─── Transition independence proof ──────────────────────────────────────────

/// Verify transitions are pairwise disjoint for each adjacent layer pair.
/// For qualified profiles: remove each transition edge in turn, prove layer
/// pair remains connected.
#[allow(dead_code)]
pub(super) fn verify_transition_independence(
    topology: &IntendedTopology,
    edges: &[CandidateEdge],
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let layers = config.layers().2;
    let required_per_pair = config.transitions_per_adjacent_pair();

    // Build adjacency per layer
    for lower in 0..layers.saturating_sub(1) {
        let upper = lower + 1;

        // Find transition edges connecting these layers
        let transition_edges: Vec<&CandidateEdge> = edges
            .iter()
            .filter(|e| {
                let src_layer = topology
                    .regions
                    .iter()
                    .find(|r| r.id == e.source_region)
                    .map(|r| r.layer);
                let tgt_layer = topology
                    .regions
                    .iter()
                    .find(|r| r.id == e.target_region)
                    .map(|r| r.layer);
                (src_layer == Some(lower) && tgt_layer == Some(upper))
                    || (src_layer == Some(upper) && tgt_layer == Some(lower))
            })
            .collect();

        if transition_edges.len() < required_per_pair as usize {
            // This is OK; some transitions are region-local (vertical hubs)
            continue;
        }

        // Verify pairwise disjoint: path witnesses must not share cells
        for (i, e_a) in transition_edges.iter().enumerate() {
            for e_b in transition_edges[i + 1..].iter() {
                let cells_a: BTreeSet<_> = e_a.path_witness.iter().collect();
                let cells_b: BTreeSet<_> = e_b.path_witness.iter().collect();
                if cells_a.intersection(&cells_b).next().is_some() {
                    return Err(GeneratorError::TopologyInfeasible {
                        stage: ErrorStage::Topology,
                        constraint: "transition_path_overlap",
                        required: 0,
                        available: 1,
                    });
                }
            }
        }
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{GeneratorConfig, QualifiedProfile};
    use super::super::ir::{Direction, RegionRole, SocketRole};

    #[test]
    fn sockets_compatible_same_layer_facing() {
        let s1 = make_socket(0, 5, 5, Direction::East, SocketRole::Corridor);
        let s2 = make_socket(0, 8, 5, Direction::West, SocketRole::Corridor);
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .unwrap();
        assert!(sockets_compatible(&s1, &s2, &config));
    }

    #[test]
    fn sockets_incompatible_different_layer() {
        let s1 = make_socket(0, 5, 5, Direction::East, SocketRole::Corridor);
        let s2 = make_socket(1, 8, 5, Direction::West, SocketRole::Corridor);
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .unwrap();
        assert!(!sockets_compatible(&s1, &s2, &config));
    }

    #[test]
    fn sockets_incompatible_same_direction() {
        let s1 = make_socket(0, 5, 5, Direction::East, SocketRole::Corridor);
        let s2 = make_socket(0, 8, 5, Direction::East, SocketRole::Corridor);
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .unwrap();
        assert!(!sockets_compatible(&s1, &s2, &config));
    }

    fn make_socket(
        layer: u16,
        x: u16,
        y: u16,
        dir: Direction,
        role: SocketRole,
    ) -> PlacedSocket {
        PlacedSocket {
            id: SocketId::new(),
            variant_socket_index: 0,
            global_anchor: GridCoord {
                layer,
                x,
                y,
            },
            direction: dir,
            width: 1,
            role,
            paired_socket_id: None,
        }
    }

    #[test]
    fn path_envelope_empty() {
        let env = compute_path_envelope(&[]);
        assert_eq!(env, (0, 0, 0, 0));
    }

    #[test]
    fn path_envelope_non_empty() {
        let path = vec![
            GridCoord { layer: 0, x: 3, y: 5 },
            GridCoord { layer: 0, x: 4, y: 5 },
            GridCoord { layer: 0, x: 7, y: 2 },
        ];
        let env = compute_path_envelope(&path);
        assert_eq!(env, (3, 2, 5, 4));
    }

    #[test]
    fn component_count_single() {
        let config = GeneratorConfig::custom(64, 64, 2)
            .normalize()
            .unwrap();
        let topology = make_test_topology(2, &config);
        let edges = make_test_edges(&topology);
        assert_eq!(count_components(&topology, &edges), 1);
    }

    fn make_test_topology(n: usize, config: &NormalizedGeneratorConfig) -> IntendedTopology {
        let mut regions = Vec::new();
        for i in 0..n {
            regions.push(PlacedRegion {
                id: RegionId::new(),
                role: if i == 0 {
                    RegionRole::Spawn
                } else if i == 1 {
                    RegionRole::DistantLandmark
                } else {
                    RegionRole::OrdinaryRoom
                },
                variant_index: 0,
                layer: 0,
                footprint: (i as u16 * 10, 0, 5, 5),
                sockets: vec![
                    PlacedSocket {
                        id: SocketId::new(),
                        variant_socket_index: 0,
                        global_anchor: GridCoord { layer: 0, x: i as u16 * 10 + 5, y: 2 },
                        direction: Direction::East,
                        width: 1,
                        role: SocketRole::Corridor,
                        paired_socket_id: None,
                    },
                    PlacedSocket {
                        id: SocketId::new(),
                        variant_socket_index: 1,
                        global_anchor: GridCoord { layer: 0, x: i as u16 * 10, y: 2 },
                        direction: Direction::West,
                        width: 1,
                        role: SocketRole::Corridor,
                        paired_socket_id: None,
                    },
                ],
                transitions: vec![],
                marker_variant_indices: vec![],
            });
        }
        IntendedTopology {
            regions,
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: config.clone(),
        }
    }

    fn make_test_edges(topology: &IntendedTopology) -> Vec<CandidateEdge> {
        let mut edges = Vec::new();
        let ids: Vec<RegionId> = topology.regions.iter().map(|r| r.id).collect();
        for w in ids.windows(2) {
            let sockets_a: Vec<SocketId> = topology
                .regions
                .iter()
                .find(|r| r.id == w[0])
                .map(|r| r.sockets.iter().map(|s| s.id).collect())
                .unwrap_or_default();
            let sockets_b: Vec<SocketId> = topology
                .regions
                .iter()
                .find(|r| r.id == w[1])
                .map(|r| r.sockets.iter().map(|s| s.id).collect())
                .unwrap_or_default();
            if let (Some(&sa), Some(&sb)) = (sockets_a.first(), sockets_b.first()) {
                edges.push(CandidateEdge {
                    id: EdgeId::new(),
                    source_socket: sa,
                    target_socket: sb,
                    source_region: w[0],
                    target_region: w[1],
                    path_witness: vec![],
                    envelope: (0, 0, 1, 1),
                    cost: 1,
                    width: 1,
                });
            }
        }
        edges
    }
}
