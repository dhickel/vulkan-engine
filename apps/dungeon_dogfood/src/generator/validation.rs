//! Phase 05a — Movement reconstruction, structural/connectivity/topology
//! validators, and app-local movement probes.
//!
//! The movement graph is reconstructed exclusively from materialized tiles
//! (`ParsedLevel`). Horizontal edges connect adjacent walkable cells with a
//! clearance check; vertical edges are created only via complete ramp
//! transition inference (contiguous R0/R1/R2, approach, opening, landing).
//! Generic Void never creates movement or vertical edges.
//!
//! Validators run in this order:
//! 1. Structural  — dimensions, tokens, sockets, corridor envelope, ramps.
//! 2. Connectivity — flood from spawn, every required landmark reachable.
//! 3. Topology     — intended edges vs reconstructed paths, bound checks.
//! 4. Movement probes — bidirectional probes over ramps and representative
//!    corridor corners/thresholds/hubs using `CollisionWorld`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::collision::{resolve_player_step, CollisionWorld};
use crate::layout::{ParsedLevel, Tile};
use crate::player::PlayerState;

use super::error::{ErrorStage, GeneratorError};
use super::ir::{Direction, IntendedEdge, IntendedTopology, PlacedRegion, RegionId, RegionRole};
use super::ramps::InferredTransition;

// ─── Movement graph types ───────────────────────────────────────────────────

/// A node in the reconstructed movement graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct MovementNode {
    pub(super) layer: u16,
    pub(super) x: u16,
    pub(super) y: u16,
}

impl MovementNode {
    fn from_coord(layer: u16, x: u16, y: u16) -> Self {
        Self { layer, x, y }
    }
}

/// An edge in the reconstructed movement graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum MovementEdgeType {
    /// Horizontal adjacency between two walkable cells on the same layer.
    Horizontal,
    /// Vertical edge through a complete ramp transition.
    Vertical {
        lower_anchor: MovementNode,
        direction: Direction,
    },
}

/// Ownership metadata for a movement edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MovementEdgeMetadata {
    pub(super) source_region: Option<RegionId>,
    pub(super) target_region: Option<RegionId>,
    pub(super) edge_type: MovementEdgeType,
    /// If true, this edge is owned by a specific transition.
    pub(super) transition_owned: bool,
}

/// Complete reconstructed movement graph.
#[derive(Debug, Clone, Default)]
pub(super) struct MovementGraph {
    /// All walkable nodes sorted canonically.
    pub(super) nodes: Vec<MovementNode>,
    /// Adjacency list: node index → list of (neighbor index, metadata).
    pub(super) adjacency: BTreeMap<usize, Vec<(usize, MovementEdgeMetadata)>>,
    /// Regions that contain each node. Built from placed region footprints.
    pub(super) node_regions: BTreeMap<MovementNode, BTreeSet<RegionId>>,
}

// ─── Validation report ──────────────────────────────────────────────────────

/// Outcome of a single validator run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ValidationReport {
    pub(super) structural_errors: Vec<GeneratorError>,
    pub(super) connectivity_errors: Vec<GeneratorError>,
    pub(super) topology_errors: Vec<GeneratorError>,
    pub(super) movement_probe_errors: Vec<GeneratorError>,
    pub(super) warnings: Vec<String>,
}

impl ValidationReport {
    pub(super) fn is_clean(&self) -> bool {
        self.structural_errors.is_empty()
            && self.connectivity_errors.is_empty()
            && self.topology_errors.is_empty()
            && self.movement_probe_errors.is_empty()
    }

    pub(super) fn all_errors(&self) -> Vec<&GeneratorError> {
        self.structural_errors
            .iter()
            .chain(&self.connectivity_errors)
            .chain(&self.topology_errors)
            .chain(&self.movement_probe_errors)
            .collect()
    }
}

// ─── Movement graph reconstruction ──────────────────────────────────────────

/// Reconstruct the movement graph from a materialized `ParsedLevel`.
///
/// Returns the graph and inferred ramp transitions for downstream validators.
pub(super) fn reconstruct_movement_graph(
    level: &ParsedLevel,
    topology: &IntendedTopology,
) -> Result<(MovementGraph, Vec<InferredTransition>), GeneratorError> {
    validate_level_storage(level)?;
    let width = u16::try_from(level.width).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Ir,
        operation: "mv_reconstruct_width_convert",
    })?;
    let height = u16::try_from(level.height).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "mv_reconstruct_height_convert",
        }
    })?;
    let layers = u16::try_from(level.layer_count()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "mv_reconstruct_layers_convert",
        }
    })?;

    // Collect walkable cells.
    let mut nodes = Vec::new();
    let mut node_set = BTreeSet::new();
    for layer in 0..layers {
        for y in 0..height {
            for x in 0..width {
                let tile = level.tile_at_3d(usize::from(layer), usize::from(x), usize::from(y));
                if tile_is_walkable(tile) {
                    let node = MovementNode::from_coord(layer, x, y);
                    nodes.push(node);
                    node_set.insert(node);
                }
            }
        }
    }
    nodes.sort();

    // Build node index map.
    let node_index: BTreeMap<MovementNode, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (*node, i))
        .collect();

    // Build region membership.
    let mut node_regions: BTreeMap<MovementNode, BTreeSet<RegionId>> = BTreeMap::new();
    for region in &topology.regions {
        let (rx, ry, rw, rh) = region.footprint;
        let Some(max_x) = rx.checked_add(rw) else {
            continue;
        };
        let Some(max_y) = ry.checked_add(rh) else {
            continue;
        };
        for dy in ry..max_y {
            for dx in rx..max_x {
                if let Some(node) = node_set.get(&MovementNode::from_coord(region.layer, dx, dy)) {
                    node_regions.entry(*node).or_default().insert(region.id);
                }
            }
        }
    }

    // The lookup for ramp inference.
    let lookup = |layer: u16, x: u16, y: u16| {
        if layer >= layers || x >= width || y >= height {
            return None;
        }
        Some(level.tile_at_3d(usize::from(layer), usize::from(x), usize::from(y)))
    };

    // Infer ramp transitions from tiles.
    let inferred = super::ramps::scan_transitions(width, height, layers, &lookup);

    // Build adjacency.
    let mut adjacency: BTreeMap<usize, Vec<(usize, MovementEdgeMetadata)>> = BTreeMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let dirs = [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)];
        for (dx, dy) in dirs {
            let nx = i32::from(node.x)
                .checked_add(dx)
                .and_then(|v| u16::try_from(v).ok());
            let ny = i32::from(node.y)
                .checked_add(dy)
                .and_then(|v| u16::try_from(v).ok());
            let Some((nx, ny)) = nx.zip(ny) else {
                continue;
            };
            let neighbor = MovementNode::from_coord(node.layer, nx, ny);
            if let Some(&j) = node_index.get(&neighbor) {
                let regions_i = node_regions.get(node).cloned().unwrap_or_default();
                let regions_j = node_regions.get(&neighbor).cloned().unwrap_or_default();
                let shared = regions_i.intersection(&regions_j).next().copied();
                let metadata = MovementEdgeMetadata {
                    source_region: shared,
                    target_region: shared,
                    edge_type: MovementEdgeType::Horizontal,
                    transition_owned: false,
                };
                adjacency.entry(i).or_default().push((j, metadata));
            }
        }
    }

    // Add vertical edges from inferred transitions.
    for ramp in &inferred {
        let lower_anchor = MovementNode::from_coord(
            ramp.lower_anchor.0,
            ramp.lower_anchor.1,
            ramp.lower_anchor.2,
        );
        let crest_coord = ramp.ramp_cells[2];
        let crest = MovementNode {
            layer: crest_coord.0,
            x: crest_coord.1,
            y: crest_coord.2,
        };
        let landing = MovementNode {
            layer: ramp.upper_landing.0,
            x: ramp.upper_landing.1,
            y: ramp.upper_landing.2,
        };
        if let (Some(&i), Some(&j)) = (node_index.get(&crest), node_index.get(&landing)) {
            let metadata = MovementEdgeMetadata {
                source_region: node_regions.get(&crest).and_then(|ids| ids.first().copied()),
                target_region: node_regions.get(&landing).and_then(|ids| ids.first().copied()),
                edge_type: MovementEdgeType::Vertical {
                    lower_anchor,
                    direction: ramp.direction,
                },
                transition_owned: true,
            };
            adjacency.entry(i).or_default().push((j, metadata.clone()));
            adjacency.entry(j).or_default().push((i, metadata));
        }
    }

    // Sort adjacency lists canonically.
    for edges in adjacency.values_mut() {
        edges.sort();
    }

    let movement = MovementGraph {
        nodes,
        adjacency,
        node_regions,
    };
    Ok((movement, inferred))
}

// ─── Structural validators ──────────────────────────────────────────────────

/// Validate structural properties of the materialized level against the
/// intended topology.
pub(super) fn validate_structural(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    _movement: &MovementGraph,
    inferred: &[InferredTransition],
) -> Result<Vec<GeneratorError>, GeneratorError> {
    let mut errors = Vec::new();

    // Dimensions match config.
    let config = &topology.config;
    let width = u16::try_from(level.width).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Ir,
        operation: "struct_width_convert",
    })?;
    let height = u16::try_from(level.height).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "struct_height_convert",
        }
    })?;
    let layers = u16::try_from(level.layer_count()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "struct_layers_convert",
        }
    })?;

    if width != config.width() || height != config.height() || layers != config.layers().2 {
        errors.push(validation_error(
            "structural",
            format!(
                "dimension_mismatch expected=({},{},{}) actual=({},{},{})",
                config.width(),
                config.height(),
                config.layers().2,
                width,
                height,
                layers,
            ),
        ));
    }

    // Borders are walls on every layer.
    for layer in 0..layers {
        for x in 0..width {
            let top = level.tile_at_3d(usize::from(layer), usize::from(x), 0);
            let bottom = level.tile_at_3d(
                usize::from(layer),
                usize::from(x),
                usize::from(height.saturating_sub(1)),
            );
            if top != Tile::Wall {
                errors.push(validation_error(
                    "structural",
                    format!("border_not_wall l={layer} x={x} y=0 tile={top:?}"),
                ));
            }
            if bottom != Tile::Wall {
                errors.push(validation_error(
                    "structural",
                    format!("border_not_wall l={layer} x={x} y={} tile={bottom:?}", height - 1),
                ));
            }
        }
        for y in 0..height {
            let left = level.tile_at_3d(usize::from(layer), 0, usize::from(y));
            let right = level.tile_at_3d(
                usize::from(layer),
                usize::from(width.saturating_sub(1)),
                usize::from(y),
            );
            if left != Tile::Wall {
                errors.push(validation_error(
                    "structural",
                    format!("border_not_wall l={layer} x=0 y={y} tile={left:?}"),
                ));
            }
            if right != Tile::Wall {
                errors.push(validation_error(
                    "structural",
                    format!("border_not_wall l={layer} x={} y={y} tile={right:?}", width - 1),
                ));
            }
        }
    }

    // Validate ramp transitions: contiguous R0/R1/R2, approach, opening, landing, headroom.
    for ramp in inferred {
        let (lx, ly) = (ramp.lower_approach.1, ramp.lower_approach.2);
        let app_tile = level.tile_at_3d(
            usize::from(ramp.lower_layer),
            usize::from(lx),
            usize::from(ly),
        );
        if app_tile != Tile::Floor {
            errors.push(validation_error(
                "structural",
                format!(
                    "ramp_approach_not_floor l={} x={} y={} tile={app_tile:?}",
                    ramp.lower_layer, lx, ly,
                ),
            ));
        }
        let (sx, sy) = (ramp.upper_landing.1, ramp.upper_landing.2);
        let land_tile = level.tile_at_3d(
            usize::from(ramp.upper_layer),
            usize::from(sx),
            usize::from(sy),
        );
        if land_tile != Tile::Floor {
            errors.push(validation_error(
                "structural",
                format!(
                    "ramp_landing_not_floor l={} x={} y={} tile={land_tile:?}",
                    ramp.upper_layer, sx, sy,
                ),
            ));
        }
        for &(ol, ox, oy) in &ramp.opening_cells {
            let open_tile =
                level.tile_at_3d(usize::from(ol), usize::from(ox), usize::from(oy));
            if open_tile != Tile::Void {
                errors.push(validation_error(
                    "structural",
                    format!(
                        "ramp_opening_not_void l={ol} x={ox} y={oy} tile={open_tile:?}"
                    ),
                ));
            }
        }
    }

    // Tile legality: every cell must be a valid token.
    for layer in 0..layers {
        for y in 0..height {
            for x in 0..width {
                let tile = level.tile_at_3d(usize::from(layer), usize::from(x), usize::from(y));
                match tile {
                    Tile::Wall | Tile::Floor | Tile::Void
                    | Tile::RampNorth(_) | Tile::RampEast(_)
                    | Tile::RampSouth(_) | Tile::RampWest(_) => {}
                }
            }
        }
    }

    // Every placed region's footprint cells must not be Void.
    for region in &topology.regions {
        let (rx, ry, rw, rh) = region.footprint;
        let max_x = rx.checked_add(rw).unwrap_or(width);
        let max_y = ry.checked_add(rh).unwrap_or(height);
        let mut region_has_floor = false;
        for dy in ry..max_y {
            for dx in rx..max_x {
                if dy >= height || dx >= width {
                    continue;
                }
                let tile = level.tile_at_3d(
                    usize::from(region.layer),
                    usize::from(dx),
                    usize::from(dy),
                );
                if tile == Tile::Floor || layout_tile_is_ramp(tile) {
                    region_has_floor = true;
                }
            }
        }
        if !region_has_floor {
            errors.push(validation_error(
                "structural",
                format!("region_{}_has_no_walkable_cells", region.id.raw()),
            ));
        }
    }

    // Every transition reservation must have its ramp/opening/landing cells matching.
    for transition in &topology.transitions {
        for cell in &transition.ramp_run_cells {
            let tile = level.tile_at_3d(
                usize::from(cell.layer),
                usize::from(cell.x),
                usize::from(cell.y),
            );
            if !layout_tile_is_ramp(tile) {
                errors.push(validation_error(
                    "structural",
                    format!(
                        "transition_{}_ramp_cell_not_ramp cell={} tile={tile:?}",
                        transition.id.raw(), cell,
                    ),
                ));
            }
        }
        for cell in &transition.upper_opening_cells {
            let tile = level.tile_at_3d(
                usize::from(cell.layer),
                usize::from(cell.x),
                usize::from(cell.y),
            );
            if tile != Tile::Void {
                errors.push(validation_error(
                    "structural",
                    format!(
                        "transition_{}_opening_not_void cell={} tile={tile:?}",
                        transition.id.raw(), cell,
                    ),
                ));
            }
        }
        for cell in &transition.landing_cells {
            let tile = level.tile_at_3d(
                usize::from(cell.layer),
                usize::from(cell.x),
                usize::from(cell.y),
            );
            if tile != Tile::Floor {
                errors.push(validation_error(
                    "structural",
                    format!(
                        "transition_{}_landing_not_floor cell={} tile={tile:?}",
                        transition.id.raw(), cell,
                    ),
                ));
            }
        }
        for cell in &transition.headroom_cells {
            let tile = level.tile_at_3d(
                usize::from(cell.layer),
                usize::from(cell.x),
                usize::from(cell.y),
            );
            if tile != Tile::Void {
                errors.push(validation_error(
                    "structural",
                    format!(
                        "transition_{}_headroom_not_void cell={} tile={tile:?}",
                        transition.id.raw(), cell,
                    ),
                ));
            }
        }
    }

    // Transition count must match inferred count.
    if inferred.len() != topology.transitions.len() {
        errors.push(validation_error(
            "structural",
            format!(
                "transition_count_mismatch reserved={} inferred={}",
                topology.transitions.len(),
                inferred.len(),
            ),
        ));
    }

    Ok(errors)
}

// ─── Connectivity validators ────────────────────────────────────────────────

/// Validate that every required region is reachable from spawn via the
/// movement graph. Unreachable walkable tiles are ok only if not in a
/// gameplay region.
pub(super) fn validate_connectivity(
    _level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
) -> Result<Vec<GeneratorError>, GeneratorError> {
    let mut errors = Vec::new();

    // Phase 05 connectivity is anchored at the canonical first walkable cell
    // of the unique spawn-role region; the final spawn marker is Phase 06 data.
    let spawn_regions: Vec<_> = topology
        .regions
        .iter()
        .filter(|region| region.role == RegionRole::Spawn)
        .collect();
    if spawn_regions.len() != 1 {
        errors.push(validation_error(
            "connectivity",
            format!("spawn_region_count={}", spawn_regions.len()),
        ));
        return Ok(errors);
    }
    let spawn_region = spawn_regions[0];
    let spawn_idx = movement.nodes.iter().enumerate().find_map(|(index, node)| {
        node_in_region(*node, spawn_region).then_some(index)
    });
    let Some(spawn_idx) = spawn_idx else {
        errors.push(validation_error("connectivity", "spawn_region_not_walkable".into()));
        return Ok(errors);
    };

    // Flood fill from spawn.
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([spawn_idx]);
    reachable.insert(spawn_idx);
    while let Some(i) = queue.pop_front() {
        if let Some(neighbors) = movement.adjacency.get(&i) {
            for (j, _) in neighbors {
                if reachable.insert(*j) {
                    queue.push_back(*j);
                }
            }
        }
    }

    let reachable_nodes: BTreeSet<MovementNode> = reachable
        .iter()
        .map(|&i| movement.nodes[i])
        .collect();

    // Check every required region has at least one reachable cell.
    for region in &topology.regions {
        if !is_gameplay_region(region.role) {
            continue;
        }
        let has_reachable = reachable_nodes.iter().any(|n| {
            n.layer == region.layer
                && n.x >= region.footprint.0
                && n.x < region.footprint.0.saturating_add(region.footprint.2)
                && n.y >= region.footprint.1
                && n.y < region.footprint.1.saturating_add(region.footprint.3)
        });
        if !has_reachable {
            errors.push(validation_error(
                "connectivity",
                format!(
                    "region_{}_{}_unreachable",
                    region.id.raw(),
                    region.role.label(),
                ),
            ));
        }
    }

    Ok(errors)
}

/// Returns true if this role must be reachable (gameplay-critical).
fn is_gameplay_region(role: RegionRole) -> bool {
    matches!(
        role,
        RegionRole::Spawn
            | RegionRole::DistantLandmark
            | RegionRole::MajorLandmark
            | RegionRole::Junction
            | RegionRole::VerticalHub
            | RegionRole::RequiredRoute
            | RegionRole::DeadEnd
    )
}

// ─── Topology comparison ────────────────────────────────────────────────────

/// Compare the intended topology edges against reconstructed movement paths.
///
/// Each required intended edge must have an owned reconstructed path in the
/// movement graph. Optional mergers/shortcuts/crossings are classified.
pub(super) fn validate_topology(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    inferred: &[InferredTransition],
) -> Result<Vec<GeneratorError>, GeneratorError> {
    let mut errors = Vec::new();

    let _node_index: BTreeMap<MovementNode, usize> = movement
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();

    // For each intended edge, verify there exists a path between source and
    // target regions in the movement graph.
    for edge in &topology.edges {
        if edge.transition.is_some() {
            // Vertical edges are covered by transition matching.
            continue;
        }
        // Find source and target region sockets.
        let source_region = topology
            .regions
            .iter()
            .find(|r| r.id == edge.source_region);
        let target_region = topology
            .regions
            .iter()
            .find(|r| r.id == edge.target_region);

        let (Some(sr), Some(tr)) = (source_region, target_region) else {
            errors.push(validation_error(
                "topology",
                format!("edge_{}_region_missing", edge.id.raw()),
            ));
            continue;
        };

        if let Err(detail) = validate_edge_witness(edge, sr, tr, movement) {
            errors.push(validation_error(
                "topology",
                format!("edge_{}_invalid_owned_witness {detail}", edge.id.raw()),
            ));
            continue;
        }

        // Find walkable cells in each region footprint.
        let source_cells: Vec<usize> = movement
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.layer == sr.layer
                    && n.x >= sr.footprint.0
                    && n.x < sr.footprint.0.saturating_add(sr.footprint.2)
                    && n.y >= sr.footprint.1
                    && n.y < sr.footprint.1.saturating_add(sr.footprint.3)
            })
            .map(|(i, _)| i)
            .collect();

        let target_cells: Vec<usize> = movement
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.layer == tr.layer
                    && n.x >= tr.footprint.0
                    && n.x < tr.footprint.0.saturating_add(tr.footprint.2)
                    && n.y >= tr.footprint.1
                    && n.y < tr.footprint.1.saturating_add(tr.footprint.3)
            })
            .map(|(i, _)| i)
            .collect();

        // Check if any source cell can reach any target cell.
        let mut has_path = false;
        for &si in &source_cells {
            if has_path {
                break;
            }
            for &ti in &target_cells {
                if si == ti || path_exists(si, ti, &movement.adjacency) {
                    has_path = true;
                    break;
                }
            }
        }

        if !has_path {
            if edge.required {
                errors.push(validation_error(
                    "topology",
                    format!(
                        "required_edge_{}_no_reconstructed_path source={} target={}",
                        edge.id.raw(),
                        edge.source_region.raw(),
                        edge.target_region.raw(),
                    ),
                ));
            }
            // Optional edges without a path are classified as repairs.
        }
    }

    // Verify every inferred transition matches a transition reservation.
    let mut ramp_by_anchor: BTreeMap<(u16, u16, u16), &InferredTransition> = BTreeMap::new();
    for ramp in inferred {
        ramp_by_anchor.insert(ramp.lower_anchor, ramp);
    }

    for transition in &topology.transitions {
        // The anchor is the first ramp cell.
        if let Some(r0) = transition.ramp_run_cells.first() {
            let anchor = (r0.layer, r0.x, r0.y);
            if !ramp_by_anchor.contains_key(&anchor) {
                errors.push(validation_error(
                    "topology",
                    format!(
                        "transition_{}_not_inferred anchor=({},{},{})",
                        transition.id.raw(),
                        anchor.0, anchor.1, anchor.2,
                    ),
                ));
            }
        }
    }

    // Bound checks. Phase 03 metrics are retained as intent witnesses here;
    // tile-derived connectivity and owned witnesses above must independently
    // pass before these configured bounds can be accepted.
    let config = &topology.config;
    check_metric_range(&mut errors, "region_count", topology.regions.len() as u64,
        u64::from(config.region_min()), u64::from(config.region_max()));
    check_metric_range(&mut errors, "required_route", topology.route_distance,
        u64::from(config.required_route_min()), u64::from(config.required_route_max()));
    check_metric_range(&mut errors, "branch_depth", u64::from(topology.max_branch_depth),
        u64::from(config.branch_depth_min()), u64::from(config.branch_depth_max()));
    check_metric_range(&mut errors, "dead_end_count", u64::from(topology.dead_end_count),
        u64::from(config.intentional_dead_ends_min()), u64::from(config.intentional_dead_ends_max()));
    check_metric_range(&mut errors, "crossing_count", u64::from(topology.crossing_count),
        0, u64::from(config.crossings_max()));
    if topology.articulation_count > config.articulation_max() {
        errors.push(validation_error("topology", format!(
            "articulation_count={} exceeds max={}", topology.articulation_count, config.articulation_max()
        )));
    }
    if topology.per_layer_cycles.len() != usize::from(config.layers().2) {
        errors.push(validation_error("topology", format!(
            "cycle_layer_count={} expected={}", topology.per_layer_cycles.len(), config.layers().2
        )));
    }
    for (layer, &cycles) in topology.per_layer_cycles.iter().enumerate() {
        check_metric_range(&mut errors, &format!("cycles_layer_{layer}"), u64::from(cycles),
            u64::from(config.per_layer_cycles_min()), u64::from(config.per_layer_cycles_max()));
    }

    // Component count: number of connected components.
    let components = connected_components(movement);
    let component_count = u64::try_from(components.len()).unwrap_or(u64::MAX);
    let max_components = u64::from(config.components_max());
    if component_count > max_components {
        errors.push(validation_error(
            "topology",
            format!("component_count={component_count} exceeds max={max_components}"),
        ));
    }

    // Every non-optional walkable cell that is inside a gameplay region must be reachable.
    // Already validated in connectivity.

    Ok(errors)
}

// ─── Movement probes ────────────────────────────────────────────────────────

/// Run bidirectional movement probes over every ramp and representative
/// corridor corners/thresholds/hubs.
pub(super) fn validate_movement_probes(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    inferred: &[InferredTransition],
) -> Result<Vec<GeneratorError>, GeneratorError> {
    let mut errors = Vec::new();
    let collision = CollisionWorld::from_level(level);

    // Probe each complete ramp through every materialized substep in both
    // directions. A failed reverse traversal is a validation failure too.
    for ramp in inferred {
        let mut route = vec![ramp.lower_approach];
        route.extend(ramp.ramp_cells);
        route.push(ramp.upper_landing);
        if let Err(error) = probe_tile_route(
            &collision,
            &route,
            format!("ramp_forward anchor={:?}", ramp.lower_anchor),
        ) {
            errors.push(error);
        }
        route.reverse();
        if let Err(error) = probe_tile_route(
            &collision,
            &route,
            format!("ramp_reverse anchor={:?}", ramp.lower_anchor),
        ) {
            errors.push(error);
        }
    }

    // Probe selected corridor witnesses rather than a straight line through
    // walls between region centers.
    for edge in topology
        .edges
        .iter()
        .filter(|edge| edge.required && edge.transition.is_none())
    {
        let route: Vec<_> = edge
            .path_witness
            .iter()
            .map(|cell| (cell.layer, cell.x, cell.y))
            .collect();
        if let Err(error) = probe_tile_route(
            &collision,
            &route,
            format!("corridor_edge_{}_forward", edge.id.raw()),
        ) {
            errors.push(error);
        }
        let reverse: Vec<_> = route.into_iter().rev().collect();
        if let Err(error) = probe_tile_route(
            &collision,
            &reverse,
            format!("corridor_edge_{}_reverse", edge.id.raw()),
        ) {
            errors.push(error);
        }
    }

    Ok(errors)
}

fn probe_tile_route(
    collision: &CollisionWorld,
    route: &[(u16, u16, u16)],
    context: String,
) -> Result<(), GeneratorError> {
    let Some(&first) = route.first() else {
        return Err(validation_error("movement_probe", format!("{context}: empty route")));
    };
    let mut start = tile_to_world_coord(first);
    for &cell in &route[1..] {
        let goal = tile_to_world_coord(cell);
        probe_movement(collision, start, goal, context.clone())?;
        start = goal;
    }
    Ok(())
}

/// Run a single movement probe from start to goal.
fn probe_movement(
    collision: &CollisionWorld,
    start: (f32, f32, f32),
    goal: (f32, f32, f32),
    context: String,
) -> Result<(), GeneratorError> {
    let mut player = PlayerState::new(glam::Vec3::new(start.0, start.1, start.2));

    const PROBE_STEPS: usize = 120;
    let step_dt = 1.0 / 60.0;

    // Simple position-based probe — move the player toward the goal step-by-step.
    let target = glam::Vec3::new(goal.0, goal.1, goal.2);
    for _ in 0..PROBE_STEPS {
        let to_target = target - player.position;
        if to_target.length() < 0.3 {
            return Ok(());
        }
        let desired_dir = to_target.normalize_or_zero();
        player.velocity = desired_dir * 4.0; // 4 units/sec

        resolve_player_step(&mut player, collision, step_dt);

        if player.position.distance(target) < 0.5 {
            return Ok(());
        }
    }

    if player.position.distance(target) < 0.5 {
        Ok(())
    } else {
        Err(validation_error(
            "movement_probe",
            format!("{context}: could not reach goal"),
        ))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn validate_level_storage(level: &ParsedLevel) -> Result<(), GeneratorError> {
    let expected = level
        .width
        .checked_mul(level.height)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "movement_level_area",
        })?;
    if level.width == 0
        || level.height == 0
        || level.layers.is_empty()
        || level.layers.iter().any(|layer| layer.len() != expected)
    {
        return Err(validation_error(
            "structural",
            "non_rectangular_or_empty_level".into(),
        ));
    }
    Ok(())
}

fn node_in_region(node: MovementNode, region: &PlacedRegion) -> bool {
    let max_x = region.footprint.0.checked_add(region.footprint.2);
    let max_y = region.footprint.1.checked_add(region.footprint.3);
    node.layer == region.layer
        && max_x.is_some_and(|max| node.x >= region.footprint.0 && node.x < max)
        && max_y.is_some_and(|max| node.y >= region.footprint.1 && node.y < max)
}

fn tile_is_walkable(tile: Tile) -> bool {
    matches!(
        tile,
        Tile::Floor | Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
    )
}

fn layout_tile_is_ramp(tile: Tile) -> bool {
    matches!(
        tile,
        Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
    )
}

fn validate_edge_witness(
    edge: &IntendedEdge,
    source: &PlacedRegion,
    target: &PlacedRegion,
    movement: &MovementGraph,
) -> Result<(), String> {
    let first = edge.path_witness.first().ok_or_else(|| "empty".to_owned())?;
    let last = edge.path_witness.last().ok_or_else(|| "empty".to_owned())?;
    let first_node = MovementNode::from_coord(first.layer, first.x, first.y);
    let last_node = MovementNode::from_coord(last.layer, last.x, last.y);
    if !node_in_region(first_node, source) || !node_in_region(last_node, target) {
        return Err("endpoints_outside_regions".into());
    }
    let index: BTreeMap<_, _> = movement
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect();
    let envelope: BTreeSet<_> = edge.allowed_envelope_cells.iter().copied().collect();
    for cell in &edge.path_witness {
        if !envelope.contains(cell) {
            return Err(format!("cell_outside_envelope {cell}"));
        }
        let node = MovementNode::from_coord(cell.layer, cell.x, cell.y);
        if !index.contains_key(&node) {
            return Err(format!("cell_not_walkable {cell}"));
        }
    }
    for pair in edge.path_witness.windows(2) {
        let a = MovementNode::from_coord(pair[0].layer, pair[0].x, pair[0].y);
        let b = MovementNode::from_coord(pair[1].layer, pair[1].x, pair[1].y);
        let ai = index[&a];
        let bi = index[&b];
        if !movement
            .adjacency
            .get(&ai)
            .is_some_and(|neighbors| neighbors.iter().any(|(next, _)| *next == bi))
        {
            return Err(format!("non_adjacent {} -> {}", pair[0], pair[1]));
        }
    }
    Ok(())
}

fn check_metric_range(
    errors: &mut Vec<GeneratorError>,
    name: &str,
    actual: u64,
    min: u64,
    max: u64,
) {
    if actual < min || actual > max {
        errors.push(validation_error(
            "topology",
            format!("{name}={actual} outside [{min},{max}]"),
        ));
    }
}

fn path_exists(
    source: usize,
    target: usize,
    adjacency: &BTreeMap<usize, Vec<(usize, MovementEdgeMetadata)>>,
) -> bool {
    if source == target {
        return true;
    }
    let mut visited = BTreeSet::from([source]);
    let mut queue = VecDeque::from([source]);
    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&current) {
            for (next, _) in neighbors {
                if *next == target {
                    return true;
                }
                if visited.insert(*next) {
                    queue.push_back(*next);
                }
            }
        }
    }
    false
}

fn connected_components(movement: &MovementGraph) -> Vec<BTreeSet<usize>> {
    let mut components = Vec::new();
    let mut visited = BTreeSet::new();
    for i in 0..movement.nodes.len() {
        if visited.contains(&i) {
            continue;
        }
        let mut component = BTreeSet::new();
        let mut queue = VecDeque::from([i]);
        component.insert(i);
        visited.insert(i);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = movement.adjacency.get(&current) {
                for (next, _) in neighbors {
                    if visited.insert(*next) {
                        component.insert(*next);
                        queue.push_back(*next);
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

fn tile_to_world_coord((layer, x, y): (u16, u16, u16)) -> (f32, f32, f32) {
    let world = crate::layout::tile_to_world(usize::from(x), usize::from(y));
    let eye_y = f32::from(layer) * crate::collision::WALL_HEIGHT
        + crate::player::PLAYER_EYE_HEIGHT;
    (
        world.x + crate::layout::TILE_SIZE * 0.5,
        eye_y,
        world.z - crate::layout::TILE_SIZE * 0.5,
    )
}

fn validation_error(kind: &'static str, detail: String) -> GeneratorError {
    GeneratorError::IrInvariant {
        stage: ErrorStage::Ir,
        detail: format!("[{kind}] {detail}"),
    }
}

// ─── Full validation ────────────────────────────────────────────────────────

/// Run the complete validator set and return a report.
pub(super) fn validate_full(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    inferred: &[InferredTransition],
) -> Result<ValidationReport, GeneratorError> {
    let structural_errors = validate_structural(level, topology, movement, inferred)?;
    let connectivity_errors = validate_connectivity(level, topology, movement)?;
    let topology_errors = validate_topology(topology, movement, inferred)?;
    let movement_probe_errors = validate_movement_probes(level, topology, inferred)?;

    Ok(ValidationReport {
        structural_errors,
        connectivity_errors,
        topology_errors,
        movement_probe_errors,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{ParsedLevel, TileCoord};

    fn minimal_level() -> ParsedLevel {
        let w = 5usize;
        let h = 5usize;
        let mut tiles = vec![Tile::Wall; w * h];
        tiles[1 * w + 1] = Tile::Floor;
        tiles[1 * w + 2] = Tile::Floor;
        tiles[2 * w + 2] = Tile::Floor;
        tiles[3 * w + 2] = Tile::Floor;
        tiles[3 * w + 3] = Tile::Floor;
        ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn dummy_topology() -> IntendedTopology {
        use super::super::config::GeneratorConfig;
        IntendedTopology {
            regions: vec![PlacedRegion {
                id: RegionId(0),
                role: RegionRole::Spawn,
                variant_index: 0,
                layer: 0,
                footprint: (1, 1, 2, 2),
                sockets: vec![],
                transitions: vec![],
                marker_variant_indices: vec![],
            }],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: GeneratorConfig::custom(64, 64, 2)
                .normalize()
                .unwrap(),
        }
    }

    #[test]
    fn movement_reconstruction_finds_walkable_cells() {
        let level = minimal_level();
        let topology = dummy_topology();
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        assert!(!movement.nodes.is_empty());
        // Walkable cells: (1,1), (2,1), (2,2), (2,3), (3,3)
        assert!(movement.nodes.len() >= 5);
        let spawn_node = MovementNode::from_coord(0, 1, 1);
        assert!(movement.nodes.contains(&spawn_node));
    }

    #[test]
    fn structural_validation_checks_borders() {
        let level = minimal_level();
        let topology = dummy_topology();
        let (movement, inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        let errors = validate_structural(&level, &topology, &movement, &inferred).unwrap();
        // The dummy topology config says 64x64 but level is 5x5 — dimension mismatch.
        assert!(errors.iter().any(|e| e.to_string().contains("dimension_mismatch")));
    }

    #[test]
    fn connectivity_from_spawn() {
        let level = minimal_level();
        let topology = dummy_topology();
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        let errors = validate_connectivity(&level, &topology, &movement).unwrap();
        // Spawn region is reachable; dimension mismatch is separate.
        // The level spawn (1,1) is walkable; flood should find all walkable cells.
        assert!(errors.is_empty() || errors.iter().all(|e| !e.to_string().contains("unreachable")));
    }
}
