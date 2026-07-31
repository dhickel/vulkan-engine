use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use super::config::NormalizedGeneratorConfig;
use super::context::AttemptContext;
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    Direction, EdgeId, GridCoord, IntendedEdge, IntendedTopology, PlacedRegion, PlacedSocket,
    RegionId, TransitionId,
};
use super::prefab::{PrefabCatalog, ReservationKind};
use super::ramps::{prefab_tile_to_layout, TileBufferWrite};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CellOwnership {
    regions: BTreeSet<RegionId>,
    corridors: BTreeSet<EdgeId>,
    transition: Option<TransitionId>,
}

/// Attempt-local, isolated grids. Each layer owns independent tile and
/// ownership vectors; all indexing is bounds checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TileBuffer {
    width: u16,
    height: u16,
    layers: u16,
    cells: Vec<Vec<crate::layout::Tile>>,
    ownership: Vec<Vec<CellOwnership>>,
}

impl TileBuffer {
    pub(super) fn new(width: u16, height: u16, layers: u16) -> Result<Self, GeneratorError> {
        if width == 0 || height == 0 || layers == 0 {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Materialization,
                detail: format!("tile_buffer_zero_dimensions w={width} h={height} l={layers}"),
            });
        }
        let layer_capacity = usize::from(width).checked_mul(usize::from(height)).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "tile_buffer_layer_capacity",
            },
        )?;
        let mut cells = Vec::with_capacity(usize::from(layers));
        let mut ownership = Vec::with_capacity(usize::from(layers));
        for _ in 0..layers {
            cells.push(vec![crate::layout::Tile::Void; layer_capacity]);
            ownership.push(vec![CellOwnership::default(); layer_capacity]);
        }
        Ok(Self {
            width,
            height,
            layers,
            cells,
            ownership,
        })
    }

    pub(super) fn into_parsed_level(self, spawn: (u16, u16)) -> crate::layout::ParsedLevel {
        crate::layout::ParsedLevel {
            width: usize::from(self.width),
            height: usize::from(self.height),
            layers: self.cells,
            spawn: crate::layout::TileCoord {
                layer: 0,
                x: usize::from(spawn.0),
                y: usize::from(spawn.1),
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn cell_index(&self, layer: u16, x: u16, y: u16) -> Result<usize, GeneratorError> {
        if layer >= self.layers || x >= self.width || y >= self.height {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Materialization,
                detail: format!(
                    "tile_buffer_oob l={layer} x={x} y={y} dimensions=({},{},{})",
                    self.layers, self.width, self.height
                ),
            });
        }
        usize::from(y)
            .checked_mul(usize::from(self.width))
            .and_then(|row| row.checked_add(usize::from(x)))
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "tile_buffer_index",
            })
    }

    fn ownership(&self, cell: GridCoord) -> Option<&CellOwnership> {
        let index = self.cell_index(cell.layer, cell.x, cell.y).ok()?;
        self.ownership.get(usize::from(cell.layer))?.get(index)
    }

    fn ownership_mut(&mut self, cell: GridCoord) -> Result<&mut CellOwnership, GeneratorError> {
        let index = self.cell_index(cell.layer, cell.x, cell.y)?;
        self.ownership
            .get_mut(usize::from(cell.layer))
            .and_then(|layer| layer.get_mut(index))
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Materialization,
                detail: format!("tile_buffer_ownership_missing {cell}"),
            })
    }

    fn clear_transition_cell(
        &mut self,
        cell: GridCoord,
        transition: TransitionId,
    ) -> Result<(), GeneratorError> {
        let index = self.cell_index(cell.layer, cell.x, cell.y)?;
        let slot = self
            .cells
            .get_mut(usize::from(cell.layer))
            .and_then(|cells| cells.get_mut(index))
            .ok_or_else(|| {
                materialization_error("transition_clearance_cell_missing", cell.to_string())
            })?;
        if !matches!(*slot, crate::layout::Tile::Void | crate::layout::Tile::Wall) {
            return Err(materialization_error(
                "transition_clearance_blocked",
                format!("transition={} cell={cell} tile={slot:?}", transition.raw()),
            ));
        }
        *slot = crate::layout::Tile::Void;
        self.mark_transition_cell(cell, transition)
    }

    fn mark_transition_cell(
        &mut self,
        cell: GridCoord,
        transition: TransitionId,
    ) -> Result<(), GeneratorError> {
        let owner = &mut self.ownership_mut(cell)?.transition;
        if owner.is_some_and(|existing| existing != transition) {
            return Err(materialization_error(
                "transition_ownership_conflict",
                format!(
                    "transition={} cell={cell} existing={:?}",
                    transition.raw(),
                    owner.map(TransitionId::raw)
                ),
            ));
        }
        *owner = Some(transition);
        Ok(())
    }

    fn seal_borders(&mut self) -> Result<(), GeneratorError> {
        for layer in 0..self.layers {
            for x in 0..self.width {
                self.seal_border_cell(layer, x, 0)?;
                self.seal_border_cell(layer, x, self.height - 1)?;
            }
            for y in 0..self.height {
                self.seal_border_cell(layer, 0, y)?;
                self.seal_border_cell(layer, self.width - 1, y)?;
            }
        }
        Ok(())
    }

    fn seal_border_cell(&mut self, layer: u16, x: u16, y: u16) -> Result<(), GeneratorError> {
        match self.get_tile(layer, x, y) {
            Some(crate::layout::Tile::Void | crate::layout::Tile::Wall) => {
                self.set_tile(layer, x, y, crate::layout::Tile::Wall)
            }
            Some(tile) => Err(materialization_error(
                "border_not_sealable",
                format!("l={layer} x={x} y={y} tile={tile:?}"),
            )),
            None => Err(materialization_error(
                "border_cell_missing",
                format!("l={layer} x={x} y={y}"),
            )),
        }
    }

    /// After all rooms, corridors, ramps, and borders are materialized, fill
    /// void cells orthogonal-adjacent to walkable tiles with Wall tiles so
    /// corridors and room exteriors produce wall geometry.
    pub(super) fn seal_corridor_walls(&mut self) -> Result<(), GeneratorError> {
        let mut walls = Vec::new();
        for layer in 0..self.layers {
            for y in 1..self.height.saturating_sub(1) {
                for x in 1..self.width.saturating_sub(1) {
                    if self.get_tile(layer, x, y) != Some(crate::layout::Tile::Void) {
                        continue;
                    }
                    let cell = GridCoord { layer, x, y };
                    if self
                        .ownership(cell)
                        .is_some_and(|ownership| ownership.transition.is_some())
                        || layer.checked_sub(1).is_some_and(|lower_layer| {
                            self.get_tile(lower_layer, x, y).is_some_and(|tile| {
                                tile == crate::layout::Tile::Void || layout_tile_is_ramp(tile)
                            })
                        })
                    {
                        continue;
                    }
                    let neighbors = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)];
                    if neighbors.into_iter().any(|(neighbor_x, neighbor_y)| {
                        self.get_tile(layer, neighbor_x, neighbor_y)
                            .is_some_and(layout_tile_is_walkable)
                    }) {
                        walls.push(cell);
                    }
                }
            }
        }
        for cell in walls {
            self.set_tile(cell.layer, cell.x, cell.y, crate::layout::Tile::Wall)?;
        }
        Ok(())
    }
}

impl TileBufferWrite for TileBuffer {
    fn set_tile(
        &mut self,
        layer: u16,
        x: u16,
        y: u16,
        tile: crate::layout::Tile,
    ) -> Result<(), GeneratorError> {
        let index = self.cell_index(layer, x, y)?;
        let slot = self
            .cells
            .get_mut(usize::from(layer))
            .and_then(|cells| cells.get_mut(index))
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Materialization,
                detail: format!("tile_buffer_cell_missing l={layer} x={x} y={y}"),
            })?;
        if *slot != crate::layout::Tile::Void && *slot != tile {
            return Err(GeneratorError::TileBufferConflict {
                stage: ErrorStage::Materialization,
                detail: format!(
                    "tile_buffer_conflict l={layer} x={x} y={y} existing={slot:?} wanted={tile:?}"
                ),
            });
        }
        *slot = tile;
        Ok(())
    }

    fn get_tile(&self, layer: u16, x: u16, y: u16) -> Option<crate::layout::Tile> {
        let index = self.cell_index(layer, x, y).ok()?;
        self.cells.get(usize::from(layer))?.get(index).copied()
    }

    fn dimensions(&self) -> (u16, u16, u16) {
        (self.width, self.height, self.layers)
    }
}

/// Stamp one fully transformed prefab variant atomically. Variant tiles and
/// reservation coordinates are already rotated by the catalog.
pub(super) fn stamp_prefab_region(
    region: &PlacedRegion,
    catalog: &PrefabCatalog,
    buffer: &mut TileBuffer,
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let variant = catalog
        .variants()
        .get(usize::from(region.variant_index))
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Materialization,
            detail: format!("stamp_variant_missing index={}", region.variant_index),
        })?;
    if region.footprint.2 != variant.width || region.footprint.3 != variant.height {
        return Err(materialization_error(
            "stamp_footprint_variant_mismatch",
            format!("region={}", region.id.raw()),
        ));
    }

    let mut staged = buffer.clone();
    for (layer_offset, layer_grid) in variant.layers.iter().enumerate() {
        let layer_offset =
            u16::try_from(layer_offset).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "stamp_layer_offset",
            })?;
        let global_layer =
            region
                .layer
                .checked_add(layer_offset)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "stamp_global_layer",
                })?;
        for (local_y, row) in layer_grid.iter().enumerate() {
            for (local_x, tile) in row.iter().enumerate() {
                let cell = translated_cell(
                    global_layer,
                    region.footprint.0,
                    region.footprint.1,
                    local_x,
                    local_y,
                    config,
                    "stamp_cell_oob",
                )?;
                let wanted = prefab_tile_to_layout(*tile, 0);
                let existing = staged
                    .get_tile(cell.layer, cell.x, cell.y)
                    .ok_or_else(|| materialization_error("stamp_cell_missing", cell.to_string()))?;
                if existing != crate::layout::Tile::Void && existing != wanted {
                    return Err(GeneratorError::TileBufferConflict {
                        stage: ErrorStage::Materialization,
                        detail: format!(
                            "stamp_conflict region={} cell={} existing={existing:?} wanted={wanted:?}",
                            region.id.raw(), cell
                        ),
                    });
                }
                staged.set_tile(cell.layer, cell.x, cell.y, wanted)?;
                staged.ownership_mut(cell)?.regions.insert(region.id);
            }
        }
    }

    // Recheck transformed reservation volumes against the stamped tile view.
    // Funnel and corridor-approach cells must remain clear Floor cells.
    for reservation in &variant.reservations {
        for local in &reservation.cells {
            let layer = region.layer.checked_add(local.layer).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "stamp_reservation_layer",
                },
            )?;
            let x = region.footprint.0.checked_add(local.x).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "stamp_reservation_x",
                },
            )?;
            let y = region.footprint.1.checked_add(local.y).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "stamp_reservation_y",
                },
            )?;
            let cell = GridCoord::new(
                layer,
                x,
                y,
                config.width(),
                config.height(),
                config.layers().2,
            )
            .map_err(|_| {
                materialization_error(
                    "stamp_reservation_oob",
                    format!("region={}", region.id.raw()),
                )
            })?;
            if matches!(
                reservation.kind,
                ReservationKind::SocketFunnel | ReservationKind::CorridorApproach
            ) && staged.get_tile(cell.layer, cell.x, cell.y) != Some(crate::layout::Tile::Floor)
            {
                return Err(materialization_error(
                    "socket_funnel_blocked",
                    format!("region={} cell={cell}", region.id.raw()),
                ));
            }
        }
    }

    *buffer = staged;
    Ok(())
}

pub(super) fn carve_corridors(
    topology: &IntendedTopology,
    buffer: &mut TileBuffer,
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let mut edges: Vec<_> = topology
        .edges
        .iter()
        .filter(|edge| edge.transition.is_none())
        .collect();
    edges.sort_by_key(|edge| edge.id);
    for edge in edges {
        let mut staged = buffer.clone();
        carve_single_edge(edge, topology, &mut staged, config)?;
        *buffer = staged;
    }
    Ok(())
}

fn carve_single_edge(
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &mut TileBuffer,
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let source = region_and_socket(topology, edge.source_region, edge.source_socket)?;
    let target = region_and_socket(topology, edge.target_region, edge.target_socket)?;
    let witness_start = edge.path_witness.first().copied().ok_or_else(|| {
        materialization_error("corridor_witness_empty", format!("edge={}", edge.id.raw()))
    })?;
    let witness_goal = edge.path_witness.last().copied().ok_or_else(|| {
        materialization_error("corridor_witness_empty", format!("edge={}", edge.id.raw()))
    })?;
    let route_start = edge
        .path_witness
        .iter()
        .copied()
        .find(|cell| {
            buffer.get_tile(cell.layer, cell.x, cell.y) == Some(crate::layout::Tile::Floor)
        })
        .ok_or_else(|| {
            materialization_error(
                "corridor_source_floor_connector_missing",
                format!("edge={}", edge.id.raw()),
            )
        })?;
    let route_goal = edge
        .path_witness
        .iter()
        .rev()
        .copied()
        .find(|cell| {
            buffer.get_tile(cell.layer, cell.x, cell.y) == Some(crate::layout::Tile::Floor)
        })
        .ok_or_else(|| {
            materialization_error(
                "corridor_target_floor_connector_missing",
                format!("edge={}", edge.id.raw()),
            )
        })?;
    if route_start.layer != route_goal.layer {
        return Err(materialization_error(
            "corridor_cross_layer_witness",
            format!("edge={}", edge.id.raw()),
        ));
    }
    validate_socket_terminal(witness_start, source.1, config, edge.id, "source")?;
    validate_socket_terminal(witness_goal, target.1, config, edge.id, "target")?;

    let envelope: BTreeSet<GridCoord> = edge.allowed_envelope_cells.iter().copied().collect();
    if envelope.len() != edge.allowed_envelope_cells.len() {
        return Err(materialization_error(
            "corridor_envelope_duplicate",
            format!("edge={}", edge.id.raw()),
        ));
    }
    for cell in &envelope {
        ensure_materialization_bounds(*cell, config, "corridor_envelope_oob")?;
    }

    let protected = protected_transition_cells(topology);
    let max_steps =
        edge.path_witness
            .len()
            .checked_sub(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "corridor_max_steps",
            })?;
    let path = target_seeking_path(
        route_start,
        route_goal,
        max_steps,
        edge,
        topology,
        buffer,
        config,
        &envelope,
        &protected,
    )
    .filter(|path| {
        expanded_path(path, edge.width, config)
            .is_ok_and(|cells| cells.iter().all(|cell| envelope.contains(cell)))
    })
    .or_else(|| {
        astar_path(
            route_start,
            route_goal,
            max_steps,
            edge,
            topology,
            buffer,
            config,
            &envelope,
            &protected,
        )
        .ok()
        .flatten()
    })
    .ok_or_else(|| {
        let trimmed_witness: Vec<_> = edge
            .path_witness
            .iter()
            .copied()
            .skip_while(|cell| *cell != route_start)
            .take_while(|cell| *cell != route_goal)
            .chain(std::iter::once(route_goal))
            .collect();
        let witness_detail = expanded_path(&trimmed_witness, edge.width, config)
            .ok()
            .and_then(|cells| {
                cells.into_iter().find_map(|cell| {
                    validate_corridor_cell(cell, edge, topology, buffer, config, &protected)
                        .err()
                        .map(|error| error.to_string())
                })
            })
            .unwrap_or_else(|| "witness_width_or_search_bound_failure".into());
        GeneratorError::CorridorInvariant {
            stage: ErrorStage::Materialization,
            edge: edge.id.raw(),
            detail: format!("phase03_realizable_envelope_has_no_legal_connector {witness_detail}"),
        }
    })?;

    let cells = expanded_path(&path, edge.width, config)?;
    if cells.iter().any(|cell| !envelope.contains(cell)) {
        return Err(GeneratorError::CorridorInvariant {
            stage: ErrorStage::Materialization,
            edge: edge.id.raw(),
            detail: "width_expansion_escaped_envelope".into(),
        });
    }
    for cell in &cells {
        validate_corridor_cell(*cell, edge, topology, buffer, config, &protected)?;
    }
    for cell in cells {
        if buffer.get_tile(cell.layer, cell.x, cell.y) == Some(crate::layout::Tile::Void) {
            buffer.set_tile(cell.layer, cell.x, cell.y, crate::layout::Tile::Floor)?;
        }
        buffer.ownership_mut(cell)?.corridors.insert(edge.id);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn target_seeking_path(
    start: GridCoord,
    goal: GridCoord,
    max_steps: usize,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &TileBuffer,
    config: &NormalizedGeneratorConfig,
    envelope: &BTreeSet<GridCoord>,
    protected: &BTreeSet<GridCoord>,
) -> Option<Vec<GridCoord>> {
    let mut path = vec![start];
    let mut visited = BTreeSet::from([start]);
    let mut previous_direction = None;
    while path.len().checked_sub(1)? < max_steps {
        let current = *path.last()?;
        if current == goal {
            return Some(path);
        }
        let mut choices = Vec::new();
        for direction in canonical_directions() {
            let next = offset(current, direction, config)?;
            if visited.contains(&next)
                || !center_cell_legal(next, edge, topology, buffer, config, envelope, protected)
                || !segment_width_legal(
                    current, next, edge, topology, buffer, config, envelope, protected,
                )
            {
                continue;
            }
            choices.push((
                next.x.abs_diff(goal.x) + next.y.abs_diff(goal.y),
                u8::from(previous_direction != Some(direction)),
                direction_rank(direction),
                next.y,
                next.x,
                direction,
                next,
            ));
        }
        choices.sort();
        let (_, _, _, _, _, direction, next) = choices.into_iter().next()?;
        path.push(next);
        visited.insert(next);
        previous_direction = Some(direction);
    }
    (path.last() == Some(&goal)).then_some(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SearchState {
    coord: GridCoord,
    direction: Option<Direction>,
    run_length: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Frontier {
    state: SearchState,
    f: u64,
    g: u64,
    turns: u64,
    sequence: u64,
}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .cmp(&self.f)
            .then_with(|| other.g.cmp(&self.g))
            .then_with(|| other.turns.cmp(&self.turns))
            .then_with(|| other.state.coord.cmp(&self.state.coord))
            .then_with(|| other.state.direction.cmp(&self.state.direction))
            .then_with(|| other.state.run_length.cmp(&self.state.run_length))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(clippy::too_many_arguments)]
fn astar_path(
    start: GridCoord,
    goal: GridCoord,
    max_steps: usize,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &TileBuffer,
    config: &NormalizedGeneratorConfig,
    envelope: &BTreeSet<GridCoord>,
    protected: &BTreeSet<GridCoord>,
) -> Result<Option<Vec<GridCoord>>, GeneratorError> {
    let start_state = SearchState {
        coord: start,
        direction: None,
        run_length: 0,
    };
    let mut frontier = BinaryHeap::from([Frontier {
        state: start_state,
        f: manhattan(start, goal),
        g: 0,
        turns: 0,
        sequence: 0,
    }]);
    let mut best = BTreeMap::from([(start_state, (0u64, 0u64))]);
    let mut parent = BTreeMap::new();
    let mut sequence = 0u64;
    while let Some(current) = frontier.pop() {
        if best.get(&current.state) != Some(&(current.g, current.turns)) {
            continue;
        }
        if current.state.coord == goal {
            let mut states = vec![current.state];
            let mut cursor = current.state;
            while cursor != start_state {
                cursor = *parent
                    .get(&cursor)
                    .ok_or(GeneratorError::CorridorInvariant {
                        stage: ErrorStage::Materialization,
                        edge: edge.id.raw(),
                        detail: "astar_parent_missing".into(),
                    })?;
                states.push(cursor);
            }
            states.reverse();
            return Ok(Some(states.into_iter().map(|state| state.coord).collect()));
        }
        if usize::try_from(current.g)
            .ok()
            .is_none_or(|g| g >= max_steps)
        {
            continue;
        }
        for direction in canonical_directions() {
            let Some(next_coord) = offset(current.state.coord, direction, config) else {
                continue;
            };
            if !center_cell_legal(
                next_coord, edge, topology, buffer, config, envelope, protected,
            ) || !segment_width_legal(
                current.state.coord,
                next_coord,
                edge,
                topology,
                buffer,
                config,
                envelope,
                protected,
            ) {
                continue;
            }
            let run_length = if current.state.direction == Some(direction) {
                current.state.run_length.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Materialization,
                        operation: "corridor_run_length",
                    },
                )?
            } else {
                1
            };
            let next = SearchState {
                coord: next_coord,
                direction: Some(direction),
                run_length,
            };
            let g = current
                .g
                .checked_add(1)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "corridor_astar_g",
                })?;
            let turns = current
                .turns
                .checked_add(u64::from(
                    current
                        .state
                        .direction
                        .is_some_and(|previous| previous != direction),
                ))
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "corridor_astar_turns",
                })?;
            let score = (g, turns);
            if best.get(&next).is_some_and(|existing| *existing <= score) {
                continue;
            }
            best.insert(next, score);
            parent.insert(next, current.state);
            sequence = sequence
                .checked_add(1)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "corridor_astar_sequence",
                })?;
            frontier.push(Frontier {
                state: next,
                f: g.checked_add(manhattan(next_coord, goal)).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Materialization,
                        operation: "corridor_astar_f",
                    },
                )?,
                g,
                turns,
                sequence,
            });
        }
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn segment_width_legal(
    from: GridCoord,
    to: GridCoord,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &TileBuffer,
    config: &NormalizedGeneratorConfig,
    envelope: &BTreeSet<GridCoord>,
    protected: &BTreeSet<GridCoord>,
) -> bool {
    expanded_path(&[from, to], edge.width, config).is_ok_and(|cells| {
        cells.iter().all(|cell| {
            envelope.contains(cell)
                && validate_corridor_cell(*cell, edge, topology, buffer, config, protected).is_ok()
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn center_cell_legal(
    cell: GridCoord,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &TileBuffer,
    config: &NormalizedGeneratorConfig,
    envelope: &BTreeSet<GridCoord>,
    protected: &BTreeSet<GridCoord>,
) -> bool {
    envelope.contains(&cell)
        && validate_corridor_cell(cell, edge, topology, buffer, config, protected).is_ok()
}

fn validate_corridor_cell(
    cell: GridCoord,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
    buffer: &TileBuffer,
    config: &NormalizedGeneratorConfig,
    protected: &BTreeSet<GridCoord>,
) -> Result<(), GeneratorError> {
    ensure_materialization_bounds(cell, config, "corridor_cell_oob")?;
    if cell.x == 0
        || cell.y == 0
        || cell.x.checked_add(1) == Some(config.width())
        || cell.y.checked_add(1) == Some(config.height())
    {
        return Err(materialization_error(
            "corridor_border_violation",
            cell.to_string(),
        ));
    }
    if protected.contains(&cell) && !transition_connector_cell(cell, edge, topology) {
        return Err(materialization_error(
            "corridor_transition_reservation_conflict",
            format!("edge={} cell={cell}", edge.id.raw()),
        ));
    }
    let tile = buffer
        .get_tile(cell.layer, cell.x, cell.y)
        .ok_or_else(|| materialization_error("corridor_cell_missing", cell.to_string()))?;
    let transition_connector = transition_connector_cell(cell, edge, topology);
    if !matches!(tile, crate::layout::Tile::Void | crate::layout::Tile::Floor)
        && !(transition_connector && layout_tile_is_ramp(tile))
    {
        return Err(GeneratorError::TileBufferConflict {
            stage: ErrorStage::Materialization,
            detail: format!(
                "corridor_blocked edge={} cell={cell} tile={tile:?}",
                edge.id.raw()
            ),
        });
    }

    let in_source = region_by_id(topology, edge.source_region)
        .is_some_and(|region| cell_in_region(cell, region));
    let in_target = region_by_id(topology, edge.target_region)
        .is_some_and(|region| cell_in_region(cell, region));
    if tile == crate::layout::Tile::Floor && !in_source && !in_target {
        let owners = buffer
            .ownership(cell)
            .ok_or_else(|| materialization_error("corridor_ownership_missing", cell.to_string()))?;
        if owners.corridors.is_empty()
            || !owners
                .corridors
                .iter()
                .all(|other| authorized_merger(cell, edge, *other, topology))
        {
            return Err(materialization_error(
                "corridor_unplanned_crossing",
                format!("edge={} cell={cell}", edge.id.raw()),
            ));
        }
    }
    Ok(())
}

fn transition_connector_cell(
    cell: GridCoord,
    edge: &IntendedEdge,
    topology: &IntendedTopology,
) -> bool {
    topology.transitions.iter().any(|transition| {
        [edge.source_region, edge.target_region].contains(&transition.upper_region)
            && transition.landing_cells.contains(&cell)
    })
}

fn authorized_merger(
    cell: GridCoord,
    edge: &IntendedEdge,
    other_id: EdgeId,
    topology: &IntendedTopology,
) -> bool {
    let Some(other) = topology
        .edges
        .iter()
        .find(|candidate| candidate.id == other_id)
    else {
        return false;
    };
    if shared_socket_funnel_cell(cell, edge, other) {
        return true;
    }
    [edge.source_region, edge.target_region]
        .into_iter()
        .filter(|region| [other.source_region, other.target_region].contains(region))
        .any(|shared| {
            region_by_id(topology, shared).is_some_and(|region| cell_in_region(cell, region))
        })
        || (edge.allowed_envelope_cells.contains(&cell)
            && other.allowed_envelope_cells.contains(&cell))
}

fn shared_socket_funnel_cell(cell: GridCoord, left: &IntendedEdge, right: &IntendedEdge) -> bool {
    let common_prefix_contains = |a: &[GridCoord], b: &[GridCoord]| {
        a.iter()
            .zip(b)
            .take_while(|(left, right)| left == right)
            .any(|(candidate, _)| *candidate == cell)
    };
    let left_reverse: Vec<_> = left.path_witness.iter().rev().copied().collect();
    let right_reverse: Vec<_> = right.path_witness.iter().rev().copied().collect();
    (left.source_socket == right.source_socket
        && common_prefix_contains(&left.path_witness, &right.path_witness))
        || (left.target_socket == right.target_socket
            && common_prefix_contains(&left_reverse, &right_reverse))
        || (left.source_socket == right.target_socket
            && common_prefix_contains(&left.path_witness, &right_reverse))
        || (left.target_socket == right.source_socket
            && common_prefix_contains(&left_reverse, &right.path_witness))
}

fn layout_tile_is_ramp(tile: crate::layout::Tile) -> bool {
    matches!(
        tile,
        crate::layout::Tile::RampNorth(_)
            | crate::layout::Tile::RampEast(_)
            | crate::layout::Tile::RampSouth(_)
            | crate::layout::Tile::RampWest(_)
    )
}

fn layout_tile_is_walkable(tile: crate::layout::Tile) -> bool {
    tile == crate::layout::Tile::Floor || layout_tile_is_ramp(tile)
}

fn expanded_path(
    path: &[GridCoord],
    width: u16,
    config: &NormalizedGeneratorConfig,
) -> Result<BTreeSet<GridCoord>, GeneratorError> {
    if width == 0 || path.is_empty() {
        return Err(materialization_error(
            "corridor_width_or_path_empty",
            "".into(),
        ));
    }
    let mut cells = BTreeSet::new();
    if path.len() == 1 {
        cells.insert(path[0]);
        return Ok(cells);
    }
    let left = -i32::from(width / 2);
    let right = i32::from(
        width
            .checked_sub(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "corridor_width_sub",
            })?
            / 2,
    );
    for pair in path.windows(2) {
        let from = pair[0];
        let to = pair[1];
        if from.layer != to.layer || from.x.abs_diff(to.x) + from.y.abs_diff(to.y) != 1 {
            return Err(materialization_error(
                "corridor_centerline_noncardinal",
                format!("from={from} to={to}"),
            ));
        }
        let perpendicular = if from.x == to.x { (1, 0) } else { (0, 1) };
        for center in [from, to] {
            for offset_value in left..=right {
                let x = checked_component_offset(center.x, perpendicular.0, offset_value)?;
                let y = checked_component_offset(center.y, perpendicular.1, offset_value)?;
                cells.insert(GridCoord::new(
                    center.layer,
                    x,
                    y,
                    config.width(),
                    config.height(),
                    config.layers().2,
                )?);
            }
        }
    }
    Ok(cells)
}

fn checked_component_offset(
    value: u16,
    perpendicular: i32,
    offset_value: i32,
) -> Result<u16, GeneratorError> {
    let delta =
        perpendicular
            .checked_mul(offset_value)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "corridor_width_offset_mul",
            })?;
    let value = i32::from(value)
        .checked_add(delta)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "corridor_width_offset_add",
        })?;
    u16::try_from(value).map_err(|_| materialization_error("corridor_width_oob", value.to_string()))
}

fn validate_socket_terminal(
    terminal: GridCoord,
    socket: &PlacedSocket,
    config: &NormalizedGeneratorConfig,
    edge: EdgeId,
    side: &'static str,
) -> Result<(), GeneratorError> {
    let (dx, dy) = socket.direction.delta();
    let x = u16::try_from(i32::from(socket.global_anchor.x).checked_sub(dx).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "socket_terminal_x",
        },
    )?)
    .map_err(|_| {
        materialization_error(
            "socket_terminal_oob",
            format!("edge={} side={side}", edge.raw()),
        )
    })?;
    let y = u16::try_from(i32::from(socket.global_anchor.y).checked_sub(dy).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "socket_terminal_y",
        },
    )?)
    .map_err(|_| {
        materialization_error(
            "socket_terminal_oob",
            format!("edge={} side={side}", edge.raw()),
        )
    })?;
    let expected = GridCoord::new(
        socket.global_anchor.layer,
        x,
        y,
        config.width(),
        config.height(),
        config.layers().2,
    )?;
    if terminal != expected {
        return Err(GeneratorError::CorridorInvariant {
            stage: ErrorStage::Materialization,
            edge: edge.raw(),
            detail: format!("{side}_terminal_mismatch expected={expected} actual={terminal}"),
        });
    }
    Ok(())
}

fn protected_transition_cells(topology: &IntendedTopology) -> BTreeSet<GridCoord> {
    topology
        .transitions
        .iter()
        .flat_map(|transition| {
            transition
                .ramp_run_cells
                .iter()
                .chain(&transition.upper_opening_cells)
                .chain(&transition.landing_cells)
                .chain(&transition.headroom_cells)
                .copied()
        })
        .collect()
}

fn region_and_socket(
    topology: &IntendedTopology,
    region_id: RegionId,
    socket_id: super::ir::SocketId,
) -> Result<(&PlacedRegion, &PlacedSocket), GeneratorError> {
    let region = region_by_id(topology, region_id).ok_or(GeneratorError::IrInvariant {
        stage: ErrorStage::Materialization,
        detail: format!("corridor_region_missing {}", region_id.raw()),
    })?;
    let socket = region
        .sockets
        .iter()
        .find(|socket| socket.id == socket_id)
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Materialization,
            detail: format!("corridor_socket_missing {}", socket_id.raw()),
        })?;
    Ok((region, socket))
}

fn region_by_id(topology: &IntendedTopology, id: RegionId) -> Option<&PlacedRegion> {
    topology.regions.iter().find(|region| region.id == id)
}

fn cell_in_region(cell: GridCoord, region: &PlacedRegion) -> bool {
    let Some(max_x) = region.footprint.0.checked_add(region.footprint.2) else {
        return false;
    };
    let Some(max_y) = region.footprint.1.checked_add(region.footprint.3) else {
        return false;
    };
    cell.layer == region.layer
        && cell.x >= region.footprint.0
        && cell.x < max_x
        && cell.y >= region.footprint.1
        && cell.y < max_y
}

fn offset(
    cell: GridCoord,
    direction: Direction,
    config: &NormalizedGeneratorConfig,
) -> Option<GridCoord> {
    let (dx, dy) = direction.delta();
    let x = u16::try_from(i32::from(cell.x).checked_add(dx)?).ok()?;
    let y = u16::try_from(i32::from(cell.y).checked_add(dy)?).ok()?;
    GridCoord::new(
        cell.layer,
        x,
        y,
        config.width(),
        config.height(),
        config.layers().2,
    )
    .ok()
}

fn canonical_directions() -> [Direction; 4] {
    [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ]
}

fn direction_rank(direction: Direction) -> u8 {
    match direction {
        Direction::North => 0,
        Direction::East => 1,
        Direction::South => 2,
        Direction::West => 3,
    }
}

fn manhattan(left: GridCoord, right: GridCoord) -> u64 {
    u64::from(left.x.abs_diff(right.x)) + u64::from(left.y.abs_diff(right.y))
}

fn translated_cell(
    layer: u16,
    origin_x: u16,
    origin_y: u16,
    local_x: usize,
    local_y: usize,
    config: &NormalizedGeneratorConfig,
    constraint: &'static str,
) -> Result<GridCoord, GeneratorError> {
    let local_x = u16::try_from(local_x).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Materialization,
        operation: "stamp_local_x_convert",
    })?;
    let local_y = u16::try_from(local_y).map_err(|_| GeneratorError::ArithmeticOverflow {
        stage: ErrorStage::Materialization,
        operation: "stamp_local_y_convert",
    })?;
    let x = origin_x
        .checked_add(local_x)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "stamp_x_add",
        })?;
    let y = origin_y
        .checked_add(local_y)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "stamp_y_add",
        })?;
    GridCoord::new(
        layer,
        x,
        y,
        config.width(),
        config.height(),
        config.layers().2,
    )
    .map_err(|_| materialization_error(constraint, format!("l={layer} x={x} y={y}")))
}

fn ensure_materialization_bounds(
    cell: GridCoord,
    config: &NormalizedGeneratorConfig,
    constraint: &'static str,
) -> Result<(), GeneratorError> {
    if cell.layer >= config.layers().2 || cell.x >= config.width() || cell.y >= config.height() {
        return Err(materialization_error(constraint, cell.to_string()));
    }
    Ok(())
}

fn materialization_error(constraint: &'static str, detail: String) -> GeneratorError {
    GeneratorError::MaterializationInfeasible {
        stage: ErrorStage::Materialization,
        constraint,
        detail,
    }
}

/// Tag every reserved transition cell plus inferred landing headroom so final
/// wall sealing cannot close a vertical opening.
fn mark_transition_ownership(
    transitions: &[super::ir::TransitionReservation],
    buffer: &mut TileBuffer,
) -> Result<(), GeneratorError> {
    for transition in transitions {
        for cell in transition
            .ramp_run_cells
            .iter()
            .chain(&transition.upper_opening_cells)
            .chain(&transition.landing_cells)
            .chain(&transition.headroom_cells)
        {
            buffer.mark_transition_cell(*cell, transition.id)?;
        }
        for landing in &transition.landing_cells {
            if let Some(headroom_layer) = landing
                .layer
                .checked_add(1)
                .filter(|layer| *layer < buffer.layers)
            {
                buffer.mark_transition_cell(
                    GridCoord {
                        layer: headroom_layer,
                        x: landing.x,
                        y: landing.y,
                    },
                    transition.id,
                )?;
            }
        }
    }
    Ok(())
}

/// Open the lower-layer cell beneath each upper landing. Leaving the prefab's
/// wall support there blocks the runtime capsule before it can finish climbing
/// R2 and step onto the upper floor.
fn clear_transition_crest_exits(
    transitions: &[super::ir::TransitionReservation],
    buffer: &mut TileBuffer,
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    for transition in transitions {
        let mut crest = None;
        for cell in &transition.ramp_run_cells {
            let direction = match buffer.get_tile(cell.layer, cell.x, cell.y) {
                Some(crate::layout::Tile::RampNorth(2)) => Some(Direction::North),
                Some(crate::layout::Tile::RampEast(2)) => Some(Direction::East),
                Some(crate::layout::Tile::RampSouth(2)) => Some(Direction::South),
                Some(crate::layout::Tile::RampWest(2)) => Some(Direction::West),
                _ => None,
            };
            if let Some(direction) = direction {
                if crest.replace((*cell, direction)).is_some() {
                    return Err(materialization_error(
                        "transition_crest_duplicate",
                        format!("transition={}", transition.id.raw()),
                    ));
                }
            }
        }
        let (crest, direction) = crest.ok_or_else(|| {
            materialization_error(
                "transition_crest_missing",
                format!("transition={}", transition.id.raw()),
            )
        })?;
        let exit = offset(crest, direction, config).ok_or_else(|| {
            materialization_error(
                "transition_crest_exit_oob",
                format!("transition={} crest={crest}", transition.id.raw()),
            )
        })?;
        let upper_layer =
            transition
                .lower_layer
                .checked_add(1)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Materialization,
                    operation: "transition_crest_exit_upper_layer",
                })?;
        if !transition.landing_cells.iter().any(|landing| {
            landing.layer == upper_layer && landing.x == exit.x && landing.y == exit.y
        }) {
            return Err(materialization_error(
                "transition_crest_exit_landing_mismatch",
                format!("transition={} exit={exit}", transition.id.raw()),
            ));
        }
        buffer.clear_transition_cell(exit, transition.id)?;
    }
    Ok(())
}

/// Full attempt transaction. A failure drops the local buffer; no caller state
/// is mutated.
pub(super) fn materialize_topology(
    topology: &IntendedTopology,
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
    ctx: &mut AttemptContext,
) -> Result<TileBuffer, GeneratorError> {
    let mut buffer = TileBuffer::new(config.width(), config.height(), config.layers().2)?;
    let upper_endpoint_ids: BTreeSet<RegionId> = topology
        .transitions
        .iter()
        .map(|transition| transition.upper_region)
        .collect();
    let mut regions: Vec<_> = topology
        .regions
        .iter()
        .filter(|region| !upper_endpoint_ids.contains(&region.id))
        .collect();
    regions.sort_by_key(|region| region.id);
    for region in regions {
        stamp_prefab_region(region, catalog, &mut buffer, config)?;
        ctx.region_stamped();
    }
    carve_corridors(topology, &mut buffer, config)?;
    ctx.corridor_carved();
    super::ramps::materialize_all_transitions(&topology.transitions, config, &mut buffer)?;
    mark_transition_ownership(&topology.transitions, &mut buffer)?;
    clear_transition_crest_exits(&topology.transitions, &mut buffer, config)?;
    buffer.seal_borders()?;
    buffer.seal_corridor_walls()?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::super::config::GeneratorConfig;
    use super::super::context::{AttemptContext, TelemetryMode};
    use super::super::determinism::{
        AttemptIdentity, GeneratorIdentity, SemanticStage, SemanticStreamFactory,
    };
    use super::super::placement::place_regions;
    use super::super::topology::{build_candidate_graph, select_topology};
    use super::*;

    fn catalog() -> PrefabCatalog {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        PrefabCatalog::load(&root).expect("bundled prefab catalog")
    }

    fn end_to_end_config() -> NormalizedGeneratorConfig {
        let mut raw = GeneratorConfig::custom(64, 64, 2);
        raw.single_bottleneck = true;
        raw.relax_route_redundancy = true;
        raw.relax_transition_redundancy = true;
        raw.region_min = Some(12);
        raw.region_max = Some(12);
        raw.required_route_min = Some(50);
        raw.required_route_max = Some(180);
        raw.branch_depth_min = Some(2);
        raw.branch_depth_max = Some(12);
        raw.articulation_max = Some(12);
        raw.crossings_max = Some(16);
        raw.intentional_dead_ends_min = Some(1);
        raw.intentional_dead_ends_max = Some(4);
        raw.normalize().expect("config")
    }

    fn selected(seed: u64) -> (NormalizedGeneratorConfig, PrefabCatalog, IntendedTopology) {
        let config = end_to_end_config();
        let catalog = catalog();
        let identity = GeneratorIdentity::new(&config, catalog.identity_bytes(), seed);
        let factory = SemanticStreamFactory::new(AttemptIdentity::new(identity, 0));
        let mut roles = factory.stream(SemanticStage::Roles, &[]);
        let (placed, grid) = place_regions(
            &config,
            &catalog,
            &mut roles,
            factory,
            &mut AttemptContext::new(TelemetryMode::Off),
        )
        .expect("placement");
        let graph =
            build_candidate_graph(&placed, &grid, &mut AttemptContext::new(TelemetryMode::Off))
                .expect("graph");
        let mut topology_rng = factory.stream(SemanticStage::Topology, &[]);
        let topology = select_topology(
            placed,
            &config,
            &graph,
            &mut topology_rng,
            &mut AttemptContext::new(TelemetryMode::Off),
        )
        .expect("topology");
        (config, catalog, topology)
    }

    #[test]
    fn tile_buffer_layers_bounds_and_conflicts_are_isolated() {
        let mut buffer = TileBuffer::new(4, 4, 2).expect("buffer");
        buffer
            .set_tile(0, 1, 1, crate::layout::Tile::Floor)
            .expect("first write");
        assert_eq!(buffer.get_tile(0, 1, 1), Some(crate::layout::Tile::Floor));
        assert_eq!(buffer.get_tile(1, 1, 1), Some(crate::layout::Tile::Void));
        assert!(buffer.set_tile(0, 1, 1, crate::layout::Tile::Wall).is_err());
        assert!(buffer
            .set_tile(2, 0, 0, crate::layout::Tile::Floor)
            .is_err());
        assert_eq!(buffer.get_tile(2, 0, 0), None);
    }

    #[test]
    fn corridor_wall_sealing_is_orthogonal_walkable_and_border_scoped() {
        let mut buffer = TileBuffer::new(7, 7, 1).expect("buffer");
        buffer
            .set_tile(0, 3, 3, crate::layout::Tile::Floor)
            .expect("floor");
        buffer
            .set_tile(0, 5, 5, crate::layout::Tile::RampNorth(0))
            .expect("ramp");

        buffer.seal_corridor_walls().expect("seal corridor walls");

        for (x, y) in [(3, 2), (4, 3), (3, 4), (2, 3), (5, 4), (4, 5)] {
            assert_eq!(buffer.get_tile(0, x, y), Some(crate::layout::Tile::Wall));
        }
        assert_eq!(buffer.get_tile(0, 2, 2), Some(crate::layout::Tile::Void));
        assert_eq!(buffer.get_tile(0, 6, 5), Some(crate::layout::Tile::Void));
    }

    #[test]
    fn corridor_wall_sealing_preserves_vertical_openings() {
        let mut buffer = TileBuffer::new(7, 7, 3).expect("buffer");
        buffer
            .set_tile(0, 3, 3, crate::layout::Tile::RampEast(1))
            .expect("lower ramp");
        buffer
            .set_tile(1, 3, 2, crate::layout::Tile::Floor)
            .expect("upper floor beside ramp opening");
        buffer
            .set_tile(2, 4, 3, crate::layout::Tile::Floor)
            .expect("floor beside stacked void opening");
        let headroom = GridCoord {
            layer: 2,
            x: 5,
            y: 5,
        };
        buffer
            .set_tile(2, 5, 4, crate::layout::Tile::Floor)
            .expect("floor beside transition headroom");
        buffer
            .mark_transition_cell(headroom, TransitionId(0))
            .expect("headroom ownership");

        let transition_owned = GridCoord {
            layer: 0,
            x: 2,
            y: 3,
        };
        buffer
            .mark_transition_cell(transition_owned, TransitionId(0))
            .expect("transition ownership");
        buffer
            .set_tile(0, 1, 3, crate::layout::Tile::Floor)
            .expect("floor beside transition-owned opening");

        buffer.seal_corridor_walls().expect("seal corridor walls");

        assert_eq!(buffer.get_tile(1, 3, 3), Some(crate::layout::Tile::Void));
        assert_eq!(buffer.get_tile(2, 3, 3), Some(crate::layout::Tile::Void));
        assert_eq!(
            buffer.get_tile(headroom.layer, headroom.x, headroom.y),
            Some(crate::layout::Tile::Void)
        );
        assert_eq!(
            buffer.get_tile(
                transition_owned.layer,
                transition_owned.x,
                transition_owned.y
            ),
            Some(crate::layout::Tile::Void)
        );
    }

    #[test]
    fn exact_width_expansion_uses_n_cells_and_checked_bounds() {
        let config = GeneratorConfig::custom(64, 64, 2)
            .normalize()
            .expect("config");
        let path = [
            GridCoord {
                layer: 0,
                x: 5,
                y: 5,
            },
            GridCoord {
                layer: 0,
                x: 6,
                y: 5,
            },
        ];
        assert_eq!(
            expanded_path(&path, 1, &config).expect("width one").len(),
            2
        );
        assert_eq!(
            expanded_path(&path, 2, &config).expect("width two").len(),
            4
        );
        let boundary = [
            GridCoord {
                layer: 0,
                x: 0,
                y: 0,
            },
            GridCoord {
                layer: 0,
                x: 1,
                y: 0,
            },
        ];
        assert!(expanded_path(&boundary, 2, &config).is_err());
    }

    #[test]
    fn materialization_succeeds_with_border_reservation() {
        // After reserving border cells, Phase 03 must produce a complete legal
        // witness that Phase 04 can materialize, not merely a non-border error.
        let (config, catalog, topology) = selected(77);
        let interior = |cell: &GridCoord| {
            cell.x > 0
                && cell.y > 0
                && cell.x.checked_add(1) < Some(config.width())
                && cell.y.checked_add(1) < Some(config.height())
        };
        for edge in topology
            .edges
            .iter()
            .filter(|edge| edge.transition.is_none())
        {
            assert!(
                edge.path_witness.iter().all(interior),
                "edge {} has a border witness",
                edge.id.raw()
            );
            assert!(
                edge.allowed_envelope_cells.iter().all(interior),
                "edge {} has a border envelope cell",
                edge.id.raw()
            );
        }
        let level = materialize_topology(
            &topology,
            &catalog,
            &config,
            &mut AttemptContext::new(TelemetryMode::Off),
        )
        .expect("seed 77 topology should materialize after border reservation")
        .into_parsed_level((1, 1));
        let lookup = |layer, x, y| {
            (layer < config.layers().2 && x < config.width() && y < config.height())
                .then(|| level.tile_at_3d(usize::from(layer), usize::from(x), usize::from(y)))
        };
        let inferred = super::super::ramps::scan_transitions(
            config.width(),
            config.height(),
            config.layers().2,
            &lookup,
        );
        assert_eq!(inferred.len(), topology.transitions.len());
        for ramp in inferred {
            assert_eq!(
                level.tile_at_3d(
                    usize::from(ramp.lower_layer),
                    usize::from(ramp.upper_landing.1),
                    usize::from(ramp.upper_landing.2),
                ),
                crate::layout::Tile::Void,
                "lower crest exit must be open beneath the upper landing",
            );
        }
        for layer in 0..level.layer_count() {
            for x in 0..level.width {
                assert_eq!(level.tile_at_3d(layer, x, 0), crate::layout::Tile::Wall);
                assert_eq!(
                    level.tile_at_3d(layer, x, level.height - 1),
                    crate::layout::Tile::Wall
                );
            }
            for y in 0..level.height {
                assert_eq!(level.tile_at_3d(layer, 0, y), crate::layout::Tile::Wall);
                assert_eq!(
                    level.tile_at_3d(layer, level.width - 1, y),
                    crate::layout::Tile::Wall
                );
            }
        }
    }
}
