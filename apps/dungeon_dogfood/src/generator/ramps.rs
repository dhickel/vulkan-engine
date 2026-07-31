use std::collections::BTreeSet;

use super::config::NormalizedGeneratorConfig;
use super::error::{ErrorStage, GeneratorError};
use super::ir::{Direction, GridCoord, TransitionReservation};
use super::prefab::Direction as PrefabDirection;

pub(super) type TileLookup<'a> = &'a dyn Fn(u16, u16, u16) -> Option<crate::layout::Tile>;

/// A complete tile-derived transition. This intentionally contains no generator
/// ownership metadata so geometry and later validation can share the contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InferredTransition {
    pub(super) lower_anchor: (u16, u16, u16),
    pub(super) lower_layer: u16,
    pub(super) upper_layer: u16,
    pub(super) direction: Direction,
    pub(crate) ramp_cells: [(u16, u16, u16); 3],
    pub(super) opening_cells: [(u16, u16, u16); 3],
    pub(super) lower_approach: (u16, u16, u16),
    pub(super) upper_landing: (u16, u16, u16),
}

/// Pure recognition of the only tile pattern that creates a traversable ramp
/// opening. Missing and malformed patterns return `None`, never an error.
pub(super) fn infer_ramp_transition(
    width: u16,
    height: u16,
    layers: u16,
    lookup: TileLookup<'_>,
    lower_layer: u16,
    anchor_x: u16,
    anchor_y: u16,
    direction: Direction,
) -> Option<InferredTransition> {
    if width == 0 || height == 0 || layers < 2 || anchor_x >= width || anchor_y >= height {
        return None;
    }
    let upper_layer = lower_layer.checked_add(1)?;
    if upper_layer >= layers {
        return None;
    }
    let (dx, dy) = direction.delta();
    let mut ramps = [(0, 0, 0); 3];
    let mut x = anchor_x;
    let mut y = anchor_y;
    for step in 0..3u8 {
        if x >= width
            || y >= height
            || lookup(lower_layer, x, y)? != layout_ramp_tile(direction, step)
        {
            return None;
        }
        ramps[usize::from(step)] = (lower_layer, x, y);
        if step != 2 {
            x = checked_offset(x, dx)?;
            y = checked_offset(y, dy)?;
        }
    }

    let approach_x = checked_offset(anchor_x, dx.checked_neg()?)?;
    let approach_y = checked_offset(anchor_y, dy.checked_neg()?)?;
    if approach_x >= width
        || approach_y >= height
        || lookup(lower_layer, approach_x, approach_y)? != crate::layout::Tile::Floor
    {
        return None;
    }

    let mut openings = [(0, 0, 0); 3];
    for (index, (_, ramp_x, ramp_y)) in ramps.iter().copied().enumerate() {
        if lookup(upper_layer, ramp_x, ramp_y)? != crate::layout::Tile::Void {
            return None;
        }
        openings[index] = (upper_layer, ramp_x, ramp_y);
    }

    let landing_x = checked_offset(x, dx)?;
    let landing_y = checked_offset(y, dy)?;
    if landing_x >= width
        || landing_y >= height
        || lookup(upper_layer, landing_x, landing_y)? != crate::layout::Tile::Floor
    {
        return None;
    }

    // An extra lower ramp beyond R2 makes the sequence ambiguous.
    if lookup(lower_layer, landing_x, landing_y).is_some_and(layout_tile_is_ramp) {
        return None;
    }

    // Where another layer exists, the landing must have clear vertical
    // headroom. The aligned Void cells above the run provide run headroom.
    if let Some(headroom_layer) = upper_layer.checked_add(1).filter(|layer| *layer < layers) {
        if lookup(headroom_layer, landing_x, landing_y)? != crate::layout::Tile::Void {
            return None;
        }
    }

    Some(InferredTransition {
        lower_anchor: (lower_layer, anchor_x, anchor_y),
        lower_layer,
        upper_layer,
        direction,
        ramp_cells: ramps,
        opening_cells: openings,
        lower_approach: (lower_layer, approach_x, approach_y),
        upper_landing: (upper_layer, landing_x, landing_y),
    })
}

/// Canonical whole-grid inference in `(layer, y, x, direction-rank)` order.
/// A low-end anchor can produce at most one result; malformed multi-direction
/// ambiguity is discarded rather than exposed nondeterministically.
pub(crate) fn scan_transitions(
    width: u16,
    height: u16,
    layers: u16,
    lookup: TileLookup<'_>,
) -> Vec<InferredTransition> {
    let mut by_anchor = std::collections::BTreeMap::new();
    let Some(lower_layer_count) = layers.checked_sub(1) else {
        return Vec::new();
    };
    for layer in 0..lower_layer_count {
        for y in 0..height {
            for x in 0..width {
                for direction in canonical_directions() {
                    if let Some(found) =
                        infer_ramp_transition(width, height, layers, lookup, layer, x, y, direction)
                    {
                        by_anchor
                            .entry(found.lower_anchor)
                            .or_insert_with(Vec::new)
                            .push(found);
                    }
                }
            }
        }
    }
    by_anchor
        .into_values()
        .filter_map(|mut matches| (matches.len() == 1).then(|| matches.remove(0)))
        .collect()
}

pub(super) trait TileBufferWrite {
    fn set_tile(
        &mut self,
        layer: u16,
        x: u16,
        y: u16,
        tile: crate::layout::Tile,
    ) -> Result<(), GeneratorError>;
    fn get_tile(&self, layer: u16, x: u16, y: u16) -> Option<crate::layout::Tile>;
    fn dimensions(&self) -> (u16, u16, u16);
}

/// Validate and commit one complete reservation. All checks happen before the
/// first write, so a failed transition cannot leak partial tiles.
pub(super) fn materialize_transition(
    transition: &TransitionReservation,
    config: &NormalizedGeneratorConfig,
    buffer: &mut dyn TileBufferWrite,
) -> Result<(), GeneratorError> {
    let dimensions = (config.width(), config.height(), config.layers().2);
    if buffer.dimensions() != dimensions {
        return Err(materialization_error(
            "tile_buffer_dimension_mismatch",
            format!("transition={}", transition.id.raw()),
        ));
    }
    let upper_layer =
        transition
            .lower_layer
            .checked_add(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Materialization,
                operation: "ramp_upper_layer",
            })?;
    if upper_layer >= dimensions.2 {
        return Err(materialization_error(
            "ramp_upper_layer_oob",
            format!("transition={}", transition.id.raw()),
        ));
    }

    let mut step_cells = [None; 3];
    let mut direction = None;
    for cell in &transition.ramp_run_cells {
        require_in_bounds(*cell, dimensions, "ramp_cell_oob")?;
        if cell.layer != transition.lower_layer {
            return Err(materialization_error(
                "ramp_cell_wrong_layer",
                cell.to_string(),
            ));
        }
        let tile = buffer
            .get_tile(cell.layer, cell.x, cell.y)
            .ok_or_else(|| materialization_error("ramp_cell_missing", cell.to_string()))?;
        let (tile_direction, step) = ramp_tile_parts(tile)
            .ok_or_else(|| materialization_error("ramp_tile_missing", cell.to_string()))?;
        if step > 2 || direction.is_some_and(|found| found != tile_direction) {
            return Err(materialization_error(
                "ramp_sequence_malformed",
                cell.to_string(),
            ));
        }
        direction = Some(tile_direction);
        let slot = &mut step_cells[usize::from(step)];
        if slot.replace(*cell).is_some() {
            return Err(materialization_error(
                "ramp_step_duplicate",
                cell.to_string(),
            ));
        }
    }
    if transition.ramp_run_cells.len() != 3 || step_cells.iter().any(Option::is_none) {
        return Err(materialization_error(
            "ramp_sequence_incomplete",
            format!("transition={}", transition.id.raw()),
        ));
    }
    let direction = direction.ok_or_else(|| {
        materialization_error(
            "ramp_direction_missing",
            format!("transition={}", transition.id.raw()),
        )
    })?;
    let r0 = step_cells[0].ok_or_else(|| materialization_error("ramp_r0_missing", "".into()))?;
    let (dx, dy) = direction.delta();
    let expected_r1 = offset_coord(r0, dx, dy, dimensions, "ramp_r1_oob")?;
    let expected_r2 = offset_coord(expected_r1, dx, dy, dimensions, "ramp_r2_oob")?;
    if step_cells != [Some(r0), Some(expected_r1), Some(expected_r2)] {
        return Err(materialization_error(
            "ramp_sequence_noncontiguous",
            format!("transition={}", transition.id.raw()),
        ));
    }

    let approach = offset_coord(
        r0,
        dx.checked_neg().ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "ramp_approach_dx_neg",
        })?,
        dy.checked_neg().ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "ramp_approach_dy_neg",
        })?,
        dimensions,
        "ramp_approach_oob",
    )?;
    if buffer.get_tile(approach.layer, approach.x, approach.y) != Some(crate::layout::Tile::Floor) {
        return Err(materialization_error(
            "ramp_approach_blocked",
            approach.to_string(),
        ));
    }
    let expected_openings: BTreeSet<GridCoord> = [r0, expected_r1, expected_r2]
        .into_iter()
        .map(|cell| GridCoord {
            layer: upper_layer,
            x: cell.x,
            y: cell.y,
        })
        .collect();
    let actual_openings: BTreeSet<GridCoord> =
        transition.upper_opening_cells.iter().copied().collect();
    if actual_openings != expected_openings {
        return Err(materialization_error(
            "ramp_opening_mask_mismatch",
            format!("transition={}", transition.id.raw()),
        ));
    }

    let landing = offset_coord(
        GridCoord {
            layer: upper_layer,
            x: expected_r2.x,
            y: expected_r2.y,
        },
        dx,
        dy,
        dimensions,
        "ramp_landing_oob",
    )?;
    if !transition.landing_cells.contains(&landing) {
        return Err(materialization_error(
            "ramp_landing_unreserved",
            landing.to_string(),
        ));
    }

    let mut writes = Vec::new();
    for (step, cell) in [r0, expected_r1, expected_r2].into_iter().enumerate() {
        let step = u8::try_from(step).map_err(|_| GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Materialization,
            operation: "ramp_step_convert",
        })?;
        validate_exact_or_clear(buffer, cell, layout_ramp_tile(direction, step), false)?;
        writes.push((cell, layout_ramp_tile(direction, step)));
    }
    for cell in expected_openings {
        validate_exact_or_clear(buffer, cell, crate::layout::Tile::Void, false)?;
        writes.push((cell, crate::layout::Tile::Void));
    }
    validate_exact_or_clear(buffer, landing, crate::layout::Tile::Floor, false)?;
    writes.push((landing, crate::layout::Tile::Floor));

    for cell in &transition.landing_cells {
        require_in_bounds(*cell, dimensions, "landing_cell_oob")?;
        if cell.layer != upper_layer
            || buffer.get_tile(cell.layer, cell.x, cell.y) != Some(crate::layout::Tile::Floor)
        {
            return Err(materialization_error(
                "landing_cell_blocked",
                cell.to_string(),
            ));
        }
    }
    for cell in &transition.headroom_cells {
        require_in_bounds(*cell, dimensions, "headroom_cell_oob")?;
        if buffer.get_tile(cell.layer, cell.x, cell.y) != Some(crate::layout::Tile::Void) {
            return Err(materialization_error(
                "headroom_cell_blocked",
                cell.to_string(),
            ));
        }
    }

    let lookup = |layer, x, y| buffer.get_tile(layer, x, y);
    let inferred = infer_ramp_transition(
        dimensions.0,
        dimensions.1,
        dimensions.2,
        &lookup,
        transition.lower_layer,
        r0.x,
        r0.y,
        direction,
    )
    .ok_or_else(|| {
        materialization_error(
            "materialized_ramp_not_inferred",
            format!("transition={}", transition.id.raw()),
        )
    })?;
    if inferred.upper_landing != (landing.layer, landing.x, landing.y) {
        return Err(materialization_error(
            "materialized_ramp_landing_mismatch",
            landing.to_string(),
        ));
    }

    for (cell, tile) in writes {
        buffer.set_tile(cell.layer, cell.x, cell.y, tile)?;
    }
    Ok(())
}

pub(super) fn materialize_all_transitions(
    transitions: &[TransitionReservation],
    config: &NormalizedGeneratorConfig,
    buffer: &mut dyn TileBufferWrite,
) -> Result<(), GeneratorError> {
    let mut sorted: Vec<_> = transitions.iter().collect();
    sorted.sort_by_key(|transition| transition.id);
    for transition in sorted {
        materialize_transition(transition, config, buffer)?;
    }

    let (width, height, layers) = buffer.dimensions();
    let lookup = |layer, x, y| buffer.get_tile(layer, x, y);
    let inferred = scan_transitions(width, height, layers, &lookup);
    if inferred.len() != transitions.len() {
        return Err(materialization_error(
            "transition_inference_count_mismatch",
            format!("reserved={} inferred={}", transitions.len(), inferred.len()),
        ));
    }
    Ok(())
}

fn validate_exact_or_clear(
    buffer: &dyn TileBufferWrite,
    cell: GridCoord,
    wanted: crate::layout::Tile,
    allow_clear: bool,
) -> Result<(), GeneratorError> {
    match buffer.get_tile(cell.layer, cell.x, cell.y) {
        Some(existing) if existing == wanted => Ok(()),
        Some(crate::layout::Tile::Void) if allow_clear => Ok(()),
        Some(existing) => Err(GeneratorError::TileBufferConflict {
            stage: ErrorStage::Materialization,
            detail: format!(
                "transition_write_conflict cell={} existing={:?} wanted={:?}",
                cell, existing, wanted
            ),
        }),
        None => Err(materialization_error(
            "transition_cell_missing",
            cell.to_string(),
        )),
    }
}

fn require_in_bounds(
    cell: GridCoord,
    dimensions: (u16, u16, u16),
    constraint: &'static str,
) -> Result<(), GeneratorError> {
    if cell.layer >= dimensions.2 || cell.x >= dimensions.0 || cell.y >= dimensions.1 {
        return Err(materialization_error(constraint, cell.to_string()));
    }
    Ok(())
}

fn offset_coord(
    cell: GridCoord,
    dx: i32,
    dy: i32,
    dimensions: (u16, u16, u16),
    constraint: &'static str,
) -> Result<GridCoord, GeneratorError> {
    let x = checked_offset(cell.x, dx)
        .ok_or_else(|| materialization_error(constraint, cell.to_string()))?;
    let y = checked_offset(cell.y, dy)
        .ok_or_else(|| materialization_error(constraint, cell.to_string()))?;
    let next = GridCoord {
        layer: cell.layer,
        x,
        y,
    };
    require_in_bounds(next, dimensions, constraint)?;
    Ok(next)
}

fn materialization_error(constraint: &'static str, detail: String) -> GeneratorError {
    GeneratorError::MaterializationInfeasible {
        stage: ErrorStage::Materialization,
        constraint,
        detail,
    }
}

fn canonical_directions() -> [Direction; 4] {
    [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ]
}

fn layout_tile_is_ramp(tile: crate::layout::Tile) -> bool {
    ramp_tile_parts(tile).is_some()
}

fn ramp_tile_parts(tile: crate::layout::Tile) -> Option<(Direction, u8)> {
    match tile {
        crate::layout::Tile::RampNorth(step) => Some((Direction::North, step)),
        crate::layout::Tile::RampEast(step) => Some((Direction::East, step)),
        crate::layout::Tile::RampSouth(step) => Some((Direction::South, step)),
        crate::layout::Tile::RampWest(step) => Some((Direction::West, step)),
        _ => None,
    }
}

fn layout_ramp_tile(direction: Direction, step: u8) -> crate::layout::Tile {
    match direction {
        Direction::North => crate::layout::Tile::RampNorth(step),
        Direction::East => crate::layout::Tile::RampEast(step),
        Direction::South => crate::layout::Tile::RampSouth(step),
        Direction::West => crate::layout::Tile::RampWest(step),
    }
}

fn checked_offset(value: u16, delta: i32) -> Option<u16> {
    u16::try_from(i32::from(value).checked_add(delta)?).ok()
}

/// Convert a prefab tile to a layout tile. Variants are already transformed;
/// callers stamping a variant pass zero. The rotation argument remains useful
/// for direct conversion tests and non-variant callers.
pub(super) fn prefab_tile_to_layout(
    tile: super::prefab::Tile,
    rotation_quarter_turns: u8,
) -> crate::layout::Tile {
    match tile {
        super::prefab::Tile::Wall => crate::layout::Tile::Wall,
        super::prefab::Tile::Floor => crate::layout::Tile::Floor,
        super::prefab::Tile::Void => crate::layout::Tile::Void,
        super::prefab::Tile::Ramp { direction, step } => layout_ramp_tile(
            rotate_prefab_direction(direction, rotation_quarter_turns),
            step,
        ),
    }
}

fn rotate_prefab_direction(direction: PrefabDirection, quarter_turns: u8) -> Direction {
    let mut direction = match direction {
        PrefabDirection::North => Direction::North,
        PrefabDirection::East => Direction::East,
        PrefabDirection::South => Direction::South,
        PrefabDirection::West => Direction::West,
    };
    for _ in 0..(quarter_turns % 4) {
        direction = match direction {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
        };
    }
    direction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Tile;

    fn fixture(direction: Direction, lower_layer: u16, layers: u16) -> (u16, u16, Vec<Tile>) {
        let width = 9u16;
        let height = 9u16;
        let mut tiles =
            vec![Tile::Wall; usize::from(width) * usize::from(height) * usize::from(layers)];
        let (anchor_x, anchor_y) = (4u16, 4u16);
        let (dx, dy) = direction.delta();
        let set = |tiles: &mut Vec<Tile>, layer: u16, x: u16, y: u16, tile| {
            let index = usize::from(layer) * usize::from(width) * usize::from(height)
                + usize::from(y) * usize::from(width)
                + usize::from(x);
            tiles[index] = tile;
        };
        let approach_x = checked_offset(anchor_x, -dx).expect("approach x");
        let approach_y = checked_offset(anchor_y, -dy).expect("approach y");
        set(&mut tiles, lower_layer, approach_x, approach_y, Tile::Floor);
        let mut x = anchor_x;
        let mut y = anchor_y;
        for step in 0..3 {
            set(
                &mut tiles,
                lower_layer,
                x,
                y,
                layout_ramp_tile(direction, step),
            );
            set(&mut tiles, lower_layer + 1, x, y, Tile::Void);
            x = checked_offset(x, dx).expect("ramp x");
            y = checked_offset(y, dy).expect("ramp y");
        }
        set(&mut tiles, lower_layer + 1, x, y, Tile::Floor);
        if lower_layer + 2 < layers {
            set(&mut tiles, lower_layer + 2, x, y, Tile::Void);
        }
        (anchor_x, anchor_y, tiles)
    }

    #[test]
    fn inference_and_scan_cover_every_direction_and_lower_layer() {
        for layers in 2..=4 {
            for lower in 0..layers - 1 {
                for direction in canonical_directions() {
                    let (x, y, tiles) = fixture(direction, lower, layers);
                    let lookup = |layer: u16, x: u16, y: u16| {
                        if layer >= layers || x >= 9 || y >= 9 {
                            return None;
                        }
                        let index = usize::from(layer) * 81 + usize::from(y) * 9 + usize::from(x);
                        tiles.get(index).copied()
                    };
                    let inferred =
                        infer_ramp_transition(9, 9, layers, &lookup, lower, x, y, direction)
                            .expect("complete transition");
                    assert_eq!(inferred.opening_cells.len(), 3);
                    assert_eq!(scan_transitions(9, 9, layers, &lookup), vec![inferred]);
                }
            }
        }
    }

    #[test]
    fn inference_rejects_malformed_and_generic_void_patterns() {
        let (x, y, base) = fixture(Direction::East, 0, 3);
        let cases: [(u16, u16, u16, Tile); 6] = [
            (0, 5, 4, Tile::RampWest(1)),
            (0, 6, 4, Tile::RampEast(1)),
            (0, 3, 4, Tile::Wall),
            (1, 4, 4, Tile::Floor),
            (1, 7, 4, Tile::Void),
            (2, 7, 4, Tile::Wall),
        ];
        for (layer, tx, ty, replacement) in cases {
            let mut tiles = base.clone();
            let index = usize::from(layer) * 81 + usize::from(ty) * 9 + usize::from(tx);
            tiles[index] = replacement;
            let lookup = |layer: u16, x: u16, y: u16| {
                (layer < 3 && x < 9 && y < 9)
                    .then(|| tiles[usize::from(layer) * 81 + usize::from(y) * 9 + usize::from(x)])
            };
            assert!(infer_ramp_transition(9, 9, 3, &lookup, 0, x, y, Direction::East).is_none());
        }
        let generic = vec![Tile::Void; 2 * 9 * 9];
        let lookup = |layer: u16, x: u16, y: u16| {
            (layer < 2 && x < 9 && y < 9)
                .then(|| generic[usize::from(layer) * 81 + usize::from(y) * 9 + usize::from(x)])
        };
        assert!(scan_transitions(9, 9, 2, &lookup).is_empty());
    }

    #[test]
    fn inference_rejects_bounds_and_top_layer_without_panicking() {
        let lookup = |_layer, _x, _y| Some(Tile::RampNorth(0));
        assert!(infer_ramp_transition(0, 9, 2, &lookup, 0, 0, 0, Direction::North).is_none());
        assert!(infer_ramp_transition(9, 9, 2, &lookup, 1, 4, 4, Direction::North).is_none());
        assert!(infer_ramp_transition(9, 9, 2, &lookup, 0, 0, 0, Direction::North).is_none());
    }

    #[test]
    fn prefab_tile_conversion_rotates_when_explicitly_requested() {
        let ramp = super::super::prefab::Tile::Ramp {
            direction: PrefabDirection::North,
            step: 1,
        };
        assert_eq!(prefab_tile_to_layout(ramp, 0), Tile::RampNorth(1));
        assert_eq!(prefab_tile_to_layout(ramp, 1), Tile::RampEast(1));
        assert_eq!(prefab_tile_to_layout(ramp, 2), Tile::RampSouth(1));
        assert_eq!(prefab_tile_to_layout(ramp, 3), Tile::RampWest(1));
    }
}
