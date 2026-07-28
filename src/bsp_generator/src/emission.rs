//! Build the final brush/entity emission intent for a generated BSP dungeon.
//!
//! Rooms are emitted as explicit floor, ceiling, and four wall slabs.  Wall
//! cells intersected by a routed corridor are omitted from the wall mask, so
//! portals are real apertures rather than additive "opening" brushes.
//! Corridors use a grid union only for their own shell: this keeps turns and
//! L/T/X junctions open without merging room floors or ceilings into broad
//! scene-spanning slabs.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::CONSTRUCTION_QUANTUM;
use crate::error::GeneratorError;
use crate::intent::{
    Brush, Corridor, EmissionIntent, EntityIntent, LayoutIntent, RoomIntent, RoutedIntent,
};
use crate::junction;
use crate::routing::{CORRIDOR_HEIGHT, CORRIDOR_WIDTH};

/// CC0 Stone Beta role bindings from `themes/cc0_stone_beta/theme.toml`.
const FLOOR_TEXTURE: &str = "stone_floor";
const WALL_TEXTURE: &str = "stone_wall";
const CEILING_TEXTURE: &str = "stone_ceiling";

const SLAB: i32 = CONSTRUCTION_QUANTUM as i32;
const EYE_OFFSET: i32 = 24;

type Cell = (i32, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoomWall {
    North,
    South,
    East,
    West,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Opening {
    tangent_min: i32,
    tangent_max: i32,
    bottom: i32,
    top: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GridRect<T> {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    value: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CorridorCell {
    floor_z: i32,
    ceiling_bottom: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WallSpan {
    floor_z: i32,
    shell_top: i32,
}

/// Build the complete worldspawn brush set and point entities.
///
/// Direct public-IR callers are validated before any source geometry is
/// constructed. In particular, a corridor must be level and exactly 64×80,
/// which is the fixed vertical contract approved by `DECISION-20260726-02`.
pub fn build_emission(
    layout: &LayoutIntent,
    routed: &RoutedIntent,
) -> Result<EmissionIntent, GeneratorError> {
    validate_corridor_contract(layout, routed)?;

    let mut brushes = Vec::new();
    let corridor_cells = build_corridor_cells(&routed.corridors);

    for room in &layout.rooms {
        brushes.extend(build_room_brushes(room, &routed.corridors, &corridor_cells));
    }

    // Every XY cell in a room shell is room-owned. Corridor shell components
    // are omitted there, leaving room floor/ceiling/wall pieces as the only
    // source-solid owners at portals and endpoint turning chambers.
    let room_owned_cells = build_room_owned_cells(layout);
    brushes.extend(build_corridor_slabs(&corridor_cells, &room_owned_cells));
    brushes.extend(build_corridor_boundary_walls(
        &corridor_cells,
        &room_owned_cells,
    ));

    validate_no_positive_volume_overlap(&brushes)?;

    let mut entities = Vec::new();
    if let Some(first) = layout.rooms.first() {
        entities.push(point_entity(
            "info_player_start",
            spawn_origin(first),
            Vec::new(),
        ));
    }
    for room in &layout.rooms {
        entities.push(point_entity(
            "light",
            light_origin(room),
            vec![("light".to_string(), "300".to_string())],
        ));
    }

    Ok(EmissionIntent {
        brushes,
        entities,
        wad: "cc0_stone_beta.wad".to_string(),
    })
}

fn validate_corridor_contract(
    layout: &LayoutIntent,
    routed: &RoutedIntent,
) -> Result<(), GeneratorError> {
    let common_floor = layout.rooms.first().map(|room| room.position.2);

    for (index, corridor) in routed.corridors.iter().enumerate() {
        if corridor.height != CORRIDOR_HEIGHT {
            return Err(GeneratorError::InvariantViolation(format!(
                "corridor {index} height {} != {CORRIDOR_HEIGHT} (DECISION-20260726-02)",
                corridor.height,
            )));
        }
        if corridor.width != CORRIDOR_WIDTH {
            return Err(GeneratorError::InvariantViolation(format!(
                "corridor {index} width {} != frozen width {CORRIDOR_WIDTH}",
                corridor.width,
            )));
        }
        if corridor.start.2 != corridor.end.2 {
            return Err(GeneratorError::InvariantViolation(format!(
                "corridor {index} has unequal endpoint Z values {} and {}",
                corridor.start.2, corridor.end.2,
            )));
        }
        if let Some(floor_z) = common_floor {
            if corridor.start.2 != floor_z {
                return Err(GeneratorError::InvariantViolation(format!(
                    "corridor {index} floor Z {} differs from the single-layer room floor {floor_z}",
                    corridor.start.2,
                )));
            }
        }

        let q = SLAB;
        for (label, point) in [("start", corridor.start), ("end", corridor.end)] {
            if point.0.rem_euclid(q) != 0
                || point.1.rem_euclid(q) != 0
                || point.2.rem_euclid(q) != 0
            {
                return Err(GeneratorError::InvariantViolation(format!(
                    "corridor {index} {label} {point:?} is not quantum-aligned",
                )));
            }
        }

        let same_x = corridor.start.0 == corridor.end.0;
        let same_y = corridor.start.1 == corridor.end.1;
        if same_x == same_y {
            return Err(GeneratorError::InvariantViolation(format!(
                "corridor {index} must be a non-zero axis-aligned segment: {:?} -> {:?}",
                corridor.start, corridor.end,
            )));
        }
    }

    Ok(())
}

fn point_entity(
    classname: &str,
    origin: (i32, i32, i32),
    mut extra: Vec<(String, String)>,
) -> EntityIntent {
    let mut properties = vec![
        ("classname".to_string(), classname.to_string()),
        (
            "origin".to_string(),
            format!("{} {} {}", origin.0, origin.1, origin.2),
        ),
    ];
    properties.append(&mut extra);
    properties.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    EntityIntent {
        classname: classname.to_string(),
        origin,
        properties,
        brushes: Vec::new(),
    }
}

fn spawn_origin(room: &RoomIntent) -> (i32, i32, i32) {
    (
        room.position.0 + room.dimensions.0 as i32 / 2,
        room.position.1 + room.dimensions.1 as i32 / 2,
        room.position.2 + SLAB + EYE_OFFSET,
    )
}

fn light_origin(room: &RoomIntent) -> (i32, i32, i32) {
    let clear_height = room.dimensions.2 as i32 - 2 * SLAB;
    (
        room.position.0 + room.dimensions.0 as i32 / 2,
        room.position.1 + room.dimensions.1 as i32 / 2,
        room.position.2 + SLAB + clear_height / 2,
    )
}

// ── Room shell ────────────────────────────────────────────────────────────

fn build_room_brushes(
    room: &RoomIntent,
    corridors: &[Corridor],
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
) -> Vec<Brush> {
    let (x, y, z) = room.position;
    let dx = room.dimensions.0 as i32;
    let dy = room.dimensions.1 as i32;
    let dz = room.dimensions.2 as i32;
    let mut brushes = Vec::new();

    push_box(
        &mut brushes,
        (x, y, z),
        (x + dx, y + dy, z + SLAB),
        FLOOR_TEXTURE,
    );
    push_box(
        &mut brushes,
        (x, y, z + dz - SLAB),
        (x + dx, y + dy, z + dz),
        CEILING_TEXTURE,
    );

    for wall in [
        RoomWall::North,
        RoomWall::South,
        RoomWall::East,
        RoomWall::West,
    ] {
        let openings = openings_for_wall(room, corridors, wall);
        brushes.extend(build_split_wall(room, wall, &openings, corridor_cells));
    }

    brushes
}

fn openings_for_wall(room: &RoomIntent, corridors: &[Corridor], wall: RoomWall) -> Vec<Opening> {
    let min_x = room.position.0;
    let max_x = min_x + room.dimensions.0 as i32;
    let min_y = room.position.1;
    let max_y = min_y + room.dimensions.1 as i32;
    let room_bottom = room.position.2;
    let room_ceiling_bottom = room_bottom + room.dimensions.2 as i32 - SLAB;
    let (wall_coord, tangent_min, tangent_max, expected_orientation) = match wall {
        RoomWall::West => (min_x, min_y, max_y, Orientation::Horizontal),
        RoomWall::East => (max_x, min_y, max_y, Orientation::Horizontal),
        RoomWall::South => (min_y, min_x, max_x, Orientation::Vertical),
        RoomWall::North => (max_y, min_x, max_x, Orientation::Vertical),
    };

    let mut openings = Vec::new();
    for corridor in corridors {
        let orientation = corridor_orientation(corridor);
        if orientation != expected_orientation {
            continue;
        }

        let (line_min, line_max, tangent_center) = match orientation {
            Orientation::Horizontal => (
                corridor.start.0.min(corridor.end.0),
                corridor.start.0.max(corridor.end.0),
                corridor.start.1,
            ),
            Orientation::Vertical => (
                corridor.start.1.min(corridor.end.1),
                corridor.start.1.max(corridor.end.1),
                corridor.start.0,
            ),
        };
        if wall_coord < line_min || wall_coord > line_max {
            continue;
        }

        let half = corridor.width as i32 / 2;
        let opening = Opening {
            tangent_min: (tangent_center - half).max(tangent_min + SLAB),
            tangent_max: (tangent_center + half).min(tangent_max - SLAB),
            bottom: (corridor.start.2.min(corridor.end.2) + SLAB).max(room_bottom + SLAB),
            top: (corridor.start.2.min(corridor.end.2) + SLAB + corridor.height as i32)
                .min(room_ceiling_bottom),
        };
        if opening.tangent_min < opening.tangent_max && opening.bottom < opening.top {
            openings.push(opening);
        }
    }

    openings.sort_unstable_by_key(|opening| {
        (
            opening.tangent_min,
            opening.tangent_max,
            opening.bottom,
            opening.top,
        )
    });
    openings.dedup();
    openings
}

fn build_split_wall(
    room: &RoomIntent,
    wall: RoomWall,
    openings: &[Opening],
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
) -> Vec<Brush> {
    let min_x = room.position.0;
    let max_x = min_x + room.dimensions.0 as i32;
    let min_y = room.position.1;
    let max_y = min_y + room.dimensions.1 as i32;
    let z0 = room.position.2;
    let z1 = z0 + room.dimensions.2 as i32;
    // G3: Walls span only the vertical range between floor and ceiling slabs.
    // The floor slab (z0..z0+SLAB) and ceiling slab (z1-SLAB..z1) already
    // provide solid geometry; wall cells in those bands would overlap them.
    let wall_z0 = z0 + SLAB;
    let wall_z1 = z1 - SLAB;
    // North/south walls own the four corner columns. Restricting east/west
    // walls to the interior tangent interval makes every room-shell source
    // box disjoint while retaining a sealed one-cell corner on every side.
    let (tangent_min, tangent_max) = match wall {
        RoomWall::West | RoomWall::East => (min_y + SLAB, max_y - SLAB),
        RoomWall::South | RoomWall::North => (min_x, max_x),
    };

    let mut solid_cells = BTreeMap::new();
    for z_cell in wall_z0.div_euclid(SLAB)..wall_z1.div_euclid(SLAB) {
        for tangent_cell in tangent_min.div_euclid(SLAB)..tangent_max.div_euclid(SLAB) {
            solid_cells.insert((tangent_cell, z_cell), 0);
        }
    }
    for opening in openings {
        for z_cell in opening.bottom.div_euclid(SLAB)..opening.top.div_euclid(SLAB) {
            for tangent_cell in
                opening.tangent_min.div_euclid(SLAB)..opening.tangent_max.div_euclid(SLAB)
            {
                solid_cells.remove(&(tangent_cell, z_cell));
            }
        }
    }

    // Endpoint turning chambers can touch an endpoint room wall even when the
    // segment centerline itself turns just outside it. Preserve the routed
    // 64×64 clear footprint at the compiled boundary by subtracting every
    // corridor-open wall cell, not only centerline-normal portal intervals.
    // The room's floor and lintel still bound the aperture vertically.
    let opening_bottom = z0 + SLAB;
    let opening_limit = z1 - SLAB;
    for tangent_cell in tangent_min.div_euclid(SLAB)..tangent_max.div_euclid(SLAB) {
        let wall_cell = match wall {
            RoomWall::West => (min_x.div_euclid(SLAB), tangent_cell),
            RoomWall::East => (max_x.div_euclid(SLAB) - 1, tangent_cell),
            RoomWall::South => (tangent_cell, min_y.div_euclid(SLAB)),
            RoomWall::North => (tangent_cell, max_y.div_euclid(SLAB) - 1),
        };
        let Some(corridor_cell) = corridor_cells.get(&wall_cell) else {
            continue;
        };
        let opening_top = corridor_cell.ceiling_bottom.min(opening_limit);
        for z_cell in opening_bottom.div_euclid(SLAB)..opening_top.div_euclid(SLAB) {
            solid_cells.remove(&(tangent_cell, z_cell));
        }
    }

    let mut brushes = Vec::new();
    for rect in merge_cells(&solid_cells) {
        let tangent0 = rect.x0 * SLAB;
        let tangent1 = rect.x1 * SLAB;
        let wall_z0 = rect.y0 * SLAB;
        let wall_z1 = rect.y1 * SLAB;
        let (min, max) = match wall {
            RoomWall::West => (
                (min_x, tangent0, wall_z0),
                (min_x + SLAB, tangent1, wall_z1),
            ),
            RoomWall::East => (
                (max_x - SLAB, tangent0, wall_z0),
                (max_x, tangent1, wall_z1),
            ),
            RoomWall::South => (
                (tangent0, min_y, wall_z0),
                (tangent1, min_y + SLAB, wall_z1),
            ),
            RoomWall::North => (
                (tangent0, max_y - SLAB, wall_z0),
                (tangent1, max_y, wall_z1),
            ),
        };
        push_box(&mut brushes, min, max, WALL_TEXTURE);
    }
    brushes
}

// ── Corridor shell ────────────────────────────────────────────────────────

fn corridor_orientation(corridor: &Corridor) -> Orientation {
    if corridor.start.1 == corridor.end.1 {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    }
}

fn mark_open_rect(
    cells: &mut BTreeMap<Cell, CorridorCell>,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    span: CorridorCell,
) {
    debug_assert_eq!(min_x.rem_euclid(SLAB), 0);
    debug_assert_eq!(min_y.rem_euclid(SLAB), 0);
    debug_assert_eq!(max_x.rem_euclid(SLAB), 0);
    debug_assert_eq!(max_y.rem_euclid(SLAB), 0);

    for gy in min_y.div_euclid(SLAB)..max_y.div_euclid(SLAB) {
        for gx in min_x.div_euclid(SLAB)..max_x.div_euclid(SLAB) {
            cells
                .entry((gx, gy))
                .and_modify(|existing| debug_assert_eq!(*existing, span))
                .or_insert(span);
        }
    }
}

fn build_corridor_cells(corridors: &[Corridor]) -> BTreeMap<Cell, CorridorCell> {
    let mut cells = BTreeMap::new();

    for corridor in corridors {
        let half = corridor.width as i32 / 2;
        let span = CorridorCell {
            floor_z: corridor.start.2,
            ceiling_bottom: corridor.start.2 + SLAB + corridor.height as i32,
        };
        match corridor_orientation(corridor) {
            Orientation::Horizontal => mark_open_rect(
                &mut cells,
                corridor.start.0.min(corridor.end.0),
                corridor.start.1 - half,
                corridor.start.0.max(corridor.end.0),
                corridor.start.1 + half,
                span,
            ),
            Orientation::Vertical => mark_open_rect(
                &mut cells,
                corridor.start.0 - half,
                corridor.start.1.min(corridor.end.1),
                corridor.start.0 + half,
                corridor.start.1.max(corridor.end.1),
                span,
            ),
        }

        // A full endpoint square preserves the complete 64×64 turning chamber
        // at L/T/X junctions instead of leaving a blocked inner quadrant.
        for endpoint in [corridor.start, corridor.end] {
            mark_open_rect(
                &mut cells,
                endpoint.0 - half,
                endpoint.1 - half,
                endpoint.0 + half,
                endpoint.1 + half,
                span,
            );
        }
    }

    cells
}

/// Build the complete room-owned XY footprint in slab-cell coordinates.
///
/// Every cell within a room's outer bounding box (including wall ring) is
/// owned by that room. Corridor slabs and boundary walls must not emit
/// geometry into these cells — `build_split_wall` is the sole owner of
/// portal columns and lintels.
fn build_room_owned_cells(layout: &LayoutIntent) -> BTreeSet<Cell> {
    let mut cells = BTreeSet::new();
    for room in &layout.rooms {
        let min_x = room.position.0;
        let min_y = room.position.1;
        let max_x = room.position.0 + room.dimensions.0 as i32;
        let max_y = room.position.1 + room.dimensions.1 as i32;
        for gy in min_y.div_euclid(SLAB)..max_y.div_euclid(SLAB) {
            for gx in min_x.div_euclid(SLAB)..max_x.div_euclid(SLAB) {
                cells.insert((gx, gy));
            }
        }
    }
    cells
}

fn build_corridor_slabs(
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
    room_owned_cells: &BTreeSet<Cell>,
) -> Vec<Brush> {
    let shell_cells: BTreeMap<Cell, CorridorCell> = corridor_cells
        .iter()
        .filter(|(cell, _)| !room_owned_cells.contains(cell))
        .map(|(&cell, &span)| (cell, span))
        .collect();

    let floor_cells: BTreeMap<Cell, i32> = shell_cells
        .iter()
        .map(|(&cell, span)| (cell, span.floor_z))
        .collect();
    let ceiling_cells: BTreeMap<Cell, i32> = shell_cells
        .iter()
        .map(|(&cell, span)| (cell, span.ceiling_bottom))
        .collect();

    let mut brushes = Vec::new();
    for rect in merge_cells(&floor_cells) {
        push_box(
            &mut brushes,
            (rect.x0 * SLAB, rect.y0 * SLAB, rect.value),
            (rect.x1 * SLAB, rect.y1 * SLAB, rect.value + SLAB),
            FLOOR_TEXTURE,
        );
    }
    for rect in merge_cells(&ceiling_cells) {
        push_box(
            &mut brushes,
            (rect.x0 * SLAB, rect.y0 * SLAB, rect.value),
            (rect.x1 * SLAB, rect.y1 * SLAB, rect.value + SLAB),
            CEILING_TEXTURE,
        );
    }
    brushes
}

fn build_corridor_boundary_walls(
    corridor_cells: &BTreeMap<Cell, CorridorCell>,
    room_owned_cells: &BTreeSet<Cell>,
) -> Vec<Brush> {
    let mut wall_cells: BTreeMap<Cell, WallSpan> = BTreeMap::new();

    for (&cell, &corridor_cell) in corridor_cells {
        let span = WallSpan {
            floor_z: corridor_cell.floor_z,
            shell_top: corridor_cell.ceiling_bottom + SLAB,
        };
        for neighbor in [
            (cell.0 - 1, cell.1),
            (cell.0 + 1, cell.1),
            (cell.0, cell.1 - 1),
            (cell.0, cell.1 + 1),
        ] {
            if corridor_cells.contains_key(&neighbor) || room_owned_cells.contains(&neighbor) {
                continue;
            }
            wall_cells
                .entry(neighbor)
                .and_modify(|existing| {
                    debug_assert_eq!(existing.floor_z, span.floor_z);
                    existing.shell_top = existing.shell_top.max(span.shell_top);
                })
                .or_insert(span);
        }
    }

    let mut brushes = Vec::new();
    for rect in merge_cells(&wall_cells) {
        push_box(
            &mut brushes,
            (rect.x0 * SLAB, rect.y0 * SLAB, rect.value.floor_z),
            (rect.x1 * SLAB, rect.y1 * SLAB, rect.value.shell_top),
            WALL_TEXTURE,
        );
    }
    brushes
}

// ── Grid rectangle merging ────────────────────────────────────────────────

fn merge_cells<T: Copy + Ord>(cells: &BTreeMap<Cell, T>) -> Vec<GridRect<T>> {
    let mut rows: BTreeMap<i32, Vec<(i32, T)>> = BTreeMap::new();
    for (&(x, y), &value) in cells {
        rows.entry(y).or_default().push((x, value));
    }

    let mut active: BTreeMap<(i32, i32, T), GridRect<T>> = BTreeMap::new();
    let mut finished = Vec::new();
    let mut previous_y = None;

    for (y, mut row) in rows {
        row.sort_unstable();
        if previous_y.is_some_and(|previous| y != previous + 1) {
            finished.extend(active.into_values());
            active = BTreeMap::new();
        }

        let mut runs = Vec::new();
        let mut index = 0;
        while index < row.len() {
            let (x0, value) = row[index];
            let mut x1 = x0 + 1;
            index += 1;
            while index < row.len() && row[index] == (x1, value) {
                x1 += 1;
                index += 1;
            }
            runs.push((x0, x1, value));
        }

        let mut next = BTreeMap::new();
        for (x0, x1, value) in runs {
            let key = (x0, x1, value);
            let rect = if let Some(mut rect) = active.remove(&key) {
                rect.y1 = y + 1;
                rect
            } else {
                GridRect {
                    x0,
                    x1,
                    y0: y,
                    y1: y + 1,
                    value,
                }
            };
            next.insert(key, rect);
        }
        finished.extend(active.into_values());
        active = next;
        previous_y = Some(y);
    }
    finished.extend(active.into_values());
    finished.sort_unstable();
    finished
}

fn push_box(brushes: &mut Vec<Brush>, min: (i32, i32, i32), max: (i32, i32, i32), texture: &str) {
    if min.0 < max.0 && min.1 < max.1 && min.2 < max.2 {
        brushes.push(junction::make_brush(min, max, texture));
    }
}

fn brush_bounds(brush: &Brush) -> ((i32, i32, i32), (i32, i32, i32)) {
    let mut min = (i32::MAX, i32::MAX, i32::MAX);
    let mut max = (i32::MIN, i32::MIN, i32::MIN);
    for face in &brush.faces {
        for &(x, y, z) in &face.plane_points {
            min = (min.0.min(x), min.1.min(y), min.2.min(z));
            max = (max.0.max(x), max.1.max(y), max.2.max(z));
        }
    }
    (min, max)
}

/// Reject positive-volume intersections between any two emitted source boxes.
/// Shared boundary planes remain legal; all three axes must overlap strictly.
fn validate_no_positive_volume_overlap(brushes: &[Brush]) -> Result<(), GeneratorError> {
    let bounds: Vec<_> = brushes.iter().map(brush_bounds).collect();
    for first in 0..bounds.len() {
        for second in first + 1..bounds.len() {
            let (a_min, a_max) = bounds[first];
            let (b_min, b_max) = bounds[second];
            let overlaps = a_min.0.max(b_min.0) < a_max.0.min(b_max.0)
                && a_min.1.max(b_min.1) < a_max.1.min(b_max.1)
                && a_min.2.max(b_min.2) < a_max.2.min(b_max.2);
            if overlaps {
                return Err(GeneratorError::InvariantViolation(format!(
                    "emitted source brushes {first} and {second} have a positive-volume overlap: {a_min:?}..{a_max:?} vs {b_min:?}..{b_max:?}",
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
        RoomIntent {
            position: (x, y, z),
            dimensions: (dx, dy, dz),
        }
    }

    fn corridor_h(x: i32, y: i32, z: i32, length: i32) -> Corridor {
        Corridor {
            start: (x, y, z),
            end: (x + length, y, z),
            width: 64,
            height: 80,
        }
    }

    fn bounds(brush: &Brush) -> ((i32, i32, i32), (i32, i32, i32)) {
        let mut min = (i32::MAX, i32::MAX, i32::MAX);
        let mut max = (i32::MIN, i32::MIN, i32::MIN);
        for face in &brush.faces {
            for &(x, y, z) in &face.plane_points {
                min = (min.0.min(x), min.1.min(y), min.2.min(z));
                max = (max.0.max(x), max.1.max(y), max.2.max(z));
            }
        }
        (min, max)
    }

    fn contains(brush: &Brush, point: (i32, i32, i32)) -> bool {
        let (min, max) = bounds(brush);
        point.0 > min.0
            && point.0 < max.0
            && point.1 > min.1
            && point.1 < max.1
            && point.2 > min.2
            && point.2 < max.2
    }

    #[test]
    fn isolated_room_emits_six_role_correct_brushes() {
        let room = room(0, 0, 0, 112, 112, 192);
        let brushes = build_room_brushes(&room, &[], &BTreeMap::new());
        assert_eq!(brushes.len(), 6);
        assert_eq!(bounds(&brushes[0]), ((0, 0, 0), (112, 112, 16)));
        assert_eq!(bounds(&brushes[1]), ((0, 0, 176), (112, 112, 192)));
        assert!(brushes[0]
            .faces
            .iter()
            .all(|face| face.texture == FLOOR_TEXTURE));
        assert!(brushes[1]
            .faces
            .iter()
            .all(|face| face.texture == CEILING_TEXTURE));
        assert!(brushes[2..]
            .iter()
            .all(|brush| brush.faces.iter().all(|face| face.texture == WALL_TEXTURE)));
    }

    #[test]
    fn portal_wall_mask_omits_only_the_aperture() {
        let room = room(0, 0, 0, 112, 112, 192);
        let corridor = corridor_h(112, 64, 0, 96);
        let corridor_cells = build_corridor_cells(std::slice::from_ref(&corridor));
        let brushes = build_room_brushes(&room, &[corridor], &corridor_cells);

        assert!(brushes.iter().all(|brush| !contains(brush, (104, 64, 40))));
        assert!(brushes.iter().any(|brush| contains(brush, (104, 8, 40))));
        assert!(brushes.iter().any(|brush| contains(brush, (104, 64, 120))));
        assert!(brushes.iter().any(|brush| contains(brush, (104, 64, 8))));
    }

    #[test]
    fn corridor_ceiling_stops_at_room_clear_interior() {
        let layout = LayoutIntent {
            rooms: vec![room(0, 0, 0, 112, 112, 192)],
            edges: Vec::new(),
            loop_count: 0,
        };
        let cells = build_corridor_cells(&[corridor_h(112, 64, 0, 96)]);
        let room_owned = build_room_owned_cells(&layout);
        let slabs = build_corridor_slabs(&cells, &room_owned);

        assert!(slabs
            .iter()
            .filter(|brush| brush.faces[0].texture == CEILING_TEXTURE)
            .all(|brush| bounds(brush).0 .0 >= 96));
    }

    #[test]
    fn l_turn_keeps_full_junction_square_clear() {
        let corridors = vec![
            corridor_h(0, 0, 0, 64),
            Corridor {
                start: (64, 0, 0),
                end: (64, 64, 0),
                width: 64,
                height: 80,
            },
        ];
        let cells = build_corridor_cells(&corridors);
        let walls = build_corridor_boundary_walls(&cells, &BTreeSet::new());
        for point in [(48, -16, 40), (64, 0, 40), (80, 16, 40)] {
            assert!(walls.iter().all(|brush| !contains(brush, point)));
        }
    }

    #[test]
    fn spawn_and_light_are_in_clear_volume() {
        let room = room(0, 0, 0, 112, 112, 192);
        assert_eq!(spawn_origin(&room), (56, 56, 40));
        assert_eq!(light_origin(&room), (56, 56, 96));
    }

    #[test]
    fn merged_rectangles_preserve_all_cells() {
        let cells = BTreeMap::from([((0, 0), 96), ((1, 0), 96), ((0, 1), 96), ((1, 1), 96)]);
        assert_eq!(
            merge_cells(&cells),
            vec![GridRect {
                x0: 0,
                x1: 2,
                y0: 0,
                y1: 2,
                value: 96,
            }]
        );
    }
}
