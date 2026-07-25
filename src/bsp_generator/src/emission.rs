//! Build the final brush/entity emission intent for a generated BSP dungeon.
//!
//! Quake brushes are additive, so overlapping a corridor with a solid room
//! wall cannot carve a portal. Emission instead rasterizes the complete clear
//! floor plan on the 16-unit construction grid, then emits floors, ceilings,
//! and only the boundary wall pieces around that union. Room portals and
//! L/T/X junction openings therefore exist by omission rather than by fake
//! subtraction brushes.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::CONSTRUCTION_QUANTUM;
use crate::intent::{
    Brush, Corridor, EmissionIntent, EntityIntent, LayoutIntent, RoomIntent, RoutedIntent,
};
use crate::junction;

const FLOOR_TEXTURE: &str = "stone_floor";
const CEILING_TEXTURE: &str = "stone_ceiling";
const WALL_TEXTURE: &str = "stone_wall";
const SLAB: i32 = CONSTRUCTION_QUANTUM as i32;
const EYE_OFFSET: i32 = 24;

type Cell = (i32, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GridRect {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    value: i32,
}

/// Build the complete worldspawn brush set and point entities.
pub fn build_emission(layout: &LayoutIntent, routed: &RoutedIntent) -> EmissionIntent {
    let open_cells = build_open_cells(layout, routed);
    let brushes = build_shell_brushes(&open_cells, layout);

    for (index, brush) in brushes.iter().enumerate() {
        debug_assert_eq!(brush.faces.len(), 6, "brush {index} is not a box");
    }

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

    EmissionIntent {
        brushes,
        entities,
        wad: "cc0_stone_beta.wad".to_string(),
    }
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

fn corridor_orientation(corridor: &Corridor) -> Orientation {
    if (corridor.end.0 - corridor.start.0).abs() >= (corridor.end.1 - corridor.start.1).abs() {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    }
}

fn mark_open_rect(
    cells: &mut BTreeMap<Cell, i32>,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    ceiling_bottom: i32,
) {
    debug_assert_eq!(min_x.rem_euclid(SLAB), 0);
    debug_assert_eq!(min_y.rem_euclid(SLAB), 0);
    debug_assert_eq!(max_x.rem_euclid(SLAB), 0);
    debug_assert_eq!(max_y.rem_euclid(SLAB), 0);

    for gy in min_y.div_euclid(SLAB)..max_y.div_euclid(SLAB) {
        for gx in min_x.div_euclid(SLAB)..max_x.div_euclid(SLAB) {
            cells
                .entry((gx, gy))
                .and_modify(|height| *height = (*height).max(ceiling_bottom))
                .or_insert(ceiling_bottom);
        }
    }
}

fn build_open_cells(layout: &LayoutIntent, routed: &RoutedIntent) -> BTreeMap<Cell, i32> {
    let mut cells = BTreeMap::new();

    // Corridors contribute their 64-unit clear rectangles. Endpoint squares
    // fill the full turn chamber, so short L/T/X segments cannot leave a solid
    // quadrant in the junction centre.
    for corridor in &routed.corridors {
        let half = corridor.width as i32 / 2;
        let ceiling_bottom = corridor.start.2 + SLAB + corridor.height as i32;
        match corridor_orientation(corridor) {
            Orientation::Horizontal => mark_open_rect(
                &mut cells,
                corridor.start.0.min(corridor.end.0),
                corridor.start.1 - half,
                corridor.start.0.max(corridor.end.0),
                corridor.start.1 + half,
                ceiling_bottom,
            ),
            Orientation::Vertical => mark_open_rect(
                &mut cells,
                corridor.start.0 - half,
                corridor.start.1.min(corridor.end.1),
                corridor.start.0 + half,
                corridor.start.1.max(corridor.end.1),
                ceiling_bottom,
            ),
        }
        for endpoint in [corridor.start, corridor.end] {
            mark_open_rect(
                &mut cells,
                endpoint.0 - half,
                endpoint.1 - half,
                endpoint.0 + half,
                endpoint.1 + half,
                ceiling_bottom,
            );
        }
    }

    // Room clear interiors override the lower corridor ceiling wherever a
    // route crosses a room. Wall cells remain absent unless a corridor portal
    // explicitly marks them open.
    for room in &layout.rooms {
        mark_open_rect(
            &mut cells,
            room.position.0 + SLAB,
            room.position.1 + SLAB,
            room.position.0 + room.dimensions.0 as i32 - SLAB,
            room.position.1 + room.dimensions.1 as i32 - SLAB,
            room.position.2 + room.dimensions.2 as i32 - SLAB,
        );
    }

    cells
}

fn build_shell_brushes(open_cells: &BTreeMap<Cell, i32>, layout: &LayoutIntent) -> Vec<Brush> {
    if open_cells.is_empty() {
        return Vec::new();
    }
    let floor_z = layout.rooms.first().map_or(0, |room| room.position.2);
    let mut brushes = Vec::new();

    let floor_cells: BTreeMap<Cell, i32> = open_cells
        .keys()
        .copied()
        .map(|cell| (cell, floor_z))
        .collect();
    for rect in merge_cells(&floor_cells) {
        push_box(
            &mut brushes,
            (rect.x0 * SLAB, rect.y0 * SLAB, floor_z),
            (rect.x1 * SLAB, rect.y1 * SLAB, floor_z + SLAB),
            FLOOR_TEXTURE,
        );
    }

    for rect in merge_cells(open_cells) {
        push_box(
            &mut brushes,
            (rect.x0 * SLAB, rect.y0 * SLAB, rect.value),
            (rect.x1 * SLAB, rect.y1 * SLAB, rect.value + SLAB),
            CEILING_TEXTURE,
        );
    }

    brushes.extend(build_boundary_walls(open_cells, floor_z));
    brushes.extend(build_height_transition_walls(open_cells));
    brushes
}

fn merge_cells(cells: &BTreeMap<Cell, i32>) -> Vec<GridRect> {
    let mut rows: BTreeMap<i32, Vec<(i32, i32)>> = BTreeMap::new();
    for (&(x, y), &value) in cells {
        rows.entry(y).or_default().push((x, value));
    }

    let mut active: BTreeMap<(i32, i32, i32), GridRect> = BTreeMap::new();
    let mut finished = Vec::new();
    let mut previous_y: Option<i32> = None;

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

fn build_boundary_walls(open_cells: &BTreeMap<Cell, i32>, floor_z: i32) -> Vec<Brush> {
    let mut vertical: BTreeMap<(i32, i32), BTreeSet<i32>> = BTreeMap::new();
    let mut horizontal: BTreeMap<(i32, i32), BTreeSet<i32>> = BTreeMap::new();

    for (&(x, y), &ceiling_bottom) in open_cells {
        let top = ceiling_bottom + SLAB;
        if !open_cells.contains_key(&(x - 1, y)) {
            vertical.entry((x - 1, top)).or_default().insert(y);
        }
        if !open_cells.contains_key(&(x + 1, y)) {
            vertical.entry((x + 1, top)).or_default().insert(y);
        }
        if !open_cells.contains_key(&(x, y - 1)) {
            horizontal.entry((y - 1, top)).or_default().insert(x);
        }
        if !open_cells.contains_key(&(x, y + 1)) {
            horizontal.entry((y + 1, top)).or_default().insert(x);
        }
    }

    let mut brushes = Vec::new();
    for ((wall_x, top), coordinates) in vertical {
        for (start, end) in contiguous_ranges(coordinates) {
            push_box(
                &mut brushes,
                (wall_x * SLAB, start * SLAB, floor_z),
                ((wall_x + 1) * SLAB, end * SLAB, top),
                WALL_TEXTURE,
            );
        }
    }
    for ((wall_y, top), coordinates) in horizontal {
        for (start, end) in contiguous_ranges(coordinates) {
            push_box(
                &mut brushes,
                (start * SLAB, wall_y * SLAB, floor_z),
                (end * SLAB, (wall_y + 1) * SLAB, top),
                WALL_TEXTURE,
            );
        }
    }
    brushes
}

fn build_height_transition_walls(open_cells: &BTreeMap<Cell, i32>) -> Vec<Brush> {
    let mut vertical: BTreeMap<(i32, i32, i32), BTreeSet<i32>> = BTreeMap::new();
    let mut horizontal: BTreeMap<(i32, i32, i32), BTreeSet<i32>> = BTreeMap::new();

    for (&(x, y), &high) in open_cells {
        for (neighbor, vertical_wall) in [
            ((x - 1, y), Some((x - 1, y))),
            ((x + 1, y), Some((x + 1, y))),
            ((x, y - 1), None),
            ((x, y + 1), None),
        ] {
            let Some(&low) = open_cells.get(&neighbor) else {
                continue;
            };
            if high <= low {
                continue;
            }
            let top = high + SLAB;
            if let Some((wall_x, coordinate)) = vertical_wall {
                vertical
                    .entry((wall_x, low, top))
                    .or_default()
                    .insert(coordinate);
            } else {
                horizontal
                    .entry((neighbor.1, low, top))
                    .or_default()
                    .insert(neighbor.0);
            }
        }
    }

    let mut brushes = Vec::new();
    for ((wall_x, z0, z1), coordinates) in vertical {
        for (start, end) in contiguous_ranges(coordinates) {
            push_box(
                &mut brushes,
                (wall_x * SLAB, start * SLAB, z0),
                ((wall_x + 1) * SLAB, end * SLAB, z1),
                WALL_TEXTURE,
            );
        }
    }
    for ((wall_y, z0, z1), coordinates) in horizontal {
        for (start, end) in contiguous_ranges(coordinates) {
            push_box(
                &mut brushes,
                (start * SLAB, wall_y * SLAB, z0),
                (end * SLAB, (wall_y + 1) * SLAB, z1),
                WALL_TEXTURE,
            );
        }
    }
    brushes
}

fn contiguous_ranges(values: BTreeSet<i32>) -> Vec<(i32, i32)> {
    let mut ranges = Vec::new();
    let mut iter = values.into_iter();
    let Some(mut start) = iter.next() else {
        return ranges;
    };
    let mut end = start + 1;
    for value in iter {
        if value == end {
            end += 1;
        } else {
            ranges.push((start, end));
            start = value;
            end = value + 1;
        }
    }
    ranges.push((start, end));
    ranges
}

fn push_box(brushes: &mut Vec<Brush>, min: (i32, i32, i32), max: (i32, i32, i32), texture: &str) {
    if min.0 < max.0 && min.1 < max.1 && min.2 < max.2 {
        brushes.push(junction::make_brush(min, max, texture));
    }
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

    fn point_in_open_cell(cells: &BTreeMap<Cell, i32>, point: (i32, i32)) -> bool {
        let candidates = [
            (point.0.div_euclid(SLAB), point.1.div_euclid(SLAB)),
            ((point.0 - 1).div_euclid(SLAB), point.1.div_euclid(SLAB)),
            (point.0.div_euclid(SLAB), (point.1 - 1).div_euclid(SLAB)),
            (
                (point.0 - 1).div_euclid(SLAB),
                (point.1 - 1).div_euclid(SLAB),
            ),
        ];
        candidates.iter().all(|cell| cells.contains_key(cell))
    }

    #[test]
    fn room_centre_is_open() {
        let layout = LayoutIntent {
            rooms: vec![room(0, 0, 0, 112, 112, 192)],
            edges: Vec::new(),
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: Vec::new(),
            junctions: Vec::new(),
        };
        let cells = build_open_cells(&layout, &routed);
        assert!(point_in_open_cell(&cells, (56, 56)));
    }

    #[test]
    fn corridor_portal_clears_room_wall() {
        let layout = LayoutIntent {
            rooms: vec![room(0, 0, 0, 112, 112, 192)],
            edges: vec![(0, 0)],
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: vec![corridor_h(112, 64, 0, 96)],
            junctions: Vec::new(),
        };
        let cells = build_open_cells(&layout, &routed);
        assert!(point_in_open_cell(&cells, (104, 64)));
        assert!(point_in_open_cell(&cells, (112, 64)));
    }

    #[test]
    fn l_junction_keeps_full_central_square_open() {
        let layout = LayoutIntent {
            rooms: Vec::new(),
            edges: Vec::new(),
            loop_count: 0,
        };
        let routed = RoutedIntent {
            corridors: vec![
                corridor_h(0, 0, 0, 64),
                Corridor {
                    start: (64, 0, 0),
                    end: (64, 64, 0),
                    width: 64,
                    height: 80,
                },
            ],
            junctions: Vec::new(),
        };
        let cells = build_open_cells(&layout, &routed);
        for point in [(48, -16), (64, 0), (80, 16)] {
            assert!(point_in_open_cell(&cells, point), "closed point {point:?}");
        }
    }

    #[test]
    fn spawn_and_light_are_above_floor_slab() {
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
