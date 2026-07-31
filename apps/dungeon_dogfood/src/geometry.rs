use std::collections::BTreeSet;

use renderer::prelude::{MaterialHandle, ProceduralMeshData, ProceduralVertex};

use crate::collision::{ramp_height, CEILING_HEIGHT, CHUNK_SIZE, TILE_SIZE, WALL_HEIGHT};
use crate::generator::ramps::scan_transitions;
use crate::layout::{tile_to_world, ParsedLevel, Tile};
use glam::{Vec2, Vec3, Vec4};

const MULTILAYER_SURFACE_GAP: f32 = 0.01;
const FLOOR_THICKNESS: f32 = 0.10;

pub struct ChunkBuild {
    pub name: String,
    pub mesh: ProceduralMeshData,
    pub world_origin: Vec3,
}

/// Material-independent geometry plan for an entire level.
pub(crate) struct ChunkGeometryPlan {
    pub leaves: Vec<LeafGeometry>,
}

/// Pre-materialized geometry for one chunk leaf, split by material domain.
pub(crate) struct LeafGeometry {
    pub leaf_name: String,
    pub wall_verts: Vec<ProceduralVertex>,
    pub wall_indices: Vec<u32>,
    pub floor_verts: Vec<ProceduralVertex>,
    pub floor_indices: Vec<u32>,
    /// Bounding box before centering (world-space).
    pub local_bounds: (Vec3, Vec3),
    /// Centroid of local_bounds; vertices are relative to this.
    pub world_origin: Vec3,
}

pub fn build_level_chunks(
    level: &ParsedLevel,
    floor_material: MaterialHandle,
    wall_material: MaterialHandle,
) -> Vec<ChunkBuild> {
    let plan = build_chunk_geometry_plan(level);
    let mut out = Vec::new();

    for leaf in &plan.leaves {
        debug_assert_eq!(
            leaf.world_origin,
            (leaf.local_bounds.0 + leaf.local_bounds.1) * 0.5
        );
        if !leaf.floor_verts.is_empty() {
            out.push(ChunkBuild {
                name: format!("floor_{}", leaf.leaf_name),
                mesh: ProceduralMeshData {
                    name: format!("floor_{}", leaf.leaf_name),
                    vertices: leaf.floor_verts.clone(),
                    indices: leaf.floor_indices.clone(),
                    material: Some(floor_material),
                },
                world_origin: leaf.world_origin,
            });
        }
        if !leaf.wall_verts.is_empty() {
            out.push(ChunkBuild {
                name: format!("struct_{}", leaf.leaf_name),
                mesh: ProceduralMeshData {
                    name: format!("struct_{}", leaf.leaf_name),
                    vertices: leaf.wall_verts.clone(),
                    indices: leaf.wall_indices.clone(),
                    material: Some(wall_material),
                },
                world_origin: leaf.world_origin,
            });
        }
    }
    out
}

/// Build a complete, material-independent geometry plan.
pub(crate) fn build_chunk_geometry_plan(level: &ParsedLevel) -> ChunkGeometryPlan {
    let suppressed_ceiling_cells = inferred_ramp_ceiling_openings(level);
    let mut leaves = Vec::new();

    for layer_idx in 0..level.layer_count() {
        plan_chunk_geometry(
            level,
            layer_idx,
            0,
            0,
            level.width,
            level.height,
            &format!("l{layer_idx}"),
            &suppressed_ceiling_cells,
            &mut leaves,
        );
    }

    ChunkGeometryPlan { leaves }
}

/// Return lower-layer cells whose ceilings are removed by complete shared ramp
/// inference. Generic upper-layer `Void` never enters this set.
fn inferred_ramp_ceiling_openings(level: &ParsedLevel) -> BTreeSet<(usize, usize, usize)> {
    let Ok(width) = u16::try_from(level.width) else {
        return BTreeSet::new();
    };
    let Ok(height) = u16::try_from(level.height) else {
        return BTreeSet::new();
    };
    let Ok(layers) = u16::try_from(level.layer_count()) else {
        return BTreeSet::new();
    };
    let lookup = |layer: u16, x: u16, y: u16| {
        Some(level.tile_at_3d(usize::from(layer), usize::from(x), usize::from(y)))
    };
    scan_transitions(width, height, layers, &lookup)
        .into_iter()
        .flat_map(|transition| transition.ramp_cells)
        .map(|(layer, x, y)| (usize::from(layer), usize::from(x), usize::from(y)))
        .collect()
}

fn plan_chunk_geometry(
    level: &ParsedLevel,
    layer_idx: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    base_name: &str,
    opening_cells: &BTreeSet<(usize, usize, usize)>,
    leaves: &mut Vec<LeafGeometry>,
) {
    let width = x1 - x0;
    let height = y1 - y0;

    if width <= CHUNK_SIZE && height <= CHUNK_SIZE {
        let mut wall_verts = Vec::new();
        let mut wall_indices = Vec::new();
        let mut floor_verts = Vec::new();
        let mut floor_indices = Vec::new();

        for y in y0..y1 {
            for x in x0..x1 {
                emit_tile_geometry(
                    level,
                    layer_idx,
                    x,
                    y,
                    opening_cells,
                    &mut wall_verts,
                    &mut wall_indices,
                    &mut floor_verts,
                    &mut floor_indices,
                );
            }
        }

        if wall_verts.is_empty() && floor_verts.is_empty() {
            return;
        }

        // Compute combined bounds from both domains.
        let all_positions = wall_verts
            .iter()
            .map(|v| v.position)
            .chain(floor_verts.iter().map(|v| v.position));
        let (bounds_min, bounds_max) = all_positions.fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), pos| (min.min(pos), max.max(pos)),
        );
        let world_origin = (bounds_min + bounds_max) * 0.5;

        // Center all vertices around the shared world origin.
        for v in &mut wall_verts {
            v.position -= world_origin;
        }
        for v in &mut floor_verts {
            v.position -= world_origin;
        }

        leaves.push(LeafGeometry {
            leaf_name: base_name.to_string(),
            wall_verts,
            wall_indices,
            floor_verts,
            floor_indices,
            local_bounds: (bounds_min, bounds_max),
            world_origin,
        });
    } else {
        // Subdivide
        if width > CHUNK_SIZE {
            let mid = x0 + width / 2;
            plan_chunk_geometry(
                level,
                layer_idx,
                x0,
                y0,
                mid,
                y1,
                &format!("{base_name}_a"),
                opening_cells,
                leaves,
            );
            plan_chunk_geometry(
                level,
                layer_idx,
                mid,
                y0,
                x1,
                y1,
                &format!("{base_name}_b"),
                opening_cells,
                leaves,
            );
            return;
        }

        if height > CHUNK_SIZE {
            let mid = y0 + height / 2;
            plan_chunk_geometry(
                level,
                layer_idx,
                x0,
                y0,
                x1,
                mid,
                &format!("{base_name}_a"),
                opening_cells,
                leaves,
            );
            plan_chunk_geometry(
                level,
                layer_idx,
                x0,
                mid,
                x1,
                y1,
                &format!("{base_name}_b"),
                opening_cells,
                leaves,
            );
            return;
        }
    }
}

fn emit_tile_geometry(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    opening_cells: &BTreeSet<(usize, usize, usize)>,
    wall_verts: &mut Vec<ProceduralVertex>,
    wall_indices: &mut Vec<u32>,
    floor_verts: &mut Vec<ProceduralVertex>,
    floor_indices: &mut Vec<u32>,
) {
    let tile = level.tile_at_3d(layer_idx, x, y);
    let y_offset = layer_idx as f32 * WALL_HEIGHT;

    // ── Floor domain: slabs and ramp wedges ──
    if is_walkable(tile) {
        match tile {
            Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_) => {
                emit_ramp_wedge(
                    level,
                    layer_idx,
                    x,
                    y,
                    y_offset,
                    tile,
                    floor_verts,
                    floor_indices,
                );
            }
            _ => emit_floor_slab(level, layer_idx, x, y, y_offset, floor_verts, floor_indices),
        }
    }

    // ── Wall domain: ceiling closures + wall voxels ──
    if is_walkable(tile) {
        let is_ramp = matches!(
            tile,
            Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
        );
        if !opening_cells.contains(&(layer_idx, x, y)) {
            if is_ramp {
                emit_ceiling_for_tile(level, layer_idx, x, y, y_offset, wall_verts, wall_indices);
            } else {
                emit_ceiling_closure(
                    level,
                    layer_idx,
                    x,
                    y,
                    y_offset,
                    opening_cells,
                    wall_verts,
                    wall_indices,
                );
            }
        }
    }

    if is_solid(tile) {
        emit_wall_voxel(level, layer_idx, x, y, y_offset, wall_verts, wall_indices);
    }
}

fn is_solid(tile: Tile) -> bool {
    matches!(tile, Tile::Wall)
}

fn is_walkable(tile: Tile) -> bool {
    matches!(
        tile,
        Tile::Floor
            | Tile::RampNorth(_)
            | Tile::RampEast(_)
            | Tile::RampSouth(_)
            | Tile::RampWest(_)
    )
}

fn has_layer_above(level: &ParsedLevel, layer_idx: usize) -> bool {
    layer_idx + 1 < level.layer_count()
}

fn emit_ceiling_for_tile(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let height = if has_layer_above(level, layer_idx) {
        y_offset + CEILING_HEIGHT - MULTILAYER_SURFACE_GAP
    } else {
        y_offset + CEILING_HEIGHT
    };
    emit_ceiling_at_height(x, y, height, verts, inds);
}

fn push_quad(
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
    v0: ProceduralVertex,
    v1: ProceduralVertex,
    v2: ProceduralVertex,
    v3: ProceduralVertex,
) {
    let b = verts.len() as u32;
    verts.push(v0);
    verts.push(v1);
    verts.push(v2);
    verts.push(v3);

    inds.push(b);
    inds.push(b + 1);
    inds.push(b + 2);
    inds.push(b);
    inds.push(b + 2);
    inds.push(b + 3);
}

fn push_triangle(
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
    v0: ProceduralVertex,
    v1: ProceduralVertex,
    v2: ProceduralVertex,
) {
    let b = verts.len() as u32;
    verts.extend([v0, v1, v2]);
    inds.extend([b, b + 1, b + 2]);
}

fn make_vertex(pos: Vec3, normal: Vec3, tangent: Vec4, tex: Vec2) -> ProceduralVertex {
    ProceduralVertex {
        position: pos,
        normal,
        tangent,
        uv0: tex,
        uv1: Vec2::ZERO,
        color: Vec4::ONE,
    }
}

// ─── Floor slab (0.10-thick box) ────────────────────────────────────────────

fn is_floor_side_exposed(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
) -> bool {
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= level.width as isize || ny >= level.height as isize {
        return true;
    }
    // Only another slab covers this face below y_offset. Ramp and wall
    // volumes begin at y_offset, so the slab edge beneath them remains exposed.
    level.tile_at_3d(layer_idx, nx as usize, ny as usize) != Tile::Floor
}

fn is_floor_bottom_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    if layer_idx == 0 {
        return true;
    }

    // Only a full-height wall below covers this horizontal face. Lower floors
    // and ramps do not: the upper slab underside is their layer's ceiling.
    !is_solid(level.tile_at_3d(layer_idx - 1, x, y))
}

fn emit_floor_slab(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let y_bottom = y_offset - FLOOR_THICKNESS;

    // Top face (walkable surface) — always visible.
    {
        let normal = Vec3::Y;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // Bottom face — visible when lower layer has no covering volume.
    if is_floor_bottom_exposed(level, layer_idx, x, y) {
        let normal = -Vec3::Y;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // North face (normal = -Z, z = z1).
    if is_floor_side_exposed(level, layer_idx, x, y, 0, 1) {
        let normal = -Vec3::Z;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // South face (normal = +Z, z = z0).
    if is_floor_side_exposed(level, layer_idx, x, y, 0, -1) {
        let normal = Vec3::Z;
        let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // East face (normal = +X, x = x1).
    if is_floor_side_exposed(level, layer_idx, x, y, 1, 0) {
        let normal = Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // West face (normal = -X, x = x0).
    if is_floor_side_exposed(level, layer_idx, x, y, -1, 0) {
        let normal = -Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }
}

// ─── Ceiling closure (thin box, wall material) ──────────────────────────────

fn upper_layer_covers(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    if !has_layer_above(level, layer_idx) {
        return false;
    }
    level.tile_at_3d(layer_idx + 1, x, y) != Tile::Void
}

fn cell_emits_ceiling_closure(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    opening_cells: &BTreeSet<(usize, usize, usize)>,
) -> bool {
    let tile = level.tile_at_3d(layer_idx, x, y);
    if !is_walkable(tile) {
        return false;
    }
    if matches!(
        tile,
        Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
    ) {
        return false;
    }
    if opening_cells.contains(&(layer_idx, x, y)) {
        return false;
    }
    !upper_layer_covers(level, layer_idx, x, y)
}

fn is_ceiling_closure_side_exposed(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
    opening_cells: &BTreeSet<(usize, usize, usize)>,
) -> bool {
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= level.width as isize || ny >= level.height as isize {
        return true;
    }
    !cell_emits_ceiling_closure(level, layer_idx, nx as usize, ny as usize, opening_cells)
}

fn emit_ceiling_closure(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    opening_cells: &BTreeSet<(usize, usize, usize)>,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    // Only emit when this cell actually qualifies.
    if !cell_emits_ceiling_closure(level, layer_idx, x, y, opening_cells) {
        return;
    }

    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let y_bottom = y_offset + CEILING_HEIGHT;
    let y_top = y_bottom + FLOOR_THICKNESS;

    // Bottom face (visible ceiling surface, normal = -Y).
    {
        let normal = -Vec3::Y;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // Top face (normal = +Y).
    {
        let normal = Vec3::Y;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_top, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_top, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // North face (normal = -Z, z = z1).
    if is_ceiling_closure_side_exposed(level, layer_idx, x, y, 0, 1, opening_cells) {
        let normal = -Vec3::Z;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_top, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // South face (normal = +Z, z = z0).
    if is_ceiling_closure_side_exposed(level, layer_idx, x, y, 0, -1, opening_cells) {
        let normal = Vec3::Z;
        let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z0),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_top, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // East face (normal = +X, x = x1).
    if is_ceiling_closure_side_exposed(level, layer_idx, x, y, 1, 0, opening_cells) {
        let normal = Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_top, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // West face (normal = -X, x = x0).
    if is_ceiling_closure_side_exposed(level, layer_idx, x, y, -1, 0, opening_cells) {
        let normal = -Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_bottom, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_bottom, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_top, z0),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, y_top, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }
}

fn emit_ceiling_at_height(
    x: usize,
    y: usize,
    ceiling_y: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let normal = -Vec3::Y;
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x0, ceiling_y, z0),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x0, ceiling_y, z1),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, ceiling_y, z1),
            normal,
            tangent,
            Vec2::new(1.0, 1.0),
        ),
        make_vertex(
            Vec3::new(x1, ceiling_y, z0),
            normal,
            tangent,
            Vec2::new(0.0, 1.0),
        ),
    );
}

fn emit_wall_north(
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z = origin.z - TILE_SIZE;

    let normal = -Vec3::Z;
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x1, y_offset, z),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x0, y_offset, z),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x0, y_offset + WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x1, y_offset + WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_south(
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z = origin.z;

    let normal = Vec3::Z;
    let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x0, y_offset, z),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, y_offset, z),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, y_offset + WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x0, y_offset + WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_east(
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let normal = Vec3::X;
    let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x, y_offset, z0),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x, y_offset, z1),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x, y_offset + WALL_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x, y_offset + WALL_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_west(
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x = origin.x;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let normal = -Vec3::X;
    let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x, y_offset, z1),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x, y_offset, z0),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x, y_offset + WALL_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x, y_offset + WALL_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
    );
}

// ─── Wall voxel (closed 6-face box) ─────────────────────────────────────────

fn emit_wall_voxel(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let y_top = y_offset + WALL_HEIGHT;

    if is_bottom_face_exposed(level, layer_idx, x, y) {
        emit_wall_bottom(x0, x1, z0, z1, y_offset, verts, inds);
    }
    if is_top_face_exposed(level, layer_idx, x, y) {
        emit_wall_top(x0, x1, z0, z1, y_top, verts, inds);
    }
    if is_north_face_exposed(level, layer_idx, x, y)
        && !emit_wall_face_above_neighbor_ramp(
            level,
            layer_idx,
            x,
            y,
            y_offset,
            RampEdge::RowPositive,
            verts,
            inds,
        )
    {
        emit_wall_north(x, y, y_offset, verts, inds);
    }
    if is_south_face_exposed(level, layer_idx, x, y)
        && !emit_wall_face_above_neighbor_ramp(
            level,
            layer_idx,
            x,
            y,
            y_offset,
            RampEdge::RowNegative,
            verts,
            inds,
        )
    {
        emit_wall_south(x, y, y_offset, verts, inds);
    }
    if is_east_face_exposed(level, layer_idx, x, y)
        && !emit_wall_face_above_neighbor_ramp(
            level,
            layer_idx,
            x,
            y,
            y_offset,
            RampEdge::East,
            verts,
            inds,
        )
    {
        emit_wall_east(x, y, y_offset, verts, inds);
    }
    if is_west_face_exposed(level, layer_idx, x, y)
        && !emit_wall_face_above_neighbor_ramp(
            level,
            layer_idx,
            x,
            y,
            y_offset,
            RampEdge::West,
            verts,
            inds,
        )
    {
        emit_wall_west(x, y, y_offset, verts, inds);
    }
}

fn is_bottom_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    if layer_idx == 0 {
        return true;
    }
    let below = layer_idx - 1;
    !is_solid(level.tile_at_3d(below, x, y))
}

fn is_top_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    if !has_layer_above(level, layer_idx) {
        return true;
    }
    let above = layer_idx + 1;
    !is_solid(level.tile_at_3d(above, x, y))
}

fn is_north_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    let ny = y as isize + 1;
    if ny >= level.height as isize {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, x, ny as usize))
}

fn is_south_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    let ny = y as isize - 1;
    if ny < 0 {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, x, ny as usize))
}

fn is_east_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    let nx = x as isize + 1;
    if nx >= level.width as isize {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, nx as usize, y))
}

fn is_west_face_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    let nx = x as isize - 1;
    if nx < 0 {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, nx as usize, y))
}

fn emit_wall_bottom(
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    y: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let normal = -Vec3::Y;
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x0, y, z0), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x0, y, z1), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(Vec3::new(x1, y, z1), normal, tangent, Vec2::new(1.0, 1.0)),
        make_vertex(Vec3::new(x1, y, z0), normal, tangent, Vec2::new(0.0, 1.0)),
    );
}

fn emit_wall_top(
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    y: f32,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let normal = Vec3::Y;
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x0, y, z0), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x1, y, z0), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(Vec3::new(x1, y, z1), normal, tangent, Vec2::new(1.0, 1.0)),
        make_vertex(Vec3::new(x0, y, z1), normal, tangent, Vec2::new(0.0, 1.0)),
    );
}

// ─── Ramp wedge (closed 3D sloped volume) ───────────────────────────────────

fn is_ramp_bottom_exposed(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    if layer_idx == 0 {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx - 1, x, y))
}

fn ramp_edge_cover_heights(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    edge: RampEdge,
) -> (f32, f32) {
    let (dx, dy) = edge.neighbor_offset();
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= level.width as isize || ny >= level.height as isize {
        return (y_offset, y_offset);
    }

    match level.tile_at_3d(layer_idx, nx as usize, ny as usize) {
        Tile::Wall => {
            let wall_top = y_offset + WALL_HEIGHT;
            (wall_top, wall_top)
        }
        neighbor @ (Tile::RampNorth(_)
        | Tile::RampEast(_)
        | Tile::RampSouth(_)
        | Tile::RampWest(_)) => surface_edge_heights(neighbor, y_offset, edge.opposite()),
        Tile::Floor | Tile::Void => (y_offset, y_offset),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_clipped_ramp_edge(
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    top_a: f32,
    top_b: f32,
    cover_a: f32,
    cover_b: f32,
    y_offset: f32,
    normal: Vec3,
    tangent: Vec4,
) {
    const EPSILON: f32 = 1e-4;
    let exposed_a = top_a - cover_a;
    let exposed_b = top_b - cover_b;
    let a_visible = exposed_a > EPSILON;
    let b_visible = exposed_b > EPSILON;

    if !a_visible && !b_visible {
        return;
    }

    let vertex = |position: Vec3, u: f32| {
        make_vertex(
            position,
            normal,
            tangent,
            Vec2::new(u, position.y - y_offset),
        )
    };
    let top_a_pos = Vec3::new(a.x, top_a, a.z);
    let top_b_pos = Vec3::new(b.x, top_b, b.z);
    let cover_a_pos = Vec3::new(a.x, cover_a, a.z);
    let cover_b_pos = Vec3::new(b.x, cover_b, b.z);

    match (a_visible, b_visible) {
        (true, true) => push_quad(
            verts,
            inds,
            vertex(cover_a_pos, 0.0),
            vertex(cover_b_pos, 1.0),
            vertex(top_b_pos, 1.0),
            vertex(top_a_pos, 0.0),
        ),
        (true, false) => {
            let t = exposed_a / (exposed_a - exposed_b);
            let intersection = top_a_pos.lerp(top_b_pos, t);
            push_triangle(
                verts,
                inds,
                vertex(cover_a_pos, 0.0),
                vertex(intersection, t),
                vertex(top_a_pos, 0.0),
            );
        }
        (false, true) => {
            let t = exposed_a / (exposed_a - exposed_b);
            let intersection = top_a_pos.lerp(top_b_pos, t);
            push_triangle(
                verts,
                inds,
                vertex(intersection, t),
                vertex(cover_b_pos, 1.0),
                vertex(top_b_pos, 1.0),
            );
        }
        (false, false) => unreachable!(),
    }
}

fn emit_wall_face_above_neighbor_ramp(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    edge: RampEdge,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) -> bool {
    let (dx, dy) = edge.neighbor_offset();
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= level.width as isize || ny >= level.height as isize {
        return false;
    }

    let neighbor = level.tile_at_3d(layer_idx, nx as usize, ny as usize);
    if !matches!(
        neighbor,
        Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
    ) {
        return false;
    }

    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let wall_top = y_offset + WALL_HEIGHT;
    let cover = surface_edge_heights(neighbor, y_offset, edge.opposite());

    let (a, b, cover_a, cover_b, normal, tangent) = match edge {
        RampEdge::RowPositive => (
            Vec3::new(x1, 0.0, z1),
            Vec3::new(x0, 0.0, z1),
            cover.1,
            cover.0,
            -Vec3::Z,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        ),
        RampEdge::RowNegative => (
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x1, 0.0, z0),
            cover.0,
            cover.1,
            Vec3::Z,
            Vec4::new(-1.0, 0.0, 0.0, 1.0),
        ),
        RampEdge::East => (
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x1, 0.0, z1),
            cover.0,
            cover.1,
            Vec3::X,
            Vec4::new(0.0, 0.0, -1.0, 1.0),
        ),
        RampEdge::West => (
            Vec3::new(x0, 0.0, z1),
            Vec3::new(x0, 0.0, z0),
            cover.1,
            cover.0,
            -Vec3::X,
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        ),
    };
    emit_clipped_ramp_edge(
        verts, inds, a, b, wall_top, wall_top, cover_a, cover_b, y_offset, normal, tangent,
    );
    true
}

fn emit_ramp_wedge(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    y_offset: f32,
    tile: Tile,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let corners = ramp_corner_heights(tile, y_offset);

    // ── Sloped top face ──
    {
        let p0 = Vec3::new(x0, corners.x0_z0, z0);
        let p1 = Vec3::new(x1, corners.x1_z0, z0);
        let p2 = Vec3::new(x1, corners.x1_z1, z1);
        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        if normal.y < 0.0 {
            normal = -normal;
        }
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, corners.x0_z0, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z0, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z1, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x0, corners.x0_z1, z1),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // ── Bottom face (planar at y_offset) ──
    if is_ramp_bottom_exposed(level, layer_idx, x, y) {
        let normal = -Vec3::Y;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 1.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 1.0),
            ),
        );
    }

    // Side and end faces are clipped against adjacent volume. This avoids both
    // shared interior faces and full quads hidden only in part by another ramp.
    let west_cover = ramp_edge_cover_heights(level, layer_idx, x, y, y_offset, RampEdge::West);
    emit_clipped_ramp_edge(
        verts,
        inds,
        Vec3::new(x0, 0.0, z1),
        Vec3::new(x0, 0.0, z0),
        corners.x0_z1,
        corners.x0_z0,
        west_cover.1,
        west_cover.0,
        y_offset,
        -Vec3::X,
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    );

    let east_cover = ramp_edge_cover_heights(level, layer_idx, x, y, y_offset, RampEdge::East);
    emit_clipped_ramp_edge(
        verts,
        inds,
        Vec3::new(x1, 0.0, z0),
        Vec3::new(x1, 0.0, z1),
        corners.x1_z0,
        corners.x1_z1,
        east_cover.0,
        east_cover.1,
        y_offset,
        Vec3::X,
        Vec4::new(0.0, 0.0, -1.0, 1.0),
    );

    let north_cover =
        ramp_edge_cover_heights(level, layer_idx, x, y, y_offset, RampEdge::RowPositive);
    emit_clipped_ramp_edge(
        verts,
        inds,
        Vec3::new(x1, 0.0, z1),
        Vec3::new(x0, 0.0, z1),
        corners.x1_z1,
        corners.x0_z1,
        north_cover.1,
        north_cover.0,
        y_offset,
        -Vec3::Z,
        Vec4::new(1.0, 0.0, 0.0, 1.0),
    );

    let south_cover =
        ramp_edge_cover_heights(level, layer_idx, x, y, y_offset, RampEdge::RowNegative);
    emit_clipped_ramp_edge(
        verts,
        inds,
        Vec3::new(x0, 0.0, z0),
        Vec3::new(x1, 0.0, z0),
        corners.x0_z0,
        corners.x1_z0,
        south_cover.0,
        south_cover.1,
        y_offset,
        Vec3::Z,
        Vec4::new(-1.0, 0.0, 0.0, 1.0),
    );
}

#[derive(Clone, Copy, Debug)]
struct RampCorners {
    x0_z0: f32,
    x1_z0: f32,
    x0_z1: f32,
    x1_z1: f32,
}

fn ramp_corner_heights(tile: Tile, y_offset: f32) -> RampCorners {
    RampCorners {
        x0_z0: ramp_height(tile, 0.0, 0.0, y_offset).unwrap_or(y_offset),
        x1_z0: ramp_height(tile, 1.0, 0.0, y_offset).unwrap_or(y_offset),
        x0_z1: ramp_height(tile, 0.0, 1.0, y_offset).unwrap_or(y_offset),
        x1_z1: ramp_height(tile, 1.0, 1.0, y_offset).unwrap_or(y_offset),
    }
}

#[derive(Clone, Copy, Debug)]
enum RampEdge {
    RowNegative,
    RowPositive,
    East,
    West,
}

impl RampEdge {
    fn neighbor_offset(self) -> (isize, isize) {
        match self {
            Self::RowNegative => (0, -1),
            Self::RowPositive => (0, 1),
            Self::East => (1, 0),
            Self::West => (-1, 0),
        }
    }

    fn opposite(self) -> Self {
        match self {
            Self::RowNegative => Self::RowPositive,
            Self::RowPositive => Self::RowNegative,
            Self::East => Self::West,
            Self::West => Self::East,
        }
    }
}

/// Returns the surface height pair for a tile along a given edge.
fn surface_edge_heights(tile: Tile, y_offset: f32, edge: RampEdge) -> (f32, f32) {
    let corners = ramp_corner_heights(tile, y_offset);
    match edge {
        RampEdge::RowNegative => (corners.x0_z0, corners.x1_z0),
        RampEdge::RowPositive => (corners.x0_z1, corners.x1_z1),
        RampEdge::East => (corners.x1_z0, corners.x1_z1),
        RampEdge::West => (corners.x0_z0, corners.x0_z1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_level(width: usize, height: usize, tiles: Vec<Tile>) -> ParsedLevel {
        parsed_level_layers(width, height, vec![tiles])
    }

    fn parsed_level_layers(width: usize, height: usize, layers: Vec<Vec<Tile>>) -> ParsedLevel {
        ParsedLevel {
            width,
            height,
            layers,
            spawn: crate::layout::TileCoord {
                layer: 0,
                x: 0,
                y: 0,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn fake_material() -> renderer::MaterialHandle {
        renderer::MaterialHandle::new(0, 0)
    }

    fn emit_floor_tile(
        level: &ParsedLevel,
        layer_idx: usize,
        x: usize,
        y: usize,
    ) -> Vec<ProceduralVertex> {
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut floor_verts = Vec::new();
        let mut floor_inds = Vec::new();
        emit_tile_geometry(
            level,
            layer_idx,
            x,
            y,
            &BTreeSet::new(),
            &mut wall_verts,
            &mut wall_inds,
            &mut floor_verts,
            &mut floor_inds,
        );
        floor_verts
    }

    fn emit_structure_tile(
        level: &ParsedLevel,
        layer_idx: usize,
        x: usize,
        y: usize,
    ) -> Vec<ProceduralVertex> {
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut floor_verts = Vec::new();
        let mut floor_inds = Vec::new();
        let empty_openings = BTreeSet::new();
        emit_tile_geometry(
            level,
            layer_idx,
            x,
            y,
            &empty_openings,
            &mut wall_verts,
            &mut wall_inds,
            &mut floor_verts,
            &mut floor_inds,
        );
        wall_verts
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_quad_winding_matches_normal(quad: &[ProceduralVertex]) {
        let tri_normal = (quad[1].position - quad[0].position)
            .cross(quad[2].position - quad[0].position)
            .normalize_or_zero();
        assert!(
            tri_normal.dot(quad[0].normal) > 0.999,
            "triangle normal {:?} does not match stored normal {:?}",
            tri_normal,
            quad[0].normal
        );
    }

    fn quad_all(quad: &[ProceduralVertex], predicate: impl Fn(&ProceduralVertex) -> bool) -> bool {
        quad.iter().all(predicate)
    }

    #[test]
    fn index_bounds_and_triangle_sanity() {
        let level = parsed_level(
            4,
            4,
            vec![
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Floor,
                Tile::Floor,
                Tile::Wall,
                Tile::Wall,
                Tile::Floor,
                Tile::RampEast(0),
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
            ],
        );

        let chunks = build_level_chunks(&level, fake_material(), fake_material());
        assert!(!chunks.is_empty());

        for chunk in chunks {
            assert_eq!(chunk.mesh.indices.len() % 3, 0);
            let v_count = chunk.mesh.vertices.len() as u32;
            for i in &chunk.mesh.indices {
                assert!(*i < v_count, "out-of-bounds index {i} >= {v_count}");
            }
        }
    }

    #[test]
    fn interior_wall_faces_are_culled() {
        let level = parsed_level(2, 2, vec![Tile::Wall, Tile::Wall, Tile::Wall, Tile::Wall]);
        let chunks = build_level_chunks(&level, fake_material(), fake_material());

        // We now have 2 chunks per layer (floor + struct)
        // Layer 0 structure chunk
        let struct_chunk = chunks
            .iter()
            .find(|c| c.name.starts_with("struct"))
            .unwrap();
        let mesh = &struct_chunk.mesh;

        // 2x2 wall block: 4 boxes * 4 exposed faces each (bottom+top always, 2 of 4 sides culled) = 16 faces.
        // 16 quads * 6 indices = 96 indices.
        assert_eq!(mesh.indices.len(), 96);
    }

    #[test]
    fn wall_faces_match_physical_edges_normals_and_winding() {
        let level = parsed_level(
            3,
            3,
            vec![
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Wall,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
            ],
        );

        let verts = emit_structure_tile(&level, 0, 1, 1);
        // Single isolated wall on layer 0: all 6 faces exposed (bottom, top, N, S, E, W).
        // 6 quads * 4 vertices = 24 vertices.
        assert_eq!(verts.len(), 24);

        // Bottom face (y=0, normal=-Y)
        let bottom: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| q[0].normal == -Vec3::Y)
            .collect();
        assert_eq!(bottom.len(), 1);
        assert_quad_winding_matches_normal(bottom[0]);

        // Top face (y=WALL_HEIGHT, normal=+Y)
        let top: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| q[0].normal == Vec3::Y)
            .collect();
        assert_eq!(top.len(), 1);
        assert_quad_winding_matches_normal(top[0]);

        // North face (z=-2, normal=-Z)
        let north: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| {
                quad_all(q, |v| (v.position.z + 2.0).abs() <= 1e-4) && q[0].normal == -Vec3::Z
            })
            .collect();
        assert_eq!(north.len(), 1);
        assert_quad_winding_matches_normal(north[0]);

        // South face (z=-1, normal=+Z)
        let south: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| {
                quad_all(q, |v| (v.position.z + 1.0).abs() <= 1e-4) && q[0].normal == Vec3::Z
            })
            .collect();
        assert_eq!(south.len(), 1);
        assert_quad_winding_matches_normal(south[0]);

        // East face (x=2, normal=+X)
        let east: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| {
                quad_all(q, |v| (v.position.x - 2.0).abs() <= 1e-4) && q[0].normal == Vec3::X
            })
            .collect();
        assert_eq!(east.len(), 1);
        assert_quad_winding_matches_normal(east[0]);

        // West face (x=1, normal=-X)
        let west: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| {
                quad_all(q, |v| (v.position.x - 1.0).abs() <= 1e-4) && q[0].normal == -Vec3::X
            })
            .collect();
        assert_eq!(west.len(), 1);
        assert_quad_winding_matches_normal(west[0]);
    }

    #[test]
    fn ramp_wedge_suppresses_compatible_shared_side_and_preserves_exterior() {
        let level = parsed_level(2, 1, vec![Tile::RampEast(0), Tile::RampEast(1)]);
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        let empty_openings = BTreeSet::new();
        for tile_coord in [(0, 0), (1, 0)] {
            emit_tile_geometry(
                &level,
                0,
                tile_coord.0,
                tile_coord.1,
                &empty_openings,
                &mut wall_verts,
                &mut wall_inds,
                &mut verts,
                &mut inds,
            );
        }

        assert!(!verts.iter().any(|vertex| {
            (vertex.position.x - 1.0).abs() <= 1e-4 && vertex.normal.x.abs() > 0.999
        }));
        assert!(verts
            .iter()
            .any(|vertex| { (vertex.position.x - 2.0).abs() <= 1e-4 && vertex.normal == Vec3::X }));
        assert!(verts
            .iter()
            .any(|vertex| { vertex.position.z.abs() <= 1e-4 && vertex.normal == Vec3::Z }));
        assert!(verts.iter().any(|vertex| vertex.normal == -Vec3::Y));
    }

    #[test]
    fn ramp_wedge_single_closed_each_direction() {
        for tile in [
            Tile::RampNorth(0),
            Tile::RampEast(0),
            Tile::RampSouth(0),
            Tile::RampWest(0),
            Tile::RampNorth(1),
            Tile::RampEast(1),
            Tile::RampSouth(1),
            Tile::RampWest(1),
        ] {
            let level = parsed_level(1, 1, vec![tile]);
            let mut wall_verts = Vec::new();
            let mut wall_inds = Vec::new();
            let mut verts = Vec::new();
            let mut inds = Vec::new();
            emit_tile_geometry(
                &level,
                0,
                0,
                0,
                &BTreeSet::new(),
                &mut wall_verts,
                &mut wall_inds,
                &mut verts,
                &mut inds,
            );

            assert_eq!(inds.len() % 3, 0, "indices must form complete triangles");
            assert!(!verts.is_empty(), "wedge must emit at least one face");

            assert!(
                verts.iter().any(|vertex| vertex.normal.y > 0.0),
                "missing sloped top face"
            );
            assert!(
                verts.iter().any(|vertex| vertex.normal == -Vec3::Y),
                "missing bottom face"
            );

            for tri in inds.chunks_exact(3) {
                let p0 = verts[tri[0] as usize].position;
                let p1 = verts[tri[1] as usize].position;
                let p2 = verts[tri[2] as usize].position;
                let face_normal = (p1 - p0).cross(p2 - p0);
                assert!(
                    face_normal.length() > 1e-6,
                    "wedge emitted a degenerate triangle for {tile:?}"
                );
                assert!(
                    face_normal.normalize().dot(verts[tri[0] as usize].normal) > 0.999,
                    "wedge triangle winding disagrees with its normal for {tile:?}: positions {p0:?}, {p1:?}, {p2:?}; computed {:?}, stored {:?}",
                    face_normal.normalize(),
                    verts[tri[0] as usize].normal
                );
            }
        }
    }

    #[test]
    fn ramp_wedge_clips_crossing_neighbor_profiles() {
        let level = parsed_level(2, 1, vec![Tile::RampSouth(0), Tile::RampNorth(0)]);
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        for x in 0..2 {
            emit_tile_geometry(
                &level,
                0,
                x,
                0,
                &BTreeSet::new(),
                &mut wall_verts,
                &mut wall_inds,
                &mut verts,
                &mut inds,
            );
        }

        let east_half = verts
            .iter()
            .filter(|vertex| (vertex.position.x - 1.0).abs() <= 1e-4 && vertex.normal == Vec3::X)
            .count();
        let west_half = verts
            .iter()
            .filter(|vertex| (vertex.position.x - 1.0).abs() <= 1e-4 && vertex.normal == -Vec3::X)
            .count();
        assert_eq!(
            east_half, 3,
            "left ramp must emit only its exposed triangle"
        );
        assert_eq!(
            west_half, 3,
            "right ramp must emit only its exposed triangle"
        );
    }

    #[test]
    fn ramp_wedge_face_is_hidden_by_adjacent_wall() {
        let level = parsed_level(2, 1, vec![Tile::RampEast(1), Tile::Wall]);
        let verts = emit_floor_tile(&level, 0, 0, 0);
        assert!(!verts
            .iter()
            .any(|vertex| { (vertex.position.x - 1.0).abs() <= 1e-4 && vertex.normal == Vec3::X }));
    }

    #[test]
    fn ramp_top_uv_uses_xz_projection() {
        let level = parsed_level(1, 1, vec![Tile::RampNorth(0)]);
        let verts = emit_floor_tile(&level, 0, 0, 0);
        for vertex in &verts[..4] {
            assert_close(vertex.uv0.x, vertex.position.x / TILE_SIZE);
            assert_close(vertex.uv0.y, -vertex.position.z / TILE_SIZE);
        }
    }

    #[test]
    fn ramp_structure_emits_ceiling_only() {
        let level = parsed_level(1, 1, vec![Tile::RampWest(0)]);
        let verts = emit_structure_tile(&level, 0, 0, 0);
        assert_eq!(verts.len(), 4);
        assert!(verts.iter().all(|vertex| vertex.normal == -Vec3::Y));
        assert!(verts
            .iter()
            .all(|vertex| (vertex.position.y - CEILING_HEIGHT).abs() <= 1e-4));
    }

    #[test]
    fn ramp_wedge_bottom_culled_by_wall_below() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Wall], vec![Tile::RampEast(0)]]);
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        emit_tile_geometry(
            &level,
            1,
            0,
            0,
            &BTreeSet::new(),
            &mut wall_verts,
            &mut wall_inds,
            &mut verts,
            &mut inds,
        );
        assert!(
            !verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::Y),
            "ramp bottom must be culled when Wall tile is below"
        );
    }

    #[test]
    fn ramp_wedge_bottom_exposed_when_void_below() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Void], vec![Tile::RampNorth(0)]]);
        let mut wall_verts = Vec::new();
        let mut wall_inds = Vec::new();
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        emit_tile_geometry(
            &level,
            1,
            0,
            0,
            &BTreeSet::new(),
            &mut wall_verts,
            &mut wall_inds,
            &mut verts,
            &mut inds,
        );
        assert!(
            verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::Y),
            "ramp bottom must be visible when Void is below"
        );
    }

    #[test]
    fn isolated_floor_emits_closed_slab_with_normalized_side_uvs() {
        let level = parsed_level(1, 1, vec![Tile::Floor]);
        let verts = emit_floor_tile(&level, 0, 0, 0);

        assert_eq!(verts.len(), 24, "isolated slab must emit all six faces");
        for normal in [Vec3::Y, -Vec3::Y, Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
            assert!(verts.chunks_exact(4).any(|quad| quad[0].normal == normal));
        }
        assert!(verts.iter().all(|vertex| {
            vertex.position.y >= -FLOOR_THICKNESS - 1e-4 && vertex.position.y <= 1e-4
        }));
        for vertex in verts.iter().filter(|vertex| vertex.normal.y.abs() < 0.5) {
            assert!((0.0..=1.0).contains(&vertex.uv0.x));
            assert!((0.0..=1.0).contains(&vertex.uv0.y));
        }
    }

    #[test]
    fn floor_slab_keeps_side_below_adjacent_ramp() {
        let level = parsed_level(2, 1, vec![Tile::Floor, Tile::RampEast(0)]);
        let floor = emit_floor_tile(&level, 0, 0, 0);
        let east_side = floor
            .chunks_exact(4)
            .find(|quad| quad[0].normal == Vec3::X)
            .expect("slab edge below ramp bottom must remain closed");
        assert!(east_side.iter().all(|vertex| {
            vertex.position.y >= -FLOOR_THICKNESS - 1e-4 && vertex.position.y <= 1e-4
        }));
    }

    #[test]
    fn adjacent_floor_slabs_cull_their_shared_side_faces() {
        let level = parsed_level(2, 1, vec![Tile::Floor, Tile::Floor]);
        let left = emit_floor_tile(&level, 0, 0, 0);
        let right = emit_floor_tile(&level, 0, 1, 0);

        assert_eq!(left.len(), 20);
        assert_eq!(right.len(), 20);
        assert!(!left.chunks_exact(4).any(|quad| quad[0].normal == Vec3::X));
        assert!(!right.chunks_exact(4).any(|quad| quad[0].normal == -Vec3::X));
    }

    #[test]
    fn upper_floor_slab_underside_closes_lower_walkable_layer() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Floor], vec![Tile::Floor]]);

        assert!(emit_structure_tile(&level, 0, 0, 0).is_empty());
        let upper_floor = emit_floor_tile(&level, 1, 0, 0);
        let underside = upper_floor
            .chunks_exact(4)
            .find(|quad| quad[0].normal == -Vec3::Y)
            .expect("upper slab underside must serve as the lower-layer ceiling");
        assert!(quad_all(underside, |vertex| {
            (vertex.position.y - (WALL_HEIGHT - FLOOR_THICKNESS)).abs() <= 1e-4
        }));
    }

    #[test]
    fn floor_slab_bottom_is_culled_only_by_full_wall_volume() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Wall], vec![Tile::Floor]]);
        let upper_floor = emit_floor_tile(&level, 1, 0, 0);
        assert!(!upper_floor
            .chunks_exact(4)
            .any(|quad| quad[0].normal == -Vec3::Y));
    }

    #[test]
    fn topmost_floor_gets_volumetric_ceiling_closure() {
        let level = parsed_level(1, 1, vec![Tile::Floor]);
        let ceiling = emit_structure_tile(&level, 0, 0, 0);

        assert_eq!(
            ceiling.len(),
            24,
            "isolated closure must emit all six faces"
        );
        let underside = ceiling
            .chunks_exact(4)
            .find(|quad| quad[0].normal == -Vec3::Y)
            .expect("closure must include a visible underside");
        assert!(quad_all(underside, |vertex| {
            (vertex.position.y - CEILING_HEIGHT).abs() <= 1e-4
        }));
    }

    #[test]
    fn wall_face_is_clipped_above_adjacent_ramp() {
        let level = parsed_level(2, 1, vec![Tile::Wall, Tile::RampEast(1)]);
        let wall = emit_structure_tile(&level, 0, 0, 0);
        let east_face: Vec<_> = wall
            .iter()
            .filter(|vertex| vertex.normal == Vec3::X)
            .collect();
        assert_eq!(east_face.len(), 4);
        let expected_bottom = ramp_height(Tile::RampEast(1), 0.0, 0.0, 0.0).unwrap();
        assert!(east_face
            .iter()
            .all(|vertex| vertex.position.y >= expected_bottom - 1e-4));
        assert!(east_face
            .iter()
            .any(|vertex| (vertex.position.y - expected_bottom).abs() <= 1e-4));
        assert!(!east_face
            .iter()
            .any(|vertex| vertex.position.y.abs() <= 1e-4));
    }

    #[test]
    fn wall_floor_junction_faces_only_meet_at_an_edge() {
        let level = parsed_level(2, 1, vec![Tile::Floor, Tile::Wall]);
        let floor = emit_floor_tile(&level, 0, 0, 0);
        let wall = emit_structure_tile(&level, 0, 1, 0);
        let floor_edge = floor
            .chunks_exact(4)
            .find(|quad| quad[0].normal == Vec3::X)
            .expect("floor side toward wall must remain closed");
        let wall_edge = wall
            .chunks_exact(4)
            .find(|quad| quad[0].normal == -Vec3::X)
            .expect("wall side toward floor must remain closed");
        let floor_max_y = floor_edge
            .iter()
            .map(|vertex| vertex.position.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let wall_min_y = wall_edge
            .iter()
            .map(|vertex| vertex.position.y)
            .fold(f32::INFINITY, f32::min);

        assert_close(floor_max_y, wall_min_y);
        assert_close(floor_max_y, 0.0);
    }

    #[test]
    fn lower_layer_ceiling_offset_is_uniform_under_next_layer() {
        // Tile (0,0): Floor on layer 0, Floor on layer 1 above → covered → no closure.
        // Tile (1,0): Floor on layer 0, Void on layer 1 above → closure emitted.
        let level = parsed_level_layers(
            2,
            1,
            vec![
                vec![Tile::Floor, Tile::Floor],
                vec![Tile::Floor, Tile::Void],
            ],
        );

        // Overlapped tile (0,0): no ceiling closure emitted.
        let overlapped = emit_structure_tile(&level, 0, 0, 0);
        assert!(
            overlapped.is_empty(),
            "ceiling suppressed when upper layer Floor covers"
        );

        // Exposed tile (1,0): closure emitted with bottom at CEILING_HEIGHT.
        let exposed = emit_structure_tile(&level, 0, 1, 0);
        assert!(!exposed.is_empty());
        // Bottom face must be present with normal = -Y at y = CEILING_HEIGHT.
        let bottom_quad: Vec<_> = exposed
            .chunks_exact(4)
            .filter(|q| q[0].normal == -Vec3::Y)
            .collect();
        assert_eq!(bottom_quad.len(), 1);
        assert!(quad_all(bottom_quad[0], |v| {
            (v.position.y - CEILING_HEIGHT).abs() <= 1e-4
        }));
        assert_quad_winding_matches_normal(bottom_quad[0]);
    }

    #[test]
    fn upper_layer_shaft_adjacent_wall_faces_over_lower_ramps_are_suppressed() {
        let level = parsed_level_layers(
            3,
            1,
            vec![
                vec![Tile::Floor, Tile::RampEast(1), Tile::Floor],
                vec![Tile::Wall, Tile::Void, Tile::Wall],
            ],
        );

        let left_wall = emit_structure_tile(&level, 1, 0, 0);
        // Wall boxes always emit faces toward same-layer Void (shaft walls are legitimate).
        assert!(
            left_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 1.0).abs() <= 1e-4) && quad[0].normal == Vec3::X
            }),
            "upper wall east face toward same-layer Void should emit"
        );
        assert!(
            left_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 0.0).abs() <= 1e-4) && quad[0].normal == -Vec3::X
            }),
            "ordinary outer wall face should remain"
        );

        let right_wall = emit_structure_tile(&level, 1, 2, 0);
        assert!(
            right_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 2.0).abs() <= 1e-4) && quad[0].normal == -Vec3::X
            }),
            "upper wall west face toward same-layer Void should emit"
        );
        assert!(
            right_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 3.0).abs() <= 1e-4) && quad[0].normal == Vec3::X
            }),
            "ordinary outer wall face should remain"
        );
    }

    #[test]
    fn upper_layer_void_boundary_walls_emit_when_no_lower_open_space_is_exposed() {
        let level = parsed_level_layers(
            2,
            1,
            vec![vec![Tile::Wall, Tile::Wall], vec![Tile::Wall, Tile::Void]],
        );

        let wall = emit_structure_tile(&level, 1, 0, 0);
        assert!(
            wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 1.0).abs() <= 1e-4) && quad[0].normal == Vec3::X
            }),
            "upper same-layer boundary wall should emit when it is not over lower open space"
        );
    }

    #[test]
    fn complete_ramp_inference_suppresses_exactly_run_ceilings() {
        let level = parsed_level_layers(
            7,
            1,
            vec![
                vec![
                    Tile::Wall,
                    Tile::Floor,
                    Tile::RampEast(0),
                    Tile::RampEast(1),
                    Tile::RampEast(2),
                    Tile::Wall,
                    Tile::Wall,
                ],
                vec![
                    Tile::Wall,
                    Tile::Wall,
                    Tile::Void,
                    Tile::Void,
                    Tile::Void,
                    Tile::Floor,
                    Tile::Wall,
                ],
            ],
        );
        let openings = inferred_ramp_ceiling_openings(&level);
        assert_eq!(openings, BTreeSet::from([(0, 2, 0), (0, 3, 0), (0, 4, 0)]));

        for x in 2..=4 {
            let mut floor_verts = Vec::new();
            let mut floor_inds = Vec::new();
            let mut verts = Vec::new();
            let mut inds = Vec::new();
            emit_tile_geometry(
                &level,
                0,
                x,
                0,
                &openings,
                &mut verts,
                &mut inds,
                &mut floor_verts,
                &mut floor_inds,
            );
            assert!(!verts.chunks_exact(4).any(|quad| quad[0].normal == -Vec3::Y));
        }
    }

    // ─── Wall voxel (box) tests ─────────────────────────────────────────

    #[test]
    fn isolated_wall_all_six_faces_exposed() {
        // Single wall on layer 0 with no neighbors: all 6 faces visible.
        let level = parsed_level(
            3,
            3,
            vec![
                Tile::Void,
                Tile::Void,
                Tile::Void,
                Tile::Void,
                Tile::Wall,
                Tile::Void,
                Tile::Void,
                Tile::Void,
                Tile::Void,
            ],
        );
        let verts = emit_structure_tile(&level, 0, 1, 1);
        // 6 faces * 4 vertices = 24 vertices.
        assert_eq!(verts.len(), 24);

        let normals: Vec<_> = verts.chunks_exact(4).map(|q| q[0].normal).collect();
        for expected in [Vec3::Y, -Vec3::Y, Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
            assert!(
                normals.iter().any(|&n| n.dot(expected) > 0.99),
                "missing normal {expected:?} in isolated wall"
            );
        }
    }

    #[test]
    fn wall_box_bottom_face_culled_by_wall_below() {
        // Layer 1 wall with Wall directly below: bottom face hidden.
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Wall], vec![Tile::Wall]]);
        let verts = emit_structure_tile(&level, 1, 0, 0);
        // 5 faces (no bottom): 20 vertices.
        assert_eq!(verts.len(), 20);
        assert!(
            !verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::Y),
            "bottom face must be culled when Wall tile is below"
        );
    }

    #[test]
    fn wall_box_bottom_face_exposed_when_floor_below() {
        // Layer 1 wall with Floor below: bottom face visible.
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Floor], vec![Tile::Wall]]);
        let verts = emit_structure_tile(&level, 1, 0, 0);
        assert!(
            verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::Y),
            "bottom face must be visible when Floor is below"
        );
    }

    #[test]
    fn wall_box_top_face_closes_full_height_under_non_wall_layer() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Wall], vec![Tile::Floor]]);
        let verts = emit_structure_tile(&level, 0, 0, 0);
        let top = verts
            .chunks_exact(4)
            .find(|quad| quad[0].normal == Vec3::Y)
            .expect("top face must be exposed below a non-wall layer");
        assert!(quad_all(top, |vertex| {
            (vertex.position.y - WALL_HEIGHT).abs() <= 1e-4
        }));
        assert_quad_winding_matches_normal(top);
    }

    #[test]
    fn wall_box_top_face_culled_by_wall_above() {
        // Wall with another Wall directly above: top face hidden.
        let level = parsed_level_layers(
            1,
            1,
            vec![vec![Tile::Wall], vec![Tile::Wall], vec![Tile::Wall]],
        );
        let verts = emit_structure_tile(&level, 1, 0, 0);
        // 4 lateral faces + bottom face (no top, no layer above is wall):
        // Bottom exposed (layer 1 below is Wall). Wait:
        // Layer 1 wall: below is Wall on layer 0 → bottom HIDDEN. Above is Wall on layer 2 → top HIDDEN.
        // So only 4 lateral faces = 16 vertices.
        assert_eq!(verts.len(), 16);
        assert!(
            !verts.chunks_exact(4).any(|q| q[0].normal == Vec3::Y),
            "top face must be culled when Wall tile is above"
        );
        assert!(
            !verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::Y),
            "bottom face must be culled when Wall tile is below"
        );
    }

    #[test]
    fn wall_box_side_face_culled_by_same_layer_wall() {
        // Two adjacent walls: shared face culled.
        let level = parsed_level(2, 1, vec![Tile::Wall, Tile::Wall]);
        // Wall at (0,0) east face hidden; Wall at (1,0) west face hidden.
        let left = emit_structure_tile(&level, 0, 0, 0);
        let right = emit_structure_tile(&level, 0, 1, 0);

        // Left wall: 5 faces (east culled) = 20 vertices.
        assert_eq!(left.len(), 20);
        assert!(
            !left.chunks_exact(4).any(|q| q[0].normal == Vec3::X),
            "east face of left wall must be culled by adjacent Wall"
        );

        // Right wall: 5 faces (west culled) = 20 vertices.
        assert_eq!(right.len(), 20);
        assert!(
            !right.chunks_exact(4).any(|q| q[0].normal == -Vec3::X),
            "west face of right wall must be culled by adjacent Wall"
        );
    }

    #[test]
    fn wall_box_face_toward_void_is_not_culled() {
        // Wall adjacent to Void on same layer: face stays visible.
        let level = parsed_level(2, 1, vec![Tile::Wall, Tile::Void]);
        let verts = emit_structure_tile(&level, 0, 0, 0);
        // East face toward Void must be present.
        assert!(
            verts.chunks_exact(4).any(|q| q[0].normal == Vec3::X),
            "east face toward same-layer Void must remain visible"
        );
    }

    #[test]
    fn wall_box_border_wall_outward_faces_at_edge() {
        // Wall at (0,0) in 3x3: west and south faces border level edge.
        let level = parsed_level(
            3,
            3,
            vec![
                Tile::Wall,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
            ],
        );
        let verts = emit_structure_tile(&level, 0, 0, 0);
        assert!(
            verts.chunks_exact(4).any(|q| q[0].normal == -Vec3::X),
            "west border face must be visible"
        );
        assert!(
            verts.chunks_exact(4).any(|q| q[0].normal == Vec3::Z),
            "south border face must be visible"
        );
    }

    #[test]
    fn wall_box_deterministic_same_output_same_input() {
        let level = parsed_level(3, 3, vec![Tile::Wall; 9]);
        let a = emit_structure_tile(&level, 0, 1, 1);
        let b = emit_structure_tile(&level, 0, 1, 1);
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va.position - vb.position).length() < 1e-6);
            assert_eq!(va.normal, vb.normal);
            assert!((va.uv0 - vb.uv0).length() < 1e-6);
        }
    }

    #[test]
    fn malformed_ramp_and_generic_void_preserve_ceilings() {
        let malformed = parsed_level_layers(
            6,
            1,
            vec![
                vec![
                    Tile::Floor,
                    Tile::RampEast(0),
                    Tile::RampEast(2),
                    Tile::RampEast(1),
                    Tile::Wall,
                    Tile::Floor,
                ],
                vec![
                    Tile::Void,
                    Tile::Void,
                    Tile::Void,
                    Tile::Void,
                    Tile::Floor,
                    Tile::Void,
                ],
            ],
        );
        assert!(inferred_ramp_ceiling_openings(&malformed).is_empty());
        let generic_void = parsed_level_layers(1, 1, vec![vec![Tile::Floor], vec![Tile::Void]]);
        assert!(inferred_ramp_ceiling_openings(&generic_void).is_empty());
        // Floor tile with Void above: ceiling closure emitted.
        let verts = emit_structure_tile(&generic_void, 0, 0, 0);
        assert!(
            verts.chunks_exact(4).any(|quad| quad[0].normal == -Vec3::Y),
            "ceiling closure must have downward-facing bottom face"
        );
        // Verify closure has side faces (not just a thin quad).
        let side_normals: Vec<_> = verts
            .chunks_exact(4)
            .filter(|q| q[0].normal != Vec3::Y && q[0].normal != -Vec3::Y)
            .map(|q| q[0].normal)
            .collect();
        assert!(
            !side_normals.is_empty(),
            "ceiling closure should have side faces"
        );
    }
}
