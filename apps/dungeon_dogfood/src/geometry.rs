use renderer::prelude::{MaterialHandle, ProceduralMeshData, ProceduralVertex};

use crate::collision::{ramp_height, CEILING_HEIGHT, CHUNK_SIZE, TILE_SIZE, WALL_HEIGHT};
use crate::layout::{tile_to_world, ParsedLevel, Tile};
use glam::{Vec2, Vec3, Vec4};

const MULTILAYER_SURFACE_GAP: f32 = 0.01;

pub struct ChunkBuild {
    pub name: String,
    pub mesh: ProceduralMeshData,
    pub world_origin: Vec3,
}

#[derive(Clone, Copy)]
pub enum SubLayer {
    Floor,
    Structure,
}

pub fn build_level_chunks(
    level: &ParsedLevel,
    floor_material: MaterialHandle,
    wall_material: MaterialHandle,
) -> Vec<ChunkBuild> {
    let mut out = Vec::new();

    for layer_idx in 0..level.layer_count() {
        emit_chunk(
            level,
            layer_idx,
            floor_material,
            SubLayer::Floor,
            0,
            0,
            level.width,
            level.height,
            &format!("floor_l{}", layer_idx),
            &mut out,
        );

        emit_chunk(
            level,
            layer_idx,
            wall_material,
            SubLayer::Structure,
            0,
            0,
            level.width,
            level.height,
            &format!("struct_l{}", layer_idx),
            &mut out,
        );
    }

    out
}

fn emit_chunk(
    level: &ParsedLevel,
    layer_idx: usize,
    material: renderer::MaterialHandle,
    sub_layer: SubLayer,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    base_name: &str,
    out: &mut Vec<ChunkBuild>,
) {
    let width = x1 - x0;
    let height = y1 - y0;

    if width <= CHUNK_SIZE && height <= CHUNK_SIZE {
        let mut verts = Vec::new();
        let mut inds = Vec::new();

        for y in y0..y1 {
            for x in x0..x1 {
                emit_tile(level, layer_idx, x, y, sub_layer, &mut verts, &mut inds);
            }
        }

        if verts.is_empty() {
            return;
        }

        let (bounds_min, bounds_max) = verts.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), vertex| (min.min(vertex.position), max.max(vertex.position)),
        );
        let world_origin = (bounds_min + bounds_max) * 0.5;
        for vertex in &mut verts {
            vertex.position -= world_origin;
        }

        out.push(ChunkBuild {
            name: base_name.to_string(),
            mesh: ProceduralMeshData {
                name: base_name.to_string(),
                vertices: verts,
                indices: inds,
                material: Some(material),
            },
            world_origin,
        });
    } else {
        // Subdivide
        if width > CHUNK_SIZE {
            let mid = x0 + width / 2;
            emit_chunk(
                level,
                layer_idx,
                material,
                sub_layer,
                x0,
                y0,
                mid,
                y1,
                &format!("{base_name}_a"),
                out,
            );
            emit_chunk(
                level,
                layer_idx,
                material,
                sub_layer,
                mid,
                y0,
                x1,
                y1,
                &format!("{base_name}_b"),
                out,
            );
            return;
        }

        if height > CHUNK_SIZE {
            let mid = y0 + height / 2;
            emit_chunk(
                level,
                layer_idx,
                material,
                sub_layer,
                x0,
                y0,
                x1,
                mid,
                &format!("{base_name}_a"),
                out,
            );
            emit_chunk(
                level,
                layer_idx,
                material,
                sub_layer,
                x0,
                mid,
                x1,
                y1,
                &format!("{base_name}_b"),
                out,
            );
            return;
        }
    }
}

fn emit_tile(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    sub_layer: SubLayer,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let tile = level.tile_at_3d(layer_idx, x, y);
    let y_offset = layer_idx as f32 * WALL_HEIGHT;

    match sub_layer {
        SubLayer::Floor => {
            if !is_walkable(tile) {
                return;
            }

            match tile {
                Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_) => {
                    emit_ramp_top(x, y, y_offset, tile, verts, inds);
                }
                _ => emit_floor(x, y, y_offset, verts, inds),
            }
        }
        SubLayer::Structure => {
            if is_walkable(tile) {
                if matches!(
                    tile,
                    Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
                ) {
                    emit_ramp_caps(level, layer_idx, x, y, y_offset, tile, verts, inds);
                } else {
                    emit_ceiling_for_tile(level, layer_idx, x, y, y_offset, verts, inds);
                }
            }

            if !is_solid(tile) {
                return;
            }

            // Walls
            if should_emit_north_face(level, layer_idx, x, y) {
                emit_wall_north(x, y, y_offset, verts, inds);
            }
            if should_emit_south_face(level, layer_idx, x, y) {
                emit_wall_south(x, y, y_offset, verts, inds);
            }
            if should_emit_east_face(level, layer_idx, x, y) {
                emit_wall_east(x, y, y_offset, verts, inds);
            }
            if should_emit_west_face(level, layer_idx, x, y) {
                emit_wall_west(x, y, y_offset, verts, inds);
            }

            // Top cap for walls
            emit_ceiling_for_tile(level, layer_idx, x, y, y_offset, verts, inds);
        }
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

fn neighbor_open(level: &ParsedLevel, layer_idx: usize, x: isize, y: isize) -> bool {
    if x < 0 || y < 0 || x >= level.width as isize || y >= level.height as isize {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, x as usize, y as usize))
}

fn should_emit_wall_face_toward(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
) -> bool {
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if !neighbor_open(level, layer_idx, nx, ny) {
        return false;
    }

    !upper_layer_shaft_face_hangs_over_lower_space(level, layer_idx, x, y, nx, ny)
}

fn upper_layer_shaft_face_hangs_over_lower_space(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    nx: isize,
    ny: isize,
) -> bool {
    if layer_idx == 0
        || nx < 0
        || ny < 0
        || nx >= level.width as isize
        || ny >= level.height as isize
    {
        return false;
    }

    if level.tile_at_3d(layer_idx, nx as usize, ny as usize) != Tile::Void {
        return false;
    }

    let lower_layer = layer_idx - 1;
    lower_tile_is_open_space(level.tile_at_3d(lower_layer, nx as usize, ny as usize))
        || lower_tile_is_open_space(level.tile_at_3d(lower_layer, x, y))
}

fn lower_tile_is_open_space(tile: Tile) -> bool {
    is_walkable(tile) || matches!(tile, Tile::Void)
}

fn should_emit_north_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    should_emit_wall_face_toward(level, layer_idx, x, y, 0, 1)
}
fn should_emit_south_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    should_emit_wall_face_toward(level, layer_idx, x, y, 0, -1)
}
fn should_emit_east_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    should_emit_wall_face_toward(level, layer_idx, x, y, 1, 0)
}
fn should_emit_west_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    should_emit_wall_face_toward(level, layer_idx, x, y, -1, 0)
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

    inds.push(b + 0);
    inds.push(b + 1);
    inds.push(b + 2);
    inds.push(b + 0);
    inds.push(b + 2);
    inds.push(b + 3);
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

fn emit_floor(
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

fn emit_ramp_top(
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

fn emit_ramp_caps(
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

    // West face
    if should_emit_ramp_cap(level, layer_idx, x, y, tile, y_offset, RampEdge::West) {
        let normal = -Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, corners.x0_z0, z0),
                normal,
                tangent,
                Vec2::new(1.0, corners.x0_z0 - y_offset),
            ),
            make_vertex(
                Vec3::new(x0, corners.x0_z1, z1),
                normal,
                tangent,
                Vec2::new(0.0, corners.x0_z1 - y_offset),
            ),
        );
    }

    // East face
    if should_emit_ramp_cap(level, layer_idx, x, y, tile, y_offset, RampEdge::East) {
        let normal = Vec3::X;
        let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z1, z1),
                normal,
                tangent,
                Vec2::new(1.0, corners.x1_z1 - y_offset),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z0, z0),
                normal,
                tangent,
                Vec2::new(0.0, corners.x1_z0 - y_offset),
            ),
        );
    }

    // Row +Y edge: world -Z.
    if should_emit_ramp_cap(
        level,
        layer_idx,
        x,
        y,
        tile,
        y_offset,
        RampEdge::RowPositive,
    ) {
        let normal = -Vec3::Z;
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x0, y_offset, z1),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, y_offset, z1),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z1, z1),
                normal,
                tangent,
                Vec2::new(1.0, corners.x1_z1 - y_offset),
            ),
            make_vertex(
                Vec3::new(x0, corners.x0_z1, z1),
                normal,
                tangent,
                Vec2::new(0.0, corners.x0_z1 - y_offset),
            ),
        );
    }

    // Row -Y edge: world +Z.
    if should_emit_ramp_cap(
        level,
        layer_idx,
        x,
        y,
        tile,
        y_offset,
        RampEdge::RowNegative,
    ) {
        let normal = Vec3::Z;
        let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(
                Vec3::new(x1, y_offset, z0),
                normal,
                tangent,
                Vec2::new(0.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, y_offset, z0),
                normal,
                tangent,
                Vec2::new(1.0, 0.0),
            ),
            make_vertex(
                Vec3::new(x0, corners.x0_z0, z0),
                normal,
                tangent,
                Vec2::new(1.0, corners.x0_z0 - y_offset),
            ),
            make_vertex(
                Vec3::new(x1, corners.x1_z0, z0),
                normal,
                tangent,
                Vec2::new(0.0, corners.x1_z0 - y_offset),
            ),
        );
    }
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

fn should_emit_ramp_cap(
    level: &ParsedLevel,
    layer_idx: usize,
    x: usize,
    y: usize,
    tile: Tile,
    y_offset: f32,
    edge: RampEdge,
) -> bool {
    let this_edge = surface_edge_heights(tile, y_offset, edge);
    if edge_at_base(this_edge, y_offset) {
        return false;
    }

    let (dx, dy) = edge.neighbor_offset();
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= level.width as isize || ny >= level.height as isize {
        return true;
    }

    let neighbor = level.tile_at_3d(layer_idx, nx as usize, ny as usize);
    if !is_walkable(neighbor) {
        return true;
    }

    let neighbor_edge = surface_edge_heights(neighbor, y_offset, edge.opposite());
    !height_pair_matches(this_edge, neighbor_edge)
}

fn surface_edge_heights(tile: Tile, y_offset: f32, edge: RampEdge) -> (f32, f32) {
    let corners = ramp_corner_heights(tile, y_offset);
    match edge {
        RampEdge::RowNegative => (corners.x0_z0, corners.x1_z0),
        RampEdge::RowPositive => (corners.x0_z1, corners.x1_z1),
        RampEdge::East => (corners.x1_z0, corners.x1_z1),
        RampEdge::West => (corners.x0_z0, corners.x0_z1),
    }
}

fn height_pair_matches(a: (f32, f32), b: (f32, f32)) -> bool {
    const EPSILON: f32 = 1e-4;
    (a.0 - b.0).abs() <= EPSILON && (a.1 - b.1).abs() <= EPSILON
}

fn edge_at_base(edge: (f32, f32), y_offset: f32) -> bool {
    const EPSILON: f32 = 1e-4;
    (edge.0 - y_offset).abs() <= EPSILON && (edge.1 - y_offset).abs() <= EPSILON
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

    fn emit_structure_tile(
        level: &ParsedLevel,
        layer_idx: usize,
        x: usize,
        y: usize,
    ) -> Vec<ProceduralVertex> {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        emit_tile(
            level,
            layer_idx,
            x,
            y,
            SubLayer::Structure,
            &mut verts,
            &mut inds,
        );
        verts
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

        // 2x2 wall block emits perimeter wall sides (8 quads) plus 4 top caps.
        // 12 quads * 6 indices = 72 indices.
        assert_eq!(mesh.indices.len(), 72);
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
        assert_eq!(verts.len(), 20);

        let row_positive = &verts[0..4];
        assert!(quad_all(row_positive, |v| {
            (v.position.z + 2.0).abs() <= 1e-4 && v.normal == -Vec3::Z
        }));
        assert_quad_winding_matches_normal(row_positive);

        let row_negative = &verts[4..8];
        assert!(quad_all(row_negative, |v| {
            (v.position.z + 1.0).abs() <= 1e-4 && v.normal == Vec3::Z
        }));
        assert_quad_winding_matches_normal(row_negative);

        let east = &verts[8..12];
        assert!(quad_all(east, |v| {
            (v.position.x - 2.0).abs() <= 1e-4 && v.normal == Vec3::X
        }));
        assert_quad_winding_matches_normal(east);

        let west = &verts[12..16];
        assert!(quad_all(west, |v| {
            (v.position.x - 1.0).abs() <= 1e-4 && v.normal == -Vec3::X
        }));
        assert_quad_winding_matches_normal(west);
    }

    #[test]
    fn ramp_top_corner_heights_match_collision_for_all_directions() {
        for tile in [
            Tile::RampNorth(0),
            Tile::RampEast(0),
            Tile::RampSouth(0),
            Tile::RampWest(0),
        ] {
            let mut verts = Vec::new();
            let mut inds = Vec::new();
            emit_ramp_top(2, 3, WALL_HEIGHT, tile, &mut verts, &mut inds);

            assert_eq!(verts.len(), 4);
            assert_eq!(inds.len(), 6);
            for vertex in &verts {
                assert!(vertex.normal.is_finite());
                assert!(vertex.normal.y > 0.0);

                let origin = tile_to_world(2, 3);
                let local_x = ((vertex.position.x - origin.x) / TILE_SIZE).clamp(0.0, 1.0);
                let local_z = ((origin.z - vertex.position.z) / TILE_SIZE).clamp(0.0, 1.0);
                let expected = ramp_height(tile, local_x, local_z, WALL_HEIGHT).unwrap();
                assert_close(vertex.position.y, expected);
            }

            for tri in inds.chunks_exact(3) {
                let p0 = verts[tri[0] as usize].position;
                let p1 = verts[tri[1] as usize].position;
                let p2 = verts[tri[2] as usize].position;
                assert!((p1 - p0).cross(p2 - p0).length() > 1e-4);
            }
        }
    }

    #[test]
    fn ramp_caps_suppress_compatible_shared_edge_and_preserve_exterior_caps() {
        let level = parsed_level(2, 1, vec![Tile::RampEast(0), Tile::RampEast(1)]);
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        emit_tile(&level, 0, 0, 0, SubLayer::Structure, &mut verts, &mut inds);
        emit_tile(&level, 0, 1, 0, SubLayer::Structure, &mut verts, &mut inds);

        let quads: Vec<&[ProceduralVertex]> = verts.chunks_exact(4).collect();
        assert!(!quads.iter().any(|quad| {
            quad_all(quad, |v| (v.position.x - 1.0).abs() <= 1e-4) && quad[0].normal.x.abs() > 0.999
        }));
        assert!(quads.iter().any(|quad| {
            quad_all(quad, |v| (v.position.x - 2.0).abs() <= 1e-4) && quad[0].normal == Vec3::X
        }));
        assert!(quads.iter().any(|quad| {
            quad_all(quad, |v| (v.position.z - 0.0).abs() <= 1e-4) && quad[0].normal == Vec3::Z
        }));
    }

    #[test]
    fn lower_layer_ceiling_offset_is_uniform_under_next_layer() {
        let level = parsed_level_layers(
            2,
            1,
            vec![
                vec![Tile::Floor, Tile::Floor],
                vec![Tile::Floor, Tile::Void],
            ],
        );

        let overlapped = emit_structure_tile(&level, 0, 0, 0);
        assert_eq!(overlapped.len(), 4);
        assert!(quad_all(&overlapped, |v| {
            (v.position.y - (CEILING_HEIGHT - MULTILAYER_SURFACE_GAP)).abs() <= 1e-4
                && v.normal == -Vec3::Y
        }));
        assert_quad_winding_matches_normal(&overlapped);

        let exposed = emit_structure_tile(&level, 0, 1, 0);
        assert_eq!(exposed.len(), 4);
        assert!(quad_all(&exposed, |v| {
            (v.position.y - (CEILING_HEIGHT - MULTILAYER_SURFACE_GAP)).abs() <= 1e-4
                && v.normal == -Vec3::Y
        }));
        assert_quad_winding_matches_normal(&exposed);
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
        assert!(
            !left_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 1.0).abs() <= 1e-4) && quad[0].normal == Vec3::X
            }),
            "upper wall east face should not hang over the lower ramp shaft"
        );
        assert!(
            left_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 0.0).abs() <= 1e-4) && quad[0].normal == -Vec3::X
            }),
            "ordinary outer wall face should remain"
        );

        let right_wall = emit_structure_tile(&level, 1, 2, 0);
        assert!(
            !right_wall.chunks_exact(4).any(|quad| {
                quad_all(quad, |v| (v.position.x - 2.0).abs() <= 1e-4) && quad[0].normal == -Vec3::X
            }),
            "upper wall west face should not hang over the lower ramp shaft"
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
}
