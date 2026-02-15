use glam::{Vec2, Vec3, Vec4};
use renderer::{MaterialHandle, ProceduralMeshData, ProceduralVertex};

use crate::collision::{CEILING_HEIGHT, CHUNK_SIZE, RAMP_RISE, TILE_SIZE, WALL_HEIGHT};
use crate::layout::{tile_to_world, ParsedLevel, Tile};

const MAX_CHUNK_VERTICES: usize = 65_535;

pub struct ChunkBuild {
    pub name: String,
    pub mesh: ProceduralMeshData,
    pub world_origin: Vec3,
}

pub fn build_level_chunks(level: &ParsedLevel, material: MaterialHandle) -> Vec<ChunkBuild> {
    let chunk_cols = level.width.div_ceil(CHUNK_SIZE);
    let chunk_rows = level.height.div_ceil(CHUNK_SIZE);

    let mut out = Vec::new();

    for chunk_y in 0..chunk_rows {
        for chunk_x in 0..chunk_cols {
            let x0 = chunk_x * CHUNK_SIZE;
            let y0 = chunk_y * CHUNK_SIZE;
            let x1 = (x0 + CHUNK_SIZE).min(level.width);
            let y1 = (y0 + CHUNK_SIZE).min(level.height);
            let base_name = format!("dungeon_chunk_{chunk_x}_{chunk_y}");

            emit_chunk(level, material, x0, y0, x1, y1, &base_name, &mut out);
        }
    }

    out
}

fn emit_chunk(
    level: &ParsedLevel,
    material: MaterialHandle,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    base_name: &str,
    out: &mut Vec<ChunkBuild>,
) {
    let mut verts = Vec::new();
    let mut inds = Vec::new();

    for y in y0..y1 {
        for x in x0..x1 {
            emit_tile(level, x, y, &mut verts, &mut inds);
        }
    }

    if verts.is_empty() {
        return;
    }

    if verts.len() > MAX_CHUNK_VERTICES {
        let width = x1 - x0;
        let height = y1 - y0;

        if width >= height && width > 1 {
            let mid = x0 + width / 2;
            emit_chunk(level, material, x0, y0, mid, y1, &format!("{base_name}_a"), out);
            emit_chunk(level, material, mid, y0, x1, y1, &format!("{base_name}_b"), out);
            return;
        }

        if height > 1 {
            let mid = y0 + height / 2;
            emit_chunk(level, material, x0, y0, x1, mid, &format!("{base_name}_a"), out);
            emit_chunk(level, material, x0, mid, x1, y1, &format!("{base_name}_b"), out);
            return;
        }
    }

    out.push(ChunkBuild {
        name: base_name.to_string(),
        mesh: ProceduralMeshData {
            name: base_name.to_string(),
            vertices: verts,
            indices: inds,
            material: Some(material),
        },
        world_origin: tile_to_world(x0, y0),
    });
}

fn emit_tile(
    level: &ParsedLevel,
    x: usize,
    y: usize,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let tile = level.tile_at(x, y);

    if is_walkable(tile) {
        match tile {
            Tile::RampNorth | Tile::RampEast | Tile::RampSouth | Tile::RampWest => {
                emit_ramp_top(x, y, tile, verts, inds);
                emit_ramp_caps(x, y, tile, verts, inds);
            }
            _ => emit_floor(x, y, verts, inds),
        }

        emit_ceiling(x, y, verts, inds);
    }

    if !is_solid(tile) {
        return;
    }

    if should_emit_north_face(level, x, y) {
        emit_wall_north(x, y, verts, inds);
    }
    if should_emit_south_face(level, x, y) {
        emit_wall_south(x, y, verts, inds);
    }
    if should_emit_east_face(level, x, y) {
        emit_wall_east(x, y, verts, inds);
    }
    if should_emit_west_face(level, x, y) {
        emit_wall_west(x, y, verts, inds);
    }
}

fn is_solid(tile: Tile) -> bool {
    matches!(tile, Tile::Wall)
}

fn is_walkable(tile: Tile) -> bool {
    !is_solid(tile)
}

fn neighbor_walkable(level: &ParsedLevel, x: isize, y: isize) -> bool {
    if x < 0 || y < 0 || x >= level.width as isize || y >= level.height as isize {
        return true;
    }
    is_walkable(level.tile_at(x as usize, y as usize))
}

fn should_emit_north_face(level: &ParsedLevel, x: usize, y: usize) -> bool {
    neighbor_walkable(level, x as isize, y as isize - 1)
}

fn should_emit_south_face(level: &ParsedLevel, x: usize, y: usize) -> bool {
    neighbor_walkable(level, x as isize, y as isize + 1)
}

fn should_emit_east_face(level: &ParsedLevel, x: usize, y: usize) -> bool {
    neighbor_walkable(level, x as isize + 1, y as isize)
}

fn should_emit_west_face(level: &ParsedLevel, x: usize, y: usize) -> bool {
    neighbor_walkable(level, x as isize - 1, y as isize)
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
    verts.extend_from_slice(&[v0, v1, v2, v3]);
    inds.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 1, b + 3]);
}

fn make_vertex(position: Vec3, normal: Vec3, tangent: Vec4, uv0: Vec2) -> ProceduralVertex {
    ProceduralVertex {
        position,
        normal,
        tangent,
        uv0,
        uv1: Vec2::ZERO,
        color: Vec4::ONE,
    }
}

fn emit_floor(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
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
        make_vertex(Vec3::new(x0, 0.0, z0), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x1, 0.0, z0), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(Vec3::new(x0, 0.0, z1), normal, tangent, Vec2::new(0.0, 1.0)),
        make_vertex(Vec3::new(x1, 0.0, z1), normal, tangent, Vec2::new(1.0, 1.0)),
    );
}

fn emit_ceiling(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
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
            Vec3::new(x0, CEILING_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x0, CEILING_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(0.0, 1.0),
        ),
        make_vertex(
            Vec3::new(x1, CEILING_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, CEILING_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(1.0, 1.0),
        ),
    );
}

fn emit_wall_north(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z = origin.z;
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x0, 0.0, z), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x1, 0.0, z), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(
            Vec3::new(x0, WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x1, WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_south(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z = origin.z - TILE_SIZE;
    let normal = Vec3::new(0.0, 0.0, -1.0);
    let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x1, 0.0, z), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x0, 0.0, z), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(
            Vec3::new(x1, WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x0, WALL_HEIGHT, z),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_east(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
    let origin = tile_to_world(x, y);
    let x = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let normal = Vec3::new(1.0, 0.0, 0.0);
    let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x, 0.0, z0), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x, 0.0, z1), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(
            Vec3::new(x, WALL_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x, WALL_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
    );
}

fn emit_wall_west(x: usize, y: usize, verts: &mut Vec<ProceduralVertex>, inds: &mut Vec<u32>) {
    let origin = tile_to_world(x, y);
    let x = origin.x;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;
    let normal = Vec3::new(-1.0, 0.0, 0.0);
    let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(Vec3::new(x, 0.0, z1), normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(Vec3::new(x, 0.0, z0), normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(
            Vec3::new(x, WALL_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(0.0, WALL_HEIGHT),
        ),
        make_vertex(
            Vec3::new(x, WALL_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(1.0, WALL_HEIGHT),
        ),
    );
}

fn emit_ramp_top(
    x: usize,
    y: usize,
    tile: Tile,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let (h_nw, h_ne, h_sw, h_se) = ramp_corner_heights(tile);

    let nw = Vec3::new(x0, h_nw, z0);
    let ne = Vec3::new(x1, h_ne, z0);
    let sw = Vec3::new(x0, h_sw, z1);
    let se = Vec3::new(x1, h_se, z1);

    let mut normal = (ne - nw).cross(sw - nw).normalize_or_zero();
    if normal.y < 0.0 {
        normal = -normal;
    }
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(nw, normal, tangent, Vec2::new(0.0, 0.0)),
        make_vertex(ne, normal, tangent, Vec2::new(1.0, 0.0)),
        make_vertex(sw, normal, tangent, Vec2::new(0.0, 1.0)),
        make_vertex(se, normal, tangent, Vec2::new(1.0, 1.0)),
    );
}

fn emit_ramp_caps(
    x: usize,
    y: usize,
    tile: Tile,
    verts: &mut Vec<ProceduralVertex>,
    inds: &mut Vec<u32>,
) {
    let origin = tile_to_world(x, y);
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    let (h_nw, h_ne, h_sw, h_se) = ramp_corner_heights(tile);

    if h_nw != h_sw {
        let normal = Vec3::new(-1.0, 0.0, 0.0);
        let tangent = Vec4::new(0.0, 0.0, 1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(Vec3::new(x0, 0.0, z1), normal, tangent, Vec2::new(0.0, 0.0)),
            make_vertex(Vec3::new(x0, 0.0, z0), normal, tangent, Vec2::new(1.0, 0.0)),
            make_vertex(Vec3::new(x0, h_sw, z1), normal, tangent, Vec2::new(0.0, h_sw)),
            make_vertex(Vec3::new(x0, h_nw, z0), normal, tangent, Vec2::new(1.0, h_nw)),
        );
    }

    if h_ne != h_se {
        let normal = Vec3::new(1.0, 0.0, 0.0);
        let tangent = Vec4::new(0.0, 0.0, -1.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(Vec3::new(x1, 0.0, z0), normal, tangent, Vec2::new(0.0, 0.0)),
            make_vertex(Vec3::new(x1, 0.0, z1), normal, tangent, Vec2::new(1.0, 0.0)),
            make_vertex(Vec3::new(x1, h_ne, z0), normal, tangent, Vec2::new(0.0, h_ne)),
            make_vertex(Vec3::new(x1, h_se, z1), normal, tangent, Vec2::new(1.0, h_se)),
        );
    }

    if h_nw != h_ne {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(Vec3::new(x0, 0.0, z0), normal, tangent, Vec2::new(0.0, 0.0)),
            make_vertex(Vec3::new(x1, 0.0, z0), normal, tangent, Vec2::new(1.0, 0.0)),
            make_vertex(Vec3::new(x0, h_nw, z0), normal, tangent, Vec2::new(0.0, h_nw)),
            make_vertex(Vec3::new(x1, h_ne, z0), normal, tangent, Vec2::new(1.0, h_ne)),
        );
    }

    if h_sw != h_se {
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let tangent = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        push_quad(
            verts,
            inds,
            make_vertex(Vec3::new(x1, 0.0, z1), normal, tangent, Vec2::new(0.0, 0.0)),
            make_vertex(Vec3::new(x0, 0.0, z1), normal, tangent, Vec2::new(1.0, 0.0)),
            make_vertex(Vec3::new(x1, h_se, z1), normal, tangent, Vec2::new(0.0, h_se)),
            make_vertex(Vec3::new(x0, h_sw, z1), normal, tangent, Vec2::new(1.0, h_sw)),
        );
    }
}

fn ramp_corner_heights(tile: Tile) -> (f32, f32, f32, f32) {
    match tile {
        Tile::RampNorth => (RAMP_RISE, RAMP_RISE, 0.0, 0.0),
        Tile::RampSouth => (0.0, 0.0, RAMP_RISE, RAMP_RISE),
        Tile::RampEast => (0.0, RAMP_RISE, 0.0, RAMP_RISE),
        Tile::RampWest => (RAMP_RISE, 0.0, RAMP_RISE, 0.0),
        _ => (0.0, 0.0, 0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_level(width: usize, height: usize, tiles: Vec<Tile>) -> ParsedLevel {
        ParsedLevel {
            width,
            height,
            tiles,
            spawn: (0, 0),
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn fake_material() -> MaterialHandle {
        MaterialHandle {
            slot: 999,
            generation: 1,
        }
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
                Tile::RampEast,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
            ],
        );

        let chunks = build_level_chunks(&level, fake_material());
        assert!(!chunks.is_empty());

        for chunk in chunks {
            assert_eq!(chunk.mesh.indices.len() % 3, 0);
            let v_count = chunk.mesh.vertices.len() as u32;
            for i in chunk.mesh.indices {
                assert!(i < v_count, "out-of-bounds index {i} >= {v_count}");
            }
        }
    }

    #[test]
    fn interior_wall_faces_are_culled() {
        let level = parsed_level(2, 2, vec![Tile::Wall, Tile::Wall, Tile::Wall, Tile::Wall]);
        let chunks = build_level_chunks(&level, fake_material());
        assert_eq!(chunks.len(), 1);

        let mesh = &chunks[0].mesh;
        // 2x2 wall block emits only perimeter faces: 8 quads => 16 triangles => 48 indices.
        assert_eq!(mesh.indices.len(), 48);
    }

    #[test]
    fn ramp_orientation_geometry_correctness() {
        let (nw, ne, sw, se) = ramp_corner_heights(Tile::RampNorth);
        assert!((nw - RAMP_RISE).abs() < 1e-4);
        assert!((ne - RAMP_RISE).abs() < 1e-4);
        assert!(sw.abs() < 1e-4);
        assert!(se.abs() < 1e-4);
    }
}
