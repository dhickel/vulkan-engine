use renderer::{MaterialHandle, ProceduralMeshData, ProceduralVertex};

use crate::collision::{CEILING_HEIGHT, CHUNK_SIZE, RAMP_RISE, TILE_SIZE, WALL_HEIGHT};
use crate::layout::{tile_to_world, ParsedLevel, Tile};
use glam::{Vec2, Vec3, Vec4};

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

        out.push(ChunkBuild {
            name: base_name.to_string(),
            mesh: ProceduralMeshData {
                name: base_name.to_string(),
                vertices: verts,
                indices: inds,
                material: Some(material),
            },
            world_origin: tile_to_world(x0, y0)
                + Vec3::new(0.0, layer_idx as f32 * WALL_HEIGHT, 0.0),
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
                    emit_ramp_caps(x, y, y_offset, tile, verts, inds);
                } else {
                    let opens_above = if layer_idx + 1 < level.layer_count() {
                        tile_opens_ceiling(level.tile_at_3d(layer_idx + 1, x, y))
                    } else {
                        false
                    };

                    if !opens_above {
                        emit_ceiling(x, y, y_offset, verts, inds);
                    }
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
            emit_ceiling(x, y, y_offset, verts, inds);
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

fn tile_opens_ceiling(tile: Tile) -> bool {
    matches!(
        tile,
        Tile::Void
            | Tile::RampNorth(_)
            | Tile::RampEast(_)
            | Tile::RampSouth(_)
            | Tile::RampWest(_)
    )
}

fn neighbor_open(level: &ParsedLevel, layer_idx: usize, x: isize, y: isize) -> bool {
    if x < 0 || y < 0 || x >= level.width as isize || y >= level.height as isize {
        return true;
    }
    !is_solid(level.tile_at_3d(layer_idx, x as usize, y as usize))
}

fn should_emit_north_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    neighbor_open(level, layer_idx, x as isize, y as isize - 1)
}
fn should_emit_south_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    neighbor_open(level, layer_idx, x as isize, y as isize + 1)
}
fn should_emit_east_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    neighbor_open(level, layer_idx, x as isize + 1, y as isize)
}
fn should_emit_west_face(level: &ParsedLevel, layer_idx: usize, x: usize, y: usize) -> bool {
    neighbor_open(level, layer_idx, x as isize - 1, y as isize)
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

fn emit_ceiling(
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

    let normal = -Vec3::Y;
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x0, y_offset + CEILING_HEIGHT, z0),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x0, y_offset + CEILING_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, y_offset + CEILING_HEIGHT, z1),
            normal,
            tangent,
            Vec2::new(1.0, 1.0),
        ),
        make_vertex(
            Vec3::new(x1, y_offset + CEILING_HEIGHT, z0),
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

    let (h_nw, h_ne, h_sw, h_se) = ramp_corner_heights(tile);
    let h_nw = h_nw + y_offset;
    let h_ne = h_ne + y_offset;
    let h_sw = h_sw + y_offset;
    let h_se = h_se + y_offset;

    // Normal calculation for the ramp plane
    let p0 = Vec3::new(x0, h_nw, z1);
    let p1 = Vec3::new(x1, h_ne, z1);
    let p2 = Vec3::new(x0, h_sw, z0);
    let normal = (p1 - p0).cross(p2 - p0).normalize();
    let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

    push_quad(
        verts,
        inds,
        make_vertex(
            Vec3::new(x0, h_sw, z0),
            normal,
            tangent,
            Vec2::new(0.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, h_se, z0),
            normal,
            tangent,
            Vec2::new(1.0, 0.0),
        ),
        make_vertex(
            Vec3::new(x1, h_ne, z1),
            normal,
            tangent,
            Vec2::new(1.0, 1.0),
        ),
        make_vertex(
            Vec3::new(x0, h_nw, z1),
            normal,
            tangent,
            Vec2::new(0.0, 1.0),
        ),
    );
}

fn emit_ramp_caps(
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

    let (h_nw, h_ne, h_sw, h_se) = ramp_corner_heights(tile);
    let h_nw = h_nw + y_offset;
    let h_ne = h_ne + y_offset;
    let h_sw = h_sw + y_offset;
    let h_se = h_se + y_offset;

    // We emit side caps for the ramp so it isn't "paper thin" from the side
    // West face
    if h_nw > 0.0 || h_sw > 0.0 {
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
                Vec3::new(x0, h_sw, z0),
                normal,
                tangent,
                Vec2::new(1.0, h_sw - y_offset),
            ),
            make_vertex(
                Vec3::new(x0, h_nw, z1),
                normal,
                tangent,
                Vec2::new(0.0, h_nw - y_offset),
            ),
        );
    }

    // East face
    if h_ne > 0.0 || h_se > 0.0 {
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
                Vec3::new(x1, h_ne, z1),
                normal,
                tangent,
                Vec2::new(1.0, h_ne - y_offset),
            ),
            make_vertex(
                Vec3::new(x1, h_se, z0),
                normal,
                tangent,
                Vec2::new(0.0, h_se - y_offset),
            ),
        );
    }

    // North face
    if h_nw > 0.0 || h_ne > 0.0 {
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
                Vec3::new(x1, h_ne, z1),
                normal,
                tangent,
                Vec2::new(1.0, h_ne - y_offset),
            ),
            make_vertex(
                Vec3::new(x0, h_nw, z1),
                normal,
                tangent,
                Vec2::new(0.0, h_nw - y_offset),
            ),
        );
    }

    // South face
    if h_sw > 0.0 || h_se > 0.0 {
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
                Vec3::new(x0, h_sw, z0),
                normal,
                tangent,
                Vec2::new(1.0, h_sw - y_offset),
            ),
            make_vertex(
                Vec3::new(x1, h_se, z0),
                normal,
                tangent,
                Vec2::new(0.0, h_se - y_offset),
            ),
        );
    }
}

fn ramp_corner_heights(tile: Tile) -> (f32, f32, f32, f32) {
    match tile {
        Tile::RampNorth(lvl) => {
            let h0 = lvl as f32 * RAMP_RISE;
            let h1 = (lvl as f32 + 1.0) * RAMP_RISE;
            (h1, h1, h0, h0)
        }
        Tile::RampSouth(lvl) => {
            let h0 = lvl as f32 * RAMP_RISE;
            let h1 = (lvl as f32 + 1.0) * RAMP_RISE;
            (h0, h0, h1, h1)
        }
        Tile::RampEast(lvl) => {
            let h0 = lvl as f32 * RAMP_RISE;
            let h1 = (lvl as f32 + 1.0) * RAMP_RISE;
            (h0, h1, h0, h1)
        }
        Tile::RampWest(lvl) => {
            let h0 = lvl as f32 * RAMP_RISE;
            let h1 = (lvl as f32 + 1.0) * RAMP_RISE;
            (h1, h0, h1, h0)
        }
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
            layers: vec![tiles],
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
    fn ramp_orientation_geometry_correctness() {
        let (nw, ne, sw, se) = ramp_corner_heights(Tile::RampNorth(0));
        assert!((nw - RAMP_RISE).abs() < 1e-4);
        assert!((ne - RAMP_RISE).abs() < 1e-4);
        assert!(sw.abs() < 1e-4);
        assert!(se.abs() < 1e-4);
    }
}
