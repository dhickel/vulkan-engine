use glam::{Vec2, Vec3};

use crate::layout::{tile_to_world, ParsedLevel, Tile};
use crate::player::{PlayerState, PLAYER_EYE_HEIGHT, PLAYER_RADIUS};
use physics::{BodyDescriptor, BodyKind, ColliderDescriptor, ColliderShape, PhysicsBodyId, PhysicsColliderId, PhysicsWorld};

pub const TILE_SIZE: f32 = 1.0;
pub const WALL_HEIGHT: f32 = 2.5;
pub const CEILING_HEIGHT: f32 = 2.5;
pub const RAMP_RISE: f32 = 0.833333; // 2.5 / 3 (takes 3 tiles to go full height)

pub const COLLISION_MAX_ITERS: usize = 4;
pub const COLLISION_EPSILON: f32 = 1e-4;
pub const CHUNK_SIZE: usize = 16;

const MAX_STEP_DOWN_PER_FRAME: f32 = 0.15;
const MAX_STEP_UP_HEIGHT: f32 = RAMP_RISE + 0.1;

#[derive(Clone, Copy, Debug)]
pub struct WallCollider {
    pub min: Vec3,
    pub max: Vec3,
    pub tile: (usize, usize),
}

#[derive(Clone, Copy, Debug)]
pub struct RampCollider {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub normal: Vec3,
    pub d: f32,
    pub tile: (usize, usize),
    pub tile_kind: Tile,
    pub y_offset: f32,
}

#[derive(Default)]
pub struct CollisionWorld {
    pub walls: Vec<WallCollider>,
    pub ramps: Vec<RampCollider>,
    pub floor_tiles: Vec<(usize, usize, f32)>, // x, y, y_offset
}

impl CollisionWorld {
    pub fn from_level(level: &ParsedLevel) -> Self {
        let mut walls = Vec::new();
        let mut ramps = Vec::new();
        let mut floor_tiles = Vec::new();

        for layer_idx in 0..level.layer_count() {
            let y_offset = layer_idx as f32 * WALL_HEIGHT;

            for y in 0..level.height {
                for x in 0..level.width {
                    let origin = tile_to_world(x, y);
                    let min = Vec3::new(origin.x, y_offset, origin.z - TILE_SIZE);
                    let max = Vec3::new(origin.x + TILE_SIZE, y_offset + WALL_HEIGHT, origin.z);

                    let tile = level.tile_at_3d(layer_idx, x, y);
                    match tile {
                        Tile::Wall => {
                            walls.push(WallCollider {
                                min,
                                max,
                                tile: (x, y),
                            });
                        }
                        Tile::Floor => {
                            floor_tiles.push((x, y, y_offset));
                        }
                        Tile::Void => {}
                        Tile::RampNorth(_)
                        | Tile::RampEast(_)
                        | Tile::RampSouth(_)
                        | Tile::RampWest(_) => {
                            let (p0, p1, p2) = ramp_plane_points(tile, origin, y_offset);
                            let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                            if normal.y < 0.0 {
                                normal = -normal;
                            }

                            let h0 = ramp_height(tile, 0.0, 0.0, y_offset).unwrap();
                            let h1 = ramp_height(tile, 1.0, 1.0, y_offset).unwrap();
                            let h_min = h0.min(h1);
                            let h_max = h0.max(h1);

                            ramps.push(RampCollider {
                                bounds_min: Vec3::new(min.x, h_min, min.z),
                                bounds_max: Vec3::new(max.x, h_max, max.z),
                                normal,
                                d: -normal.dot(p0),
                                tile: (x, y),
                                tile_kind: tile,
                                y_offset,
                            });
                        }
                    }
                }
            }
        }

        Self {
            walls,
            ramps,
            floor_tiles,
        }
    }
}

/// Seed level geometry as static cuboid colliders in a physics world.
///
/// Creates simple AABB colliders for walls, floors, and ramp planes.
/// This provides a single collision-authority path for the character
/// controller.  Returns the number of created colliders.
pub fn seed_level_colliders(world: &mut PhysicsWorld, level: &ParsedLevel) -> Result<usize, physics::PhysicsError> {
    let mut count = 0;
    for layer_idx in 0..level.layer_count() {
        let y_offset = layer_idx as f32 * WALL_HEIGHT;
        for y in 0..level.height {
            for x in 0..level.width {
                let origin = tile_to_world(x, y);
                let tile = level.tile_at_3d(layer_idx, x, y);
                let body_id = PhysicsBodyId::new(format!("body.level_L{layer_idx}_{x}_{y}"));
                match tile {
                    Tile::Wall => {
                        let cx = origin.x + TILE_SIZE * 0.5;
                        let cy = y_offset + WALL_HEIGHT * 0.5;
                        let cz = origin.z - TILE_SIZE * 0.5;
                        world.create_body(BodyDescriptor::new(
                            body_id.clone(),
                            BodyKind::Static,
                            [cx, cy, cz],
                        ))?;
                        let collider_id = PhysicsColliderId::new(format!("collider.level_L{layer_idx}_{x}_{y}"));
                        world.create_collider(ColliderDescriptor::new(
                            collider_id,
                            body_id,
                            ColliderShape::Cuboid {
                                half_extents: [TILE_SIZE * 0.5, WALL_HEIGHT * 0.5, TILE_SIZE * 0.5],
                            },
                        ))?;
                        count += 1;
                    }
                    Tile::Floor => {
                        let cx = origin.x + TILE_SIZE * 0.5;
                        let cy = y_offset;
                        let cz = origin.z - TILE_SIZE * 0.5;
                        world.create_body(BodyDescriptor::new(
                            body_id.clone(),
                            BodyKind::Static,
                            [cx, cy, cz],
                        ))?;
                        let collider_id = PhysicsColliderId::new(format!("collider.level_L{layer_idx}_{x}_{y}"));
                        world.create_collider(ColliderDescriptor::new(
                            collider_id,
                            body_id,
                            ColliderShape::Cuboid {
                                half_extents: [TILE_SIZE * 0.5, 0.05, TILE_SIZE * 0.5],
                            },
                        ))?;
                        count += 1;
                    }
                    Tile::Void => {}
                    Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_) => {
                        // Ramps are approximated as thin cuboid planes.
                        let (p0, p1, p2) = ramp_plane_points(tile, origin, y_offset);
                        let center = (p0 + p1 + p2) / 3.0;
                        world.create_body(BodyDescriptor::new(
                            body_id.clone(),
                            BodyKind::Static,
                            [center.x, center.y, center.z],
                        ))?;
                        let collider_id = PhysicsColliderId::new(format!("collider.level_L{layer_idx}_{x}_{y}"));
                        world.create_collider(ColliderDescriptor::new(
                            collider_id,
                            body_id,
                            ColliderShape::Cuboid {
                                half_extents: [TILE_SIZE * 0.5, 0.1, TILE_SIZE * 0.5],
                            },
                        ))?;
                        count += 1;
                    }
                }
            }
        }
    }
    log::info!("Seeded {} level colliders into physics world", count);
    Ok(count)
}

pub fn resolve_player_step(player: &mut PlayerState, world: &CollisionWorld, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    let desired = player.velocity * dt;
    if player.noclip {
        player.position += desired;
        return;
    }

    let mut pos = player.position + Vec3::new(desired.x, 0.0, desired.z);

    for _ in 0..COLLISION_MAX_ITERS {
        let correction = resolve_wall_penetration_iter(&mut pos, PLAYER_RADIUS, world);
        if correction < COLLISION_EPSILON {
            break;
        }
    }

    let ground = solve_ground_height(pos, world);
    let target_y = ground + PLAYER_EYE_HEIGHT;
    if target_y < pos.y {
        pos.y = (pos.y - MAX_STEP_DOWN_PER_FRAME).max(target_y);
    } else {
        pos.y = target_y;
    }

    player.position = pos;
}

fn solve_ground_height(pos: Vec3, world: &CollisionWorld) -> f32 {
    let mut ground: f32 = 0.0;
    let player_base_y = pos.y - PLAYER_EYE_HEIGHT;

    // Check flat floors
    let tx = (pos.x / TILE_SIZE).floor() as usize;
    let ty = ((-pos.z) / TILE_SIZE).floor() as usize;

    for &(fx, fy, fy_offset) in &world.floor_tiles {
        if fx == tx && fy == ty {
            // Found a floor at this tile. If it's below or at player's current base Y, it's a candidate.
            if fy_offset <= player_base_y + MAX_STEP_UP_HEIGHT {
                ground = ground.max(fy_offset);
            }
        }
    }

    // Check ramps
    for ramp in &world.ramps {
        if pos.x < ramp.bounds_min.x - COLLISION_EPSILON
            || pos.x > ramp.bounds_max.x + COLLISION_EPSILON
            || pos.z < ramp.bounds_min.z - COLLISION_EPSILON
            || pos.z > ramp.bounds_max.z + COLLISION_EPSILON
        {
            continue;
        }

        let (rx, ry) = ramp.tile;
        let origin = tile_to_world(rx, ry);
        let local_x = ((pos.x - origin.x) / TILE_SIZE).clamp(0.0, 1.0);
        let local_z = ((origin.z - pos.z) / TILE_SIZE).clamp(0.0, 1.0);

        if let Some(h) = ramp_height(ramp.tile_kind, local_x, local_z, ramp.y_offset) {
            if h <= player_base_y + MAX_STEP_UP_HEIGHT {
                ground = ground.max(h);
            }
        }
    }

    ground
}

pub fn ramp_height(tile: Tile, local_x: f32, local_z: f32, y_offset: f32) -> Option<f32> {
    let (level, slope) = match tile {
        Tile::RampNorth(lvl) => (lvl as f32, (1.0 - local_z).clamp(0.0, 1.0)),
        Tile::RampSouth(lvl) => (lvl as f32, local_z.clamp(0.0, 1.0)),
        Tile::RampEast(lvl) => (lvl as f32, local_x.clamp(0.0, 1.0)),
        Tile::RampWest(lvl) => (lvl as f32, (1.0 - local_x).clamp(0.0, 1.0)),
        _ => return None,
    };
    Some(y_offset + (level + slope) * RAMP_RISE)
}

fn world_to_tile(pos: Vec3) -> (isize, isize) {
    (
        (pos.x / TILE_SIZE).floor() as isize,
        ((-pos.z) / TILE_SIZE).floor() as isize,
    )
}

fn nearby_walls<'a>(
    world: &'a CollisionWorld,
    center_tile: (isize, isize),
) -> impl Iterator<Item = &'a WallCollider> {
    world.walls.iter().filter(move |wall| {
        let dx = wall.tile.0 as isize - center_tile.0;
        let dy = wall.tile.1 as isize - center_tile.1;
        dx.abs() <= 1 && dy.abs() <= 1
    })
}

fn ramp_plane_points(tile: Tile, origin: Vec3, y_offset: f32) -> (Vec3, Vec3, Vec3) {
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    match tile {
        Tile::RampNorth(lvl) => {
            let h0 = y_offset + lvl as f32 * RAMP_RISE;
            let h1 = y_offset + (lvl as f32 + 1.0) * RAMP_RISE;
            (
                Vec3::new(x0, h1, z0),
                Vec3::new(x1, h1, z0),
                Vec3::new(x0, h0, z1),
            )
        }
        Tile::RampSouth(lvl) => {
            let h0 = y_offset + lvl as f32 * RAMP_RISE;
            let h1 = y_offset + (lvl as f32 + 1.0) * RAMP_RISE;
            (
                Vec3::new(x0, h0, z0),
                Vec3::new(x1, h0, z0),
                Vec3::new(x0, h1, z1),
            )
        }
        Tile::RampEast(lvl) => {
            let h0 = y_offset + lvl as f32 * RAMP_RISE;
            let h1 = y_offset + (lvl as f32 + 1.0) * RAMP_RISE;
            (
                Vec3::new(x0, h0, z0),
                Vec3::new(x1, h1, z0),
                Vec3::new(x0, h0, z1),
            )
        }
        Tile::RampWest(lvl) => {
            let h0 = y_offset + lvl as f32 * RAMP_RISE;
            let h1 = y_offset + (lvl as f32 + 1.0) * RAMP_RISE;
            (
                Vec3::new(x0, h1, z0),
                Vec3::new(x1, h0, z0),
                Vec3::new(x0, h1, z1),
            )
        }
        _ => (
            Vec3::new(x0, y_offset, z0),
            Vec3::new(x1, y_offset, z0),
            Vec3::new(x0, y_offset, z1),
        ),
    }
}

pub fn is_penetrating_wall(pos: Vec3, radius: f32, wall: &WallCollider) -> bool {
    let capsule_min = pos.y - PLAYER_EYE_HEIGHT;
    let capsule_max = capsule_min + 1.8;

    if capsule_max <= wall.min.y + COLLISION_EPSILON
        || capsule_min >= wall.max.y - COLLISION_EPSILON
    {
        return false;
    }

    // AABB vs Circle overlap in XZ plane
    let closest_x = pos.x.clamp(wall.min.x, wall.max.x);
    let closest_z = pos.z.clamp(wall.min.z, wall.max.z);

    let dx = pos.x - closest_x;
    let dz = pos.z - closest_z;

    let dist_sq = dx * dx + dz * dz;
    dist_sq < radius * radius
}

fn resolve_wall_penetration_iter(pos: &mut Vec3, radius: f32, world: &CollisionWorld) -> f32 {
    let mut max_correction: f32 = 0.0;
    let tile = world_to_tile(*pos);

    for wall in nearby_walls(world, tile) {
        if is_penetrating_wall(*pos, radius, wall) {
            let correction = calculate_wall_correction(*pos, radius, wall);
            *pos += Vec3::new(correction.x, 0.0, correction.y);
            max_correction = max_correction.max(correction.length());
        }
    }

    max_correction
}

fn calculate_wall_correction(pos: Vec3, radius: f32, wall: &WallCollider) -> Vec2 {
    let closest_x = pos.x.clamp(wall.min.x, wall.max.x);
    let closest_z = pos.z.clamp(wall.min.z, wall.max.z);

    let dx = pos.x - closest_x;
    let dz = pos.z - closest_z;

    let dist_sq = dx * dx + dz * dz;
    let dist = dist_sq.sqrt();

    if dist < 1e-6 {
        // Center is inside or on edge. Push out along the shallowest axis.
        let left = pos.x - wall.min.x;
        let right = wall.max.x - pos.x;
        let down = pos.z - wall.min.z;
        let up = wall.max.z - pos.z;

        let min_axis = left.min(right).min(down.min(up));

        if (min_axis - left).abs() < 1e-6 {
            Vec2::new(-(radius + left), 0.0)
        } else if (min_axis - right).abs() < 1e-6 {
            Vec2::new(radius + right, 0.0)
        } else if (min_axis - down).abs() < 1e-6 {
            Vec2::new(0.0, -(radius + down))
        } else {
            Vec2::new(0.0, radius + up)
        }
    } else if dist < radius {
        let buffer = 1.001; // Tiny buffer to push slightly beyond the radius
        let push_dist = radius * buffer - dist;
        Vec2::new(dx / dist * push_dist, dz / dist * push_dist)
    } else {
        Vec2::ZERO
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

    #[test]
    fn wall_collision_sliding() {
        let level = parsed_level(
            3,
            3,
            vec![
                Tile::Floor,
                Tile::Wall,
                Tile::Floor,
                Tile::Wall,
                Tile::Wall,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
                Tile::Floor,
            ],
        );
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(0.6, PLAYER_EYE_HEIGHT, -2.4));
        player.velocity = Vec3::new(4.0, 0.0, 4.0);

        let before_z = player.position.z;
        for _ in 0..5 {
            resolve_player_step(&mut player, &world, 0.02);
        }

        assert!(player.position.x < 1.0 + PLAYER_RADIUS + 1e-3);
        assert!(
            player.position.z > before_z,
            "player should slide along non-blocked axis"
        );
    }

    #[test]
    fn ramp_ascend() {
        let level = parsed_level(3, 1, vec![Tile::Floor, Tile::RampEast(0), Tile::Floor]);
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(1.05, PLAYER_EYE_HEIGHT, -0.5));
        player.velocity = Vec3::new(2.0, 0.0, 0.0);

        let before = player.position.y;
        resolve_player_step(&mut player, &world, 0.1);

        assert!(player.position.y > before);
    }

    #[test]
    fn ramp_descend_without_large_snap() {
        let level = parsed_level(3, 1, vec![Tile::Floor, Tile::RampWest(0), Tile::Floor]);
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(1.9, PLAYER_EYE_HEIGHT + RAMP_RISE, -0.5));
        player.velocity = Vec3::new(-0.8, 0.0, 0.0);

        let mut max_snap = 0.0f32;
        for _ in 0..5 {
            let before = player.position.y;
            resolve_player_step(&mut player, &world, 0.05);
            let snap = (before - player.position.y).max(0.0);
            max_snap = max_snap.max(snap);
        }

        assert!(max_snap <= 0.15 + 1e-3);
    }

    #[test]
    fn no_penetration_invariant_after_solver() {
        let level = parsed_level(
            3,
            3,
            vec![
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Floor,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
                Tile::Wall,
            ],
        );
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(1.5, PLAYER_EYE_HEIGHT, -1.5));
        player.velocity = Vec3::new(2.0, 0.0, 2.0);

        for _ in 0..10 {
            resolve_player_step(&mut player, &world, 0.02);
        }

        for wall in &world.walls {
            assert!(
                !is_penetrating_wall(player.position, PLAYER_RADIUS, wall),
                "player penetrates wall at {:?}",
                wall.tile
            );
        }
    }

    #[test]
    fn noclip_applies_raw_velocity_without_collision_or_ground_resolution() {
        let level = parsed_level(1, 1, vec![Tile::Wall]);
        let world = CollisionWorld::from_level(&level);
        let mut player = PlayerState::new(Vec3::new(0.5, PLAYER_EYE_HEIGHT, -0.5));
        player.noclip = true;
        player.velocity = Vec3::new(2.0, 3.0, -4.0);

        resolve_player_step(&mut player, &world, 0.25);

        assert_eq!(player.position, Vec3::new(1.0, PLAYER_EYE_HEIGHT + 0.75, -1.5));
    }

    #[test]
    fn second_floor_player_does_not_collide_with_lower_wall_touching_base_height() {
        let level = parsed_level_layers(1, 1, vec![vec![Tile::Wall], vec![Tile::Floor]]);
        let world = CollisionWorld::from_level(&level);
        let player_pos = Vec3::new(0.5, WALL_HEIGHT + PLAYER_EYE_HEIGHT, -0.5);

        let lower_wall = world
            .walls
            .iter()
            .find(|wall| (wall.min.y - 0.0).abs() <= COLLISION_EPSILON)
            .expect("lower wall collider should exist");

        assert!(
            !is_penetrating_wall(player_pos, PLAYER_RADIUS, lower_wall),
            "lower floor wall top should not collide with a second-floor player"
        );
    }
}
