use glam::{Vec2, Vec3};

use crate::layout::{tile_to_world, ParsedLevel, Tile};
use crate::player::{PlayerState, PLAYER_EYE_HEIGHT, PLAYER_RADIUS};

pub const TILE_SIZE: f32 = 1.0;
pub const WALL_HEIGHT: f32 = 2.5;
pub const CEILING_HEIGHT: f32 = 2.5;
pub const RAMP_RISE: f32 = 2.5;

pub const COLLISION_MAX_ITERS: usize = 4;
pub const COLLISION_EPSILON: f32 = 1e-4;
pub const CHUNK_SIZE: usize = 16;

const MAX_PUSH_PER_ITER: f32 = 0.25;
const MAX_STEP_DOWN_PER_FRAME: f32 = 0.15;

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
}

#[derive(Default)]
pub struct CollisionWorld {
    pub walls: Vec<WallCollider>,
    pub ramps: Vec<RampCollider>,
}

impl CollisionWorld {
    pub fn from_level(level: &ParsedLevel) -> Self {
        let mut walls = Vec::new();
        let mut ramps = Vec::new();

        for y in 0..level.height {
            for x in 0..level.width {
                let origin = tile_to_world(x, y);
                let min = Vec3::new(origin.x, 0.0, origin.z - TILE_SIZE);
                let max = Vec3::new(origin.x + TILE_SIZE, WALL_HEIGHT, origin.z);

                let tile = level.tile_at(x, y);
                match tile {
                    Tile::Wall => walls.push(WallCollider {
                        min,
                        max,
                        tile: (x, y),
                    }),
                    Tile::RampNorth | Tile::RampEast | Tile::RampSouth | Tile::RampWest => {
                        let (p0, p1, p2) = ramp_plane_points(tile, origin);
                        let mut normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                        if normal.y < 0.0 {
                            normal = -normal;
                        }
                        ramps.push(RampCollider {
                            bounds_min: min,
                            bounds_max: Vec3::new(max.x, RAMP_RISE, max.z),
                            normal,
                            d: -normal.dot(p0),
                            tile: (x, y),
                            tile_kind: tile,
                        });
                    }
                    _ => {}
                }
            }
        }

        Self { walls, ramps }
    }
}

pub fn resolve_player_step(player: &mut PlayerState, world: &CollisionWorld, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    let desired = player.velocity * dt;
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

fn resolve_wall_penetration_iter(pos: &mut Vec3, radius: f32, world: &CollisionWorld) -> f32 {
    let tile = world_to_tile(*pos);
    let mut correction_sum = 0.0f32;

    for wall in nearby_walls(world, tile) {
        if !capsule_y_overlaps_wall(pos.y, wall.min.y, wall.max.y) {
            continue;
        }
        let correction = wall_pushout(Vec2::new(pos.x, pos.z), radius, wall);
        if correction.length_squared() > COLLISION_EPSILON * COLLISION_EPSILON {
            let c = correction.clamp_length_max(MAX_PUSH_PER_ITER);
            pos.x += c.x;
            pos.z += c.y;
            correction_sum += c.length();
        }
    }

    correction_sum
}

fn wall_pushout(center: Vec2, radius: f32, wall: &WallCollider) -> Vec2 {
    let nearest_x = center.x.clamp(wall.min.x, wall.max.x);
    let nearest_z = center.y.clamp(wall.min.z, wall.max.z);
    let delta = Vec2::new(center.x - nearest_x, center.y - nearest_z);
    let dist_sq = delta.length_squared();
    let r_sq = radius * radius;

    if dist_sq >= r_sq {
        return Vec2::ZERO;
    }

    if dist_sq > 1e-8 {
        let dist = dist_sq.sqrt();
        let dir = delta / dist;
        return dir * (radius - dist);
    }

    let left = center.x - wall.min.x;
    let right = wall.max.x - center.x;
    let down = center.y - wall.min.z;
    let up = wall.max.z - center.y;
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
}

fn capsule_y_overlaps_wall(camera_y: f32, wall_min_y: f32, wall_max_y: f32) -> bool {
    let capsule_min = camera_y - PLAYER_EYE_HEIGHT;
    let capsule_max = capsule_min + 1.8;
    capsule_max >= wall_min_y && capsule_min <= wall_max_y
}

fn solve_ground_height(pos: Vec3, world: &CollisionWorld) -> f32 {
    let mut ground: f32 = 0.0;
    let tile = world_to_tile(pos);

    for ramp in nearby_ramps(world, tile) {
        if pos.x < ramp.bounds_min.x - COLLISION_EPSILON
            || pos.x > ramp.bounds_max.x + COLLISION_EPSILON
            || pos.z < ramp.bounds_min.z - COLLISION_EPSILON
            || pos.z > ramp.bounds_max.z + COLLISION_EPSILON
        {
            continue;
        }

        let (tx, ty) = ramp.tile;
        let origin = tile_to_world(tx, ty);
        let local_x = ((pos.x - origin.x) / TILE_SIZE).clamp(0.0, 1.0);
        let local_z = ((origin.z - pos.z) / TILE_SIZE).clamp(0.0, 1.0);

        if let Some(h) = ramp_height(ramp.tile_kind, local_x, local_z) {
            ground = ground.max(h);
        }
    }

    ground.max(0.0)
}

pub fn ramp_height(tile: Tile, local_x: f32, local_z: f32) -> Option<f32> {
    match tile {
        Tile::RampNorth => Some((1.0 - local_z).clamp(0.0, 1.0) * RAMP_RISE),
        Tile::RampSouth => Some(local_z.clamp(0.0, 1.0) * RAMP_RISE),
        Tile::RampEast => Some(local_x.clamp(0.0, 1.0) * RAMP_RISE),
        Tile::RampWest => Some((1.0 - local_x).clamp(0.0, 1.0) * RAMP_RISE),
        _ => None,
    }
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

fn nearby_ramps<'a>(
    world: &'a CollisionWorld,
    center_tile: (isize, isize),
) -> impl Iterator<Item = &'a RampCollider> {
    world.ramps.iter().filter(move |ramp| {
        let dx = ramp.tile.0 as isize - center_tile.0;
        let dy = ramp.tile.1 as isize - center_tile.1;
        dx.abs() <= 1 && dy.abs() <= 1
    })
}

fn ramp_plane_points(tile: Tile, origin: Vec3) -> (Vec3, Vec3, Vec3) {
    let x0 = origin.x;
    let x1 = origin.x + TILE_SIZE;
    let z0 = origin.z;
    let z1 = origin.z - TILE_SIZE;

    match tile {
        Tile::RampNorth => (
            Vec3::new(x0, RAMP_RISE, z0),
            Vec3::new(x1, RAMP_RISE, z0),
            Vec3::new(x0, 0.0, z1),
        ),
        Tile::RampSouth => (
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x0, RAMP_RISE, z1),
        ),
        Tile::RampEast => (
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x1, RAMP_RISE, z0),
            Vec3::new(x0, 0.0, z1),
        ),
        Tile::RampWest => (
            Vec3::new(x0, RAMP_RISE, z0),
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x0, RAMP_RISE, z1),
        ),
        _ => (
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x0, 0.0, z1),
        ),
    }
}

pub fn is_penetrating_wall(pos: Vec3, radius: f32, wall: &WallCollider) -> bool {
    let nearest_x = pos.x.clamp(wall.min.x, wall.max.x);
    let nearest_z = pos.z.clamp(wall.min.z, wall.max.z);
    let delta = Vec2::new(pos.x - nearest_x, pos.z - nearest_z);
    delta.length_squared() < radius * radius - 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::PlayerState;

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

    #[test]
    fn wall_blocking() {
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
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(0.6, PLAYER_EYE_HEIGHT, -1.5));
        player.velocity = Vec3::new(8.0, 0.0, 0.0);

        resolve_player_step(&mut player, &world, 0.1);

        let wall = &world.walls[0];
        assert!(player.position.x <= wall.min.x - PLAYER_RADIUS + 1e-3);
    }

    #[test]
    fn corner_slide_behavior() {
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
        let level = parsed_level(3, 1, vec![Tile::Floor, Tile::RampEast, Tile::Floor]);
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(1.05, PLAYER_EYE_HEIGHT, -0.5));
        player.velocity = Vec3::new(2.0, 0.0, 0.0);

        let before = player.position.y;
        resolve_player_step(&mut player, &world, 0.1);

        assert!(player.position.y > before);
    }

    #[test]
    fn ramp_descend_without_large_snap() {
        let level = parsed_level(3, 1, vec![Tile::Floor, Tile::RampWest, Tile::Floor]);
        let world = CollisionWorld::from_level(&level);

        let mut player = PlayerState::new(Vec3::new(1.9, PLAYER_EYE_HEIGHT + 2.5, -0.5));
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
        player.velocity = Vec3::new(6.0, 0.0, 6.0);

        for _ in 0..5 {
            resolve_player_step(&mut player, &world, 0.02);
        }

        for wall in nearby_walls(&world, world_to_tile(player.position)) {
            assert!(
                !is_penetrating_wall(player.position, PLAYER_RADIUS, wall),
                "player penetrates wall at {:?}",
                wall.tile
            );
        }
    }
}
