//! Player navigation: fixed-step movement using point_contents and trace_line.
//!
//! Implements Quake-style player movement using BSP clipnode point traces
//! (hull 0) with application-level hull expansion. The ericw-tools qbsp
//! compiler does not generate separate hull 1 clipnode trees; player
//! movement uses the point hull with explicit bounding-box validation.
//!
//! # Hull Contract
//!
//! The player bounding box is ±(16, 16, 24) Quake units (symmetric).
//! This is the reference Quake player size. The FGD variant
//! (-16,-16,-24)-(16,16,32) is not used because the BSP pre-expanded
//! hull 1 is not available.

use bsp::coords::QuakeToEngine;
use bsp::{self, StoredHull, TraceResult};
use glam::Vec3;

/// Player hull half-extents in engine units (symmetric player hull).
pub const PLAYER_HALF_EXTENTS_ENGINE: Vec3 = Vec3::new(
    16.0 * 0.0254,
    24.0 * 0.0254, // Z extent → engine Y (up)
    16.0 * 0.0254, // Y extent → engine -Z
);

/// Fixed-step player mover using BSP clipnode traces.
pub struct PlayerMover {
    /// Current position in engine space.
    pub position: Vec3,
    /// Player hull half-extents in engine units.
    pub half_extents: Vec3,
    /// Current velocity (engine units per second).
    pub velocity: Vec3,
    /// Whether the mover is on the ground.
    pub on_ground: bool,
}

impl PlayerMover {
    /// Create a new player mover at the given engine-space position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            half_extents: PLAYER_HALF_EXTENTS_ENGINE,
            velocity: Vec3::ZERO,
            on_ground: false,
        }
    }

    /// Check whether a point in engine space is in empty (walkable) space.
    pub fn is_clear(
        &self,
        position: Vec3,
        nodes: &[bsp::lumps::Node],
        leaves: &[bsp::lumps::Leaf],
        planes: &[bsp::lumps::Plane],
    ) -> bool {
        let contents = bsp::point_contents(position, nodes, leaves, planes);
        !contents.is_solid()
    }

    /// Trace a line from current position by a delta, returning the
    /// trace result for point hull (hull 0).
    pub fn trace_move(
        &self,
        delta: Vec3,
        clipnodes: &[bsp::lumps::Clipnode],
        planes: &[bsp::lumps::Plane],
        models: &[bsp::lumps::Model],
        qte: &QuakeToEngine,
    ) -> TraceResult {
        let end = self.position + delta;
        bsp::trace_line(
            self.position,
            end,
            StoredHull::Point,
            clipnodes,
            planes,
            models,
            qte,
        )
    }

    /// Attempt a fixed-step move. Returns the new position after resolving
    /// collisions via simple slide-along-wall.
    ///
    /// If `resolve_sliding` is true, the mover will attempt to slide along
    /// the hit plane instead of stopping dead.
    pub fn step(
        &mut self,
        delta: Vec3,
        clipnodes: &[bsp::lumps::Clipnode],
        planes_data: &[bsp::lumps::Plane],
        models: &[bsp::lumps::Model],
        _nodes: &[bsp::lumps::Node],
        _leaves: &[bsp::lumps::Leaf],
        _bsp_planes: &[bsp::lumps::Plane],
        qte: &QuakeToEngine,
        resolve_sliding: bool,
    ) {
        if delta.length_squared() < 1e-10 {
            return;
        }

        let result = self.trace_move(delta, clipnodes, planes_data, models, qte);

        if result.starts_solid {
            // Try to nudge out — should not happen in normal operation
            return;
        }

        if result.no_hit {
            // Full move succeeded
            self.position += delta;
            self.on_ground = false;
        } else if result.hit_fraction < 1e-6 {
            // Blocked immediately; don't move
        } else if resolve_sliding {
            // Move to the hit point, then attempt to slide
            let move_fraction = (result.hit_fraction - 0.001).max(0.0);
            self.position += delta * move_fraction;

            // Compute slide vector: remove the component along the hit normal
            let remaining = delta * (1.0 - move_fraction);
            let normal = result.plane_normal;
            let dot = remaining.dot(normal);
            if dot < 0.0 {
                let slide = remaining - normal * dot;
                // Recurse with the slide vector (one level, no infinite recursion)
                if slide.length_squared() > 1e-10 {
                    let slide_result = self.trace_move(slide, clipnodes, planes_data, models, qte);
                    if slide_result.no_hit {
                        self.position += slide;
                    } else if slide_result.hit_fraction > 1e-6 {
                        self.position += slide * (slide_result.hit_fraction - 0.001).max(0.0);
                    }
                }
            }
        } else {
            // Move up to the hit point but not through
            let move_fraction = (result.hit_fraction - 0.001).max(0.0);
            self.position += delta * move_fraction;
        }
    }

    /// Verify that the current position is in non-solid space.
    pub fn validate_position(
        &self,
        nodes: &[bsp::lumps::Node],
        leaves: &[bsp::lumps::Leaf],
        planes: &[bsp::lumps::Plane],
    ) -> bool {
        self.is_clear(self.position, nodes, leaves, planes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp::coords::QuakeToEngine;
    use bsp::{BspLoader, LoadOptions};
    use glam::Vec3;
    use std::path::Path;

    fn load_fixture(name: &str) -> bsp::BspWorld {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/bsp/tests/fixtures/compiled")
            .join(name);
        let data = std::fs::read(&path).unwrap();
        let options = LoadOptions {
            strict: true,
            source_identity: name.into(),
            ..LoadOptions::default()
        };
        BspLoader::load(&data, &options).unwrap()
    }

    #[test]
    fn player_mover_spawn_valid() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mover = PlayerMover::new(spawn_eng);
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_simple_east_move() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 100.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east by 200 quake units, north of the pillar (clear path)
        let delta_q = Vec3::new(200.0, 0.0, 0.0);
        let delta_eng = Vec3::new(
            qte.scale * delta_q.x,
            qte.scale * delta_q.z,
            -qte.scale * delta_q.y,
        );

        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        // Should have moved east
        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        assert!(
            moved_q.x > -100.0,
            "mover should have moved east, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_blocked_by_pillar() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east directly toward the pillar
        let delta_eng = Vec3::new(qte.scale * 200.0, 0.0, 0.0);
        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        // Should have been stopped by the pillar
        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        // The pillar is at x=-16..16. Mover should not have reached past x=-17
        assert!(
            moved_q.x < 0.0,
            "mover should be blocked by pillar, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_slide_along_wall() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        // Start near but not at the west wall, heading north-west at an angle.
        let spawn_q = Vec3::new(-220.0, -100.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move north-west (-20 X, +200 Y in Quake)
        let delta_q = Vec3::new(-20.0, 200.0, 0.0);
        let delta_eng = Vec3::new(
            qte.scale * delta_q.x,
            qte.scale * delta_q.z,
            -qte.scale * delta_q.y,
        );

        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        let pos_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        // Should slide north along west wall
        assert!(
            pos_q.x <= -210.0,
            "mover must not pass through west wall, x={}",
            pos_q.x
        );
        assert!(pos_q.y > -90.0, "mover must slide north, y={}", pos_q.y);
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_validate_all_positions() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();

        // Test multiple clear positions around the room
        let clear_positions = [
            Vec3::new(-200.0, 0.0, 0.0),
            Vec3::new(200.0, 0.0, 0.0),
            Vec3::new(0.0, 200.0, 0.0),
            Vec3::new(0.0, -200.0, 0.0),
            Vec3::new(-128.0, 128.0, 0.0),
            Vec3::new(128.0, -128.0, 0.0),
        ];

        for q_pos in &clear_positions {
            let eng_pos = qte.position_vec3(*q_pos);
            let mover = PlayerMover::new(eng_pos);
            assert!(
                mover.validate_position(&world.nodes, &world.leaves, &world.planes),
                "position {:?} should be clear",
                q_pos
            );
        }
    }

    #[test]
    fn player_mover_straight_junction_traversal() {
        let world = load_fixture("dungeon-junction-straight-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-192.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east through the corridor to the other room
        let delta_eng = Vec3::new(qte.scale * 384.0, 0.0, 0.0);
        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        assert!(
            moved_q.x > 100.0,
            "mover should reach east room, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }
}
